"""
Churn model tuning pipeline with Optuna + MLflow.
Supports XGBoost, SVM, SGD, and KNN.
"""

import os
import logging
from typing import Dict, Any, Tuple
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from shap_charts import log_explainability_artifacts
from sklearn.model_selection import train_test_split
from data_ingestion import DataIngestor
from preprocessing import Preprocessor
import json, pathlib

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tune_model")

const_handler = logging_stream_handler = logging.StreamHandler()
const_handler.setLevel(logging.DEBUG)
logger.addHandler(const_handler)

file_handler = logging.FileHandler("tune_model_error.log")
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
const_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


class ModelFactory:

    SUPPORTED_MODELS = ["xgboost", "svm", "sgd", "knn"]

    @staticmethod
    def create_model(model_name: str, params: Dict[str, Any]) -> Any:
        if model_name == "xgboost":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                **params,
            )
        if model_name == "svm":
            return SVC(probability=True, random_state=42, **params)
        if model_name == "sgd":
            return SGDClassifier(loss="log_loss", random_state=42, **params)
        if model_name == "knn":
            return KNeighborsClassifier(**params)

        raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def get_param_prefix(model_name: str) -> str:
        return {"svm": "svm_", "sgd": "sgd_", "knn": "knn_", "xgboost": ""}.get(
            model_name, ""
        )


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


class TuningOrchestrator:

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 20,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.study: optuna.Study | None = None

    def _suggest_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if model_name == "svm":
            return {
                "C": trial.suggest_float("svm_C", 0.1, 10.0, log=True),
                "kernel": trial.suggest_categorical("svm_kernel", ["linear", "rbf"]),
            }
        if model_name == "sgd":
            return {
                "alpha": trial.suggest_float("sgd_alpha", 1e-5, 1e-1, log=True),
            }
        if model_name == "knn":
            return {
                "n_neighbors": trial.suggest_int("knn_n_neighbors", 3, 15),
                "weights": trial.suggest_categorical(
                    "knn_weights", ["uniform", "distance"]
                ),
            }
        raise logger.error(f"Unknown model: {model_name}")

    def _objective(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        model_name = trial.suggest_categorical("model", ModelFactory.SUPPORTED_MODELS)
        params = self._suggest_params(trial, model_name)
        model = ModelFactory.create_model(model_name, params)

        logging.info(f"[Trial {trial.number}] model={model_name} params={params}")

        cv_results = cross_validate(
            model,
            self.X_train,
            self.y_train,
            cv=3,
            scoring=["roc_auc", "recall", "precision"],
            error_score="raise",
        )

        auc = cv_results["test_roc_auc"].mean()
        recall = cv_results["test_recall"].mean()
        precision = cv_results["test_precision"].mean()

        with mlflow.start_run(
            nested=True,
            run_name=f"{model_name}_trial_{trial.number}",
        ):
            mlflow.log_params({"model": model_name, **params})
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)

        logger.info(
            f"[Trial {trial.number}] "
            f"auc={auc:.4f} recall={recall:.4f} precision={precision:.4f}"
        )

        return auc, recall, precision

    def run_tuning(self) -> optuna.Study:
        self.study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"]
        )
        self.study.optimize(self._objective, n_trials=self.n_trials)
        return self.study


# ---------------------------------------------------------------------------
# Final model trainer
# ---------------------------------------------------------------------------


class FinalModelTrainer:

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        best_trial: optuna.trial.FrozenTrial,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = best_trial.params
        self.best_value = best_trial.values[0]

    def _clean_params(
        self, model_name: str, raw_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        prefix = ModelFactory.get_param_prefix(model_name)
        return {
            (k[len(prefix) :] if prefix and k.startswith(prefix) else k): v
            for k, v in raw_params.items()
            if k != "model"
        }

    def train_and_save(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_path: str = "best_model",
    ):
        model_name = self.best_params["model"]
        clean_params = self._clean_params(model_name, self.best_params)

        model = ModelFactory.create_model(model_name, clean_params)
        model.fit(self.X_train, self.y_train)

        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, "model.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

        pathlib.Path("config").mkdir(exist_ok=True)
        pathlib.Path("config/best_params.json").write_text(
            json.dumps({"model": model_name, **clean_params}, indent=2)
        )
        logging.info("Wrote best params to config/best_params.json")
        # --- end add ---

        with mlflow.start_run(run_name="final_model"):
            mlflow.log_param("final_model_type", model_name)
            mlflow.log_params(clean_params)

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=output_path,
                input_example=X_test[:5],
            )

            eval_data = X_test.copy()
            eval_data["label"] = y_test.values

            try:
                mlflow.models.evaluate(
                    model_info.model_uri,
                    eval_data,
                    targets="label",
                    model_type="classifier",
                    evaluator_config={"log_explainer": False},
                )
            except Exception as e:
                logging.warning(f"MLflow evaluation skipped: {e}")

            log_explainability_artifacts(
                model=model,
                model_name=model_name,
                X_train=self.X_train,
                X_test=X_test,
                y_test=y_test,
            )

        return model


if __name__ == "__main__":
    ingestor = None
    try:
        ingestor = DataIngestor()
        df = ingestor.load_customer_features(add_churn_labels=True)
        print(df.head())

        # load_customer_features() produces "churn_label", not "Churn".
        preprocessor = Preprocessor(label_col="churn_label", exclude_cols=["user_id"])

        # Full flow: dedupe -> null check (conditional imputer) -> outlier
        # report -> imbalance check -> continue (with SMOTE only if warranted).
        clean_df, transformer, report = preprocessor.run(df)

        print("Duplicates removed:", report["duplicates"]["n_duplicates"])
        print("Rows with outliers:", report["outliers"]["n_rows_with_outliers"])
        print("Class balance:", report["imbalance"])

        # Fit the transformer here (train split only, in real usage) to see
        # the actual output shape and feature names.
        X = clean_df.drop(columns=["churn_label"])
        Xt = transformer.fit_transform(X)

        print("Transformed shape:", Xt.shape)
        print("Feature names:", list(transformer.get_feature_names_out()))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            Xt, clean_df["churn_label"], test_size=0.2, random_state=42
        )

        tuner = TuningOrchestrator(X_train, y_train, n_trials=50)
        study = tuner.run_tuning()

        if study.best_trials:
            best_trial = max(study.best_trials, key=lambda t: t.values[0])
            trainer = FinalModelTrainer(X_train, y_train, best_trial)
            model = trainer.train_and_save(X_test, y_test)

    except Exception:
        logger.exception("Tuning failed.")
        raise
    finally:
        if ingestor is not None:
            ingestor.close()
