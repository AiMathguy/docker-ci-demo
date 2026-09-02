import os
import tempfile
import logging
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

from data_ingestion import DataIngestor

logging.basicConfig(level=logging.INFO)


class ModelExplainer:
    def __init__(self, model_path: str = "xgb_best_model.pkl"):
        self.model = self._load_model(model_path)
        self.X = None
        self.shap_values = None
        self.lime_explainer = None

    def _load_model(self, path: str):
        if not Path(path).exists():
            logging.error(f"Model file not found: {path}")
            return None
        return joblib.load(path)

    def load_and_prepare_data(self):
        """Ingests and prepares features via DataIngestor."""
        ingestor = DataIngestor()
        try:
            df = ingestor.load_customer_features(add_churn_labels=False)
            if df.empty:
                raise ValueError("No data returned from DataIngestor.")
            self.X = df
            logging.info(f"Features prepared. Shape: {self.X.shape}")
        finally:
            ingestor.close()

    def initialize_explainers(self):
        """Initializes SHAP and LIME engines."""
        if self.X is None or self.model is None:
            raise RuntimeError("Load data and model before initializing explainers.")

        self.explainer_shap = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer_shap(self.X)

        self.lime_explainer = LimeTabularExplainer(
            training_data=np.array(self.X),
            feature_names=list(self.X.columns),
            class_names=["Low Risk", "High Risk"],
            mode="classification",
        )

    def plot_shap_summary(self):
        """Generates global feature importance plot."""
        fig, _ = plt.subplots()
        shap.summary_plot(self.shap_values, self.X, show=False)
        st.pyplot(plt.gcf())
        plt.close(fig)

    def plot_local_explanation(self, user_index: int = 0):
        """Combines SHAP waterfall and LIME explanation for a single user."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SHAP Waterfall")
            fig = plt.figure()
            shap.plots.waterfall(self.shap_values[user_index], show=False)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("LIME Explanation")
            exp = self.lime_explainer.explain_instance(
                self.X.iloc[user_index].values,
                self.model.predict_proba,
                num_features=5,
            )
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            plt.close(fig)

    def plot_dependency(self, feature_name: str):
        """Plots how a specific feature impacts predictions."""
        fig, ax = plt.subplots()
        shap.dependence_plot(
            feature_name,
            self.shap_values.values,
            self.X,
            show=False,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)


def log_explainability_artifacts(model, model_name: str, X_train, X_test, y_test):
    """Logs SHAP explainability artifacts to MLflow for supported model types."""
    model_name = str(model_name).lower()

    X_test_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)

    # FIX: entire artifact generation happens inside the tempdir context so
    # tmpdir is still valid when mlflow.log_artifacts is called.
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.log_param("explainability_sample_size", len(X_test_sample))

        if model_name == "xgboost":
            logging.info("Generating SHAP artifacts (TreeExplainer)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_sample)

            shap.plots.bar(shap_values, show=False)
            plt.gcf().savefig(
                os.path.join(tmpdir, "shap_bar.png"), bbox_inches="tight", dpi=150
            )
            plt.close()

            shap.plots.beeswarm(shap_values, show=False)
            plt.gcf().savefig(
                os.path.join(tmpdir, "shap_beeswarm.png"), bbox_inches="tight", dpi=150
            )
            plt.close()

            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().savefig(
                os.path.join(tmpdir, "shap_waterfall_first_row.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

            pd.DataFrame(shap_values.values, columns=X_test_sample.columns).to_csv(
                os.path.join(tmpdir, "shap_values_sample.csv"), index=False
            )

        elif model_name == "sgd":
            logging.info("Generating linear coefficient artifacts")
            coef = pd.Series(model.coef_.ravel(), index=X_test_sample.columns)
            coef.sort_values().plot(kind="barh")
            plt.tight_layout()
            plt.gcf().savefig(
                os.path.join(tmpdir, "sgd_coefficients.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

        elif model_name in ["svm", "knn"]:
            logging.info("Generating permutation importance artifacts")
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model,
                X_test_sample,
                y_test.loc[X_test_sample.index],
                n_repeats=10,
                random_state=42,
            )
            imp = pd.Series(
                result.importances_mean, index=X_test_sample.columns
            ).sort_values()
            imp.plot(kind="barh")
            plt.tight_layout()
            plt.gcf().savefig(
                os.path.join(tmpdir, "permutation_importance.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

        else:
            logging.warning(
                f"No explainability method available for model: {model_name}"
            )
            return

        mlflow.log_artifacts(tmpdir, artifact_path="explainability")


if __name__ == "__main__":
    explainer_tool = ModelExplainer()
    explainer_tool.load_and_prepare_data()
    explainer_tool.initialize_explainers()

    st.title("Churn Model Explainability")
    explainer_tool.plot_shap_summary()

    user_id_idx = st.number_input("Select User Index", min_value=0, value=0)
    explainer_tool.plot_local_explanation(user_index=int(user_id_idx))
