import os
import joblib
import pandas as pd
import shap
from dotenv import load_dotenv
from sqlalchemy import create_engine
from dashboard_ui.ml_model import prepare_features
from data_ingestion import DataIngestor

# Load environment variables
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

MODEL_PATH = "xgb_best_model.pkl"

# Human-readable labels
# FEATURE_LABELS = {
#     "login_count_7d": "Low weekly activity",
#     "login_count_30d": "Low monthly activity",
#     "failed_login_attempts": "Multiple failed logins",
#     "days_since_last_login": "Recent inactivity",
#     "is_verified": "Account verification status",
# }


class Explainability:
    def load_data() -> pd.DataFrame:
        # """Load customer_features table from DB."""
        # engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        # # raw_connection() returns a DBAPI2 connection, but may not support context manager
        # conn = engine.raw_connection()
        # try:
        #     df = pd.read_sql("SELECT * FROM customer_features", conn)
        # finally:
        #     conn.close()
        # return df
        try:
            ingestor = DataIngestor()
            df = ingestor.load_customer_features(add_churn_labels=True)
            if df.empty:
                raise ValueError("customer_features table is empty.")
            logger.info(f"Loaded {len(df)} rows from customer_features.")
            return df
        except Exception as e:
            logger.error(f"Error occurred while loading customer_features table: {e}")
            raise

        ingestor = DataIngestor()
        df = ingestor.master_df()
        y = df["churn_label"]
        X = df.drop(["churn_label"], axis=1)

    def explain_predictions(model, df: pd.DataFrame):
        """Compute SHAP values and return top features per row."""
        X = prepare_features(df)

        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)

        values = explanation.values
        if len(values.shape) == 3:
            # For binary classification, take class 1 (index 1) or class 0 as needed
            values = values[:, :, 0]  # adjust index if necessary

        results = []
        for i in range(len(X)):
            row_shap = values[i]
            pairs = list(zip(X.columns, row_shap))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_pairs = pairs[:3]

            results.append(
                {
                    "top_features": [
                        {
                            "feature": feature,
                            "label": FEATURE_LABELS.get(feature, feature),
                            "impact": float(value),
                        }
                        for feature, value in top_pairs
                    ]
                }
            )
        return results

    def log_explainability_artifacts(model, model_name, X_train, X_test, y_test):
        model_name = str(model_name).lower()

        if model_name == "xgboost":
            logging.info("Using SHAP TreeExplainer for XGBoost model explainability")
            print("Using SHAP TreeExplainer", flush=True)
        elif model_name == "sgd":
            logging.info("Using linear coefficients for SGD model explainability")
            print("Using linear coefficients", flush=True)
        elif model_name in ["svm", "knn"]:
            logging.info(
                "Using permutation importance for {model_name} model explainability"
            )
            print("Using permutation importance", flush=True)
        else:
            logging.error("Unknown model type")
            print("Unknown model type", flush=True)

        X_test_sample = X_test.sample(
            n=min(100, len(X_test)),
            random_state=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:

            mlflow.log_param("explainability_mode", "custom_artifacts")
            mlflow.log_param("explainability_sample_size", len(X_test_sample))

            if model_name == "xgboost":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_test_sample)

                shap.plots.bar(shap_values, show=False)
                bar_path = os.path.join(tmpdir, "shap_bar.png")
                plt.gcf().savefig(bar_path, bbox_inches="tight", dpi=150)
                plt.close()

                shap.plots.beeswarm(shap_values, show=False)
                beeswarm_path = os.path.join(tmpdir, "shap_beeswarm.png")
                plt.gcf().savefig(beeswarm_path, bbox_inches="tight", dpi=150)
                plt.close()

                shap.plots.waterfall(shap_values[0], show=False)
                waterfall_path = os.path.join(tmpdir, "shap_waterfall_first_row.png")
                plt.gcf().savefig(waterfall_path, bbox_inches="tight", dpi=150)
                plt.close()

                shap_df = pd.DataFrame(
                    shap_values.values,
                    columns=X_test_sample.columns,
                )
                shap_csv_path = os.path.join(tmpdir, "shap_values_sample.csv")
                shap_df.to_csv(shap_csv_path, index=False)

                mlflow.log_artifacts(tmpdir, artifact_path="explainability")
                logging.info(
                    "Logged explainability artifacts for XGBoost model SHAP values"
                )

            elif model_name == "sgd":
                if hasattr(model, "coef_"):
                    coef_df = pd.DataFrame(
                        {
                            "feature": X_test.columns,
                            "coefficient": model.coef_.ravel(),
                        }
                    ).sort_values("coefficient", key=abs, ascending=False)

                    coef_path = os.path.join(tmpdir, "sgd_coefficients.csv")
                    coef_df.to_csv(coef_path, index=False)

                    ax = coef_df.plot(
                        kind="barh",
                        x="feature",
                        y="coefficient",
                        legend=False,
                        title="SGD Feature Coefficients",
                    )
                    fig = ax.get_figure()
                    coef_plot_path = os.path.join(tmpdir, "sgd_coefficients.png")
                    fig.savefig(coef_plot_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)

                mlflow.log_artifacts(tmpdir, artifact_path="explainability")
                logging.info(
                    "Logged explainability artifacts for SGD model coefficients"
                )

            elif model_name in ["svm", "knn"]:
                sample_y = y_test.loc[X_test_sample.index]

                result = permutation_importance(
                    model,
                    X_test_sample,
                    sample_y,
                    scoring="roc_auc",
                    n_repeats=10,
                    random_state=42,
                )

                importance_df = pd.DataFrame(
                    {
                        "feature": X_test.columns,
                        "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std,
                    }
                ).sort_values("importance_mean", ascending=False)

                importance_path = os.path.join(tmpdir, "permutation_importance.csv")
                importance_df.to_csv(importance_path, index=False)

                ax = importance_df.plot(
                    kind="barh",
                    x="feature",
                    y="importance_mean",
                    legend=False,
                    title=f"{model_name.upper()} Permutation Importance",
                )
                fig = ax.get_figure()
                importance_plot_path = os.path.join(
                    tmpdir, "permutation_importance.png"
                )
                fig.savefig(importance_plot_path, bbox_inches="tight", dpi=150)
                plt.close(fig)

                mlflow.log_artifacts(tmpdir, artifact_path="explainability")

            else:
                mlflow.log_param("explainability_status", "unsupported_model")


if __name__ == "__main__":
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    print("Loading customer data...")
    df = Explainability.load_data()
    print(f"Loaded {len(df)} customers.")

    print("Computing SHAP explanations...")
    results = Explainability.explain_predictions(model, df)

    # Print results
    for i, res in enumerate(results[:10]):  # first 10 customers
        print(f"\n--- Customer {i+1} ---")
        for feat in res["top_features"]:
            print(f"  {feat['label']}: impact = {feat['impact']:.5f}")
