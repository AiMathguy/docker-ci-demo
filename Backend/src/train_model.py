import json, logging, joblib
from pathlib import Path
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from config import MLFLOW_TRACKING_URI
from tune_model import ModelFactory  # reuse the factory, NOT the orchestrator
from data_ingestion import DataIngestor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

PARAMS_PATH = Path("config/best_params.json")
MODEL_OUT = Path("best_model/model.joblib")

# Only these five columns get used. The old Preprocessor was one-hot-encoding
# full_name, email, created_at and last_login_at -- that's what produced the
# 900-column model that couldn't score a new customer. churned and churn_score
# are the answer itself, so they stay out too.
FEATURES = [
    "is_verified",
    "days_since_last_login",
    "login_count_7d",
    "failed_login_attempts",
    "has_active_subscription",
]
LABEL = "churn_label"


def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text())  # written by tune_model.py


def train():
    # 1. fresh data from the same source
    ingestor = DataIngestor()
    try:
        df = ingestor.load_customer_features(add_churn_labels=True)
    finally:
        ingestor.close()

    # 2. just the five real features
    X = df[FEATURES]
    y = df[LABEL]

    # 3. pinned params -> factory. NO optuna.
    saved = load_params()
    model_name = saved.pop("model")
    model = ModelFactory.create_model(model_name, saved)

    # 4. bundle imputation + model into ONE artifact, so the API can send the
    #    five raw fields and the pipeline preprocesses them itself.
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    pipe.fit(X, y)

    # 5. save the model FIRST -- this is what serve.py loads.
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    logger.info(
        "Retrained %s on %d rows, %d features -> %s",
        model_name,
        len(y),
        len(FEATURES),
        MODEL_OUT,
    )

    # 6. experiment tracking is best-effort: a missing MLflow server must never
    #    kill a training run. The joblib model above is already on disk.
    try:
        with mlflow.start_run(run_name="drift_retrain"):
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(saved)
            mlflow.sklearn.log_model(pipe, artifact_path="best_model")
    except Exception as e:
        logger.warning(
            "Skipped MLflow logging (%s: server not reachable). "
            "Model still saved to %s.",
            type(e).__name__,
            MODEL_OUT,
        )

    return pipe


if __name__ == "__main__":
    train()
