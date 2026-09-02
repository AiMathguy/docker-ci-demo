import logging, joblib
from pathlib import Path
import pandas as pd
import mlflow
from config import MLFLOW_TRACKING_URI
from data_ingestion import DataIngestor
from preprocessing import Preprocessor

logger = logging.getLogger("predict")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_PATH = Path("best_model/model.joblib")
TRANSFORMER_PATH = Path(
    "best_model/transformer.joblib"
)  # produced by train.py (see note)
OUTPUT_PATH = Path("backend/data/predictions.csv")

LABEL_COL = "churn_label"
ID_COL = "user_id"
THRESHOLD = 0.5


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(
            f"Fitted transformer not found at {TRANSFORMER_PATH}. "
            "train.py must persist it: add "
            "`joblib.dump(transformer, 'best_model/transformer.joblib')` "
            "right after transformer.fit_transform(...)."
        )
    model = joblib.load(MODEL_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)
    logger.info("Loaded model + transformer from best_model/")
    return model, transformer


def _recover_ids(raw: pd.DataFrame, clean: pd.DataFrame):
    # user_id is excluded from features but we still want it on the output.
    if ID_COL in clean.columns:
        return clean[ID_COL].values
    if ID_COL in raw.columns and clean.index.isin(raw.index).all():
        return raw.loc[clean.index, ID_COL].values
    logger.warning("Could not recover %s; falling back to row index.", ID_COL)
    return range(len(clean))


def predict():
    model, transformer = load_artifacts()

    # 1. fresh data from the same source as training
    ingestor = DataIngestor()
    try:
        df = ingestor.load_customer_features(add_churn_labels=True)
    finally:
        ingestor.close()

    # 2. same cleaning as tuning/training, but transform() with the FITTED
    #    transformer — never fit_transform, or you'd relearn state at score time.
    pre = Preprocessor(label_col=LABEL_COL, exclude_cols=[ID_COL])
    clean_df, _, _ = pre.run(df)  # reuse cleaning; ignore fresh transformer
    features = clean_df.drop(columns=[LABEL_COL], errors="ignore")
    X = transformer.transform(features)

    # 3. score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        labels = (proba >= THRESHOLD).astype(int)
    else:
        labels = model.predict(X)
        proba = [None] * len(labels)

    out = pd.DataFrame(
        {
            ID_COL: _recover_ids(df, clean_df),
            "churn_probability": proba,
            "churn_label": labels,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info("Scored %d rows -> %s", len(out), OUTPUT_PATH)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    predict()
