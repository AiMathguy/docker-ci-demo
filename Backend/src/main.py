"""
FastAPI churn prediction service.

Loads a trained model once at startup (from a local joblib file) and exposes
single + batch predict endpoints. Feature order, null-fill values, and the
decision threshold are read from features.yaml so this stays consistent with
the ingestion pipeline.

Run:
    # MODEL_PATH is optional; defaults to best_model/model.joblib
    export MODEL_PATH="best_model/model.joblib"
    export FEATURES_CONFIG="src/config/features.yaml"
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("churn_api")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

CONFIG_PATH = Path(os.getenv("FEATURES_CONFIG", "src/config/features.yaml"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "src/best_model/model.joblib"))


def load_config(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Features config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


CONFIG = load_config(CONFIG_PATH)

FEATURE_ORDER: list[str] = CONFIG["features"]
THRESHOLD: float = CONFIG["target"]["probability_threshold"]
TARGET_NAME: str = CONFIG["target"]["name"]

# Reuse the null-fill values defined for the labeling rules so inference
# handles missing values the same way the training data was built.
FILLNA: dict[str, float] = {
    c["column"]: c["fillna"]
    for c in CONFIG["churn_rules"]["conditions"].values()
    if "fillna" in c
}

# --------------------------------------------------------------------------- #
# Model lifecycle
# --------------------------------------------------------------------------- #

state: dict = {"model": None, "has_proba": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        logger.error(
            "Model file not found at %s; predictions will be unavailable.", MODEL_PATH
        )
    else:
        try:
            model = joblib.load(MODEL_PATH)
            state["model"] = model
            state["has_proba"] = hasattr(model, "predict_proba")
            logger.info(
                "Loaded model from %s (predict_proba=%s)",
                MODEL_PATH,
                state["has_proba"],
            )
        except Exception:
            logger.exception("Failed to load model from %s", MODEL_PATH)
    yield
    state["model"] = None


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class CustomerFeatures(BaseModel):
    is_verified: int = Field(..., ge=0, le=1)
    days_since_last_login: Optional[float] = Field(None, ge=0)
    login_count_7d: Optional[int] = Field(None, ge=0)
    failed_login_attempts: int = Field(..., ge=0)
    has_active_subscription: int = Field(..., ge=0, le=1)


class Prediction(BaseModel):
    churn_label: int
    churn_probability: Optional[float]
    threshold: float


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def to_frame(rows: list[CustomerFeatures]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    for col, value in FILLNA.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    # Enforce the exact column order the model was trained on.
    return df[FEATURE_ORDER]


def run_inference(df: pd.DataFrame) -> list[Prediction]:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if state["has_proba"]:
        probs = model.predict_proba(df)[:, 1]
        labels = (probs >= THRESHOLD).astype(int)
        return [
            Prediction(
                churn_label=int(l), churn_probability=float(p), threshold=THRESHOLD
            )
            for p, l in zip(probs, labels)
        ]

    # Model exposes only hard labels.
    labels = model.predict(df)
    return [
        Prediction(churn_label=int(l), churn_probability=None, threshold=THRESHOLD)
        for l in labels
    ]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_path": str(MODEL_PATH),
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures):
    try:
        df = to_frame([customer])
        return run_inference(df)[0]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    try:
        df = to_frame(request.customers)
        return BatchResponse(predictions=run_inference(df))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch prediction failed.")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
