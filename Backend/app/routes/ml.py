from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd

from app.db import get_db
from ml_model import load_model, predict
from explainability import explain_predictions

router = APIRouter(prefix="/api/ml", tags=["ML"])


def risk_level_from_probability(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


@router.get("/predict")
def predict_churn(db: Session = Depends(get_db)):
    model = load_model()

    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    rows = db.execute(text("""
            SELECT
                cf.user_id,
                u.full_name,
                u.email,
                cf.days_since_last_login,
                cf.login_count_7d,
                cf.login_count_30d,
                cf.failed_login_attempts,
                cf.is_verified,
                cf.status,
                cf.role,
                cf.created_days_ago
            FROM customer_features cf
            JOIN users u ON u.id = cf.user_id
            """)).mappings().all()

    if not rows:
        raise HTTPException(status_code=400, detail="No customer features found")

    df = pd.DataFrame(rows)

    probs = predict(model, df)
    explanations = [{"top_features": []} for _ in range(len(df))]

    results = []
    for i, row in df.iterrows():
        results.append(
            {
                "user_id": str(row["user_id"]),
                "full_name": row["full_name"],
                "email": row["email"],
                "role": row["role"],
                "status": row["status"],
                "churn_probability": float(probs[i]),
                "risk_level": risk_level_from_probability(float(probs[i])),
                "top_features": explanations[i]["top_features"],
            }
        )

    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return results
