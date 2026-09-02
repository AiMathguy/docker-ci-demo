from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL, echo=False)


def days_since(dt: Any) -> int:
    if dt is None:
        return 999
    now = datetime.now(timezone.utc)
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((now - dt).days, 0)


def score_user(row: Any) -> dict[str, Any]:
    reasons: list[str | None] = []
    score = 0

    if row.status == "suspended":
        score += 40
        reasons.append("Account is suspended")

    if not row.is_verified:
        score += 20
        reasons.append("User is not verified")

    failed = row.failed_login_attempts or 0
    if failed >= 3:
        score += 15
        reasons.append(f"{failed} failed login attempts")

    inactivity_days = days_since(row.last_login_at)
    if inactivity_days > 30:
        score += 15
        reasons.append(f"No login for {inactivity_days} days")
    elif inactivity_days > 14:
        score += 10
        reasons.append(f"Low recent activity ({inactivity_days} days since login)")

    if (row.login_count_7d or 0) == 0:
        score += 10
        reasons.append("No logins in last 7 days")

    score = min(score, 100)
    churn_probability = round(score / 100.0, 2)

    if score >= 60:
        risk_level = "High"
    elif score >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    while len(reasons) < 3:
        reasons.append(None)

    return {
        "user_id": row.user_id,
        "churn_probability": churn_probability,
        "risk_score": float(score),
        "risk_level": risk_level,
        "top_reason_1": reasons[0],
        "top_reason_2": reasons[1],
        "top_reason_3": reasons[2],
        "model_version": "rules_v1",
    }


def rebuild_features() -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM customer_features"))

        conn.execute(text("""
                INSERT INTO customer_features (
                    user_id,
                    days_since_last_login,
                    login_count_7d,
                    login_count_30d,
                    failed_login_attempts,
                    is_verified,
                    status,
                    role,
                    created_days_ago,
                    updated_at
                )
                SELECT
                    u.id AS user_id,
                    CASE
                        WHEN u.last_login_at IS NULL THEN 999
                        ELSE CAST(EXTRACT(DAY FROM (NOW() - u.last_login_at)) AS INT)
                    END AS days_since_last_login,
                    COALESCE(SUM(CASE
                        WHEN l.activity_type = 'login'
                         AND l.activity_time >= NOW() - INTERVAL '7 days'
                        THEN 1 ELSE 0 END), 0) AS login_count_7d,
                    COALESCE(SUM(CASE
                        WHEN l.activity_type = 'login'
                         AND l.activity_time >= NOW() - INTERVAL '30 days'
                        THEN 1 ELSE 0 END), 0) AS login_count_30d,
                    COALESCE(u.failed_login_attempts, 0) AS failed_login_attempts,
                    u.is_verified,
                    u.status,
                    u.role,
                    CAST(EXTRACT(DAY FROM (NOW() - u.created_at)) AS INT) AS created_days_ago,
                    NOW() AS updated_at
                FROM users u
                LEFT JOIN user_activity_log l ON l.user_id = u.id
                GROUP BY
                    u.id, u.last_login_at, u.failed_login_attempts,
                    u.is_verified, u.status, u.role, u.created_at
                """))


def rebuild_predictions() -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("""
                SELECT
                    cf.user_id,
                    cf.login_count_7d,
                    cf.login_count_30d,
                    cf.failed_login_attempts,
                    cf.is_verified,
                    cf.status,
                    cf.role,
                    u.last_login_at
                FROM customer_features cf
                JOIN users u ON u.id = cf.user_id
                """)).fetchall()

        conn.execute(text("DELETE FROM customer_predictions"))

        for row in rows:
            pred = score_user(row)
            conn.execute(
                text("""
                    INSERT INTO customer_predictions (
                        user_id,
                        churn_probability,
                        risk_score,
                        risk_level,
                        top_reason_1,
                        top_reason_2,
                        top_reason_3,
                        model_version,
                        scored_at
                    ) VALUES (
                        :user_id,
                        :churn_probability,
                        :risk_score,
                        :risk_level,
                        :top_reason_1,
                        :top_reason_2,
                        :top_reason_3,
                        :model_version,
                        NOW()
                    )
                    """),
                pred,
            )


def rebuild_eda_snapshot() -> None:
    with engine.begin() as conn:
        counts = conn.execute(text("""
                SELECT
                    COUNT(*) AS total_users,
                    COUNT(*) FILTER (WHERE cp.risk_level = 'High') AS high_risk_users,
                    AVG(cf.login_count_7d)::float AS avg_logins_7d,
                    AVG(cf.failed_login_attempts)::float AS avg_failed_logins
                FROM customer_features cf
                JOIN customer_predictions cp ON cp.user_id = cf.user_id
                """)).fetchone()

        top_status = conn.execute(text("""
                SELECT status, COUNT(*) AS total
                FROM users
                GROUP BY status
                ORDER BY total DESC
                """)).fetchall()

        payload = {
            "summary": {
                "total_users": counts.total_users or 0,
                "high_risk_users": counts.high_risk_users or 0,
                "avg_logins_7d": round(counts.avg_logins_7d or 0, 2),
                "avg_failed_logins": round(counts.avg_failed_logins or 0, 2),
            },
            "status_breakdown": [
                {"status": r.status, "total": r.total} for r in top_status
            ],
            "top_signals": [
                "days_since_last_login",
                "failed_login_attempts",
                "login_count_7d",
                "is_verified",
                "status",
            ],
        }

        conn.execute(
            text("""
                INSERT INTO eda_snapshots (snapshot_type, payload, created_at)
                VALUES ('risk_overview', CAST(:payload AS JSONB), NOW())
                """),
            {"payload": json.dumps(payload)},
        )


if __name__ == "__main__":
    rebuild_features()
    rebuild_predictions()
    rebuild_eda_snapshot()
    print("Scoring complete.")
