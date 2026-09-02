from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db
from app.auth import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/kpis")
def get_kpis(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT
            COUNT(*) AS total_users,
            COUNT(*) FILTER (WHERE status = 'active') AS active_users,
            COUNT(*) FILTER (WHERE status = 'suspended') AS suspended_users,
            COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') AS new_users_last_7_days
        FROM users
    """)
    row = db.execute(query).fetchone()

    return {
        "total_users": row.total_users or 0,
        "active_users": row.active_users or 0,
        "suspended_users": row.suspended_users or 0,
        "new_users_last_7_days": row.new_users_last_7_days or 0,
    }


@router.get("/charts/user-growth")
def user_growth(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT
            day,
            SUM(new_users) OVER (ORDER BY day) AS total_users
        FROM (
            SELECT DATE(created_at) AS day, COUNT(*) AS new_users
            FROM users
            GROUP BY DATE(created_at)
        ) grouped
        ORDER BY day
    """)

    rows = db.execute(query).fetchall()

    return {
        "labels": [str(r.day) for r in rows],
        "datasets": [
            {"label": "Total Users Growth", "data": [r.total_users for r in rows]}
        ],
    }


@router.get("/charts/users-by-status")
def users_by_status(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT status, COUNT(*) AS total
        FROM users
        GROUP BY status
        ORDER BY total DESC
    """)
    rows = db.execute(query).fetchall()

    return {
        "labels": [r.status for r in rows],
        "datasets": [{"label": "Users by Status", "data": [r.total for r in rows]}],
    }


@router.get("/charts/daily-logins")
def daily_logins(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT DATE(activity_time) AS day, COUNT(DISTINCT user_id) AS active_users
        FROM user_activity_log
        WHERE activity_type = 'login'
        GROUP BY DATE(activity_time)
        ORDER BY day
    """)
    rows = db.execute(query).fetchall()

    return {
        "labels": [str(r.day) for r in rows],
        "datasets": [{"label": "Daily Logins", "data": [r.active_users for r in rows]}],
    }


@router.get("/charts/users-by-role")
def users_by_role(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT role, COUNT(*) AS total
        FROM users
        GROUP BY role
        ORDER BY total DESC
    """)
    rows = db.execute(query).fetchall()

    return {
        "labels": [r.role for r in rows],
        "datasets": [{"label": "Users by Role", "data": [r.total for r in rows]}],
    }


@router.get("/attention")
def attention_items(
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    query = text("""
        SELECT
            full_name,
            role,
            status,
            last_login_at,
            is_verified,
            failed_login_attempts,
            CASE
                WHEN status = 'suspended' THEN 'High'
                WHEN is_verified = false THEN 'High'
                WHEN failed_login_attempts >= 3 THEN 'High'
                WHEN last_login_at < NOW() - INTERVAL '30 days' THEN 'Medium'
                WHEN status = 'inactive' THEN 'Medium'
                ELSE 'Low'
            END AS priority
        FROM users
        WHERE
            status IN ('inactive', 'suspended')
            OR is_verified = false
            OR failed_login_attempts >= 3
            OR last_login_at < NOW() - INTERVAL '30 days'
        ORDER BY
            CASE
                WHEN status = 'suspended' THEN 1
                WHEN is_verified = false THEN 2
                WHEN failed_login_attempts >= 3 THEN 3
                WHEN last_login_at < NOW() - INTERVAL '30 days' THEN 4
                ELSE 5
            END,
            last_login_at ASC NULLS FIRST
    """)
    rows = db.execute(query).fetchall()

    return [
        {
            "full_name": r.full_name,
            "role": r.role,
            "status": r.status,
            "last_login_at": str(r.last_login_at) if r.last_login_at else None,
            "is_verified": r.is_verified,
            "failed_login_attempts": r.failed_login_attempts,
            "priority": r.priority,
        }
        for r in rows
    ]
