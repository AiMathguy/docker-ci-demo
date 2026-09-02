import pandas as pd
import streamlit as st
from sqlalchemy import text
from Backend.streamlit.utils.config import engine
from datetime import datetime, timedelta


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    query = text(f"SELECT * FROM {name}")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return pd.DataFrame(rows)


@st.cache_data
def load_all_data():
    users = load_table("users")
    activity = load_table("user_activity_log")
    features = load_table("customer_features")
    preds = load_table("customer_predictions")
    try:
        subs = load_table("subscriptions")
    except Exception:
        subs = pd.DataFrame()
    return users, activity, features, preds, subs


def apply_filters(users, features, preds, subs, date_range, role_filter, status_filter):
    users["id"] = users["id"].astype(str)
    users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")

    filtered_users = users[
        (users["created_at"].dt.date >= date_range[0])
        & (users["created_at"].dt.date <= date_range[1])
        & (users["role"].isin(role_filter))
        & (users["status"].isin(status_filter))
    ]

    df = users.copy()
    if not features.empty:
        features["user_id"] = features["user_id"].astype(str)
        df = df.merge(
            features,
            left_on="id",
            right_on="user_id",
            how="left",
            suffixes=("", "_feat"),
        )
    if not preds.empty:
        preds["user_id"] = preds["user_id"].astype(str)
        df = df.merge(
            preds, left_on="id", right_on="user_id", how="left", suffixes=("", "_pred")
        )
    if not subs.empty:
        subs["user_id"] = subs["user_id"].astype(str)
        df = df.merge(
            subs, left_on="id", right_on="user_id", how="left", suffixes=("", "_sub")
        )

    filtered_df = df[df["id"].isin(filtered_users["id"])].copy()
    return filtered_df
