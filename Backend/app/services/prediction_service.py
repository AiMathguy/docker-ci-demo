import pandas as pd


def get_total_users(filtered_df: pd.DataFrame) -> int:
    return len(filtered_df)


def get_inactive_users(filtered_df: pd.DataFrame) -> int:
    if "status" not in filtered_df.columns:
        return 0
    return int((filtered_df["status"] == "inactive").sum())


def get_high_risk_users(filtered_df: pd.DataFrame) -> int:
    if "risk_level" not in filtered_df.columns:
        return 0
    return int((filtered_df["risk_level"] == "High").sum())


def get_total_revenue(filtered_df: pd.DataFrame) -> float:
    if "amount" not in filtered_df.columns:
        return 0.0
    return float(filtered_df["amount"].fillna(0).sum())
