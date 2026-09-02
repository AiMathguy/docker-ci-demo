"""
drift_check.py

Feature-drift detection for the churn pipeline.

Two tests, one per feature type:
  - PSI (Population Stability Index) for numeric features. Bins the REFERENCE
    distribution, applies those same edges to the CURRENT data, and measures
    how much mass moved between bins.
  - Chi-square for categorical features. Builds a reference-vs-current
    contingency of category counts and tests whether the split changed.

The reference is a frozen snapshot of the data the live model was trained on.
The current window is whatever you're scoring now. Drift = the current window
has moved far enough from the reference that the model's assumptions may no
longer hold.

Output is a per-feature report plus a single `should_retrain` boolean, so an
Airflow @task.short_circuit can gate the retrain DAG on it directly:

    report = detect_drift(reference_df, current_df, numeric_cols, categorical_cols)
    if report["should_retrain"]:
        trigger_retrain()   # -> calls train.py, which refits on pinned params
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

logger = logging.getLogger("drift_check")

# Anchor any file IO (reference snapshot, saved report) to THIS file's folder,
# not the current working directory — same lesson as the params path.
HERE = Path(__file__).resolve().parent

# --- thresholds -------------------------------------------------------------
# PSI convention: <0.10 stable, 0.10-0.25 moderate shift, >0.25 significant.
PSI_THRESHOLD = 0.25
# Chi-square: reject "same distribution" when p < alpha.
CHI2_ALPHA = 0.05
# Guard against log(0) / division-by-zero when a bin or category is empty in
# one of the two windows. Proportions are floored to this before the PSI math.
EPS = 1e-6


# ---------------------------------------------------------------------------
# Numeric drift: PSI
# ---------------------------------------------------------------------------
def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """PSI between a reference and current numeric distribution.

    Bins are quantile edges of the REFERENCE (so each reference bin holds
    ~equal mass), then the SAME edges are applied to `current`. This is the
    part people get wrong: if you re-bin the current data independently the
    number is meaningless — both windows must be scored against one fixed
    set of edges.

    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%)
    """
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        logger.warning("PSI: empty reference or current series; returning 0.0.")
        return 0.0

    # Quantile edges from the reference. Deduplicate: a spiky/low-cardinality
    # column can produce repeated edges, which would create zero-width bins.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    # Open the outer edges so values beyond the reference range still land in
    # the first/last bin instead of falling out entirely.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Floor both sides so an empty bin can't send a term to inf/nan.
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Categorical drift: chi-square
# ---------------------------------------------------------------------------
def chi_square_drift(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Chi-square test of independence on a reference-vs-current contingency.

    Returns (chi2_stat, p_value). Low p = the category distribution differs
    between the two windows = drift.

    The alignment step matters: a category present in one window but not the
    other must appear in both rows of the contingency (as a 0), or the test
    is comparing mismatched shapes. We union the categories and reindex both
    to fill_value=0.
    """
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    categories = ref_counts.index.union(cur_counts.index)
    ref_aligned = ref_counts.reindex(categories, fill_value=0)
    cur_aligned = cur_counts.reindex(categories, fill_value=0)

    contingency = np.vstack([ref_aligned.values, cur_aligned.values])

    # A category that's zero in BOTH windows gives an all-zero column, which
    # chi2_contingency rejects (zero expected frequency). Drop those columns.
    non_empty = contingency.sum(axis=0) > 0
    contingency = contingency[:, non_empty]

    if contingency.shape[1] < 2:
        logger.warning("chi-square: <2 non-empty categories; treating as no drift.")
        return 0.0, 1.0

    chi2, p_value, _, _ = chi2_contingency(contingency)
    return float(chi2), float(p_value)


# ---------------------------------------------------------------------------
# Orchestration: run both across all features -> single retrain decision
# ---------------------------------------------------------------------------
def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    psi_threshold: float = PSI_THRESHOLD,
    chi2_alpha: float = CHI2_ALPHA,
) -> dict:
    """Run PSI on numeric cols and chi-square on categorical cols.

    Returns a report dict:
        {
          "numeric":   {col: {"psi": float, "drift": bool}},
          "categorical": {col: {"chi2": float, "p_value": float, "drift": bool}},
          "drifted_features": [...],
          "should_retrain": bool,   # <- what the DAG short-circuits on
        }

    `should_retrain` is True if ANY feature drifted. That's the simplest
    policy; see the note at the bottom about tightening it (e.g. require N
    features, or weight by importance) once the skeleton is wired up.
    """
    numeric_result: dict = {}
    for col in numeric_cols:
        if col not in reference_df or col not in current_df:
            logger.warning("Skipping numeric '%s': missing in one frame.", col)
            continue
        psi = population_stability_index(reference_df[col], current_df[col])
        drift = psi > psi_threshold
        numeric_result[col] = {"psi": psi, "drift": drift}
        logger.info("PSI %-25s = %.4f  drift=%s", col, psi, drift)

    categorical_result: dict = {}
    for col in categorical_cols:
        if col not in reference_df or col not in current_df:
            logger.warning("Skipping categorical '%s': missing in one frame.", col)
            continue
        chi2, p = chi_square_drift(reference_df[col], current_df[col])
        drift = p < chi2_alpha
        categorical_result[col] = {"chi2": chi2, "p_value": p, "drift": drift}
        logger.info("chi2 %-25s p=%.4f  drift=%s", col, p, drift)

    drifted = [c for c, r in numeric_result.items() if r["drift"]] + [
        c for c, r in categorical_result.items() if r["drift"]
    ]

    return {
        "numeric": numeric_result,
        "categorical": categorical_result,
        "drifted_features": drifted,
        "should_retrain": len(drifted) > 0,
    }


# ---------------------------------------------------------------------------
# Seam for the DAG: load the frozen reference snapshot.
# ---------------------------------------------------------------------------
def load_reference(
    path: str | Path = HERE / "reference" / "reference_snapshot.parquet",
) -> pd.DataFrame:
    """Load the frozen training-time snapshot the live model was trained on.

    TODO: point this at however you persist the reference — a parquet dropped
    at train time, a table in admin_dashboard, or an MLflow artifact tied to
    the deployed model. Whatever you choose, it must be the data the CURRENT
    model saw, not 'latest' — otherwise you're comparing now against now.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference snapshot not found at {path}. "
            "Write one at train time so drift has something to compare against."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Synthetic smoke test so the skeleton runs standalone before it's wired
    # to real data. `current` has a deliberately shifted numeric column and a
    # shifted category split, so you should see drift fire on both.
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "days_since_last_login": rng.normal(30, 5, 1000),
            "login_count_7d": rng.poisson(4, 1000),
            "has_active_subscription": rng.choice(["yes", "no"], 1000, p=[0.7, 0.3]),
        }
    )
    current = pd.DataFrame(
        {
            "days_since_last_login": rng.normal(45, 5, 1000),  # shifted up
            "login_count_7d": rng.poisson(4, 1000),  # ~same
            "has_active_subscription": rng.choice(
                ["yes", "no"], 1000, p=[0.4, 0.6]
            ),  # shifted
        }
    )

    report = detect_drift(
        reference,
        current,
        numeric_cols=["days_since_last_login", "login_count_7d"],
        categorical_cols=["has_active_subscription"],
    )

    print("\n--- drift report ---")
    print("drifted features:", report["drifted_features"])
    print("should_retrain:  ", report["should_retrain"])
