"""
preprocessing.py
Feature preprocessing for the churn pipeline.

Builds an (unfit) sklearn ColumnTransformer that:
  - median-imputes and standard-scales numeric features
  - most-frequent-imputes and one-hot-encodes categorical features
  - leaves ID / leakage / label columns out entirely (remainder="drop")

For an imbalanced target, SMOTE is offered as a *train-only* resampling step,
wired through an imbalanced-learn Pipeline so it never touches test / serving
data. See the SMOTE section below for why it can't live inside the
ColumnTransformer.

Fit on the TRAIN split only, then transform train/test/serving inputs.
Persist the fitted transformer alongside the model to keep train/serve
category maps and scaler stats identical.
"""

from __future__ import annotations
import logging
from collections import Counter

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_ingestion import DataIngestor

logger = logging.getLogger("preprocessing")


class Preprocessor:
    """Builds an (unfit) sklearn ColumnTransformer for churn features.

    Parameters
    ----------
    label_col : str
        Target column. Always excluded from features.
    exclude_cols : list[str] | None
        ID and leakage columns to keep out of the transformer. They can
        stay in the DataFrame — remainder="drop" ensures they never reach
        an encoder or scaler.
    categorical_cols : list[str] | None
        Manual override for categoricals. If None, detected by dtype
        (object / category).
    numeric_cols : list[str] | None
        Manual override for numerics. If None, everything not categorical
        (and not excluded) is treated as numeric.
    """

    def __init__(
        self,
        label_col: str,
        exclude_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        numeric_cols: list[str] | None = None,
    ) -> None:
        self.label_col = label_col
        self.exclude_cols = set(exclude_cols or []) | {label_col}
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

    def _split_columns(self, df) -> tuple[list[str], list[str]]:
        from pandas.api.types import is_numeric_dtype

        feature_cols = [c for c in df.columns if c not in self.exclude_cols]

        if self.categorical_cols is not None:
            categorical = [c for c in self.categorical_cols if c in feature_cols]
        else:
            # Anything not numeric is categorical. Using is_numeric_dtype (rather
            # than dtype == "object") keeps this correct across pandas versions:
            # newer pandas gives string columns the "str"/"string" dtype, which an
            # object-only check would miss and wrongly send to the numeric branch.
            categorical = [c for c in feature_cols if not is_numeric_dtype(df[c])]

        if self.numeric_cols is not None:
            numeric = [c for c in self.numeric_cols if c in feature_cols]
        else:
            numeric = [c for c in feature_cols if c not in categorical]

        return categorical, numeric

    def _null_report(self, df, categorical: list[str], numeric: list[str]) -> dict:
        """Per-column null counts for the feature columns, plus a rollup of
        whether the numeric side and the categorical side each need an
        imputer at all. Columns with zero nulls don't need one — adding an
        imputer that never fires is harmless but pointless, and skipping it
        keeps the pipeline (and its persisted state) simpler to reason about.
        """
        per_column = {}
        for col in numeric + categorical:
            n_null = int(df[col].isna().sum())
            per_column[col] = {
                "n_null": n_null,
                "null_share": (n_null / len(df)) if len(df) else 0.0,
            }

        numeric_needs_impute = any(per_column[c]["n_null"] > 0 for c in numeric)
        categorical_needs_impute = any(per_column[c]["n_null"] > 0 for c in categorical)

        return {
            "per_column": per_column,
            "numeric_needs_impute": numeric_needs_impute,
            "categorical_needs_impute": categorical_needs_impute,
        }

    def build(self, df) -> ColumnTransformer:
        """Return an unfit ColumnTransformer shaped to `df`'s columns.

        Whether an imputer step is included is decided by actually checking
        for nulls (via _null_report), not assumed. A column set with no
        missing values gets scale/encode only. The null report is stashed
        on self.null_report_ so callers can inspect what was found.
        """
        categorical, numeric = self._split_columns(df)
        null_report = self._null_report(df, categorical, numeric)
        self.null_report_ = null_report

        numeric_steps = []
        if null_report["numeric_needs_impute"]:
            logger.info("Nulls found in numeric features; adding median imputer.")
            numeric_steps.append(("impute", SimpleImputer(strategy="median")))
        else:
            logger.info("No nulls in numeric features; skipping imputer.")
        numeric_steps.append(("scale", StandardScaler()))
        numeric_pipe = Pipeline(numeric_steps)

        categorical_steps = []
        if null_report["categorical_needs_impute"]:
            logger.info(
                "Nulls found in categorical features; adding most-frequent imputer."
            )
            categorical_steps.append(
                ("impute", SimpleImputer(strategy="most_frequent"))
            )
        else:
            logger.info("No nulls in categorical features; skipping imputer.")
        categorical_steps.append(
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        )
        categorical_pipe = Pipeline(categorical_steps)

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
            ],
            remainder="drop",
        ).set_output(transform="pandas")

    def run(
        self,
        df,
        estimator=None,
        drop_duplicates: bool = True,
        duplicate_subset: list[str] | None = None,
        outlier_method: str = "iqr",
        imbalance_threshold: float = 0.35,
        smote_kwargs: dict | None = None,
        random_state: int = 42,
    ):
        """End-to-end preprocessing flow, in order:

          1. Remove duplicates.
          2. Check for nulls -> decide whether imputing is needed
             (done inside build(), per numeric/categorical side).
          3. Check outliers on the numeric features (report-only: flagged,
             not dropped — that call belongs to the caller/estimator choice).
          4. Check class imbalance -> resample (SMOTE) only if the minority
             class is below `imbalance_threshold`; otherwise continue
             without resampling.

        Returns (clean_df, transformer_or_pipeline, report).
        `transformer_or_pipeline` is a plain ColumnTransformer if `estimator`
        is None, or a full imblearn Pipeline (preprocess [+ SMOTE] + estimator)
        if you pass one.
        """
        df = df.copy()

        # 1. Duplicates
        dup_report = check_duplicates(df, subset=duplicate_subset)
        if drop_duplicates and dup_report["n_duplicates"]:
            before = len(df)
            df = df.drop(index=dup_report["duplicate_row_indices"]).reset_index(
                drop=True
            )
            logger.info(
                f"Removed {before - len(df)} duplicate rows ({before} -> {len(df)})."
            )
        elif dup_report["n_duplicates"]:
            logger.warning(
                f"{dup_report['n_duplicates']} duplicate rows found but "
                "drop_duplicates=False; leaving them in."
            )

        # 2. Nulls -> imputer decision happens inside build()
        transformer = self.build(df)
        categorical, numeric = self._split_columns(df)

        # 3. Outliers (numeric features only; report-only, nothing is dropped)
        if numeric:
            outlier_report = detect_outliers(df, columns=numeric, method=outlier_method)
        else:
            logger.info("No numeric feature columns; skipping outlier check.")
            outlier_report = {
                "method": outlier_method,
                "per_column": {},
                "n_rows_with_outliers": 0,
            }

        # 4. Imbalance -> decide on SMOTE, else continue without it
        imbalance_report = None
        use_smote = False
        if self.label_col in df.columns:
            y = df[self.label_col]
            imbalance_report = class_balance(y)
            use_smote = needs_resampling(y, threshold=imbalance_threshold)
            imbalance_report["needs_resampling"] = use_smote
            logger.info(
                f"Class balance: {imbalance_report['counts']}, "
                f"minority_share={imbalance_report['minority_share']:.3f}, "
                f"needs_resampling={use_smote}"
            )
        else:
            logger.warning(
                f"Label column '{self.label_col}' not found in df; skipping imbalance check."
            )

        report = {
            "duplicates": dup_report,
            "nulls": self.null_report_,
            "outliers": outlier_report,
            "imbalance": imbalance_report,
            "n_rows_final": len(df),
        }
        self.quality_report_ = report

        if estimator is not None:
            pipeline = build_training_pipeline(
                transformer,
                estimator,
                use_smote=use_smote,
                random_state=random_state,
                smote_kwargs=smote_kwargs,
            )
            return df, pipeline, report

        # No estimator given: just continue with the fitted-shape transformer.
        return df, transformer, report


# ---------------------------------------------------------------------------
# Data quality checks (duplicates, outliers)
# ---------------------------------------------------------------------------
# Run these on the TRAIN split (or full df, pre-split) before fitting anything.
# Duplicates inflate the effective weight of repeated rows and can straddle a
# train/test split, leaking identical rows across both. Outliers get flagged,
# not silently dropped or clipped here — that decision (drop, cap, winsorize,
# leave alone for tree models) belongs to the caller, not to a shared utility.


def check_duplicates(
    df,
    subset: list[str] | None = None,
    keep: str = "first",
) -> dict:
    """Report duplicate rows in `df`.

    Parameters
    ----------
    subset : list[str] | None
        Columns to consider when identifying duplicates. Use an ID column
        (e.g. ["user_id"]) to catch duplicate entities even if other fields
        drifted; leave None to require every column to match.
    keep : str
        Passed to pandas' duplicated(): "first"/"last" mark all but one copy
        as a duplicate, "False" marks every copy of a repeated row. Only
        used to decide which rows land in the returned "duplicate_rows"
        frame — the counts always reflect all repeated rows.

    Returns
    -------
    dict with:
        n_duplicates      : count of duplicate rows (all repeats beyond the
                             first occurrence, regardless of `keep`)
        duplicate_share    : n_duplicates / len(df)
        duplicate_row_indices : index labels of the duplicate rows returned
        duplicate_rows     : DataFrame of the duplicate rows themselves
    """
    if df.empty:
        logger.info("check_duplicates called on an empty DataFrame.")
        return {
            "n_duplicates": 0,
            "duplicate_share": 0.0,
            "duplicate_row_indices": [],
            "duplicate_rows": df,
        }

    if subset is not None:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            logger.error(f"check_duplicates: subset columns not in df: {missing}")
            raise KeyError(f"subset columns not in df: {missing}")

    # Full count of every repeated row (all copies past the first), independent
    # of `keep`, so n_duplicates is stable regardless of which rows get returned.
    all_dupe_mask = df.duplicated(subset=subset, keep="first")
    n_duplicates = int(all_dupe_mask.sum())
    duplicate_share = n_duplicates / len(df)

    returned_mask = df.duplicated(subset=subset, keep=keep)
    duplicate_rows = df[returned_mask]

    if n_duplicates:
        logger.warning(
            f"Found {n_duplicates} duplicate rows "
            f"({duplicate_share:.2%} of {len(df)}), subset={subset or 'all columns'}."
        )
    else:
        logger.info(f"No duplicate rows found (subset={subset or 'all columns'}).")

    return {
        "n_duplicates": n_duplicates,
        "duplicate_share": duplicate_share,
        "duplicate_row_indices": duplicate_rows.index.tolist(),
        "duplicate_rows": duplicate_rows,
    }


def detect_outliers(
    df,
    columns: list[str] | None = None,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_thresh: float = 3.0,
) -> dict:
    """Flag outliers per numeric column using IQR or z-score.

    Parameters
    ----------
    columns : list[str] | None
        Numeric columns to check. If None, every numeric column in `df` is
        used (via pandas.api.types.is_numeric_dtype).
    method : str
        "iqr"    -> flag values outside [Q1 - k*IQR, Q3 + k*IQR], robust to
                    skewed churn-style features (tenure, spend, counts).
        "zscore" -> flag values with |z| > z_thresh. Assumes roughly normal
                    data; sensitive to the outliers it's trying to detect,
                    so prefer "iqr" for skewed distributions.
    iqr_multiplier : float
        The "k" above. 1.5 is Tukey's standard fence; 3.0 is a looser
        "extreme outlier" fence if 1.5 flags too much on skewed data.
    z_thresh : float
        Threshold for the zscore method.

    Returns
    -------
    dict with:
        method        : method used
        per_column    : {col: {"n_outliers", "outlier_share", "bounds"/"stats",
                                "outlier_indices"}}
        any_outlier_mask : boolean Series, True where a row is an outlier on
                           at least one checked column
        n_rows_with_outliers : count of rows flagged on any column
    """
    from pandas.api.types import is_numeric_dtype, is_bool_dtype

    def _is_true_numeric(series) -> bool:
        # bool is technically numeric to pandas (is_numeric_dtype(bool) == True),
        # but IQR/z-score quantile math on a boolean Series crashes in numpy
        # ("boolean subtract... not supported"). Treat bool as non-numeric here.
        return is_numeric_dtype(series) and not is_bool_dtype(series)

    if df.empty:
        logger.info("detect_outliers called on an empty DataFrame.")
        return {
            "method": method,
            "per_column": {},
            "any_outlier_mask": df.index.to_series().iloc[0:0].astype(bool),
            "n_rows_with_outliers": 0,
        }

    if columns is None:
        columns = [c for c in df.columns if _is_true_numeric(df[c])]
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.error(f"detect_outliers: columns not in df: {missing}")
            raise KeyError(f"columns not in df: {missing}")

        bool_cols = [c for c in columns if is_bool_dtype(df[c])]
        if bool_cols:
            logger.info(
                f"detect_outliers: skipping boolean column(s) {bool_cols} "
                "(outlier detection doesn't apply to flags)."
            )
            columns = [c for c in columns if c not in bool_cols]

        non_numeric = [c for c in columns if not is_numeric_dtype(df[c])]
        if non_numeric:
            logger.error(f"detect_outliers: non-numeric columns given: {non_numeric}")
            raise TypeError(
                f"detect_outliers only supports numeric columns, got: {non_numeric}"
            )

    if method not in ("iqr", "zscore"):
        raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}")

    per_column: dict = {}
    combined_mask = pd_series_false_like(df)

    for col in columns:
        series = df[col]
        valid = series.dropna()

        if valid.empty:
            logger.warning(
                f"detect_outliers: column '{col}' is entirely NaN, skipping."
            )
            per_column[col] = {
                "n_outliers": 0,
                "outlier_share": 0.0,
                "outlier_indices": [],
            }
            continue

        if method == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            col_mask = (series < lower) | (series > upper)
            bounds_info = {
                "lower": float(lower),
                "upper": float(upper),
                "iqr": float(iqr),
            }
        else:  # zscore
            mean = valid.mean()
            std = valid.std(ddof=0)
            if std == 0 or np.isnan(std):
                logger.warning(
                    f"detect_outliers: column '{col}' has zero/NaN std, "
                    "skipping z-score check (no variation to flag against)."
                )
                per_column[col] = {
                    "n_outliers": 0,
                    "outlier_share": 0.0,
                    "outlier_indices": [],
                }
                continue
            z = (series - mean) / std
            col_mask = z.abs() > z_thresh
            bounds_info = {"mean": float(mean), "std": float(std), "z_thresh": z_thresh}

        col_mask = col_mask.fillna(False)
        n_outliers = int(col_mask.sum())

        per_column[col] = {
            "n_outliers": n_outliers,
            "outlier_share": n_outliers / len(df),
            **({"bounds": bounds_info} if method == "iqr" else {"stats": bounds_info}),
            "outlier_indices": df.index[col_mask].tolist(),
        }

        if n_outliers:
            logger.warning(
                f"Column '{col}': {n_outliers} outliers "
                f"({n_outliers / len(df):.2%}) via {method}."
            )

        combined_mask = combined_mask | col_mask

    n_rows_with_outliers = int(combined_mask.sum())
    logger.info(
        f"detect_outliers ({method}): {n_rows_with_outliers} rows flagged "
        f"across {len(columns)} column(s)."
    )

    return {
        "method": method,
        "per_column": per_column,
        "any_outlier_mask": combined_mask,
        "n_rows_with_outliers": n_rows_with_outliers,
    }


def pd_series_false_like(df):
    """All-False boolean Series aligned to df's index, used as an OR-accumulator."""
    import pandas as pd

    return pd.Series(False, index=df.index)


def data_quality_report(
    df,
    outlier_columns: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    outlier_method: str = "iqr",
) -> dict:
    """Convenience wrapper: run check_duplicates + detect_outliers together
    and log a one-line summary. Handy to call once right after ingestion,
    before splitting/fitting anything.
    """
    dup_report = check_duplicates(df, subset=duplicate_subset)
    outlier_report = detect_outliers(df, columns=outlier_columns, method=outlier_method)

    logger.info(
        f"Data quality: {dup_report['n_duplicates']} duplicate rows, "
        f"{outlier_report['n_rows_with_outliers']} rows with at least one "
        f"outlier (out of {len(df)} total rows)."
    )

    return {"duplicates": dup_report, "outliers": outlier_report}


# ---------------------------------------------------------------------------
# Class imbalance handling (SMOTE)
# ---------------------------------------------------------------------------
# SMOTE creates synthetic minority-class rows by interpolating between existing
# ones. Two rules this module enforces by construction:
#
#   1. Resample TRAIN ONLY. Never oversample the validation / test / serving
#      split — that leaks synthetic points into evaluation and quietly inflates
#      your metrics. imbalanced-learn's Pipeline guarantees this: samplers run
#      during .fit() but are skipped at .transform() / .predict(). A plain
#      sklearn Pipeline cannot hold a sampler at all — that's why the builder
#      below imports imblearn's Pipeline.
#
#   2. SMOTE needs an all-numeric matrix and uses distance to pick neighbours,
#      so it runs AFTER the ColumnTransformer (impute + OHE + scale). Order:
#          ColumnTransformer  ->  SMOTE (train-only)  ->  estimator
#
# Caveat on SMOTE + one-hot: plain SMOTE interpolates, so it can produce
# fractional values on the 0/1 one-hot columns (e.g. 0.4). It's common and
# usually fine for churn with linear / tree models, but if it bothers you, use
# SMOTENC instead — see build_training_pipeline_smotenc().


def class_balance(y) -> dict:
    """Report class counts and the minority share, to decide if SMOTE is worth it.

    Returns {"counts": {label: n}, "minority_share": float}.
    """
    counts = Counter(y)
    total = sum(counts.values())
    minority = min(counts.values()) / total if total else 0.0
    return {"counts": dict(counts), "minority_share": minority}


def needs_resampling(y, threshold: float = 0.35) -> bool:
    """True when the minority class is below `threshold` of the rows.

    Roughly balanced data (~0.5) returns False, so you can gate SMOTE on this:
    only resample when the target is actually skewed. 0.35 is a sane default
    for binary churn — tune to taste.
    """
    return class_balance(y)["minority_share"] < threshold


def build_training_pipeline(
    preprocessor: ColumnTransformer,
    estimator,
    use_smote: bool = True,
    random_state: int = 42,
    smote_kwargs: dict | None = None,
):
    """Wrap preprocessing (+ optional SMOTE) + estimator in one imblearn Pipeline.

    Order: ColumnTransformer -> SMOTE (train-only) -> estimator.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The (unfit) transformer from Preprocessor.build().
    estimator : sklearn-compatible classifier
        e.g. LogisticRegression, RandomForestClassifier, XGBClassifier.
    use_smote : bool
        Set False (e.g. when needs_resampling(y_train) is False) to get a plain
        preprocess + model pipeline with no resampling.
    smote_kwargs : dict | None
        Passed to SMOTE, e.g. {"k_neighbors": 3} for a tiny minority class, or
        {"sampling_strategy": 0.5} to only partially balance.

    Notes
    -----
    imblearn is imported lazily so this module still imports without it when you
    only need the Preprocessor. Install with: pip install imbalanced-learn
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(
            ("smote", SMOTE(random_state=random_state, **(smote_kwargs or {})))
        )
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def build_training_pipeline_smotenc(
    df,
    label_col: str,
    estimator,
    exclude_cols: list[str] | None = None,
    random_state: int = 42,
    smotenc_kwargs: dict | None = None,
):
    """SMOTENC variant: resample BEFORE encoding so categoricals stay valid.

    SMOTENC treats categorical columns natively (no fractional one-hots). It
    must run before OHE and needs no NaNs, so the order is:

        impute  ->  SMOTENC (train-only)  ->  OHE + scale  ->  estimator

    We impute up front (categoricals: most_frequent, numerics: median) so
    SMOTENC sees a clean matrix, then hand the still-categorical frame to the
    ColumnTransformer for encoding + scaling.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC

    from pandas.api.types import is_numeric_dtype

    exclude = set(exclude_cols or []) | {label_col}
    feature_cols = [c for c in df.columns if c not in exclude]
    categorical = [c for c in feature_cols if not is_numeric_dtype(df[c])]
    numeric = [c for c in feature_cols if c not in categorical]

    # Impute per type first (SMOTENC rejects NaNs), keeping columns intact.
    impute = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # After the impute step, columns are ordered [numeric..., categorical...],
    # so the categorical indices are the last len(categorical) positions.
    cat_index = list(range(len(numeric), len(numeric) + len(categorical)))

    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    return ImbPipeline(
        [
            ("impute", impute),
            (
                "smotenc",
                SMOTENC(
                    categorical_features=cat_index,
                    random_state=random_state,
                    **(smotenc_kwargs or {}),
                ),
            ),
            ("encode", encode),
            ("model", estimator),
        ]
    )


def save_preprocessor(transformer: ColumnTransformer, path: str) -> None:
    """Persist a fitted ColumnTransformer (joblib)."""
    joblib.dump(transformer, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a persisted ColumnTransformer."""
    return joblib.load(path)


if __name__ == "__main__":
    ingestor = None
    try:
        ingestor = DataIngestor()
        df = ingestor.load_customer_features(add_churn_labels=True)
        print(df.head())

        # load_customer_features() produces "churn_label", not "Churn".
        preprocessor = Preprocessor(label_col="churn_label", exclude_cols=["user_id"])

        # Full flow: dedupe -> null check (conditional imputer) -> outlier
        # report -> imbalance check -> continue (with SMOTE only if warranted).
        clean_df, transformer, report = preprocessor.run(df)

        print("Duplicates removed:", report["duplicates"]["n_duplicates"])
        print("Rows with outliers:", report["outliers"]["n_rows_with_outliers"])
        print("Class balance:", report["imbalance"])

        # Fit the transformer here (train split only, in real usage) to see
        # the actual output shape and feature names.
        X = clean_df.drop(columns=["churn_label"])
        Xt = transformer.fit_transform(X)

        print("Transformed shape:", Xt.shape)
        print("Feature names:", list(transformer.get_feature_names_out()))

    except Exception:
        logger.exception("Preprocessing run failed.")
        raise
    finally:
        if ingestor is not None:
            ingestor.close()
