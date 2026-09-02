# tests/test_smoke_pipeline.py

import pandas as pd
import numpy as np
import tune_model as tm

import pytest

pytestmark = pytest.mark.skip(
    reason="Temporarily skipped while ML pipeline tests are being refactored"
)


def test_main_smoke(monkeypatch):
    fake_df = pd.DataFrame(
        {
            "is_verified": [1, 0, 1, 1, 0, 1],
            "days_since_last_login": [5, 30, 50, 2, 40, 1],
            "login_count_7d": [3, 0, 1, 5, 0, 6],
            "failed_login_attempts": [0, 4, 1, 0, 3, 0],
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_b": [1, 2, 3, 4, 5, 6],
        }
    )

    monkeypatch.setenv("DATABASE_URL", "sqlite:///fake.db")
    monkeypatch.setattr(tm, "FEATURE_COLUMNS", ["feature_a", "feature_b"])

    def fake_read_sql(query, conn):
        return fake_df

    monkeypatch.setattr("pandas.read_sql", fake_read_sql)

    class DummyStudy:
        best_params = {"model": "knn", "knn_n_neighbors": 3, "knn_weights": "uniform"}
        best_value = 0.75

    monkeypatch.setattr(tm.TuningOrchestrator, "run_tuning", lambda self: DummyStudy())

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), 0.7), np.full(len(X), 0.7)])

    monkeypatch.setattr(
        tm.FinalModelTrainer, "train_and_save", lambda self, output_path: DummyModel()
    )

    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("mlflow.set_experiment", lambda name: None)
    monkeypatch.setattr(
        "mlflow.start_run", lambda **kwargs: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr("mlflow.log_params", lambda params: None)
    monkeypatch.setattr("mlflow.log_metric", lambda key, value: None)
    monkeypatch.setattr("mlflow.log_artifacts", lambda *args, **kwargs: None)

    tm.main()
