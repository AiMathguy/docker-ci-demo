# tests/test_final_trainer.py
import numpy as np
import optuna

from pathlib import Path
from tune_model import FinalModelTrainer
import pytest

pytestmark = pytest.mark.skip(
    reason="Temporarily skipped while ML pipeline tests are being refactored"
)


def test_clean_params_for_prefixed_model():
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"model": "svm", "sv_C": 1.0, "sv_kernel": "linear"})
    study.optimize(lambda trial: 0.9, n_trials=1)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    trainer = FinalModelTrainer(X, y, study)
    cleaned = trainer._clean_params(
        "svm", {"model": "svm", "sv_C": 1.0, "sv_kernel": "linear"}
    )

    assert cleaned == {"C": 1.0, "kernel": "linear"}


def test_train_and_save_sklearn_model(tmp_path):
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {"model": "knn", "knn_n_neighbors": 3, "knn_weights": "uniform"}
    )
    study.optimize(lambda trial: 0.8, n_trials=1)

    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)

    trainer = FinalModelTrainer(X, y, study)
    model = trainer.train_and_save(str(tmp_path / "best_model"))

    assert model is not None
    assert Path(tmp_path / "best_model.pkl").exists()
