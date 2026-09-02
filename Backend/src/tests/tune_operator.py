# tests/test_tuning_orchestrator.py
import numpy as np
from tune_model import TuningOrchestrator


def test_orchestrator_init():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    tuner = TuningOrchestrator(X, y, n_trials=1)

    assert tuner.X_train.shape == (20, 3)
    assert len(tuner.y_train) == 20
    assert tuner.n_trials == 1
    assert tuner.study is None


def test_run_tuning_smoke(monkeypatch):
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    tuner = TuningOrchestrator(X, y, n_trials=1)

    def fake_objective(trial):
        return 0.8

    monkeypatch.setattr(tuner, "_objective", fake_objective)

    study = tuner.run_tuning()

    assert study is not None
    assert len(study.trials) == 1
    assert study.best_value == 0.8
