# tests/test_model_factory.py

import pytest
from tune_model import ModelFactory
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


def test_create_xgboost():
    model = ModelFactory.create_model("xgboost", {"n_estimators": 100, "max_depth": 3})
    assert isinstance(model, XGBClassifier)


def test_create_svm():
    model = ModelFactory.create_model("svm", {"C": 1.0, "kernel": "linear"})
    assert isinstance(model, SVC)


def test_create_sgd():
    model = ModelFactory.create_model("sgd", {"alpha": 0.001})
    assert isinstance(model, SGDClassifier)


def test_create_knn():
    model = ModelFactory.create_model("knn", {"n_neighbors": 5, "weights": "uniform"})
    assert isinstance(model, KNeighborsClassifier)


def test_create_neural_network_requires_input_dim():
    with pytest.raises(ValueError, match="input_dim required for neural network"):
        ModelFactory.create_model(
            "neural_network", {"hidden_layers": (64, 32), "optimizer": "adam"}
        )


def test_create_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        ModelFactory.create_model("banana_model", {})
