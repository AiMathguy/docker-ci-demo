import os
import pytest
from sqlalchemy import create_engine
import pytest

pytestmark = pytest.mark.skip(
    reason="Temporarily skipped while ML pipeline tests are being refactored"
)


def test_database_url_exists():
    db_url = os.getenv("DATABASE_URL")
    assert db_url is not None
    assert db_url.strip() != ""


def test_engine_can_be_created():
    db_url = os.getenv("DATABASE_URL")
    assert db_url is not None
    engine = create_engine(db_url, pool_pre_ping=True)
    assert engine is not None


def test_tune_model_imports():
    import tune_model

    assert tune_model is not None


def test_feature_columns_exists():
    from dashboard_ui.ml_model import FEATURE_COLUMNS

    assert isinstance(FEATURE_COLUMNS, list)
    assert len(FEATURE_COLUMNS) > 0
