import streamlit as st
import logging
from pathlib import Path
import logging
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ✅ Define BASE_DIR first
BASE_DIR = Path(__file__).resolve().parent.parent

# Now you can use it
LOGS_DIR = BASE_DIR / "logs"


MLFLOW_TRACKING_URI = "file:./mlruns"

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}


def set_page_config():
    st.set_page_config(
        page_title="CEO Growth & Churn Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# madewithml/config.py
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default=None, required: bool = False):
    value = os.getenv(key, default)

    if required and value is None:
        raise RuntimeError(f"{key} is required but not set in .env")

    return value


# ===== CONFIG VALUES =====

DATABASE_URL = get_env("DATABASE_URL", required=True)

MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI", default="file:./mlruns")

JWT_SECRET = get_env("JWT_SECRET", default="dev_secret")
ENV = get_env("ENV", default="development")
