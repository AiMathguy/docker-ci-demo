"""Churn drift-monitoring DAG: detect drift, branch, retrain or cooldown."""

from __future__ import annotations
from airflow import DAG
import logging
from datetime import datetime, timedelta

import pendulum
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator


def get_data(**context):
    from src.data_ingestion import DataIngestor

    ingestor = DataIngestor()
    df = ingestor.load_customer_features(add_churn_labels=True)
    # push to XCom for downstream tasks (serialize as needed)
    context["ti"].xcom_push(key="raw_data_ready", value=True)
    logging.info("Loaded %d rows", len(df))


def preprocess_data(**context):
    from src.preprocessing import Preprocessor

    pre = Preprocessor(label_col="Churn", exclude_cols=["customerID"])
    # ... run preprocessing, persist output where drift_check can read it
    logging.info("Preprocessing complete")


def run_drift_check(**context):
    from src.drift_check import detect_drift, load_reference

    # load reference + current window, call detect_drift, push report
    # report = detect_drift(reference_df, current_df, numeric_cols, categorical_cols)
    # context["ti"].xcom_push(key="report", value=report)
    logging.info("Drift check complete")


def decide_branch(**context):
    report = context["ti"].xcom_pull(task_ids="drift_check", key="report")
    if report and report["should_retrain"]:
        return "train_model"
    return "cooldown_hold"


def trigger_retrain(**context):
    from src.train_model import ChurnModelManager

    logging.info("Retraining due to detected drift.")
    manager = ChurnModelManager()
    manager.fetch_data()
    return manager.run_training_pipeline()


def cooldown_hold(**context):
    logging.info("No drift — holding, no retrain.")


with DAG(
    dag_id="drift_check_dag",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Drift monitoring for churn model",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["MLOps", "drift_check"],
) as dag:

    get_data_task = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
    )

    preprocessor_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    drift_check_task = PythonOperator(
        task_id="drift_check",
        python_callable=run_drift_check,
    )

    drift_branch_task = BranchPythonOperator(
        task_id="branching",
        python_callable=decide_branch,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=trigger_retrain,
    )

    notify_task = EmptyOperator(
        task_id="notify",
        trigger_rule="none_failed_min_one_success",
    )

    cooldown_task = PythonOperator(
        task_id="cooldown_hold",
        python_callable=cooldown_hold,
    )

    get_data_task >> preprocessor_task >> drift_check_task >> drift_branch_task
    drift_branch_task >> [train_model_task, cooldown_task]
    train_model_task >> notify_task
