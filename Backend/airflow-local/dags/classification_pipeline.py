"""Churn drift-monitoring DAG: detect drift, branch, retrain or cooldown."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.providers.smtp.notifications.smtp import SmtpNotifier
import pendulum
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
from preprocessing import Preprocessor
from src.data_ingestion import DataIngestor
from src.train_model import ChurnModelManager
from src.drift_check import detect_drift
import src.tune_model


def skip_retrain():
    """ "
    This function is used to skip the retraining process.
    """
    logging.info("Skipping retraining due to cooldown period.")
    return "cooldown_hold"


def trigger_retrain():
    logging.info("Retraining due to detected drift.")
    manager = ChurnModelManager()
    manager.fetch_data()
    results = manager.run_training_pipeline()  # but see below
    return results


def decide_branch(**context):
    report = context["ti"].xcom_pull(task_ids="drift_check")
    if report["should_retrain"]:
        return "email_notification"
    else:
        return "cooldown_hold"
        # <- must match a downstream task_id exactly


def notifier():
    logging.info("Drift detected. Notifying stakeholders.")
    EmptyOperator(
        task_id="task",
        on_failure_callback=SmtpNotifier(
            from_email="tbilankulu@linkfields.com", to="aawasthi@linkfields.com"
        ),
    )


def skip_notification():
    logging.info("skipping notification as no drift detected.")
    return "cooldown_hold"


with DAG(
    dag_id="drift_check_dag",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="scheduling for drift checking",
    schedule=timedelta(days=60),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["MLOps", "drift_check"],
) as dag:

    get_data_task = PythonOperator(
        task_id="get_data",
        python_callable=DataIngestor.load_customer_features(add_churn_labels=True),
    )
preprocessor_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=Preprocessor(label_col="Churn", exclude_cols=["customerID"]),
)

drift_branch_task = BranchPythonOperator(
    task_id="branching", python_callable=decide_branch
)

train_model_task = PythonOperator(
    task_id="train_model", python_callable=trigger_retrain
)

email = EmptyOperator(
    task_id="email_notification",
    on_failure_callback=SmtpNotifier(
        from_email="tbilankulu@linkfields.com",
        to="thatobilankulu01@gmail.com",
        subject="Drift detected in Churn Model",
        trigger_rule="none_failed",
    ),
)
skip_retrain_task = PythonOperator(
    task_id="cooldown_hold", python_callable=skip_retrain
)

get_data_task >> preprocessor_task >> drift_branch_task >> [email, skip_retrain_task]
email >> train_model_task
