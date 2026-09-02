import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger("data_ingestion")
logger.propagate = False  # avoid duplicate lines if basicConfig also attaches a handler

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler("data_ingestion_error.log")
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


class DataIngestor:

    def __init__(self, output_dir: str = "backend/data"):
        load_dotenv()
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            logger.error("DATABASE_URL not found in environment.")
            raise RuntimeError("DATABASE_URL not found in environment.")

        logger.info("Using database URL: %s", self._redact_url(self.db_url))

        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
        except SQLAlchemyError:
            logger.exception("Failed to create database engine.")
            raise

        try:
            self.output_path = Path(output_dir).resolve()
            self.output_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception(f"Failed to create output directory: {output_dir}")
            self.engine.dispose()
            raise

        try:
            self.tables = inspect(self.engine).get_table_names()
            logger.info(f"Discovered {len(self.tables)} tables: {sorted(self.tables)}")
        except OperationalError:
            logger.exception("Could not connect to the database to inspect tables.")
            self.engine.dispose()
            raise
        except SQLAlchemyError:
            logger.exception("Unexpected error while inspecting database schema.")
            self.engine.dispose()
            raise

    @staticmethod
    def _redact_url(url: str) -> str:
        # Avoid printing credentials to logs, e.g. postgresql://user:pass@host/db
        if "@" in url and "//" in url:
            scheme, rest = url.split("//", 1)
            if "@" in rest:
                _, host_part = rest.split("@", 1)
                return f"{scheme}//***:***@{host_part}"
        return url

    def close(self):
        try:
            self.engine.dispose()
            logger.info("Database engine disposed.")
        except SQLAlchemyError:
            logger.exception("Error while disposing database engine.")

    def list_available_tables(self) -> list[str]:
        return sorted(self.tables)

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.tables

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.table_exists(table_name):
            logger.error(f"Table not found: {table_name}")
            raise ValueError(f"Table not found: {table_name}")

        query = f"SELECT * FROM {table_name}"
        if limit is not None:
            query += f" LIMIT {limit}"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
        except OperationalError:
            logger.exception(
                f"Database connection error while loading table {table_name}."
            )
            raise
        except SQLAlchemyError:
            logger.exception(f"Query failed while loading table {table_name}.")
            raise
        except Exception:
            logger.exception(f"Unexpected error while loading table {table_name}.")
            raise

        logger.info(f"Loaded {len(df)} rows from {table_name}")
        return df

    def load_customer_features(self, add_churn_labels: bool = True) -> pd.DataFrame:
        try:
            df = self.load_table("customer_features")
        except Exception:
            logger.exception("Error occurred while loading customer_features table.")
            raise

        if df.empty:
            logger.error("customer_features table is empty.")
            raise RuntimeError("customer_features table is empty")

        if "has_active_subscription" not in df.columns:
            if "churned" in df.columns:
                df["has_active_subscription"] = (df["churned"] == 0).astype(int)
            else:
                logger.warning(
                    "Neither 'has_active_subscription' nor 'churned' found; "
                    "defaulting has_active_subscription to 0 for all rows."
                )
                df["has_active_subscription"] = 0

        if add_churn_labels:
            if "churn_probability" in df.columns:
                df["churn_label"] = (df["churn_probability"] >= 0.5).astype(int)
                logger.info("churn_label derived from churn_probability column.")
            else:
                required_cols = [
                    "is_verified",
                    "days_since_last_login",
                    "login_count_7d",
                    "failed_login_attempts",
                ]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    logger.error(
                        f"Cannot derive churn_label: missing required columns {missing}. "
                        f"Available columns: {df.columns.tolist()}"
                    )
                    raise KeyError(
                        f"customer_features is missing columns needed to derive churn_label: {missing}"
                    )

                conditions = pd.DataFrame(
                    {
                        "not_verified": df["is_verified"] == 0,
                        "inactive_60d": df["days_since_last_login"].fillna(999) > 60,
                        "no_recent_logins": df["login_count_7d"].fillna(0) == 0,
                        "many_failed_logins": df["failed_login_attempts"] >= 3,
                        "no_active_subscription": df["has_active_subscription"] == 0,
                    }
                )

                df["churn_score"] = conditions.sum(axis=1)
                df["churn_label"] = (df["churn_score"] >= 3).astype(int)

            logger.info(
                f"Churn label distribution: {df['churn_label'].value_counts().to_dict()}"
            )

        logger.info(f"customer_features columns: {df.columns.tolist()}")
        return df

    def export_all(self):
        failed_tables = []

        try:
            available_tables = set(self.list_available_tables())
            logger.info(f"Available tables: {sorted(available_tables)}")

            for table in self.tables:
                if table not in available_tables:
                    logger.error(f"Table not found: {table}")
                    failed_tables.append(table)
                    continue

                try:
                    self._export_table(table)
                except Exception:
                    logger.exception(f"Failed to export {table}.")
                    failed_tables.append(table)

            # NOTE: original code referenced a "churn_labels" table that does not
            # exist in this database. Predictions/scores live in "customer_predictions".
            # Update the table name below if your actual predictions table differs.
            try:
                master_df = self.build_master_df(
                    self.load_table("users"),
                    self.load_table("customer_features"),
                    self.load_table("customer_predictions"),
                )
                master_path = self.output_path / "master_customer_data.csv"
                master_df.to_csv(master_path, index=False)
                logger.info(f"Exported master_customer_data -> {master_path}")
            except Exception:
                logger.exception("Failed to build or export master_customer_data.")
                failed_tables.append("master_customer_data")

            if failed_tables:
                raise RuntimeError(f"Export failed for tables: {failed_tables}")

            logger.info("All requested tables exported successfully.")

        finally:
            self.close()

    def export_table(self, table_name: str):
        if not self.table_exists(table_name):
            logger.error(f"Table not found: {table_name}")
            raise ValueError(f"Table not found: {table_name}")
        self._export_table(table_name)

    def _export_table(self, table_name: str):
        file_path = self.output_path / f"{table_name}.csv"
        tmp_path = self.output_path / f"{table_name}.csv.tmp"
        total_rows = 0
        first_chunk = True

        try:
            with self.engine.connect() as conn:
                for chunk in pd.read_sql_query(
                    f"SELECT * FROM {table_name}", conn, chunksize=5000
                ):
                    chunk.to_csv(
                        tmp_path,
                        mode="w" if first_chunk else "a",
                        index=False,
                        header=first_chunk,
                    )
                    total_rows += len(chunk)
                    first_chunk = False
        except OperationalError:
            logger.exception(f"Database connection error while exporting {table_name}.")
            self._cleanup_tmp(tmp_path)
            raise
        except SQLAlchemyError:
            logger.exception(f"Query failed while exporting {table_name}.")
            self._cleanup_tmp(tmp_path)
            raise
        except OSError:
            logger.exception(f"Filesystem error while writing CSV for {table_name}.")
            self._cleanup_tmp(tmp_path)
            raise

        # Only replace the real file once the export fully succeeded,
        # so a failed run never leaves a partial/corrupt CSV in place.
        if total_rows == 0:
            logger.warning(f"Table {table_name} is empty. No CSV rows written.")
            self._cleanup_tmp(tmp_path)
            return

        try:
            tmp_path.replace(file_path)
        except OSError:
            logger.exception(f"Failed to finalize CSV for {table_name}.")
            self._cleanup_tmp(tmp_path)
            raise

        logger.info(f"Exported {table_name}: {total_rows} rows -> {file_path}")

    @staticmethod
    def _cleanup_tmp(tmp_path: Path):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.warning(f"Could not remove temp file {tmp_path}")

    def build_master_df(
        self,
        users: pd.DataFrame,
        features: pd.DataFrame,
        preds: pd.DataFrame,
    ) -> pd.DataFrame:
        for df_name, df, key_col in (
            ("users", users, "id"),
            ("features", features, "user_id"),
            ("preds", preds, "user_id"),
        ):
            if key_col not in df.columns:
                logger.error(f"'{key_col}' column missing from {df_name} dataframe.")
                raise KeyError(f"'{key_col}' column missing from {df_name} dataframe")

        try:
            users = users.copy()
            features = features.copy()
            preds = preds.copy()

            users["id"] = users["id"].astype(str)
            features["user_id"] = features["user_id"].astype(str)
            preds["user_id"] = preds["user_id"].astype(str)

            df = users.merge(
                features, left_on="id", right_on="user_id", how="left"
            ).merge(
                preds,
                left_on="id",
                right_on="user_id",
                how="left",
                suffixes=("", "_pred"),
            )

            df.drop(columns=[c for c in df.columns if c == "user_id"], inplace=True)
        except Exception:
            logger.exception("Failed to build master dataframe.")
            raise

        logger.info(
            f"Built master dataframe with {len(df)} rows, {len(df.columns)} columns."
        )
        return df


if __name__ == "__main__":
    ingestor = None
    try:
        ingestor = DataIngestor()
        df = ingestor.load_customer_features(add_churn_labels=True)
        print(df.head())
        ingestor.export_all()
    except Exception:
        logger.exception("Data ingestion run failed.")
        raise
    finally:
        if ingestor is not None:
            ingestor.close()
