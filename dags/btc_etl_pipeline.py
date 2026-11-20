"""Daily BTC data ingestion, feature generation, training, validation, and API reload."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow.decorators import dag, task


PROJECT = Path("/opt/airflow/project")


@dag(
    dag_id="btc_daily_forecast_pipeline",
    schedule="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["bitcoin", "mlops", "forecast"],
)
def btc_pipeline():
    @task
    def fetch_data() -> str:
        from src.etl_fetch import fetch_btc_data

        return str(fetch_btc_data(PROJECT / "data/raw/btc_usd.csv"))

    @task
    def preprocess(path: str) -> str:
        from src.features import add_technical_features

        source = pd.read_csv(path)
        output = PROJECT / "data/processed/btc_features.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        add_technical_features(source).to_csv(output, index=False)
        return str(output)

    @task
    def train_and_evaluate(raw_path: str, _features_path: str) -> dict:
        from src.train_forecast import train_model

        return train_model(raw_path, PROJECT / "models")

    @task
    def validate(metrics_and_metadata: dict) -> dict:
        metrics = metrics_and_metadata["metrics"]
        if metrics["mape_percent"] > 30:
            raise ValueError(f"MAPE {metrics['mape_percent']} exceeds the 30% registration threshold")
        return metrics

    @task
    def reload_api(metrics: dict) -> str:
        token = __import__("os").getenv("RELOAD_TOKEN", "local-development-token")
        response = requests.post(
            "http://api:5000/reload",
            headers={"X-Reload-Token": token},
            timeout=15,
        )
        response.raise_for_status()
        return json.dumps({"api": response.json(), "metrics": metrics})

    raw = fetch_data()
    features = preprocess(raw)
    trained = train_and_evaluate(raw, features)
    reload_api(validate(trained))


btc_pipeline()
