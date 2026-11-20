"""Thread-safe model artifact loader and recursive forecast service."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np


class ModelService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.metadata: dict = {}
        self.metrics: dict = {"status": "model_not_trained"}

    @property
    def lags(self) -> int:
        return int(self.metadata.get("lags", 30))

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unavailable"))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        model_path = self.model_dir / "btc_forecast.joblib"
        metadata_path = self.model_dir / "metadata.json"
        metrics_path = self.model_dir / "metrics.json"
        self.model = joblib.load(model_path) if model_path.exists() else None
        self.metadata = self._read_json(metadata_path, {})
        self.metrics = self._read_json(metrics_path, {"status": "model_not_trained"})

    @staticmethod
    def _read_json(path: Path, fallback: dict) -> dict:
        if not path.exists():
            return fallback
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def predict(self, days: int, history: list[float] | None = None) -> dict:
        if self.model is None:
            raise RuntimeError("model is not available; run the training pipeline first")
        values = list(history if history is not None else self.metadata.get("recent_close", []))
        if len(values) < self.lags:
            raise RuntimeError("model artifact does not contain enough recent history")

        predictions = []
        for _ in range(days):
            features = np.asarray(values[-self.lags :], dtype=float).reshape(1, -1)
            next_price = max(0.01, float(self.model.predict(features)[0]))
            predictions.append(round(next_price, 2))
            values.append(next_price)

        start = date.today() + timedelta(days=1)
        return {
            "symbol": "BTC-USD",
            "model_version": self.version,
            "forecast_days": days,
            "predictions": [
                {"date": (start + timedelta(days=i)).isoformat(), "close": price}
                for i, price in enumerate(predictions)
            ],
        }
