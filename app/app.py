"""Flask API serving the latest Bitcoin forecast model."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

from src.model_service import ModelService


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        MODEL_DIR=os.getenv("MODEL_DIR", "/app/models"),
        MAX_FORECAST_DAYS=int(os.getenv("MAX_FORECAST_DAYS", "30")),
        MODEL_STAGE=os.getenv("MODEL_STAGE", "production"),
    )
    if test_config:
        app.config.update(test_config)

    started_at = time.monotonic()
    service = ModelService(Path(app.config["MODEL_DIR"]))
    service.load()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            uptime_seconds=round(time.monotonic() - started_at, 3),
            model_version=service.version,
            model_loaded=service.is_loaded,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify(error="JSON body must be an object"), 400

        days = payload.get("forecast_days", 7)
        if isinstance(days, bool) or not isinstance(days, int):
            return jsonify(error="forecast_days must be an integer"), 400
        if not 1 <= days <= app.config["MAX_FORECAST_DAYS"]:
            return (
                jsonify(error=f"forecast_days must be between 1 and {app.config['MAX_FORECAST_DAYS']}"),
                400,
            )

        history = payload.get("history")
        if history is not None:
            if not isinstance(history, list) or len(history) < service.lags:
                return jsonify(error=f"history must contain at least {service.lags} prices"), 400
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in history):
                return jsonify(error="history values must be positive numbers"), 400

        try:
            result = service.predict(days=days, history=history)
            return jsonify(result)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(error="prediction failed"), 500

    @app.get("/metrics")
    def metrics():
        return jsonify(service.metrics)

    @app.post("/reload")
    def reload_model():
        expected = os.getenv("RELOAD_TOKEN", "local-development-token")
        supplied = request.headers.get("X-Reload-Token", "")
        if supplied != expected:
            return jsonify(error="unauthorized"), 401
        with lock:
            service.load()
        return jsonify(status="reloaded", model_version=service.version)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
