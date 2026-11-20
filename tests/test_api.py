import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from app.app import create_app


def _model_dir(tmp_path: Path) -> Path:
    lags = 5
    x = np.asarray([[i + j for j in range(lags)] for i in range(1, 30)], dtype=float)
    y = np.asarray([i + lags for i in range(1, 30)], dtype=float)
    model = Ridge(alpha=1.0).fit(x, y)
    joblib.dump(model, tmp_path / "btc_forecast.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"version": "test-v1", "lags": lags, "recent_close": [10, 11, 12, 13, 14]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"mae": 10.5, "rmse": 12.0}), encoding="utf-8"
    )
    return tmp_path


def test_health(tmp_path):
    client = create_app(
        {"TESTING": True, "MODEL_DIR": str(_model_dir(tmp_path))}
    ).test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["model_loaded"] is True


def test_predict_defaults_to_seven_days(tmp_path):
    client = create_app(
        {"TESTING": True, "MODEL_DIR": str(_model_dir(tmp_path))}
    ).test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert response.json["forecast_days"] == 7
    assert len(response.json["predictions"]) == 7


def test_predict_validates_days_and_history(tmp_path):
    client = create_app(
        {"TESTING": True, "MODEL_DIR": str(_model_dir(tmp_path))}
    ).test_client()
    assert client.post("/predict", json={"forecast_days": 0}).status_code == 400
    assert client.post("/predict", json={"forecast_days": "7"}).status_code == 400
    assert client.post("/predict", json={"history": [1, 2]}).status_code == 400


def test_metrics(tmp_path):
    client = create_app(
        {"TESTING": True, "MODEL_DIR": str(_model_dir(tmp_path))}
    ).test_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json["mae"] == 10.5
