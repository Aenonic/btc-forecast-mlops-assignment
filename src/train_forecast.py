"""Train and evaluate an autoregressive Ridge forecast model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import add_technical_features, lag_matrix


def _recursive_forecast(
    model: Ridge, history: list[float], days: int, lags: int
) -> np.ndarray:
    result = []
    values = list(history)
    for _ in range(days):
        prediction = max(
            0.01, float(model.predict(np.asarray(values[-lags:]).reshape(1, -1))[0])
        )
        values.append(prediction)
        result.append(prediction)
    return np.asarray(result)


def train_model(
    input_path: str | Path = "data/raw/btc_usd.csv",
    output_dir: str | Path = "models",
    lags: int = 30,
    test_days: int = 7,
) -> dict:
    frame = pd.read_csv(input_path)
    frame.columns = [str(c).lower() for c in frame.columns]
    engineered = add_technical_features(frame)
    close = engineered["close"].astype(float).to_numpy()
    if len(close) <= lags + test_days:
        raise ValueError("Dataset is too small for training and 7-day validation")

    train_values = close[:-test_days]
    actual = close[-test_days:]
    x_train, y_train = lag_matrix(train_values, lags)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    predicted = _recursive_forecast(model, train_values.tolist(), test_days, lags)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_percent": round(mape, 4),
        "validation_days": test_days,
        "evaluated_at": now.isoformat(),
    }
    metadata = {
        "symbol": "BTC-USD",
        "version": version,
        "algorithm": "Ridge autoregression",
        "lags": lags,
        "trained_rows": len(train_values),
        "recent_close": close[-lags:].tolist(),
        "trained_at": now.isoformat(),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "btc_forecast.joblib")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/btc_usd.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    result = train_model(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
