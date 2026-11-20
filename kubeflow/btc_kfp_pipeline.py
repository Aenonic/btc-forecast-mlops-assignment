"""Kubeflow Pipelines v2 specification for the BTC forecasting workflow."""

from kfp import compiler, dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests==2.32.5", "pandas==2.3.3"],
)
def ingest_data(output_data: dsl.Output[dsl.Dataset]):
    from datetime import datetime, timezone

    import pandas as pd
    import requests

    response = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1000},
        headers={"User-Agent": "btc-forecast-kfp/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    candles = response.json()
    data = pd.DataFrame(
        {
            "Date": [datetime.fromtimestamp(row[0] / 1000, timezone.utc).date() for row in candles],
            "Open": [float(row[1]) for row in candles],
            "High": [float(row[2]) for row in candles],
            "Low": [float(row[3]) for row in candles],
            "Close": [float(row[4]) for row in candles],
            "Volume": [float(row[5]) for row in candles],
        }
    ).dropna()
    data.to_csv(output_data.path, index=False)


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas==2.3.3"])
def generate_features(input_data: dsl.Input[dsl.Dataset], output_data: dsl.Output[dsl.Dataset]):
    import pandas as pd

    frame = pd.read_csv(input_data.path)
    close = frame["Close"].astype(float)
    frame["ma_7"] = close.rolling(7).mean()
    frame["ma_30"] = close.rolling(30).mean()
    frame["volatility_7"] = close.pct_change().rolling(7).std()
    frame.dropna().to_csv(output_data.path, index=False)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.3.3", "scikit-learn==1.7.2", "joblib==1.5.2"],
)
def train_model(
    input_data: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model], metrics: dsl.Output[dsl.Metrics]
):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    values = pd.read_csv(input_data.path)["Close"].astype(float).to_numpy()
    lags = 30
    x = np.asarray([values[i - lags : i] for i in range(lags, len(values))])
    y = values[lags:]
    split = len(x) - 7
    fitted = Ridge(alpha=1.0).fit(x[:split], y[:split])
    prediction = fitted.predict(x[split:])
    mae = float(mean_absolute_error(y[split:], prediction))
    joblib.dump(fitted, model.path)
    metrics.log_metric("mae", mae)


@dsl.pipeline(name="btc-seven-day-forecast", description="Daily BTC forecast training pipeline")
def btc_forecast_pipeline():
    ingestion = ingest_data()
    features = generate_features(input_data=ingestion.outputs["output_data"])
    train_model(input_data=features.outputs["output_data"])


if __name__ == "__main__":
    compiler.Compiler().compile(btc_forecast_pipeline, "kubeflow/btc_kfp_pipeline.yaml")
