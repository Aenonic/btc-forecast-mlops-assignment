"""Fetch historical BTC-USD daily candles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


def fetch_btc_data(
    output_path: str | Path = "data/raw/btc_usd.csv", period: str = "5y"
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Binance's public endpoint requires no API key and provides daily OHLCV candles.
    # Its 1,000-row response gives roughly 2.7 years of training history.
    url = "https://api.binance.com/api/v3/klines"
    response = requests.get(
        url,
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1000},
        headers={"User-Agent": "btc-forecast-mlops/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    candles = response.json()
    frame = pd.DataFrame(
        {
            "date": [
                datetime.fromtimestamp(row[0] / 1000, timezone.utc).date()
                for row in candles
            ],
            "open": [float(row[1]) for row in candles],
            "high": [float(row[2]) for row in candles],
            "low": [float(row[3]) for row in candles],
            "close": [float(row[4]) for row in candles],
            "volume": [float(row[5]) for row in candles],
        }
    ).dropna()
    if frame.empty:
        raise RuntimeError("No BTC-USDT rows were returned by Binance")
    frame.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/btc_usd.csv")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    path = fetch_btc_data(args.output, args.period)
    print(f"Saved BTC-USD data to {path}")


if __name__ == "__main__":
    main()
