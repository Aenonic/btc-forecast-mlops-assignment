"""Feature engineering utilities for BTC forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"].astype(float)
    result["return_1d"] = close.pct_change()
    result["ma_7"] = close.rolling(7).mean()
    result["ma_30"] = close.rolling(30).mean()
    result["volatility_7"] = result["return_1d"].rolling(7).std()
    result["volatility_30"] = result["return_1d"].rolling(30).std()
    return result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def lag_matrix(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= lags:
        raise ValueError(f"At least {lags + 1} close prices are required")
    x_rows, y_rows = [], []
    for index in range(lags, len(values)):
        x_rows.append(values[index - lags : index])
        y_rows.append(values[index])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)
