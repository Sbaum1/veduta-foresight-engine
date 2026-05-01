# ==================================================
# FILE: foresight_engine/models/naive.py
# VERSION: 3.0.0
# MODEL: NAIVE (Random Walk / Persistence)
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# v3.0.0 FIXES:
#   - Confidence level now handled by interpolating _get_z() function.
#     Was: hard-coded dict lookup — crash on any value not in {0.80, 0.90, 0.95}
#     Fixed: interpolates between z-score map entries, supports any value in (0,1)
#   - File header and path corrected (was streamlit_sandbox/models/naive.py)
#   - CRLF line endings replaced with LF
#   - Dead _compute_metrics() removed — metrics computed by backtest.py, not here
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ForecastResult

_SEASONAL_PERIODS = {
    "MS": 12, "M": 12, "ME": 12,
    "QS": 4,  "Q": 4,  "QE": 4,
    "AS": 1,  "A": 1,  "YS": 1, "YE": 1,
    "W": 52,  "W-SUN": 52, "W-MON": 52,
    "D": 7,   "B": 5,
    "H": 24,
}

def _get_seasonal_period(freq: str) -> int:
    """Return seasonal period for a given pandas frequency string."""
    if freq in _SEASONAL_PERIODS:
        return _SEASONAL_PERIODS[freq]
    # Handle suffixed variants like 'QS-OCT', 'A-DEC'
    base = freq.split("-")[0].split("_")[0].upper()
    for key in _SEASONAL_PERIODS:
        if key.upper() == base:
            return _SEASONAL_PERIODS[key]
    return 1  # Unknown frequency — no seasonality assumed



_Z = {
    0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.99: 2.576,
}


def _get_z(confidence_level: float) -> float:
    if confidence_level in _Z:
        return _Z[confidence_level]
    levels = sorted(_Z.keys())
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        if lo <= confidence_level <= hi:
            t = (confidence_level - lo) / (hi - lo)
            return _Z[lo] + t * (_Z[hi] - _Z[lo])
    return 1.960


def run_naive(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Naive model requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected.")

    y = df["value"].astype(float)

    if len(y) < 3:
        raise ValueError("Minimum 3 observations required.")

    # Historical fitted (lag-1)
    hist_fitted = y.shift(1)
    hist_fitted.iloc[0] = y.iloc[0]

    # Future forecast
    last_value   = float(y.iloc[-1])
    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    future_forecast = np.full(horizon, last_value)

    # Random walk CI
    residuals = y - hist_fitted
    sigma     = float(np.std(residuals, ddof=1))
    z         = _get_z(confidence_level)

    steps          = np.arange(1, horizon + 1)
    interval_width = z * sigma * np.sqrt(steps)

    ci_low  = future_forecast - interval_width
    ci_high = future_forecast + interval_width

    hist_block = pd.DataFrame({
        "date":      y.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_forecast,
        "ci_low":    ci_low,
        "ci_mid":    future_forecast,
        "ci_high":   ci_high,
        "error_pct": np.nan,
    })

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "Naive",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "model_type":       "random_walk",
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "random_walk_variance",
            "output_contract":  "ForecastResult",
        },
    )
