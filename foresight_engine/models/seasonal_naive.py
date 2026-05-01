# ==================================================
# FILE: foresight_engine/models/seasonal_naive.py
# VERSION: 1.0.0
# MODEL: SEASONAL NAIVE (Last Same-Season Value)
# ENGINE: Foresight Engine v3.0.0
# TIER: essentials
# ==================================================
#
# PURPOSE:
#   Seasonal Naïve forecasts each future period with the value
#   from the same period in the prior season. For monthly data
#   with period=12, January is forecast with last January, etc.
#
#   This is the standard MASE denominator model — the benchmark
#   every other model is measured against. If any model scores
#   MASE > 1.0 it is literally worse than this baseline.
#
#   Required for M3 certification methodology.
#
# CI METHOD:
#   Residuals from in-sample seasonal-naïve fitted values.
#   CI width grows with sqrt(h) — correct for independent errors.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Accepts any frequency with a definable seasonal period
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ForecastResult

_Z = {
    0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.99: 2.576,
}

_SEASONAL_PERIODS = {
    "MS": 12, "M": 12, "QS": 4, "Q": 4,
    "AS": 1,  "A": 1,  "W": 52, "D": 7,
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


def run_seasonal_naive(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("SeasonalNaive requires 'date' and 'value' columns.")

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
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")
    n = len(y)

    if not np.isfinite(y.values).all():
        raise ValueError("Non-finite values detected in series.")

    # Determine seasonal period
    period = _SEASONAL_PERIODS.get(inferred, 12)

    if n < period + 1:
        raise ValueError(
            f"SeasonalNaive requires >= {period + 1} observations "
            f"(one full season + 1). Got {n}."
        )

    # ── Historical fitted values ──────────────────────────────────────────────
    # Fitted value at time t = value at time t-period
    hist_fitted = y.shift(period)
    # First `period` positions have no fitted value — use actual (conservative)
    hist_fitted.iloc[:period] = y.iloc[:period].values

    # ── Future forecast ───────────────────────────────────────────────────────
    # For each step h=1..horizon, forecast = y[n - period + ((h-1) % period)]
    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    future_forecast = np.array([
        float(y.iloc[n - period + (h % period)])
        for h in range(horizon)
    ], dtype="float64")

    # ── CI from residuals ─────────────────────────────────────────────────────
    residuals    = (y.values[period:] - hist_fitted.values[period:]).astype("float64")
    finite_resid = residuals[np.isfinite(residuals)]
    sigma        = float(np.std(finite_resid, ddof=1)) if len(finite_resid) > 1 else float(np.abs(finite_resid).mean())
    if sigma < 1e-10:
        sigma = float(np.std(y.values)) * 0.1 + 1e-8

    z       = _get_z(confidence_level)
    steps   = np.arange(1, horizon + 1, dtype="float64")
    ci_low  = (future_forecast - z * sigma * np.sqrt(steps)).astype("float64")
    ci_high = (future_forecast + z * sigma * np.sqrt(steps)).astype("float64")
    ci_low  = np.minimum(ci_low,  future_forecast)
    ci_high = np.maximum(ci_high, future_forecast)

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in SeasonalNaive forecast output.")

    # ── Output assembly ───────────────────────────────────────────────────────
    hist_block = pd.DataFrame({
        "date":      y.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values.astype("float64"),
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values.astype("float64"),
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

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in SeasonalNaive output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in SeasonalNaive future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in SeasonalNaive output.")

    return ForecastResult(
        model_name  = "SeasonalNaive",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "model_type":       "seasonal_naive",
            "seasonal_period":  period,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "residual_based_sqrt_h",
            "min_tier":         "essentials",
            "output_contract":  "ForecastResult",
        },
    )
