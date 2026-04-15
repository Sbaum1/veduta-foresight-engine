# ==================================================
# FILE: foresight_engine/models/drift.py
# VERSION: 1.0.0
# MODEL: DRIFT (Random Walk with Drift)
# ENGINE: Foresight Engine v3.0.0
# TIER: essentials
# ==================================================
#
# PURPOSE:
#   Drift model projects the average rate of change observed
#   in the training series linearly into the future.
#
#   forecast(h) = y_n + h * ( (y_n - y_1) / (n - 1) )
#
#   This is equivalent to drawing a straight line between the
#   first and last observed values and extending it forward.
#   The drift model outperforms Naïve on clearly trended series.
#
# CI METHOD:
#   Random walk with drift variance — uncertainty grows as
#   sigma * sqrt(h * (1 + h/n)) per Hyndman & Athanasopoulos (2021).
#   The 1 + h/n correction accounts for uncertainty in estimating
#   the drift parameter from finite data.
#
# GOVERNANCE:
#   - Minimum 2 observations (to define a drift)
#   - No Streamlit dependencies
#   - No session state dependencies
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ForecastResult

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


def run_drift(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Drift model requires 'date' and 'value' columns.")

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

    if n < 2:
        raise ValueError("Drift model requires at least 2 observations.")

    # ── Drift parameter ───────────────────────────────────────────────────────
    # Average change per period from first to last observation
    y1    = float(y.iloc[0])
    yn    = float(y.iloc[-1])
    drift = (yn - y1) / (n - 1)

    # ── Historical fitted values ──────────────────────────────────────────────
    # Fitted value at time t: y_1 + t * drift (t = 0, 1, ..., n-1)
    t_hist       = np.arange(n, dtype="float64")
    hist_fitted  = y1 + t_hist * drift

    # ── Residual sigma ────────────────────────────────────────────────────────
    residuals    = y.values - hist_fitted
    finite_resid = residuals[np.isfinite(residuals)]
    sigma        = float(np.std(finite_resid, ddof=2)) if len(finite_resid) > 2 else \
                   float(np.abs(finite_resid).mean() + 1e-8)
    if sigma < 1e-10:
        sigma = float(np.std(y.values)) * 0.1 + 1e-8

    # ── Future forecast ───────────────────────────────────────────────────────
    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    steps           = np.arange(1, horizon + 1, dtype="float64")
    future_forecast = (yn + steps * drift).astype("float64")

    # CI: sigma * sqrt(h * (1 + h/n)) — Hyndman & Athanasopoulos (2021) §5.3
    z       = _get_z(confidence_level)
    ci_half = z * sigma * np.sqrt(steps * (1.0 + steps / n))
    ci_low  = (future_forecast - ci_half).astype("float64")
    ci_high = (future_forecast + ci_half).astype("float64")
    ci_low  = np.minimum(ci_low,  future_forecast)
    ci_high = np.maximum(ci_high, future_forecast)

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in Drift forecast output.")

    # ── Output assembly ───────────────────────────────────────────────────────
    hist_block = pd.DataFrame({
        "date":      y.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.astype("float64"),
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.astype("float64"),
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
        raise RuntimeError("Duplicate dates in Drift output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in Drift future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in Drift output.")

    return ForecastResult(
        model_name  = "Drift",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "model_type":       "random_walk_with_drift",
            "drift":            round(float(drift), 6),
            "sigma":            round(float(sigma), 6),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "hyndman_athanasopoulos_drift_variance",
            "min_tier":         "essentials",
            "output_contract":  "ForecastResult",
        },
    )
