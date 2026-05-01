# ==================================================
# FILE: foresight_engine/models/quantile_baseline.py
# VERSION: 1.0.0
# MODEL: QUANTILE BASELINE
# ENGINE: Foresight Engine v3.0.0
# TIER: essentials
# ==================================================
#
# PURPOSE:
#   Quantile Baseline forecasts using the empirical distribution
#   of the training series. Point forecast = median (p50) of the
#   most recent `window` observations. CIs use empirical quantiles.
#
#   Captures the central tendency without assuming any trend or
#   seasonality. Provides a robust non-parametric baseline.
#
#   Most useful on stationary series, volatile series where trend
#   extrapolation is dangerous, and as an ensemble diversity member.
#
# CI METHOD:
#   Empirical quantile intervals from the rolling window.
#   For the M3, this is a diagnostic check — if this model ranks
#   high, the series is fundamentally unpredictable beyond its
#   historical distribution.
#
# WINDOW SELECTION:
#   Default: min(24, n // 2) — at least 12 if available, never more
#   than half the series to capture recent behavior.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Minimum 4 observations
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ForecastResult

_Z = {
    0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.99: 2.576,
}

DEFAULT_WINDOW = 24


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


def run_quantile_baseline(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("QuantileBaseline requires 'date' and 'value' columns.")

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

    if n < 4:
        raise ValueError("QuantileBaseline requires at least 4 observations.")

    # ── Window selection ──────────────────────────────────────────────────────
    window = max(4, min(DEFAULT_WINDOW, n // 2))
    recent = y.values[-window:].astype("float64")

    # ── Point forecast ─────────────────────────────────────────────────────────
    # Median of the most recent window — robust to outliers
    point_value = float(np.median(recent))

    # ── Empirical quantile CI ─────────────────────────────────────────────────
    alpha  = 1.0 - confidence_level
    q_lo   = float(np.percentile(recent, 100 * alpha / 2))
    q_hi   = float(np.percentile(recent, 100 * (1 - alpha / 2)))

    # Ensure CI brackets the point forecast
    q_lo = min(q_lo, point_value)
    q_hi = max(q_hi, point_value)

    # CI width is constant (distributional forecast — no time dependence)
    future_forecast = np.full(horizon, point_value, dtype="float64")
    ci_low          = np.full(horizon, q_lo,         dtype="float64")
    ci_high         = np.full(horizon, q_hi,         dtype="float64")

    # ── Historical fitted values ──────────────────────────────────────────────
    # Rolling median of preceding `window` values
    hist_fitted = y.rolling(window=window, min_periods=1).median()

    # ── Output assembly ───────────────────────────────────────────────────────
    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

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
        raise RuntimeError("Duplicate dates in QuantileBaseline output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in QuantileBaseline future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in QuantileBaseline output.")

    return ForecastResult(
        model_name  = "QuantileBaseline",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "model_type":       "empirical_quantile_baseline",
            "window":           window,
            "point_value":      round(point_value, 4),
            "q_lo":             round(q_lo, 4),
            "q_hi":             round(q_hi, 4),
            "confidence_level": confidence_level,
            "ci_method":        "empirical_quantile",
            "frequency":        inferred,
            "min_tier":         "essentials",
            "output_contract":  "ForecastResult",
        },
    )
