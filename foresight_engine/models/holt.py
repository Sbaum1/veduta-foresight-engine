# ==================================================
# FILE: foresight_engine/models/holt.py
# VERSION: 1.0.0
# MODEL: HOLT — LINEAR TREND EXPONENTIAL SMOOTHING
# ENGINE: Foresight Engine v3.0.0
# TIER: essentials
# ==================================================
#
# PURPOSE:
#   Holt's method (double exponential smoothing) extends SES with
#   a linear trend component. Two smoothing parameters:
#     alpha — level smoothing (0 < alpha < 1)
#     beta  — trend smoothing (0 < beta < 1)
#
#   Unlike HW_Damped, the trend is not damped — it extrapolates
#   linearly into the future. Best on series with a clear,
#   sustained linear trend and no seasonality.
#
#   Distinct from HW_Damped in the ensemble: Holt adds full-trend
#   extrapolation which HW_Damped suppresses. On series with strong
#   sustained trends, Holt outperforms HW_Damped significantly.
#
# PARAMETER SELECTION:
#   AIC-minimising optimisation via statsmodels ExponentialSmoothing
#   with trend='add', damped_trend=False, seasonal=None.
#   Consistent with the Hyndman-Athanasopoulos (2021) framework.
#
# CI METHOD:
#   Residual-based: sigma * sqrt(h) scaled by Normal quantile.
#   Conservative — no covariance between level and trend uncertainty.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .contracts import ForecastResult

_Z = {
    0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.99: 2.576,
}
_MIN_OBS = 6


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


def run_holt(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Holt requires 'date' and 'value' columns.")

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

    if n < _MIN_OBS:
        raise ValueError(f"Holt requires >= {_MIN_OBS} observations. Got {n}.")

    # ── Fit Holt model ────────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = ExponentialSmoothing(
                y,
                trend           = "add",
                damped_trend    = False,
                seasonal        = None,
                initialization_method = "estimated",
            )
            fitted = model.fit(
                optimized     = True,
                use_brute     = True,
                remove_bias   = True,
            )
    except Exception as e:
        raise RuntimeError(f"Holt model fit failed: {e}") from e

    # ── Historical fitted values ───────────────────────────────────────────────
    fv = np.asarray(fitted.fittedvalues).astype("float64")

    # ── Future forecast ────────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fitted.forecast(horizon)
    except Exception as e:
        raise RuntimeError(f"Holt forecast failed: {e}") from e

    future_mean = np.asarray(pred).astype("float64")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values from Holt.")

    # ── CI from residuals ─────────────────────────────────────────────────────
    residuals    = (y.values - fv).astype("float64")
    finite_resid = residuals[np.isfinite(residuals)]
    sigma        = float(np.std(finite_resid, ddof=1)) if len(finite_resid) > 1 \
                   else float(np.abs(finite_resid).mean() + 1e-8)
    if sigma < 1e-10:
        sigma = float(np.std(y.values)) * 0.1 + 1e-8

    z       = _get_z(confidence_level)
    steps   = np.arange(1, horizon + 1, dtype="float64")
    ci_low  = (future_mean - z * sigma * np.sqrt(steps)).astype("float64")
    ci_high = (future_mean + z * sigma * np.sqrt(steps)).astype("float64")
    ci_low  = np.minimum(ci_low,  future_mean)
    ci_high = np.maximum(ci_high, future_mean)

    # ── Output assembly ───────────────────────────────────────────────────────
    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    hist_block = pd.DataFrame({
        "date":      y.index,
        "actual":    np.nan,
        "forecast":  fv,
        "ci_low":    np.nan,
        "ci_mid":    fv,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_mean,
        "ci_low":    ci_low,
        "ci_mid":    future_mean,
        "ci_high":   ci_high,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in Holt output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in Holt future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in Holt output.")

    return ForecastResult(
        model_name  = "Holt",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "model_type":       "holt_linear_trend",
            "alpha":            round(float(fitted.params.get("smoothing_level", 0)), 6),
            "beta":             round(float(fitted.params.get("smoothing_trend", 0)), 6),
            "damped":           False,
            "aic":              round(float(fitted.aic), 4),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "residual_based_sqrt_h",
            "min_tier":         "essentials",
            "output_contract":  "ForecastResult",
        },
    )
