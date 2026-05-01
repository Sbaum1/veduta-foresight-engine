# ==================================================
# FILE: foresight_engine/models/prophet.py
# VERSION: 4.1.0
# MODEL: PROPHET — SUPPRESSOR REMOVED, TUNED CHANGEPOINTS
# ENGINE: Foresight Engine v3.0.0
# UPDATED: M1 — Remove G1 suppressor, optimise hyperparameters
# UPDATED: v4.1.0 — uncertainty_samples=0 for deterministic CI
# ==================================================
#
# M1 UPGRADE — REMOVE STRUCTURAL BREAK SUPPRESSOR:
#
#   Previous (v3.0.0):
#     G1 suppressor: when a structural break >= 2.5σ was detected,
#     Prophet was suppressed entirely and the forecast was flat-lined
#     at the last observed value. This was intended to prevent
#     Prophet extrapolating regime shocks, but had the opposite
#     effect — a flat forecast on a series that has already broken
#     to a new level is worse than letting Prophet adapt.
#     Evidence: Series 02 MASE remained 4.4x after hardening.
#
#   Fixed (v4.0.0):
#     Suppressor removed entirely. Prophet runs on every series.
#     Changepoint parameters tuned to be more aggressive so Prophet
#     adapts to level shifts rather than ignoring them:
#
#     changepoint_prior_scale:  0.15 → 0.30
#       Higher = more responsive to trend changes. Prophet can now
#       detect and follow a sharp level shift within the training
#       window rather than averaging across pre/post break levels.
#
#     changepoint_range:        0.90 → 0.95
#       Allows changepoint detection in the last 5% of training data
#       (previously 10% was excluded). Captures very recent breaks.
#
#     n_changepoints:           (default 25) → 30
#       More candidate changepoint locations = finer detection grid.
#
#     seasonality_mode:         multiplicative (unchanged)
#     seasonality_prior_scale:  5.0 → 3.0
#       Tighter seasonal prior — prevents seasonal component
#       absorbing trend energy on volatile series.
#
#   Why this is the right fix:
#     Prophet's core value is piecewise linear trend with automatic
#     changepoint detection. Suppressing it on exactly the series
#     where it is most needed (regime change series) is contradictory.
#     The correct approach is to tune it to detect and follow breaks.
#
# GOVERNANCE:
#   - tuning_version updated to "m1_changepoint_optimised"
#   - All metadata keys preserved and extended
#   - Output contract: ForecastResult unchanged
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

from foresight_engine.models.contracts import ForecastResult

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



# --------------------------------------------------
# TUNED HYPERPARAMETERS — M1
# --------------------------------------------------

CHANGEPOINT_PRIOR_SCALE  = 0.30   # was 0.15 — more aggressive break detection
CHANGEPOINT_RANGE        = 0.95   # was 0.90 — detect breaks nearer end of series
N_CHANGEPOINTS           = 30     # was default 25
SEASONALITY_MODE         = "multiplicative"
SEASONALITY_PRIOR_SCALE  = 3.0    # was 5.0 — tighter seasonal prior
UNCERTAINTY_SAMPLES      = 1000  # Monte Carlo CI — ensures yhat_lower/yhat_upper always present
                                  # v4.2.0: was 0 (closed-form) but Prophet 1.3.0 omits
                                  # yhat_lower/yhat_upper columns when uncertainty_samples=0,
                                  # causing KeyError in the CI validation check below.
                                  # 1000 samples adds ~1-2s but guarantees the full output schema.

TUNING_VERSION = "m1_changepoint_optimised"


# ==================================================
# MODEL RUNNER
# ==================================================

def run_prophet(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Prophet requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date")

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    if df["value"].isna().any():
        raise ValueError("Missing values detected in input series.")

    y = df["value"].astype("float64")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")
    if len(df) < 24:
        raise ValueError("Minimum 24 observations required.")

    # --------------------------------------------------
    # PROPHET MODEL — M1 TUNED
    # --------------------------------------------------

    prophet_df = df.rename(columns={"date": "ds", "value": "y"})[["ds", "y"]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            interval_width          = confidence_level,
            uncertainty_samples     = UNCERTAINTY_SAMPLES,
            changepoint_prior_scale = CHANGEPOINT_PRIOR_SCALE,
            changepoint_range       = CHANGEPOINT_RANGE,
            n_changepoints          = N_CHANGEPOINTS,
            seasonality_mode        = SEASONALITY_MODE,
            seasonality_prior_scale = SEASONALITY_PRIOR_SCALE,
            daily_seasonality       = False,
            weekly_seasonality      = False,
            yearly_seasonality      = True,
        )
        model.fit(prophet_df)

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_hist = model.predict(prophet_df)

    # Guard: uncertainty_samples=0 in some Prophet versions omits yhat_lower/yhat_upper
    if "yhat_lower" not in fitted_hist.columns:
        residuals    = (prophet_df["y"].values - fitted_hist["yhat"].values)
        sigma        = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.abs(residuals).mean())
        from scipy import stats as _st
        z            = float(_st.norm.ppf((1 + confidence_level) / 2))
        fitted_hist["yhat_lower"] = fitted_hist["yhat"] - z * sigma
        fitted_hist["yhat_upper"] = fitted_hist["yhat"] + z * sigma

    if fitted_hist[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid fitted output detected.")

    hist_block = pd.DataFrame({
        "date":      fitted_hist["ds"],
        "actual":    np.nan,
        "forecast":  fitted_hist["yhat"].astype("float64").values,
        "ci_low":    fitted_hist["yhat_lower"].astype("float64").values,
        "ci_mid":    fitted_hist["yhat"].astype("float64").values,
        "ci_high":   fitted_hist["yhat_upper"].astype("float64").values,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future = model.make_future_dataframe(
        periods=horizon, freq=inferred, include_history=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_future = model.predict(future)

    # Guard: ensure CI columns present (same as fitted_hist guard above)
    if "yhat_lower" not in forecast_future.columns:
        residuals    = (prophet_df["y"].values - fitted_hist["yhat"].values)
        sigma        = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.abs(residuals).mean())
        from scipy import stats as _st
        z            = float(_st.norm.ppf((1 + confidence_level) / 2))
        h_arr        = np.arange(1, horizon + 1, dtype="float64")
        forecast_future["yhat_lower"] = forecast_future["yhat"] - z * sigma * np.sqrt(h_arr)
        forecast_future["yhat_upper"] = forecast_future["yhat"] + z * sigma * np.sqrt(h_arr)

    if forecast_future[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid future forecast output detected.")
    if not np.isfinite(forecast_future["yhat"]).all():
        raise RuntimeError("Non-finite forecast values detected.")
    if forecast_future["ds"].min() <= df["date"].max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_block = pd.DataFrame({
        "date":      forecast_future["ds"],
        "actual":    np.nan,
        "forecast":  forecast_future["yhat"].astype("float64").values,
        "ci_low":    forecast_future["yhat_lower"].astype("float64").values,
        "ci_mid":    forecast_future["yhat"].astype("float64").values,
        "ci_high":   forecast_future["yhat_upper"].astype("float64").values,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast","ci_low","ci_mid","ci_high"]] = \
            b[["forecast","ci_low","ci_mid","ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "Prophet",
        forecast_df = forecast_df[
            ["date","actual","forecast","ci_low","ci_mid","ci_high","error_pct"]
        ],
        metrics  = None,
        metadata = {
            "trend":                    "piecewise_linear",
            "seasonality":              "yearly",
            "seasonality_mode":         SEASONALITY_MODE,
            "changepoint_prior_scale":  CHANGEPOINT_PRIOR_SCALE,
            "changepoint_range":        CHANGEPOINT_RANGE,
            "n_changepoints":           N_CHANGEPOINTS,
            "seasonality_prior_scale":  SEASONALITY_PRIOR_SCALE,
            "uncertainty_samples":      UNCERTAINTY_SAMPLES,
            "ci_method":                "closed_form_analytical",
            "regime_suppressor":        False,
            "tuning_version":           TUNING_VERSION,
            "frequency":                inferred,
            "confidence_level":         confidence_level,
            "output_contract":          "ForecastResult",
        },
    )
