# ==================================================
# FILE: foresight_engine/models/local_linear_trend.py
# VERSION: 3.0.0
# MODEL: LOCAL LINEAR TREND (State-Space, MLE)
# ENGINE: Foresight Engine v3.0.0
# TIER: essentials (minimum)
# ==================================================
#
# MODEL OVERVIEW:
#   Local Linear Trend State-Space model fitted by Maximum Likelihood
#   Estimation (MLE) via statsmodels UnobservedComponents.
#
#   Structure: level + slope (local linear trend) + seasonal(12)
#   Estimation: frequentist MLE — NOT Bayesian
#
#   This model was previously misidentified as "BSTS" (Bayesian
#   Structural Time Series). True BSTS uses MCMC posterior sampling
#   and is only available in R. This implementation uses the
#   equivalent state-space structure with MLE optimisation —
#   a well-established and production-grade approach.
#
# MULTI-START VARIANCE SEARCH:
#   Grid search over 9 (level_var_scale, trend_var_scale) combinations.
#   Best model selected by log-likelihood. Addresses sensitivity of
#   state-space models to variance initialisation.
#   Total fits: 9 (3 level × 3 trend scales).
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
from statsmodels.tsa.statespace.structural import UnobservedComponents

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



# --------------------------------------------------
# VARIANCE SEARCH GRID
# --------------------------------------------------

LEVEL_VAR_SCALES = [0.01, 0.1, 1.0]
TREND_VAR_SCALES = [0.001, 0.01, 0.1]


def _fit_with_scales(
    y,
    level_scale: float,
    trend_scale: float,
    inferred: str,
) -> tuple:
    """
    Fit Local Linear Trend with given variance initialisations.
    Returns (result, log_likelihood). Returns (None, -inf) on failure.
    """
    try:
        y_mean = float(np.mean(np.abs(y))) + 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                y,
                level="local linear trend",
                seasonal=12,
            )
            start = model.start_params.copy()
            param_names = model.param_names
            for i, name in enumerate(param_names):
                if "sigma2_level" in name or name == "sigma2.level":
                    start[i] = y_mean * level_scale
                elif "sigma2_trend" in name or name == "sigma2.trend":
                    start[i] = y_mean * trend_scale
                elif "sigma2_season" in name or "seasonal" in name:
                    start[i] = y_mean * 0.01
                elif "sigma2_irregular" in name or "irregular" in name:
                    start[i] = y_mean * 0.1

            start = np.clip(start, 1e-6, None)
            res   = model.fit(start_params=start, disp=False, maxiter=200)

        if not np.isfinite(res.llf):
            return None, float("-inf")
        if not np.isfinite(np.asarray(res.fittedvalues)).all():
            return None, float("-inf")

        return res, float(res.llf)

    except Exception:
        return None, float("-inf")


def _select_best_fit(
    y,
    inferred: str,
) -> tuple:
    """
    Grid search over variance scales, return best by log-likelihood.
    """
    best_res, best_llf = None, float("-inf")
    best_ls, best_ts   = LEVEL_VAR_SCALES[1], TREND_VAR_SCALES[1]

    for ls in LEVEL_VAR_SCALES:
        for ts in TREND_VAR_SCALES:
            res, llf = _fit_with_scales(y, ls, ts, inferred)
            if res is not None and llf > best_llf:
                best_res, best_llf = res, llf
                best_ls,  best_ts  = ls, ts

    if best_res is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model    = UnobservedComponents(y, level="local linear trend", seasonal=12)
                best_res = model.fit(disp=False)
                best_llf = float(best_res.llf)
        except Exception as e:
            raise RuntimeError(f"LocalLinearTrend fit failed on all initialisations: {e}") from e


    return best_res, best_llf, best_ls, best_ts


# ==================================================
# MODEL RUNNER
# ==================================================

def run_local_linear_trend(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("LocalLinearTrend requires 'date' and 'value' columns.")

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
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")
    if len(y) < 24:
        raise ValueError("Minimum 24 observations required (2 seasonal cycles).")

    res, best_llf, best_ls, best_ts = _select_best_fit(y, inferred)

    _fv = res.fittedvalues
    hist_fitted = pd.Series(np.asarray(_fv).astype("float64"), index=y.index)
    if np.isnan(hist_fitted.values).any():
        raise RuntimeError("NaN in fitted values.")

    hist_block = pd.DataFrame({
        "date":      hist_fitted.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    forecast_res = res.get_forecast(steps=horizon)
    future_mean  = forecast_res.predicted_mean.astype("float64")
    ci           = forecast_res.conf_int(alpha=1.0 - confidence_level).astype("float64")

    if future_mean.index.min() <= hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")
    if ci.isna().any().any():
        raise RuntimeError("Invalid confidence intervals detected.")
    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values detected.")
    if (ci.iloc[:, 0] > ci.iloc[:, 1]).any():
        raise RuntimeError("CI bounds inverted.")

    future_block = pd.DataFrame({
        "date":      future_mean.index,
        "actual":    np.nan,
        "forecast":  future_mean.values,
        "ci_low":    ci.iloc[:, 0].values,
        "ci_mid":    future_mean.values,
        "ci_high":   ci.iloc[:, 1].values,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "LocalLinearTrend",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "structure":          "local_linear_trend + seasonal(12)",
            "estimation_method":  "mle_frequentist",
            "selection_method":   "multi_start_llf_grid",
            "best_llf":           round(best_llf, 4),
            "best_level_scale":   best_ls,
            "best_trend_scale":   best_ts,
            "n_starts":           len(LEVEL_VAR_SCALES) * len(TREND_VAR_SCALES),
            "bayesian":           False,
            "note":               "Frequentist MLE state-space — not Bayesian",
            "frequency":          inferred,
            "confidence_level":   confidence_level,
            "ci_method":          "state_space_kalman",
            "output_contract":    "ForecastResult",
        },
    )
