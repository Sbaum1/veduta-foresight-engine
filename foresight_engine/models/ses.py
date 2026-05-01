# ==================================================
# FILE: foresight_engine/models/ses.py
# VERSION: 1.0.0
# MODEL: SES — Simple Exponential Smoothing
# ROLE: PRODUCTION FORECAST MODEL
# ENGINE: Foresight Engine v3.0.0
# ADDED: Phase 4 — 20th model (Essentials tier)
# ==================================================
#
# MODEL OVERVIEW:
#   Simple Exponential Smoothing (SES) is the foundational
#   exponential smoothing method. It applies a weighted average
#   of all past observations, with weights decaying geometrically.
#   Optimal for short, level series with no systematic trend or
#   seasonality — a complement to the more complex ETS/TBATS/SARIMA
#   models already in the ensemble.
#
# ALPHA SELECTION:
#   v1.0.0 uses AIC-minimising grid search over alpha ∈ [0.01, 0.99]
#   (step 0.01) via statsmodels SimpleExpSmoothing. This mirrors the
#   AIC-based model selection philosophy used in ETS (v3.0.0) and is
#   consistent with the Hyndman-Athanasopoulos forecasting framework.
#
#   Fallback: if grid search fails for any reason, alpha = 0.2
#   (a conservative, commonly cited default). Engine never crashes.
#
# CONFIDENCE INTERVALS:
#   SES has no closed-form CI from statsmodels by default.
#   CIs are constructed via in-sample residual standard deviation,
#   scaled by the Normal quantile for the requested confidence level.
#   Width grows with sqrt(h) where h = forecast horizon step.
#   This matches the asymptotic variance formula for SES.
#
# TIER: essentials — available in all three platform tiers.
# ENSEMBLE: True — eligible for Primary Ensemble weighting.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult (unchanged)
#   - Selected alpha logged in metadata for auditability
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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
# ALPHA SEARCH GRID
# --------------------------------------------------

_ALPHA_GRID   = np.round(np.arange(0.01, 1.00, 0.01), 2)
_FALLBACK_ALPHA = 0.2
_MIN_OBS        = 8


# --------------------------------------------------
# AIC-OPTIMAL ALPHA SELECTION
# --------------------------------------------------

def _select_alpha(y: np.ndarray) -> tuple[float, float, object]:
    """
    Grid-search over alpha to minimise AIC.
    Returns (best_alpha, best_aic, fitted_result).
    Raises RuntimeError if all fits fail.
    """
    best_alpha  = None
    best_aic    = np.inf
    best_result = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alpha in _ALPHA_GRID:
            try:
                model = SimpleExpSmoothing(y, initialization_method="estimated")
                res   = model.fit(smoothing_level=alpha, optimized=False)
                aic   = res.aic
                if np.isfinite(aic) and aic < best_aic:
                    best_aic    = aic
                    best_alpha  = alpha
                    best_result = res
            except Exception:
                continue

    if best_result is None:
        raise RuntimeError("SES: all alpha grid fits failed.")

    return float(best_alpha), float(best_aic), best_result


def _fallback_fit(y: np.ndarray) -> tuple[float, object]:
    """
    Fixed alpha=0.2 fallback — conservative default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SimpleExpSmoothing(y, initialization_method="estimated")
        res   = model.fit(smoothing_level=_FALLBACK_ALPHA, optimized=False)
    return _FALLBACK_ALPHA, res


# ==================================================
# MODEL RUNNER
# ==================================================

def run_ses(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("SES requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

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

    if len(y) < _MIN_OBS:
        raise ValueError(f"Minimum {_MIN_OBS} observations required for SES.")

    # --------------------------------------------------
    # ALPHA SELECTION WITH FALLBACK
    # --------------------------------------------------

    used_auto     = False
    used_fallback = False
    best_aic      = None

    try:
        best_alpha, best_aic, res = _select_alpha(y.values)
        used_auto = True
    except Exception:
        try:
            best_alpha, res = _fallback_fit(y.values)
            used_fallback   = True
        except Exception as e:
            raise RuntimeError(f"SES fit failed (auto and fallback): {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    hist_fitted = pd.Series(
        res.fittedvalues.astype("float64"),
        index=df.index,
    )

    if np.isnan(np.asarray(hist_fitted)).any():
        raise RuntimeError("NaN in SES fitted values.")

    hist_block = pd.DataFrame({
        "date":      hist_fitted.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST + CONFIDENCE INTERVALS
    # --------------------------------------------------

    # Point forecast — SES repeats the final smoothed level
    forecast_obj = res.forecast(horizon)
    future_mean  = forecast_obj.astype("float64")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite SES forecast values detected.")

    # CI construction via residual std dev scaled by sqrt(h)
    in_sample_errors = y.values - hist_fitted.values
    sigma            = float(np.std(in_sample_errors, ddof=1))

    # Normal quantile for the requested confidence level
    z = float(norm.ppf((1.0 + confidence_level) / 2.0))

    # Build forecast date index from the data's last date
    last_date    = df.index[-1]
    freq_offset  = pd.tseries.frequencies.to_offset(inferred)
    future_dates = pd.date_range(
        start=last_date + freq_offset,
        periods=horizon,
        freq=inferred,
    )

    if len(future_dates) != horizon:
        raise RuntimeError("Forecast date index length mismatch.")

    step_widths = np.array([sigma * z * np.sqrt(h) for h in range(1, horizon + 1)])

    lower = future_mean - step_widths
    upper = future_mean + step_widths

    future_block = pd.DataFrame({
        "date":      future_dates,
        "actual":    np.nan,
        "forecast":  future_mean,
        "ci_low":    lower,
        "ci_mid":    future_mean,
        "ci_high":   upper,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # DTYPE GOVERNANCE
    # --------------------------------------------------

    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols]   = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in SES final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "SES",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "alpha":            best_alpha,
            "best_aic":         best_aic,
            "selection_method": "aic_grid_search" if used_auto else "fixed_fallback",
            "used_auto":        used_auto,
            "used_fallback":    used_fallback,
            "ci_method":        "residual_sigma_sqrt_h",
            "ci_z":             z,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "output_contract":  "ForecastResult",
        },
    )
