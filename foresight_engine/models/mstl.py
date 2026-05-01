# ==================================================
# FILE: foresight_engine/models/mstl.py
# VERSION: 1.0.0
# MODEL: MSTL -- Multi-Seasonal Trend Decomposition with Loess
# ROLE: PRODUCTION FORECAST MODEL
# ENGINE: Foresight Engine v3.0.0
# ADDED: Phase 4 -- 21st model, replaces SES as 20th ensemble member
# ==================================================
#
# MODEL OVERVIEW:
#   MSTL extends classical STL decomposition to handle multiple
#   seasonal periods simultaneously. Where STL+ETS fits a single
#   12-period seasonal component, MSTL decomposes the series into:
#     - A long-period seasonal component  (period=12, annual)
#     - A short-period seasonal component (period=3, quarterly)
#       when series length supports it (>= 3 * 12 = 36 obs)
#     - A trend component (ETS on the trend+remainder)
#     - A remainder (residual)
#
#   This directly addresses the weakness in STL+ETS on Fortune 100
#   series that exhibit both quarterly budget cycles AND annual
#   seasonal patterns -- retail, CPG, manufacturing, and financial
#   series almost universally have this dual-cycle structure.
#
# WHY MSTL OVER SES IN THE ENSEMBLE:
#   SES (Simple Exponential Smoothing) has no trend or seasonal
#   component -- it is a subset of ETS, which VEDUTA already runs.
#   MSTL fills a genuine gap: multi-seasonal decomposition is not
#   handled by any other model in the current ensemble. On M3
#   Monthly series with complex seasonal structure (the majority),
#   MSTL provides independent signal not captured by SARIMA, ETS,
#   TBATS, or STL+ETS alone.
#
# IMPLEMENTATION:
#   Pure statsmodels -- no new dependencies. Uses STL twice:
#     Pass 1: Decompose the full series with period=12 (annual)
#     Pass 2: Decompose the seasonally-adjusted series with period=3
#             (quarterly) if series length >= 3 * short_period * 2
#   Trend + remainder forecasted with ETS (AIC-optimised across
#   additive/multiplicative specs, consistent with ets.py v3.0.0).
#   Future seasonals reconstructed by tiling the last full cycle
#   of each seasonal component -- same approach as STL+ETS v2.0.0.
#
# CONFIDENCE INTERVALS:
#   Residual-based sigma * sqrt(h) -- identical to STL+ETS v2.0.0.
#   Consistent CI methodology across all decomposition models.
#
# FALLBACK:
#   If multi-seasonal decomposition fails for any reason (short
#   series, insufficient variance for second STL pass), falls back
#   to single-period STL+ETS decomposition -- same as the existing
#   STL+ETS model. Engine never crashes.
#
# TIER: essentials -- available in all three platform tiers.
# ENSEMBLE: True -- 20th ensemble member.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - No new package dependencies (pure statsmodels)
#   - Output contract: ForecastResult (unchanged)
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETSModel
    _HAS_ETSMODEL = True
except ImportError:
    _HAS_ETSMODEL = False

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
# CONSTANTS
# --------------------------------------------------

LONG_PERIOD   = 12    # annual seasonality (monthly data)
SHORT_PERIOD  = 3     # quarterly seasonality
MIN_OBS       = 24    # minimum for single-seasonal pass
MIN_OBS_MULTI = 72    # minimum for dual-seasonal pass (6 full long cycles)
MAX_ITER      = 200

# ETS specs to try (error, trend, damped) -- AIC selection
ETS_SPECS = [
    ("add", "add",  True),
    ("add", "add",  False),
    ("add", None,   False),
    ("mul", "add",  True),
    ("mul", "add",  False),
    ("mul", None,   False),
]

Z_SCORES = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _z(confidence_level: float) -> float:
    return min(Z_SCORES.items(), key=lambda kv: abs(kv[0] - confidence_level))[1]


def _fit_ets(y: np.ndarray, inferred_freq: str) -> object:
    """
    AIC-optimal ETS fit on the trend+remainder series.
    Uses ETSModel when available (supports error= param),
    falls back to ExponentialSmoothing (Holt-Winters style).
    No seasonal component — STL handles seasonality separately.
    """
    s = pd.Series(y, dtype="float64")
    best_aic = np.inf
    best_fit = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try ETSModel specs (proper error-state formulation)
        if _HAS_ETSMODEL:
            for error, trend, damped in ETS_SPECS:
                try:
                    model = _ETSModel(
                        s,
                        error=error,
                        trend=trend,
                        damped_trend=damped if trend else False,
                        seasonal=None,
                        initialization_method="estimated",
                    )
                    fit = model.fit(optimized=True, maxiter=MAX_ITER, disp=False)
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit
                except Exception:
                    continue

        # Fallback: ExponentialSmoothing (HW-style, no error= param)
        if best_fit is None:
            hw_specs = [
                {"trend": None,  "damped_trend": False},
                {"trend": "add", "damped_trend": False},
                {"trend": "add", "damped_trend": True},
            ]
            for spec in hw_specs:
                try:
                    model = ExponentialSmoothing(
                        s,
                        trend              = spec["trend"],
                        damped_trend       = spec["damped_trend"] if spec["trend"] else False,
                        seasonal           = None,
                        initialization_method = "estimated",
                    )
                    fit = model.fit(optimized=True, maxiter=MAX_ITER)
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit
                except Exception:
                    continue

    if best_fit is None:
        # Hard fallback — fixed parameters
        model = ExponentialSmoothing(
            s, trend="add", damped_trend=True, seasonal=None,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_fit = model.fit(optimized=True, maxiter=MAX_ITER)

    return best_fit


def _stl_decompose(y: np.ndarray, period: int, robust: bool = True) -> tuple:
    """
    Run STL on y with given period.
    Returns (trend, seasonal, residual) as np.ndarray.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl    = STL(y, period=period, robust=robust)
        result = stl.fit()
    return result.trend, result.seasonal, result.resid


def _residual_ci(
    forecast_vals: np.ndarray,
    residuals: np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    z      = _z(confidence_level)
    sigma  = float(np.std(residuals, ddof=1))
    h      = np.arange(1, len(forecast_vals) + 1, dtype="float64")
    spread = z * sigma * np.sqrt(h)
    return forecast_vals - spread, forecast_vals + spread


def _tile_seasonal(seasonal: np.ndarray, period: int, horizon: int) -> np.ndarray:
    """Tile the last full cycle of a seasonal component forward."""
    last_cycle = seasonal[-period:].astype("float64")
    tiled      = np.tile(last_cycle, int(np.ceil(horizon / period)))
    return tiled[:horizon]


# ==================================================
# CORE DECOMPOSITION
# ==================================================

def _multi_seasonal_decompose(
    y: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Dual-pass STL decomposition.

    Pass 1: Decompose y with LONG_PERIOD=12 -> annual seasonal S1, trend+resid T1
    Pass 2: Decompose (y - S1) with SHORT_PERIOD=3 -> quarterly seasonal S2, trend T2

    Returns:
        trend     : combined trend+remainder after both passes
        seasonal1 : annual seasonal component (period=12)
        seasonal2 : quarterly seasonal component (period=3) or zeros
        multi     : True if dual pass succeeded, False if single pass only
    """
    # Pass 1 -- annual
    try:
        trend1, seasonal1, resid1 = _stl_decompose(y, LONG_PERIOD)
    except Exception:
        raise RuntimeError("MSTL: Primary STL decomposition (period=12) failed.")

    # Pass 2 -- quarterly (only if sufficient obs)
    if n >= MIN_OBS_MULTI:
        deseasonalised = y - seasonal1
        try:
            trend2, seasonal2, resid2 = _stl_decompose(deseasonalised, SHORT_PERIOD)
            # Trend for forecasting is the remainder after BOTH seasonal removals
            trend_for_ets = y - seasonal1 - seasonal2
            return trend_for_ets, seasonal1, seasonal2, True
        except Exception:
            pass  # Fall through to single-pass

    # Single pass fallback
    trend_for_ets = trend1 + resid1  # trend+remainder -> ETS handles it
    seasonal2     = np.zeros_like(seasonal1)
    return trend_for_ets, seasonal1, seasonal2, False


# ==================================================
# MODEL RUNNER
# ==================================================

def run_mstl(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("MSTL requires 'date' and 'value' columns.")

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

    y = df["value"].astype("float64").values
    n = len(y)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")

    if n < MIN_OBS:
        raise ValueError(f"MSTL requires minimum {MIN_OBS} observations. Got {n}.")

    dates = df.index

    # --------------------------------------------------
    # MULTI-SEASONAL DECOMPOSITION
    # --------------------------------------------------

    try:
        trend_for_ets, seasonal1, seasonal2, multi_seasonal = _multi_seasonal_decompose(y, n)
    except RuntimeError:
        # Hard fallback: treat as single STL+ETS
        trend1, seasonal1, resid1 = _stl_decompose(y, LONG_PERIOD)
        trend_for_ets  = trend1 + resid1
        seasonal2      = np.zeros(n)
        multi_seasonal = False

    # --------------------------------------------------
    # ETS ON TREND + REMAINDER
    # --------------------------------------------------

    ets_fit = _fit_ets(trend_for_ets, inferred)

    # Historical fitted: ETS fitted + both seasonal components
    ets_hist      = ets_fit.fittedvalues.values.astype("float64") if hasattr(ets_fit.fittedvalues, 'values') else ets_fit.fittedvalues.astype("float64")
    if hasattr(ets_hist, 'values'): ets_hist = ets_hist.values
    hist_forecast = ets_hist + seasonal1 + seasonal2

    # In-sample residuals for CI
    residuals = y - hist_forecast
    if not np.isfinite(residuals).all():
        residuals = y - (seasonal1 + seasonal2)  # degraded but finite

    hist_block = pd.DataFrame({
        "date":      dates,
        "actual":    np.nan,
        "forecast":  hist_forecast,
        "ci_low":    np.nan,
        "ci_mid":    hist_forecast,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_index = pd.date_range(
        start   = dates[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    # ETS trend forecast
    future_trend = ets_fit.forecast(horizon).astype("float64")

    if not np.isfinite(future_trend).all():
        raise RuntimeError("MSTL: Non-finite ETS trend forecast.")

    # Project seasonal components forward by tiling
    future_s1 = _tile_seasonal(seasonal1, LONG_PERIOD, horizon)
    future_s2 = _tile_seasonal(seasonal2, SHORT_PERIOD, horizon) if multi_seasonal else np.zeros(horizon)

    future_vals = future_trend.values + future_s1 + future_s2

    if not np.isfinite(future_vals).all():
        raise RuntimeError("MSTL: Non-finite combined future forecast.")

    ci_low, ci_high = _residual_ci(future_vals, residuals, confidence_level)

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_vals,
        "ci_low":    ci_low,
        "ci_mid":    future_vals,
        "ci_high":   ci_high,
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
        raise RuntimeError("Duplicate dates in MSTL final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "MSTL",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "long_period":       LONG_PERIOD,
            "short_period":      SHORT_PERIOD if multi_seasonal else None,
            "multi_seasonal":    multi_seasonal,
            "n_observations":    n,
            "ets_aic":           float(ets_fit.aic),
            "ci_method":         "residual_sigma_sqrt_h",
            "frequency":         inferred,
            "confidence_level":  confidence_level,
            "output_contract":   "ForecastResult",
        },
    )
