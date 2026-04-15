# ==================================================
# FILE: foresight_engine/models/arima.py
# VERSION: 3.0.0
# MODEL: ARIMA -- AUTO ORDER SELECTION (pmdarima)
# ROLE: PRODUCTION FORECAST MODEL
# ENGINE: Foresight Engine v3.0.0
# UPDATED: Phase 4 -- Replace fixed variant with auto_arima
# ==================================================
#
# PHASE 4 UPGRADE -- AUTO ORDER SELECTION:
#
#   Previous (v2.0.0 / streamlit_sandbox):
#     Fixed ORDER=(0,1,1) selected by ARIMA_VARIANT constant.
#     No AIC/BIC optimisation. Header still referenced
#     streamlit_sandbox -- legacy artifact, now corrected.
#
#   Fixed (v3.0.0):
#     pmdarima.auto_arima -- stepwise AIC grid search over:
#       p in [0, 3]   non-seasonal AR order
#       d in [0, 2]   differencing (KPSS test)
#       q in [0, 3]   non-seasonal MA order
#       seasonal=False -- pure ARIMA, no seasonal component
#     Seasonal modelling is handled by SARIMA (v3.0.0) which
#     already performs auto seasonal order selection. Keeping
#     ARIMA non-seasonal avoids model overlap and ensures each
#     model in the ensemble contributes distinct signal.
#
#   Fallback:
#     If auto_arima fails or pmdarima is unavailable, falls back
#     to fixed (0,1,1) -- the original v2.0.0 behaviour.
#     Engine never crashes.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult (unchanged)
#   - Selected orders logged in metadata for full auditability
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

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



try:
    import pmdarima as pm
    _PMDARIMA_AVAILABLE = True
except ImportError:
    _PMDARIMA_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA


# --------------------------------------------------
# SEARCH BOUNDS
# --------------------------------------------------

MAX_P, MAX_D, MAX_Q = 3, 2, 3
MAX_ITER            = 200
FALLBACK_ORDER      = (0, 1, 1)


# --------------------------------------------------
# AUTO ORDER SELECTION
# --------------------------------------------------

def _auto_select(y) -> tuple:
    """
    Run auto_arima (non-seasonal) to select optimal ARIMA orders.
    Returns (order, fitted_statsmodels_result).
    Raises RuntimeError if pmdarima unavailable or fails.
    """
    if not _PMDARIMA_AVAILABLE:
        raise RuntimeError("pmdarima not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _y = y.values if hasattr(y, 'values') else y
        model = pm.auto_arima(
            _y,
            start_p=1, max_p=MAX_P,
            d=None,     max_d=MAX_D,
            start_q=1, max_q=MAX_Q,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            maxiter=MAX_ITER,
        )

    order = model.order

    sm_model = ARIMA(
        y,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm_model.fit(method_kwargs={"maxiter": MAX_ITER, "disp": 0})

    return order, res


def _fixed_fallback(y) -> tuple:
    """Fixed-order ARIMA fallback -- original v2.0.0 behaviour."""
    model = ARIMA(
        y,
        order=FALLBACK_ORDER,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(method_kwargs={"maxiter": MAX_ITER, "disp": 0})
    return FALLBACK_ORDER, res


# ==================================================
# MODEL RUNNER
# ==================================================

def run_arima(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("ARIMA requires 'date' and 'value' columns.")

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

    # --------------------------------------------------
    # AUTO ORDER SELECTION WITH FALLBACK
    # --------------------------------------------------

    used_auto     = False
    used_fallback = False

    try:
        order, res = _auto_select(y)
        used_auto = True
    except Exception:
        try:
            order, res = _fixed_fallback(y)
            used_fallback = True
        except Exception as e:
            raise RuntimeError(f"ARIMA fit failed (auto and fallback): {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    _fv = res.fittedvalues
    hist_fitted = pd.Series(np.asarray(_fv).astype("float64"), index=y.index)

    if np.isnan(hist_fitted.values).any():
        raise RuntimeError("NaN in ARIMA fitted values.")

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
    # FUTURE FORECAST
    # --------------------------------------------------

    forecast_res = res.get_forecast(steps=horizon)
    future_mean  = forecast_res.predicted_mean.astype("float64")
    ci           = forecast_res.conf_int(alpha=1.0 - confidence_level).astype("float64")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite ARIMA forecast values detected.")
    if ci.isna().any().any():
        raise RuntimeError("NaN in ARIMA confidence intervals.")
    if (ci.iloc[:, 0] > ci.iloc[:, 1]).any():
        raise RuntimeError("ARIMA CI bounds inverted.")
    if future_mean.index.min() <= hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_block = pd.DataFrame({
        "date":      future_mean.index,
        "actual":    np.nan,
        "forecast":  future_mean.values,
        "ci_low":    ci.iloc[:, 0].values,
        "ci_mid":    future_mean.values,
        "ci_high":   ci.iloc[:, 1].values,
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
        raise RuntimeError("Duplicate dates in ARIMA final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "ARIMA",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "order":            order,
            "seasonal":         False,
            "selection_method": "auto_arima_aic" if used_auto else "fixed_fallback",
            "used_auto":        used_auto,
            "used_fallback":    used_fallback,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "output_contract":  "ForecastResult",
        },
    )
