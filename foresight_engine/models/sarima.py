# ==================================================
# FILE: foresight_engine/models/sarima.py
# VERSION: 3.0.0
# MODEL: SARIMA — AUTO ORDER SELECTION (pmdarima)
# ENGINE: Foresight Engine v3.0.0
# UPDATED: M1 — Replace fixed (1,1,1)(1,0,1,12) with auto_arima
# ==================================================
#
# M1 UPGRADE — AUTO ORDER SELECTION:
#
#   Previous (v2.0.0):
#     Fixed ORDER=(1,1,1), SEASONAL_ORDER=(1,0,1,12).
#     One specification applied to every series regardless of
#     its actual autocorrelation structure. A series with no
#     AR component still gets an AR(1) term. A series that
#     needs d=2 differencing gets d=1. This is wrong.
#
#   Fixed (v3.0.0):
#     pmdarima.auto_arima — stepwise AIC/BIC grid search over:
#       p ∈ [0, 3]   non-seasonal AR order
#       d ∈ [0, 2]   differencing (KPSS test for d, ADF for D)
#       q ∈ [0, 3]   non-seasonal MA order
#       P ∈ [0, 2]   seasonal AR order
#       D ∈ [0, 1]   seasonal differencing
#       Q ∈ [0, 2]   seasonal MA order
#       m = 12       seasonal period (monthly)
#     Stepwise search: starts from simple model, expands only
#     when AIC improves. Fast — typically 5-20 model evaluations
#     rather than full grid (which would be 3*3*4*3*2*3=648).
#     Information criterion: AIC (consistent with ETS selection).
#
#   Fallback:
#     If auto_arima fails (e.g. non-stationary pathological series),
#     falls back to fixed (1,1,1)(1,0,1,12) — original behaviour.
#     Engine never crashes.
#
#   Why pmdarima:
#     Direct Python port of R's auto.arima (Hyndman-Khandakar).
#     The same algorithm used in academic M-competition benchmarks.
#     Selecting with this algorithm is the published standard.
#
# GOVERNANCE:
#   - Output contract: ForecastResult unchanged
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



# pmdarima for auto order selection; statsmodels SARIMAX as fallback
try:
    import pmdarima as pm
    _PMDARIMA_AVAILABLE = True
except ImportError:
    _PMDARIMA_AVAILABLE = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# SEARCH BOUNDS
# --------------------------------------------------

MAX_P, MAX_D, MAX_Q   = 3, 2, 3
MAX_SP, MAX_SD, MAX_SQ = 2, 1, 2
SEASONAL_PERIOD        = 12
MAX_ITER               = 200

# Fixed fallback (original v2.0.0 behaviour)
FALLBACK_ORDER          = (1, 1, 1)
FALLBACK_SEASONAL_ORDER = (1, 0, 1, 12)


# --------------------------------------------------
# AUTO ORDER SELECTION
# --------------------------------------------------

def _auto_select(
    y,
    seasonal: bool,
) -> tuple[tuple, tuple, object]:
    """
    Run auto_arima to select optimal SARIMA orders.
    Returns (order, seasonal_order, fitted_result).
    Raises RuntimeError if auto_arima unavailable or fails.
    """
    if not _PMDARIMA_AVAILABLE:
        raise RuntimeError("pmdarima not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _y = y.values if hasattr(y, 'values') else y
        model = pm.auto_arima(
            _y,
            start_p=1, max_p=MAX_P,
            d=None,     max_d=MAX_D,    # d selected by KPSS test
            start_q=1, max_q=MAX_Q,
            start_P=1, max_P=MAX_SP,
            D=None,     max_D=MAX_SD,   # D selected by CH test
            start_Q=1, max_Q=MAX_SQ,
            m=SEASONAL_PERIOD if seasonal else 1,
            seasonal=seasonal,
            information_criterion="aic",
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            maxiter=MAX_ITER,
        )

    order          = model.order
    seasonal_order = model.seasonal_order   # (P,D,Q,m)

    # Re-fit via statsmodels SARIMAX for full forecast API
    sm_model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm_model.fit(maxiter=MAX_ITER, disp=False)

    return order, seasonal_order, res


def _fixed_fallback(y: np.ndarray) -> tuple[tuple, tuple, object]:
    """
    Fixed-order SARIMAX fallback — original v2.0.0 behaviour.
    """
    model = SARIMAX(
        y,
        order=FALLBACK_ORDER,
        seasonal_order=FALLBACK_SEASONAL_ORDER,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(maxiter=MAX_ITER, disp=False)
    return FALLBACK_ORDER, FALLBACK_SEASONAL_ORDER, res


# ==================================================
# MODEL RUNNER
# ==================================================

def run_sarima(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("SARIMA requires 'date' and 'value' columns.")

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

    if len(y) < 24:
        raise ValueError("Minimum 24 observations required.")

    # ── Near-constant series guard ────────────────────────────────────────────
    # SARIMA differences a flat series to near-zero values, fits on noise,
    # and produces a forecast near zero — catastrophically wrong.
    # If coefficient of variation < 1%, return a level forecast (series mean).
    y_mean = float(y.mean())
    y_std  = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    cv     = y_std / abs(y_mean) if abs(y_mean) > 1e-8 else 0.0
    if cv < 0.01:
        # Near-constant: forecast the mean with narrow residual CI
        future_index = pd.date_range(
            start=y.index[-1], periods=horizon + 1, freq=inferred)[1:]
        z = 1.645  # 90% default
        ci_half = z * max(y_std, abs(y_mean) * 0.001)
        future_forecast = np.full(horizon, y_mean, dtype="float64")
        ci_low  = future_forecast - ci_half
        ci_high = future_forecast + ci_half
        model_tag = "SARIMA"
        hist_fitted_vals = np.full(len(y), y_mean, dtype="float64")
        hist_block = pd.DataFrame({
            "date": y.index, "actual": np.nan,
            "forecast": hist_fitted_vals, "ci_low": np.nan,
            "ci_mid": hist_fitted_vals, "ci_high": np.nan, "error_pct": np.nan,
        })
        future_block = pd.DataFrame({
            "date": future_index, "actual": np.nan,
            "forecast": future_forecast,
            "ci_low": ci_low, "ci_mid": future_forecast, "ci_high": ci_high,
            "error_pct": np.nan,
        })
        for b in (hist_block, future_block):
            b[["forecast","ci_low","ci_mid","ci_high"]] = \
                b[["forecast","ci_low","ci_mid","ci_high"]].astype("float64")
        fc_df = pd.concat([hist_block, future_block], ignore_index=True)
        fc_df = fc_df.sort_values("date").reset_index(drop=True)
        return ForecastResult(
            model_name  = model_tag,
            forecast_df = fc_df[["date","actual","forecast","ci_low","ci_mid","ci_high","error_pct"]],
            metrics     = None,
            metadata    = {
                "mode":             "near_constant_level_forecast",
                "cv":               round(cv, 6),
                "series_mean":      round(y_mean, 2),
                "frequency":        inferred,
                "confidence_level": confidence_level,
                "output_contract":  "ForecastResult",
            },
        )


    # --------------------------------------------------
    # M1: AUTO ORDER SELECTION WITH FALLBACK
    # --------------------------------------------------

    seasonal       = len(y) >= 2 * SEASONAL_PERIOD
    used_auto      = False
    used_fallback  = False

    try:
        order, seasonal_order, res = _auto_select(y, seasonal)
        used_auto = True
    except Exception:
        try:
            order, seasonal_order, res = _fixed_fallback(y)
            used_fallback = True
        except Exception as e:
            raise RuntimeError(f"SARIMA fit failed (auto and fallback): {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    forecast_res = res.get_forecast(steps=horizon)
    future_mean  = forecast_res.predicted_mean.astype("float64")
    ci           = forecast_res.conf_int(
                       alpha=1.0 - confidence_level
                   ).astype("float64")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values detected.")
    if ci.isna().any().any():
        raise RuntimeError("NaN in confidence intervals.")
    if (ci.iloc[:, 0] > ci.iloc[:, 1]).any():
        raise RuntimeError("CI bounds inverted.")
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
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "SARIMA",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "order":            order,
            "seasonal_order":   seasonal_order,
            "selection_method": "auto_arima_aic" if used_auto else "fixed_fallback",
            "used_auto":        used_auto,
            "used_fallback":    used_fallback,
            "seasonal":         seasonal,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "output_contract":  "ForecastResult",
        },
    )
