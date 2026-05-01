# ==================================================
# FILE: foresight_engine/models/dhr.py
# VERSION: 2.0.0
# MODEL: DYNAMIC HARMONIC REGRESSION (DHR)
# ENGINE: Foresight Engine v3.0.0
# TIER: pro (minimum)
# STATUS: VEDUTA ENGINE — PHASE 3C
# ==================================================
#
# PURPOSE:
#   Dynamic Harmonic Regression captures multiple seasonality
#   periods simultaneously using Fourier terms as external
#   regressors in a SARIMAX framework.
#
#   Advantages over standard SARIMA:
#   - Handles multiple seasonality periods (e.g. weekly + annual)
#   - More flexible seasonal shape via sine/cosine terms
#   - Works on long seasonal periods where SARIMA is slow
#   - K Fourier terms tuned automatically by AIC
#
# IMPLEMENTATION:
#   statsmodels SARIMAX with Fourier regressors.
#   ARIMA order: (1,1,1) with no seasonal ARIMA component.
#   Seasonal variation handled entirely by Fourier terms.
#   K terms (1 to K_MAX) selected via AIC minimization.
#
# CI METHOD:
#   Native SARIMAX prediction intervals via get_forecast().
#   Falls back to residual-based if native CI is invalid.
#
# FOURIER TERMS:
#   sin(2*pi*k*t/m) and cos(2*pi*k*t/m) for k=1..K
#   where m = seasonal period, t = time index.
#   Both history and future regressors constructed identically.
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from foresight_engine.models.contracts import ForecastResult

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

K_MAX     = 5      # Maximum Fourier terms to evaluate
ARIMA_P   = 1
ARIMA_D   = 1
ARIMA_Q   = 1

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


# --------------------------------------------------
# FOURIER TERM BUILDER
# --------------------------------------------------

def _fourier_terms(t: np.ndarray, m: int, K: int) -> np.ndarray:
    """
    Build Fourier regressor matrix.
    Returns array of shape (len(t), 2*K).
    Columns: sin_1, cos_1, sin_2, cos_2, ..., sin_K, cos_K
    """
    cols = []
    for k in range(1, K + 1):
        angle = 2.0 * np.pi * k * t / m
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def _select_k(
    y:        np.ndarray,
    m:        int,
    k_max:    int,
    arima_p:  int,
    arima_d:  int,
    arima_q:  int,
) -> int:
    """
    Select optimal K via AIC minimization.
    Tries K = 1 to k_max. Returns K with lowest AIC.
    """
    best_k   = 1
    best_aic = np.inf
    t        = np.arange(len(y), dtype="float64")

    for k in range(1, k_max + 1):
        X = _fourier_terms(t, m, k)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(
                    y,
                    exog=X,
                    order=(arima_p, arima_d, arima_q),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False, maxiter=200)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_k   = k
        except Exception:
            continue

    return best_k


# ==================================================
# MODEL RUNNER
# ==================================================

def run_dhr(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("DHR requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    if df["value"].isna().any():
        raise ValueError("Missing values detected in input series.")

    y = df["value"].astype("float64").values

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(df) < 24:
        raise ValueError("Minimum 24 observations required for DHR.")

    # --------------------------------------------------
    # SEASON LENGTH
    # --------------------------------------------------

    _season_map = {
        "MS": 12, "M": 12,
        "QS": 4,  "Q": 4,
        "W":  52, "W-SUN": 52, "W-MON": 52,
        "D":  365,
    }
    m = _season_map.get(inferred, 12)

    # Cap K_MAX to avoid over-parameterisation on short series
    k_max = min(K_MAX, max(1, len(y) // (2 * m)))
    k_max = max(k_max, 1)

    # --------------------------------------------------
    # SELECT OPTIMAL K
    # --------------------------------------------------

    K = _select_k(y, m, k_max, ARIMA_P, ARIMA_D, ARIMA_Q)

    # --------------------------------------------------
    # FIT FINAL MODEL WITH SELECTED K
    # --------------------------------------------------

    t_hist = np.arange(len(y), dtype="float64")
    X_hist = _fourier_terms(t_hist, m, K)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog=X_hist,
                order=(ARIMA_P, ARIMA_D, ARIMA_Q),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=500)
    except Exception as e:
        raise RuntimeError(f"DHR model fit failed: {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_values = fitted.fittedvalues.astype("float64")

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_values,
        "ci_low":    np.nan,
        "ci_mid":    fitted_values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FOURIER REGRESSORS
    # --------------------------------------------------

    t_future = np.arange(len(y), len(y) + horizon, dtype="float64")
    X_future = _fourier_terms(t_future, m, K)

    # --------------------------------------------------
    # FUTURE FORECAST WITH NATIVE CI
    # --------------------------------------------------

    alpha_ci = 1.0 - confidence_level
    ci_method = "native_sarimax"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fcast = fitted.get_forecast(steps=horizon, exog=X_future)
            pred_raw = fcast.predicted_mean
            pred  = np.asarray(pred_raw, dtype="float64")
            ci_df = fcast.conf_int(alpha=alpha_ci)
            ci_low  = _ci_to_arrays(ci_df, horizon)[0]
            ci_high = _ci_to_arrays(ci_df, horizon)[1]

        # Validate native CI
        if (not np.isfinite(pred).all()
                or not np.isfinite(ci_low).all()
                or not np.isfinite(ci_high).all()
                or (ci_low >= ci_high).any()):
            raise ValueError("Invalid native CI — falling back.")

    except Exception:
        # Residual-based fallback
        ci_method = "residual_based_fallback"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = np.asarray(fitted.forecast(steps=horizon, exog=X_future), dtype="float64")

        residuals = fitted.resid
        finite_resid = residuals[np.isfinite(residuals)]
        sigma = float(np.std(finite_resid, ddof=1)) if len(finite_resid) > 1 else 1.0
        z     = _get_z(confidence_level)
        h_arr = np.arange(1, horizon + 1, dtype="float64")
        ci_low  = (pred - z * sigma * np.sqrt(h_arr)).astype("float64")
        ci_high = (pred + z * sigma * np.sqrt(h_arr)).astype("float64")

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite forecast values detected.")

    # --------------------------------------------------
    # FUTURE DATE INDEX
    # --------------------------------------------------

    last_date  = df["date"].iloc[-1]
    freq_alias = inferred if inferred else "MS"
    future_idx = pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=freq_alias
    )[1:]

    if len(future_idx) != horizon:
        raise RuntimeError(f"Future date index mismatch: {len(future_idx)} vs {horizon}.")

    future_block = pd.DataFrame({
        "date":      future_idx,
        "actual":    np.nan,
        "forecast":  pred,
        "ci_low":    ci_low,
        "ci_mid":    pred,
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
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI values in future forecast output.")
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI detected in future forecast output.")

    return ForecastResult(
        model_name  = "DHR",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "arima_order":      (ARIMA_P, ARIMA_D, ARIMA_Q),
            "seasonal_period":  m,
            "fourier_terms_K":  K,
            "k_max_evaluated":  k_max,
            "ci_method":        ci_method,
            "aic":              round(float(fitted.aic), 4),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "pro",
            "output_contract":       "ForecastResult",
        },
    )
