# ==================================================
# FILE: foresight_engine/models/var_model.py
# VERSION: 2.0.0
# MODEL: VAR — VECTOR AUTOREGRESSION
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# STATUS: VEDUTA ENGINE — PHASE 3C
# ==================================================
#
# PURPOSE:
#   Vector Autoregression models multiple time series
#   simultaneously, capturing cross-series dependencies
#   that univariate models cannot see.
#
#   Best on: correlated series where one drives another.
#   Examples: revenue + headcount, multiple SKU categories,
#   macro indicators + sales, regional sales with spillover.
#
# INPUT CONTRACT:
#   df must contain 'date' + at least 2 numeric series columns.
#   The PRIMARY series for forecasting is the first numeric
#   column after 'date' (or column named 'value' if present).
#   All other numeric columns are treated as companion series.
#
# OUTPUT CONTRACT:
#   ForecastResult contains forecast for the PRIMARY series only.
#   Standard foresight_engine output schema preserved.
#
# LAG ORDER SELECTION:
#   AIC-based selection from maxlags=min(12, n//5).
#   Falls back to lag=1 if AIC selection fails.
#
# CI METHOD:
#   Native statsmodels VAR forecast_interval().
#   Falls back to residual-based if native CI is invalid.
#
# STATIONARITY:
#   First-difference applied if ADF test rejects stationarity.
#   Forecast is integrated back (cumsum) before output.
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

from foresight_engine.models.contracts import ForecastResult

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MAX_LAGS  = 12
MIN_OBS   = 24
ADF_ALPHA = 0.05

_Z = {
    0.50: 0.674, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.99: 2.576,
}

def _get_z(cl: float) -> float:
    if cl in _Z:
        return _Z[cl]
    levels = sorted(_Z.keys())
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        if lo <= cl <= hi:
            return _Z[lo] + (cl - lo) / (hi - lo) * (_Z[hi] - _Z[lo])
    return 1.960


# --------------------------------------------------
# STATIONARITY CHECK
# --------------------------------------------------

def _is_stationary(series: np.ndarray, alpha: float = ADF_ALPHA) -> bool:
    try:
        result = adfuller(series, autolag="AIC")
        return float(result[1]) < alpha
    except Exception:
        return True   # Assume stationary if test fails


# ==================================================
# UNIVARIATE FALLBACK — AUTOREG
# Used when only one series is provided (VAR needs ≥2).
# AutoReg captures the same AR structure univariately.
# ==================================================

def _run_autoreg_fallback(
    df:               pd.DataFrame,
    series_cols:      list,
    inferred:         str,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    from statsmodels.tsa.ar_model import AutoReg as _AutoReg

    primary_col = "value" if "value" in series_cols else series_cols[0]
    y           = df[primary_col].astype("float64").values
    n           = len(y)
    dates       = pd.to_datetime(df["date"].values)

    # AIC lag selection
    best_aic, best_lag = float("inf"), 1
    for lag in range(1, min(13, n // 4)):
        try:
            m = _AutoReg(y, lags=lag, old_names=False).fit(cov_type="HC0")
            if m.aic < best_aic:
                best_aic, best_lag = m.aic, lag
        except Exception:
            pass

    model  = _AutoReg(y, lags=best_lag, old_names=False).fit(cov_type="HC0")
    fitted = model.fittedvalues

    # Historical block
    fitted_full        = np.full(n, np.nan)
    fitted_full[best_lag:] = np.asarray(fitted).astype("float64")

    hist_block = pd.DataFrame({
        "date":      dates,
        "actual":    np.nan,
        "forecast":  fitted_full,
        "ci_low":    np.nan,
        "ci_mid":    fitted_full,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # Future forecast
    start      = n
    end        = n + horizon - 1
    future_fc  = model.predict(start=start, end=end).astype("float64")

    resid      = y[best_lag:] - np.asarray(fitted).astype("float64")
    sigma      = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.abs(resid).mean())
    z          = _get_z(confidence_level)
    h_arr      = np.arange(1, horizon + 1, dtype="float64")
    ci_lo      = (future_fc - z * sigma * np.sqrt(h_arr)).astype("float64")
    ci_hi      = (future_fc + z * sigma * np.sqrt(h_arr)).astype("float64")
    ci_lo      = np.minimum(ci_lo, future_fc)
    ci_hi      = np.maximum(ci_hi, future_fc)

    freq_alias  = inferred if inferred else "MS"
    future_idx  = pd.date_range(start=dates[-1], periods=horizon + 1, freq=freq_alias)[1:]

    future_block = pd.DataFrame({
        "date":      future_idx,
        "actual":    np.nan,
        "forecast":  future_fc,
        "ci_low":    ci_lo,
        "ci_mid":    future_fc,
        "ci_high":   ci_hi,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "VAR",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "mode":             "autoreg_univariate_fallback",
            "lag_order":        best_lag,
            "reason":           "Single series — VAR requires ≥2. AutoReg used.",
            "ci_method":        "residual_based",
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


# ==================================================
# MODEL RUNNER
# ==================================================

def run_var(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns:
        raise ValueError("VAR requires a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    # Identify numeric series columns
    series_cols = [c for c in df.columns if c != "date"
                   and pd.api.types.is_numeric_dtype(df[c])]

    if len(series_cols) < 2:
        # Univariate fallback — AutoReg when only one series provided
        # VAR is multivariate by design, but graceful degradation is better
        # than disqualification. AutoReg captures the same AR structure.
        return _run_autoreg_fallback(df, series_cols, inferred, horizon, confidence_level)

    for col in series_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values in column '{col}'.")
        if not np.isfinite(df[col].astype("float64").values).all():
            raise ValueError(f"Non-finite values in column '{col}'.")

    if len(df) < MIN_OBS:
        raise ValueError(f"Minimum {MIN_OBS} observations required for VAR.")

    # Primary series: 'value' if present, else first numeric column
    primary_col = "value" if "value" in series_cols else series_cols[0]
    companion_cols = [c for c in series_cols if c != primary_col]

    # --------------------------------------------------
    # BUILD MULTIVARIATE ARRAY
    # --------------------------------------------------

    # Primary first, companions after
    ordered_cols = [primary_col] + companion_cols
    data = df[ordered_cols].astype("float64").values   # shape (n, k)
    n, k = data.shape

    # --------------------------------------------------
    # STATIONARITY — DIFFERENCE IF NEEDED
    # --------------------------------------------------

    differenced = np.zeros(k, dtype=bool)
    data_stat   = data.copy()

    for j in range(k):
        if not _is_stationary(data[:, j]):
            data_stat[1:, j] = np.diff(data[:, j])
            data_stat[0,  j] = 0.0
            differenced[j]   = True

    # Drop first row (lost to differencing)
    data_fit = data_stat[1:, :]
    n_fit    = len(data_fit)

    # --------------------------------------------------
    # LAG ORDER SELECTION
    # --------------------------------------------------

    max_lags = min(MAX_LAGS, n_fit // 5)
    max_lags = max(max_lags, 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_selector = VAR(data_fit)
            lag_order    = var_selector.select_order(maxlags=max_lags)
            best_lag     = lag_order.aic
            best_lag     = max(1, int(best_lag))
    except Exception:
        best_lag = 1

    # --------------------------------------------------
    # FIT VAR MODEL
    # --------------------------------------------------

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = VAR(data_fit)
            fitted = model.fit(maxlags=best_lag, ic=None)
    except Exception as e:
        raise RuntimeError(f"VAR model fit failed: {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES (primary series only)
    # --------------------------------------------------

    fitted_values_stat = fitted.fittedvalues[:, 0].astype("float64")

    # Integrate back if primary series was differenced
    if differenced[0]:
        # fitted_values_stat are differences; integrate
        start_val = data[best_lag, 0]   # value at start of fit window
        fitted_primary = np.empty(len(fitted_values_stat))
        fitted_primary[0] = start_val + fitted_values_stat[0]
        for t in range(1, len(fitted_values_stat)):
            fitted_primary[t] = fitted_primary[t-1] + fitted_values_stat[t]
    else:
        fitted_primary = fitted_values_stat

    # Align to original date index
    fit_offset = n - len(fitted_primary)
    fitted_full = np.full(n, np.nan)
    fitted_full[fit_offset:] = fitted_primary

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_full,
        "ci_low":    np.nan,
        "ci_mid":    fitted_full,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    alpha_ci  = 1.0 - confidence_level
    ci_method = "native_var"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = fitted.forecast_interval(
                y     = data_fit[-best_lag:],
                steps = horizon,
                alpha = alpha_ci,
            )
        # fc is (mid, lower, upper) — each shape (horizon, k)
        fc_mid = np.asarray(fc[0], dtype="float64")[:, 0]
        fc_lo  = np.asarray(fc[1], dtype="float64")[:, 0]
        fc_hi  = np.asarray(fc[2], dtype="float64")[:, 0]

        # Integrate back if primary was differenced
        if differenced[0]:
            last_val = data[-1, 0]
            fc_mid = last_val + np.cumsum(fc_mid)
            fc_lo  = last_val + np.cumsum(fc_lo)
            fc_hi  = last_val + np.cumsum(fc_hi)

        if (not np.isfinite(fc_mid).all()
                or not np.isfinite(fc_lo).all()
                or not np.isfinite(fc_hi).all()
                or (fc_lo >= fc_hi).any()):
            raise ValueError("Invalid native CI.")

    except Exception:
        # Residual-based fallback on primary series
        ci_method = "residual_based_fallback"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_raw = fitted.forecast(
                y=data_fit[-best_lag:], steps=horizon
            )
        fc_mid = np.asarray(fc_raw, dtype="float64")[:, 0]

        if differenced[0]:
            last_val = data[-1, 0]
            fc_mid   = last_val + np.cumsum(fc_mid)

        resid_primary = data[fit_offset:, 0] - fitted_full[fit_offset:]
        finite_resid  = resid_primary[np.isfinite(resid_primary)]
        sigma  = float(np.std(finite_resid, ddof=1)) if len(finite_resid) > 1 else 1.0
        z      = _get_z(confidence_level)
        h_arr  = np.arange(1, horizon + 1, dtype="float64")
        fc_lo  = (fc_mid - z * sigma * np.sqrt(h_arr)).astype("float64")
        fc_hi  = (fc_mid + z * sigma * np.sqrt(h_arr)).astype("float64")

    fc_mid = fc_mid.astype("float64")
    fc_lo  = np.minimum(fc_lo, fc_mid).astype("float64")
    fc_hi  = np.maximum(fc_hi, fc_mid).astype("float64")

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
        "forecast":  fc_mid,
        "ci_low":    fc_lo,
        "ci_mid":    fc_mid,
        "ci_high":   fc_hi,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # DTYPE GOVERNANCE
    # --------------------------------------------------

    numeric_cols_out = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols_out]   = hist_block[numeric_cols_out].astype("float64")
    future_block[numeric_cols_out] = future_block[numeric_cols_out].astype("float64")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in future forecast output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in future forecast output.")

    return ForecastResult(
        model_name  = "VAR",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "primary_series":   primary_col,
            "companion_series": companion_cols,
            "n_series":         k,
            "lag_order":        best_lag,
            "differenced":      differenced.tolist(),
            "ci_method":        ci_method,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":       "ForecastResult",
        },
    )
