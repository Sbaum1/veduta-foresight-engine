# ==================================================
# FILE: foresight_engine/models/varima.py
# VERSION: 1.0.0
# MODEL: VARIMA — VECTOR AUTOREGRESSION WITH INTEGRATION
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# ==================================================
#
# PURPOSE:
#   VARIMA extends VAR to handle non-stationary multivariate
#   series. Where VAR operates on levels, VARIMA differences
#   each series to achieve stationarity (as indicated by ADF),
#   fits the VAR on differences, then integrates (cumsum) the
#   forecast back to the original level scale.
#
#   Best on: correlated revenue series where individual series
#   are non-stationary but the relationships between them are
#   stable over time (co-integration in the broad sense).
#
# DIFFERENCE FROM var_model.py:
#   var_model.py has basic differencing but does not perform
#   per-column ADF testing or multi-order integration.
#   VARIMA performs per-column ADF tests and applies the
#   correct integration order (0, 1, or 2) per column.
#
# UNIVARIATE FALLBACK:
#   If only one numeric column is present, falls back to
#   ARIMA on the single series — same behaviour as var_model.py.
#   This ensures VARIMA never DQs on single-series input.
#
# CI METHOD:
#   Native statsmodels VAR forecast_interval() on differenced
#   series, integrated back. Falls back to residual-based.
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
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

from .contracts import ForecastResult

def _ci_to_arrays(ci_result, n: int) -> tuple:
    """
    Extract lower/upper confidence interval arrays from statsmodels result.
    Handles both DataFrame (statsmodels <= 0.13) and ndarray (>= 0.14) return types.
    Returns (lower: np.ndarray, upper: np.ndarray).
    """
    import numpy as np
    import pandas as pd
    if hasattr(ci_result, 'iloc'):
        # DataFrame path (statsmodels <= 0.13)
        lo = np.asarray(ci_result.iloc[:, 0], dtype="float64")
        hi = np.asarray(ci_result.iloc[:, 1], dtype="float64")
    elif hasattr(ci_result, 'values'):
        lo = ci_result.values[:, 0].astype("float64")
        hi = ci_result.values[:, 1].astype("float64")
    else:
        # ndarray path (statsmodels >= 0.14)
        arr = np.asarray(ci_result, dtype="float64")
        if arr.ndim == 2 and arr.shape[1] >= 2:
            lo, hi = arr[:, 0], arr[:, 1]
        else:
            # Degenerate — widen from point forecast shape
            lo = arr.flatten()
            hi = arr.flatten()
    return lo, hi



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
            t = (cl - lo) / (hi - lo)
            return _Z[lo] + t * (_Z[hi] - _Z[lo])
    return 1.960


def _integration_order(series: np.ndarray, alpha: float = ADF_ALPHA) -> int:
    """
    Determine integration order (0, 1, or 2) via sequential ADF tests.
    Returns d such that diff(series, d) is stationary.
    """
    for d in range(3):
        try:
            test_series = np.diff(series, n=d) if d > 0 else series
            if len(test_series) < 8:
                return d
            pval = float(adfuller(test_series, autolag="AIC")[1])
            if pval < alpha:
                return d
        except Exception:
            return d
    return 2


def _arima_univariate_fallback(
    df: pd.DataFrame,
    primary_col: str,
    inferred: str,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:
    """ARIMA(1,1,1) fallback for single-series input."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    y     = df[primary_col].astype("float64").values
    dates = pd.to_datetime(df["date"].values)
    n     = len(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _ARIMA(y, order=(1, 1, 1)).fit()

    fv = np.asarray(m.fittedvalues).astype("float64")
    if len(fv) < n:
        fv = np.concatenate([np.full(n - len(fv), y[0]), fv])

    pred   = m.get_forecast(steps=horizon)
    fc     = pred.predicted_mean.astype("float64")
    ci     = pred.conf_int(alpha=1 - confidence_level)
    ci_lo, ci_hi = _ci_to_arrays(ci, horizon)
    

    future_idx = pd.date_range(start=dates[-1], periods=horizon + 1,
                               freq=inferred or "MS")[1:]

    hist_block = pd.DataFrame({
        "date": dates, "actual": np.nan,
        "forecast": fv, "ci_low": np.nan,
        "ci_mid": fv, "ci_high": np.nan, "error_pct": np.nan,
    })
    future_block = pd.DataFrame({
        "date": future_idx, "actual": np.nan,
        "forecast": fc,
        "ci_low":   np.minimum(ci_lo, fc),
        "ci_mid":   fc,
        "ci_high":  np.maximum(ci_hi, fc),
        "error_pct": np.nan,
    })
    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    fc_df = pd.concat([hist_block, future_block], ignore_index=True)
    fc_df = fc_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "VARIMA",
        forecast_df = fc_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "mode":             "arima_univariate_fallback",
            "reason":           "Single series — VARIMA requires ≥2. ARIMA(1,1,1) used.",
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


def run_varima(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns:
        raise ValueError("VARIMA requires a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")
    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    series_cols = [c for c in df.columns if c != "date"
                   and pd.api.types.is_numeric_dtype(df[c])]

    for col in series_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values in column '{col}'.")
        if not np.isfinite(df[col].astype("float64").values).all():
            raise ValueError(f"Non-finite values in column '{col}'.")

    primary_col  = "value" if "value" in series_cols else series_cols[0]

    # ── Univariate fallback ───────────────────────────────────────────────────
    if len(series_cols) < 2:
        return _arima_univariate_fallback(
            df, primary_col, inferred, horizon, confidence_level
        )

    if len(df) < MIN_OBS:
        raise ValueError(f"Minimum {MIN_OBS} observations required for VARIMA.")

    companion_cols = [c for c in series_cols if c != primary_col]
    ordered_cols   = [primary_col] + companion_cols
    data           = df[ordered_cols].astype("float64").values
    n, k           = data.shape

    # ── Per-column integration order ─────────────────────────────────────────
    d_orders = np.array([_integration_order(data[:, j]) for j in range(k)])
    max_d    = int(d_orders.max())

    # Difference each column to its required order
    data_diff = data.copy()
    for j in range(k):
        if d_orders[j] > 0:
            data_diff[:, j] = np.concatenate([
                np.zeros(d_orders[j]),
                np.diff(data[:, j], n=int(d_orders[j]))
            ])

    # Drop rows with leading zeros from differencing
    data_fit = data_diff[max_d:]
    n_fit    = len(data_fit)

    # ── Lag selection ─────────────────────────────────────────────────────────
    max_lags = min(MAX_LAGS, n_fit // 5)
    max_lags = max(max_lags, 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel      = VAR(data_fit)
            lag_res  = sel.select_order(maxlags=max_lags)
            best_lag = max(1, int(lag_res.aic))
    except Exception:
        best_lag = 1

    # ── Fit VAR on differenced data ───────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_model  = VAR(data_fit)
            var_fitted = var_model.fit(maxlags=best_lag, ic=None)
    except Exception as e:
        raise RuntimeError(f"VARIMA VAR fit failed: {e}") from e

    # ── Historical fitted values (primary series) ─────────────────────────────
    fv_diff = var_fitted.fittedvalues[:, 0].astype("float64")
    d0      = int(d_orders[0])

    # Integrate fitted differences back to level
    fit_offset  = n - len(fv_diff)
    if d0 == 0:
        fv_level = fv_diff
        pad_val  = data[fit_offset, 0]
    elif d0 == 1:
        start_val = data[fit_offset, 0]
        fv_level  = np.empty(len(fv_diff))
        fv_level[0] = start_val + fv_diff[0]
        for t in range(1, len(fv_diff)):
            fv_level[t] = fv_level[t - 1] + fv_diff[t]
    else:  # d=2
        start_level = data[fit_offset, 0]
        start_diff  = data[fit_offset, 0] - (data[fit_offset - 1, 0] if fit_offset > 0 else data[fit_offset, 0])
        fv_level    = np.empty(len(fv_diff))
        lvl, dlt    = start_level, start_diff
        for t in range(len(fv_diff)):
            dlt     = dlt + fv_diff[t]
            lvl     = lvl + dlt
            fv_level[t] = lvl

    fitted_full        = np.full(n, np.nan)
    fitted_full[fit_offset:] = fv_level

    # ── Future forecast ───────────────────────────────────────────────────────
    alpha_ci  = 1.0 - confidence_level
    ci_method = "native_var_integrated"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_res = var_fitted.forecast_interval(
                y     = data_fit[-best_lag:],
                steps = horizon,
                alpha = alpha_ci,
            )
        fc_diff_mid = np.asarray(fc_res[0], dtype="float64")[:, 0]
        fc_diff_lo  = np.asarray(fc_res[1], dtype="float64")[:, 0]
        fc_diff_hi  = np.asarray(fc_res[2], dtype="float64")[:, 0]

        # Integrate back to levels
        def _integrate(diffs: np.ndarray, start: float, d: int) -> np.ndarray:
            if d == 0:
                return diffs
            lvl = np.empty(len(diffs))
            lvl[0] = start + diffs[0]
            for t in range(1, len(diffs)):
                lvl[t] = lvl[t - 1] + diffs[t]
            return lvl

        last_level = float(data[-1, 0])
        fc_mid     = _integrate(fc_diff_mid, last_level, d0)
        fc_lo      = _integrate(fc_diff_lo,  last_level, d0)
        fc_hi      = _integrate(fc_diff_hi,  last_level, d0)

        if not (np.isfinite(fc_mid).all() and np.isfinite(fc_lo).all()
                and np.isfinite(fc_hi).all() and not (fc_lo >= fc_hi).any()):
            raise ValueError("Invalid integrated forecast.")

    except Exception:
        # Residual-based fallback
        ci_method = "residual_based_fallback"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_raw = var_fitted.forecast(
                y=data_fit[-best_lag:], steps=horizon
            )
        fc_diff_mid = np.asarray(fc_raw, dtype="float64")[:, 0]
        last_level  = float(data[-1, 0])
        fc_mid      = last_level + np.cumsum(fc_diff_mid) if d0 > 0 else fc_diff_mid

        resid_primary = data[fit_offset:, 0] - fitted_full[fit_offset:]
        finite_resid  = resid_primary[np.isfinite(resid_primary)]
        sigma         = float(np.std(finite_resid, ddof=1)) if len(finite_resid) > 1 else 1.0
        z             = _get_z(confidence_level)
        h_arr         = np.arange(1, horizon + 1, dtype="float64")
        fc_lo         = (fc_mid - z * sigma * np.sqrt(h_arr)).astype("float64")
        fc_hi         = (fc_mid + z * sigma * np.sqrt(h_arr)).astype("float64")

    fc_mid = fc_mid.astype("float64")
    fc_lo  = np.minimum(fc_lo, fc_mid).astype("float64")
    fc_hi  = np.maximum(fc_hi, fc_mid).astype("float64")

    if not np.isfinite(fc_mid).all():
        raise RuntimeError("Non-finite values in VARIMA forecast.")

    future_idx = pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1,
        freq=inferred or "MS"
    )[1:]

    hist_block = pd.DataFrame({
        "date": pd.to_datetime(df["date"].values), "actual": np.nan,
        "forecast": fitted_full, "ci_low": np.nan,
        "ci_mid": fitted_full, "ci_high": np.nan, "error_pct": np.nan,
    })
    future_block = pd.DataFrame({
        "date": future_idx, "actual": np.nan,
        "forecast": fc_mid, "ci_low": fc_lo,
        "ci_mid": fc_mid, "ci_high": fc_hi, "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in VARIMA output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in VARIMA future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in VARIMA output.")

    return ForecastResult(
        model_name  = "VARIMA",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "primary_series":   primary_col,
            "companion_series": companion_cols,
            "n_series":         k,
            "lag_order":        best_lag,
            "integration_orders": d_orders.tolist(),
            "max_integration":  int(max_d),
            "ci_method":        ci_method,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )
