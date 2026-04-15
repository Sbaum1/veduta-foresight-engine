# ==================================================
# FILE: foresight_engine/models/hybrid_models.py
# VERSION: 1.0.0
# MODELS: ARIMA+XGBoost · Prophet+XGBoost
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# ==================================================
#
# PURPOSE:
#   Hybrid models decompose the forecasting task into two stages:
#
#   Stage 1 — Structural model captures linear/periodic patterns:
#     ARIMA captures ARMA structure + integration + differencing
#     Prophet captures trend changepoints + yearly seasonality
#
#   Stage 2 — XGBoost corrects residuals:
#     The residuals from Stage 1 are a stationary time series.
#     XGBoost with lag features models the remaining non-linear
#     structure that the structural model cannot capture.
#
#   Final forecast = structural forecast + XGBoost residual correction.
#
# ACADEMIC BASIS:
#   This two-stage approach was a top performer in the M4 competition
#   (Smyl, 2020 — ES-RNN) and M5 competition. The combination of a
#   statistical model for global structure and a ML model for local
#   residual correction is now standard in production forecasting.
#
# CI METHOD:
#   Propagated uncertainty: structural CI + XGBoost residual variance.
#   The total CI is wider than either component alone — honest about
#   two sources of uncertainty.
#
# GOVERNANCE:
#   - XGBoost stage falls back to zero correction if insufficient data
#   - Prophet stage optional — falls back to ARIMA-only if prophet fails
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult per model
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

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



# Shared lag/rolling feature set — same as LightGBM/XGBoost/RF
LAG_COLS  = [1, 2, 3, 6, 12]
ROLL_WINS = [3, 6]
MIN_RESID_OBS = max(LAG_COLS) + max(ROLL_WINS) + 2   # 20

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


def _build_residual_features(resid: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build lag+seasonal features on the residual series."""
    df = pd.DataFrame({"resid": resid, "date": dates})
    feat = pd.DataFrame(index=df.index)
    for lag in LAG_COLS:
        feat[f"lag_{lag}"] = df["resid"].shift(lag)
    for w in ROLL_WINS:
        feat[f"roll_mean_{w}"] = df["resid"].shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"]  = df["resid"].shift(1).rolling(w).std()
    feat["month"]   = df["date"].dt.month.astype("float64")
    feat["quarter"] = df["date"].dt.quarter.astype("float64")
    return feat


def _fit_xgboost_on_residuals(
    resid_hist:       np.ndarray,
    dates_hist:       pd.DatetimeIndex,
    horizon:          int,
    confidence_level: float,
) -> tuple:
    """
    Fit XGBoost on historical residuals, forecast residual correction.
    Returns (correction, ci_lo_correction, ci_hi_correction).
    Falls back to (zeros, zeros, zeros) if insufficient data.
    """
    from xgboost import XGBRegressor

    n = len(resid_hist)
    if n < MIN_RESID_OBS:
        return (
            np.zeros(horizon, dtype="float64"),
            np.zeros(horizon, dtype="float64"),
            np.zeros(horizon, dtype="float64"),
        )

    alpha_ci = 1.0 - confidence_level
    q_lo     = alpha_ci / 2.0
    q_hi     = 1.0 - alpha_ci / 2.0

    feat = _build_residual_features(resid_hist, dates_hist)
    valid  = feat.notna().all(axis=1)
    X_tr   = feat[valid].values.astype("float64")
    y_tr   = resid_hist[valid]
    f_cols = feat.columns.tolist()

    if len(X_tr) < 6:
        return (
            np.zeros(horizon, dtype="float64"),
            np.zeros(horizon, dtype="float64"),
            np.zeros(horizon, dtype="float64"),
        )

    XGB_PARAMS = dict(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )

    # Check quantile support
    def _check_quantile():
        import xgboost as xgb
        major, minor = [int(x) for x in xgb.__version__.split(".")[:2]]
        return (major, minor) >= (1, 7)

    use_quantile = _check_quantile()

    model_point = XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
    model_point.fit(pd.DataFrame(X_tr, columns=f_cols), y_tr)

    model_lo = model_hi = None
    if use_quantile:
        try:
            model_lo = XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q_lo, **XGB_PARAMS)
            model_hi = XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q_hi, **XGB_PARAMS)
            model_lo.fit(pd.DataFrame(X_tr, columns=f_cols), y_tr)
            model_hi.fit(pd.DataFrame(X_tr, columns=f_cols), y_tr)
        except Exception:
            use_quantile = False
            model_lo = model_hi = None

    # Residual sigma for fallback CI
    fitted_resid = model_point.predict(pd.DataFrame(X_tr, columns=f_cols))
    resid_of_resid = y_tr - fitted_resid
    sigma_resid = float(np.std(resid_of_resid, ddof=1)) if len(resid_of_resid) > 1 \
                  else float(np.abs(resid_of_resid).mean() + 1e-8)

    # Recursive future residual correction
    from scipy.stats import norm as _norm
    z_lo = float(_norm.ppf(q_lo))
    z_hi = float(_norm.ppf(q_hi))

    resid_extended = list(resid_hist)
    future_dates   = pd.date_range(
        start=dates_hist[-1], periods=horizon + 1,
        freq=pd.infer_freq(dates_hist) or "MS"
    )[1:]

    corrections, ci_lo_corr, ci_hi_corr = [], [], []
    for step in range(horizon):
        all_dates = pd.DatetimeIndex(
            list(dates_hist) + list(future_dates[:step])
        )
        feat_ext = _build_residual_features(
            np.array(resid_extended, dtype="float64"), all_dates
        )
        row_idx = len(resid_extended) - 1
        x_row   = feat_ext.loc[row_idx, f_cols].values.astype("float64")

        if not np.isfinite(x_row).all():
            c  = corrections[-1] if corrections else 0.0
            lo = c - sigma_resid * 0.5
            hi = c + sigma_resid * 0.5
        else:
            x_df = pd.DataFrame(x_row.reshape(1, -1), columns=f_cols)
            c    = float(model_point.predict(x_df)[0])
            if use_quantile and model_lo is not None:
                lo = float(model_lo.predict(x_df)[0])
                hi = float(model_hi.predict(x_df)[0])
            else:
                grow = float(np.sqrt(step + 1))
                lo   = c + z_lo * sigma_resid * grow
                hi   = c + z_hi * sigma_resid * grow

        corrections.append(c)
        ci_lo_corr.append(lo)
        ci_hi_corr.append(hi)
        resid_extended.append(c)

    return (
        np.array(corrections,  dtype="float64"),
        np.array(ci_lo_corr,   dtype="float64"),
        np.array(ci_hi_corr,   dtype="float64"),
    )


def _assemble_forecast_df(
    df: pd.DataFrame,
    hist_fitted: np.ndarray,
    future_index: pd.DatetimeIndex,
    future_forecast: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  hist_fitted.astype("float64"),
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.astype("float64"),
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })
    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_forecast,
        "ci_low":    np.minimum(ci_low,  future_forecast),
        "ci_mid":    future_forecast,
        "ci_high":   np.maximum(ci_high, future_forecast),
        "error_pct": np.nan,
    })
    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")
    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError(f"Duplicate dates in {model_name} output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)
    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError(f"NaN CI in {model_name} future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError(f"Inverted CI in {model_name} output.")
    return forecast_df


# ==============================================================================
# ARIMA + XGBOOST
# ==============================================================================

def run_arima_xgboost(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Stage 1: Auto-ARIMA fits global structure (trend, integration, MA terms).
    Stage 2: XGBoost models the ARIMA residuals with lag/seasonal features.
    Final:   ARIMA forecast + XGBoost residual correction.
    """
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("ARIMA+XGBoost requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")
    df = df.sort_values("date").reset_index(drop=True)
    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if df["value"].isna().any():
        raise ValueError("Missing values in 'value' column.")

    y = df["value"].astype("float64").values
    n = len(y)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")
    if n < 24:
        raise ValueError("ARIMA+XGBoost requires >= 24 observations.")

    dates = pd.to_datetime(df["date"].values)

    # ── Stage 1: ARIMA ────────────────────────────────────────────────────────
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tsa.stattools import adfuller

    # ADF test for integration order
    try:
        adf_pval = float(adfuller(y, autolag="AIC")[1])
        d = 1 if adf_pval > 0.05 else 0
    except Exception:
        d = 1

    # Grid search over (p, q) with fixed d
    best_aic    = float("inf")
    best_order  = (1, d, 1)
    best_fitted = None

    for p in range(0, 4):
        for q in range(0, 3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = _ARIMA(y, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic    = m.aic
                        best_order  = (p, d, q)
                        best_fitted = m
            except Exception:
                continue

    if best_fitted is None:
        raise RuntimeError("ARIMA fitting failed for all (p,d,q) combinations.")

    # ARIMA in-sample fitted values
    arima_fitted = np.asarray(best_fitted.fittedvalues).astype("float64")
    if len(arima_fitted) < n:
        pad = np.full(n - len(arima_fitted), arima_fitted[0] if len(arima_fitted) > 0 else y[0])
        arima_fitted = np.concatenate([pad, arima_fitted])
    arima_fitted = arima_fitted[:n]

    # ARIMA future forecast + CI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_pred = best_fitted.get_forecast(steps=horizon)
    arima_mean   = arima_pred.predicted_mean.astype("float64")
    arima_ci     = arima_pred.conf_int(alpha=1 - confidence_level)
    arima_ci_lo, arima_ci_hi = _ci_to_arrays(arima_ci, horizon)
    

    # ── Stage 2: XGBoost on residuals ────────────────────────────────────────
    resid_hist = (y - arima_fitted).astype("float64")

    correction, ci_lo_corr, ci_hi_corr = _fit_xgboost_on_residuals(
        resid_hist, dates, horizon, confidence_level
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    future_forecast = (arima_mean + correction).astype("float64")
    ci_low  = (arima_ci_lo + ci_lo_corr).astype("float64")
    ci_high = (arima_ci_hi + ci_hi_corr).astype("float64")
    ci_low  = np.minimum(ci_low,  future_forecast)
    ci_high = np.maximum(ci_high, future_forecast)

    hist_fitted = (arima_fitted + resid_hist * 0.0).astype("float64")  # = arima_fitted

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in ARIMA+XGBoost forecast.")

    future_index = pd.date_range(
        start=dates[-1], periods=horizon + 1,
        freq=inferred or "MS"
    )[1:]

    forecast_df = _assemble_forecast_df(
        df, hist_fitted, future_index, future_forecast,
        ci_low, ci_high, horizon, "ARIMA+XGBoost"
    )

    return ForecastResult(
        model_name  = "ARIMA+XGBoost",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "arima_order":      best_order,
            "arima_aic":        round(float(best_aic), 4),
            "xgb_n_estimators": 200,
            "residual_correction": bool(np.any(correction != 0)),
            "ci_method":        "arima_ci_plus_xgb_residual_variance",
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# PROPHET + XGBOOST
# ==============================================================================

def run_prophet_xgboost(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Stage 1: Prophet fits piecewise trend + yearly seasonality.
    Stage 2: XGBoost models the Prophet residuals.
    Final:   Prophet forecast + XGBoost residual correction.

    Falls back to XGBoost-only if Prophet fails (not installed / fit error).
    """
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Prophet+XGBoost requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")
    df = df.sort_values("date").reset_index(drop=True)
    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if df["value"].isna().any():
        raise ValueError("Missing values in 'value' column.")

    y = df["value"].astype("float64").values
    n = len(y)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")
    if n < 24:
        raise ValueError("Prophet+XGBoost requires >= 24 observations.")

    dates      = pd.to_datetime(df["date"].values)
    prophet_ok = False

    # ── Stage 1: Prophet ─────────────────────────────────────────────────────
    try:
        from prophet import Prophet as _Prophet

        prophet_df = pd.DataFrame({"ds": dates, "y": y})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _Prophet(
                interval_width          = confidence_level,
                uncertainty_samples     = 1000,
                changepoint_prior_scale = 0.30,
                changepoint_range       = 0.95,
                n_changepoints          = 30,
                seasonality_mode        = "multiplicative",
                seasonality_prior_scale = 3.0,
                daily_seasonality       = False,
                weekly_seasonality      = False,
                yearly_seasonality      = True,
            )
            model.fit(prophet_df)

        # In-sample fitted values
        fitted_hist_df = model.predict(prophet_df)

        # Guard: CI columns may be absent if uncertainty_samples=0
        if "yhat_lower" not in fitted_hist_df.columns:
            import scipy.stats as _st
            res = y - fitted_hist_df["yhat"].values
            sig = float(np.std(res, ddof=1))
            z   = float(_st.norm.ppf((1 + confidence_level) / 2))
            fitted_hist_df["yhat_lower"] = fitted_hist_df["yhat"] - z * sig
            fitted_hist_df["yhat_upper"] = fitted_hist_df["yhat"] + z * sig

        prophet_fitted = fitted_hist_df["yhat"].values.astype("float64")

        # Future forecast
        future_dates_df = model.make_future_dataframe(
            periods=horizon, freq=inferred or "MS", include_history=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prophet_future = model.predict(future_dates_df)

        if "yhat_lower" not in prophet_future.columns:
            import scipy.stats as _st
            res = y - prophet_fitted
            sig = float(np.std(res, ddof=1))
            z   = float(_st.norm.ppf((1 + confidence_level) / 2))
            h   = np.arange(1, horizon + 1, dtype="float64")
            prophet_future["yhat_lower"] = prophet_future["yhat"] - z * sig * np.sqrt(h)
            prophet_future["yhat_upper"] = prophet_future["yhat"] + z * sig * np.sqrt(h)

        prophet_mean   = prophet_future["yhat"].values.astype("float64")
        prophet_ci_lo  = prophet_future["yhat_lower"].values.astype("float64")
        prophet_ci_hi  = prophet_future["yhat_upper"].values.astype("float64")
        future_index   = pd.DatetimeIndex(prophet_future["ds"].values)
        prophet_ok     = True

    except Exception:
        # Prophet not available or fit failed — use ARIMA as structural component
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        prophet_fitted = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                arima_m      = _ARIMA(y, order=(1, 1, 1)).fit()
                prophet_fitted = np.asarray(arima_m.fittedvalues).astype("float64")
                if len(prophet_fitted) < n:
                    pad = np.full(n - len(prophet_fitted), y[0])
                    prophet_fitted = np.concatenate([pad, prophet_fitted])
                arima_pred    = arima_m.get_forecast(steps=horizon)
                prophet_mean  = arima_pred.predicted_mean.astype("float64")
                fallback_ci   = arima_pred.conf_int(alpha=1 - confidence_level)
                prophet_ci_lo, prophet_ci_hi = _ci_to_arrays(fallback_ci, horizon)
                
                future_index  = pd.date_range(
                    start=dates[-1], periods=horizon + 1,
                    freq=inferred or "MS"
                )[1:]
            except Exception as e:
                raise RuntimeError(f"Both Prophet and ARIMA fallback failed: {e}") from e

    # ── Stage 2: XGBoost on residuals ─────────────────────────────────────────
    resid_hist = (y - prophet_fitted).astype("float64")

    correction, ci_lo_corr, ci_hi_corr = _fit_xgboost_on_residuals(
        resid_hist, dates, horizon, confidence_level
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    future_forecast = (prophet_mean + correction).astype("float64")
    ci_low  = (prophet_ci_lo + ci_lo_corr).astype("float64")
    ci_high = (prophet_ci_hi + ci_hi_corr).astype("float64")
    ci_low  = np.minimum(ci_low,  future_forecast)
    ci_high = np.maximum(ci_high, future_forecast)

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in Prophet+XGBoost forecast.")

    forecast_df = _assemble_forecast_df(
        df, prophet_fitted, future_index, future_forecast,
        ci_low, ci_high, horizon, "Prophet+XGBoost"
    )

    return ForecastResult(
        model_name  = "Prophet+XGBoost",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "structural_model":    "Prophet" if prophet_ok else "ARIMA(1,1,1)_fallback",
            "prophet_available":   prophet_ok,
            "xgb_n_estimators":    200,
            "residual_correction": bool(np.any(correction != 0)),
            "ci_method":           "prophet_ci_plus_xgb_residual_variance",
            "frequency":           inferred,
            "confidence_level":    confidence_level,
            "min_tier":            "enterprise",
            "output_contract":     "ForecastResult",
        },
    )
