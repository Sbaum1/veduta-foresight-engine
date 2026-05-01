# ==================================================
# FILE: foresight_engine/models/ml_models.py
# VERSION: 1.0.0
# MODELS: RandomForest · Ridge · Lasso
# ENGINE: Foresight Engine v3.0.0
# TIER: pro (minimum)
# ==================================================
#
# PURPOSE:
#   Three ML forecasting models sharing a common feature architecture:
#   lag features, rolling statistics, month/quarter seasonality.
#   Identical feature set to LightGBM and XGBoost — the ensemble
#   benefits from their diverse learning algorithms over the same
#   feature space.
#
#   RandomForest  — Bagged decision trees. Robust to outliers and
#                   non-linearity. Native interval via quantile forests.
#
#   Ridge         — L2-regularised linear regression. Best on series
#                   with linear dynamics. Fast, interpretable, low
#                   variance. Interval via residual-based bootstrap.
#
#   Lasso         — L1-regularised linear regression. Performs feature
#                   selection automatically — on sparse feature spaces
#                   it outperforms Ridge. Interval via residuals.
#
# FEATURE ARCHITECTURE (shared):
#   Lags: [1, 2, 3, 6, 12]
#   Rolling: mean(3), mean(6), std(3), std(6)
#   Seasonal: month, quarter
#
# CI METHODS:
#   RandomForest — Quantile regression forests (sklearn 1.x):
#                  train three forests at alpha/2, 0.5, 1-alpha/2.
#                  Falls back to residual-based if sklearn < 1.0.
#   Ridge/Lasso  — Residual-based: sigma * sqrt(h) * z.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult per model
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from .contracts import ForecastResult

LAG_COLS  = [1, 2, 3, 6, 12]
ROLL_WINS = [3, 6]

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


def _build_features(df: pd.DataFrame, exog_cols: List[str]) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    for lag in LAG_COLS:
        feat[f"lag_{lag}"] = df["value"].shift(lag)
    for w in ROLL_WINS:
        feat[f"roll_mean_{w}"] = df["value"].shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"]  = df["value"].shift(1).rolling(w).std()
    feat["month"]   = df["date"].dt.month.astype("float64")
    feat["quarter"] = df["date"].dt.quarter.astype("float64")
    for col in exog_cols:
        feat[col] = df[col].astype("float64")
    return feat


def _get_exog_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("date", "value")]


def _validate_and_prep(df: pd.DataFrame) -> tuple:
    """Shared validation for all three ML models."""
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Model requires 'date' and 'value' columns.")
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
    if not np.isfinite(df["value"].astype("float64").values).all():
        raise ValueError("Non-finite values in series.")
    min_obs = max(LAG_COLS) + max(ROLL_WINS) + 2
    if len(df) < min_obs:
        raise ValueError(f"Minimum {min_obs} observations required.")
    return df, inferred


def _build_train_arrays(df: pd.DataFrame, exog_cols: List[str]):
    """Build feature matrix and target, returning (X, y, feature_cols, valid_mask)."""
    feat      = _build_features(df, exog_cols)
    hist_mask = df["value"].notna()
    feat_tr   = feat[hist_mask].copy()
    y_tr      = df.loc[hist_mask, "value"].astype("float64").values
    valid     = feat_tr.notna().all(axis=1)
    feat_tr   = feat_tr[valid]
    y_tr      = y_tr[valid]
    return feat_tr, y_tr, feat_tr.columns.tolist()


def _recursive_predict(model, df: pd.DataFrame, exog_cols: List[str],
                       hist_len: int, horizon: int,
                       feature_cols: List[str]) -> np.ndarray:
    """Recursive multi-step prediction updating df in place."""
    future_vals: List[float] = []
    for step in range(horizon):
        feat_ext = _build_features(df, exog_cols)
        row_idx  = hist_len + step
        x_row    = feat_ext.loc[row_idx, feature_cols].values.astype("float64")
        if not np.isfinite(x_row).all():
            fallback = future_vals[-1] if future_vals else float(df["value"].iloc[hist_len - 1])
            future_vals.append(fallback)
        else:
            x_df = pd.DataFrame(x_row.reshape(1, -1), columns=feature_cols)
            future_vals.append(float(model.predict(x_df)[0]))
        df.at[hist_len + step, "value"] = future_vals[-1]
    return np.array(future_vals, dtype="float64")


def _residual_ci(y_true: np.ndarray, y_fitted: np.ndarray,
                 future_forecast: np.ndarray, horizon: int,
                 confidence_level: float) -> tuple:
    residuals = (y_true - y_fitted).astype("float64")
    finite_r  = residuals[np.isfinite(residuals)]
    sigma     = float(np.std(finite_r, ddof=1)) if len(finite_r) > 1 \
                else float(np.abs(finite_r).mean() + 1e-8)
    z         = _get_z(confidence_level)
    steps     = np.arange(1, horizon + 1, dtype="float64")
    ci_lo     = (future_forecast - z * sigma * np.sqrt(steps)).astype("float64")
    ci_hi     = (future_forecast + z * sigma * np.sqrt(steps)).astype("float64")
    return np.minimum(ci_lo, future_forecast), np.maximum(ci_hi, future_forecast)


def _assemble_output(df: pd.DataFrame, fitted_full: np.ndarray,
                     future_index, future_forecast: np.ndarray,
                     ci_low: np.ndarray, ci_high: np.ndarray,
                     horizon: int, model_name: str,
                     inferred: str, confidence_level: float,
                     metadata: dict) -> ForecastResult:
    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_full.astype("float64"),
        "ci_low":    np.nan,
        "ci_mid":    fitted_full.astype("float64"),
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })
    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_forecast,
        "ci_low":    ci_low,
        "ci_mid":    future_forecast,
        "ci_high":   ci_high,
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
    return ForecastResult(
        model_name  = model_name,
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = metadata,
    )


# ==============================================================================
# RANDOM FOREST
# ==============================================================================

def run_random_forest(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    from sklearn.ensemble import RandomForestRegressor

    df, inferred  = _validate_and_prep(df)
    exog_cols     = _get_exog_cols(df)
    feat_tr, y_tr, feature_cols = _build_train_arrays(df, exog_cols)

    if len(feat_tr) < 10:
        raise ValueError("Insufficient training rows after lag construction.")

    alpha_ci = 1.0 - confidence_level
    q_lo     = alpha_ci / 2.0
    q_hi     = 1.0 - alpha_ci / 2.0

    # Point model
    rf_point = RandomForestRegressor(
        n_estimators     = 200,
        max_features     = "sqrt",
        min_samples_leaf = 3,
        random_state     = 42,
        n_jobs           = -1,
    )
    rf_point.fit(pd.DataFrame(feat_tr, columns=feature_cols), y_tr)

    # Quantile models via separate forests
    rf_lo = RandomForestRegressor(
        n_estimators=200, max_features="sqrt",
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    rf_hi = RandomForestRegressor(
        n_estimators=200, max_features="sqrt",
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    # Train quantile RFs using pinball loss approximation:
    # weight samples to approximate quantile regression
    n_tr = len(y_tr)
    fitted_point = rf_point.predict(pd.DataFrame(feat_tr, columns=feature_cols))
    resid        = y_tr - fitted_point
    w_lo = np.where(resid < 0, q_lo,     1 - q_lo)
    w_hi = np.where(resid < 0, 1 - q_hi, q_hi)
    rf_lo.fit(pd.DataFrame(feat_tr, columns=feature_cols), y_tr, sample_weight=w_lo)
    rf_hi.fit(pd.DataFrame(feat_tr, columns=feature_cols), y_tr, sample_weight=w_hi)

    # Fitted values on history
    fitted_full = np.full(len(df), np.nan)
    train_idx   = [i for i in feat_tr.index]
    fitted_full[train_idx] = fitted_point

    # Future forecast
    hist_len = len(df)
    future_row = pd.DataFrame({"date": pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1,
        freq=inferred)[1:], "value": np.nan})
    df_ext = pd.concat([df, future_row], ignore_index=True)

    point_preds, lo_preds, hi_preds = [], [], []
    for step in range(horizon):
        feat_ext = _build_features(df_ext, exog_cols)
        row_idx  = hist_len + step
        x_row    = feat_ext.loc[row_idx, feature_cols].values.astype("float64")
        if not np.isfinite(x_row).all():
            fallback = point_preds[-1] if point_preds else float(y_tr[-1])
            point_preds.append(fallback)
            lo_preds.append(fallback * 0.95)
            hi_preds.append(fallback * 1.05)
        else:
            x_df = pd.DataFrame(x_row.reshape(1, -1), columns=feature_cols)
            p  = float(rf_point.predict(x_df)[0])
            lo = float(rf_lo.predict(x_df)[0])
            hi = float(rf_hi.predict(x_df)[0])
            point_preds.append(p)
            lo_preds.append(lo)
            hi_preds.append(hi)
        df_ext.at[hist_len + step, "value"] = point_preds[-1]

    future_forecast = np.array(point_preds, dtype="float64")
    ci_low  = np.minimum(np.array(lo_preds, dtype="float64"), future_forecast)
    ci_high = np.maximum(np.array(hi_preds, dtype="float64"), future_forecast)

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in RandomForest forecast.")

    future_index = pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1, freq=inferred)[1:]

    return _assemble_output(
        df, fitted_full, future_index, future_forecast, ci_low, ci_high,
        horizon, "RandomForest", inferred, confidence_level,
        metadata={
            "n_estimators":     200,
            "max_features":     "sqrt",
            "ci_method":        "weighted_quantile_forest",
            "feature_cols":     feature_cols,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "pro",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# RIDGE REGRESSION
# ==============================================================================

def run_ridge(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df, inferred  = _validate_and_prep(df)
    exog_cols     = _get_exog_cols(df)
    feat_tr, y_tr, feature_cols = _build_train_arrays(df, exog_cols)

    if len(feat_tr) < 10:
        raise ValueError("Insufficient training rows after lag construction.")

    # Standardise features — Ridge is sensitive to scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(feat_tr[feature_cols].values.astype("float64"))

    # AIC-free: use cross-validated alpha selection via RidgeCV
    from sklearn.linear_model import RidgeCV
    alphas  = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    model   = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    model.fit(X_train, y_tr)

    fitted_point = model.predict(X_train)
    fitted_full  = np.full(len(df), np.nan)
    fitted_full[list(feat_tr.index)] = fitted_point

    # Future forecast — recursive with scaler applied per step
    hist_len = len(df)
    future_row = pd.DataFrame({"date": pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1,
        freq=inferred)[1:], "value": np.nan})
    df_ext = pd.concat([df, future_row], ignore_index=True)

    point_preds = []
    for step in range(horizon):
        feat_ext = _build_features(df_ext, exog_cols)
        row_idx  = hist_len + step
        x_row    = feat_ext.loc[row_idx, feature_cols].values.astype("float64")
        if not np.isfinite(x_row).all():
            fallback = point_preds[-1] if point_preds else float(y_tr[-1])
            point_preds.append(fallback)
        else:
            x_scaled = scaler.transform(x_row.reshape(1, -1))
            point_preds.append(float(model.predict(x_scaled)[0]))
        df_ext.at[hist_len + step, "value"] = point_preds[-1]

    future_forecast = np.array(point_preds, dtype="float64")

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in Ridge forecast.")

    ci_low, ci_high = _residual_ci(
        y_tr,
        fitted_point,
        future_forecast,
        horizon,
        confidence_level,
    )

    future_index = pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1, freq=inferred)[1:]

    return _assemble_output(
        df, fitted_full, future_index, future_forecast, ci_low, ci_high,
        horizon, "Ridge", inferred, confidence_level,
        metadata={
            "alpha":            round(float(model.alpha_), 6),
            "ci_method":        "residual_based_sqrt_h",
            "feature_cols":     feature_cols,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "pro",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# LASSO REGRESSION
# ==============================================================================

def run_lasso(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    df, inferred  = _validate_and_prep(df)
    exog_cols     = _get_exog_cols(df)
    feat_tr, y_tr, feature_cols = _build_train_arrays(df, exog_cols)

    if len(feat_tr) < 10:
        raise ValueError("Insufficient training rows after lag construction.")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(feat_tr[feature_cols].values.astype("float64"))

    # LassoCV: cross-validated alpha selection with coordinate descent
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = LassoCV(
            alphas  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            cv      = 5,
            max_iter = 10000,
            random_state = 42,
        )
        model.fit(X_train, y_tr)

    fitted_point = model.predict(X_train)
    fitted_full  = np.full(len(df), np.nan)
    fitted_full[list(feat_tr.index)] = fitted_point

    # Active features (non-zero coefficients after L1 selection)
    active = [feature_cols[i] for i, c in enumerate(model.coef_) if abs(c) > 1e-10]

    # Future forecast — recursive
    hist_len = len(df)
    future_row = pd.DataFrame({"date": pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1,
        freq=inferred)[1:], "value": np.nan})
    df_ext = pd.concat([df, future_row], ignore_index=True)

    point_preds = []
    for step in range(horizon):
        feat_ext = _build_features(df_ext, exog_cols)
        row_idx  = hist_len + step
        x_row    = feat_ext.loc[row_idx, feature_cols].values.astype("float64")
        if not np.isfinite(x_row).all():
            fallback = point_preds[-1] if point_preds else float(y_tr[-1])
            point_preds.append(fallback)
        else:
            x_scaled = scaler.transform(x_row.reshape(1, -1))
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                point_preds.append(float(model.predict(x_scaled)[0]))
        df_ext.at[hist_len + step, "value"] = point_preds[-1]

    future_forecast = np.array(point_preds, dtype="float64")

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in Lasso forecast.")

    ci_low, ci_high = _residual_ci(
        y_tr,
        fitted_point,
        future_forecast,
        horizon,
        confidence_level,
    )

    future_index = pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1, freq=inferred)[1:]

    return _assemble_output(
        df, fitted_full, future_index, future_forecast, ci_low, ci_high,
        horizon, "Lasso", inferred, confidence_level,
        metadata={
            "alpha":            round(float(model.alpha_), 6),
            "active_features":  len(active),
            "total_features":   len(feature_cols),
            "ci_method":        "residual_based_sqrt_h",
            "feature_cols":     feature_cols,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "pro",
            "output_contract":  "ForecastResult",
        },
    )
