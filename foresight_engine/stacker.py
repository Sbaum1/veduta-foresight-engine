# ==================================================
# FILE: foresight_engine/stacker.py
# VERSION: 2.0.0
# ROLE: RIDGE REGRESSION FORECAST STACKER
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# v3.0.0 FIX:
#   _extract_fold_forecasts() early return bug fixed.
#   Was: return None, None  (2 values)
#   Fixed: return None, None, []  (3 values — matches unpacking)
# ==================================================

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .contracts import ForecastResult, ENGINE_VERSION

RIDGE_ALPHAS        = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_BASE_MODELS     = 2
MIN_FOLD_ROWS       = 6
STACKER_VERSION     = "ridge_cv_v2"

STACKER_SKIP_MODELS = {"X-13", "VAR", "Croston_SBA", "Primary Ensemble", "Stacked Ensemble"}


def _extract_fold_forecasts(
    results: Dict[str, Any],
    horizon: int,
) -> tuple:
    """
    Extract out-of-fold forecasts from backtest results.

    Returns:
        (X, y, base_model_names) — all three values always returned.
        (None, None, []) if insufficient data.
    """
    model_cols: Dict[str, np.ndarray] = {}
    actuals_arr: Optional[np.ndarray] = None

    for name, result in results.items():
        if name.startswith("_") or name in STACKER_SKIP_MODELS:
            continue
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        if result.get("diagnostic_only"):
            continue

        fdf = result.get("forecast_df")
        if fdf is None or fdf.empty:
            continue

        hist = fdf[fdf["actual"].notna()].copy()
        if len(hist) < horizon:
            continue

        fold_slice = hist.tail(horizon)
        forecasts  = fold_slice["forecast"].values.astype(float)
        actuals    = fold_slice["actual"].values.astype(float)

        if not np.isfinite(forecasts).all():
            continue
        if not np.isfinite(actuals).all():
            continue

        model_cols[name] = forecasts

        if actuals_arr is None:
            actuals_arr = actuals
        else:
            if not np.allclose(actuals_arr, actuals, rtol=1e-3):
                continue

    # BUG FIX v3.0.0: always return 3 values
    if len(model_cols) < MIN_BASE_MODELS or actuals_arr is None:
        return None, None, []

    X = np.column_stack(list(model_cols.values()))
    y = actuals_arr

    return X, y, list(model_cols.keys())


def _train_ridge_stacker(X: np.ndarray, y: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    ridge  = RidgeCV(
        alphas              = RIDGE_ALPHAS,
        fit_intercept       = True,
        scoring             = "neg_mean_absolute_error",
        cv                  = min(5, len(y)),
    )
    ridge.fit(X_sc, y)
    return ridge, scaler


def build_stacked_forecast(
    results:          Dict[str, Any],
    horizon:          int,
    confidence_level: float,
    df_historical:    pd.DataFrame,
) -> ForecastResult:
    """
    Build a stacked ensemble forecast using Ridge meta-learner.
    Falls back to Primary Ensemble on any failure.
    """
    fallback_result = results.get("Primary Ensemble")

    def _fallback(reason: str) -> ForecastResult:
        if fallback_result and fallback_result.get("status") == "success":
            fdf  = fallback_result["forecast_df"].copy()
            meta = dict(fallback_result.get("metadata", {}))
            meta.update({
                "stacker_active":   False,
                "stacker_fallback": reason,
                "stacker_version":  STACKER_VERSION,
            })
            return ForecastResult(
                model_name  = "Stacked Ensemble",
                forecast_df = fdf,
                metrics     = fallback_result.get("metrics"),
                metadata    = meta,
            )
        raise RuntimeError(f"Stacker fallback failed — Primary Ensemble also unavailable. Reason: {reason}")

    try:
        extracted = _extract_fold_forecasts(results, horizon)
        # Always 3 values now — no crash risk
        X, y, base_model_names = extracted
        if X is None:
            return _fallback("Insufficient fold data for meta-learner training")
    except Exception as e:
        return _fallback(f"Fold extraction error: {e}")

    if len(y) < MIN_FOLD_ROWS:
        return _fallback(f"Too few fold rows ({len(y)} < {MIN_FOLD_ROWS})")

    try:
        ridge, scaler = _train_ridge_stacker(X, y)
    except Exception as e:
        return _fallback(f"Ridge training error: {e}")

    future_cols:  Dict[str, np.ndarray] = {}
    future_dates: Optional[np.ndarray]  = None

    for name in base_model_names:
        result = results.get(name, {})
        if result.get("status") != "success":
            continue
        fdf    = result.get("forecast_df")
        if fdf is None:
            continue
        future = fdf[fdf["actual"].isna()].head(horizon)
        if len(future) < horizon:
            continue
        vals = future["forecast"].values.astype(float)
        if not np.isfinite(vals).all():
            continue
        future_cols[name] = vals
        if future_dates is None:
            future_dates = future["date"].values

    if len(future_cols) < MIN_BASE_MODELS or future_dates is None:
        return _fallback("Insufficient base model future forecasts for stacking")

    aligned = []
    aligned_names = []
    for name in base_model_names:
        if name in future_cols:
            aligned.append(future_cols[name])
            aligned_names.append(name)

    if len(aligned) < MIN_BASE_MODELS:
        return _fallback("Column alignment produced insufficient models")

    X_future = np.column_stack(aligned)

    try:
        X_future_sc      = scaler.transform(X_future)
        stacked_forecast = ridge.predict(X_future_sc)
    except Exception as e:
        return _fallback(f"Ridge prediction error: {e}")

    if not np.isfinite(stacked_forecast).all():
        return _fallback("Non-finite values in stacked forecast")

    base_matrix = X_future
    spread_half = np.std(base_matrix, axis=1) * 1.96
    ci_low      = stacked_forecast - spread_half
    ci_high     = stacked_forecast + spread_half

    pe_fdf     = fallback_result["forecast_df"].copy() if fallback_result else None
    hist_block = pe_fdf[pe_fdf["actual"].notna()].copy() if pe_fdf is not None else pd.DataFrame()

    future_block = pd.DataFrame({
        "date":      future_dates,
        "actual":    pd.NA,
        "forecast":  stacked_forecast.astype("float64"),
        "ci_low":    ci_low.astype("float64"),
        "ci_mid":    stacked_forecast.astype("float64"),
        "ci_high":   ci_high.astype("float64"),
        "error_pct": pd.NA,
    })

    if not hist_block.empty:
        forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    else:
        forecast_df = future_block

    coefficients = dict(zip(aligned_names, ridge.coef_.tolist()))

    metadata = {
        "engine_version":     ENGINE_VERSION,
        "stacker_active":     True,
        "stacker_version":    STACKER_VERSION,
        "base_models":        aligned_names,
        "n_base_models":      len(aligned_names),
        "ridge_alpha":        float(ridge.alpha_),
        "ridge_coefficients": {k: round(v, 6) for k, v in coefficients.items()},
        "training_rows":      int(len(y)),
        "confidence_level":   confidence_level,
        "ci_method":          "base_model_spread",
    }

    return ForecastResult(
        model_name  = "Stacked Ensemble",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = metadata,
    )
