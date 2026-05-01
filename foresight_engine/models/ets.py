# ==================================================
# FILE: foresight_engine/models/ets.py
# VERSION: 3.0.0
# MODEL: ETS — AUTOMATIC MODEL SELECTION
# ENGINE: Foresight Engine v3.0.0
# UPDATED: M1 — Full automatic ETS model selection via AIC grid search
# ==================================================
#
# M1 UPGRADE — AUTOMATIC ETS MODEL SELECTION:
#   Previous: Fixed trend="add", seasonal=None, alpha=0.3, optimized=False
#   Fixed: AIC grid search across up to 9 ETS specifications per series
#     error:    additive (A) only
#     trend:    None (N), additive (A), additive damped (Ad)
#     seasonal: None (N), additive (A), multiplicative (M) — positive series only
#   Minimum 2 seasonal cycles for seasonal candidates.
#   Best by AIC. Three-tier fallback. Never crashes.
# ==================================================

from __future__ import annotations

import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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



_Z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

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


def _residual_ci(
    forecast_values:  np.ndarray,
    residuals:        np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    z     = _get_z(confidence_level)
    sigma = float(np.std(residuals[np.isfinite(residuals)], ddof=1))
    if sigma < 1e-10:
        sigma = float(np.mean(np.abs(residuals[np.isfinite(residuals)])))
    h      = np.arange(1, len(forecast_values) + 1, dtype="float64")
    spread = z * sigma * np.sqrt(h)
    return (forecast_values - spread).astype("float64"), \
           (forecast_values + spread).astype("float64")


def _build_candidate_grid(y: np.ndarray, season_len: int, has_seasonality: bool) -> list:
    all_positive = bool((y > 0).all())
    candidates   = []
    trend_specs  = [
        {"trend": None,  "damped_trend": False, "trend_label": "N"},
        {"trend": "add", "damped_trend": False, "trend_label": "A"},
        {"trend": "add", "damped_trend": True,  "trend_label": "Ad"},
    ]
    seasonal_specs = [{"seasonal": None, "seasonal_periods": None, "seas_label": "N"}]
    if has_seasonality:
        seasonal_specs.append({"seasonal": "add", "seasonal_periods": season_len, "seas_label": "A"})
        if all_positive:
            seasonal_specs.append({"seasonal": "mul", "seasonal_periods": season_len, "seas_label": "M"})
    for ts, ss in itertools.product(trend_specs, seasonal_specs):
        if ss["seasonal"] == "mul" and ts["trend"] is None and len(y) < 3 * season_len:
            continue
        candidates.append({
            "trend":            ts["trend"],
            "damped_trend":     ts["damped_trend"],
            "seasonal":         ss["seasonal"],
            "seasonal_periods": ss["seasonal_periods"],
            "label":            f"ETS(A,{ts['trend_label']},{ss['seas_label']})"
                                + ("d" if ts["damped_trend"] else ""),
        })
    return candidates


def _fit_candidate(y: np.ndarray, spec: dict) -> tuple:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend              = spec["trend"],
                damped_trend       = spec["damped_trend"],
                seasonal           = spec["seasonal"],
                seasonal_periods   = spec["seasonal_periods"],
                initialization_method = "estimated",
            )
            fitted = model.fit(optimized=True, remove_bias=False)
        aic = fitted.aic
        if not np.isfinite(aic): return None, np.inf
        if not np.isfinite(np.asarray(fitted.fittedvalues)).all(): return None, np.inf
        if not np.isfinite(fitted.fittedvalues).all(): return None, np.inf
        return fitted, float(aic)
    except Exception:
        return None, np.inf


def _select_best_ets(y: np.ndarray, season_len: int, has_seasonality: bool) -> tuple:
    candidates  = _build_candidate_grid(y, season_len, has_seasonality)
    best_fitted = None
    best_label  = "ETS(A,A,N)"
    best_aic    = np.inf
    for spec in candidates:
        fitted, aic = _fit_candidate(y, spec)
        if fitted is not None and aic < best_aic:
            best_fitted, best_label, best_aic = fitted, spec["label"], aic
    # Fallback 1
    if best_fitted is None:
        fb = {"trend": "add", "damped_trend": False, "seasonal": None, "seasonal_periods": None, "label": "ETS(A,A,N)_fallback"}
        best_fitted, best_aic = _fit_candidate(y, fb)
        best_label = fb["label"]
    # Fallback 2
    if best_fitted is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
                best_fitted = m.fit(optimized=False, smoothing_level=0.3, smoothing_trend=0.1)
                best_label  = "ETS(A,A,N)_fixed_fallback"
                best_aic    = float("inf")
        except Exception as e:
            raise RuntimeError(f"All ETS candidates failed: {e}") from e
    return best_fitted, best_label, best_aic


def run_ets(df: pd.DataFrame, horizon: int, confidence_level: float) -> ForecastResult:
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("ETS requires 'date' and 'value' columns.")
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
    y = df["value"].astype("float64")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected.")
    if len(y) < 6:
        raise ValueError("Minimum 6 observations required.")

    season_len      = 12
    has_seasonality = len(y) >= 2 * season_len

    best_fitted, selected_model, selected_aic = _select_best_ets(
        y.values, season_len, has_seasonality
    )

    _fv = best_fitted.fittedvalues
    hist_fitted = pd.Series(np.asarray(_fv).astype("float64"), index=df.index)
    if np.isnan(hist_fitted.values).any():
        raise RuntimeError("NaN in fitted values.")
    residuals        = (y.values - np.asarray(hist_fitted).astype("float64")).astype("float64")
    finite_residuals = residuals[np.isfinite(residuals)]
    if len(finite_residuals) < 2:
        raise RuntimeError("Insufficient finite residuals.")

    hist_block = pd.DataFrame({
        "date":      hist_fitted.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    future_index = pd.date_range(start=hist_fitted.index[-1], periods=horizon + 1, freq=inferred)[1:]
    if not future_index.min() > hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")
    future_forecast = best_fitted.forecast(horizon).astype("float64")
    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values.")

    ci_low, ci_high = _residual_ci(np.asarray(future_forecast), finite_residuals, confidence_level)

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  np.asarray(future_forecast).astype('float64'),
        "ci_low":    ci_low,
        "ci_mid":    np.asarray(future_forecast).astype('float64'),
        "ci_high":   ci_high,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI values in output.")
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI detected.")

    return ForecastResult(
        model_name  = "ETS",
        forecast_df = forecast_df[["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]],
        metrics     = None,
        metadata    = {
            "selected_model":   selected_model,
            "selected_aic":     round(selected_aic, 4) if np.isfinite(selected_aic) else None,
            "selection_method": "AIC grid search",
            "season_len":       season_len,
            "has_seasonality":  has_seasonality,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "residual_based_sigma_sqrt_h",
            "output_contract":  "ForecastResult",
        },
    )
