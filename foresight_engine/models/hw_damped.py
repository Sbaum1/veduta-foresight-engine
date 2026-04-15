# ==================================================
# FILE: foresight_engine/models/hw_damped.py
# VERSION: 3.0.0
# MODEL: HOLT-WINTERS DAMPED — AIC SEASONAL MODE SELECTION
# ENGINE: Foresight Engine v3.0.0
# UPDATED: M1 — Additive vs multiplicative seasonal AIC selection
# ==================================================
#
# M1 UPGRADE — SEASONAL MODE AIC SELECTION:
#
#   Previous (v2.0.0):
#     Always used additive seasonal ("add") when seasonality
#     was present. Multiplicative seasonality — where amplitude
#     scales proportionally with level — was never tried.
#     On M3 series where seasonal swings grow with the level
#     (common in economic and supply chain data), additive
#     seasonal consistently under-fits. MASE penalty follows.
#
#   Fixed (v3.0.0):
#     When has_seasonality=True, fit both:
#       - ETS(A, Ad, A) — additive seasonal + damped trend
#       - ETS(A, Ad, M) — multiplicative seasonal + damped trend
#                         (all-positive series only)
#     Select by AIC. Lower AIC wins.
#     Non-seasonal and short-series behaviour unchanged.
#
# GOVERNANCE:
#   - selected_seasonal_mode and selected_aic logged in metadata
#   - Output contract: ForecastResult unchanged
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from foresight_engine.models.contracts import ForecastResult

_Z = {0.50:0.674, 0.80:1.282, 0.90:1.645, 0.95:1.960, 0.99:2.576}

def _get_z(cl):
    if cl in _Z: return _Z[cl]
    levels = sorted(_Z.keys())
    for i in range(len(levels)-1):
        lo, hi = levels[i], levels[i+1]
        if lo <= cl <= hi:
            t=(cl-lo)/(hi-lo); return _Z[lo]+t*(_Z[hi]-_Z[lo])
    return 1.960

_season_map = {
    "MS":12,"M":12,"QS":4,"Q":4,
    "W":52,"W-SUN":52,"W-MON":52,"A":1,"AS":1,"D":7,
}


def _fit_hw(y, trend, damped, seasonal, season_len):
    """Fit one HW variant. Returns (fitted, aic) or (None, inf)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                y,
                trend=trend,
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=season_len if seasonal else None,
                initialization_method="estimated",
            )
            f = m.fit(optimized=True, remove_bias=True)
        if not np.isfinite(f.aic): return None, np.inf
        if not np.isfinite(np.asarray(f.fittedvalues)).all(): return None, np.inf
        return f, float(f.aic)
    except Exception:
        return None, np.inf


# ==================================================
# MODEL RUNNER
# ==================================================

def run_hw_damped(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("HW Damped requires 'date' and 'value' columns.")

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
    if len(df) < 12:
        raise ValueError("Minimum 12 observations required.")

    season_len   = _season_map.get(inferred, 12)
    has_seasonal = len(y) >= 2 * season_len and season_len > 1
    all_positive = bool((y > 0).all())

    # --------------------------------------------------
    # M1: AIC SELECTION OVER SEASONAL MODE
    # --------------------------------------------------

    best_fitted       = None
    best_aic          = np.inf
    selected_seasonal = None

    if has_seasonal:
        # Try additive seasonal
        f_add, aic_add = _fit_hw(y, "add", True, "add", season_len)
        if f_add is not None and aic_add < best_aic:
            best_fitted, best_aic, selected_seasonal = f_add, aic_add, "add"

        # Try multiplicative seasonal (positive series only)
        if all_positive:
            f_mul, aic_mul = _fit_hw(y, "add", True, "mul", season_len)
            if f_mul is not None and aic_mul < best_aic:
                best_fitted, best_aic, selected_seasonal = f_mul, aic_mul, "mul"

    # No seasonality (or seasonal fits failed)
    if best_fitted is None:
        best_fitted, best_aic = _fit_hw(y, "add", True, None, season_len)
        selected_seasonal = "none"

    # Hard fallback — optimized=False
    if best_fitted is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ExponentialSmoothing(
                    y, trend="add", damped_trend=True, seasonal=None,
                    initialization_method="estimated",
                )
                best_fitted = m.fit(optimized=True)
                best_aic    = float("inf")
                selected_seasonal = "none_fallback"
        except Exception as e:
            raise RuntimeError(f"HW Damped all fits failed: {e}") from e

    # --------------------------------------------------
    # RESIDUALS AND CI PARAMS
    # --------------------------------------------------

    fitted_values = best_fitted.fittedvalues.astype("float64")
    residuals     = best_fitted.resid
    finite_resid  = residuals[np.isfinite(residuals)]
    if len(finite_resid) < 2:
        raise RuntimeError("Insufficient residuals for CI computation.")

    sigma = float(np.std(finite_resid, ddof=1))
    z     = _get_z(confidence_level)

    hist_block = pd.DataFrame({
        "date": df["date"].values, "actual": np.nan,
        "forecast": fitted_values, "ci_low": np.nan,
        "ci_mid": fitted_values, "ci_high": np.nan, "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_forecast = best_fitted.forecast(horizon).astype("float64")
    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values detected.")

    last_date  = df["date"].iloc[-1]
    future_idx = pd.date_range(start=last_date, periods=horizon+1,
                               freq=inferred if inferred else "MS")[1:]
    if len(future_idx) != horizon:
        raise RuntimeError(f"Future date index length mismatch.")

    h_arr   = np.arange(1, horizon+1, dtype="float64")
    ci_low  = future_forecast - z * sigma * np.sqrt(h_arr)
    ci_high = future_forecast + z * sigma * np.sqrt(h_arr)

    future_block = pd.DataFrame({
        "date": future_idx, "actual": np.nan,
        "forecast": future_forecast,
        "ci_low": ci_low.astype("float64"),
        "ci_mid": future_forecast,
        "ci_high": ci_high.astype("float64"),
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast","ci_low","ci_mid","ci_high"]] = \
            b[["forecast","ci_low","ci_mid","ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low","ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI values in future forecast output.")
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI detected.")

    phi = best_fitted.params.get("damping_trend", np.nan) \
          if hasattr(best_fitted, "params") else np.nan

    return ForecastResult(
        model_name  = "HW_Damped",
        forecast_df = forecast_df[
            ["date","actual","forecast","ci_low","ci_mid","ci_high","error_pct"]
        ],
        metrics  = None,
        metadata = {
            "trend":                  "additive_damped",
            "selected_seasonal_mode": selected_seasonal,
            "selected_aic":           round(best_aic, 4) if np.isfinite(best_aic) else None,
            "selection_method":       "aic" if has_seasonal else "no_seasonal",
            "seasonal_periods":       season_len if selected_seasonal not in ("none","none_fallback") else None,
            "damped_trend":           True,
            "phi":                    float(phi) if np.isfinite(phi) else None,
            "ci_method":              "residual_based_sigma_sqrt_h",
            "sigma":                  round(sigma, 6),
            "frequency":              inferred,
            "confidence_level":       confidence_level,
            "output_contract":        "ForecastResult",
        },
    )
