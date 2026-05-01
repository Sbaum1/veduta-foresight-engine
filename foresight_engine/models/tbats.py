# ==================================================
# FILE: foresight_engine/models/tbats.py
# VERSION: 2.0.0
# MODEL: TBATS (State-Space Seasonal)
# ENGINE: Foresight Engine v3.0.0
# STATUS: VEDUTA ENGINE — REAL CI — 3B-2
# ==================================================
#
# 3B-2 CI FIX:
#   Previous: ci_low = np.nan, ci_high = np.nan (hard-coded)
#   Fixed:    Native TBATS library confidence intervals
#
#   Method: fitted_model.forecast(steps, confidence_level)
#     Returns (forecast_array, confidence_interval_array)
#     confidence_interval_array shape: (steps, 2)
#     col 0 = lower bound, col 1 = upper bound
#
#   The tbats library supports confidence_level as a float
#   in [0, 1]. We pass confidence_level directly.
#   Falls back to residual-based CI if library CI is
#   unavailable or non-finite.
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
# sklearn 1.6 compatibility: patch before tbats import
def _patch_sklearn_for_tbats():
    try:
        import sklearn.utils.validation as _skval
        if getattr(_skval, '_tbats_patched', False):
            return
        _orig = _skval.check_array
        def _safe_check_array(*args, **kwargs):
            kwargs.pop('force_all_finite', None)
            return _orig(*args, **kwargs)
        _skval.check_array = _safe_check_array
        _skval._tbats_patched = True
    except Exception:
        pass

_patch_sklearn_for_tbats()
from tbats import TBATS

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




# --------------------------------------------------
# Z-SCORE MAP (fallback CI)
# --------------------------------------------------

Z_SCORES = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def _z_score(confidence_level: float) -> float:
    return min(Z_SCORES.items(), key=lambda kv: abs(kv[0] - confidence_level))[1]


def _residual_ci(
    forecast_values:  np.ndarray,
    residuals:        np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    z      = _z_score(confidence_level)
    sigma  = float(np.std(residuals, ddof=1))
    h      = np.arange(1, len(forecast_values) + 1, dtype="float64")
    spread = z * sigma * np.sqrt(h)
    return forecast_values - spread, forecast_values + spread


# ==================================================
# MODEL RUNNER
# ==================================================

def run_tbats(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("TBATS requires 'date' and 'value' columns.")

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

    seasonal_period = 12

    if len(y) < 2 * seasonal_period:
        raise ValueError("Minimum 2 full seasonal cycles required (>= 24 observations).")

    # --------------------------------------------------
    # MODEL FIT
    # --------------------------------------------------

    # sklearn 1.6 removed force_all_finite — patch tbats to handle this
    try:
        import sklearn.utils.validation as _skval
        if not hasattr(_skval, '_check_array_orig'):
            _orig_check = _skval.check_array
            def _patched_check_array(*args, **kwargs):
                kwargs.pop('force_all_finite', None)
                return _orig_check(*args, **kwargs)
            _skval.check_array = _patched_check_array
            _skval._check_array_orig = _orig_check
    except Exception:
        pass

    estimator = TBATS(
        seasonal_periods = [float(seasonal_period)],
        use_box_cox      = False,
        use_arma_errors  = False,
        n_jobs           = 1,
    )

    fitted_model = estimator.fit(y.values)

    hist_fitted = pd.Series(
        fitted_model.y_hat,
        index = y.index,
        dtype = "float64",
    )

    if np.isnan(np.asarray(hist_fitted)).any():
        raise RuntimeError("NaN in fitted values.")

    # ── Residuals for fallback CI ────────────────────────────────────────────
    residuals = (y.values - hist_fitted.values).astype("float64")

    hist_block = pd.DataFrame(
        {
            "date":      hist_fitted.index,
            "actual":    np.nan,
            "forecast":  hist_fitted.values,
            "ci_low":    np.nan,
            "ci_mid":    hist_fitted.values,
            "ci_high":   np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST WITH NATIVE CI (3B-2)
    # --------------------------------------------------

    future_index = pd.date_range(
        start   = hist_fitted.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    if not future_index.min() > hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    # ── Attempt native TBATS confidence intervals ────────────────────────────
    ci_method = "tbats_native"
    try:
        future_forecast_raw, conf_int = fitted_model.forecast(
            steps            = horizon,
            confidence_level = confidence_level,
        )
        future_forecast = np.asarray(future_forecast_raw, dtype="float64")
        ci_low          = np.asarray(conf_int[:, 0],      dtype="float64")
        ci_high         = np.asarray(conf_int[:, 1],      dtype="float64")

        # Validate native CI
        if (
            not np.isfinite(ci_low).all()
            or not np.isfinite(ci_high).all()
            or (ci_high <= ci_low).any()
        ):
            raise ValueError("Native CI invalid — falling back to residual CI.")

    except Exception:
        # ── Fallback: residual-based CI ──────────────────────────────────────
        future_forecast_raw = np.asarray(
            fitted_model.forecast(steps=horizon),
            dtype="float64",
        )
        future_forecast = future_forecast_raw
        ci_low, ci_high = _residual_ci(
            forecast_values  = future_forecast,
            residuals        = residuals,
            confidence_level = confidence_level,
        )
        ci_method = "residual_based_fallback"

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values detected.")

    future_block = pd.DataFrame(
        {
            "date":      future_index,
            "actual":    np.nan,
            "forecast":  future_forecast,
            "ci_low":    ci_low,
            "ci_mid":    future_forecast,
            "ci_high":   ci_high,
            "error_pct": np.nan,
        }
    )

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
        model_name  = "TBATS",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "seasonal_period":  seasonal_period,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "box_cox":          False,
            "arma_errors":      False,
            "ci_method":        ci_method,
            "output_contract":       "ForecastResult",
        },
    )
