# ==================================================
# FILE: foresight_engine/models/garch_model.py
# VERSION: 2.0.0
# MODEL: GARCH — GENERALIZED AUTOREGRESSIVE
#        CONDITIONAL HETEROSKEDASTICITY
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# STATUS: VEDUTA ENGINE — PHASE 3C
# ==================================================
#
# PURPOSE:
#   GARCH models time-varying volatility. It does not
#   produce a level forecast — it produces a volatility
#   forecast that is used to scale confidence interval
#   width on high-variance series.
#
#   In the ensemble, GARCH serves as a CI width modifier.
#   On calm periods it narrows intervals. On volatile
#   periods it widens them. This is more honest than
#   fixed-width sigma * sqrt(h) intervals.
#
# TWO OUTPUTS:
#   1. Level forecast: mean equation (AR(1) + GARCH mean)
#      Used as a standalone forecast in the ensemble.
#   2. Volatility forecast: conditional sigma per horizon
#      Stored in metadata for ensemble CI upgrade (Phase 3D).
#
# MODEL SPECIFICATION:
#   Mean equation:     AR(1) — captures serial dependence
#   Variance equation: GARCH(1,1) — industry standard
#   Distribution:      Normal (SkewStudent optional in Phase 5)
#
# CI METHOD:
#   GARCH conditional volatility * z_score.
#   Width varies by period — tighter in calm, wider in
#   volatile regimes. This is the correct CI for GARCH.
#
# IMPLEMENTATION:
#   arch library (Kevin Sheppard). Production standard.
#   arch_model with mean='ARX', vol='GARCH', p=1, q=1.
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

from foresight_engine.models.contracts import ForecastResult

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

GARCH_P   = 1
GARCH_Q   = 1
AR_LAGS   = 1
RESCALE   = 100.0    # Scale series for GARCH numerical stability

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


# ==================================================
# MODEL RUNNER
# ==================================================

def run_garch(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("GARCH requires 'date' and 'value' columns.")

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

    y_raw = df["value"].astype("float64").values

    if not np.isfinite(y_raw).all():
        raise ValueError("Non-finite values detected in series.")

    if len(df) < 20:
        raise ValueError("Minimum 20 observations required for GARCH.")

    # --------------------------------------------------
    # RESCALE FOR NUMERICAL STABILITY
    # --------------------------------------------------

    y_mean  = float(np.mean(y_raw))
    y_std   = float(np.std(y_raw, ddof=1))
    y_scale = y_std if y_std > 0 else 1.0

    y_scaled = (y_raw - y_mean) / y_scale * RESCALE

    # --------------------------------------------------
    # FIT GARCH(1,1) WITH AR(1) MEAN
    # --------------------------------------------------

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am     = arch_model(
                y_scaled,
                mean  = "ARX",
                lags  = AR_LAGS,
                vol   = "GARCH",
                p     = GARCH_P,
                q     = GARCH_Q,
                dist  = "normal",
                rescale = False,
            )
            result = am.fit(
                disp        = "off",
                show_warning= False,
                options     = {"maxiter": 500},
            )
    except Exception as e:
        raise RuntimeError(f"GARCH model fit failed: {e}") from e

    # --------------------------------------------------
    # VALIDATE FIT
    # --------------------------------------------------

    if not np.isfinite(result.params).all():
        raise RuntimeError("GARCH fit produced non-finite parameters.")

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    cond_mean_scaled = result.conditional_volatility   # volatility, not mean
    # Fitted level = residuals reversed
    fitted_scaled    = y_scaled - result.resid
    fitted_raw       = fitted_scaled / RESCALE * y_scale + y_mean

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_raw,
        "ci_low":    np.nan,
        "ci_mid":    fitted_raw,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST — LEVEL + VOLATILITY
    # --------------------------------------------------

    alpha_ci = 1.0 - confidence_level
    z        = _get_z(confidence_level)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecasts = result.forecast(
                horizon   = horizon,
                reindex   = False,
            )

        # Mean forecasts — shape (1, horizon)
        mean_fc_scaled = forecasts.mean.values[-1, :horizon].astype("float64")
        # Variance forecasts — shape (1, horizon)
        var_fc_scaled  = forecasts.variance.values[-1, :horizon].astype("float64")

        if not np.isfinite(mean_fc_scaled).all():
            raise ValueError("Non-finite mean forecast.")
        if not np.isfinite(var_fc_scaled).all() or (var_fc_scaled < 0).any():
            raise ValueError("Invalid variance forecast.")

        # Conditional volatility (sigma) per horizon step
        sigma_fc_scaled = np.sqrt(var_fc_scaled)

        # Rescale back to original units
        mean_fc_raw   = mean_fc_scaled  / RESCALE * y_scale + y_mean
        sigma_fc_raw  = sigma_fc_scaled / RESCALE * y_scale

        ci_low  = (mean_fc_raw - z * sigma_fc_raw).astype("float64")
        ci_high = (mean_fc_raw + z * sigma_fc_raw).astype("float64")
        ci_method = "garch_conditional_volatility"

    except Exception:
        # Fallback: use last fitted residual std as constant sigma
        ci_method = "residual_based_fallback"
        resid_raw = result.resid / RESCALE * y_scale
        sigma_const = float(np.std(resid_raw[np.isfinite(resid_raw)], ddof=1))

        # Simple AR(1) mean forecast: last value * AR coefficient
        ar_coef = float(result.params.get("y[1]",
                    result.params.iloc[1] if len(result.params) > 1 else 0.0))
        ar_coef = np.clip(ar_coef, -0.99, 0.99)

        last_val     = float(y_raw[-1])
        mean_fc_raw  = np.array([
            y_mean + ar_coef ** h * (last_val - y_mean)
            for h in range(1, horizon + 1)
        ], dtype="float64")

        h_arr   = np.arange(1, horizon + 1, dtype="float64")
        ci_low  = (mean_fc_raw - z * sigma_const * np.sqrt(h_arr)).astype("float64")
        ci_high = (mean_fc_raw + z * sigma_const * np.sqrt(h_arr)).astype("float64")
        sigma_fc_raw = np.full(horizon, sigma_const, dtype="float64")

    # Ensure CI brackets forecast
    ci_low  = np.minimum(ci_low,  mean_fc_raw)
    ci_high = np.maximum(ci_high, mean_fc_raw)

    if not np.isfinite(mean_fc_raw).all():
        raise RuntimeError("Non-finite values in final forecast.")

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
        "forecast":  mean_fc_raw.astype("float64"),
        "ci_low":    ci_low.astype("float64"),
        "ci_mid":    mean_fc_raw.astype("float64"),
        "ci_high":   ci_high.astype("float64"),
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
        raise RuntimeError("NaN CI in future forecast output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in future forecast output.")

    return ForecastResult(
        model_name  = "GARCH",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "garch_p":              GARCH_P,
            "garch_q":              GARCH_Q,
            "ar_lags":              AR_LAGS,
            "ci_method":            ci_method,
            "volatility_forecast":  sigma_fc_raw.tolist(),
            "aic":                  round(float(result.aic), 4),
            "bic":                  round(float(result.bic), 4),
            "frequency":            inferred,
            "confidence_level":     confidence_level,
            "ensemble_note":        "volatility_forecast available for CI width scaling in Phase 3D",
            "min_tier":             "enterprise",
            "output_contract":           "ForecastResult",
        },
    )
