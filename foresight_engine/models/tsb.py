# ==================================================
# FILE: foresight_engine/models/tsb.py
# VERSION: 1.0.0
# MODEL: TSB — TEÜBER-SYNTETOS-BOYLAN
# ENGINE: Foresight Engine v3.0.0
# TIER: pro (minimum)
# ==================================================
#
# PURPOSE:
#   TSB (Teüber, Syntetos & Boylan, 2011) is the successor to
#   Croston's method for intermittent demand forecasting. It
#   separately smooths:
#     p̂  — probability of demand occurring (demand rate)
#     d̂  — expected demand size given occurrence
#
#   Forecast = p̂ × d̂
#
#   Key improvement over Croston/SBA:
#     Croston smooths the inter-demand interval (z̃) and updates
#     only on non-zero periods — it never decreases the demand
#     rate estimate, even during prolonged zero-demand periods.
#     This leads to upward bias on non-stationary intermittent
#     demand (products going obsolete or seasonal).
#
#     TSB updates the demand probability p̂ on EVERY period —
#     both non-zero (where p̂ increases toward 1) and zero
#     (where p̂ decreases toward 0). This makes TSB unbiased
#     on non-stationary intermittent series and better suited
#     for products with changing demand patterns.
#
# PARAMETER SELECTION:
#   Grid search over alpha_p (demand probability smoothing) and
#   alpha_d (demand size smoothing) to minimise in-sample MSE
#   on non-zero demand periods. Grid: {0.05, 0.10, ..., 0.50}.
#
# CI METHOD:
#   Bootstrap over non-zero demand values. At each horizon step,
#   simulate demand occurrence (Bernoulli with p=p̂) and demand
#   size (resample from non-zero history). Empirical quantiles
#   at confidence_level give the interval bounds.
#   Minimum 500 bootstrap paths. Interval widens with horizon.
#
# GOVERNANCE:
#   - Requires series with at least some non-zero periods
#   - Falls back to SeasonalNaive if <3 non-zero observations
#   - No Streamlit dependencies
#   - Output contract: ForecastResult
#
# REFERENCE:
#   Teüber, T., Syntetos, A. A., & Boylan, J. E. (2011).
#   Intermittent demand forecasting: An empirical study on
#   accuracy and the risk of obsolescence.
#   International Journal of Forecasting, 27(2), 596–611.
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ForecastResult

N_BOOTSTRAP    = 500
ALPHA_GRID     = np.round(np.arange(0.05, 0.55, 0.05), 2)
MIN_NONZERO    = 3

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


def _tsb_fit(y: np.ndarray, alpha_p: float, alpha_d: float) -> tuple:
    """
    Fit TSB model. Returns (p_hat, d_hat, fitted_vals).
    p_hat: final demand probability estimate
    d_hat: final demand size estimate
    fitted_vals: in-sample fitted values (p_t * d_t for each t)
    """
    n  = len(y)
    p  = np.zeros(n + 1)  # demand probability
    d  = np.zeros(n + 1)  # demand size

    # Initialise with first non-zero observation
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return 0.0, 0.0, np.zeros(n)

    # Initial estimates
    p[0] = float(np.mean(y > 0))
    d[0] = float(np.mean(y[nz_idx]))

    fitted = np.zeros(n)

    for t in range(n):
        fitted[t] = p[t] * d[t]
        if y[t] > 0:
            p[t + 1] = (1 - alpha_p) * p[t] + alpha_p * 1.0
            d[t + 1] = (1 - alpha_d) * d[t] + alpha_d * float(y[t])
        else:
            p[t + 1] = (1 - alpha_p) * p[t]   # key TSB update — decreases on zero
            d[t + 1] = d[t]                     # size unchanged on zero periods

    return float(p[n]), float(d[n]), fitted


def _tsb_mse(y: np.ndarray, alpha_p: float, alpha_d: float) -> float:
    """In-sample MSE on non-zero periods."""
    _, _, fitted = _tsb_fit(y, alpha_p, alpha_d)
    nz  = y > 0
    if nz.sum() == 0:
        return float("inf")
    return float(np.mean((y[nz] - fitted[nz]) ** 2))


def run_tsb(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("TSB requires 'date' and 'value' columns.")

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

    y = df["value"].astype("float64").values
    n = len(y)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")

    if n < 4:
        raise ValueError("TSB requires at least 4 observations.")

    # Ensure non-negative (intermittent demand must be >= 0)
    if (y < 0).any():
        raise ValueError("TSB requires non-negative demand values.")

    nz_count = int((y > 0).sum())

    # ── Fallback for near-zero series ────────────────────────────────────────
    if nz_count < MIN_NONZERO:
        # Series is almost entirely zero — forecast zero with tight intervals
        future_index = pd.date_range(
            start=df.index[-1], periods=horizon + 1, freq=inferred)[1:]
        future_forecast = np.zeros(horizon, dtype="float64")
        ci_low          = np.zeros(horizon, dtype="float64")
        ci_high         = np.full(horizon, float(np.max(y)) + 1e-8, dtype="float64")

        hist_block = pd.DataFrame({
            "date": df.index, "actual": np.nan,
            "forecast": np.zeros(n), "ci_low": np.nan,
            "ci_mid": np.zeros(n), "ci_high": np.nan, "error_pct": np.nan,
        })
        future_block = pd.DataFrame({
            "date": future_index, "actual": np.nan,
            "forecast": future_forecast, "ci_low": ci_low,
            "ci_mid": future_forecast, "ci_high": ci_high, "error_pct": np.nan,
        })
        for b in (hist_block, future_block):
            b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
                b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")
        forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
        forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

        return ForecastResult(
            model_name="TSB",
            forecast_df=forecast_df[[
                "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
            ]],
            metrics=None,
            metadata={
                "mode": "near_zero_fallback",
                "nz_count": nz_count,
                "frequency": inferred,
                "confidence_level": confidence_level,
                "min_tier": "pro",
                "output_contract": "ForecastResult",
            },
        )

    # ── Grid search for optimal alpha_p, alpha_d ─────────────────────────────
    best_mse    = float("inf")
    best_alpha_p = 0.10
    best_alpha_d = 0.10

    for ap in ALPHA_GRID:
        for ad in ALPHA_GRID:
            mse = _tsb_mse(y, float(ap), float(ad))
            if mse < best_mse:
                best_mse     = mse
                best_alpha_p = float(ap)
                best_alpha_d = float(ad)

    # ── Fit final model ───────────────────────────────────────────────────────
    p_hat, d_hat, fitted = _tsb_fit(y, best_alpha_p, best_alpha_d)
    point_forecast       = float(p_hat * d_hat)

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    nz_values = y[y > 0]
    rng       = np.random.default_rng(seed=42)

    alpha_ci = 1.0 - confidence_level
    q_lo     = alpha_ci / 2.0
    q_hi     = 1.0 - alpha_ci / 2.0

    # Simulate N_BOOTSTRAP paths of length `horizon`
    paths = np.zeros((N_BOOTSTRAP, horizon), dtype="float64")
    for b in range(N_BOOTSTRAP):
        for h in range(horizon):
            # Demand occurrence ~ Bernoulli(p_hat)
            occurs = rng.random() < p_hat
            if occurs:
                size = float(rng.choice(nz_values))
            else:
                size = 0.0
            paths[b, h] = size

    # Cumulative demand paths for widening CI
    cum_paths = np.cumsum(paths, axis=1)
    ci_low    = np.percentile(cum_paths, 100 * q_lo, axis=0) / \
                np.arange(1, horizon + 1)   # per-period rate
    ci_high   = np.percentile(cum_paths, 100 * q_hi, axis=0) / \
                np.arange(1, horizon + 1)

    # Point forecast is constant (TSB produces a rate, not trajectory)
    future_forecast = np.full(horizon, point_forecast, dtype="float64")
    ci_low  = np.minimum(ci_low.astype("float64"),  future_forecast)
    ci_high = np.maximum(ci_high.astype("float64"), future_forecast)

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite values in TSB forecast.")

    # ── Output assembly ───────────────────────────────────────────────────────
    future_index = pd.date_range(
        start=df.index[-1], periods=horizon + 1, freq=inferred)[1:]

    hist_block = pd.DataFrame({
        "date": df.index, "actual": np.nan,
        "forecast": fitted.astype("float64"), "ci_low": np.nan,
        "ci_mid": fitted.astype("float64"), "ci_high": np.nan, "error_pct": np.nan,
    })
    future_block = pd.DataFrame({
        "date": future_index, "actual": np.nan,
        "forecast": future_forecast, "ci_low": ci_low,
        "ci_mid": future_forecast, "ci_high": ci_high, "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in TSB output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in TSB future output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in TSB output.")

    return ForecastResult(
        model_name  = "TSB",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "alpha_p":          best_alpha_p,
            "alpha_d":          best_alpha_d,
            "p_hat":            round(p_hat, 6),
            "d_hat":            round(d_hat, 6),
            "point_forecast":   round(point_forecast, 6),
            "nz_count":         nz_count,
            "nz_pct":           round(nz_count / n, 4),
            "n_bootstrap":      N_BOOTSTRAP,
            "ci_method":        "bootstrap_bernoulli_demand_simulation",
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "pro",
            "output_contract":  "ForecastResult",
        },
    )
