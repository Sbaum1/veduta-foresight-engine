# ==================================================
# FILE: foresight_engine/models/nnetar.py
# VERSION: 2.0.0
# MODEL: NNETAR — NEURAL NETWORK AUTOREGRESSION
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# STATUS: VEDUTA ENGINE — PHASE 3C
# ==================================================
#
# PURPOSE:
#   Feed-forward neural network trained on autoregressive
#   lag features. Captures non-linear patterns that SARIMA,
#   ETS, and other linear models cannot model.
#
#   Best on: volatile series, non-linear trend, complex
#   seasonal interactions, series where residuals from
#   linear models show persistent structure.
#
# ARCHITECTURE:
#   Single hidden layer feed-forward network.
#   Input: p lag features + seasonal lag features.
#   Hidden units: auto-selected as ceil(p/2) + 1.
#   Output: 1 (next period forecast).
#   Activation: tanh (hidden), linear (output).
#
# LAG SELECTION:
#   p lags from ACF: significant lags up to p_max.
#   Seasonal lag: +m (one full seasonal period).
#   Minimum p = 1, maximum p = p_max.
#
# TRAINING:
#   Multiple random restarts (n_restarts) to escape
#   local minima. Best model selected by training MSE.
#   Weights initialized from N(0, 0.1).
#
# CI METHOD:
#   Bootstrap simulation: n_boot forward paths from
#   perturbed inputs using training residual resampling.
#   Empirical quantiles at confidence_level bounds.
#
# IMPLEMENTATION:
#   Pure numpy — no deep learning framework dependency.
#   Portable, deterministic with seed, fast on CPU.
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from foresight_engine.models.contracts import ForecastResult

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

P_MAX      = 12     # Maximum AR lags to consider
N_RESTARTS = 10     # Random restarts for training
N_BOOT     = 500    # Bootstrap paths for CI
MAX_ITER   = 2000   # Training iterations per restart
LR         = 0.01   # Learning rate
RNG_SEED   = 42

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
# LAG SELECTION VIA ACF
# --------------------------------------------------

def _select_lags(y: np.ndarray, m: int, p_max: int) -> list[int]:
    """
    Select AR lags from significant ACF values.
    Always includes lag 1. Adds seasonal lag m if series long enough.
    Returns sorted list of lags (1-indexed).
    """
    n = len(y)
    lags = [1]

    # ACF-based lag selection
    for lag in range(2, min(p_max + 1, n // 3)):
        if lag >= n:
            break
        y_shifted = y[lag:]
        y_base    = y[:n - lag]
        if len(y_shifted) < 2:
            break
        corr = float(np.corrcoef(y_base, y_shifted)[0, 1])
        threshold = 1.96 / np.sqrt(n)
        if abs(corr) > threshold:
            lags.append(lag)

    # Always add seasonal lag if series is long enough
    if m > 1 and n > 2 * m and m not in lags:
        lags.append(m)

    return sorted(set(lags))


# --------------------------------------------------
# FEATURE BUILDER
# --------------------------------------------------

def _build_features(y: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lag feature matrix X and target vector y_target.
    Drops the first max(lags) observations.
    """
    max_lag = max(lags)
    n       = len(y)
    rows    = n - max_lag

    X = np.empty((rows, len(lags)), dtype="float64")
    for j, lag in enumerate(lags):
        X[:, j] = y[max_lag - lag: n - lag]

    y_target = y[max_lag:].astype("float64")
    return X, y_target


# --------------------------------------------------
# NORMALISATION
# --------------------------------------------------

def _normalise(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Min-max normalise features and target to [-1, 1]."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_rng = np.where(x_max - x_min == 0, 1.0, x_max - x_min)

    y_min = y.min()
    y_max = y.max()
    y_rng = y_max - y_min if y_max != y_min else 1.0

    X_norm = 2.0 * (X - x_min) / x_rng - 1.0
    y_norm = 2.0 * (y - y_min) / y_rng - 1.0

    scale = {"x_min": x_min, "x_max": x_max, "x_rng": x_rng,
             "y_min": y_min, "y_max": y_max, "y_rng": y_rng}
    return X_norm, y_norm, scale


def _denorm_y(y_norm: np.ndarray, scale: dict) -> np.ndarray:
    return (y_norm + 1.0) / 2.0 * scale["y_rng"] + scale["y_min"]


def _norm_x_row(x_row: np.ndarray, scale: dict) -> np.ndarray:
    return 2.0 * (x_row - scale["x_min"]) / scale["x_rng"] - 1.0


# --------------------------------------------------
# NETWORK — FORWARD PASS
# --------------------------------------------------

def _forward(X_row: np.ndarray, W1: np.ndarray, b1: np.ndarray,
             W2: np.ndarray, b2: float) -> float:
    """Single forward pass. Tanh hidden, linear output."""
    h = np.tanh(X_row @ W1 + b1)
    return float(h @ W2 + b2)


# --------------------------------------------------
# TRAINING — GRADIENT DESCENT WITH RESTARTS
# --------------------------------------------------

def _train(
    X_norm:     np.ndarray,
    y_norm:     np.ndarray,
    n_hidden:   int,
    n_restarts: int,
    max_iter:   int,
    lr:         float,
    rng:        np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Train single-hidden-layer NN with gradient descent.
    Multiple restarts — returns best weights by training MSE.
    """
    n_in   = X_norm.shape[1]
    best_mse  = np.inf
    best_W1 = best_b1 = best_W2 = best_b2 = None

    for _ in range(n_restarts):
        W1 = rng.normal(0, 0.1, (n_in, n_hidden))
        b1 = rng.normal(0, 0.1, (n_hidden,))
        W2 = rng.normal(0, 0.1, (n_hidden,))
        b2 = float(rng.normal(0, 0.1))

        for _ in range(max_iter):
            # Forward
            H   = np.tanh(X_norm @ W1 + b1)         # (n, n_hidden)
            out = H @ W2 + b2                         # (n,)
            err = out - y_norm                        # (n,)

            # Backward
            d_out = err / len(y_norm)
            d_W2  = H.T @ d_out
            d_b2  = d_out.sum()
            d_H   = np.outer(d_out, W2) * (1 - H**2)
            d_W1  = X_norm.T @ d_H
            d_b1  = d_H.sum(axis=0)

            # Update
            W1 -= lr * d_W1
            b1 -= lr * d_b1
            W2 -= lr * d_W2
            b2 -= lr * float(d_b2)

        mse = float(np.mean((np.tanh(X_norm @ W1 + b1) @ W2 + b2 - y_norm) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_W1, best_b1, best_W2, best_b2 = (
                W1.copy(), b1.copy(), W2.copy(), float(b2)
            )

    return best_W1, best_b1, best_W2, best_b2


# --------------------------------------------------
# MULTI-STEP FORECAST
# --------------------------------------------------

def _forecast_path(
    y_history:  np.ndarray,
    lags:       list[int],
    scale:      dict,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: float,
    horizon:    int,
) -> np.ndarray:
    """
    Iterative one-step-ahead forecast for h steps.
    Returns denormalised forecast array of length horizon.
    """
    history = list(y_history.astype("float64"))
    preds   = []

    for _ in range(horizon):
        x_row  = np.array([history[-lag] for lag in lags], dtype="float64")
        x_norm = _norm_x_row(x_row, scale)
        y_hat  = _forward(x_norm, W1, b1, W2, b2)
        y_real = float(_denorm_y(np.array([y_hat]), scale)[0])
        preds.append(y_real)
        history.append(y_real)

    return np.array(preds, dtype="float64")


# ==================================================
# MODEL RUNNER
# ==================================================

def run_nnetar(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("NNETAR requires 'date' and 'value' columns.")

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

    if len(df) < 16:
        raise ValueError("Minimum 16 observations required for NNETAR.")

    # --------------------------------------------------
    # SEASON LENGTH
    # --------------------------------------------------

    _season_map = {
        "MS": 12, "M": 12,
        "QS": 4,  "Q": 4,
        "W":  52, "W-SUN": 52, "W-MON": 52,
        "D":  7,
    }
    m = _season_map.get(inferred, 12)

    # --------------------------------------------------
    # LAG SELECTION
    # --------------------------------------------------

    lags = _select_lags(y, m, P_MAX)
    if not lags:
        lags = [1]

    # --------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------

    X, y_target = _build_features(y, lags)
    if len(X) < 4:
        raise ValueError("Insufficient data after lag construction.")

    n_hidden = max(2, int(np.ceil(len(lags) / 2)) + 1)

    # --------------------------------------------------
    # NORMALISE
    # --------------------------------------------------

    X_norm, y_norm, scale = _normalise(X, y_target)

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    rng = np.random.default_rng(RNG_SEED)

    W1, b1, W2, b2 = _train(
        X_norm, y_norm,
        n_hidden   = n_hidden,
        n_restarts = N_RESTARTS,
        max_iter   = MAX_ITER,
        lr         = LR,
        rng        = rng,
    )

    # --------------------------------------------------
    # TRAINING RESIDUALS
    # --------------------------------------------------

    H_hist    = np.tanh(X_norm @ W1 + b1)
    out_hist  = H_hist @ W2 + b2
    pred_hist = _denorm_y(out_hist, scale)
    residuals = y_target - pred_hist

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    max_lag = max(lags)
    fitted_full = np.full(len(y), np.nan)
    fitted_full[max_lag:] = pred_hist

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
    # POINT FORECAST
    # --------------------------------------------------

    point_forecast = _forecast_path(y, lags, scale, W1, b1, W2, b2, horizon)

    if not np.isfinite(point_forecast).all():
        raise RuntimeError("Non-finite values in point forecast.")

    # --------------------------------------------------
    # BOOTSTRAP CI
    # --------------------------------------------------

    finite_resid = residuals[np.isfinite(residuals)]
    boot_paths   = np.empty((N_BOOT, horizon), dtype="float64")

    rng_boot = np.random.default_rng(RNG_SEED + 1)
    for b in range(N_BOOT):
        noise    = rng_boot.choice(finite_resid, size=horizon, replace=True)
        y_perturbed = y.copy()
        y_perturbed[-1] += noise[0]   # perturb last observed
        path = _forecast_path(y_perturbed, lags, scale, W1, b1, W2, b2, horizon)
        path += noise
        boot_paths[b] = path

    alpha_ci = 1.0 - confidence_level
    lo_pct   = (alpha_ci / 2.0) * 100
    hi_pct   = (1.0 - alpha_ci / 2.0) * 100

    ci_low  = np.percentile(boot_paths, lo_pct, axis=0).astype("float64")
    ci_high = np.percentile(boot_paths, hi_pct, axis=0).astype("float64")

    # Ensure CI brackets point forecast
    ci_low  = np.minimum(ci_low,  point_forecast)
    ci_high = np.maximum(ci_high, point_forecast)

    if not np.isfinite(ci_low).all() or not np.isfinite(ci_high).all():
        # Fallback to sigma-based CI
        sigma   = float(np.std(finite_resid, ddof=1))
        z       = _get_z(confidence_level)
        h_arr   = np.arange(1, horizon + 1, dtype="float64")
        ci_low  = (point_forecast - z * sigma * np.sqrt(h_arr)).astype("float64")
        ci_high = (point_forecast + z * sigma * np.sqrt(h_arr)).astype("float64")

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
        "forecast":  point_forecast,
        "ci_low":    ci_low,
        "ci_mid":    point_forecast,
        "ci_high":   ci_high,
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
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in future forecast output.")

    return ForecastResult(
        model_name  = "NNETAR",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "lags":             lags,
            "n_lags":           len(lags),
            "n_hidden":         n_hidden,
            "n_restarts":       N_RESTARTS,
            "seasonal_period":  m,
            "ci_method":        "bootstrap_residual_resampling",
            "n_boot":           N_BOOT,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":       "ForecastResult",
        },
    )
