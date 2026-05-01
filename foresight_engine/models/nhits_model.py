# ==============================================================================
# FILE: foresight_engine/models/nhits_model.py
# VERSION: 1.0.1
# UPDATED: Fix ImportError — ForecastResult imported from sentinel_engine.contracts
# MODEL: N-HiTS — NEURAL HIERARCHICAL INTERPOLATION FOR TIME SERIES
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# ==============================================================================
#
# PURPOSE:
#   N-HiTS (Challu et al., 2022) is a deep neural forecasting model that
#   uses hierarchical interpolation across multiple time scales to decompose
#   the forecasting task. It consistently outperforms NNETAR (single-layer
#   FFNN) and matches or exceeds N-BEATS on M4/M5 benchmarks.
#
#   This is what "neural time series forecasting" means at the Fortune 100
#   level in 2026. NNETAR is a 1993 architecture; N-HiTS is the current
#   state of the art for pure neural univariate forecasting.
#
# ARCHITECTURE:
#   N stacks (default: 3), each containing B blocks (default: 1).
#   Each block: fully connected residual network with:
#     - Pooling (max pooling at different window sizes per stack)
#     - Basis expansion (interpolation-based forecast)
#     - Residual connections (backcast subtracted from input)
#
#   Stack 1 (trend):      large pooling window, slow components
#   Stack 2 (seasonality): medium pooling window, periodic components
#   Stack 3 (residual):   small pooling window, high-frequency components
#
#   Final forecast = sum of all stack outputs.
#
# TRAINING:
#   Single-series training using a sliding window approach.
#   Each window of length LOOKBACK_MULT * horizon is a training sample.
#   Loss: MSE (point forecast). Quantile models trained separately for CI.
#   Optimizer: Adam. Early stopping on validation loss.
#
# CI METHOD:
#   Two additional models trained at alpha/2 and 1-alpha/2 quantiles
#   using pinball (quantile) loss. Same architecture as point model.
#
# DEPENDENCY:
#   Requires PyTorch. Install with:
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# STRENGTHS: Non-linear patterns, complex seasonality, long-horizon forecasts.
#            Hierarchical decomposition captures trend + seasonality jointly.
#            Much stronger than NNETAR on hostile scenarios.
# KNOWN LIMITS: Requires ≥ 3 * horizon training observations.
#               Training takes ~2-5 seconds per model on CPU.
# CONTRACT: Returns ForecastResult — see foresight_engine/contracts.py
# ==============================================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .contracts import ForecastResult

# ==============================================================================
# WINDOWS DLL PATH REGISTRATION
# PyTorch on Windows requires the torch/lib directory to be
# registered in the DLL search path before the first import.
# This must run at module level — not inside a function —
# so it executes when nhits_model.py is first imported.
# ==============================================================================
import sys as _sys
if _sys.platform == "win32":
    import os as _os
    import importlib.util as _ilu
    _torch_spec = _ilu.find_spec("torch")
    if _torch_spec and _torch_spec.origin:
        _torch_lib_dir = _os.path.join(
            _os.path.dirname(_torch_spec.origin), "lib"
        )
        if _os.path.isdir(_torch_lib_dir):
            _os.add_dll_directory(_torch_lib_dir)
    del _os, _ilu, _torch_spec  # clean up module namespace

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

N_STACKS        = 3       # trend, seasonality, residual
N_BLOCKS        = 1       # blocks per stack
N_LAYERS        = 2       # fully connected layers per block
HIDDEN_SIZE     = 512     # hidden units per FC layer
LOOKBACK_MULT   = 3       # lookback = lookback_mult * horizon
MAX_EPOCHS      = 100     # maximum training epochs
PATIENCE        = 10      # early stopping patience
BATCH_SIZE      = 32      # training batch size
LEARNING_RATE   = 1e-3    # Adam learning rate
MIN_OBS_MULT    = 3       # minimum obs = min_obs_mult * horizon


# ------------------------------------------------------------------------------
# PYTORCH MODEL DEFINITION
# ------------------------------------------------------------------------------

def _build_nhits_model(horizon: int, lookback: int, hidden: int,
                        n_stacks: int, n_blocks: int, n_layers: int):
    """
    Build the N-HiTS model in PyTorch.
    Returns an nn.Module.
    """
    import torch
    import torch.nn as nn

    class NHiTSBlock(nn.Module):
        def __init__(self, input_size, horizon, hidden, n_layers, pool_size):
            super().__init__()
            self.pool_size = pool_size
            pooled_size = input_size // pool_size + (1 if input_size % pool_size else 0)

            layers = []
            in_dim = pooled_size
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
                in_dim = hidden

            self.fc = nn.Sequential(*layers)

            # Backcast (reconstruction of input)
            self.backcast_head = nn.Linear(hidden, input_size)
            # Forecast (future prediction)
            self.forecast_head = nn.Linear(hidden, horizon)

        def forward(self, x):
            # Max pooling for multi-scale input
            if self.pool_size > 1:
                pad = (self.pool_size - x.shape[-1] % self.pool_size) % self.pool_size
                if pad > 0:
                    x_pad = torch.nn.functional.pad(x, (pad, 0))
                else:
                    x_pad = x
                x_pooled = x_pad.reshape(x_pad.shape[0], -1, self.pool_size).max(dim=-1).values
            else:
                x_pooled = x

            h = self.fc(x_pooled)
            backcast = self.backcast_head(h)
            forecast = self.forecast_head(h)
            return backcast, forecast

    class NHiTS(nn.Module):
        def __init__(self, input_size, horizon, hidden, n_stacks, n_blocks, n_layers):
            super().__init__()
            # Pool sizes increase per stack: small → large (residual → trend)
            # Stack 0 (trend): large pool
            # Stack 1 (seasonality): medium pool
            # Stack 2 (residual): small pool (1 = no pooling)
            pool_sizes = []
            for s in range(n_stacks):
                if n_stacks == 1:
                    pool_sizes.append(1)
                else:
                    # Distribute pool sizes: last stack has pool=1, first has larger
                    p = max(1, input_size // (horizon * (s + 1)))
                    pool_sizes.append(p)
            pool_sizes = list(reversed(pool_sizes))

            self.stacks = nn.ModuleList()
            for s in range(n_stacks):
                stack_blocks = nn.ModuleList([
                    NHiTSBlock(input_size, horizon, hidden, n_layers, pool_sizes[s])
                    for _ in range(n_blocks)
                ])
                self.stacks.append(stack_blocks)

        def forward(self, x):
            forecast_total = torch.zeros(x.shape[0], self.stacks[0][0].forecast_head.out_features,
                                         device=x.device)
            residual = x
            for stack_blocks in self.stacks:
                for block in stack_blocks:
                    backcast, forecast = block(residual)
                    residual = residual - backcast
                    forecast_total = forecast_total + forecast
            return forecast_total

    return NHiTS(lookback, horizon, hidden, n_stacks, n_blocks, n_layers)


# ------------------------------------------------------------------------------
# TRAINING UTILITIES
# ------------------------------------------------------------------------------

def _make_windows(y: np.ndarray, lookback: int, horizon: int):
    """
    Create sliding window training samples.
    Returns X (n_windows, lookback), y (n_windows, horizon).
    """
    n = len(y)
    max_start = n - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError(f"Series too short for window construction. "
                         f"Need >= {lookback + horizon}, got {n}.")

    X_list, y_list = [], []
    for i in range(max_start):
        X_list.append(y[i: i + lookback])
        y_list.append(y[i + lookback: i + lookback + horizon])

    return np.array(X_list, dtype="float32"), np.array(y_list, dtype="float32")


def _train_nhits(
    y_norm:    np.ndarray,
    horizon:   int,
    lookback:  int,
    quantile:  float | None,
    device:    str,
) -> object:
    """
    Train one N-HiTS model (point or quantile).
    y_norm: normalised training series (zero-mean, unit-std).
    quantile: None for MSE (point), float for pinball loss.
    Returns trained nn.Module.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    X, y_target = _make_windows(y_norm, lookback, horizon)

    # Train/val split: 80/20
    n_total = len(X)
    n_train = max(int(n_total * 0.8), 1)
    X_tr, X_va = X[:n_train], X[n_train:]
    y_tr, y_va = y_target[:n_train], y_target[n_train:]

    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_va_t = torch.tensor(X_va, device=device) if len(X_va) > 0 else None
    y_va_t = torch.tensor(y_va, device=device) if len(y_va) > 0 else None

    model = _build_nhits_model(
        horizon, lookback, HIDDEN_SIZE, N_STACKS, N_BLOCKS, N_LAYERS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def pinball_loss(pred, target, q):
        err   = target - pred
        loss  = torch.where(err >= 0, q * err, (q - 1) * err)
        return loss.mean()

    def mse_loss(pred, target):
        return ((pred - target) ** 2).mean()

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    n_batches = max(1, n_train // BATCH_SIZE)

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Shuffle
        perm = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        for b in range(n_batches):
            idx     = perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            if len(idx) == 0:
                continue
            x_batch = X_tr_t[idx]
            y_batch = y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = pinball_loss(pred, y_batch, quantile) if quantile is not None \
                   else mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        if X_va_t is not None and len(X_va_t) > 0:
            model.eval()
            with torch.no_grad():
                pred_va = model(X_va_t)
                val_loss = (pinball_loss(pred_va, y_va_t, quantile)
                            if quantile is not None
                            else mse_loss(pred_va, y_va_t)).item()
        else:
            val_loss = epoch_loss / max(n_batches, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _forecast_nhits(
    model:    object,
    y_norm:   np.ndarray,
    lookback: int,
    horizon:  int,
    device:   str,
) -> np.ndarray:
    """
    Generate horizon-step forecast from trained N-HiTS model.
    Uses the last `lookback` observations as input.
    Returns raw (normalised) forecast array of length `horizon`.
    """
    import torch
    x = torch.tensor(y_norm[-lookback:].astype("float32"), device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy().astype("float64")
    return pred


# ==============================================================================
# MODEL RUNNER
# ==============================================================================

def run_nhits(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("N-HiTS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    df = df.set_index("date").asfreq(inferred).reset_index()

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64").values
    n = len(y)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")

    min_obs = MIN_OBS_MULT * horizon
    if n < min_obs:
        raise ValueError(
            f"N-HiTS requires >= {min_obs} observations (= {MIN_OBS_MULT} × horizon). "
            f"Got {n}."
        )

    # --------------------------------------------------------------------------
    # PYTORCH AVAILABILITY CHECK
    # --------------------------------------------------------------------------

    try:
        import torch
        device = "cpu"
    except ImportError:
        raise RuntimeError(
            "N-HiTS requires PyTorch. Install with: "
            "pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        )

    # --------------------------------------------------------------------------
    # NORMALISATION
    # Mean/std normalisation for stable training.
    # Forecasts are denormalised before output.
    # --------------------------------------------------------------------------

    y_mean = float(np.mean(y))
    y_std  = float(np.std(y))
    if y_std < 1e-6:
        # Flat series — std near zero, use range-based normalisation
        y_std = max(float(np.max(y) - np.min(y)), 1.0)

    y_norm = ((y - y_mean) / y_std).astype("float32")

    lookback = max(LOOKBACK_MULT * horizon, 24)
    lookback = min(lookback, n - horizon)  # cannot exceed available history

    # --------------------------------------------------------------------------
    # TRAIN THREE MODELS: POINT + LOWER + UPPER CI
    # --------------------------------------------------------------------------

    alpha_ci = 1.0 - confidence_level
    q_lo     = alpha_ci / 2.0
    q_hi     = 1.0 - alpha_ci / 2.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_point = _train_nhits(y_norm, horizon, lookback, None,  device)
            model_lo    = _train_nhits(y_norm, horizon, lookback, q_lo,  device)
            model_hi    = _train_nhits(y_norm, horizon, lookback, q_hi,  device)
    except Exception as exc:
        raise RuntimeError(f"N-HiTS training failed: {exc}") from exc

    # --------------------------------------------------------------------------
    # HISTORICAL FITTED VALUES
    # Using sliding window predictions on training data.
    # --------------------------------------------------------------------------

    fitted_norm  = np.full(n, np.nan)
    n_windows    = n - lookback - horizon + 1

    if n_windows > 0:
        import torch
        X_hist = np.array([
            y_norm[i: i + lookback] for i in range(n_windows)
        ], dtype="float32")
        X_t = torch.tensor(X_hist, device=device)
        model_point.eval()
        with torch.no_grad():
            preds_hist = model_point(X_t).cpu().numpy()
        # Each prediction covers positions [lookback+i : lookback+i+horizon]
        # For fitted values, use only the 1-step-ahead prediction
        for i in range(n_windows):
            pos = lookback + i
            if pos < n:
                fitted_norm[pos] = preds_hist[i, 0]

    fitted_vals = fitted_norm * y_std + y_mean
    # Where fitted is nan, use actual (historical block)
    hist_forecast = np.where(np.isfinite(fitted_vals), fitted_vals, y)

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  hist_forecast,
        "ci_low":    np.nan,
        "ci_mid":    hist_forecast,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------------------------------

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future_norm_point = _forecast_nhits(model_point, y_norm, lookback, horizon, device)
            future_norm_lo    = _forecast_nhits(model_lo,    y_norm, lookback, horizon, device)
            future_norm_hi    = _forecast_nhits(model_hi,    y_norm, lookback, horizon, device)
    except Exception as exc:
        raise RuntimeError(f"N-HiTS forecast failed: {exc}") from exc

    # Denormalise
    future_point = future_norm_point * y_std + y_mean
    future_lo    = future_norm_lo    * y_std + y_mean
    future_hi    = future_norm_hi    * y_std + y_mean

    # Enforce CI brackets point forecast
    future_lo = np.minimum(future_lo, future_point)
    future_hi = np.maximum(future_hi, future_point)

    if not np.isfinite(future_point).all():
        raise RuntimeError("N-HiTS produced non-finite forecast values.")

    # --------------------------------------------------------------------------
    # FUTURE BLOCK
    # --------------------------------------------------------------------------

    future_index = pd.date_range(
        start=df["date"].iloc[-1], periods=horizon + 1, freq=inferred
    )[1:]

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_point,
        "ci_low":    future_lo,
        "ci_mid":    future_point,
        "ci_high":   future_hi,
        "error_pct": np.nan,
    })

    # --------------------------------------------------------------------------
    # DTYPE GOVERNANCE + OUTPUT
    # --------------------------------------------------------------------------

    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols]   = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in N-HiTS output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in N-HiTS future forecast output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in N-HiTS future forecast output.")

    return ForecastResult(
        model_name  = "N-HiTS",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = {
            "n_stacks":        N_STACKS,
            "n_blocks":        N_BLOCKS,
            "n_layers":        N_LAYERS,
            "hidden_size":     HIDDEN_SIZE,
            "lookback":        lookback,
            "horizon":         horizon,
            "max_epochs":      MAX_EPOCHS,
            "patience":        PATIENCE,
            "y_mean":          round(float(y_mean), 6),
            "y_std":           round(float(y_std),  6),
            "ci_method":       "quantile_pinball_loss",
            "q_lo":            round(q_lo, 4),
            "q_hi":            round(q_hi, 4),
            "frequency":       inferred,
            "confidence_level":confidence_level,
            "device":          device,
            "min_tier":        "enterprise",
            "output_contract": "ForecastResult",
        },
    )
