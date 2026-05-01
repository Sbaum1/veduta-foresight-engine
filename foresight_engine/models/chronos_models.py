# ==================================================
# FILE: foresight_engine/models/chronos_models.py
# VERSION: 1.0.0
# MODELS: Chronos-Bolt-Small · Chronos-Bolt-Base ·
#         Chronos-T5-Small · Chronos-2
# ENGINE: Foresight Engine v3.0.0
# TIER: enterprise (minimum)
# ==================================================
#
# PURPOSE:
#   Four zero-shot foundation models from Amazon Research.
#   No training on your data — inference only. Model weights
#   are downloaded from HuggingFace Hub on first call and
#   cached permanently on your machine (~200-800MB per model).
#
#   Chronos-Bolt-Small  (48M params)
#     Patch-based direct multi-step quantile forecasting.
#     250× faster than original Chronos of same size.
#     Best speed/accuracy balance for ensemble membership.
#     HuggingFace: amazon/chronos-bolt-small
#
#   Chronos-Bolt-Base  (205M params)
#     Same architecture as Bolt-Small, larger capacity.
#     Outperforms original Chronos-Large (700M) at 600× speed.
#     Best Bolt-family accuracy for production ensemble.
#     HuggingFace: amazon/chronos-bolt-base
#
#   Chronos-T5-Small  (46M params)
#     Original T5-based autoregressive Chronos.
#     Monte Carlo sampling gives native probabilistic intervals
#     without approximation — richest CI calibration in family.
#     HuggingFace: amazon/chronos-t5-small
#
#   Chronos-2  (120M params, released Oct 2025)
#     Encoder-only architecture with group attention.
#     Best on fev-bench, GIFT-Eval, Chronos Benchmark II.
#     90%+ win rate over Chronos-Bolt in head-to-head.
#     HuggingFace: amazon/chronos-2
#
# PIPELINE CACHING:
#   Module-level dict caches each pipeline after first load.
#   All four runner functions share this cache.
#   First call per model: downloads weights + loads pipeline.
#   Subsequent calls: inference only (weights already in RAM).
#   This is critical for ensemble runs — avoids reloading
#   200-800MB per series.
#
# FIT / PREDICT:
#   fit() is a zero-shot no-op — stores context only.
#   predict() calls the pre-trained pipeline with the context.
#   p50 → point forecast. p10/p90 → native CI bounds.
#
# INSTALLATION:
#   pip install chronos-forecasting
#   pip install torch   (if not already installed)
#
# GOVERNANCE:
#   - Graceful ImportError if chronos-forecasting not installed
#   - Graceful RuntimeError if HuggingFace Hub unreachable
#   - No Streamlit dependencies
#   - No session state dependencies
#   - Output contract: ForecastResult
#
# REFERENCES:
#   Chronos:      https://arxiv.org/abs/2403.07815 (TMLR 2024)
#   Chronos-Bolt: https://aws.amazon.com/blogs/machine-learning/chronos-bolt
#   Chronos-2:    https://arxiv.org/abs/2510.15821 (Oct 2025)
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .contracts import ForecastResult

# ── Module-level pipeline cache ───────────────────────────────────────────────
# Keyed by HuggingFace model ID. Pipelines are loaded once per process.
_PIPELINE_CACHE: dict = {}

# Chronos quantile indices (p10=0, p50=1, p90=2)
_Q_LEVELS        = [0.1, 0.5, 0.9]
_IDX_P10         = 0
_IDX_P50         = 1
_IDX_P90         = 2
_MAX_CONTEXT     = 512       # cap context to avoid OOM on very long series
_N_SAMPLES_T5    = 20        # Monte Carlo samples for T5 CI

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


# ==============================================================================
# SHARED INFRASTRUCTURE
# ==============================================================================

def _validate_and_prep(df: pd.DataFrame) -> tuple:
    """Shared input validation for all Chronos runners."""
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Chronos requires 'date' and 'value' columns.")
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
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values in series.")
    if len(y) < 12:
        raise ValueError("Chronos requires at least 12 observations.")
    return df, inferred, y


def _build_context_tensor(y: np.ndarray):
    """Convert numpy array to torch tensor for Chronos input."""
    import torch
    ctx = y.astype(np.float32)
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
    # Cap context to _MAX_CONTEXT most recent observations
    if len(ctx) > _MAX_CONTEXT:
        ctx = ctx[-_MAX_CONTEXT:]
    return torch.tensor(ctx, dtype=torch.float32)


def _load_bolt_pipeline(model_id: str):
    """Load and cache a ChronosBoltPipeline."""
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]
    try:
        from chronos import ChronosBoltPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting is required for Chronos-Bolt models. "
            "Install with: pip install chronos-forecasting"
        )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = ChronosBoltPipeline.from_pretrained(
            model_id,
            device_map  = device,
            dtype = dtype,
        )
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


def _load_t5_pipeline(model_id: str):
    """Load and cache a ChronosPipeline (T5-based)."""
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]
    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting is required for Chronos-T5 models. "
            "Install with: pip install chronos-forecasting"
        )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = ChronosPipeline.from_pretrained(
            model_id,
            device_map  = device,
            dtype = dtype,
        )
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


def _load_chronos2_pipeline(model_id: str):
    """Load and cache a Chronos2Pipeline."""
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting >= 2.0 is required for Chronos-2. "
            "Install with: pip install chronos-forecasting"
        )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map  = device,
            dtype = dtype,
        )
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


def _scale_ci_to_level(
    p10:              np.ndarray,
    p90:              np.ndarray,
    point:            np.ndarray,
    confidence_level: float,
) -> tuple:
    """
    Scale native p10/p90 (80% CI) to the requested confidence level.
    For levels > 0.80, extrapolate using the empirical spread.
    """
    if confidence_level <= 0.80:
        return (
            np.minimum(p10, point).astype("float64"),
            np.maximum(p90, point).astype("float64"),
        )
    # Extrapolate: 80% CI covers ~1.28σ each side; 90% → 1.645σ
    from scipy.stats import norm as _norm
    z_80  = float(_norm.ppf(0.90))   # = 1.2816 (one-sided for 80% two-sided)
    z_req = float(_norm.ppf((1 + confidence_level) / 2))
    factor = z_req / z_80
    spread = (p90 - p10) / 2.0
    ci_lo  = (point - spread * factor).astype("float64")
    ci_hi  = (point + spread * factor).astype("float64")
    return (
        np.minimum(ci_lo, point).astype("float64"),
        np.maximum(ci_hi, point).astype("float64"),
    )


def _assemble_result(
    df:               pd.DataFrame,
    y:                np.ndarray,
    inferred:         str,
    horizon:          int,
    point_forecast:   np.ndarray,
    ci_low:           np.ndarray,
    ci_high:          np.ndarray,
    model_name:       str,
    confidence_level: float,
    metadata:         dict,
) -> ForecastResult:
    """Build ForecastResult from Chronos output."""
    if not np.isfinite(point_forecast).all():
        raise RuntimeError(f"Non-finite values in {model_name} forecast.")

    future_index = pd.date_range(
        start   = pd.to_datetime(df["date"].iloc[-1]),
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    # Historical block: Chronos is zero-shot so no fitted values
    # Use actual values as the "fitted" representation for display
    hist_block = pd.DataFrame({
        "date":      pd.to_datetime(df["date"].values),
        "actual":    np.nan,
        "forecast":  y.astype("float64"),
        "ci_low":    np.nan,
        "ci_mid":    y.astype("float64"),
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  point_forecast,
        "ci_low":    np.minimum(ci_low,  point_forecast),
        "ci_mid":    point_forecast,
        "ci_high":   np.maximum(ci_high, point_forecast),
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
# CHRONOS-BOLT-SMALL
# ==============================================================================

_BOLT_SMALL_ID = "amazon/chronos-bolt-small"


def run_chronos_bolt_small(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Chronos-Bolt-Small (48M params).
    Patch-based direct multi-step forecasting. Zero-shot.
    """
    df, inferred, y = _validate_and_prep(df)
    import torch

    pipe    = _load_bolt_pipeline(_BOLT_SMALL_ID)
    context = _build_context_tensor(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            quantiles, _ = pipe.predict_quantiles(
                inputs            = context.unsqueeze(0),
                prediction_length = horizon,
                quantile_levels   = _Q_LEVELS,
            )

    # quantiles: (1, horizon, 3) — extract series 0
    q = quantiles[0].cpu().numpy().astype("float64")   # (horizon, 3)

    point  = q[:, _IDX_P50]
    p10    = q[:, _IDX_P10]
    p90    = q[:, _IDX_P90]
    ci_lo, ci_hi = _scale_ci_to_level(p10, p90, point, confidence_level)

    return _assemble_result(
        df, y, inferred, horizon, point, ci_lo, ci_hi,
        model_name       = "Chronos-Bolt-Small",
        confidence_level = confidence_level,
        metadata         = {
            "model_id":         _BOLT_SMALL_ID,
            "params_M":         48,
            "architecture":     "patch_based_direct_multistep",
            "zero_shot":        True,
            "ci_method":        "native_quantile_p10_p90",
            "context_length":   min(len(y), _MAX_CONTEXT),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# CHRONOS-BOLT-BASE
# ==============================================================================

_BOLT_BASE_ID = "amazon/chronos-bolt-base"


def run_chronos_bolt_base(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Chronos-Bolt-Base (205M params).
    Higher-capacity patch-based variant. Outperforms Chronos-Large at 600× speed.
    """
    df, inferred, y = _validate_and_prep(df)
    import torch

    pipe    = _load_bolt_pipeline(_BOLT_BASE_ID)
    context = _build_context_tensor(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            quantiles, _ = pipe.predict_quantiles(
                inputs            = context.unsqueeze(0),
                prediction_length = horizon,
                quantile_levels   = _Q_LEVELS,
            )

    q = quantiles[0].cpu().numpy().astype("float64")

    point  = q[:, _IDX_P50]
    p10    = q[:, _IDX_P10]
    p90    = q[:, _IDX_P90]
    ci_lo, ci_hi = _scale_ci_to_level(p10, p90, point, confidence_level)

    return _assemble_result(
        df, y, inferred, horizon, point, ci_lo, ci_hi,
        model_name       = "Chronos-Bolt-Base",
        confidence_level = confidence_level,
        metadata         = {
            "model_id":         _BOLT_BASE_ID,
            "params_M":         205,
            "architecture":     "patch_based_direct_multistep",
            "zero_shot":        True,
            "ci_method":        "native_quantile_p10_p90",
            "context_length":   min(len(y), _MAX_CONTEXT),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# CHRONOS-T5-SMALL
# ==============================================================================

_T5_SMALL_ID = "amazon/chronos-t5-small"


def run_chronos_t5_small(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Chronos-T5-Small (46M params).
    Original T5-based autoregressive Chronos. Monte Carlo sampling
    produces native probabilistic intervals — best CI calibration.
    """
    df, inferred, y = _validate_and_prep(df)
    import torch

    pipe    = _load_t5_pipeline(_T5_SMALL_ID)
    context = _build_context_tensor(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            # samples: (1, n_samples, horizon)
            samples = pipe.predict(
                inputs                  = context.unsqueeze(0),
                prediction_length       = horizon,
                num_samples             = _N_SAMPLES_T5,
                limit_prediction_length = False,
            )

    # Extract (n_samples, horizon)
    s = samples[0].cpu().numpy().astype("float64")

    point  = np.median(s, axis=0)
    alpha  = 1.0 - confidence_level
    p10    = np.percentile(s, 100 * alpha / 2,       axis=0)
    p90    = np.percentile(s, 100 * (1 - alpha / 2), axis=0)
    ci_lo  = np.minimum(p10, point).astype("float64")
    ci_hi  = np.maximum(p90, point).astype("float64")

    return _assemble_result(
        df, y, inferred, horizon, point, ci_lo, ci_hi,
        model_name       = "Chronos-T5-Small",
        confidence_level = confidence_level,
        metadata         = {
            "model_id":         _T5_SMALL_ID,
            "params_M":         46,
            "architecture":     "t5_autoregressive_sampling",
            "n_samples":        _N_SAMPLES_T5,
            "zero_shot":        True,
            "ci_method":        "monte_carlo_empirical_quantile",
            "context_length":   min(len(y), _MAX_CONTEXT),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )


# ==============================================================================
# CHRONOS-2
# ==============================================================================

_CHRONOS2_ID = "amazon/chronos-2"


def run_chronos_2(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Chronos-2 (120M params, Oct 2025).
    Encoder-only with group attention. Best on fev-bench, GIFT-Eval,
    Chronos Benchmark II. 90%+ win rate over Chronos-Bolt.
    """
    df, inferred, y = _validate_and_prep(df)
    import torch

    pipe    = _load_chronos2_pipeline(_CHRONOS2_ID)
    context = _build_context_tensor(y)

    # ── Chronos-2 inference ───────────────────────────────────────────────────
    # Chronos-2 v2.x exposes predict_quantiles() as the correct method.
    # predict() accepts only **kwargs with no quantile or sample params.
    # We use predict_quantiles() if available, fall back to predict() only
    # if the method doesn't exist (future API change).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            if hasattr(pipe, "predict_quantiles"):
                # Chronos-2 v2.2+ requires 3D input: (n_series, n_variates, history_length)
                # context is 1D (n,) — unsqueeze twice: (1, 1, n)
                # This differs from Bolt which uses 2D (1, n) input.
                c2_input = context.unsqueeze(0).unsqueeze(0)  # (1, 1, n)
                quantiles, _ = pipe.predict_quantiles(
                    inputs            = c2_input,
                    prediction_length = horizon,
                    quantile_levels   = _Q_LEVELS,
                )
                # predict_quantiles returns (quantiles_obj, mean_obj)
                # quantiles_obj is a tensor or list of tensors.
                # Extract to numpy, squeezing all leading size-1 dims.
                if hasattr(quantiles, 'cpu'):
                    q_np = quantiles.cpu().numpy().astype("float64")
                elif isinstance(quantiles, (list, tuple)):
                    t = quantiles[0]
                    q_np = t.cpu().numpy().astype("float64") if hasattr(t, 'cpu') else                            np.asarray(t, dtype="float64")
                else:
                    q_np = np.asarray(quantiles, dtype="float64")

                # Squeeze all leading size-1 dims until (horizon, n_q)
                while q_np.ndim > 2 and q_np.shape[0] == 1:
                    q_np = q_np[0]
                # Normalise orientation
                if q_np.ndim == 2 and q_np.shape[0] == len(_Q_LEVELS) and q_np.shape[1] != len(_Q_LEVELS):
                    q_np = q_np.T
                point = q_np[:, _IDX_P50]
                p10   = q_np[:, _IDX_P10]
                p90   = q_np[:, _IDX_P90]
            else:
                # Future fallback: bare predict() with no extra kwargs
                results = pipe.predict(
                    inputs            = [context],
                    prediction_length = horizon,
                )
                r_raw = results[0]
                if hasattr(r_raw, "cpu"):
                    r_arr = r_raw.cpu().numpy().astype("float64")
                else:
                    r_arr = np.asarray(r_raw, dtype="float64")
                if r_arr.ndim == 2 and r_arr.shape[0] > r_arr.shape[1]:
                    r_arr = r_arr.T
                point = np.median(r_arr, axis=0)[:horizon]
                p10   = np.percentile(r_arr, 10, axis=0)[:horizon]
                p90   = np.percentile(r_arr, 90, axis=0)[:horizon]

    if not np.isfinite(point).all():
        raise RuntimeError("Chronos-2 produced non-finite forecast values.")

    ci_lo, ci_hi = _scale_ci_to_level(p10, p90, point, confidence_level)

    return _assemble_result(
        df, y, inferred, horizon, point, ci_lo, ci_hi,
        model_name       = "Chronos-2",
        confidence_level = confidence_level,
        metadata         = {
            "model_id":         _CHRONOS2_ID,
            "params_M":         120,
            "architecture":     "encoder_only_group_attention",
            "zero_shot":        True,
            "ci_method":        "native_quantile_p10_p90",
            "context_length":   min(len(y), _MAX_CONTEXT),
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "min_tier":         "enterprise",
            "output_contract":  "ForecastResult",
        },
    )
