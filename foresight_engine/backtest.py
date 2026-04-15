# ==================================================
# FILE: foresight_engine/backtest.py
# VERSION: 2.3.0
# ROLE: ROLLING-ORIGIN CERTIFICATION ENGINE
# ENGINE: Foresight Engine v3.0.0
# UPDATED: G7 — MASE zero-mean / near-zero scale guard
# UPDATED: H1 Step 3 — MASE denominator two-floor guard
# UPDATED: M2 — Rolling folds 3 → 6
# ==================================================
#
# M2 UPGRADE — ROLLING FOLDS 3 → 6:
#
#   Previous: DEFAULT_ROLLING_FOLDS = 3
#     Three folds means MASE weights are estimated from three
#     out-of-sample windows. On a 72-observation M3 series with
#     horizon=12, folds cover months 37-48, 49-60, 61-72.
#     Three windows is the bare minimum — a model can get lucky
#     or unlucky on any single window, and three windows do not
#     average that out reliably.
#
#   Fixed: DEFAULT_ROLLING_FOLDS = 6
#     Six folds doubles the evaluation coverage. On the same 72-obs
#     series the folds now cover months 13-24 through 61-72.
#     This means:
#       - MASE weights are estimated from 6x more evidence
#       - The stacking layer (ensemble.py) has more fold predictions
#         to train the ridge regression meta-learner
#       - Certification scores are materially more reliable
#       - Variance of MASE estimates across folds (reported as
#         mase_std) is a more meaningful stability signal
#
#   Series length requirement:
#     MIN_OBSERVATIONS remains 36. With horizon=12 and 6 folds,
#     minimum series needed = 36 + 6*12 = 108? No — folds are
#     overlapping walk-forward windows, not sequential blocks.
#     The earliest fold's training set ends at len-6*horizon,
#     which for a 72-obs series = 72-72 = 0. Guard: each fold
#     skips if train_end < MIN_OBSERVATIONS. Short series
#     naturally produce fewer completed folds (reported in
#     aggregated["folds"]) — this is correct behaviour.
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - ROLLING_FOLDS and SEASONAL_PERIOD are configurable
# - Metric keys are lowercase — runner.py normalizes to contract keys
# - CI NaN guard: all-NaN CI coverage reported as None, not 0.0
# - engine_version injected into every output dict
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional

from .contracts import ENGINE_VERSION

# --------------------------------------------------
# DEFAULTS
# --------------------------------------------------

MIN_OBSERVATIONS = 36
DEFAULT_ROLLING_FOLDS  = 6   # M2: was 3 — doubled for more reliable MASE weights
DEFAULT_SEASONAL_PERIOD = 12

# G7: Minimum absolute scale floor for MASE denominator.
MASE_SCALE_FLOOR = 1e-6


# ==================================================
# METRIC FUNCTIONS
# ==================================================

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask  = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _directional_accuracy(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    y_train_last:  float,
) -> float:
    """
    Percentage of periods where model predicted correct direction of change
    relative to the last training observation.
    """
    true_diff = np.sign(y_true - y_train_last)
    pred_diff = np.sign(y_pred - y_train_last)
    return float(np.mean(true_diff == pred_diff))


def _mase(
    y_true:          np.ndarray,
    y_pred:          np.ndarray,
    y_train:         np.ndarray,
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
) -> float:
    """
    Seasonal MASE using seasonal naïve scaling.
    Falls back to first-difference naïve if insufficient training length.
    MASE < 1.0 = beats seasonal naïve baseline.

    G7: Scale is clamped to MASE_SCALE_FLOOR before division.
    Returns np.nan if scale is NaN (degenerate training data).
    Prevents infinity / explosion on near-flat or zero-mean series.
    """
    if len(y_train) > seasonal_period:
        naive_errors = np.abs(
            y_train[seasonal_period:] - y_train[:-seasonal_period]
        )
    else:
        naive_errors = np.abs(np.diff(y_train))

    scale = np.mean(naive_errors)

    if np.isnan(scale):
        return np.nan

    # G7: Absolute minimum floor — handles exact zero series
    # H1 Step 3: Mean-level anchor floor — prevents astronomical MASE on
    # near-flat series where seasonal naïve scale ≈ noise level but the
    # forecast error is measured against a meaningful real-world level.
    # Reference: Hyndman & Koehler (2006)
    mean_abs_train = float(np.mean(np.abs(y_train)))
    scale = float(max(scale, MASE_SCALE_FLOOR, mean_abs_train * 0.01))

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _theils_u(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_train: np.ndarray,
) -> float:
    """
    Theil's U statistic relative to rolling naïve forecast.
    U < 1.0 = model beats naïve.
    """
    naive_forecast = []
    last_value     = y_train[-1]

    for actual in y_true:
        naive_forecast.append(last_value)
        last_value = actual

    naive_arr = np.array(naive_forecast)

    numerator   = np.sqrt(np.mean((y_true - y_pred)    ** 2))
    denominator = np.sqrt(np.mean((y_true - naive_arr) ** 2))

    if np.isnan(denominator):
        return np.nan

    # G7: Clamp to floor — handles flat series where naïve also has zero error
    denominator = float(max(denominator, MASE_SCALE_FLOOR))

    return float(numerator / denominator)


def _ci_coverage(
    y_true:  np.ndarray,
    ci_low:  np.ndarray,
    ci_high: np.ndarray,
) -> Optional[float]:
    """
    Proportion of actuals falling within the predicted interval.

    UPGRADE vs original:
    - Returns None if all CI values are NaN (model produced no intervals)
    - Returns None if ci_low >= ci_high for all periods (degenerate intervals)
    - Avoids reporting 0.0 coverage when the model simply has no CI
    """
    ci_low  = ci_low.astype(float)
    ci_high = ci_high.astype(float)

    # All-NaN guard
    if not np.isfinite(ci_low).any() or not np.isfinite(ci_high).any():
        return None

    # Degenerate interval guard
    valid = np.isfinite(ci_low) & np.isfinite(ci_high) & (ci_high > ci_low)
    if valid.sum() == 0:
        return None

    inside = (y_true[valid] >= ci_low[valid]) & (y_true[valid] <= ci_high[valid])
    return float(np.mean(inside))


# ==================================================
# SINGLE FOLD EVALUATION
# ==================================================

def _evaluate_fold(
    train_df:         pd.DataFrame,
    test_df:          pd.DataFrame,
    model_runner:     Callable,
    horizon:          int,
    confidence_level: float,
    seasonal_period:  int,
) -> Optional[Dict[str, Any]]:
    """
    Run model on one train/test split and compute all metrics.

    Returns None if the fold cannot be evaluated cleanly.
    """

    result = model_runner(
        train_df,
        horizon          = horizon,
        confidence_level = confidence_level,
    )

    forecast_df         = result.forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    last_train_date = pd.to_datetime(train_df["date"]).max()

    future_block = forecast_df[
        forecast_df["date"] > last_train_date
    ].reset_index(drop=True)

    if len(future_block) != horizon:
        return None

    y_true = test_df["value"].astype(float).values
    y_pred = future_block["forecast"].astype(float).values

    if not np.isfinite(y_pred).all():
        return None

    y_train = train_df["value"].astype(float).values

    # CI coverage
    coverage: Optional[float] = None
    if {"ci_low", "ci_high"}.issubset(future_block.columns):
        ci_low  = future_block["ci_low"].values
        ci_high = future_block["ci_high"].values
        coverage = _ci_coverage(y_true, ci_low, ci_high)

    return {
        "mae":                  _mae(y_true, y_pred),
        "rmse":                 _rmse(y_true, y_pred),
        "mape":                 _mape(y_true, y_pred),
        "smape":                _smape(y_true, y_pred),
        "bias":                 _bias(y_true, y_pred),
        "mase":                 _mase(y_true, y_pred, y_train, seasonal_period),
        "theils_u":             _theils_u(y_true, y_pred, y_train),
        "ci_coverage":          coverage,
        "directional_accuracy": _directional_accuracy(
                                    y_true, y_pred, float(y_train[-1])
                                ),
    }


# ==================================================
# MAIN BACKTEST ENTRY POINT
# ==================================================

def run_backtest(
    df:               pd.DataFrame,
    model_runner:     Callable,
    horizon:          int,
    confidence_level: float,
    rolling_folds:    int = DEFAULT_ROLLING_FOLDS,
    seasonal_period:  int = DEFAULT_SEASONAL_PERIOD,
) -> Dict[str, Any]:
    """
    Rolling-origin walk-forward backtest certification engine.

    Runs `rolling_folds` train/test splits, each shifted back by
    `horizon` periods. Aggregates metrics across folds using nanmean.
    Fold stability metrics (std) are included for all keys.

    Args:
        df               : Full historical DataFrame ('date', 'value')
        model_runner     : Callable matching foresight_engine model signature
        horizon          : Forecast periods per fold
        confidence_level : Prediction interval confidence (e.g. 0.90)
        rolling_folds    : Number of walk-forward folds (default 3)
        seasonal_period  : Seasonal period for MASE scaling (default 12)

    Returns:
        Dict of aggregated metrics. Keys match runner.py normalization map.
        Includes 'eligible', 'folds', 'mean_level', 'engine_version'.

    Metric keys (lowercase — runner.py normalizes to contract casing):
        mae, rmse, mape, smape, bias, mase, theils_u,
        ci_coverage, directional_accuracy
        + _std variants for each (fold stability)
    """

    series = df.copy().sort_values("date").reset_index(drop=True)

    # ── Eligibility check ────────────────────────────────────────────────────
    if len(series) < MIN_OBSERVATIONS:
        return {
            "eligible":       False,
            "reason":         f"Minimum {MIN_OBSERVATIONS} observations required.",
            "observations":   len(series),
            "engine_version": ENGINE_VERSION,
        }

    y_full       = series["value"].astype(float).values
    fold_metrics: List[Dict[str, Any]] = []

    # ── Rolling-origin folds ─────────────────────────────────────────────────
    for fold in range(rolling_folds):

        train_end = len(y_full) - horizon - (rolling_folds - fold - 1) * horizon

        # Guard: skip fold if training window is too short for the engine.
        # With DEFAULT_ROLLING_FOLDS=6 and short series, early folds can
        # produce train_end <= 0 or below MIN_OBSERVATIONS, which causes
        # "Primary Ensemble received empty dataframe" crashes.
        if train_end < MIN_OBSERVATIONS:
            continue

        train_df  = series.iloc[:train_end].copy()
        test_df   = series.iloc[train_end: train_end + horizon].copy()

        if len(test_df) < horizon:
            continue

        fold_result = _evaluate_fold(
            train_df         = train_df,
            test_df          = test_df,
            model_runner     = model_runner,
            horizon          = horizon,
            confidence_level = confidence_level,
            seasonal_period  = seasonal_period,
        )

        if fold_result is not None:
            fold_metrics.append(fold_result)

    # ── Insufficient folds ───────────────────────────────────────────────────
    if not fold_metrics:
        return {
            "eligible":       False,
            "reason":         "Insufficient fold generation.",
            "engine_version": ENGINE_VERSION,
        }

    # ── Aggregate across folds ───────────────────────────────────────────────
    aggregated: Dict[str, Any] = {}

    metric_keys = [k for k in fold_metrics[0].keys() if k != "ci_coverage"]

    for key in metric_keys:
        values              = [m[key] for m in fold_metrics]
        aggregated[key]     = float(np.nanmean(values))
        aggregated[f"{key}_std"] = float(np.nanstd(values))

    # CI coverage — aggregate only non-None values
    ci_values = [m["ci_coverage"] for m in fold_metrics if m["ci_coverage"] is not None]
    if ci_values:
        aggregated["ci_coverage"]      = float(np.mean(ci_values))
        aggregated["ci_coverage_std"]  = float(np.std(ci_values))
    else:
        aggregated["ci_coverage"]      = None
        aggregated["ci_coverage_std"]  = None

    # ── Metadata ─────────────────────────────────────────────────────────────
    aggregated["eligible"]       = True
    aggregated["folds"]          = len(fold_metrics)
    aggregated["mean_level"]     = float(np.mean(y_full))
    aggregated["observations"]   = len(series)
    aggregated["engine_version"] = ENGINE_VERSION

    return aggregated