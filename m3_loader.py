# ==============================================================================
# FILE: m3_loader.py
# VERSION: 2.0.0
# ROLE: M3 MONTHLY DATASET LOADER — SHARED MODULE
# ENGINE: VEDUTA Foresight Engine v3.0.0
#
# CITATIONS:
#   Dataset: Makridakis, S. & Hibon, M. (2000).
#            'The M3-Competition: results, conclusions and implications.'
#            International Journal of Forecasting, 16(4), 451-476.
#            DOI: 10.1016/S0169-2070(00)00057-1
#
#   Metric (MASE): Hyndman, R.J. & Koehler, A.B. (2006).
#            'Another look at measures of forecast accuracy.'
#            International Journal of Forecasting, 22(4), 679-688.
#            DOI: 10.1016/j.ijforecast.2006.03.001
#
#   Metric (sMAPE): Makridakis, S. (1993).
#            'Accuracy measures: theoretical and practical concerns.'
#            International Journal of Forecasting, 9(4), 527-529.
#
# GOVERNANCE:
#   - No engine imports — pure data loading
#   - Data returned as-is from the M3 dataset (no imputation)
#   - Series with NaN in train split are flagged and skipped
#   - Timezone-naive dates enforced (engine requirement)
#   - Official M3 monthly horizon: h=18
#   - Official M3 monthly minimum observations: 48 total (30 train + 18 test)
# ==============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

M3_HORIZON   = 18    # Official M3 monthly forecast horizon (Makridakis & Hibon, 2000)
M3_FREQUENCY = "MS"  # Month-start frequency
M3_PERIOD    = 12    # Seasonal period for monthly MASE denominator
M3_MIN_OBS   = 30    # Minimum train observations (48 total - 18 test)


def load_m3_monthly(
    tsf_path:   str,
    max_series: Optional[int] = None,
    verbose:    bool = True,
) -> List[Dict[str, Any]]:
    """
    Load M3 monthly series from the TSF file.

    Applies the official M3 train/test split:
      train = all observations EXCEPT the last 18
      test  = the last 18 observations (the official holdout)

    Args:
        tsf_path:   Path to m3_monthly_dataset.tsf
        max_series: If set, return only the first N series (pilot/test mode)
        verbose:    Print loading progress

    Returns:
        List of dicts, each containing:
          series_id : str             — series identifier (e.g. 'T1')
          df        : pd.DataFrame    — training data (date, value), tz-naive
          actuals   : np.ndarray      — hold-out actuals (length = horizon)
          horizon   : int             — forecast horizon (18)
          n_train   : int             — training observations
    """
    path = Path(tsf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"M3 dataset not found: {tsf_path}\n"
            f"Expected: C:\\Dev\\VEDUTA\\_shared\\sample_data\\m3\\"
        )

    series_list = []
    skipped     = 0
    in_data     = False

    with open(tsf_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue

            parts = line.split(":")
            if len(parts) < 3:
                continue

            series_name = parts[0].strip()

            # Parse start date — strip timezone if present
            try:
                s = parts[1].strip()
                start = pd.Timestamp(
                    f"{s}-01-01" if len(s) == 4
                    else f"{s}-01" if len(s) == 7
                    else s
                )
                if start.tzinfo is not None:
                    start = start.tz_convert("UTC").tz_localize(None)
            except Exception:
                start = pd.Timestamp("1982-01-01")

            # Parse values
            try:
                vals = np.array(
                    [float(v.strip()) for v in parts[2].strip().split(",")
                     if v.strip()],
                    dtype=np.float64
                )
            except ValueError:
                skipped += 1
                continue

            # Must have enough for at least train + full horizon
            if len(vals) < M3_MIN_OBS + M3_HORIZON:
                skipped += 1
                continue

            # Official M3 train/test split
            train_vals = vals[:-M3_HORIZON]
            test_vals  = vals[-M3_HORIZON:]

            # Validate
            if not np.isfinite(train_vals).all():
                skipped += 1
                continue

            # Build training DataFrame with enforced monthly frequency
            dates = pd.date_range(
                start   = start,
                periods = len(train_vals),
                freq    = M3_FREQUENCY,
            )
            df = pd.DataFrame({
                "date":  dates,
                "value": train_vals,
            })

            series_list.append({
                "series_id": series_name,
                "df":        df,
                "actuals":   test_vals,
                "horizon":   M3_HORIZON,
                "n_train":   len(train_vals),
            })

            if max_series and len(series_list) >= max_series:
                break

    if verbose:
        print(f"  M3 monthly series loaded: {len(series_list)}  Skipped: {skipped}")

    return series_list


def stratified_sample(
    series_list: List[Dict],
    n:           int = 30,
    seed:        int = 42,
) -> List[Dict]:
    """
    Draw n series stratified by training length bucket.

    Buckets:
      Short  (n_train < 60):    represents series with ~48 total obs
      Medium (n_train 60-99):   represents mid-length series
      Long   (n_train >= 100):  represents series up to 144 total obs

    Fixed seed ensures reproducibility across runs.
    """
    rng    = np.random.default_rng(seed)
    short  = [s for s in series_list if s["n_train"] < 60]
    medium = [s for s in series_list if 60 <= s["n_train"] < 100]
    long_  = [s for s in series_list if s["n_train"] >= 100]

    per_bucket = n // 3
    remainder  = n - per_bucket * 3

    sampled = []
    for bucket, extra in zip([short, medium, long_], [remainder, 0, 0]):
        k   = min(per_bucket + extra, len(bucket))
        idx = rng.choice(len(bucket), size=k, replace=False)
        sampled.extend([bucket[i] for i in sorted(idx)])

    # Fill remainder from any bucket if a bucket was too small
    if len(sampled) < n:
        remaining = [s for s in series_list if s not in sampled]
        if remaining:
            fill = rng.choice(len(remaining),
                              size=min(n - len(sampled), len(remaining)),
                              replace=False)
            sampled.extend([remaining[i] for i in fill])

    return sampled[:n]


# ==============================================================================
# METRIC FUNCTIONS
# ==============================================================================

def compute_mase(
    forecast:        np.ndarray,
    actuals:         np.ndarray,
    train:           np.ndarray,
    seasonal_period: int = M3_PERIOD,
) -> float:
    """
    Seasonal MASE per Hyndman & Koehler (2006).

    Formula:
      MASE = MAE(forecast) / MAE(seasonal_naive)
      where MAE(seasonal_naive) = mean(|y_t - y_{t-m}|) for t = m+1..T

    A value < 1.0 means the forecast beats the seasonal naive baseline.
    Cap applied at 10.0 to prevent degenerate series from distorting median.
    """
    n   = min(len(forecast), len(actuals))
    mae = float(np.mean(np.abs(forecast[:n] - actuals[:n])))

    if len(train) > seasonal_period:
        scale = float(np.mean(
            np.abs(train[seasonal_period:] - train[:-seasonal_period])
        ))
    else:
        diff  = np.diff(train)
        scale = float(np.mean(np.abs(diff))) if len(diff) > 0 else 1.0

    scale = max(scale, 1e-8)
    mase  = mae / scale
    return min(float(mase), 10.0)


def compute_mase_raw(
    forecast:        np.ndarray,
    actuals:         np.ndarray,
    train:           np.ndarray,
    seasonal_period: int = M3_PERIOD,
) -> float:
    """Uncapped MASE — for audit records."""
    n   = min(len(forecast), len(actuals))
    mae = float(np.mean(np.abs(forecast[:n] - actuals[:n])))
    if len(train) > seasonal_period:
        scale = float(np.mean(
            np.abs(train[seasonal_period:] - train[:-seasonal_period])
        ))
    else:
        diff  = np.diff(train)
        scale = float(np.mean(np.abs(diff))) if len(diff) > 0 else 1.0
    return mae / max(scale, 1e-8)


def compute_smape(
    forecast: np.ndarray,
    actuals:  np.ndarray,
) -> float:
    """
    sMAPE per Makridakis (1993) — primary metric of original M3 competition.

    Formula: mean(200 * |actual - forecast| / (|actual| + |forecast|))
    Bounds:  [0%, 200%]

    Note: sMAPE has known asymmetry bias (Koehler, 2001) but is included
    for direct comparability with original M3 competition results.
    """
    n     = min(len(forecast), len(actuals))
    num   = np.abs(actuals[:n] - forecast[:n])
    denom = np.abs(actuals[:n]) + np.abs(forecast[:n])
    valid = denom > 1e-8
    if not valid.any():
        return float("nan")
    return float(np.mean(200.0 * num[valid] / denom[valid]))


def compute_mae(forecast: np.ndarray, actuals: np.ndarray) -> float:
    n = min(len(forecast), len(actuals))
    return float(np.mean(np.abs(forecast[:n] - actuals[:n])))


def compute_rmse(forecast: np.ndarray, actuals: np.ndarray) -> float:
    n = min(len(forecast), len(actuals))
    return float(np.sqrt(np.mean((forecast[:n] - actuals[:n]) ** 2)))
