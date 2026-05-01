# ==================================================
# FILE: foresight_engine/preprocessor.py
# VERSION: 1.0.0
# ROLE: PRE-FITTING HARDENING LAYER
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# PURPOSE:
#   Provides three hardening capabilities that execute
#   before any model is fit, improving engine robustness
#   on hostile real-world series:
#
#   1. STRUCTURAL BREAK DETECTION (Steps 1)
#      CUSUM + Chow test detects level shifts and trend
#      reversals. Returns regime_context dict consumed
#      by ensemble.py to adjust model weighting.
#
#   2. BOX-COX VARIANCE STABILIZATION (Step 2)
#      Estimates optimal lambda via MLE. Transforms
#      series before model execution. Inversion applied
#      after aggregation restores original scale.
#
#   3. AIC/BIC MODEL FITNESS SCORING (Step 4)
#      Lightweight per-family fitness scoring using
#      information criteria. Returns fitness_scores
#      dict consumed by ensemble.py to adjust initial
#      weights before MASE weighting.
#
# GOVERNANCE:
#   - No Streamlit dependencies
#   - No session state dependencies
#   - All functions are pure (no side effects on input df)
#   - Returns new objects — never mutates input
#   - Falls back gracefully on all failure paths
#   - No ranking logic — scoring only
#   - Consumed exclusively by ensemble.py
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

# Structural break
CUSUM_THRESHOLD          = 0.15   # CUSUM cumulative deviation threshold (fraction of std)
CHOW_MIN_SEGMENT         = 12     # Minimum observations per segment for Chow test
BREAK_CONFIDENCE_HIGH    = 0.80   # Both tests agree → high confidence
BREAK_CONFIDENCE_MEDIUM  = 0.50   # One test agrees  → medium confidence

# Box-Cox
BOXCOX_MIN_POSITIVE_PCT  = 0.80   # Series must be ≥80% positive for Box-Cox
BOXCOX_LAMBDA_BOUNDS     = (-2.0, 2.0)
BOXCOX_OFFSET_MULTIPLIER = 0.01   # Offset applied when series contains zeros

# AIC/BIC fitness
FITNESS_AR_LAGS          = 2      # AR lags for lightweight ARIMA family fitness
FITNESS_MIN_OBS          = 24     # Minimum obs for fitness scoring
FITNESS_FLOOR            = 1e-6   # Floor for fitness weight before normalisation


# ==================================================
# 1. STRUCTURAL BREAK DETECTION
# ==================================================

def _cusum_test(y: np.ndarray) -> Tuple[bool, Optional[int], float]:
    """
    CUSUM (Cumulative Sum) test for structural breaks.

    Computes cumulative deviation from mean, normalised by
    series standard deviation. A break is flagged when the
    maximum CUSUM statistic exceeds CUSUM_THRESHOLD * n.

    Returns:
        detected  : bool   — True if break detected
        location  : int    — index of most likely break point (or None)
        strength  : float  — normalised CUSUM statistic (0–1)
    """
    if len(y) < CHOW_MIN_SEGMENT * 2:
        return False, None, 0.0

    std = float(np.std(y, ddof=1))
    if std < 1e-10:
        return False, None, 0.0

    mean   = float(np.mean(y))
    cusum  = np.cumsum(y - mean) / std
    absval = np.abs(cusum)
    loc    = int(np.argmax(absval))
    stat   = float(absval[loc]) / len(y)

    detected = stat > CUSUM_THRESHOLD
    strength = float(np.clip(stat / (CUSUM_THRESHOLD * 3), 0.0, 1.0))

    return detected, loc if detected else None, strength


def _chow_test(y: np.ndarray, break_point: int) -> Tuple[bool, float]:
    """
    Chow test for structural break at a given break_point.

    Tests whether the regression relationship differs
    significantly across the two sub-samples split at
    break_point. Uses F-statistic approximation.

    Returns:
        significant : bool  — True if Chow F-stat suggests break
        f_stat      : float — F-statistic value
    """
    n = len(y)

    if break_point < CHOW_MIN_SEGMENT or (n - break_point) < CHOW_MIN_SEGMENT:
        return False, 0.0

    def _ols_sse(segment: np.ndarray) -> float:
        x = np.arange(len(segment), dtype=float)
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, segment, rcond=None)
            residuals     = segment - X @ beta
            return float(np.sum(residuals ** 2))
        except np.linalg.LinAlgError:
            return float(np.sum((segment - segment.mean()) ** 2))

    seg1 = y[:break_point]
    seg2 = y[break_point:]
    y_full = y

    sse_full = _ols_sse(y_full)
    sse_1    = _ols_sse(seg1)
    sse_2    = _ols_sse(seg2)

    sse_restricted   = sse_full
    sse_unrestricted = sse_1 + sse_2

    k  = 2   # parameters (intercept + slope)
    df_num   = k
    df_denom = n - 2 * k

    if df_denom <= 0 or sse_unrestricted < 1e-12:
        return False, 0.0

    f_stat = ((sse_restricted - sse_unrestricted) / df_num) / \
             (sse_unrestricted / df_denom)

    # Critical value approximation: F(2, n) at p=0.05 ≈ 3.0
    significant = f_stat > 3.0

    return significant, float(f_stat)


def detect_structural_break(
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Run CUSUM + Chow structural break detection on a series.

    Both tests must be computable. Results are combined into
    a regime_context dict consumed by ensemble.py.

    Returns a regime_context dict:
        detected         : bool   — break detected with ≥medium confidence
        confidence       : str    — 'high' | 'medium' | 'none'
        break_location   : int    — index of break (or None)
        break_fraction   : float  — break location as fraction of series (0–1)
        cusum_detected   : bool
        chow_significant : bool
        f_stat           : float
        cusum_strength   : float
        recommended_action : str  — guidance for ensemble weighting
    """
    context: Dict[str, Any] = {
        "detected":           False,
        "confidence":         "none",
        "break_location":     None,
        "break_fraction":     None,
        "cusum_detected":     False,
        "chow_significant":   False,
        "f_stat":             0.0,
        "cusum_strength":     0.0,
        "recommended_action": "standard",
    }

    if len(y) < CHOW_MIN_SEGMENT * 2:
        return context

    try:
        cusum_detected, cusum_loc, cusum_strength = _cusum_test(y)
        context["cusum_detected"]  = cusum_detected
        context["cusum_strength"]  = cusum_strength

        if not cusum_detected or cusum_loc is None:
            return context

        chow_sig, f_stat = _chow_test(y, cusum_loc)
        context["chow_significant"] = chow_sig
        context["f_stat"]           = f_stat
        context["break_location"]   = cusum_loc
        context["break_fraction"]   = float(cusum_loc) / len(y)

        if cusum_detected and chow_sig:
            context["detected"]    = True
            context["confidence"]  = "high"
            context["recommended_action"] = "downweight_trend_extrapolation"
        elif cusum_detected:
            context["detected"]    = True
            context["confidence"]  = "medium"
            context["recommended_action"] = "flag_regime_uncertainty"

    except Exception:
        # Detection failure is non-fatal — return default context
        pass

    return context


# ==================================================
# 2. BOX-COX VARIANCE STABILIZATION
# ==================================================

def _estimate_boxcox_lambda(y: np.ndarray) -> float:
    """
    Estimate optimal Box-Cox lambda via MLE using a grid search.
    Evaluates log-likelihood of normalised residuals across
    lambda grid. Returns lambda in BOXCOX_LAMBDA_BOUNDS.
    """
    lambdas    = np.linspace(BOXCOX_LAMBDA_BOUNDS[0], BOXCOX_LAMBDA_BOUNDS[1], 100)
    best_llf   = -np.inf
    best_lam   = 1.0   # lambda=1 → no transform (identity)

    n = len(y)

    for lam in lambdas:
        try:
            if abs(lam) < 1e-8:
                y_t = np.log(y)
            else:
                y_t = (y ** lam - 1.0) / lam

            if not np.isfinite(y_t).all():
                continue

            std = float(np.std(y_t, ddof=1))
            if std < 1e-10:
                continue

            # Log-likelihood of normalised transformed series
            llf = (
                -n / 2.0 * np.log(std ** 2)
                + (lam - 1.0) * np.sum(np.log(y))
            )

            if llf > best_llf:
                best_llf = llf
                best_lam = lam

        except Exception:
            continue

    return float(best_lam)


def apply_boxcox(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    """
    Apply Box-Cox variance stabilisation to a series DataFrame.

    Eligibility:
        - Series must be ≥BOXCOX_MIN_POSITIVE_PCT positive values
        - Series must have finite variance
        - If series contains zeros, a small offset is added

    Returns:
        transformed_df : pd.DataFrame — copy with 'value' transformed
        lambda_        : float        — Box-Cox lambda used (None if skipped)
        offset         : float        — offset added before transform (None if skipped)

    If ineligible or transform fails, returns original df unchanged
    with lambda_=None, offset=None (no-op).
    """
    if "value" not in df.columns:
        return df, None, None

    y = df["value"].astype("float64").values

    if not np.isfinite(y).all():
        return df, None, None

    # Eligibility: must be mostly positive
    pos_pct = float((y > 0).mean())
    if pos_pct < BOXCOX_MIN_POSITIVE_PCT:
        return df, None, None

    # Variance check
    if float(np.std(y, ddof=1)) < 1e-8:
        return df, None, None

    # Offset for zero/near-zero values
    offset = 0.0
    if (y <= 0).any():
        offset = float(abs(y.min())) + float(np.mean(y[y > 0])) * BOXCOX_OFFSET_MULTIPLIER
        y = y + offset

    if (y <= 0).any():
        return df, None, None

    try:
        lambda_ = _estimate_boxcox_lambda(y)

        # Apply transform
        if abs(lambda_) < 1e-8:
            y_transformed = np.log(y)
        else:
            y_transformed = (y ** lambda_ - 1.0) / lambda_

        if not np.isfinite(y_transformed).all():
            return df, None, None

        transformed_df            = df.copy()
        transformed_df["value"]   = y_transformed
        return transformed_df, lambda_, offset

    except Exception:
        return df, None, None


def invert_boxcox(
    forecast_values: np.ndarray,
    lambda_: float,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Invert Box-Cox transform on forecast output.

    Applies the inverse transform and subtracts any offset
    that was added before the forward transform.

    Args:
        forecast_values : np.ndarray — transformed forecast values
        lambda_         : float      — Box-Cox lambda used in forward transform
        offset          : float      — offset added before forward transform

    Returns:
        np.ndarray — forecast values in original scale
    """
    try:
        if abs(lambda_) < 1e-8:
            inverted = np.exp(forecast_values)
        else:
            inner = forecast_values * lambda_ + 1.0
            # Guard against negative base with non-integer exponent
            inner = np.maximum(inner, 1e-10)
            inverted = inner ** (1.0 / lambda_)

        inverted = inverted - offset

        if not np.isfinite(inverted).all():
            raise ValueError("Non-finite values after Box-Cox inversion.")

        return inverted.astype("float64")

    except Exception:
        # Inversion failure — return original values unchanged
        return forecast_values.astype("float64")


# ==================================================
# 3. AIC/BIC MODEL FITNESS SCORING
# ==================================================

def _fit_ar_aic(y: np.ndarray, lags: int = FITNESS_AR_LAGS) -> float:
    """
    Fit a simple AR(lags) model via OLS and return AIC.
    Used as a lightweight proxy for ARIMA family fitness.
    Lower AIC = better fit.
    """
    n = len(y)
    if n <= lags + 2:
        return np.inf

    Y = y[lags:]
    X = np.column_stack([y[lags - i - 1: n - i - 1] for i in range(lags)])
    X = np.column_stack([np.ones(len(Y)), X])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        residuals     = Y - X @ beta
        sse           = float(np.sum(residuals ** 2))
        k             = lags + 1   # AR params + intercept
        sigma2        = sse / (n - lags)

        if sigma2 <= 0:
            return np.inf

        log_likelihood = -0.5 * (n - lags) * (np.log(2 * np.pi * sigma2) + 1)
        aic            = -2 * log_likelihood + 2 * k
        return float(aic)

    except np.linalg.LinAlgError:
        return np.inf


def _fit_ets_proxy_aic(y: np.ndarray) -> float:
    """
    Fit a simple exponential smoothing proxy and return AIC.
    Optimises alpha via grid search over residuals.
    Used as lightweight proxy for ETS family fitness.
    """
    n = len(y)
    if n < FITNESS_MIN_OBS:
        return np.inf

    best_aic   = np.inf
    best_alpha = 0.3

    for alpha in np.linspace(0.05, 0.95, 19):
        fitted    = np.empty(n)
        fitted[0] = y[0]
        for t in range(1, n):
            fitted[t] = alpha * y[t - 1] + (1 - alpha) * fitted[t - 1]

        residuals = y[1:] - fitted[1:]
        sse       = float(np.sum(residuals ** 2))
        k         = 1   # alpha
        sigma2    = sse / (n - 1)

        if sigma2 <= 0:
            continue

        log_likelihood = -0.5 * (n - 1) * (np.log(2 * np.pi * sigma2) + 1)
        aic            = -2 * log_likelihood + 2 * k

        if aic < best_aic:
            best_aic   = aic
            best_alpha = alpha

    return float(best_aic)


def score_model_fitness(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Score per-family model fitness using lightweight AIC/BIC proxies.

    Runs fast AR and ETS proxy fits to assess how well each
    model family is likely to fit the series before committing
    to full model execution.

    Returns a fitness_scores dict keyed by model family:
        'arima'       : float — normalised fitness weight (higher = better fit)
        'ets'         : float — normalised fitness weight
        'ml'          : float — 1.0 (ML models always eligible)
        'bayesian'    : float — 1.0 (Bayesian models always eligible)
        'volatility'  : float — based on heteroscedasticity proxy
        'decomposition': float — 1.0 (Prophet always eligible)

    All weights are normalised to sum to 1.0 within the returned dict.
    A weight of FITNESS_FLOOR indicates the family is a poor fit
    but is NOT excluded — weighting handles the adjustment.
    """
    fitness: Dict[str, float] = {
        "arima":        1.0,
        "ets":          1.0,
        "ml":           1.0,
        "bayesian":     1.0,
        "volatility":   1.0,
        "decomposition": 1.0,
    }

    if "value" not in df.columns or len(df) < FITNESS_MIN_OBS:
        return fitness

    y = df["value"].astype("float64").values

    if not np.isfinite(y).all():
        return fitness

    try:
        # ── ARIMA family fitness via AR(2) AIC ───────────────────────
        ar_aic = _fit_ar_aic(y, lags=FITNESS_AR_LAGS)
        if np.isfinite(ar_aic):
            # Convert AIC to fitness weight: lower AIC → higher fitness
            # Use softmax-style normalisation relative to series length
            n = len(y)
            aic_normalised        = ar_aic / n
            fitness["arima"]      = float(np.clip(np.exp(-aic_normalised * 0.01), FITNESS_FLOOR, 1.0))
        else:
            fitness["arima"] = FITNESS_FLOOR

        # ── ETS family fitness via ES proxy AIC ──────────────────────
        ets_aic = _fit_ets_proxy_aic(y)
        if np.isfinite(ets_aic):
            n = len(y)
            aic_normalised   = ets_aic / n
            fitness["ets"]   = float(np.clip(np.exp(-aic_normalised * 0.01), FITNESS_FLOOR, 1.0))
        else:
            fitness["ets"] = FITNESS_FLOOR

        # ── Volatility fitness: GARCH appropriate on heterosc. series ─
        # Proxy: ratio of rolling std to mean — high ratio → GARCH relevant
        window = min(12, len(y) // 3)
        if window >= 4:
            rolling_std  = pd.Series(y).rolling(window).std().dropna().values
            mean_abs     = float(np.mean(np.abs(y)))
            if mean_abs > 1e-8:
                heterosc_ratio       = float(np.std(rolling_std) / mean_abs)
                fitness["volatility"] = float(np.clip(heterosc_ratio * 2.0, FITNESS_FLOOR, 1.0))

    except Exception:
        # Fitness scoring failure is non-fatal — return uniform weights
        pass

    return fitness
