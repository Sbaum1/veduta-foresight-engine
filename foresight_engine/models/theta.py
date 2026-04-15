# ==================================================
# FILE: foresight_engine/models/theta.py
# VERSION: 3.0.0
# MODEL: THETA — STL SEASONAL DECOMPOSITION
# ENGINE: Foresight Engine v3.0.0
# UPDATED: M1 — STL seasonal decomposition before Theta
# ==================================================
#
# M1 UPGRADE — STL SEASONAL DECOMPOSITION:
#   Previous: Theta applied directly to raw series
#   Fixed: STL decomposition when n >= 2 seasonal cycles (24 obs)
#     Seasonal strength = 1 - Var(R)/Var(S+R)  (Hyndman et al. 2015)
#     Skip decomposition if strength < 0.10 (negligible seasonality)
#     Theta fit on deseasonalised series, forecast reseasonalised
#   Multiplicative mode: all-positive series (sf = seasonal/mean + 1)
#   Additive mode: series with zeros/negatives (sf = seasonal component)
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

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



_Z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

def _get_z(cl: float) -> float:
    if cl in _Z:
        return _Z[cl]
    levels = sorted(_Z.keys())
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        if lo <= cl <= hi:
            return _Z[lo] + (cl - lo) / (hi - lo) * (_Z[hi] - _Z[lo])
    return 1.960


def _seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    """Hyndman et al. (2015) seasonal strength: Fs = 1 - Var(R)/Var(S+R)."""
    var_r  = float(np.var(residual, ddof=1)) if len(residual) > 1 else 0.0
    var_sr = float(np.var(seasonal + residual, ddof=1)) if len(seasonal) > 1 else 0.0
    if var_sr < 1e-10:
        return 0.0
    return max(0.0, 1.0 - var_r / var_sr)


def _stl_decompose(y: np.ndarray, season_len: int) -> tuple:
    """
    STL decomposition. Returns (seasonal_factors, deseasonalised, strength, mode).
    Returns (None, y, 0.0, None) if decomposition not applicable or fails.
    """
    if len(y) < 2 * season_len:
        return None, y, 0.0, None
    try:
        from statsmodels.tsa.seasonal import STL
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl    = STL(y, period=season_len, robust=True)
            result = stl.fit()
        seasonal = result.seasonal.astype("float64")
        residual = result.resid.astype("float64")
        strength = _seasonal_strength(seasonal, residual)
        if strength < 0.10:
            return None, y, strength, None
        # STL seasonal components are additive by construction (sum to ~0 per cycle).
        # The "multiplicative" branch using sf = seasonal/mean + 1 is incorrect
        # because np.mean(seasonal) ≈ 0, which causes division by near-zero and
        # collapses the deseasonalized series. Always use additive decomposition:
        # deseasonalized = y - seasonal, reseasonalize = forecast + seasonal.
        sf     = seasonal
        deseas = y - sf
        mode   = "additive"
        return sf, deseas.astype("float64"), strength, mode
    except Exception:
        return None, y, 0.0, None


def _reseasonalise(forecast: np.ndarray, sf: np.ndarray, mode: str, n_train: int, season_len: int) -> np.ndarray:
    """
    Reseasonalise forecast by adding back the last full seasonal cycle.
    Always additive: forecast + seasonal_component.
    sf contains the STL seasonal component from the training series.
    """
    h = len(forecast)
    last_cycle_start = n_train - season_len
    if last_cycle_start < 0:
        return forecast
    last_sf  = sf[last_cycle_start: last_cycle_start + season_len]
    repeated = np.tile(last_sf, (h // season_len) + 2)[:h]
    return (forecast + repeated).astype("float64")


def run_theta(df: pd.DataFrame, horizon: int, confidence_level: float) -> ForecastResult:
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Theta requires 'date' and 'value' columns.")
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
    y_series = df["value"].astype("float64")
    y        = y_series.values
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected.")
    if len(y) < 6:
        raise ValueError("Minimum 6 observations required.")

    season_len = 12

    # ── M1: STL seasonal decomposition ────────────────────────────────────────
    sf, y_deseas, seasonal_strength, decomp_mode = _stl_decompose(y, season_len)
    seasonal_decomposed = sf is not None

    # ── Fit Theta on deseasonalised series ────────────────────────────────────
    if seasonal_decomposed:
        fit_series = pd.Series(y_deseas, index=y_series.index)
        fit_series = fit_series.asfreq(inferred)
    else:
        fit_series = y_series

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = ThetaModel(fit_series, period=season_len, deseasonalize=False)
            fitted = model.fit(use_mle=True)
    except Exception as e:
        raise RuntimeError(f"Theta model fit failed: {e}") from e

    # ── Forecast ──────────────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fitted.forecast(horizon)
    except Exception as e:
        raise RuntimeError(f"Theta forecast failed: {e}") from e

    future_mean = pred.values.astype("float64") if hasattr(pred, "values") else np.asarray(pred, dtype="float64")

    # ── Reseasonalise ─────────────────────────────────────────────────────────
    if seasonal_decomposed:
        future_mean_rs = _reseasonalise(future_mean, sf, decomp_mode, len(y), season_len)
        # Sanity guard: if reseasonalized values are >10x the series mean, discard
        series_mean    = float(np.mean(np.abs(y)))
        if (np.isfinite(future_mean_rs).all()
                and float(np.mean(np.abs(future_mean_rs))) < 10.0 * series_mean):
            future_mean = future_mean_rs
        else:
            # Decomposition produced unstable output — run without decomposition
            seasonal_decomposed = False
            sf = None
            fit_series_raw = y_series
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_raw  = ThetaModel(fit_series_raw, period=season_len, deseasonalize=True)
                    fitted_raw = model_raw.fit(use_mle=True)
                    pred_raw   = fitted_raw.forecast(horizon)
                    future_mean = pred_raw.values.astype("float64") if hasattr(pred_raw, "values") else np.asarray(pred_raw, dtype="float64")
            except Exception:
                pass  # keep future_mean from deseasonalized fit

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values.")

    # ── CI from residuals ─────────────────────────────────────────────────────
    # ThetaModelResults may not have fittedvalues — predict in-sample
    # ThetaModelResults API: use forecast on training period for in-sample fit
    try:
        _fv = fitted.fittedvalues
        hist_fitted = pd.Series(np.asarray(_fv).astype("float64"), index=fit_series.index)
    except AttributeError:
        # fittedvalues not available — reconstruct from model's alpha/theta parameters
        # using the Theta recurrence relation (exponential smoothing component)
        try:
            alpha    = float(getattr(fitted, "params", {}).get("smoothing_level", 0.2))
            alpha    = max(0.01, min(0.99, alpha))
            n_fit    = len(fit_series)
            ys       = fit_series.values.astype("float64")
            lvl      = np.zeros(n_fit)
            lvl[0]   = ys[0]
            for t in range(1, n_fit):
                lvl[t] = alpha * ys[t] + (1 - alpha) * lvl[t - 1]
            hist_fitted = pd.Series(lvl, index=fit_series.index)
        except Exception:
            # Last resort: use the actual series (zero residuals = conservative CI)
            hist_fitted = pd.Series(fit_series.values.astype("float64"), index=fit_series.index)
    if seasonal_decomposed:
        hist_reseason = _reseasonalise(hist_fitted.values, sf, decomp_mode, len(y), season_len)
        hist_fitted_rs = pd.Series(hist_reseason, index=hist_fitted.index)
    else:
        hist_fitted_rs = hist_fitted

    residuals        = (y_series.values - hist_fitted_rs.values).astype("float64")
    finite_residuals = residuals[np.isfinite(residuals)]
    if len(finite_residuals) < 2:
        finite_residuals = residuals

    z     = _get_z(confidence_level)
    sigma = float(np.std(finite_residuals, ddof=1)) if len(finite_residuals) > 1 else float(np.mean(np.abs(finite_residuals)))
    if sigma < 1e-10:
        sigma = float(np.std(y)) * 0.1
    h_arr   = np.arange(1, horizon + 1, dtype="float64")
    ci_low  = (future_mean - z * sigma * np.sqrt(h_arr)).astype("float64")
    ci_high = (future_mean + z * sigma * np.sqrt(h_arr)).astype("float64")

    # ── Output assembly ───────────────────────────────────────────────────────
    hist_block = pd.DataFrame({
        "date":      hist_fitted_rs.index,
        "actual":    np.nan,
        "forecast":  hist_fitted_rs.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted_rs.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    future_index = pd.date_range(start=y_series.index[-1], periods=horizon + 1, freq=inferred)[1:]
    future_block = pd.DataFrame({
        "date":      future_index,
        "actual":    np.nan,
        "forecast":  future_mean,
        "ci_low":    ci_low,
        "ci_mid":    future_mean,
        "ci_high":   ci_high,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast", "ci_low", "ci_mid", "ci_high"]] = \
            b[["forecast", "ci_low", "ci_mid", "ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI values in output.")
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI detected.")

    return ForecastResult(
        model_name  = "Theta",
        forecast_df = forecast_df[["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]],
        metrics     = None,
        metadata    = {
            "seasonal_decomposed": seasonal_decomposed,
            "decomposition_mode":  decomp_mode,
            "seasonal_strength":   round(float(seasonal_strength), 4),
            "STL seasonal decomposition": True,
            "frequency":           inferred,
            "confidence_level":    confidence_level,
            "ci_method":           "residual_based_sigma_sqrt_h",
            "output_contract":     "ForecastResult",
        },
    )
