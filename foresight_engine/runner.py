# ==================================================
# FILE: foresight_engine/runner.py
# VERSION: 3.0.0
# ROLE: MODEL EXECUTION ORCHESTRATOR
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
#
# v3.0.0 CHANGES:
#   - Preprocessor wired in: structural break detection and model
#     fitness scoring run before model execution. Results passed
#     to ensemble for weight adjustment and logged in _engine metadata.
#   - Package renamed sentinel_engine → foresight_engine
#   - Config renamed SentinelConfig → ForesightConfig
# ==================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .contracts        import ForecastResult, ENGINE_VERSION
from .registry         import get_model_registry, get_ensemble_members, get_models_by_tier, get_ensemble_members_by_tier
from .backtest         import run_backtest as _default_backtest
from .ensemble         import run_primary_ensemble
from .foresight_config import get_config
from .stacker          import build_stacked_forecast
from .preprocessor     import detect_structural_break, score_model_fitness

MIN_OBSERVATIONS = 36
MASE_FLOOR       = 1e-6
MASE_CAP         = 10.0


def _build_failure_record(model_name: str, error: Exception) -> dict:
    return {
        "model_name":    model_name,
        "error_type":    type(error).__name__,
        "error_message": str(error),
    }


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Input dataframe must contain 'date' and 'value'.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred not in ("MS", "M"):
        raise ValueError(f"Monthly frequency required. Inferred: '{inferred}'.")

    df = df.set_index("date").asfreq(inferred).reset_index()

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    return df


def _inject_actuals(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    df   = forecast_df.copy()
    if "actual" not in df.columns:
        df["actual"] = pd.NA
    hist   = historical_df[["date", "value"]].rename(columns={"value": "actual_hist"})
    merged = pd.merge(df, hist, on="date", how="left")
    merged["actual"] = merged["actual"].combine_first(merged["actual_hist"])
    merged = merged.drop(columns=["actual_hist"])
    if merged["date"].duplicated().any():
        raise ValueError("Duplicate dates after actual injection.")
    return merged


def _validate_forward_boundary(forecast_df: pd.DataFrame) -> None:
    df     = forecast_df.sort_values("date")
    hist   = df[df["actual"].notna()]
    future = df[df["actual"].isna()]
    if not hist.empty and not future.empty:
        # Normalise to datetime64[ns] before comparison — pandas 2.x raises
        # TypeError when comparing datetime64[us] against Timestamp directly.
        future_min = pd.to_datetime(future["date"].min()).to_datetime64()
        hist_max   = pd.to_datetime(hist["date"].max()).to_datetime64()
        if future_min <= hist_max:
            raise ValueError("Forecast overlaps historical period — lookahead detected.")


def _normalize_metric_keys(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}
    mapping = {
        "mae":                      "MAE",
        "rmse":                     "RMSE",
        "mape":                     "MAPE",
        "bias":                     "Bias",
        "mase":                     "MASE",
        "theils_u":                 "Theils_U",
        "ci_coverage":              "CI_Coverage",
        "smape":                    "SMAPE",
        "folds":                    "Folds",
        "observations":             "Observations",
        "mean_level":               "Mean_Level",
        "directional_accuracy":     "Directional_Accuracy",
        "mae_std":                  "MAE_Std",
        "rmse_std":                 "RMSE_Std",
        "mape_std":                 "MAPE_Std",
        "bias_std":                 "Bias_Std",
        "mase_std":                 "MASE_Std",
        "theils_u_std":             "Theils_U_Std",
        "ci_coverage_std":          "CI_Coverage_Std",
        "smape_std":                "SMAPE_Std",
        "directional_accuracy_std": "Directional_Accuracy_Std",
    }
    return {mapping.get(k, k): v for k, v in metrics.items()}


def _assign_readiness(metrics: dict, confidence_level: float) -> str:
    if metrics.get("eligible") is False:
        return "Ineligible — Minimum Data Not Met"
    mase       = metrics.get("MASE")
    theils_u   = metrics.get("Theils_U")
    coverage   = metrics.get("CI_Coverage")
    bias       = metrics.get("Bias")
    folds      = metrics.get("Folds")
    mean_level = metrics.get("Mean_Level")
    if mase is None or theils_u is None:
        return "Unscored"
    if coverage is not None and coverage < 0.50:
        return "Tier 4 — Structural Failure"
    bias_ok = True
    if bias is not None and mean_level not in (None, 0):
        if abs(bias) / abs(mean_level) > 0.02:
            bias_ok = False
    if folds is not None and int(folds) < 3:
        return "Tier 3 — Weak (Limited Fold Validation)"
    if (mase < 0.8 and theils_u < 1.0 and coverage is not None
            and abs(coverage - confidence_level) <= 0.05 and bias_ok):
        return "Tier 1 — Production Ready"
    if mase < 1.0:
        return "Tier 2 — Acceptable"
    return "Tier 3 — Weak"


def _assign_confidence(metrics: dict, confidence_level: float) -> dict:
    if metrics.get("eligible") is False:
        return {
            "confidence_posture": "Not Eligible",
            "risk_flags":         ["Minimum data threshold not met"],
            "decision_guidance":  "Increase historical data before executive use.",
        }
    mase       = metrics.get("MASE")
    theils_u   = metrics.get("Theils_U")
    coverage   = metrics.get("CI_Coverage")
    bias       = metrics.get("Bias")
    mean_level = metrics.get("Mean_Level")
    risk_flags: List[str] = []
    if mase is not None and mase > 1.0:
        risk_flags.append("Error exceeds naïve baseline")
    if coverage is not None:
        deviation = abs(coverage - confidence_level)
        if deviation > 0.10:
            risk_flags.append("Confidence interval materially miscalibrated")
        elif deviation > 0.05:
            risk_flags.append("Minor confidence interval drift")
        if coverage < 0.50:
            risk_flags.append("Confidence band structural failure")
    if bias is not None and mean_level not in (None, 0):
        if abs(bias) / abs(mean_level) > 0.02:
            risk_flags.append("Structural forecast bias (>2%)")
    if mase is not None and theils_u is not None:
        if mase < 0.8 and theils_u < 1.0 and not risk_flags:
            posture  = "High Confidence — Production Safe"
            guidance = "Model suitable for executive planning use."
        elif mase < 1.0:
            posture  = "Moderate Confidence — Monitor"
            guidance = "Model acceptable but monitor volatility and calibration."
        else:
            posture  = "Low Confidence — Elevated Risk"
            guidance = "Exercise caution in executive decisions."
    else:
        posture  = "Unscored"
        guidance = "Metrics incomplete."
    return {
        "confidence_posture": posture,
        "risk_flags":         risk_flags,
        "decision_guidance":  guidance,
    }


def apply_stress(results: Dict[str, Any], stress_pct: float = 0.15) -> Dict[str, Any]:
    """Stress-test results by widening CIs. Operates on copy — never mutates original."""
    stressed = {}
    for name, result in results.items():
        if name.startswith("_") or not isinstance(result, dict):
            stressed[name] = result
            continue
        if result.get("status") != "success":
            stressed[name] = result
            continue
        df_out = result.get("forecast_df")
        if df_out is None or df_out.empty:
            stressed[name] = result
            continue
        df_stressed            = df_out.copy()
        width                  = df_stressed["ci_high"] - df_stressed["ci_low"]
        shock                  = width * stress_pct
        df_stressed["ci_low"]  = df_stressed["ci_low"]  - shock
        df_stressed["ci_high"] = df_stressed["ci_high"] + shock
        stressed[name] = {**result, "forecast_df": df_stressed}
    return stressed


def _build_ensemble_weights(
    model_mase_scores: Dict[str, Optional[float]],
    ensemble_member_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Build inverse-MASE normalized weights for ensemble members.
    G3: Median-fallback for missing MASE scores.
    """
    valid_raw: Dict[str, float] = {}
    missing:   List[str]        = []

    for name in ensemble_member_names:
        mase = model_mase_scores.get(name)
        if mase is not None and np.isfinite(mase) and mase > 0:
            valid_raw[name] = 1.0 / float(np.clip(mase, MASE_FLOOR, MASE_CAP))
        else:
            missing.append(name)

    if not valid_raw:
        return None

    if not missing:
        total = sum(valid_raw.values())
        return {n: w / total for n, w in valid_raw.items()}

    median_inv_mase = float(np.median(list(valid_raw.values())))
    raw: Dict[str, float] = dict(valid_raw)
    for name in missing:
        raw[name] = median_inv_mase

    total = sum(raw.values())
    if total <= 0:
        return None

    weights = {n: w / total for n, w in raw.items()}
    weights["_median_fallback_models"] = missing  # type: ignore[assignment]
    return weights


# ==================================================
# MAIN RUNNER
# ==================================================

def run_all_models(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
    backtest_fn:      Optional[Any] = None,
    diagnostics_fn:   Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute all registered models and return scored results.

    v3.0.0: Preprocessor is now wired in. Before model execution,
    detect_structural_break() and score_model_fitness() are called.
    Results are passed to the ensemble for weight adjustment and
    logged in the _engine metadata block.
    """

    config       = get_config()
    active_tier  = config.ACTIVE_TIER

    if backtest_fn is None:
        backtest_fn = _default_backtest if config.BACKTEST_ENABLED else None

    if diagnostics_fn is None:
        diagnostics_fn = lambda forecast_df, metrics: {}

    results:  Dict[str, Any] = {}
    failures: List[dict]     = []

    hist_df           = _validate_input(df)
    # Normalize dates to timezone-naive — engine requires tz-naive datetime64.
    # External datasets (M3, FRED) may carry UTC timezone info after parsing.
    if hist_df["date"].dt.tz is not None:
        hist_df["date"] = hist_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    observation_count = len(hist_df)

    # ── Preprocessor: structural break + fitness scoring ─────────────────────
    regime_context: Dict[str, Any] = {}
    fitness_scores: Dict[str, float] = {}

    if config.PREPROCESSOR_ENABLED:
        try:
            regime_context = detect_structural_break(hist_df["value"].values)
        except Exception as e:
            regime_context = {"detected": False, "error": str(e)}
        try:
            fitness_scores = score_model_fitness(hist_df)
        except Exception as e:
            fitness_scores = {}

    # ── Tier-filtered registry ────────────────────────────────────────────────
    tier_registry         = get_models_by_tier(active_tier)
    tier_ensemble_entries = get_ensemble_members_by_tier(active_tier)
    ensemble_member_names = [e["name"] for e in tier_ensemble_entries]

    model_mase_scores: Dict[str, Optional[float]] = {}

    for model_meta in tier_registry:
        name            = model_meta["name"]
        runner          = model_meta["runner"]
        diagnostic_only = model_meta.get("diagnostic_only", False)

        if name == "Primary Ensemble":
            continue

        try:
            output = runner(
                df               = hist_df,
                horizon          = horizon,
                confidence_level = confidence_level,
            )

            if not isinstance(output, ForecastResult):
                raise TypeError(
                    f"{name} did not return ForecastResult. Got: {type(output).__name__}"
                )

            forecast_df         = output.forecast_df.copy()
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            if diagnostic_only:
                results[name] = {
                    "status":               "success",
                    "forecast_df":          forecast_df,
                    "metrics":              {},
                    "diagnostics":          {},
                    "executive_assessment": {},
                    "metadata":             output.metadata or {},
                    "diagnostic_only":      True,
                }
                continue

            forecast_df = _inject_actuals(forecast_df, hist_df)
            _validate_forward_boundary(forecast_df)

            if observation_count < MIN_OBSERVATIONS:
                raw_metrics = {
                    "eligible":     False,
                    "reason":       f"Minimum {MIN_OBSERVATIONS} observations required.",
                    "observations": observation_count,
                }
            elif not config.BACKTEST_ENABLED or backtest_fn is None:
                raw_metrics = {
                    "eligible":         True,
                    "observations":     observation_count,
                    "backtest_skipped": True,
                    "reason":           "BACKTEST_ENABLED=False in ForesightConfig",
                }
            else:
                raw_metrics = backtest_fn(
                    df               = hist_df,
                    model_runner     = runner,
                    horizon          = horizon,
                    confidence_level = confidence_level,
                )
                if not isinstance(raw_metrics, dict):
                    raise ValueError(f"Backtest returned {type(raw_metrics).__name__}.")
                raw_metrics["eligible"]     = True
                raw_metrics["observations"] = observation_count

            normalized_metrics = _normalize_metric_keys(raw_metrics)

            if name in ensemble_member_names:
                model_mase_scores[name] = normalized_metrics.get("MASE")

            readiness         = _assign_readiness(normalized_metrics, confidence_level)
            confidence_bundle = _assign_confidence(normalized_metrics, confidence_level)
            diagnostics       = diagnostics_fn(
                forecast_df = forecast_df,
                metrics     = normalized_metrics,
            )

            metadata                     = dict(output.metadata or {})
            metadata["confidence_level"] = confidence_level
            metadata["engine_version"]   = ENGINE_VERSION

            results[name] = {
                "status":      "success",
                "forecast_df": forecast_df,
                "metrics":     normalized_metrics,
                "diagnostics": diagnostics,
                "executive_assessment": {
                    "readiness_tier": readiness,
                    **confidence_bundle,
                },
                "metadata":        metadata,
                "diagnostic_only": False,
            }

        except Exception as e:
            failure_record = _build_failure_record(name, e)
            failures.append(failure_record)
            results[name] = {
                "status":         "failed",
                "error":          str(e),
                "exception_type": type(e).__name__,
                "error_message":  str(e),
            }
            if name in ensemble_member_names:
                model_mase_scores[name] = None

    # ── Build pre-computed weights ────────────────────────────────────────────
    pre_computed_weights = _build_ensemble_weights(
        model_mase_scores     = model_mase_scores,
        ensemble_member_names = ensemble_member_names,
    )

    # ── Run Primary Ensemble ──────────────────────────────────────────────────
    try:
        ensemble_result = run_primary_ensemble(
            df                   = hist_df,
            horizon              = horizon,
            confidence_level     = confidence_level,
            pre_computed_weights = pre_computed_weights,
            active_tier          = active_tier,
            regime_context       = regime_context,
            fitness_scores       = fitness_scores,
        )

        ensemble_df         = ensemble_result.forecast_df.copy()
        ensemble_df["date"] = pd.to_datetime(ensemble_df["date"])
        ensemble_df         = _inject_actuals(ensemble_df, hist_df)

        results["Primary Ensemble"] = {
            "status":      "success",
            "forecast_df": ensemble_df,
            "metrics":     ensemble_result.metrics or {},
            "diagnostics": {},
            "executive_assessment": {
                "readiness_tier":     "Ensemble",
                "confidence_posture": ensemble_result.metadata.get(
                    "aggregation_method", "unknown"
                ),
                "risk_flags":        [],
                "decision_guidance": "See individual model assessments.",
            },
            "metadata":        ensemble_result.metadata or {},
            "diagnostic_only": False,
        }

    except Exception as e:
        failures.append(_build_failure_record("Primary Ensemble", e))
        results["Primary Ensemble"] = {
            "status":         "failed",
            "error":          str(e),
            "exception_type": type(e).__name__,
            "error_message":  str(e),
        }

    # ── Ridge Stacker ─────────────────────────────────────────────────────────
    try:
        stacker_result = build_stacked_forecast(
            results          = results,
            horizon          = horizon,
            confidence_level = confidence_level,
            df_historical    = hist_df,
        )
        stacker_df         = stacker_result.forecast_df.copy()
        stacker_df["date"] = pd.to_datetime(stacker_df["date"])
        stacker_df         = _inject_actuals(stacker_df, hist_df)

        results["Stacked Ensemble"] = {
            "status":      "success",
            "forecast_df": stacker_df,
            "metrics":     stacker_result.metrics or {},
            "diagnostics": {},
            "executive_assessment": {
                "readiness_tier":     "Stacked Ensemble",
                "confidence_posture": "stacked_meta_learner",
                "risk_flags":        [],
                "decision_guidance": "Ridge meta-learner on base model fold forecasts.",
            },
            "metadata":        stacker_result.metadata or {},
            "diagnostic_only": False,
        }
    except Exception as e:
        failures.append(_build_failure_record("Stacked Ensemble", e))
        results["Stacked Ensemble"] = {
            "status":         "failed",
            "error":          str(e),
            "exception_type": type(e).__name__,
            "error_message":  str(e),
        }

    if failures:
        results["_failures"] = failures

    results["_engine"] = {
        "engine_version":      ENGINE_VERSION,
        "observation_count":   observation_count,
        "horizon":             horizon,
        "confidence_level":    confidence_level,
        "active_tier":         active_tier,
        "config_snapshot":     config.as_dict(),
        "models_attempted":    len(tier_registry),
        "models_succeeded":    sum(
            1 for k, v in results.items()
            if not k.startswith("_") and isinstance(v, dict)
            and v.get("status") == "success"
        ),
        "regime_context":      regime_context,
        "fitness_scores":      fitness_scores,
    }

    return results
