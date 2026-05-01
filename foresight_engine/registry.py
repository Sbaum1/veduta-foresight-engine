# ==================================================
# FILE: foresight_engine/registry.py
# VERSION: 3.0.0
# ROLE: AUTHORITATIVE MODEL REGISTRY
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - Explicit imports only
# - Stable deterministic order
# - No dynamic imports
# - No conditional gating
# - No ranking logic
# - Failures handled upstream in runner.py
# - X-13 registered as diagnostic_only
# - BSTS renamed to LocalLinearTrend (v3.0.0)
#
# TIER TAGS (min_tier):
#   essentials — available in all three platform tiers
#   pro        — available in Pro and Enterprise
#   enterprise — available in Enterprise only
# ==================================================

from __future__ import annotations
from typing import Any, Dict, List

from .models.naive               import run_naive
from .models.seasonal_naive      import run_seasonal_naive
from .models.drift               import run_drift
from .models.quantile_baseline   import run_quantile_baseline
from .models.ets                 import run_ets
from .models.arima               import run_arima
from .models.sarima              import run_sarima
from .models.sarimax             import run_sarimax
from .models.theta               import run_theta
from .models.stl_ets             import run_stl_ets
from .models.tbats               import run_tbats
from .models.prophet             import run_prophet
from .models.local_linear_trend  import run_local_linear_trend
from .models.x13                 import run_x13
from .models.ses                 import run_ses
from .models.mstl                import run_mstl
from .models.hw_damped           import run_hw_damped
from .models.croston             import run_croston
from .models.dhr                 import run_dhr
from .models.nnetar              import run_nnetar
from .models.nhits_model         import run_nhits
from .models.hybrid_models       import run_arima_xgboost, run_prophet_xgboost
from .models.lightgbm_model      import run_lightgbm
from .models.var_model           import run_var
from .models.varima              import run_varima
from .models.tsb                 import run_tsb
from .models.garch_model         import run_garch
from .models.xgboost_model       import run_xgboost
from .models.holt                import run_holt
from .models.ml_models           import run_random_forest, run_ridge, run_lasso
from .models.chronos_models     import (
    run_chronos_bolt_small,
    run_chronos_bolt_base,
    run_chronos_t5_small,
    run_chronos_2,
)
from .ensemble                   import run_primary_ensemble


def get_model_registry() -> List[Dict[str, Any]]:
    """
    Canonical forecasting model registry for Foresight Engine v3.0.0.

    Registry Entry Schema:
        name            : str      — display name, used as result key
        runner          : callable(df, horizon, confidence_level) -> ForecastResult
        diagnostic_only : bool     — excluded from scoring and ensemble if True
        ensemble_member : bool     — eligible for Primary Ensemble if True
        min_tier        : str      — minimum platform tier
        routing_note    : str|None — special input requirements or routing conditions
    """

    return [

        # --------------------------------------------------
        # EXECUTIVE DEFAULT (PRIMARY ENSEMBLE)
        # --------------------------------------------------
        {
            "name":             "Primary Ensemble",
            "runner":           run_primary_ensemble,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # ENSEMBLE MEMBERS — ESSENTIALS TIER
        # --------------------------------------------------
        {
            "name":             "LocalLinearTrend",
            "runner":           run_local_linear_trend,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     "Frequentist MLE state-space. Local linear trend + seasonal(12). Multi-start variance search.",
        },
        {
            "name":             "ETS",
            "runner":           run_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "Prophet",
            "runner":           run_prophet,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "SARIMA",
            "runner":           run_sarima,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "STL+ETS",
            "runner":           run_stl_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "TBATS",
            "runner":           run_tbats,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "Theta",
            "runner":           run_theta,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "HW_Damped",
            "runner":           run_hw_damped,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     "Preferred over ETS on short-horizon volatile series.",
        },
        {
            "name":             "Holt",
            "runner":           run_holt,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     "Linear trend ETS (undamped). Outperforms HW_Damped on strongly trended series.",
        },
        {
            "name":             "MSTL",
            "runner":           run_mstl,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     "Multi-seasonal decomposition. Best on series with both quarterly and annual cycles.",
        },

        # --------------------------------------------------
        # INDIVIDUAL ONLY — ESSENTIALS TIER
        # --------------------------------------------------
        {
            "name":             "SES",
            "runner":           run_ses,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     "Simple Exponential Smoothing. Diagnostic: if SES is best, series is a pure level process.",
        },
        {
            "name":             "Naive",
            "runner":           run_naive,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "SeasonalNaive",
            "runner":           run_seasonal_naive,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     "M3 MASE denominator model. If SeasonalNaive wins, series is fundamentally unpredictable.",
        },
        {
            "name":             "Drift",
            "runner":           run_drift,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     "Random walk with drift. Outperforms Naive on strongly trended series.",
        },
        {
            "name":             "QuantileBaseline",
            "runner":           run_quantile_baseline,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     "Empirical median/quantile forecast. Robust non-parametric baseline.",
        },

        # --------------------------------------------------
        # DIAGNOSTIC ONLY — ESSENTIALS TIER
        # --------------------------------------------------
        {
            "name":             "X-13",
            "runner":           run_x13,
            "diagnostic_only":  True,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # PRO TIER
        # --------------------------------------------------
        {
            "name":             "Croston_SBA",
            "runner":           run_croston,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "pro",
            "routing_note":     "Route series with >30% zero periods to this model.",
        },
        {
            "name":             "DHR",
            "runner":           run_dhr,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Preferred on series with multiple seasonality periods.",
        },
        {
            "name":             "LightGBM",
            "runner":           run_lightgbm,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Accepts exogenous columns in df. Falls back to lag-only if future exog not provided.",
        },
        {
            "name":             "XGBoost",
            "runner":           run_xgboost,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Gradient boosted trees. Same feature set as LightGBM — complementary in ensemble.",
        },

        # --------------------------------------------------
        # ENTERPRISE TIER
        # --------------------------------------------------
        {
            "name":             "RandomForest",
            "runner":           run_random_forest,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Bagged trees with lag/rolling features. Robust to outliers and non-linearity.",
        },
        {
            "name":             "Ridge",
            "runner":           run_ridge,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "L2-regularised linear regression. Best on series with linear dynamics.",
        },
        {
            "name":             "Lasso",
            "runner":           run_lasso,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "L1-regularised linear regression with automatic feature selection.",
        },
        {
            "name":             "ARIMA",
            "runner":           run_arima,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     None,
        },
        {
            "name":             "SARIMAX",
            "runner":           run_sarimax,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     "Requires exogenous columns in df alongside 'date' and 'value'.",
        },
        {
            "name":             "NNETAR",
            "runner":           run_nnetar,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Best on volatile non-linear series. Requires >= 16 obs.",
        },
        {
            "name":             "N-HiTS",
            "runner":           run_nhits,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Neural Hierarchical Interpolation. State-of-the-art neural univariate forecasting. Requires PyTorch + >= 3x horizon observations.",
        },
        {
            "name":             "ARIMA+XGBoost",
            "runner":           run_arima_xgboost,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "ARIMA captures linear structure; XGBoost corrects residuals. M4/M5 proven hybrid.",
        },
        {
            "name":             "Prophet+XGBoost",
            "runner":           run_prophet_xgboost,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Prophet captures trend+seasonality; XGBoost corrects residuals. Falls back to ARIMA+XGBoost if Prophet unavailable.",
        },
        {
            "name":             "VAR",
            "runner":           run_var,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     "Requires df with 2+ numeric series columns.",
        },
        {
            "name":             "VARIMA",
            "runner":           run_varima,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "VAR with per-column ADF integration. Multivariate non-stationary series. Falls back to ARIMA(1,1,1) on single-series input.",
        },
        {
            "name":             "TSB",
            "runner":           run_tsb,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Teüber-Syntetos-Boylan. Superior to Croston on non-stationary intermittent demand. Updates demand probability on every period.",
        },
        {
            "name":             "Chronos-Bolt-Small",
            "runner":           run_chronos_bolt_small,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Chronos-Bolt-Small (48M). Zero-shot. 250x faster than original Chronos. Requires: pip install chronos-forecasting torch",
        },
        {
            "name":             "Chronos-Bolt-Base",
            "runner":           run_chronos_bolt_base,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Chronos-Bolt-Base (205M). Zero-shot. Outperforms Chronos-Large at 600x speed. Requires: pip install chronos-forecasting torch",
        },
        {
            "name":             "Chronos-T5-Small",
            "runner":           run_chronos_t5_small,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Chronos-T5-Small (46M). Zero-shot. Monte Carlo sampling — best CI calibration in Chronos family. Requires: pip install chronos-forecasting torch",
        },
        {
            "name":             "Chronos-2",
            "runner":           run_chronos_2,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Chronos-2 (120M, Oct 2025). Zero-shot. 90%+ win rate over Chronos-Bolt. Best on fev-bench + GIFT-Eval. Requires: pip install chronos-forecasting torch",
        },
        {
            "name":             "GARCH",
            "runner":           run_garch,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Volatility CI modifier. volatility_forecast in metadata for CI scaling.",
        },

    ]


def get_ensemble_members() -> List[Dict[str, Any]]:
    return [e for e in get_model_registry() if e.get("ensemble_member", False)]


def get_production_models() -> List[Dict[str, Any]]:
    return [e for e in get_model_registry() if not e.get("diagnostic_only", False)]


def get_diagnostic_models() -> List[Dict[str, Any]]:
    return [e for e in get_model_registry() if e.get("diagnostic_only", False)]


def get_models_by_tier(tier: str) -> List[Dict[str, Any]]:
    tier_order = {"essentials": 0, "pro": 1, "enterprise": 2}
    tier_level = tier_order.get(tier, 2)
    return [
        e for e in get_model_registry()
        if tier_order.get(e.get("min_tier", "enterprise"), 2) <= tier_level
    ]


def get_ensemble_members_by_tier(tier: str) -> List[Dict[str, Any]]:
    return [e for e in get_models_by_tier(tier) if e.get("ensemble_member", False)]
