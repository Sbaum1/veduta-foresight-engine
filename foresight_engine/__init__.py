# ==================================================
# FILE: foresight_engine/__init__.py
# VERSION: 3.0.0
# ROLE: PUBLIC INTERFACE
# ENGINE: Foresight Engine v3.0.0
# ==================================================

from .contracts        import ForecastResult, ENGINE_VERSION
from .registry         import (
    get_model_registry,
    get_ensemble_members,
    get_models_by_tier,
    get_ensemble_members_by_tier,
)
from .runner           import run_all_models, apply_stress
from .ensemble         import run_primary_ensemble
from .certifier        import certify, verify_certificates, generate_certificates
from .backtest         import run_backtest
from .stacker          import build_stacked_forecast
from .foresight_config import (
    get_config,
    set_tier,
    set_flag,
    reset_config,
    get_active_tier,
)

__all__ = [
    "ForecastResult",
    "ENGINE_VERSION",
    "run_all_models",
    "run_primary_ensemble",
    "apply_stress",
    "run_backtest",
    "build_stacked_forecast",
    "certify",
    "verify_certificates",
    "generate_certificates",
    "get_model_registry",
    "get_ensemble_members",
    "get_models_by_tier",
    "get_ensemble_members_by_tier",
    "get_config",
    "set_tier",
    "set_flag",
    "reset_config",
    "get_active_tier",
]
