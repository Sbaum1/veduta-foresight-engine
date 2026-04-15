# ==================================================
# FILE: foresight_engine/models/contracts.py
# VERSION: 3.0.0
# ROLE: CONTRACT RE-EXPORT SHIM
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# PURPOSE:
#   Re-exports ForecastResult from the parent package so that
#   model files importing from this local path receive the SAME
#   class object as foresight_engine.contracts.
#
#   Uses a relative import — portable regardless of package name.
#
# GOVERNANCE:
#   - Never define ForecastResult here directly
#   - Always re-export from foresight_engine.contracts
#   - This file must never contain any logic
# ==================================================

from ..contracts import ForecastResult, ENGINE_VERSION

__all__ = ["ForecastResult", "ENGINE_VERSION"]
