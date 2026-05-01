# ==================================================
# FILE: foresight_engine/foresight_config.py
# VERSION: 2.0.1
# UPDATED: April 20, 2026 — tier model counts corrected to match
#          registry.py v3.0.0 (38 registered models)
# ROLE: FEATURE FLAG AND TIER CONFIGURATION
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - Single source of truth for platform tier and feature flags
# - All mutations go through set_tier() or set_flag()
#
# TIER HIERARCHY:
#   essentials  — 16 models, 10 ensemble members
#   pro         — 24 models, 17 ensemble members
#   enterprise  — 38 models, 27 ensemble members (default)
#
# FEATURE FLAGS:
#   BACKTEST_ENABLED          — if False, skips backtest (fast dev mode)
#   MASE_EXCLUSION_ENABLED    — if False, bypasses Phase 3D auto-exclusion
#   DIVERSITY_CAP_ENABLED     — if False, ARIMA family cap not applied
#   INTERMITTENT_ROUTING_ENABLED — if False, Croston routing bypassed
#   PREPROCESSOR_ENABLED      — if False, structural break detection and
#                               fitness scoring are skipped
# ==================================================

from __future__ import annotations
from typing import List, Optional

VALID_TIERS = {"essentials", "pro", "enterprise"}

VALID_FLAGS = {
    "BACKTEST_ENABLED",
    "MASE_EXCLUSION_ENABLED",
    "DIVERSITY_CAP_ENABLED",
    "INTERMITTENT_ROUTING_ENABLED",
    "PREPROCESSOR_ENABLED",
}


class ForesightConfig:
    """
    Platform configuration for Foresight Engine v3.0.0.

    Holds the active tier and all feature flags.
    Constructed once at module load. Mutated via
    set_tier() and set_flag() accessors.
    """

    def __init__(
        self,
        active_tier:                   str  = "enterprise",
        ensemble_members_override:     Optional[List[str]] = None,
        backtest_enabled:              bool = True,
        mase_exclusion_enabled:        bool = True,
        diversity_cap_enabled:         bool = True,
        intermittent_routing_enabled:  bool = True,
        preprocessor_enabled:          bool = True,
    ) -> None:
        self.ACTIVE_TIER:                   str              = active_tier
        self.ENSEMBLE_MEMBERS_OVERRIDE:     Optional[List[str]] = ensemble_members_override
        self.BACKTEST_ENABLED:              bool             = backtest_enabled
        self.MASE_EXCLUSION_ENABLED:        bool             = mase_exclusion_enabled
        self.DIVERSITY_CAP_ENABLED:         bool             = diversity_cap_enabled
        self.INTERMITTENT_ROUTING_ENABLED:  bool             = intermittent_routing_enabled
        self.PREPROCESSOR_ENABLED:          bool             = preprocessor_enabled

    def as_dict(self) -> dict:
        return {
            "active_tier":                  self.ACTIVE_TIER,
            "ensemble_members_override":    self.ENSEMBLE_MEMBERS_OVERRIDE,
            "backtest_enabled":             self.BACKTEST_ENABLED,
            "mase_exclusion_enabled":       self.MASE_EXCLUSION_ENABLED,
            "diversity_cap_enabled":        self.DIVERSITY_CAP_ENABLED,
            "intermittent_routing_enabled": self.INTERMITTENT_ROUTING_ENABLED,
            "preprocessor_enabled":         self.PREPROCESSOR_ENABLED,
        }

    def __repr__(self) -> str:
        return (
            f"ForesightConfig("
            f"tier={self.ACTIVE_TIER!r}, "
            f"backtest={self.BACKTEST_ENABLED}, "
            f"mase_exclusion={self.MASE_EXCLUSION_ENABLED}, "
            f"diversity_cap={self.DIVERSITY_CAP_ENABLED}, "
            f"intermittent_routing={self.INTERMITTENT_ROUTING_ENABLED}, "
            f"preprocessor={self.PREPROCESSOR_ENABLED})"
        )


_config = ForesightConfig()


def get_config() -> ForesightConfig:
    """Return the active ForesightConfig singleton."""
    return _config


def set_tier(tier: str) -> None:
    """
    Set the active platform tier.
    Args: tier — 'essentials' | 'pro' | 'enterprise'
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier: {tier!r}. Must be one of: {sorted(VALID_TIERS)}")
    _config.ACTIVE_TIER = tier


def set_flag(flag: str, value: bool) -> None:
    """Set a feature flag by name."""
    if flag not in VALID_FLAGS:
        raise ValueError(f"Unknown flag: {flag!r}. Valid flags: {sorted(VALID_FLAGS)}")
    if not isinstance(value, bool):
        raise TypeError(f"Flag value must be bool, got {type(value).__name__}")
    setattr(_config, flag, value)


def reset_config() -> None:
    """Reset all config to production defaults."""
    global _config
    _config = ForesightConfig()


def get_active_tier() -> str:
    """Convenience accessor for the active tier string."""
    return _config.ACTIVE_TIER
