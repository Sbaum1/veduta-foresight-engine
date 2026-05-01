# ==================================================
# FILE: foresight_engine/contracts.py
# VERSION: 3.0.0
# ROLE: CANONICAL FORECAST RESULT CONTRACT
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# GOVERNANCE:
# - Frozen dataclass — immutable after construction
# - No Streamlit dependencies
# - No session state dependencies
# - All fields explicitly typed
# ==================================================

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


ENGINE_VERSION = "3.0.0"


@dataclass(frozen=True)
class ForecastResult:
    """
    Canonical forecast result contract for Foresight Engine v3.0.0.

    REQUIRED FIELDS:
        model_name   : Name of the model that produced this result
        forecast_df  : Full DataFrame — history + future rows
                       Columns: date, actual, forecast, ci_low, ci_mid,
                                ci_high, error_pct
        metrics      : Accuracy metrics computed on realized data only
        metadata     : Transparency-only descriptors — no ranking logic

    ENGINE FIELDS (auto-populated):
        engine_version : Foresight Engine version that produced this result
        engine_id      : Unique UUID for this execution run

    GOVERNANCE:
        - frozen=True enforces immutability after construction
        - No logic lives in this contract
        - metadata is transparency-only — never drives decisions
    """

    model_name:     str
    forecast_df:    pd.DataFrame
    metrics:        Dict[str, float]
    metadata:       Dict[str, Any]

    engine_version: str = field(default=ENGINE_VERSION)
    engine_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
