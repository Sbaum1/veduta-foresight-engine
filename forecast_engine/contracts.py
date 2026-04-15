"""
forecast_engine/contracts.py
Simple typed containers for the Foresight X UI.
ForecastInput carries series data from app.py to the engine adapter.
No engine logic lives here.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class Frequency(Enum):
    MONTHLY   = "Monthly"
    QUARTERLY = "Quarterly"
    ANNUAL    = "Annual"
    WEEKLY    = "Weekly"
    DAILY     = "Daily"


@dataclass
class ForecastInput:
    series_id: str
    values:    pd.Series
    horizon:   int
    frequency: Frequency = Frequency.MONTHLY
