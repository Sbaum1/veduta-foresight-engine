# FILE: foresight_engine/models/x13.py
# MODEL: X-13 ARIMA-SEATS (Diagnostic Only)
# ENGINE: Foresight Engine v3.0.0
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.x13 import x13_arima_analysis

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




def run_x13(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("X-13 requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred. Explicit monthly index required.")
        df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype(float)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    # --------------------------------------------------
    # X-13 DIAGNOSTIC EXECUTION
    # --------------------------------------------------

    try:
        res = x13_arima_analysis(y)
        trend = res.trend.dropna()

        if trend.empty:
            raise RuntimeError("X-13 produced no usable trend component.")

    except Exception as e:
        # Controlled diagnostic failure (still structurally valid output)
        empty_df = pd.DataFrame(
            columns=["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        )
        return ForecastResult(
            model_name="X-13",
            forecast_df=empty_df,
            metrics=None,
            metadata={
                "diagnostic_only": True,
                "status": "unavailable",
                "reason": str(e),
                "output_contract": "ForecastResult",
            },
        )

    # --------------------------------------------------
    # DIAGNOSTIC OUTPUT (NO TRUE FORECAST)
    # --------------------------------------------------

    hist_block = pd.DataFrame(
        {
            "date": trend.index,
            "actual": np.nan,
            "forecast": trend.values,
            "ci_low": np.nan,
            "ci_mid": trend.values,
            "ci_high": np.nan,
            "error_pct": np.nan,
        }
    )

    hist_block = hist_block.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name="X-13",
        forecast_df=hist_block[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "diagnostic_only": True,
            "role": "seasonal_adjustment_authority",
            "frequency": inferred,
            "ci_method": "not_applicable",
            "output_contract": "ForecastResult",
        },
    )
