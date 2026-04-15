"""
veduta_foresight_app/charts_utils.py
VEDUTA Foresight — Charts + Utilities
"""
from __future__ import annotations
import io
import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veduta_foresight_app.styles import (
    PLOTLY_BASE, FS_SILVER, FS_ELECTRIC, FS_GOLD,
    FS_GREEN, FS_RED, FS_AMBER, NOCTURNE_BLACK, WHITE_55, WHITE_80,
)


# ==============================================================================
# CHARTS
# ==============================================================================

def _base(title: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure()
    layout = dict(**PLOTLY_BASE)
    layout.update(height=height, title=dict(
        text=title, font=dict(family="DM Mono, monospace", size=11,
        color="rgba(200,212,224,0.55)"), x=0.0, xanchor="left",
    ))
    fig.update_layout(**layout)
    return fig


def series_chart(
    series: pd.Series,
    outlier_indices: List[int] = None,
    show_trend: bool = True,
    title: str = "SERIES",
) -> go.Figure:
    fig = _base(title, 300)
    x = series.index.tolist(); y = series.values.tolist()
    fig.add_trace(go.Scatter(x=x, y=y, name="Actual",
        line=dict(color=FS_SILVER, width=2),
        hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra></extra>"))
    if show_trend:
        n = len(y); t = np.arange(n,dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        trend_y = (intercept + slope * t).tolist()
        fig.add_trace(go.Scatter(x=x, y=trend_y, name="Trend",
            line=dict(color="rgba(200,212,224,0.3)", width=1.5, dash="dot"),
            hoverinfo="skip"))
    if outlier_indices:
        ox = [x[i] for i in outlier_indices if i < len(x)]
        oy = [y[i] for i in outlier_indices if i < len(y)]
        fig.add_trace(go.Scatter(x=ox, y=oy, name="Outlier",
            mode="markers", marker=dict(color=FS_AMBER, size=9,
            symbol="circle-open", line=dict(width=2, color=FS_AMBER)),
            hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Outlier</extra>"))
    return fig


def seasonal_chart(series: pd.Series, title: str = "SEASONAL PROFILE") -> go.Figure:
    fig = _base(title, 260)
    labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    means = [series[series.index.month == m].mean() for m in range(1, 13)]
    overall = np.nanmean(means)
    devs = [(v - overall) / (overall + 1e-8) * 100 for v in means]
    colors = [FS_GREEN if d >= 0 else FS_RED for d in devs]
    fig.add_trace(go.Bar(x=labels, y=devs, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>%{y:+.1f}% vs avg<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_yaxes(tickformat="+.0f%%")
    return fig


def forecast_chart(
    historical: pd.Series,
    forecast_index: pd.DatetimeIndex,
    point_forecast: np.ndarray,
    scenarios: List = None,
    intervals: List = None,
    show_hist: bool = True,
    show_hist_trend: bool = True,
    show_base: bool = True,
    show_upside: bool = True,
    show_downside: bool = True,
    title: str = "FORESIGHT FORECAST",
) -> go.Figure:
    fig = _base(title, 400)
    hx = historical.index.tolist()
    hy = historical.values.tolist()
    fx = list(forecast_index)

    if show_hist:
        fig.add_trace(go.Scatter(x=hx, y=hy, name="Historical",
            line=dict(color=FS_SILVER, width=2),
            hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Historical</extra>"))

    if show_hist_trend and len(hy) > 2:
        n = len(hy); t = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t, hy, 1)
        fig.add_trace(go.Scatter(x=hx, y=(intercept + slope*t).tolist(),
            line=dict(color="rgba(200,212,224,0.2)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"))

    # Prediction intervals
    if intervals:
        for intv in intervals:
            try:
                if hasattr(intv, 'lower') and hasattr(intv, 'upper'):
                    lo = np.array(intv.lower); hi = np.array(intv.upper)
                    if len(lo) == len(fx) and len(hi) == len(fx):
                        fig.add_trace(go.Scatter(
                            x=fx + fx[::-1], y=hi.tolist() + lo.tolist()[::-1],
                            fill="toself", fillcolor="rgba(74,158,224,0.06)",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False, hoverinfo="skip", name="PI"))
                        break
            except Exception:
                pass

    # Scenarios (upside / downside)
    if scenarios:
        for sc in scenarios:
            try:
                sc_val = str(sc.scenario.value) if hasattr(sc.scenario, 'value') else str(sc.scenario)
                fc_arr = np.array(sc.point_forecast)
                if sc_val == "UPSIDE" and show_upside and len(fc_arr) == len(fx):
                    fig.add_trace(go.Scatter(x=fx, y=fc_arr.tolist(), name="Upside",
                        line=dict(color=FS_GREEN, width=1.5, dash="dash"),
                        hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Upside</extra>"))
                elif sc_val == "DOWNSIDE" and show_downside and len(fc_arr) == len(fx):
                    fig.add_trace(go.Scatter(x=fx, y=fc_arr.tolist(), name="Downside",
                        line=dict(color=FS_RED, width=1.5, dash="dash"),
                        hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Downside</extra>"))
            except Exception:
                pass

    # Base forecast
    if show_base:
        fig.add_trace(go.Scatter(x=fx, y=point_forecast.tolist(), name="Base Forecast",
            line=dict(color=FS_ELECTRIC, width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Forecast</extra>"))

    # Connect
    if fx and hx:
        fig.add_trace(go.Scatter(
            x=[hx[-1], fx[0]], y=[hy[-1], float(point_forecast[0])],
            line=dict(color="rgba(74,158,224,0.3)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"))

    if fx:
        fig.add_vline(x=fx[0], line_color="rgba(200,212,224,0.15)",
                      line_dash="dash", line_width=1)
    return fig


def model_weight_chart(rankings: List, title: str = "ENSEMBLE WEIGHTS") -> go.Figure:
    fig = _base(title, 300)
    members = [r for r in rankings if not r.disqualified and r.ensemble_weight]
    members.sort(key=lambda r: -(r.ensemble_weight or 0))
    if not members:
        return fig
    names   = [r.model_id.replace("_"," ").title()[:20] for r in members[:12]]
    weights = [r.ensemble_weight for r in members[:12]]
    colors  = [FS_ELECTRIC] * len(names)
    fig.add_trace(go.Bar(x=names, y=weights,
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>Weight: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(tickformat=".2f")
    return fig


def mase_distribution_chart(rankings: List, title: str = "MODEL MASE DISTRIBUTION") -> go.Figure:
    fig = _base(title, 280)
    valid = [r for r in rankings if not r.disqualified and not np.isnan(r.mase)]
    names  = [r.model_id.replace("_"," ").title()[:18] for r in valid]
    mases  = [r.mase for r in valid]
    colors = [FS_GREEN if m < 1.0 else (FS_AMBER if m < 2.0 else FS_RED) for m in mases]
    fig.add_trace(go.Bar(y=names, x=mases, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>MASE: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=1.0, line_color="rgba(200,212,224,0.3)", line_dash="dash", line_width=1)
    fig.update_xaxes(title_text="MASE (< 1.0 beats naive baseline)")
    fig.update_layout(height=max(280, len(valid) * 22 + 60))
    return fig


def individual_forecasts_chart(
    historical: pd.Series,
    forecast_index: pd.DatetimeIndex,
    constituent_forecasts: Dict[str, np.ndarray],
    point_forecast: np.ndarray,
    rankings: List,
    n_show: int = 5,
    title: str = "INDIVIDUAL MODEL FORECASTS",
) -> go.Figure:
    fig = _base(title, 380)
    hx = historical.index.tolist()
    hy = historical.values.tolist()
    fx = list(forecast_index)

    fig.add_trace(go.Scatter(x=hx, y=hy, name="Historical",
        line=dict(color="rgba(200,212,224,0.4)", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Historical</extra>"))

    colors = ["#4A9EE0","#52B8B2","#C8974A","#A882E0","#82C852",
              "#E88A6A","#E8C842","#B8D4E8","#8AC852","#C852A8"]

    top_models = [r.model_id for r in rankings if not r.disqualified][:n_show]
    for i, mid in enumerate(top_models):
        fc = constituent_forecasts.get(mid)
        if fc is not None and len(fc) == len(fx):
            fig.add_trace(go.Scatter(x=fx, y=fc.tolist(),
                name=mid.replace("_"," ").title()[:18],
                line=dict(color=colors[i % len(colors)], width=1.2, dash="dot"),
                opacity=0.6,
                hovertemplate=f"%{{x|%b %Y}}: %{{y:,.2f}}<extra>{mid}</extra>"))

    # Ensemble on top
    if fx:
        fig.add_trace(go.Scatter(x=fx, y=point_forecast.tolist(), name="Ensemble",
            line=dict(color=FS_ELECTRIC, width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Ensemble</extra>"))

    if fx:
        fig.add_vline(x=fx[0], line_color="rgba(200,212,224,0.15)",
                      line_dash="dash", line_width=1)
    return fig


# ==============================================================================
# DATA UTILITIES
# ==============================================================================

def parse_series(raw: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    raw = raw.strip()
    if not raw:
        return None, "No data provided."
    lines = [l.strip() for l in raw.replace('\r\n','\n').split('\n') if l.strip()]
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.replace('\t',',').split(',')]
        if len(parts) >= 2:
            rows.append(parts[:2])
        elif len(parts) == 1:
            rows.append([None, parts[0]])

    dates, values = [], []
    for row in rows:
        date_str = row[0]; val_str = row[1] if len(row) > 1 else row[0]
        val_str = val_str.replace(',','').replace('$','').replace('%','').strip()
        try: v = float(val_str)
        except: continue
        dates.append(date_str); values.append(v)

    if len(values) < 12:
        return None, f"Need at least 12 observations. Found {len(values)}."

    parsed = []
    if dates[0] is not None:
        for d in dates:
            try: parsed.append(pd.to_datetime(d))
            except: parsed = []; break

    idx = pd.DatetimeIndex(parsed) if len(parsed) == len(values) else \
          pd.date_range("2015-01-01", periods=len(values), freq="MS")
    s = pd.Series(np.array(values, dtype=float), index=idx).dropna()
    if len(s) < 12:
        return None, "Not enough valid values."
    return s, None


def series_summary(series: pd.Series) -> Dict[str, Any]:
    vals = series.values.astype(float); n = len(vals)
    from scipy import stats as sp
    slope, intercept, r, _, _ = sp.linregress(np.arange(n), vals)
    monthly_means = [series[series.index.month == m].mean()
                     for m in range(1,13) if len(series[series.index.month==m])>0]
    seasonal_str = np.std(monthly_means) / (np.mean(vals)+1e-8) if monthly_means else 0
    return {
        "n_obs": n, "start": series.index[0].strftime("%b %Y"),
        "end": series.index[-1].strftime("%b %Y"),
        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "trend": "↑ Upward" if slope > 0 else "↓ Downward",
        "trend_r2": float(r**2),
        "monthly_growth_pct": slope / (np.mean(vals)+1e-8) * 100,
        "seasonal": seasonal_str > 0.03,
        "seasonal_str": float(seasonal_str),
        "cv": float(np.std(vals)/(np.mean(vals)+1e-8)),
    }


def detect_outliers(series: pd.Series) -> List[int]:
    vals = series.values.astype(float)
    pct = pd.Series(vals).pct_change().fillna(0).values
    mu = np.mean(pct[1:]); sig = np.std(pct[1:]) + 1e-8
    outliers = set(i for i, p in enumerate(pct) if abs((p-mu)/sig) > 2.5)
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        _, trend = hpfilter(vals, lamb=1600)
        res = vals - trend
        q1, q3 = np.percentile(res,25), np.percentile(res,75)
        iqr = q3-q1
        outliers.update(i for i,r in enumerate(res) if r < q1-2.5*iqr or r > q3+2.5*iqr)
    except: pass
    return sorted(outliers)


# ==============================================================================
# ENGINE ADAPTER — calls foresight_engine directly
# ==============================================================================

class _SeriesType(Enum):
    STANDARD     = "Standard"
    INTERMITTENT = "Intermittent"
    TRENDED      = "Trended"
    SEASONAL     = "Seasonal"


class _Strategy(Enum):
    ENSEMBLE = "Ensemble"
    SINGLE   = "Single"


class _ScenarioLabel(Enum):
    UPSIDE   = "UPSIDE"
    DOWNSIDE = "DOWNSIDE"


@dataclass
class _RankingEntry:
    rank:                    int
    model_id:                str
    mase:                    float
    smape:                   float
    ensemble_weight:         float
    disqualified:            bool
    disqualification_reason: Optional[str]
    composite_score:         float = 0.0
    readiness_tier:          str   = "Unscored"
    confidence_posture:      str   = "Unscored"
    risk_flags:              List[str] = None

    def __post_init__(self):
        if self.risk_flags is None:
            self.risk_flags = []


@dataclass
class _Interval:
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class _Scenario:
    scenario:       Any
    point_forecast: np.ndarray


class _ForesightResult:
    """Result object with the exact attributes app.py expects."""
    def __init__(
        self,
        rankings:              List[_RankingEntry],
        point_forecast:        np.ndarray,
        forecast_index:        pd.DatetimeIndex,
        constituent_forecasts: Dict[str, np.ndarray],
        scenarios:             List[_Scenario],
        intervals:             List[_Interval],
        series_type:           _SeriesType,
        strategy:              _Strategy,
        engine_version:        str,
        run_id:                str,
        series_id:             str,
        data_hash:             str,
        regime_context:        Optional[Dict] = None,
        stacked_forecast:      Optional[np.ndarray] = None,
    ):
        self.rankings              = rankings
        self.point_forecast        = point_forecast
        self.forecast_index        = forecast_index
        self.constituent_forecasts = constituent_forecasts
        self.scenarios             = scenarios
        self.intervals             = intervals
        self.series_type           = series_type
        self.strategy              = strategy
        self.engine_version        = engine_version
        self.run_id                = run_id
        self.series_id             = series_id
        self.data_hash             = data_hash
        self.regime_context        = regime_context or {}
        self.stacked_forecast      = stacked_forecast


class _ForesightEngineAdapter:
    """
    Adapter: receives ForecastInput from app.py,
    calls foresight_engine.run_all_models,
    returns _ForesightResult shaped for the existing UI.
    """

    def run(self, fi, run_id: str = "") -> _ForesightResult:
        from foresight_engine.runner import run_all_models

        series = fi.values  # pd.Series with DatetimeIndex

        # Build DataFrame for foresight_engine
        df = pd.DataFrame({
            "date":  series.index,
            "value": series.values,
        }).reset_index(drop=True)

        # Call the engine
        raw: Dict[str, Any] = run_all_models(
            df               = df,
            horizon          = fi.horizon,
            confidence_level = 0.90,
        )

        # Extract engine metadata (regime context, fitness scores)
        engine_meta    = raw.get("_engine", {})
        regime_context = engine_meta.get("regime_context", {})

        # Build forecast_index from last historical date + horizon
        last_date      = series.index.max()
        forecast_index = pd.date_range(
            start   = last_date + pd.DateOffset(months=1),
            periods = fi.horizon,
            freq    = "MS",
        )

        # Build rankings — pass full raw dict so readiness tiers can be extracted
        rankings = self._build_rankings(raw)

        # Build ensemble point forecast
        ensemble_raw = raw.get("Primary Ensemble")
        if ensemble_raw and ensemble_raw.get("status") == "success":
            point_forecast = self._extract_future(
                ensemble_raw.get("forecast_df", pd.DataFrame()),
                forecast_index,
            )
        else:
            # Fall back to top-ranked model
            point_forecast = np.zeros(fi.horizon)
            for entry in rankings:
                if not entry.disqualified:
                    model_raw = raw.get(entry.model_id, {})
                    if model_raw.get("status") == "success":
                        point_forecast = self._extract_future(
                            model_raw.get("forecast_df", pd.DataFrame()),
                            forecast_index,
                        )
                        break

        # Extract stacked ensemble forecast
        stacked_raw      = raw.get("Stacked Ensemble")
        stacked_forecast = None
        if stacked_raw and stacked_raw.get("status") == "success":
            stacked_forecast = self._extract_future(
                stacked_raw.get("forecast_df", pd.DataFrame()),
                forecast_index,
            )

        # Build constituent forecasts
        constituent_forecasts: Dict[str, np.ndarray] = {}
        skip_names = {"Primary Ensemble", "Stacked Ensemble"}
        for name, result in raw.items():
            if name.startswith("_") or name in skip_names:
                continue
            if not isinstance(result, dict) or result.get("status") != "success":
                continue
            if result.get("diagnostic_only"):
                continue
            fc = self._extract_future(
                result.get("forecast_df", pd.DataFrame()),
                forecast_index,
            )
            if len(fc) == fi.horizon:
                constituent_forecasts[name] = fc

        # Build CI intervals from ensemble forecast_df
        intervals: List[_Interval] = []
        if ensemble_raw and ensemble_raw.get("status") == "success":
            fc_df = ensemble_raw.get("forecast_df", pd.DataFrame())
            if not fc_df.empty and "ci_low" in fc_df.columns:
                future = fc_df[fc_df["actual"].isna()].copy() \
                    if "actual" in fc_df.columns else fc_df.copy()
                future = future.iloc[:fi.horizon]
                if len(future) > 0:
                    lo = future["ci_low"].ffill().values.astype(float)
                    hi = future["ci_high"].ffill().values.astype(float) \
                        if "ci_high" in future.columns else lo
                    intervals.append(_Interval(lower=lo, upper=hi))

        # Build upside / downside scenarios from CI
        scenarios: List[_Scenario] = []
        if intervals:
            scenarios.append(_Scenario(
                scenario       = _ScenarioLabel.UPSIDE,
                point_forecast = intervals[0].upper,
            ))
            scenarios.append(_Scenario(
                scenario       = _ScenarioLabel.DOWNSIDE,
                point_forecast = intervals[0].lower,
            ))

        # Series type classification
        zero_pct = float((series == 0).mean())
        if zero_pct > 0.20:
            series_type = _SeriesType.INTERMITTENT
        elif len(series) > 12:
            coeffs = np.polyfit(np.arange(len(series)), series.values, 1)
            trend_strength = abs(coeffs[0]) * len(series) / (series.mean() + 1e-8)
            series_type = _SeriesType.TRENDED if trend_strength > 0.10 else _SeriesType.SEASONAL
        else:
            series_type = _SeriesType.STANDARD

        data_hash = hashlib.md5(series.values.tobytes()).hexdigest()

        return _ForesightResult(
            rankings              = rankings,
            point_forecast        = point_forecast,
            forecast_index        = forecast_index,
            constituent_forecasts = constituent_forecasts,
            scenarios             = scenarios,
            intervals             = intervals,
            series_type           = series_type,
            strategy              = _Strategy.ENSEMBLE,
            engine_version        = "Foresight Engine v3.0.0",
            run_id                = run_id or "—",
            series_id             = fi.series_id,
            data_hash             = data_hash,
            regime_context        = regime_context,
            stacked_forecast      = stacked_forecast,
        )

    def _extract_future(
        self,
        fc_df: pd.DataFrame,
        forecast_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Extract future forecast values aligned to forecast_index."""
        if fc_df.empty or "forecast" not in fc_df.columns:
            return np.zeros(len(forecast_index))
        fc_df = fc_df.copy()
        fc_df["date"] = pd.to_datetime(fc_df["date"])
        if "actual" in fc_df.columns:
            future = fc_df[fc_df["actual"].isna()].copy()
        else:
            future = fc_df.copy()
        if future.empty:
            return np.zeros(len(forecast_index))
        future = future.set_index("date").reindex(forecast_index)
        vals = future["forecast"].ffill().fillna(0).values
        return vals.astype(float)

    def _build_rankings(self, raw: Dict[str, Any]) -> List[_RankingEntry]:
        """Convert run_all_models output to ranked _RankingEntry list."""
        rows = []
        skip_names = {"Primary Ensemble", "Stacked Ensemble"}
        for name, result in raw.items():
            if name.startswith("_") or name in skip_names:
                continue
            if not isinstance(result, dict):
                continue

            status  = result.get("status", "unknown")
            metrics = result.get("metrics", {})
            mase    = metrics.get("MASE")
            smape   = metrics.get("SMAPE", float("nan"))

            # Extract executive assessment from runner output
            exec_assessment   = result.get("executive_assessment", {})
            readiness_tier    = exec_assessment.get("readiness_tier", "Unscored")
            confidence_posture = exec_assessment.get("confidence_posture", "Unscored")
            risk_flags        = exec_assessment.get("risk_flags", [])

            if status != "success" or mase is None:
                rows.append({
                    "name":              name,
                    "mase":              float("inf"),
                    "smape":             float("nan"),
                    "weight":            0.0,
                    "dq":                True,
                    "reason":            result.get("error_message", "Did not complete"),
                    "readiness_tier":    "Ineligible",
                    "confidence_posture": "Not Eligible",
                    "risk_flags":        [],
                    "composite":         float("inf"),
                    "diagnostic_only":   result.get("diagnostic_only", False),
                })
            else:
                mase_f  = float(mase)
                smape_f = float(smape) if smape is not None else float("nan")
                # Composite: 70% MASE + 30% normalized SMAPE (SMAPE is 0-200 scale)
                smape_norm = smape_f / 200.0 if np.isfinite(smape_f) else 1.0
                composite  = round(0.7 * min(mase_f, 5.0) + 0.3 * min(smape_norm, 1.0), 4)
                rows.append({
                    "name":              name,
                    "mase":              mase_f,
                    "smape":             smape_f,
                    "weight":            0.0,
                    "dq":                False,
                    "reason":            None,
                    "readiness_tier":    readiness_tier,
                    "confidence_posture": confidence_posture,
                    "risk_flags":        risk_flags if isinstance(risk_flags, list) else [],
                    "composite":         composite,
                    "diagnostic_only":   result.get("diagnostic_only", False),
                })

        # Sort: success by MASE, failures at end
        success = [r for r in rows if not r["dq"]]
        failed  = [r for r in rows if r["dq"]]
        success.sort(key=lambda x: x["mase"])

        # Inverse-MASE weights for non-diagnostic ensemble members
        eligible = [r for r in success if not r["diagnostic_only"]]
        mase_vals = np.array([r["mase"] for r in eligible], dtype=float)
        if len(mase_vals) > 0:
            inv_mase  = np.where(mase_vals > 0, 1.0 / (mase_vals + 1e-8), 0.0)
            total     = inv_mase.sum()
            weights   = inv_mase / (total + 1e-8) if total > 0 else inv_mase
            for i, row in enumerate(eligible):
                row["weight"] = float(weights[i])

        all_rows = success + failed
        return [
            _RankingEntry(
                rank                    = i + 1,
                model_id                = r["name"],
                mase                    = r["mase"],
                smape                   = r["smape"],
                ensemble_weight         = r["weight"],
                disqualified            = r["dq"],
                disqualification_reason = r["reason"],
                composite_score         = r["composite"],
                readiness_tier          = r["readiness_tier"],
                confidence_posture      = r["confidence_posture"],
                risk_flags              = r["risk_flags"],
            )
            for i, r in enumerate(all_rows)
        ]


def get_foresight_engine() -> _ForesightEngineAdapter:
    """Return the Foresight engine adapter backed by foresight_engine."""
    return _ForesightEngineAdapter()
def export_excel(series: pd.Series, result, series_id: str) -> bytes:
    import io, openpyxl
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Historical
        pd.DataFrame({"Date":series.index,"Value":series.values}).to_excel(
            writer, sheet_name="Historical Data", index=False)
        # Forecast
        fc_df = pd.DataFrame({
            "Date":    result.forecast_index,
            "Forecast": result.point_forecast,
        })
        # Add scenarios
        for sc in result.scenarios:
            sc_name = str(sc.scenario.value) if hasattr(sc.scenario,'value') else str(sc.scenario)
            if sc_name in {"BASE","UPSIDE","DOWNSIDE"}:
                if len(sc.point_forecast) == len(result.forecast_index):
                    fc_df[sc_name] = sc.point_forecast
        fc_df.to_excel(writer, sheet_name="Forecast", index=False)
        # Rankings
        rank_rows = [{"Rank":r.rank,"Model":r.model_id,"MASE":round(r.mase,4),
                      "SMAPE":round(r.smape,4) if hasattr(r,'smape') else "",
                      "Weight":round(r.ensemble_weight,4) if r.ensemble_weight else 0,
                      "Disqualified":r.disqualified,
                      "Reason":r.disqualification_reason or ""}
                     for r in result.rankings]
        pd.DataFrame(rank_rows).to_excel(writer, sheet_name="Model Rankings", index=False)
    return buf.getvalue()


def generate_sample_data() -> str:
    """FRED-calibrated US Durable Goods Manufacturing Orders sample."""
    rng = np.random.default_rng(42)
    n = 96
    t = np.arange(n, dtype=float)
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    # Calibrated to FRED DGORDER characteristics
    vals = (248_000 + t * 620
            + 18_500 * np.sin(2*np.pi*t/12 - 0.3)
            + 6_200 * np.sin(2*np.pi*t/6)
            + rng.normal(0, 4_500, n))
    # COVID shock
    vals[51:54] -= 38_000
    vals[54:60] += np.linspace(-20_000, 0, 6)
    vals = np.maximum(vals, 180_000)
    lines = [f"{d.strftime('%Y-%m-%d')},{v:.0f}"
             for d, v in zip(dates, vals)]
    return "Date,Value\n" + "\n".join(lines)
