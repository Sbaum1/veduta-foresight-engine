"""
veduta_foresight_app/app.py
================================================================================
VEDUTA Foresight X — Upgraded Streamlit Platform
"22 models. One answer. Elevated."

Upgrades over Foresight v1:
  · st.fragment — engine run isolated, UI never blocks
  · Native bslib-style theme via config.toml
  · CSS custom properties (design tokens) — no more string injection hacks
  · Animated metric cards, progress indicators, weight bars
  · Inline chart controls per module
  · Model ranking table rebuilt with gradient weight bars
  · Better locked-tab UX with breathing animation

Engine: UNCHANGED — zero modifications to forecast_engine/
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List
import uuid

st.set_page_config(
    page_title="VEDUTA Foresight X",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from veduta_foresight_app.styles import (
    FORESIGHT_CSS, PLOTLY_BASE,
    FS_SILVER, FS_ELECTRIC, FS_GREEN, FS_RED, FS_AMBER, FS_GOLD,
    NOCTURNE_BLACK, CHARCOAL,
)
from veduta_foresight_app.charts_utils import (
    series_chart, seasonal_chart, forecast_chart,
    model_weight_chart, mase_distribution_chart, individual_forecasts_chart,
    parse_series, series_summary, detect_outliers,
    get_foresight_engine, export_excel, generate_sample_data,
)

st.markdown(FORESIGHT_CSS, unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "fx_mode":            "Signal",
        "fx_series":          None,
        "fx_series_id":       "SERIES_001",
        "fx_summary":         None,
        "fx_outliers":        [],
        "fx_outlier_notes":   {},
        "fx_data_committed":  False,
        "fx_result":          None,
        "fx_run_id":          None,
        "fx_horizon":         12,
        "fx_frequency":       "Monthly",
        "fx_engine_running":  False,
        # Chart line toggles
        "fx_show_hist":       True,
        "fx_show_hist_trend": True,
        "fx_show_base":       True,
        "fx_show_up":         True,
        "fx_show_down":       True,
        "fx_n_ind_models":    5,
        # Tab 1 module visibility
        "fx_mod1_open":       True,
        "fx_mod2_open":       True,
        "fx_mod3_open":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Engine helpers ────────────────────────────────────────────────────────────
@st.cache_resource
def _get_engine():
    return get_foresight_engine()


def _metric_card(label: str, value: str, sub: str = "",
                 color_class: str = "") -> str:
    """Render a VEDUTA metric card as HTML."""
    return f"""
    <div class="fx-metric-card">
        <div class="fx-metric-label">{label}</div>
        <div class="fx-metric-value {color_class}">{value}</div>
        {'<div class="fx-metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""


def _section(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="fx-section-header">{title}</div>'
        f'{"<div class=fx-section-sub>" + sub + "</div>" if sub else ""}',
        unsafe_allow_html=True,
    )


def _locked(title: str, sub: str) -> None:
    st.markdown(f"""
    <div class="fx-locked">
        <div class="fx-locked-icon">◈</div>
        <div class="fx-locked-title">{title}</div>
        <div class="fx-locked-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def _mase_class(mase: float) -> str:
    if mase < 0.8:  return "fx-mase-good"
    if mase < 1.0:  return "fx-mase-ok"
    if mase < 2.0:  return "fx-mase-poor"
    return "fx-mase-bad"


# ── Navigation ────────────────────────────────────────────────────────────────
def render_nav():
    col_logo, col_badge, col_spacer, col_mode = st.columns([3, 2, 2, 1])

    with col_logo:
        st.markdown("""
        <div style="padding:8px 0 6px 0">
            <div class="fx-wordmark">VEDUTA &nbsp;·&nbsp; Foresight</div>
            <div class="fx-edition">◈ &nbsp;X Edition</div>
            <div class="fx-tagline">22 Models &nbsp;·&nbsp; One Answer &nbsp;·&nbsp; Elevated</div>
        </div>""", unsafe_allow_html=True)

    with col_badge:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="m3-badge">
            <div class="m3-dot"></div>
            M3 · MASE 0.6913 · #1 Modern Published
        </div>""", unsafe_allow_html=True)

    with col_spacer:
        pass

    with col_mode:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        mode = st.radio(
            "Mode", ["Signal", "Source"],
            index=0 if st.session_state.fx_mode == "Signal" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="fx_mode_radio",
        )
        st.session_state.fx_mode = mode
        st.caption("◈ " + ("Executive" if mode == "Signal" else "Full audit"))

    st.markdown("<hr class='fx-nav-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA INPUT & SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def render_tab1():
    source_mode = st.session_state.fx_mode == "Source"
    _section("Data Input & Series Analysis", "Import · Review · Commit · Discover")

    # ── Import panel ──────────────────────────────────────────────────────────
    if not st.session_state.fx_data_committed:
        with st.expander("◈  Import Data", expanded=True):
            col_in, col_help = st.columns([3, 1])

            with col_in:
                col_a, col_b, col_c = st.columns([3, 2, 2])
                with col_a:
                    sid = st.text_input(
                        "Series Name",
                        value=st.session_state.fx_series_id,
                        key="fx_sid_input",
                        placeholder="e.g. US_Durable_Goods",
                    )
                    st.session_state.fx_series_id = sid
                with col_b:
                    freq = st.selectbox(
                        "Frequency",
                        ["Monthly", "Quarterly", "Annual", "Weekly", "Daily"],
                        key="fx_freq_select",
                    )
                    st.session_state.fx_frequency = freq
                with col_c:
                    horizon = st.number_input(
                        "Horizon (periods)", 1, 60, 12, key="fx_horizon_input"
                    )
                    st.session_state.fx_horizon = horizon

                raw = st.text_area(
                    "Paste series data",
                    height=200,
                    key="fx_raw",
                    placeholder=(
                        "2020-01-01, 248500\n"
                        "2020-02-01, 251300\n"
                        "...\n\n"
                        "Or values only (monthly from Jan 2015 assumed):\n"
                        "248500\n251300\n..."
                    ),
                )

                col_parse, col_sample, col_gap = st.columns([2, 2, 6])
                with col_parse:
                    parse_clicked = st.button(
                        "→ Parse & Review",
                        key="fx_parse_btn",
                        type="primary",
                        use_container_width=True,
                    )
                with col_sample:
                    if st.button("Load Sample", key="fx_sample_btn",
                                 use_container_width=True):
                        st.session_state.fx_raw = generate_sample_data()
                        st.rerun()

            with col_help:
                st.markdown("""
                <div class="fx-controls" style="margin-top:28px">
                    <div class="fx-controls-label">Format Guide</div>
                    <div style="font-family:'DM Mono',monospace;font-size:9px;
                                color:rgba(255,255,255,0.5);line-height:1.8">
                        <code>2024-01-01, 248500</code><br>
                        <code>Jan 2024, 248500</code><br>
                        <code>248500</code> (value only)<br><br>
                        Min 24 observations<br>
                        Values may include $, commas
                    </div>
                </div>""", unsafe_allow_html=True)

            if parse_clicked:
                text = st.session_state.get("fx_raw", raw)
                series, err = parse_series(text)
                if err:
                    st.error(err)
                else:
                    st.session_state.fx_series  = series
                    st.session_state.fx_summary  = series_summary(series)
                    st.session_state.fx_outliers = detect_outliers(series)
                    st.rerun()

    # ── Review panel ──────────────────────────────────────────────────────────
    if st.session_state.fx_series is not None and not st.session_state.fx_data_committed:
        s  = st.session_state.fx_series
        sm = st.session_state.fx_summary or series_summary(s)

        st.markdown("<hr>", unsafe_allow_html=True)
        _section("Series Review", "Confirm before committing")

        # Metric row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        cells = [
            (c1, "Observations", str(sm["n_obs"]),         sm["start"] + " — " + sm["end"]),
            (c2, "Mean",         f"{sm['mean']:,.2f}",     f"σ = {sm['std']:,.2f}"),
            (c3, "Min",          f"{sm['min']:,.2f}",      ""),
            (c4, "Max",          f"{sm['max']:,.2f}",      ""),
            (c5, "Trend",        sm["trend"],              f"{sm['monthly_growth_pct']:+.3f}%/period"),
            (c6, "Seasonality",  "Detected ✓" if sm["seasonal"] else "Weak",
                                 f"Strength: {sm['seasonal_str']:.1%}"),
        ]
        for col, label, value, sub in cells:
            with col:
                st.markdown(_metric_card(label, value, sub), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Preview chart
        col_chart, col_ctrl = st.columns([5, 1])
        with col_ctrl:
            st.markdown('<div class="fx-controls"><div class="fx-controls-label">Controls</div>',
                        unsafe_allow_html=True)
            show_val   = st.checkbox("Values",   True, key="fx_rv_val")
            show_trend = st.checkbox("Trend",    True, key="fx_rv_trend")
            show_out   = st.checkbox("Outliers", True, key="fx_rv_out")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_chart:
            st.plotly_chart(
                series_chart(
                    s,
                    st.session_state.fx_outliers if show_out else [],
                    show_trend=show_trend,
                    title=f"RAW DATA — {st.session_state.fx_series_id}",
                ),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True},
            )

        # Outlier notes
        n_out = len(st.session_state.fx_outliers)
        if n_out > 0:
            with st.expander(f"◈  {n_out} Outlier(s) Detected — Add Notes"):
                for idx in st.session_state.fx_outliers[:6]:
                    if idx >= len(s): continue
                    date_label = s.index[idx].strftime("%b %Y")
                    val        = s.values[idx]
                    col_n, col_h = st.columns([3, 1])
                    with col_n:
                        note = st.text_input(
                            f"{date_label} · {val:,.2f}",
                            value=st.session_state.fx_outlier_notes.get(idx, ""),
                            key=f"fx_note_{idx}",
                            placeholder="What caused this? e.g. COVID, pricing error, one-time order",
                        )
                        st.session_state.fx_outlier_notes[idx] = note
                    with col_h:
                        st.selectbox(
                            "Handle as",
                            ["Keep with note", "Interpolate",
                             "Replace with seasonal avg", "Ignore in analysis"],
                            key=f"fx_handle_{idx}",
                        )

        col_commit, col_back, col_gap = st.columns([2, 2, 6])
        with col_commit:
            if st.button("✓ Commit Data", key="fx_commit_btn",
                         type="primary", use_container_width=True):
                st.session_state.fx_data_committed = True
                st.rerun()
        with col_back:
            if st.button("← Re-enter", key="fx_back_btn",
                         use_container_width=True):
                st.session_state.fx_series = None
                st.rerun()

    # ── Post-commit intelligence modules ──────────────────────────────────────
    if st.session_state.fx_data_committed:
        s  = st.session_state.fx_series
        sm = st.session_state.fx_summary or series_summary(s)

        st.markdown("<hr>", unsafe_allow_html=True)
        _section("Series Intelligence", "Layer 1 — Data Analysis · All modules unlocked")

        # ── Module 1: Overview ────────────────────────────────────────────────
        with st.expander("◈  Module 1 — Series Overview", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            mets = [
                (c1, "Observations", str(sm["n_obs"]),
                 f"{sm['start']} — {sm['end']}", ""),
                (c2, "Mean Value", f"{sm['mean']:,.2f}",
                 f"σ = {sm['std']:,.2f}", "gold"),
                (c3, "Trend", sm["trend"],
                 f"{sm['monthly_growth_pct']:+.3f}%/period", ""),
                (c4, "Seasonality",
                 "Detected ✓" if sm["seasonal"] else "Weak",
                 f"Strength: {sm['seasonal_str']:.1%}",
                 "teal" if sm["seasonal"] else ""),
            ]
            for col, label, value, sub, color in mets:
                with col:
                    st.markdown(
                        _metric_card(label, value, sub, color),
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            col_v, col_c = st.columns([5, 1])
            with col_c:
                st.markdown('<div class="fx-controls"><div class="fx-controls-label">Controls</div>',
                            unsafe_allow_html=True)
                sv = st.checkbox("Values",  True, key="fx_m1_val")
                st_= st.checkbox("Trend",   True, key="fx_m1_trend")
                so = st.checkbox("Outliers",True, key="fx_m1_out")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_v:
                if sv:
                    st.plotly_chart(
                        series_chart(
                            s,
                            st.session_state.fx_outliers if so else [],
                            show_trend=st_,
                            title=f"SERIES — {st.session_state.fx_series_id}",
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": True},
                    )

            if source_mode:
                st.markdown(f"""
                <div class="fx-audit">
                    CV: {sm['cv']:.4f} &nbsp;·&nbsp;
                    Trend R²: {sm['trend_r2']:.4f} &nbsp;·&nbsp;
                    Seasonal strength: {sm['seasonal_str']:.4f} &nbsp;·&nbsp;
                    Outliers flagged: {len(st.session_state.fx_outliers)}
                </div>""", unsafe_allow_html=True)

        # ── Module 2: Seasonality ─────────────────────────────────────────────
        with st.expander("◈  Module 2 — Seasonality & Rhythms"):
            col_v2, col_c2 = st.columns([5, 1])
            with col_c2:
                st.markdown('<div class="fx-controls"><div class="fx-controls-label">Controls</div>',
                            unsafe_allow_html=True)
                show_seas = st.checkbox("Profile", True, key="fx_m2_seas")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_v2:
                if show_seas:
                    st.plotly_chart(
                        seasonal_chart(s, title="SEASONAL PROFILE"),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # Peak / trough callout
            labels = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            means   = [s[s.index.month == m].mean() for m in range(1, 13)]
            overall = float(np.mean(means))
            peaks   = [labels[i] for i, v in enumerate(means) if v > overall * 1.05]
            troughs = [labels[i] for i, v in enumerate(means) if v < overall * 0.95]

            cp, ct = st.columns(2)
            with cp:
                if peaks:
                    st.markdown(_metric_card(
                        "Peak Months ↑",
                        " · ".join(peaks), "", "teal"
                    ), unsafe_allow_html=True)
            with ct:
                if troughs:
                    st.markdown(f"""
                    <div class="fx-metric-card">
                        <div class="fx-metric-label">Trough Months ↓</div>
                        <div style="font-family:'DM Mono',monospace;font-size:14px;
                                    color:{FS_RED}">{" · ".join(troughs)}</div>
                        <div class="fx-metric-sub">Intervention opportunity</div>
                    </div>""", unsafe_allow_html=True)

        # ── Module 3: Engine config & run ─────────────────────────────────────
        with st.expander("◈  Module 3 — Engine Configuration & Run", expanded=True):
            col_h, col_f, col_gap2 = st.columns([2, 2, 4])
            with col_h:
                horizon = st.selectbox(
                    "Forecast horizon",
                    [6, 12, 18, 24, 36, 48], index=1, key="fx_h_sel"
                )
                st.session_state.fx_horizon = horizon
            with col_f:
                freq2 = st.selectbox(
                    "Frequency",
                    ["Monthly", "Quarterly", "Annual", "Weekly", "Daily"],
                    index=["Monthly","Quarterly","Annual","Weekly","Daily"].index(
                        st.session_state.fx_frequency),
                    key="fx_freq_sel2"
                )
                st.session_state.fx_frequency = freq2

            st.markdown("<br>", unsafe_allow_html=True)

            # ── st.fragment — engine run is isolated ──────────────────────────
            # This fragment reruns independently when the button is pressed.
            # The rest of the page does NOT rerun. UI stays responsive.
            @st.fragment
            def engine_run_fragment():
                col_run, col_info = st.columns([2, 6])
                with col_run:
                    run_clicked = st.button(
                        "→ Run Foresight X",
                        key="fx_run_btn",
                        type="primary",
                        use_container_width=True,
                    )
                with col_info:
                    st.caption(
                        "22 models · MASE-inverse ensemble · "
                        f"Horizon {st.session_state.fx_horizon} periods · "
                        "Engine cached between runs"
                    )

                if run_clicked:
                    run_id = f"FX_{uuid.uuid4().hex[:8].upper()}"
                    from forecast_engine.contracts import ForecastInput, Frequency
                    freq_map = {
                        "Monthly":   Frequency.MONTHLY,
                        "Quarterly": Frequency.QUARTERLY,
                        "Annual":    Frequency.ANNUAL,
                        "Weekly":    Frequency.WEEKLY,
                        "Daily":     Frequency.DAILY,
                    }
                    freq = freq_map.get(st.session_state.fx_frequency, Frequency.MONTHLY)
                    fi   = ForecastInput(
                        series_id = st.session_state.fx_series_id,
                        values    = st.session_state.fx_series,
                        horizon   = st.session_state.fx_horizon,
                        frequency = freq,
                    )
                    with st.spinner(
                        f"Running 22-model Foresight X ensemble · "
                        f"Horizon: {st.session_state.fx_horizon} · "
                        f"Run: {run_id}"
                    ):
                        engine = _get_engine()
                        result = engine.run(fi, run_id=run_id)

                    st.session_state.fx_result = result
                    st.session_state.fx_run_id  = run_id

                    if result:
                        n_q = sum(1 for r in result.rankings if not r.disqualified)
                        top = result.rankings[0].model_id if result.rankings else "—"
                        st.markdown(f"""
                        <div class="fx-result-banner">
                            <div class="fx-result-dot"></div>
                            <div class="fx-result-text">
                                ENGINE COMPLETE &nbsp;·&nbsp; {n_q}/22 qualified
                                &nbsp;·&nbsp; Top: {top}
                                &nbsp;·&nbsp; Switch to Tab II
                            </div>
                        </div>""", unsafe_allow_html=True)

            engine_run_fragment()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST CANVAS
# ══════════════════════════════════════════════════════════════════════════════
def render_tab2():
    if not st.session_state.fx_result:
        _locked("Forecast Canvas Locked",
                "Run the engine in Tab I to unlock")
        return

    result      = st.session_state.fx_result
    s           = st.session_state.fx_series
    source_mode = st.session_state.fx_mode == "Source"

    _section("Forecast Canvas", "22-Model Ensemble · Scenarios · Weighted MASE")

    # ── Top metric row ─────────────────────────────────────────────────────────
    n_qualified = sum(1 for r in result.rankings if not r.disqualified)
    top         = result.rankings[0] if result.rankings else None
    fc_mean     = float(np.mean(result.point_forecast))

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, sub, color in [
        (c1, "Forecast Mean",   f"{fc_mean:,.2f}",       "Per period",        "electric"),
        (c2, "Horizon",         f"{len(result.point_forecast)}",
                                                          f"{st.session_state.fx_frequency}", ""),
        (c3, "Models Qualified",f"{n_qualified}/22",
                                                          f"Top: {top.model_id if top else '—'}", ""),
        (c4, "Top MASE",        f"{top.mase:.4f}" if top else "—",
                                                          "Best backtest MASE", "green"),
        (c5, "Series Type",     str(result.series_type.value).replace("SeriesType.", "") if result.series_type else "—",
                                                          str(result.strategy.value), ""),
    ]:
        with col:
            st.markdown(_metric_card(label, value, sub, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart with inline controls ───────────────────────────────────
    col_chart, col_ctrl = st.columns([5, 1])
    with col_ctrl:
        st.markdown('<div class="fx-controls"><div class="fx-controls-label">Line Controls</div>',
                    unsafe_allow_html=True)
        show_hist  = st.checkbox("Historical",    True, key="fx_fc_hist")
        show_ht    = st.checkbox("Hist. Trend",   True, key="fx_fc_ht")
        show_base  = st.checkbox("Base Forecast", True, key="fx_fc_base")
        show_up    = st.checkbox("Upside",        True, key="fx_fc_up")
        show_down  = st.checkbox("Downside",      True, key="fx_fc_down")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart:
        st.plotly_chart(
            forecast_chart(
                historical    = s,
                forecast_index = result.forecast_index,
                point_forecast = result.point_forecast,
                scenarios      = result.scenarios,
                intervals      = result.intervals,
                show_hist      = show_hist,
                show_hist_trend= show_ht,
                show_base      = show_base,
                show_upside    = show_up,
                show_downside  = show_down,
                title          = f"FORESIGHT X — {st.session_state.fx_series_id}",
            ),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": True},
        )

    # ── Model rankings — redesigned table ─────────────────────────────────────
    with st.expander("◈  Model Rankings — All 22 Models", expanded=True):
        # Header
        st.markdown("""
        <div class="fx-model-header">
            <div style="min-width:32px">Rank</div>
            <div style="flex:1">Model</div>
            <div style="min-width:72px;text-align:right">MASE</div>
            <div style="min-width:100px;padding-left:12px">Weight</div>
            <div style="min-width:90px;text-align:right">Status</div>
        </div>""", unsafe_allow_html=True)

        max_weight = max((r.ensemble_weight or 0) for r in result.rankings)

        for r in result.rankings:
            dq  = r.disqualified
            w   = r.ensemble_weight or 0
            bar = int((w / (max_weight + 1e-8)) * 80) if not dq else 0
            rank_class  = "fx-rank-top" if r.rank <= 3 else "fx-rank-num"
            mase_class  = _mase_class(r.mase) if not np.isnan(r.mase) else "fx-mase-bad"
            weight_text = f"{w:.3f}" if not dq else "—"

            badge = ""
            if dq:
                reason = (r.disqualification_reason or "")[:22]
                badge  = f'<span class="fx-dq-badge">DQ{" · " + reason if reason else ""}</span>'
            elif w > 0.01:
                badge = '<span class="fx-ensemble-badge">ENSEMBLE</span>'

            st.markdown(f"""
            <div class="fx-model-row">
                <div class="{rank_class}">{r.rank}</div>
                <div class="fx-model-name">{r.model_id}{badge}</div>
                <div class="fx-mase-val {mase_class}">{r.mase:.4f}</div>
                <div class="fx-weight-col">
                    <div style="font-family:'DM Mono',monospace;font-size:9px;
                                color:rgba(200,212,224,0.55);text-align:right">
                        {weight_text}
                    </div>
                    <div class="fx-weight-bar-bg">
                        <div class="fx-weight-bar-fill" style="width:{bar}px"></div>
                    </div>
                </div>
                <div style="min-width:90px;text-align:right;font-size:9px;
                            font-family:'DM Mono',monospace;
                            color:{'#4AB88A' if not dq and w>0 else ('#C85A4A' if dq else 'rgba(255,255,255,0.3)')}">
                    {'◈ WEIGHTED' if not dq and w > 0 else ('RANKED' if not dq else 'DISQUALIFIED')}
                </div>
            </div>""", unsafe_allow_html=True)

        if source_mode:
            st.markdown(f"""
            <div class="fx-audit" style="margin-top:12px">
                Strategy: {result.strategy.value} &nbsp;·&nbsp;
                Engine: {result.engine_version} &nbsp;·&nbsp;
                Run: {result.run_id} &nbsp;·&nbsp;
                Hash: {(result.data_hash or '')[:20]}...
            </div>""", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    col_w, col_m = st.columns(2)
    with col_w:
        st.plotly_chart(
            model_weight_chart(result.rankings, title="ENSEMBLE WEIGHTS"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_m:
        st.plotly_chart(
            mase_distribution_chart(result.rankings, title="MASE DISTRIBUTION"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Individual model overlays ──────────────────────────────────────────────
    with st.expander("◈  Individual Model Forecasts"):
        col_iv, col_ic = st.columns([5, 1])
        with col_ic:
            st.markdown('<div class="fx-controls"><div class="fx-controls-label">Controls</div>',
                        unsafe_allow_html=True)
            n_show = st.slider(
                "Models", 2, min(10, n_qualified), 5, key="fx_n_ind"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_iv:
            st.plotly_chart(
                individual_forecasts_chart(
                    s, result.forecast_index,
                    result.constituent_forecasts,
                    result.point_forecast,
                    result.rankings,
                    n_show=n_show,
                    title=f"TOP {n_show} MODELS + ENSEMBLE",
                ),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True},
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS, AUDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def render_tab3():
    if not st.session_state.fx_result:
        _locked("Results Locked", "Run the engine in Tab I to unlock")
        return

    result      = st.session_state.fx_result
    s           = st.session_state.fx_series
    source_mode = st.session_state.fx_mode == "Source"

    _section("Results, Audit & Export",
             "Forecast Table · Credential · Model Audit · Export")

    # ── Forecast table ─────────────────────────────────────────────────────────
    with st.expander("◈  Forecast Table", expanded=True):
        fc_data = {
            "Period":   [d.strftime("%b %Y") for d in result.forecast_index],
            "Forecast": [round(v, 2) for v in result.point_forecast],
        }
        for sc in result.scenarios:
            sc_name = str(sc.scenario.value) if hasattr(sc.scenario, "value") else str(sc.scenario)
            if sc_name in {"UPSIDE", "DOWNSIDE"}:
                if len(sc.point_forecast) == len(result.forecast_index):
                    if sc_name not in fc_data:
                        fc_data[sc_name] = [round(v, 2) for v in sc.point_forecast]

        st.dataframe(
            pd.DataFrame(fc_data),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Sum",  f"{sum(result.point_forecast):,.0f}", ""),
            (c2, "Mean", f"{np.mean(result.point_forecast):,.2f}", "electric"),
            (c3, "Min",  f"{min(result.point_forecast):,.2f}", ""),
            (c4, "Max",  f"{max(result.point_forecast):,.2f}", "gold"),
        ]:
            with col:
                st.markdown(_metric_card(label, val, "", color), unsafe_allow_html=True)

    # ── M3 Credential ──────────────────────────────────────────────────────────
    with st.expander("◈  M3 Benchmark Credential"):
        st.markdown("""
        <div class="fx-credential">
            <div class="fx-credential-label">◈ M3 Competition Benchmark — VEDUTA Foresight X</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">
                <div>
                    <div class="fx-credential-number" style="color:#C8974A">0.6913</div>
                    <div class="fx-credential-desc">MEDIAN MASE</div>
                </div>
                <div>
                    <div class="fx-credential-number" style="color:#C8D4E0">1,428</div>
                    <div class="fx-credential-desc">MONTHLY SERIES TESTED</div>
                </div>
                <div>
                    <div class="fx-credential-number" style="color:#4AB88A">0</div>
                    <div class="fx-credential-desc">CRASHES</div>
                </div>
                <div>
                    <div class="fx-credential-number" style="color:#C8974A">#1</div>
                    <div class="fx-credential-desc">MODERN PUBLISHED RANK</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Model audit ────────────────────────────────────────────────────────────
    with st.expander("◈  Model Audit" + (" — Full Detail" if source_mode else "")):
        rows = []
        for r in result.rankings:
            rows.append({
                "Rank":          r.rank,
                "Model":         r.model_id,
                "MASE":          round(r.mase, 4),
                "SMAPE":         round(r.smape, 4) if hasattr(r, "smape") and r.smape else "—",
                "Composite":     round(r.composite_score, 4) if hasattr(r, "composite_score") else "—",
                "Weight":        round(r.ensemble_weight, 4) if r.ensemble_weight else 0,
                "DQ":            "Yes" if r.disqualified else "No",
                "Reason":        (r.disqualification_reason or "")[:35] if r.disqualified else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if source_mode:
            st.markdown(f"""
            <div class="fx-audit">
                Engine Version: {result.engine_version} &nbsp;·&nbsp;
                Run ID: {result.run_id} &nbsp;·&nbsp;
                Series: {result.series_id} &nbsp;·&nbsp;
                Strategy: {result.strategy.value} &nbsp;·&nbsp;
                Series Type: {str(result.series_type.value) if result.series_type else 'Unknown'} &nbsp;·&nbsp;
                Data Hash: {(result.data_hash or '')[:28]}...
            </div>""", unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    with st.expander("◈  Export"):
        col_e1, col_e2, col_gap = st.columns([2, 2, 6])

        with col_e1:
            if st.button("↓ Generate Excel", key="fx_export_btn",
                         use_container_width=True):
                xlsx = export_excel(s, result, st.session_state.fx_series_id)
                st.download_button(
                    "Download Excel",
                    xlsx,
                    file_name=(
                        f"foresight_x_{st.session_state.fx_series_id}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )

        with col_e2:
            if st.button("↓ Generate CSV", key="fx_csv_btn",
                         use_container_width=True):
                fc_df = pd.DataFrame({
                    "Date":     result.forecast_index.strftime("%Y-%m-%d"),
                    "Forecast": result.point_forecast,
                })
                st.download_button(
                    "Download CSV",
                    fc_df.to_csv(index=False).encode(),
                    file_name=f"foresight_x_{st.session_state.fx_series_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_nav()

    has_result = st.session_state.fx_result is not None
    t1 = "◈  I — Data & Analysis"
    t2 = "◈  II — Forecast Canvas" + ("" if has_result else " 🔒")
    t3 = "◈  III — Results & Export" + ("" if has_result else " 🔒")

    tab1, tab2, tab3 = st.tabs([t1, t2, t3])
    with tab1: render_tab1()
    with tab2: render_tab2()
    with tab3: render_tab3()


if __name__ == "__main__":
    main()
