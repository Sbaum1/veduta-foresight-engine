"""
veduta_foresight_app/styles.py
VEDUTA Foresight X — Upgraded Brand Identity
Native config.toml theme + enhanced CSS for 8.5/10 UI quality.
"""

# ── Brand Colors ──────────────────────────────────────────────────────────────
NOCTURNE_BLACK  = "#07080F"
VENETIAN_GOLD   = "#C8974A"
CHARCOAL        = "#1A1D2E"
SLATE           = "#2A2D3E"

FS_SILVER       = "#C8D4E0"
FS_MIDNIGHT     = "#0D1829"
FS_ELECTRIC     = "#4A9EE0"
FS_GOLD         = "#C8974A"
FS_GREEN        = "#4AB88A"
FS_RED          = "#C85A4A"
FS_AMBER        = "#E8A832"

WHITE_80        = "rgba(255,255,255,0.80)"
WHITE_55        = "rgba(255,255,255,0.55)"
WHITE_25        = "rgba(255,255,255,0.25)"

# ── Plotly Base Layout ────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor = NOCTURNE_BLACK,
    plot_bgcolor  = "#0A1020",
    font          = dict(family="DM Mono, monospace", color=WHITE_55, size=11),
    margin        = dict(l=60, r=20, t=44, b=40),
    xaxis         = dict(
        gridcolor = "rgba(255,255,255,0.05)",
        linecolor = "rgba(255,255,255,0.08)",
        tickfont  = dict(color=WHITE_55, size=10),
        showgrid  = True,
    ),
    yaxis         = dict(
        gridcolor = "rgba(255,255,255,0.05)",
        linecolor = "rgba(255,255,255,0.08)",
        tickfont  = dict(color=WHITE_55, size=10),
        showgrid  = True,
    ),
    legend        = dict(
        bgcolor     = "rgba(7,8,15,0.85)",
        bordercolor = "rgba(200,212,224,0.15)",
        borderwidth = 1,
        font        = dict(color=WHITE_55, size=10),
        itemsizing  = "constant",
    ),
    hoverlabel    = dict(
        bgcolor  = "#1A1D2E",
        font     = dict(color=WHITE_80, size=11, family="DM Mono, monospace"),
        bordercolor = FS_SILVER,
    ),
    hovermode     = "x unified",
    dragmode      = "pan",
)

# ── Enhanced CSS ──────────────────────────────────────────────────────────────
FORESIGHT_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap');

/* ── CSS Custom Properties — VEDUTA Design Tokens ── */
:root {
    --veduta-black:      #07080F;
    --veduta-charcoal:   #1A1D2E;
    --veduta-midnight:   #0D1829;
    --veduta-slate:      #2A2D3E;
    --veduta-silver:     #C8D4E0;
    --veduta-gold:       #C8974A;
    --veduta-electric:   #4A9EE0;
    --veduta-teal:       #52B8B2;
    --veduta-green:      #4AB88A;
    --veduta-red:        #C85A4A;
    --veduta-amber:      #E8A832;
    --veduta-white-80:   rgba(255,255,255,0.80);
    --veduta-white-55:   rgba(255,255,255,0.55);
    --veduta-white-30:   rgba(255,255,255,0.30);
    --veduta-silver-20:  rgba(200,212,224,0.20);
    --veduta-silver-10:  rgba(200,212,224,0.10);
    --veduta-gold-20:    rgba(200,151,74,0.20);
    --veduta-gold-10:    rgba(200,151,74,0.10);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    background-color: var(--veduta-black) !important;
    color: var(--veduta-white-80) !important;
}
.stApp {
    background: var(--veduta-black) !important;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Wordmark & Branding ── */
.fx-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px;
    font-weight: 300;
    letter-spacing: 0.12em;
    color: var(--veduta-silver);
    text-transform: uppercase;
    line-height: 1.1;
    transition: opacity 0.2s;
}
.fx-edition {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.35em;
    color: var(--veduta-electric);
    text-transform: uppercase;
    margin-top: 1px;
}
.fx-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: rgba(200,212,224,0.4);
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── M3 Badge ── */
.m3-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(200,151,74,0.08);
    border: 1px solid rgba(200,151,74,0.28);
    border-radius: 4px;
    padding: 7px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--veduta-gold);
    transition: background 0.2s, border-color 0.2s;
}
.m3-badge:hover {
    background: rgba(200,151,74,0.12);
    border-color: rgba(200,151,74,0.45);
}
.m3-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--veduta-gold);
    animation: m3-pulse 2.5s ease-in-out infinite;
}
@keyframes m3-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── Navigation divider ── */
.fx-nav-divider {
    border: none;
    border-bottom: 1px solid rgba(200,212,224,0.1);
    margin: 0 0 28px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid rgba(200,212,224,0.1) !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.28) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 16px 28px !important;
    margin: 0 !important;
    transition: color 0.2s, border-color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(200,212,224,0.6) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--veduta-silver) !important;
    border-bottom: 2px solid var(--veduta-silver) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding: 32px 0 !important;
}

/* ── Section headers ── */
.fx-section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 24px;
    font-weight: 300;
    color: rgba(255,255,255,0.92);
    letter-spacing: 0.03em;
    margin-bottom: 4px;
    line-height: 1.2;
}
.fx-section-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(200,212,224,0.38);
    margin-bottom: 24px;
}

/* ── Metric Cards ── */
.fx-metric-card {
    background: rgba(13,24,41,0.75);
    border: 1px solid rgba(200,212,224,0.1);
    border-radius: 8px;
    padding: 16px 18px;
    transition: border-color 0.2s, background 0.2s;
}
.fx-metric-card:hover {
    border-color: rgba(200,212,224,0.2);
    background: rgba(13,24,41,0.9);
}
.fx-metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 8px;
}
.fx-metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 300;
    color: var(--veduta-silver);
    line-height: 1;
}
.fx-metric-value.gold  { color: var(--veduta-gold); }
.fx-metric-value.teal  { color: var(--veduta-teal); }
.fx-metric-value.green { color: var(--veduta-green); }
.fx-metric-value.electric { color: var(--veduta-electric); }
.fx-metric-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.28);
    margin-top: 5px;
}

/* ── Progress bar (engine run) ── */
.fx-progress-bar {
    height: 2px;
    background: linear-gradient(90deg, var(--veduta-electric), var(--veduta-teal));
    border-radius: 1px;
    animation: fx-progress 1.8s ease-in-out infinite;
    margin: 12px 0;
}
@keyframes fx-progress {
    0%   { width: 0%; opacity: 1; }
    70%  { width: 85%; opacity: 1; }
    100% { width: 100%; opacity: 0; }
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid rgba(200,212,224,0.3) !important;
    color: var(--veduta-silver) !important;
    padding: 10px 22px !important;
    border-radius: 4px !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: rgba(200,212,224,0.07) !important;
    border-color: rgba(200,212,224,0.5) !important;
}
.stButton > button:active {
    background: rgba(200,212,224,0.12) !important;
}
/* Primary button */
[data-testid="baseButton-primary"] > button,
.stButton > button[kind="primary"] {
    background: var(--veduta-electric) !important;
    border-color: var(--veduta-electric) !important;
    color: var(--veduta-black) !important;
    font-weight: 500 !important;
}
[data-testid="baseButton-primary"] > button:hover,
.stButton > button[kind="primary"]:hover {
    background: #5AAEF0 !important;
    border-color: #5AAEF0 !important;
}

/* ── Inputs ── */
.stTextArea textarea,
.stTextInput input {
    background: var(--veduta-midnight) !important;
    border: 1px solid rgba(200,212,224,0.15) !important;
    border-radius: 6px !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    transition: border-color 0.18s !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(74,158,224,0.5) !important;
    box-shadow: 0 0 0 2px rgba(74,158,224,0.1) !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--veduta-midnight) !important;
    border: 1px solid rgba(200,212,224,0.15) !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 6px !important;
}
.stSelectbox [data-baseweb="popover"] {
    background: var(--veduta-charcoal) !important;
    border: 1px solid rgba(200,212,224,0.15) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: rgba(74,158,224,0.25) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--veduta-electric) !important;
}

/* ── Checkboxes ── */
.stCheckbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    color: rgba(255,255,255,0.55) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: rgba(10,16,32,0.65) !important;
    border: 1px solid rgba(200,212,224,0.1) !important;
    border-radius: 7px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: rgba(200,212,224,0.6) !important;
    transition: background 0.18s, border-color 0.18s !important;
    padding: 14px 18px !important;
}
.streamlit-expanderHeader:hover,
[data-testid="stExpander"] summary:hover {
    background: rgba(10,16,32,0.9) !important;
    border-color: rgba(200,212,224,0.2) !important;
    color: var(--veduta-silver) !important;
}
.streamlit-expanderContent,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: rgba(10,16,32,0.35) !important;
    border: 1px solid rgba(200,212,224,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 7px 7px !important;
    padding: 16px 20px !important;
}

/* ── Model rank table ── */
.fx-model-header {
    display: flex;
    align-items: center;
    padding: 8px 16px 8px 16px;
    border-bottom: 1px solid rgba(200,212,224,0.15);
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(200,212,224,0.45);
    margin-bottom: 2px;
}
.fx-model-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    transition: background 0.15s;
    border-radius: 3px;
}
.fx-model-row:hover {
    background: rgba(200,212,224,0.04);
}
.fx-rank-num {
    color: rgba(200,212,224,0.35);
    min-width: 32px;
    font-size: 9px;
}
.fx-rank-top {
    color: var(--veduta-electric);
    font-size: 10px;
    font-weight: 500;
}
.fx-model-name {
    color: rgba(255,255,255,0.78);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.fx-mase-val  { min-width: 72px; text-align: right; }
.fx-mase-good { color: var(--veduta-green); }
.fx-mase-ok   { color: var(--veduta-amber); }
.fx-mase-poor { color: #E88A6A; }
.fx-mase-bad  { color: var(--veduta-red); }
.fx-weight-col { min-width: 100px; padding-left: 12px; }
.fx-weight-bar-bg {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 3px;
    overflow: hidden;
}
.fx-weight-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--veduta-electric), var(--veduta-teal));
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.fx-dq-badge {
    font-size: 8px;
    color: var(--veduta-red);
    background: rgba(200,90,74,0.1);
    border: 1px solid rgba(200,90,74,0.22);
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 8px;
    letter-spacing: 0.1em;
}
.fx-ensemble-badge {
    font-size: 8px;
    color: var(--veduta-teal);
    background: rgba(82,184,178,0.1);
    border: 1px solid rgba(82,184,178,0.22);
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 8px;
    letter-spacing: 0.1em;
}

/* ── Signal/Source toggle ── */
.fx-mode-container {
    display: flex;
    align-items: center;
    background: rgba(13,24,41,0.8);
    border: 1px solid rgba(200,212,224,0.18);
    border-radius: 20px;
    padding: 3px 4px;
    gap: 2px;
}

/* ── Fragment spinner ── */
.fx-engine-running {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: rgba(74,158,224,0.06);
    border: 1px solid rgba(74,158,224,0.2);
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: rgba(74,158,224,0.8);
}
.fx-spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(74,158,224,0.2);
    border-top-color: var(--veduta-electric);
    border-radius: 50%;
    animation: fx-spin 0.8s linear infinite;
}
@keyframes fx-spin {
    to { transform: rotate(360deg); }
}

/* ── Engine result banner ── */
.fx-result-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 20px;
    background: rgba(74,184,138,0.06);
    border: 1px solid rgba(74,184,138,0.2);
    border-radius: 8px;
    margin-top: 12px;
}
.fx-result-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--veduta-green);
}
.fx-result-text {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: rgba(74,184,138,0.85);
}

/* ── Chart controls panel ── */
.fx-controls {
    background: rgba(13,24,41,0.6);
    border: 1px solid rgba(200,212,224,0.08);
    border-radius: 7px;
    padding: 14px 16px;
}
.fx-controls-label {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(200,212,224,0.35);
    margin-bottom: 10px;
}

/* ── Locked tab ── */
.fx-locked {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 100px 40px;
    text-align: center;
}
.fx-locked-icon {
    font-size: 52px;
    opacity: 0.18;
    margin-bottom: 20px;
    animation: fx-breathe 3s ease-in-out infinite;
}
@keyframes fx-breathe {
    0%, 100% { opacity: 0.18; }
    50%       { opacity: 0.25; }
}
.fx-locked-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 300;
    color: rgba(255,255,255,0.28);
    margin-bottom: 10px;
}
.fx-locked-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    color: rgba(200,212,224,0.18);
    text-transform: uppercase;
}

/* ── Credential panel ── */
.fx-credential {
    background: rgba(200,151,74,0.04);
    border: 1px solid rgba(200,151,74,0.18);
    border-radius: 10px;
    padding: 22px 26px;
}
.fx-credential-label {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(200,151,74,0.55);
    margin-bottom: 18px;
}
.fx-credential-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px;
    font-weight: 300;
    line-height: 1;
}
.fx-credential-desc {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    color: rgba(255,255,255,0.28);
    margin-top: 5px;
    letter-spacing: 0.1em;
}

/* ── Audit trail block ── */
.fx-audit {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: rgba(200,212,224,0.32);
    padding: 12px 16px;
    background: rgba(10,16,32,0.55);
    border-radius: 5px;
    border-left: 2px solid rgba(200,212,224,0.15);
    line-height: 1.8;
}

/* ── Dataframe overrides ── */
.stDataFrame { background: transparent !important; }
[data-testid="stDataFrame"] th {
    background: rgba(13,24,41,0.9) !important;
    color: rgba(200,212,224,0.65) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(200,212,224,0.12) !important;
}
[data-testid="stDataFrame"] td {
    background: rgba(13,24,41,0.4) !important;
    color: rgba(255,255,255,0.65) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}

hr { border: none; border-bottom: 1px solid rgba(200,212,224,0.08); margin: 20px 0; }

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid rgba(200,212,224,0.28) !important;
    color: var(--veduta-silver) !important;
    border-radius: 4px !important;
}

/* ── Notification toasts ── */
[data-testid="stNotification"] {
    background: var(--veduta-charcoal) !important;
    border: 1px solid rgba(200,212,224,0.15) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Caption text ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    color: rgba(200,212,224,0.38) !important;
    letter-spacing: 0.08em !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    color: rgba(74,158,224,0.7) !important;
    letter-spacing: 0.12em !important;
}
</style>
"""
