"""Modern, minimalist theme for the Omni_Stocks dashboard.

Token-based design system:
- Spacing scale: --space-1 (4px) ... --space-8 (40px)
- Type scale: --text-micro ... --text-3xl
- Single accent (#06b6d4 cyan); gradient kept only as legacy fallback
- Hairline borders, no glassmorphic blur on cards
- Tabular numerals via JetBrains Mono + font-variant-numeric

Public API:
  inject_theme()                          -- call once at app start
  hero(title, subtitle, kicker)           -- top-of-page hero band
  pill(text, kind)                        -- small status chip
  regime_gradient(regime)                 -- gradient for regime banner
"""
from __future__ import annotations

import streamlit as st


_FONT_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""


_CSS = """
<style>
:root {
    /* ---- spacing scale ---- */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-7: 32px;
    --space-8: 40px;

    /* ---- type scale ---- */
    --text-micro: 0.68rem;
    --text-xs: 0.78rem;
    --text-sm: 0.88rem;
    --text-base: 1rem;
    --text-lg: 1.25rem;
    --text-xl: 1.5rem;
    --text-2xl: 1.75rem;
    --text-3xl: 2.25rem;

    /* ---- radii ---- */
    --r-sm: 8px;
    --r-md: 12px;
    --r-lg: 16px;

    /* ---- accents (single solid + gradient kept for hero/highlights only) ---- */
    --accent: #06b6d4;
    --accent-strong: #0891b2;
    --accent-soft: rgba(6, 182, 212, 0.10);
    --accent-glow: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    --accent-glow-soft: linear-gradient(135deg, rgba(6,182,212,0.16) 0%, rgba(139,92,246,0.16) 100%);

    /* ---- light tokens ---- */
    --surface-0: #f8fafc;
    --surface-1: #ffffff;
    --surface-2: #f1f5f9;
    --border-subtle: rgba(15, 23, 42, 0.08);
    --border-strong: rgba(15, 23, 42, 0.16);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --kpi-positive: #059669;
    --kpi-negative: #dc2626;
    --kpi-neutral: #94a3b8;
    --regime-bull: #16a34a;
    --regime-chop: #ca8a04;
    --regime-bear: #dc2626;
    --regime-crisis: #7c3aed;

    /* ---- legacy fallbacks (so unmigrated components keep working) ---- */
    --glass-bg: var(--surface-1);
    --glass-bg-strong: var(--surface-1);
    --glass-border: var(--border-subtle);
    --glass-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

@media (prefers-color-scheme: dark) {
    :root {
        --surface-0: #0a0e1a;
        --surface-1: #11172a;
        --surface-2: #182037;
        --border-subtle: rgba(148, 163, 184, 0.10);
        --border-strong: rgba(148, 163, 184, 0.22);
        --text-primary: #e6ecf5;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --kpi-positive: #34d399;
        --kpi-negative: #f87171;
        --kpi-neutral: #64748b;
        --regime-bull: #22c55e;
        --regime-chop: #eab308;
        --regime-bear: #ef4444;
        --regime-crisis: #a855f7;
        --glass-bg: var(--surface-1);
        --glass-bg-strong: var(--surface-1);
        --glass-border: var(--border-subtle);
        --glass-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
}

/* ---- base ---- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary) !important;
    font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
}
.stApp {
    background:
      radial-gradient(800px 600px at 90% -10%, rgba(6,182,212,0.08), transparent 60%),
      radial-gradient(700px 500px at -10% 110%, rgba(139,92,246,0.07), transparent 60%),
      var(--surface-0) !important;
    background-attachment: fixed !important;
}
.main .block-container {
    padding-top: var(--space-6) !important;
    padding-bottom: var(--space-8) !important;
    max-width: 1400px !important;
}
h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.015em;
    font-feature-settings: 'cv02', 'cv03';
}
h1 { font-weight: 800 !important; font-size: var(--text-3xl) !important; }
h2 { font-weight: 700 !important; font-size: var(--text-2xl) !important; }
h3, h4, h5 { font-weight: 600 !important; }
h5 {
    font-size: var(--text-sm) !important;
    color: var(--text-secondary) !important;
    margin-bottom: var(--space-3) !important;
}
code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

/* ---- canonical card ---- */
.card, .glass, .glass-strong, .signal-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg);
    padding: var(--space-5) var(--space-6);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: var(--space-3);
    transition: border-color 150ms ease;
}
.card:hover, .signal-card:hover, .glass:hover {
    border-color: var(--border-strong);
    transform: none !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.card-tight { padding: var(--space-3) var(--space-4); }

/* ---- streamlit container border override ---- */
[data-testid="stContainer"][class*="border"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

[data-testid="stHorizontalBlock"] { gap: var(--space-4) !important; }

/* ---- micro section label ---- */
.section-label {
    font-size: var(--text-micro);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--text-muted);
    margin: 0 0 var(--space-3) 0;
}

/* ---- hairline divider ---- */
.hairline {
    border: 0;
    border-top: 1px solid var(--border-subtle);
    margin: var(--space-5) 0;
}
hr {
    border: 0 !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: var(--space-5) 0 !important;
}

/* ---- stat blocks ---- */
.stat {
    display: flex; flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-3) 0;
}
.stat-label {
    font-size: var(--text-micro);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.10em;
}
.stat-value {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--text-2xl);
    font-weight: 700;
    line-height: 1.1;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
}
.stat-value.positive { color: var(--kpi-positive); }
.stat-value.negative { color: var(--kpi-negative); }
.stat-value.accent   { color: var(--accent); }
.stat-delta {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--text-xs);
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
}

/* ---- pillar bar list ---- */
.pillar-row {
    display: grid;
    grid-template-columns: 110px 1fr 48px;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) 0;
    border-top: 1px solid var(--border-subtle);
}
.pillar-row:first-child { border-top: 0; }
.pillar-label {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    font-weight: 500;
}
.pillar-track {
    position: relative;
    height: 6px;
    background: var(--surface-2);
    border-radius: 999px;
    overflow: hidden;
}
.pillar-fill {
    position: absolute; left: 0; top: 0;
    height: 100%;
    background: var(--text-secondary);
    border-radius: 999px;
}
.pillar-fill.is-top { background: var(--accent); }
.pillar-score {
    text-align: right;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
}

/* ---- streamlit metric override (clean stat style) ---- */
[data-testid="stMetric"] {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: var(--space-3) 0 !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
[data-testid="stMetric"]:hover { transform: none !important; box-shadow: none !important; }
[data-testid="stMetricLabel"] {
    font-size: var(--text-micro) !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-weight: 700 !important;
    font-size: var(--text-2xl) !important;
    color: var(--text-primary) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-variant-numeric: tabular-nums;
    line-height: 1.15 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: var(--text-xs) !important;
    color: var(--text-muted) !important;
    font-variant-numeric: tabular-nums;
}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface-1) !important;
    border-right: 1px solid var(--border-subtle) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* ---- buttons ---- */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: 0 !important;
    border-radius: var(--r-md) !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.15rem !important;
    box-shadow: 0 1px 2px rgba(6, 182, 212, 0.25);
    transition: background 150ms ease;
}
.stButton > button:hover {
    background: var(--accent-strong) !important;
    transform: none !important;
    box-shadow: 0 2px 4px rgba(6, 182, 212, 0.30);
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: none;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--surface-2) !important;
}

/* ---- inputs ---- */
.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div > input,
.stDateInput > div > div > input, .stNumberInput > div > div > input {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-md) !important;
    backdrop-filter: none !important;
}
.stSelectbox > div > div:hover, .stTextInput > div > div > input:hover {
    border-color: var(--border-strong) !important;
}

/* ---- slider ---- */
.stSlider [data-baseweb="slider"] > div > div > div {
    background: var(--accent) !important;
}

/* ---- tabs (sticky, animated) ---- */
.stTabs [role="tablist"] {
    gap: var(--space-1);
    background: var(--surface-0);
    border-bottom: 1px solid rgba(148,163,184,0.10);
    padding: 4px 4px 0 4px;
    /* Stick to the top of the viewport when the user scrolls within a tab */
    position: sticky;
    top: 0;
    z-index: 50;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    overflow-x: auto;
    scrollbar-width: thin;
    box-shadow: 0 1px 0 rgba(148,163,184,0.06);
}
.stTabs [role="tablist"]::-webkit-scrollbar { height: 4px; }
.stTabs [role="tablist"]::-webkit-scrollbar-thumb {
    background: var(--border-subtle); border-radius: 2px;
}
.stTabs [role="tab"] {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.55rem 1rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    backdrop-filter: none;
    white-space: nowrap;
    position: relative;
    transition: color 180ms cubic-bezier(0.16, 1, 0.3, 1),
                  border-color 180ms cubic-bezier(0.16, 1, 0.3, 1),
                  transform 140ms ease;
}
.stTabs [role="tab"]:hover {
    color: var(--text-primary) !important;
    transform: translateY(-1px);
}
.stTabs [role="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    background: transparent !important;
    font-weight: 600 !important;
    /* Animated cyan→violet underline indicator */
    border-bottom: 2px solid transparent !important;
    background-image: linear-gradient(transparent calc(100% - 2px),
                                        var(--accent) calc(100% - 2px),
                                        #8b5cf6 100%) !important;
}
/* Smoother tab-panel fade with ease-out-expo curve */
.stTabs [role="tabpanel"] {
    animation: tabFadeIn 280ms cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- KPI metric tiles: subtle hover lift ---- */
[data-testid="stMetric"] {
    transition: transform 180ms cubic-bezier(0.16, 1, 0.3, 1),
                  box-shadow 220ms ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px -8px rgba(6, 182, 212, 0.25) !important;
}

/* ---- Dataframe row hover (cyan tint) ---- */
[data-testid="stDataFrame"] [data-testid="stTable"] tr:hover,
[data-testid="stDataFrame"] table tbody tr:hover {
    background: rgba(6, 182, 212, 0.06) !important;
    transition: background 120ms ease;
}

/* ---- Slider track gradient (brand cyan→violet) ---- */
.stSlider [data-baseweb="slider"] [role="progressbar"] {
    background: linear-gradient(90deg, var(--accent), #8b5cf6) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.15) !important;
    transition: box-shadow 160ms ease, transform 120ms ease;
}
.stSlider [data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(6, 182, 212, 0.22) !important;
    transform: scale(1.05);
}

/* ---- Button micro-interactions: tactile :active scale ---- */
.stButton > button {
    transition: background 160ms ease,
                  border-color 160ms ease,
                  transform 80ms cubic-bezier(0.16, 1, 0.3, 1),
                  box-shadow 160ms ease !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* ---- Progress bar: smoother brand colour ---- */
.stProgress [role="progressbar"] > div > div {
    background: linear-gradient(90deg, var(--accent), #8b5cf6) !important;
    transition: width 200ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* ---- expander ---- */
[data-testid="stExpander"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-md) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: var(--text-sm) !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] {
    border-radius: var(--r-md);
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    backdrop-filter: none;
}

/* ---- hero band ---- */
.hero {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: var(--space-6) var(--space-7);
    border-radius: var(--r-lg);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: var(--space-5);
}
.hero h1 {
    margin: 0 !important;
    font-size: var(--text-2xl) !important;
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
.hero .kicker {
    text-transform: uppercase;
    font-size: var(--text-micro);
    letter-spacing: 0.18em;
    color: var(--text-muted);
    margin-bottom: var(--space-2);
    font-weight: 600;
}
.hero .subtitle {
    margin-top: var(--space-2);
    color: var(--text-secondary);
    font-size: var(--text-sm);
}

/* ---- pill / chip ---- */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: var(--text-micro);
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid currentColor;
    background: transparent;
    backdrop-filter: none;
    line-height: 1.5;
}
.pill-bull   { color: var(--regime-bull); }
.pill-chop   { color: var(--regime-chop); }
.pill-bear   { color: var(--regime-bear); }
.pill-crisis { color: var(--regime-crisis); }
.pill-long   { color: var(--kpi-positive); }
.pill-short  { color: var(--kpi-negative); }
.pill-flat   { color: var(--text-muted); }

/* ---- regime banner ---- */
.regime-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-4) var(--space-5);
    border-radius: var(--r-md);
    color: white;
    font-weight: 600;
    margin-bottom: var(--space-3);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10);
}
.regime-banner .label-big {
    font-size: var(--text-xl);
    letter-spacing: -0.01em;
}
.regime-banner .label-meta {
    font-size: var(--text-xs);
    font-weight: 500;
    opacity: 0.92;
}

/* ---- legacy signal card (other tabs still use this) ---- */
.signal-card .ticker {
    font-size: var(--text-2xl);
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    background: none;
    -webkit-text-fill-color: var(--text-primary);
}
.signal-card .meta {
    color: var(--text-muted);
    font-size: var(--text-xs);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.signal-card .conviction-bar {
    height: 4px;
    background: var(--surface-2);
    border-radius: 999px;
    overflow: hidden;
    margin: var(--space-2) 0;
}
.signal-card .conviction-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
}

/* ---- horizon table ---- */
.horizon-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-sm);
}
.horizon-table th {
    text-align: left;
    padding: var(--space-2) var(--space-3);
    font-size: var(--text-micro);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
}
.horizon-table td {
    padding: var(--space-3);
    border-bottom: 1px solid var(--border-subtle);
    font-variant-numeric: tabular-nums;
}
.horizon-table tr:last-child td { border-bottom: 0; }
.horizon-table .h-label { font-weight: 600; color: var(--text-primary); }
.horizon-table .h-sub   { color: var(--text-muted); font-size: var(--text-xs); }
.horizon-table .h-num   { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.horizon-table .h-range { color: var(--text-muted); font-size: var(--text-xs); font-family: 'JetBrains Mono', monospace; }

/* ---- proba bar ---- */
.proba-bar {
    position: relative;
    height: 6px;
    background: var(--surface-2);
    border-radius: 999px;
    overflow: visible;
}
.proba-bar .midline {
    position: absolute; left: 50%; top: -3px;
    width: 1px; height: 12px;
    background: var(--border-strong);
}
.proba-bar .fill {
    position: absolute; left: 0; top: 0; height: 100%;
    background: var(--accent);
    border-radius: 999px;
}
.proba-bar .blended-marker {
    position: absolute; top: -2px; width: 2px; height: 10px;
    background: var(--text-primary);
}

/* ---- streamlit chrome cleanup ---- */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ---- scrollbar polish ---- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* =====================================================================
   Phase 22 — Cutting-edge CSS upgrades (UI1, UI2, UI3, UI5)
   All progressively-enhance: older browsers ignore unknown properties.
   ===================================================================== */

/* UI1 — @property registered custom property for animated gradients */
@property --glow-angle {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}
@keyframes _glow-rotate { to { --glow-angle: 360deg; } }
.prediction-card, .signal-card {
    position: relative;
    isolation: isolate;
}
.prediction-card::before, .signal-card::before {
    content: "";
    position: absolute; inset: -1px;
    border-radius: inherit;
    padding: 1px;
    background: conic-gradient(
        from var(--glow-angle, 0deg),
        transparent 0deg,
        color-mix(in oklch, var(--accent) 60%, transparent) 90deg,
        transparent 180deg,
        color-mix(in oklch, #8b5cf6 60%, transparent) 270deg,
        transparent 360deg
    );
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    animation: _glow-rotate 12s linear infinite;
    pointer-events: none;
    opacity: 0.6;
    z-index: -1;
}

/* UI2 — color-mix() utilities (replaces hardcoded rgba shades) */
.accent-tint-soft  { background: color-mix(in oklch, var(--accent) 8%,  transparent); }
.accent-tint-mid   { background: color-mix(in oklch, var(--accent) 16%, transparent); }
.accent-tint-strong{ background: color-mix(in oklch, var(--accent) 28%, transparent); }
.violet-tint-soft  { background: color-mix(in oklch, #8b5cf6 8%, transparent); }

/* UI3 — Container queries on the prediction card */
.prediction-card { container-type: inline-size; container-name: predcard; }
@container predcard (max-width: 640px) {
    .prediction-card .stats { grid-template-columns: 1fr 1fr !important; }
    .prediction-card h2 { font-size: 1.25rem !important; }
}
@container predcard (min-width: 900px) {
    .prediction-card .stats { grid-template-columns: repeat(4, 1fr) !important; }
}

/* UI5 — Scroll-driven hero parallax + KPI fade-in (Chrome 115+, falls back to no-op) */
@supports (animation-timeline: scroll()) {
    .hero {
        animation: _hero-shrink linear both;
        animation-timeline: scroll(root);
        animation-range: 0 240px;
    }
    @keyframes _hero-shrink {
        to {
            transform: scale(0.94) translateY(-6px);
            opacity: 0.62;
            filter: blur(1px);
        }
    }
    [data-testid="stMetric"] {
        animation: _kpi-rise linear both;
        animation-timeline: view();
        animation-range: entry 0% entry 60%;
    }
    @keyframes _kpi-rise {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
}

/* UI4 — View Transitions hookup (CSS half; JS half in inject_theme) */
@supports (view-transition-name: auto) {
    ::view-transition-old(tabpanel),
    ::view-transition-new(tabpanel) {
        animation-duration: 280ms;
        animation-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
    }
}

/* UI8 — Sparkline polishing */
.sparkline {
    display: inline-block; vertical-align: middle;
    width: 80px; height: 18px;
}
.sparkline polyline { fill: none; stroke-width: 1.6; }
.sparkline.up   polyline { stroke: var(--accent); }
.sparkline.down polyline { stroke: var(--regime-bear); }

/* UI9 — Cmd+K palette modal styling */
.omni-cmdk-overlay {
    position: fixed; inset: 0; background: rgba(7, 11, 22, 0.55);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    z-index: 9999; display: none; align-items: flex-start; justify-content: center;
    padding-top: 14vh;
}
.omni-cmdk-overlay.open { display: flex; }
.omni-cmdk-modal {
    width: min(560px, 92vw);
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border-subtle); border-radius: 14px;
    box-shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
    overflow: hidden; animation: _cmdk-pop 240ms cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes _cmdk-pop {
    from { opacity: 0; transform: translateY(-8px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.omni-cmdk-input {
    width: 100%; border: 0; outline: 0; background: transparent;
    padding: 18px 20px; font-size: 1.05rem; color: var(--text-primary);
    border-bottom: 1px solid var(--border-subtle);
}
.omni-cmdk-list { list-style: none; margin: 0; padding: 6px; max-height: 50vh; overflow: auto; }
.omni-cmdk-list li {
    padding: 10px 14px; border-radius: 8px; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    color: var(--text-secondary);
}
.omni-cmdk-list li.active,
.omni-cmdk-list li:hover {
    background: color-mix(in oklch, var(--accent) 14%, transparent);
    color: var(--text-primary);
}
.omni-cmdk-tag {
    font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.04em;
}

</style>
"""


_VT_AND_CMDK_JS = """
<script>
(function () {
    if (window.__omniThemeBoot) return;
    window.__omniThemeBoot = true;

    // ---- UI4: View Transitions API for tab switches -------------------
    function wrapTabClicks() {
        if (!document.startViewTransition) return;
        document.querySelectorAll('[role="tab"]').forEach(function (t) {
            if (t.__omniVT) return;
            t.__omniVT = true;
            t.addEventListener('click', function (ev) {
                if (!document.startViewTransition) return;
                document.querySelectorAll('[role="tabpanel"]').forEach(function (p) {
                    p.style.viewTransitionName = 'tabpanel';
                });
            }, { capture: true });
        });
    }

    // ---- UI9: Cmd/Ctrl+K command palette ------------------------------
    function makePalette() {
        if (document.getElementById('omni-cmdk')) return;
        var ov = document.createElement('div');
        ov.id = 'omni-cmdk';
        ov.className = 'omni-cmdk-overlay';
        ov.innerHTML =
          '<div class="omni-cmdk-modal">' +
            '<input class="omni-cmdk-input" placeholder="Jump to a ticker — try SBIN, INFY, AAPL…" />' +
            '<ul class="omni-cmdk-list"></ul>' +
          '</div>';
        document.body.appendChild(ov);
        var input = ov.querySelector('input');
        var list  = ov.querySelector('ul');

        // Static seed: filter against suggestions parsed from any visible
        // selectbox or watchlist labels on the page. Cmd+K stays read-only.
        function harvestTickers() {
            var s = new Set();
            document.querySelectorAll('table td, .pill, [data-testid="stMetricValue"]').forEach(function (e) {
                var t = (e.textContent || '').trim();
                if (/^[A-Z][A-Z0-9.\-]{1,12}$/.test(t)) s.add(t);
            });
            return Array.from(s);
        }
        var pool = [];
        function refreshPool() { pool = harvestTickers(); }
        function render(query) {
            var q = (query || '').toUpperCase();
            list.innerHTML = '';
            var filtered = q
                ? pool.filter(function (p) { return p.indexOf(q) !== -1; })
                : pool.slice(0, 12);
            if (filtered.length === 0) {
                list.innerHTML = '<li><span class="omni-cmdk-tag">no matches</span></li>';
                return;
            }
            filtered.slice(0, 30).forEach(function (t, i) {
                var li = document.createElement('li');
                if (i === 0) li.classList.add('active');
                li.innerHTML = '<span>' + t + '</span><span class="omni-cmdk-tag">jump</span>';
                li.addEventListener('click', function () {
                    try {
                        var u = new URL(window.location.href);
                        u.searchParams.set('jump', t);
                        window.location.href = u.toString();
                    } catch (e) { console.warn('cmdk nav fail', e); }
                });
                list.appendChild(li);
            });
        }
        input.addEventListener('input', function () { render(input.value); });
        function open() { refreshPool(); render(''); ov.classList.add('open'); setTimeout(function () { input.focus(); }, 50); }
        function close() { ov.classList.remove('open'); input.value = ''; }
        ov.addEventListener('click', function (e) { if (e.target === ov) close(); });
        document.addEventListener('keydown', function (e) {
            if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
                e.preventDefault(); open();
            } else if (e.key === 'Escape') {
                close();
            }
        });
    }

    function init() {
        wrapTabClicks();
        makePalette();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    // Streamlit re-renders DOM frequently — re-bind tabs on mutation
    new MutationObserver(function () { wrapTabClicks(); })
        .observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


def inject_theme() -> None:
    """Inject the modern minimalist CSS + Inter/JetBrains Mono fonts."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    # UI4 + UI9 — View Transitions and Cmd+K palette
    st.markdown(_VT_AND_CMDK_JS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", kicker: str = "") -> None:
    parts = []
    if kicker:
        parts.append(f"<div class='kicker'>{kicker}</div>")
    parts.append(f"<h1>{title}</h1>")
    if subtitle:
        parts.append(f"<div class='subtitle'>{subtitle}</div>")
    st.markdown(f"<div class='hero'>{''.join(parts)}</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "flat") -> str:
    """Return a pill HTML string. `kind` is one of bull/chop/bear/crisis/long/short/flat."""
    return f"<span class='pill pill-{kind}'>{text}</span>"


_REGIME_GRADIENTS = {
    "bull":   "linear-gradient(135deg, #16a34a 0%, #06b6d4 100%)",
    "chop":   "linear-gradient(135deg, #ca8a04 0%, #f59e0b 100%)",
    "bear":   "linear-gradient(135deg, #dc2626 0%, #f43f5e 100%)",
    "crisis": "linear-gradient(135deg, #7c3aed 0%, #ec4899 100%)",
}


def regime_gradient(regime: str) -> str:
    return _REGIME_GRADIENTS.get(str(regime).lower(),
                                 "linear-gradient(135deg, #475569 0%, #334155 100%)")
