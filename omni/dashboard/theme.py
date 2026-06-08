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

/* Phase 24.13 — UI2 accent-tint / violet-tint utility classes were
   never applied in any .py file. Removed. */

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

/* ===========================================================================
   Phase 24.16 — Predict tab polish
   - .predict-action-btn aligns Run/Deep/Refresh buttons with the selectbox
     baseline (replaces brittle inline `height:1.85rem` spacers).
   - .stStatus inherits the signal-card surface so the status block doesn't
     look like a foreign element.
   - sidebar nav line-height + padding so the radiogroup labels (Predict /
     Watchlist / Portfolio …) don't get clipped by tight line boxes.
   - cursor reset so `cursor: wait` doesn't leak after spinners resolve.
   ============================================================================= */
.predict-action-btn {
    height: 1.85rem;
}
@media (max-width: 768px) {
    .predict-action-btn { height: 0; }
}

/* st.status surface — tone it down so it sits inside the page rhythm */
[data-testid="stStatus"] {
    border: 1px solid var(--border-subtle) !important;
    background: var(--surface-2, rgba(15, 23, 42, 0.4)) !important;
    border-radius: var(--r-md, 12px) !important;
    padding: 8px 12px !important;
    margin: 8px 0 !important;
}
[data-testid="stStatus"] [data-testid="stStatusContent"] {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: var(--text-sm);
}

/* Sidebar nav: prevent clipped descenders on radiogroup labels.
   IMPORTANT: zero out the global `.stApp p { margin-bottom: 12px }` rule
   on `<p>` tags Streamlit wraps the radio label text in — that 12px was
   what was clipping descenders / cutting names in half. */
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label div p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.5 !important;
    font-size: 0.92rem !important;
    color: inherit !important;
}
/* the inner div Streamlit wraps the markdown in — let it size naturally */
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {
    line-height: 1.5 !important;
    overflow: visible !important;
    flex: 1 1 auto;
    min-width: 0;
}
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] > * {
    margin: 0 !important;
    overflow: visible !important;
}

/* ===========================================================================
   OmniNav — cutting-edge sidebar section switcher (Phase 24.16)
   Replaces the default radio bullet UI with a modern segmented-pill nav:
     - sliding 3px accent bar on the left edge of the active item
     - smooth 280ms cubic-bezier slide between items (translateY)
     - per-item monogram glyph (CSS counter / nth-child mapping)
     - hover lifts background, focus ring uses --accent
     - active item: accent-tinted background + bold label + left bar glow
   No layout shift, no JS — purely CSS-driven, anchored on Streamlit's own
   `[role="radiogroup"]` semantics so click → state-change still works.
   ============================================================================= */
[data-testid="stSidebar"] [role="radiogroup"] {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 4px 6px;
}
/* hide default native radio dot */
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    opacity: 0 !important; pointer-events: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* base label = pill button */
[data-testid="stSidebar"] [role="radiogroup"] label {
    position: relative;
    cursor: pointer;
    width: 100%;
    /* Important: explicit min-height + flex so the text never clips. */
    min-height: 40px;
    display: flex !important;
    align-items: center;
    padding: 8px 12px 8px 32px !important;
    border-radius: 10px;
    background: transparent;
    transition: background 200ms ease, color 200ms ease, transform 200ms ease;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.005em;
    /* overflow VISIBLE so descenders + monogram are never clipped */
    overflow: visible !important;
    isolation: isolate;
    line-height: 1.5;
}
/* slim left accent bar — invisible by default */
[data-testid="stSidebar"] [role="radiogroup"] label::before {
    content: "";
    position: absolute;
    left: 8px; top: 50%;
    width: 3px; height: 0;
    border-radius: 2px;
    background: linear-gradient(180deg,
        var(--accent, #06b6d4),
        color-mix(in oklch, var(--accent, #06b6d4) 60%, #8b5cf6));
    box-shadow: 0 0 8px color-mix(in oklch, var(--accent, #06b6d4) 60%, transparent);
    transform: translateY(-50%);
    transition: height 260ms cubic-bezier(0.34, 1.56, 0.64, 1),
                opacity 200ms ease;
    opacity: 0;
}
/* monogram glyph — now on label::after directly (was nested on inner div).
   Position absolutely so it doesn't fight flex layout or clip the label. */
[data-testid="stSidebar"] [role="radiogroup"] label::after {
    position: absolute;
    left: 16px; top: 50%;
    transform: translateY(-50%);
    width: 14px;
    text-align: center;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-weight: 600;
    font-size: 0.85rem;
    color: color-mix(in oklch, var(--accent, #06b6d4) 70%, var(--text-muted, #64748b));
    transition: color 200ms ease, transform 200ms ease;
    pointer-events: none;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    color: var(--text-primary, #e6ecf5);
    background: color-mix(in oklch, var(--accent, #06b6d4) 6%, transparent);
}
/* active item — Streamlit marks aria-checked=true on the input;
   target via :has(). Fallback for engines without :has() below. */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    color: var(--text-primary, #e6ecf5) !important;
    font-weight: 700;
    background: color-mix(in oklch, var(--accent, #06b6d4) 14%, transparent);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {
    height: 22px;
    opacity: 1;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::after {
    color: var(--accent, #06b6d4);
    transform: translateY(-50%) scale(1.1);
}
/* glyph map by position — 12 sections in `section_options`.
   Use ::after's content (since the glyph is now on label::after directly). */
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(1)::after  { content: "◆"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(2)::after  { content: "★"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(3)::after  { content: "▣"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(4)::after  { content: "↕"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(5)::after  { content: "⚡"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(6)::after  { content: "▦"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(7)::after  { content: "◉"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(8)::after  { content: "∞"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(9)::after  { content: "⊿"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(10)::after { content: "$"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(11)::after { content: "⌚"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(12)::after { content: "⚙"; }

/* :has() polyfill — older Chromium gets a less-fancy active state via
   :focus-within (Streamlit moves focus on click, so this still works). */
@supports not selector(:has(*)) {
    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
        background: color-mix(in oklch, var(--accent, #06b6d4) 14%, transparent);
        color: var(--text-primary, #e6ecf5);
        font-weight: 700;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within::before {
        height: 22px;
        opacity: 1;
    }
}

/* ===========================================================================
   Production loading UX (Phase 24.16) — skeleton placeholders + a slim
   indeterminate top progress bar. Replaces the perceived-stuck spinner.
   Patterns inspired by Linear / Vercel / Stripe Dashboard:
     - skeleton boxes that match the eventual content's shape
     - shimmer animation (gradient sweep) for "actively loading" cue
     - 2px top progress bar with infinite slide for indeterminate work
   ============================================================================= */

/* Skeleton primitives — drop these into st.markdown() while waiting.
   Phase 28.2: contrast bumped from 6%/12%/6% -> 12%/26%/12% so the wave
   is clearly visible on dark themes. Previous contrast made the skeletons
   look static and the page felt frozen even when animation was running. */
.skel {
    display: block;
    background: linear-gradient(
        90deg,
        color-mix(in oklch, var(--text-primary, #e6ecf5) 12%, transparent) 0%,
        color-mix(in oklch, var(--text-primary, #e6ecf5) 26%, transparent) 50%,
        color-mix(in oklch, var(--text-primary, #e6ecf5) 12%, transparent) 100%
    );
    background-size: 200% 100%;
    border-radius: 6px;
    animation: skel-shimmer 1.2s linear infinite;
}
.skel-line   { height: 14px; width: 100%; margin: 6px 0; }
.skel-line-sm{ height: 10px; width: 40%;  margin: 4px 0; }
.skel-line-lg{ height: 28px; width: 65%;  margin: 10px 0; }
.skel-tile   { height: 96px; border-radius: 12px; }
.skel-pill   { display: inline-block; height: 22px; width: 80px;
               border-radius: 999px; vertical-align: middle; }
@keyframes skel-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Top indeterminate progress bar — Phase 28.2: 3px tall (was 2px) and the
   sliding glow widened from 38% -> 45% with a stronger box-shadow so the
   progress signal is visible even on bright dashboards. */
.top-progress {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    z-index: 9999;
    overflow: hidden;
    background: color-mix(in oklch, var(--accent, #06b6d4) 8%, transparent);
    pointer-events: none;
}
.top-progress::before {
    content: "";
    display: block;
    height: 100%;
    width: 45%;
    background: linear-gradient(90deg,
        transparent,
        var(--accent, #06b6d4),
        color-mix(in oklch, var(--accent, #06b6d4) 60%, #8b5cf6),
        transparent);
    border-radius: 2px;
    animation: top-progress-slide 1.1s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    box-shadow: 0 0 12px color-mix(in oklch, var(--accent, #06b6d4) 70%, transparent);
}
@keyframes top-progress-slide {
    0%   { transform: translateX(-120%); }
    100% { transform: translateX(310%); }
}
/* Phase 28.2: when prefers-reduced-motion is on, the CSS animations would
   otherwise be killed and the loading state looks frozen. Replace the
   sliding glow with a static-but-clearly-visible accent band so the user
   still gets a visible "we're working" affordance. The skeletons stay
   visible at the new higher contrast. */
@media (prefers-reduced-motion: reduce) {
    .skel { animation: none !important; opacity: 0.9; }
    .top-progress::before {
        animation: none !important;
        width: 100%;
        opacity: 0.85;
    }
}

/* ---------------------------------------------------------------------------
   Selectbox / dropdown text clipping fix (Phase 24.16, hardened pass 2)
   Streamlit selectbox renders the chosen value through several nested
   divs whose computed line-height collapses to 0 on some builds, eating
   descenders (g/p/y/q). We force every nested element to have a real
   line-height and visible overflow. The value-container also gets extra
   vertical padding so the cap-line doesn't kiss the border.
   --------------------------------------------------------------------------- */
[data-baseweb="select"],
[data-baseweb="select"] *,
.stSelectbox [data-baseweb="select"] * {
    line-height: 1.5 !important;
    overflow: visible !important;
}
[data-baseweb="select"] [data-baseweb="select-input"],
[data-baseweb="select"] > div:first-child {
    min-height: 44px !important;
    align-items: center !important;
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    min-height: 28px !important;
    line-height: 1.5 !important;
}
/* baseweb popover (the open dropdown menu) — same hardening */
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"] {
    line-height: 1.5 !important;
    padding: 8px 12px !important;
    min-height: 38px;
    display: flex;
    align-items: center;
    overflow: visible !important;
}
[data-baseweb="popover"] li *,
[data-baseweb="popover"] [role="option"] * {
    line-height: 1.5 !important;
    overflow: visible !important;
}

/* Cursor — when a spinner/status finishes, force the document back to
   default so the OS cursor doesn't keep showing the wait/busy state. */
body, .stApp {
    cursor: default;
}
[data-testid="stSpinner"] ~ * {
    cursor: default;
}

/* Stat grid hardening — make sure cells in the Results row stretch to the
   tallest sibling so labels stay baseline-aligned even when one cell hosts
   a pill instead of a numeric value. */
.stat {
    min-height: 96px;
    justify-content: flex-start;
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
                if (/^[A-Z][A-Z0-9.\\-]{1,12}$/.test(t)) s.add(t);
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


# ============================================================================
# Phase 23 — CUTTING-EDGE FX LAYER
# ----------------------------------------------------------------------------
# Stacked on top of the base theme. Each technique below is independently
# disable-able via CSS class toggle. Designed against current (2025) baseline:
# Chromium 120+, Firefox 121+, Safari 17.4+. Falls back gracefully where
# `@supports` is absent.
#
#  1. Cursor-reactive spotlight on cards (mousemove -> --mx/--my CSS vars)
#  2. Glassmorphism with backdrop-filter saturate+blur on metric containers
#  3. Three ambient conic-gradient orbs (regime-aware hues)
#  4. CSS spring easing via `linear()` timing function
#  5. Variable-font weight breathing on interactive headings
#  6. SVG turbulence noise grain overlay (extremely subtle)
#  7. Scroll-progress bar pinned to top, scroll-driven via animation-timeline
#  8. Skeleton shimmer using mask-image gradient sweep
#  9. Live-pulse rings on data-fresh indicators
# 10. @property animated counters (count-up on number metrics)
# 11. CSS-only 3D tilt cards (perspective + transform-style)
# 12. Magnetic buttons (cursor pulls primary action toward itself)
# 13. Glow trails on focused / active KPI tiles
# 14. Theme-color meta sync so OS chrome (mobile/PWA) follows the site
# 15. prefers-reduced-motion: every animation respects user preference
# ============================================================================

_ADVANCED_FX_CSS = """
<style>
/* ---- 4. Spring easing tokens (CSS linear() timing) -------------------- */
:root {
  --spring-snappy: linear(
    0, 0.009, 0.035, 0.078, 0.137, 0.211, 0.298, 0.396, 0.501, 0.609,
    0.717, 0.819, 0.911, 0.987, 1.043, 1.077, 1.087, 1.075, 1.045, 1.005,
    0.962, 0.924, 0.895, 0.879, 0.876, 0.886, 0.905, 0.93, 0.957, 0.984, 1
  );
  --spring-soft: linear(
    0, 0.014, 0.054, 0.119, 0.205, 0.308, 0.422, 0.541, 0.658, 0.768,
    0.864, 0.943, 1.0, 1.034, 1.046, 1.04, 1.022, 0.997, 0.972, 0.953,
    0.943, 0.943, 0.951, 0.964, 0.98, 0.994, 1
  );
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
}

/* ---- 6. SVG noise grain overlay (very subtle, no animation) ----------- */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 1000;
  opacity: 0.025;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.55'/></svg>");
}

/* ---- 3. Ambient conic-gradient orbs ----------------------------------- */
@property --orb-hue { syntax: "<number>"; inherits: false; initial-value: 220; }
.fx-orb {
  position: fixed;
  pointer-events: none;
  z-index: 0;
  width: 42vmax;
  height: 42vmax;
  border-radius: 50%;
  filter: blur(80px) saturate(1.4);
  opacity: 0.32;
  mix-blend-mode: screen;
  background:
    conic-gradient(
      from 0deg,
      hsl(var(--orb-hue) 80% 60%),
      hsl(calc(var(--orb-hue) + 60) 80% 55%),
      hsl(calc(var(--orb-hue) + 120) 80% 60%),
      hsl(var(--orb-hue) 80% 60%)
    );
  animation: fx-orb-spin 28s linear infinite,
             fx-orb-hue 14s ease-in-out infinite alternate;
}
.fx-orb.o1 { top: -20vmax; left: -15vmax; --orb-hue: 220; animation-delay: 0s, 0s; }
.fx-orb.o2 { bottom: -22vmax; right: -18vmax; --orb-hue: 290; animation-delay: -9s, -3s; opacity: 0.22; }
.fx-orb.o3 { top: 30vh; right: -25vmax; --orb-hue: 175; animation-delay: -18s, -7s; opacity: 0.18; }
@keyframes fx-orb-spin { to { transform: rotate(360deg); } }
@keyframes fx-orb-hue  { to { --orb-hue: 360; } }

/* ---- 2. Glassmorphism on metric containers --------------------------- */
[data-testid="stMetric"],
[data-testid="stExpander"] > details,
.stTabs [data-baseweb="tab-panel"] > div {
  position: relative;
  border-radius: 16px !important;
  background: color-mix(in oklch, var(--surface, #0f172a) 72%, transparent) !important;
  backdrop-filter: saturate(160%) blur(14px);
  -webkit-backdrop-filter: saturate(160%) blur(14px);
  border: 1px solid color-mix(in oklch, white 8%, transparent) !important;
  transition:
    transform 380ms var(--spring-soft),
    border-color 220ms var(--ease-out-expo),
    box-shadow 220ms var(--ease-out-expo);
}

/* ---- 1 + 13. Cursor-reactive spotlight overlay ----------------------- */
[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background:
    radial-gradient(
      420px circle at var(--mx, 50%) var(--my, 50%),
      color-mix(in oklch, var(--accent, #38bdf8) 28%, transparent) 0%,
      transparent 55%
    );
  opacity: 0;
  transition: opacity 320ms var(--ease-out-expo);
  z-index: 0;
}
[data-testid="stMetric"]:hover::before { opacity: 1; }
[data-testid="stMetric"] > * { position: relative; z-index: 1; }
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: color-mix(in oklch, var(--accent, #38bdf8) 50%, transparent) !important;
  box-shadow:
    0 18px 40px -22px color-mix(in oklch, var(--accent, #38bdf8) 60%, black),
    0 0 0 1px color-mix(in oklch, var(--accent, #38bdf8) 25%, transparent) inset;
}

/* Phase 24.13 — .fx-tilt 3D card class was never applied. Removed. */

/* ---- 12. Magnetic buttons -------------------------------------------- */
.stButton > button,
button[kind="primary"] {
  position: relative;
  border-radius: 12px !important;
  transition:
    transform 260ms var(--spring-snappy),
    background 200ms var(--ease-out-expo),
    box-shadow 200ms var(--ease-out-expo) !important;
}
.stButton > button:hover {
  transform: translate(
    calc((var(--mx, 50%) - 50%) * 0.08),
    calc((var(--my, 50%) - 50%) * 0.08)
  ) scale(1.02);
  box-shadow: 0 10px 28px -10px color-mix(in oklch, var(--accent, #38bdf8) 80%, black);
}

/* ---- 5. Variable-font weight breathing ------------------------------- */
h1, h2, h3, .hero h1 {
  font-variation-settings: "wght" 600;
  transition: font-variation-settings 600ms var(--ease-out-expo),
              letter-spacing 600ms var(--ease-out-expo);
}
h1:hover, h2:hover, h3:hover, .hero:hover h1 {
  font-variation-settings: "wght" 760;
  letter-spacing: -0.005em;
}

/* ---- 7. Scroll-progress bar (top, scroll-driven) --------------------- */
@supports (animation-timeline: scroll()) {
  body::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    height: 2px;
    width: 100%;
    transform-origin: 0 50%;
    transform: scaleX(0);
    background: linear-gradient(90deg,
      color-mix(in oklch, var(--accent, #38bdf8) 100%, transparent),
      color-mix(in oklch, #a855f7 100%, transparent));
    z-index: 1001;
    animation: fx-scroll-progress linear;
    animation-timeline: scroll(root);
  }
  @keyframes fx-scroll-progress { to { transform: scaleX(1); } }
}

/* Phase 24.13 — .fx-pulse pulse-ring class never applied. Removed. */

/* Phase 24.13 — .fx-skeleton shimmer class never applied. Removed. */

/* Phase 24.13 — .fx-counter @property animated counter never applied. */

/* Phase 24.13 — .bento-grid container queries never used. Removed. */

/* Phase 24.13 — old reduced-motion block referenced now-deleted classes.
   The Phase 24.12 universal `*, *::before, *::after` rule (later in the
   file) covers everything comprehensively. */

/* ---- Sidebar polish: floating nav with frosted glass ----------------- */
section[data-testid="stSidebar"] {
  background: color-mix(in oklch, var(--bg, #0b1220) 80%, transparent) !important;
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-right: 1px solid color-mix(in oklch, white 6%, transparent);
}

/* Tab strip — pill-style with sliding indicator */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  padding: 4px;
  border-radius: 14px;
  background: color-mix(in oklch, var(--surface, #0f172a) 60%, transparent);
  backdrop-filter: blur(10px);
  border: 1px solid color-mix(in oklch, white 6%, transparent);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  padding: 8px 14px !important;
  transition: all 280ms var(--ease-out-expo);
}
.stTabs [aria-selected="true"] {
  background: color-mix(in oklch, var(--accent, #38bdf8) 18%, transparent) !important;
  color: var(--accent, #38bdf8) !important;
}

/* DataFrame: subtle row hover with gradient border-left */
[data-testid="stDataFrame"] tbody tr {
  transition: background 180ms var(--ease-out-expo);
}
[data-testid="stDataFrame"] tbody tr:hover {
  background: color-mix(in oklch, var(--accent, #38bdf8) 8%, transparent) !important;
  box-shadow: inset 3px 0 0 var(--accent, #38bdf8);
}
</style>

<!-- The three ambient orbs -->
<div class="fx-orb o1" aria-hidden="true"></div>
<div class="fx-orb o2" aria-hidden="true"></div>
<div class="fx-orb o3" aria-hidden="true"></div>

<!-- 14. theme-color sync for browser chrome -->
<meta name="theme-color" content="#0b1220" id="fx-theme-meta">
"""

_ADVANCED_FX_JS = """
<script>
/* Phase 24.16 cull — _ADVANCED_FX_JS reduced from 75 lines to ~25.
   Dropped:
     - `bindCardSpotlight` mousemove on every metric/button/expander
       (4 listeners per element × tens of elements × every Streamlit rerun
       = wasted CPU for an effect that's barely visible). The spotlight
       --mx/--my CSS vars are still keyed off mouse position via plain
       :hover, so cards still highlight; we just don't need 100Hz updates.
     - dead `bindCounters` for .fx-counter (class no Python emits)
   Kept:
     - syncThemeMeta (mobile PWA chrome colour)
     - 'gd' shortcut to scroll to top
*/
(function(){
  if (window.__fxAdvancedBound) return;
  window.__fxAdvancedBound = true;

  function syncThemeMeta(){
    try {
      const accent = getComputedStyle(document.body).getPropertyValue('--bg').trim();
      const meta = document.getElementById('fx-theme-meta');
      if (meta && accent) meta.setAttribute('content', accent);
    } catch (e) {}
  }
  syncThemeMeta();

  // Keyboard: 'g' then 'd' jumps to dashboard top with smooth scroll.
  let lastKey = '', lastTime = 0;
  document.addEventListener('keydown', function(e){
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    const now = Date.now();
    if (lastKey === 'g' && e.key === 'd' && (now - lastTime) < 800) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    lastKey = e.key;
    lastTime = now;
  });
})();
</script>
"""


# ============================================================================
# Phase 23.5 — ULTRA FX
# ============================================================================
# Stacks ON TOP of Phase 23. Goal: visibly bold, modern, premium feel.
# Techniques used (all current 2025-2026 web best practices):
#  A. Aurora mesh gradient background (replaces flat bg)
#  B. Lighthouse cursor (radial gradient follows cursor across viewport)
#  C. SVG goo filter on the ambient orbs (liquid clumping)
#  D. Iridescent / holographic accent shimmer on KPI tiles
#  E. Custom cursor with delayed follower dot (premium feel)
#  F. Marquee ticker tape strip at very top of body
#  G. Slot-roll digit animation on metric values (.ultra-num)
#  H. Animated outline draw on cards on hover
#  I. Custom gradient scrollbar (WebKit + Firefox)
#  J. Staggered first-paint reveal (every direct child of main)
#  K. text-wrap: balance/pretty + variable-font extreme weight on hero H1
#  L. Confetti particle burst (canvas) - shift+C or .fx-celebrate trigger
#  M. Frosted floating bottom-right "live" status pill with pulse
#  N. Section title underline that draws on view-timeline scroll
# ============================================================================

_ULTRA_FX_SVG_DEFS = """
<!-- Phase 24.16 cull — goo filter, lighthouse, ultra-status, confetti canvas
     all removed. Goo opt-in was never set; lighthouse was already display:none;
     status indicator was hidden; confetti only fired on Shift+C and bloated CSS.
     The marquee div stays in app.py via render_marquee(). -->
"""

_ULTRA_FX_CSS = """
<style>
/* A. Aurora mesh background (Phase 24.7 — driven by --theme-c{1-4} vars
   so the user's Theme picker actually re-skins the page) */
@property --aurora-1 { syntax: "<percentage>"; inherits: false; initial-value: 0%; }
@property --aurora-2 { syntax: "<percentage>"; inherits: false; initial-value: 0%; }
:root {
  /* Default = "Auto / Cosmos" — overridden by app.py _theme_override */
  --theme-c1: oklch(0.42 0.16 260 / 0.55);
  --theme-c2: oklch(0.45 0.18 310 / 0.45);
  --theme-c3: oklch(0.40 0.14 180 / 0.40);
  --theme-c4: oklch(0.44 0.20 25 / 0.30);
  --theme-bg-stop1: #07090f;
  --theme-bg-stop2: #0a0d18;
  --theme-bg-stop3: #06070c;
  --theme-accent: #38bdf8;
  --theme-accent-2: #a855f7;
  --theme-accent-3: #22d3ee;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(at calc(20% + var(--aurora-1)) 12%,
      var(--theme-c1) 0%, transparent 55%),
    radial-gradient(at calc(80% - var(--aurora-2)) 18%,
      var(--theme-c2) 0%, transparent 55%),
    radial-gradient(at 65% calc(85% - var(--aurora-1)),
      var(--theme-c3) 0%, transparent 55%),
    radial-gradient(at 12% calc(78% + var(--aurora-2)),
      var(--theme-c4) 0%, transparent 60%),
    linear-gradient(180deg,
      var(--theme-bg-stop1) 0%,
      var(--theme-bg-stop2) 50%,
      var(--theme-bg-stop3) 100%) !important;
  background-attachment: fixed !important;
  animation: ultra-aurora 22s ease-in-out infinite alternate;
}
@keyframes ultra-aurora {
  0%   { --aurora-1:   0%; --aurora-2:   0%; }
  50%  { --aurora-1:  18%; --aurora-2:  12%; }
  100% { --aurora-1: -10%; --aurora-2:  20%; }
}

/* Phase 24.16 cull — .ultra-lighthouse rules removed (was already
   `display: none` further down anyway, plus the lighthouse DOM element
   was deleted from _ULTRA_FX_SVG_DEFS). Same for `body.fx-goo`. */

/* ===========================================================================
   PRODUCTION-GRADE METRIC TILES (Phase 24.1)
   The default Streamlit metric value runs ~44-48px which overflowed our
   tiles. Lock everything down: explicit value/label/delta sizes, generous
   padding, proper spacing rhythm. Box auto-grows; nothing escapes it.
   =========================================================================== */

[data-testid="stMetric"] {
  position: relative;
  /* tile box */
  padding: 16px 18px 14px 18px !important;
  border-radius: 16px !important;
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #0e1525 88%, transparent),
      color-mix(in oklch, #0a0f1c 92%, transparent)) !important;
  border: 1px solid color-mix(in oklch, white 7%, transparent) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.04),
    0 8px 24px -16px rgba(0, 0, 0, 0.5);
  /* layout */
  display: flex !important;
  flex-direction: column;
  gap: 6px;
  min-height: 120px;
  overflow: hidden;
  isolation: isolate;
  /* outline must always be transparent + offset 0; we use box-shadow only */
  outline: none !important;
  transition:
    transform 380ms var(--ease-out-expo),
    border-color 280ms var(--ease-out-expo),
    box-shadow 280ms var(--ease-out-expo),
    background 280ms var(--ease-out-expo);
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: color-mix(in oklch, #38bdf8 32%, transparent) !important;
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #111c30 90%, transparent),
      color-mix(in oklch, #0a0f1c 95%, transparent)) !important;
  /* Smooth multi-layer halo — no harsh outline ring */
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.07),
    0 0 0 1px color-mix(in oklch, #38bdf8 35%, transparent) inset,
    0 16px 48px -20px color-mix(in oklch, #38bdf8 55%, transparent),
    0 4px 18px -8px color-mix(in oklch, #a855f7 35%, transparent);
}
[data-testid="stMetric"] > * {
  position: relative;
  z-index: 1;
}

/* ---- LABEL: small-caps tracking ---- */
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] label {
  font-size: 10.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: rgba(148, 163, 184, 0.72) !important;
  line-height: 1.3 !important;
  margin: 0 !important;
  white-space: normal !important;
}

/* ---- VALUE: locked size, never overflows ---- */
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 30px !important;
  line-height: 1.15 !important;
  font-weight: 700 !important;
  letter-spacing: -0.015em !important;
  font-feature-settings: "tnum" 1, "ss01" 1 !important;
  font-variant-numeric: tabular-nums !important;
  color: #f1f5f9 !important;
  text-shadow: 0 1px 0 rgba(0, 0, 0, 0.4);
  margin: 2px 0 0 0 !important;
  padding: 0 !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}
/* Children of the value (Streamlit nests an extra <div>) */
[data-testid="stMetric"] [data-testid="stMetricValue"] > div,
[data-testid="stMetric"] [data-testid="stMetricValue"] > * {
  font-size: inherit !important;
  line-height: inherit !important;
  margin: 0 !important;
}

/* ---- DELTA: chip pill, INSIDE the box, with breathing room ---- */
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-size: 11px !important;
  font-weight: 600 !important;
  line-height: 1.25 !important;
  letter-spacing: 0.005em !important;
  padding: 3px 8px !important;
  border-radius: 6px !important;
  display: inline-flex !important;
  align-items: center;
  gap: 4px;
  margin: 4px 0 0 0 !important;
  width: fit-content;
  max-width: 100%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  background: color-mix(in oklch, #4ade80 14%, transparent) !important;
  color: #86efac !important;
  border: 1px solid color-mix(in oklch, #4ade80 25%, transparent);
}
/* Negative delta — red chip */
[data-testid="stMetric"] [data-testid="stMetricDelta"][class*="negative"],
[data-testid="stMetric"] [data-testid="stMetricDelta"]:has(svg[fill*="rgb(255"]) {
  background: color-mix(in oklch, #f87171 14%, transparent) !important;
  color: #fca5a5 !important;
  border-color: color-mix(in oklch, #f87171 25%, transparent) !important;
}

/* ---- Iridescent glow stays subtle and BEHIND content ---- */
[data-testid="stMetric"]::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(
      circle at var(--mx, 50%) var(--my, 50%),
      color-mix(in oklch, #38bdf8 18%, transparent) 0%,
      transparent 55%
    );
  opacity: 0;
  transition: opacity 320ms var(--ease-out-expo);
}
[data-testid="stMetric"]:hover::after { opacity: 0.7; }

/* E. Cursor & hover-halo (Phase 24.1 — production-grade)
   The previous "custom cursor" approach was fragile: Streamlit re-mounts the
   DOM constantly which orphaned the cursor div, and `mix-blend-mode` made it
   invisible on dark backgrounds. We now use the SYSTEM cursor (always
   visible, accessible, expected) and add a bold cyan halo that follows the
   cursor only when hovering interactive elements. This is the same pattern
   Linear / Vercel / Stripe Dashboard use. */

/* Hide the previous JS-driven cursor divs entirely */
.ultra-cursor, .ultra-cursor-follow { display: none !important; }

/* Make sure the system cursor is always shown */
html, body, * { cursor: auto; }
a, button, [role="button"], .stButton > button, [data-baseweb="tab"],
[data-testid="stMetric"], [data-testid="stExpander"] > details summary {
  cursor: pointer;
}
input, textarea, [contenteditable="true"], select { cursor: text; }
[disabled] { cursor: not-allowed; }

/* F. Marquee ticker tape (36px tall — large enough to read at a glance) */
.ultra-marquee {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 36px;
  z-index: 1002;
  overflow: hidden;
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #050810 92%, transparent),
      color-mix(in oklch, #0b1220 88%, transparent));
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border-bottom: 1px solid color-mix(in oklch, white 8%, transparent);
  box-shadow: 0 6px 24px -10px rgba(0, 0, 0, 0.6);
  font-feature-settings: "tnum" 1;
  font-variant-numeric: tabular-nums;
  font-size: 12.5px;
  font-weight: 500;
  letter-spacing: 0.02em;
  -webkit-mask-image: linear-gradient(90deg, transparent, black 5%, black 95%, transparent);
          mask-image: linear-gradient(90deg, transparent, black 5%, black 95%, transparent);
}
.ultra-marquee__track {
  display: inline-flex;
  gap: 32px;
  align-items: center;
  white-space: nowrap;
  padding: 10px 0;
  height: 36px;
  animation: ultra-mqr 90s linear infinite;
}
.ultra-marquee__track > span {
  color: rgba(148, 163, 184, 0.78);
  font-weight: 600;
  letter-spacing: 0.04em;
  font-size: 11px;
  text-transform: uppercase;
}
.ultra-marquee__track > b {
  color: rgba(241, 245, 249, 0.95);
  margin-right: 8px;
  font-weight: 700;
  font-size: 13px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.ultra-marquee__track > b.up {
  color: #4ade80;
  text-shadow: 0 0 12px rgba(74, 222, 128, 0.35);
}
.ultra-marquee__track > b.dn {
  color: #f87171;
  text-shadow: 0 0 12px rgba(248, 113, 113, 0.35);
}
.ultra-marquee:hover .ultra-marquee__track { animation-play-state: paused; }
@keyframes ultra-mqr { to { transform: translateX(-50%); } }
.stApp [data-testid="stHeader"] { top: 36px !important; }
.stApp { padding-top: 36px; }

/* Phase 24.13 — .ultra-digit / .ultra-digit__strip slot-roll CSS removed.
   The JS that emitted these elements was dead (no Python emits .ultra-num). */

/* H. Outline draw on cards (DEPRECATED Phase 24.4 — replaced by smooth halo
   in the Phase 24.1 metric tile block above. The outline-offset trick was
   creating a hard floating ring that looked unrefined. We now use layered
   box-shadows for a soft premium glow.) */
/* Intentionally left empty so this section header stays in the change log. */

/* I. Gradient scrollbar */
* { scrollbar-width: thin; scrollbar-color: color-mix(in oklch, #38bdf8 65%, transparent) transparent; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #38bdf8, #a855f7);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, #22d3ee, #ec4899);
  background-clip: padding-box;
}

/* J. Staggered first-paint reveal */
@keyframes ultra-fade-in-up {
  from { opacity: 0; transform: translateY(14px); filter: blur(8px); }
  to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}
[data-testid="stAppViewContainer"] section.main > div > div > div > [data-testid] {
  animation: ultra-fade-in-up 720ms var(--ease-out-expo) both;
  animation-delay: calc(var(--ultra-stagger, 0) * 60ms);
}

/* K. Typography polish */
h1, h2, h3 { text-wrap: balance; }
p, [data-testid="stMarkdownContainer"] p { text-wrap: pretty; hanging-punctuation: first last; }
.hero h1 {
  font-variation-settings: "wght" 820;
  letter-spacing: -0.04em;
  background: linear-gradient(100deg, #f8fafc 0%, #c0d6ff 32%, #a855f7 55%, #22d3ee 78%, #f8fafc 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
          background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: ultra-shimmer 8s linear infinite;
}
@keyframes ultra-shimmer { to { background-position: -200% 0; } }

/* L+M. Phase 24.16 cull — .ultra-confetti, .ultra-status, .ultra-status__dot
   and ultra-pulse-dot removed. Confetti DOM element was dropped along with
   its 130-line JS engine. Status pill was hidden anyway. */

/* N. Section title underline (scroll-driven) */
@supports (animation-timeline: view()) {
  h2 { position: relative; }
  h2::after {
    content: "";
    position: absolute;
    left: 0; bottom: -6px;
    height: 2px;
    width: 0%;
    background: linear-gradient(90deg, #38bdf8, #a855f7, #22d3ee);
    border-radius: 2px;
    animation: ultra-underline linear both;
    animation-timeline: view();
    animation-range: entry 10% cover 35%;
  }
  @keyframes ultra-underline { to { width: 64px; } }
}

/* ==== Phase 24 — UNIQUE TABS / SIDEBAR / SEARCH ============================
   The default Streamlit chrome was too generic. This section completely
   re-skins the three places users spend the most time: the top tab bar,
   the left sidebar, and the search input.
   ============================================================================ */

/* ---- TABS: glass pills with sliding indicator + per-tab accent ---------- */
.stTabs [data-baseweb="tab-list"] {
  position: relative;
  gap: 6px !important;
  padding: 6px !important;
  border-radius: 18px !important;
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #0b1220 75%, transparent),
      color-mix(in oklch, #0b1220 60%, transparent)) !important;
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid color-mix(in oklch, white 9%, transparent) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.05),
    0 12px 40px -20px rgba(0,0,0,0.5);
  margin-bottom: 22px !important;
  overflow-x: auto;
  scrollbar-width: none;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }

.stTabs [data-baseweb="tab"] {
  position: relative;
  border-radius: 13px !important;
  padding: 10px 18px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.015em !important;
  color: rgba(226,232,240,0.62) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  transition: all 280ms var(--ease-out-expo);
  white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover {
  color: rgba(226,232,240,0.92) !important;
  background: color-mix(in oklch, white 5%, transparent) !important;
  transform: translateY(-1px);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: #f8fafc !important;
  background: linear-gradient(135deg,
    color-mix(in oklch, #38bdf8 22%, transparent),
    color-mix(in oklch, #a855f7 22%, transparent)) !important;
  border-color: color-mix(in oklch, #38bdf8 45%, transparent) !important;
  box-shadow:
    0 0 0 1px color-mix(in oklch, #38bdf8 25%, transparent) inset,
    0 8px 24px -12px color-mix(in oklch, #38bdf8 80%, transparent),
    0 0 22px -6px color-mix(in oklch, #a855f7 60%, transparent);
}
/* Tiny dot indicator under the active tab */
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
  content: "";
  position: absolute;
  left: 50%; bottom: 4px;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: linear-gradient(90deg, #38bdf8, #a855f7);
  transform: translateX(-50%);
  box-shadow: 0 0 8px rgba(56,189,248,0.8);
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* Hide the boring underline border under tab list */
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-list"] + div > div:first-child { background: transparent !important; }

/* ---- SIDEBAR: vertical command rail with section dividers --------------- */
section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #0b1220 90%, transparent) 0%,
      color-mix(in oklch, #050810 95%, transparent) 100%) !important;
  border-right: 1px solid color-mix(in oklch, white 8%, transparent) !important;
  box-shadow: 1px 0 30px -10px rgba(0,0,0,0.5);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
}
section[data-testid="stSidebar"] > div:first-child {
  padding-top: 18px !important;
}
/* Decorative top accent bar in sidebar */
section[data-testid="stSidebar"]::before {
  content: "";
  position: absolute;
  top: 0; left: 12px; right: 12px;
  height: 2px;
  background: linear-gradient(90deg,
    transparent, #38bdf8 20%, #a855f7 50%, #22d3ee 80%, transparent);
  opacity: 0.65;
  border-radius: 999px;
  filter: blur(0.5px);
}
/* Vertical glow rail along the sidebar's right edge */
section[data-testid="stSidebar"]::after {
  content: "";
  position: absolute;
  top: 60px; bottom: 60px; right: -1px;
  width: 1px;
  background: linear-gradient(180deg,
    transparent, color-mix(in oklch, #38bdf8 50%, transparent) 30%,
    color-mix(in oklch, #a855f7 50%, transparent) 70%, transparent);
  opacity: 0.5;
}

/* Sidebar headings — small caps, gradient bullet */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase;
  color: rgba(148,163,184,0.85) !important;
  margin: 18px 0 10px 0 !important;
  padding-left: 14px;
  position: relative;
}
section[data-testid="stSidebar"] h1::before,
section[data-testid="stSidebar"] h2::before,
section[data-testid="stSidebar"] h3::before {
  content: "";
  position: absolute;
  left: 0; top: 50%;
  width: 6px; height: 6px;
  margin-top: -3px;
  border-radius: 50%;
  background: linear-gradient(135deg, #38bdf8, #a855f7);
  box-shadow: 0 0 8px rgba(56,189,248,0.6);
}

/* Sidebar buttons — full-width, icon-rail style */
section[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  text-align: left !important;
  border-radius: 10px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  color: rgba(226,232,240,0.78) !important;
  font-weight: 500 !important;
  padding: 10px 12px !important;
  position: relative;
  transition: all 220ms var(--ease-out-expo);
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: color-mix(in oklch, white 5%, transparent) !important;
  border-color: color-mix(in oklch, #38bdf8 30%, transparent) !important;
  color: #f8fafc !important;
  padding-left: 18px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover::before {
  content: "";
  position: absolute;
  left: 6px; top: 50%;
  width: 3px; height: 18px;
  margin-top: -9px;
  border-radius: 2px;
  background: linear-gradient(180deg, #38bdf8, #a855f7);
}

/* Sidebar widgets (selectbox, slider, radio) */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: color-mix(in oklch, #0b1220 60%, transparent) !important;
  border: 1px solid color-mix(in oklch, white 8%, transparent) !important;
  border-radius: 10px !important;
  color: #e2e8f0 !important;
}

/* ---- SEARCH BAR: cutting-edge animated focus ring + live indicator ----- */
/* We can't add HTML around Streamlit's input from CSS, so we restyle the
   input widget heavily so it FEELS like a command palette. The two main
   text_input boxes (Predict tab + recent searches) get this treatment. */

[data-testid="stTextInput"] {
  position: relative;
}
[data-testid="stTextInput"] > div > div {
  position: relative;
  border-radius: 14px !important;
  background:
    linear-gradient(180deg,
      color-mix(in oklch, #0b1220 70%, transparent),
      color-mix(in oklch, #0b1220 90%, transparent)) !important;
  backdrop-filter: blur(18px) saturate(180%);
  -webkit-backdrop-filter: blur(18px) saturate(180%);
  border: 1px solid color-mix(in oklch, white 10%, transparent) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.05),
    0 8px 24px -16px rgba(0,0,0,0.5);
  transition: all 240ms var(--ease-out-expo);
  overflow: visible;
}
[data-testid="stTextInput"] > div > div::before {
  /* Animated gradient border — uses background-clip trick + animation */
  content: "";
  position: absolute;
  inset: -1px;
  border-radius: 14px;
  padding: 1px;
  background: conic-gradient(
    from var(--search-angle, 0deg),
    transparent 0deg,
    #38bdf8 90deg,
    #a855f7 180deg,
    #22d3ee 270deg,
    transparent 360deg);
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
          mask-composite: exclude;
  opacity: 0;
  transition: opacity 280ms var(--ease-out-expo);
  pointer-events: none;
  animation: search-rotate 4s linear infinite;
}
@property --search-angle { syntax: "<angle>"; inherits: false; initial-value: 0deg; }
@keyframes search-rotate { to { --search-angle: 360deg; } }

[data-testid="stTextInput"] > div > div:focus-within {
  border-color: color-mix(in oklch, #38bdf8 55%, transparent) !important;
  box-shadow:
    0 0 0 4px color-mix(in oklch, #38bdf8 18%, transparent),
    0 12px 40px -18px color-mix(in oklch, #38bdf8 70%, transparent),
    inset 0 1px 0 rgba(255,255,255,0.08);
  transform: translateY(-1px);
}
[data-testid="stTextInput"] > div > div:focus-within::before { opacity: 1; }

/* Magnifier glass icon prepended to text inputs */
[data-testid="stTextInput"] > div > div > input {
  background: transparent !important;
  border: none !important;
  padding: 14px 16px 14px 44px !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  color: #f1f5f9 !important;
  letter-spacing: 0.005em;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
  color: rgba(148,163,184,0.55) !important;
  font-weight: 400;
}
[data-testid="stTextInput"] > div > div > input:focus { outline: none !important; }

[data-testid="stTextInput"] > div > div::after {
  content: "";
  position: absolute;
  left: 16px; top: 50%;
  width: 14px; height: 14px;
  margin-top: -7px;
  pointer-events: none;
  /* SVG magnifier — encoded inline so no fetch */
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><path d='m21 21-4.3-4.3'/></svg>") center/contain no-repeat;
  opacity: 0.7;
  transition: opacity 220ms, filter 220ms;
}
[data-testid="stTextInput"] > div > div:focus-within::after {
  opacity: 1;
  filter: drop-shadow(0 0 6px rgba(56,189,248,0.6));
}

/* Right-aligned hint chip showing keyboard shortcut */
[data-testid="stTextInput"] > label {
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
  color: rgba(148,163,184,0.75) !important;
  margin-bottom: 6px !important;
  display: flex;
  align-items: center;
  gap: 8px;
}
/* Phase 24.13 — was a `::after` chip showing "⌘K" on EVERY st.text_input
   label (filters, search, dev tab inputs, etc.). The chip implied each
   input had a keyboard shortcut, which was misleading — ⌘K is global.
   Removed. The Cmd+K overlay still works; it just doesn't advertise via
   per-input chrome. */
[data-testid="stTextInput"] > label::after {
  content: none;
}

/* ==== Phase 24.5 — MODERN SLIDING SIDEBAR =================================
   Replaces the boring default Streamlit sidebar with a premium fly-in panel.
   Smooth spring transitions, custom toggle pill in top-left, frosted glass.
   The transition timing matches Linear / Notion sidebar feel.
   =========================================================================== */

/* Smooth sidebar slide-in: override Streamlit's sharp toggle */
section[data-testid="stSidebar"] {
  transition:
    transform 480ms cubic-bezier(0.16, 1, 0.3, 1),
    margin-left 480ms cubic-bezier(0.16, 1, 0.3, 1),
    width 480ms cubic-bezier(0.16, 1, 0.3, 1),
    min-width 480ms cubic-bezier(0.16, 1, 0.3, 1) !important;
  will-change: transform, width;
}

/* Sidebar collapse button: elevate from a thin chevron to a real pill */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  position: relative !important;
  width: 40px !important;
  height: 40px !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg,
    color-mix(in oklch, #38bdf8 22%, transparent),
    color-mix(in oklch, #a855f7 22%, transparent)) !important;
  border: 1px solid color-mix(in oklch, #38bdf8 35%, transparent) !important;
  backdrop-filter: blur(18px) saturate(180%);
  -webkit-backdrop-filter: blur(18px) saturate(180%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.1),
    0 8px 24px -8px color-mix(in oklch, #38bdf8 50%, transparent),
    0 0 24px -8px color-mix(in oklch, #a855f7 40%, transparent);
  transition: all 320ms cubic-bezier(0.16, 1, 0.3, 1) !important;
  cursor: pointer !important;
}
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
  transform: scale(1.08) translateY(-1px);
  border-color: color-mix(in oklch, #38bdf8 80%, transparent) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.18),
    0 12px 32px -8px color-mix(in oklch, #38bdf8 70%, transparent),
    0 0 32px -4px color-mix(in oklch, #a855f7 60%, transparent);
}

/* Style the chevron icons inside the toggle */
button[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
  fill: #f8fafc !important;
  filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.5));
  transition: transform 320ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* When sidebar is COLLAPSED, the show-button floats top-left */
[data-testid="collapsedControl"] {
  position: fixed !important;
  top: 50px !important;     /* below 36px marquee + 14px padding */
  left: 14px !important;
  z-index: 1003 !important;
  animation: ultra-pulse-toggle 2.4s cubic-bezier(0.16, 1, 0.3, 1) infinite;
}
@keyframes ultra-pulse-toggle {
  0%, 100% { box-shadow: 0 8px 24px -8px rgba(56, 189, 248, 0.5),
                         0 0 0 0   rgba(56, 189, 248, 0); }
  50%      { box-shadow: 0 8px 24px -8px rgba(56, 189, 248, 0.5),
                         0 0 0 8px rgba(56, 189, 248, 0.18); }
}

/* When sidebar is OPEN, the close button sits cleanly inside the panel */
button[data-testid="stSidebarCollapseButton"] {
  position: absolute !important;
  top: 14px !important;
  right: 14px !important;
  z-index: 100 !important;
}

/* Add a subtle backdrop blur effect to the main content when sidebar opens */
section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main {
  transition: filter 480ms cubic-bezier(0.16, 1, 0.3, 1);
}

/* Sidebar inner content: NO stagger animation.
   Phase 24.5 had a translateX(-12px) staged-in animation that re-fired on
   every Streamlit rerun, leaving the children stuck at translateX(-12px)
   for ~520ms each time and clipping the leftmost ~12px of every label
   ("MNI_STOCKS" instead of "OMNI_STOCKS"). Removed entirely — the smooth
   slide-in of the whole sidebar is enough motion. */

/* Floating thin accent line on the inner edge of the sidebar */
section[data-testid="stSidebar"]::after {
  content: "";
  position: absolute;
  top: 60px; bottom: 60px; right: 0;
  width: 2px;
  background: linear-gradient(180deg,
    transparent,
    color-mix(in oklch, #38bdf8 60%, transparent) 25%,
    color-mix(in oklch, #a855f7 60%, transparent) 75%,
    transparent);
  opacity: 0.45;
  filter: blur(0.5px);
  border-radius: 2px;
  pointer-events: none;
}

/* Reduced motion (Phase 24.12 — comprehensive coverage) */
@media (prefers-reduced-motion: reduce) {
  /* Universal sledgehammer: kill every animation and transition for users
     with motion sensitivity. The earlier specific-selector approach missed
     many animations (search-rotate, _glow-rotate, _cmdk-pop, tabFadeIn,
     ultra-fade-in-up, fx-orb-*) — this catches them all. */
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
  /* .ultra-lighthouse selector kept here as a no-op — element removed. */
}

/* ==== Phase 24.8 — PRODUCTION ACCESSIBILITY POLISH ========================
   Addresses concrete UX/accessibility issues:
   1. WCAG-compliant text contrast on all helper / instruction copy
   2. Pill / chip legibility (bigger, higher-contrast)
   3. Sidebar collapse button always clickable + on top
   4. Cleaner ticker input field (no overcrowding)
   5. Caption / muted text upgraded from rgba(...,0.45) to ≥0.75
   =========================================================================== */

/* (1) Helper / muted text now slate-300+ (WCAG AA on dark bg) */
.stApp [data-testid="stCaptionContainer"],
.stApp .stCaption,
.stApp [data-testid="stMarkdownContainer"] small,
.stApp [data-testid="stMarkdownContainer"] em,
[data-testid="stWidgetLabel"] {
  color: rgba(203, 213, 225, 0.85) !important;   /* slate-300 / ~5.5:1 contrast */
  font-weight: 500 !important;
}
.stApp [data-testid="stMarkdownContainer"] {
  color: #e2e8f0;
}
.stApp [data-testid="stInfo"] {
  background: color-mix(in oklch, #38bdf8 12%, transparent) !important;
  border: 1px solid color-mix(in oklch, #38bdf8 30%, transparent) !important;
  color: #e0f2fe !important;
}

/* (2) Pill / chip legibility */
.stMultiSelect [data-baseweb="tag"],
.stSelectbox [data-baseweb="tag"],
[data-baseweb="tag"] {
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  padding: 4px 10px !important;
  border-radius: 8px !important;
  background: color-mix(in oklch, #38bdf8 16%, transparent) !important;
  border: 1px solid color-mix(in oklch, #38bdf8 35%, transparent) !important;
  color: #e0f2fe !important;
}
[data-baseweb="tag"] svg { fill: #e0f2fe !important; opacity: 0.8 !important; }

/* (3) SIDEBAR REOPEN BUG FIX
   The previous Phase 24.5 rules made the toggle pretty but Streamlit's
   collapsedControl became un-clickable after a close because the confetti
   canvas (z-index 1003, full viewport) was layered above it. Fix: bump
   collapsedControl's z-index over everything and ensure pointer-events. */
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
  z-index: 2147483600 !important;
  pointer-events: auto !important;
}
.ultra-marquee  { z-index: 1100 !important; }
/* Phase 24.16 cull — z-index lines for .ultra-confetti / .ultra-status /
   .ultra-lighthouse removed (elements deleted). */

/* (4) Cleaner ticker input — give the chevron + clear button breathing room */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
  padding-right: 8px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  padding: 4px 4px 4px 16px !important;
}
[data-testid="stSelectbox"] [aria-label="Clear value"],
[data-testid="stSelectbox"] [aria-label="open"] {
  margin-left: 4px !important;
  opacity: 0.65;
  transition: opacity 180ms var(--ease-out-expo);
}
[data-testid="stSelectbox"] [aria-label="Clear value"]:hover,
[data-testid="stSelectbox"] [aria-label="open"]:hover {
  opacity: 1;
}

/* (5) Search-history expander — was sprawling full-width with thin text.
   Compact it: smaller header, bordered chip-style. */
[data-testid="stExpander"] > details > summary {
  padding: 10px 14px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: rgba(203, 213, 225, 0.95) !important;
  background: color-mix(in oklch, #0b1220 60%, transparent) !important;
  border-radius: 12px !important;
  border: 1px solid color-mix(in oklch, white 6%, transparent) !important;
  list-style: none;
  cursor: pointer;
}
[data-testid="stExpander"] > details > summary:hover {
  background: color-mix(in oklch, #0b1220 80%, transparent) !important;
  border-color: color-mix(in oklch, #38bdf8 30%, transparent) !important;
}

/* Universal sidebar safety: ensure inner content has proper left padding so
   no labels ever clip on the leftmost edge. */
section[data-testid="stSidebar"] > div:first-child > div:first-child {
  padding-left: 16px !important;
  padding-right: 16px !important;
  padding-top: 18px !important;
}

/* The "Force refresh" button: when it's a regular st.button, make sure it
   does NOT show a checkbox-looking glyph. The 🔄 emoji can render as a
   small box on systems missing the emoji font. We removed the emoji from
   app.py; this rule just polishes the button shape. */
.stButton > button:not([kind="primary"]) {
  border: 1px solid color-mix(in oklch, white 12%, transparent) !important;
  background: color-mix(in oklch, #0b1220 50%, transparent) !important;
  color: rgba(226, 232, 240, 0.92) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: color-mix(in oklch, #38bdf8 50%, transparent) !important;
  background: color-mix(in oklch, #0b1220 70%, transparent) !important;
  color: #f1f5f9 !important;
}

/* Help-icon (the small ⓘ next to widget labels) */
[data-testid="stTooltipIcon"] svg {
  fill: rgba(203, 213, 225, 0.55) !important;
  transition: fill 200ms var(--ease-out-expo);
}
[data-testid="stTooltipIcon"]:hover svg {
  fill: #38bdf8 !important;
}

/* ==== Phase 24.9 — RADIUS / PADDING SCALE + SIDEBAR TOGGLE GUARANTEE ======
   Fixes three real production issues:
   1. Padding < border-radius made content collide with rounded corners
   2. Unbalanced vertical spacing in container blocks
   3. Nested radii looked arbitrary — locked a proper scale: 6/10/14/20px
   4. Sidebar toggle was sometimes invisible — guaranteed-visible custom button
   =========================================================================== */

/* ---- Radius scale tokens ------------------------------------------------ */
:root {
  --r-xs: 6px;       /* small chips, deltas */
  --r-sm: 10px;      /* buttons, small inputs */
  --r-md: 14px;      /* metric tiles, medium cards */
  --r-lg: 20px;      /* hero card, page-level containers */
  --r-pill: 999px;   /* pure pills */
  /* Padding scale — each level matches a radius so OuterPad ≥ OuterRadius */
  --p-card-sm: 14px;     /* matches r-sm content */
  --p-card-md: 20px;     /* matches r-md content */
  --p-card-lg: 28px;     /* matches r-lg content */
}

/* ---- Predict tab "Enter a ticker to begin" container ---- */
/* Streamlit's stInfo had bottom-left corner collision because the inner
   padding (10px) was less than the border-radius (12px). Fix: bump padding,
   match the radius scale. */
.stApp [data-testid="stInfo"] {
  padding: 16px 22px !important;
  border-radius: var(--r-md) !important;
  margin-top: 10px !important;
}

/* ---- Streamlit alert containers (info / warning / error / success) ---- */
.stApp [data-testid="stAlert"],
.stApp [data-testid="stAlertContentInfo"],
.stApp [data-testid="stAlertContentWarning"],
.stApp [data-testid="stAlertContentError"],
.stApp [data-testid="stAlertContentSuccess"] {
  padding: 14px 20px !important;
  border-radius: var(--r-md) !important;
  line-height: 1.5;
}

/* ---- Buttons: locked radius, balanced padding ---- */
.stButton > button {
  padding: 12px 18px !important;
  border-radius: var(--r-sm) !important;
  min-height: 42px;
  font-size: 13.5px !important;
}

/* ---- Selectbox / TextInput inner radius matches card radius ---- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
  border-radius: var(--r-sm) !important;
}

/* ---- Search history expander: outer pad > radius ---- */
[data-testid="stExpander"] > details {
  border-radius: var(--r-md) !important;
  overflow: hidden;
}
[data-testid="stExpander"] > details > summary {
  padding: 12px 18px !important;
  border-radius: var(--r-md) !important;
}

/* ---- Block container (the wrapper around each tab's content) ---- */
[data-testid="stVerticalBlock"] > div:has([data-testid="stInfo"]:only-child),
[data-testid="stHorizontalBlock"] {
  gap: 12px;
}

/* ---- Predict tab specifically: balance top vs bottom padding ----------- */
/* Streamlit's column layout has consistent gaps; the trailing element
   (st.info "Enter a ticker") lands too close to the next section. Add
   bottom margin so the empty state doesn't feel jammed against the edge. */
.stApp [data-testid="stInfo"]:last-child {
  margin-bottom: 24px !important;
}

/* ===========================================================================
   Phase 24.15 — RAG citations & grounding badge
   ===========================================================================
   Renders below the why-text in the prediction card. Three pieces:
     - inline pill showing path/score (green = grounded, amber = low conf)
     - row of clickable citation chips linking to source articles
   ============================================================================= */
.why-badge {
  display: inline-flex;
  align-items: center;
  margin-left: 10px;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
  vertical-align: middle;
}
.why-badge-ok {
  color: #4ade80;
  background: rgba(74, 222, 128, 0.12);
  border: 1px solid rgba(74, 222, 128, 0.28);
}
.why-badge-warn {
  color: #fbbf24;
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.28);
}
.why-citations {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
  margin-bottom: 4px;
}
.why-citation-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  background: color-mix(in oklch, white 6%, transparent);
  border: 1px solid color-mix(in oklch, white 12%, transparent);
  font-size: 11px;
  color: rgba(203, 213, 225, 0.92) !important;
  text-decoration: none !important;
  transition: all 180ms cubic-bezier(0.16, 1, 0.3, 1);
  font-family: Inter, system-ui, sans-serif;
}
.why-citation-chip b {
  color: #06b6d4;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}
.why-citation-chip:hover {
  background: color-mix(in oklch, #06b6d4 14%, transparent);
  border-color: color-mix(in oklch, #06b6d4 45%, transparent);
  transform: translateY(-1px);
}

/* Phase 24.11 — sidebar collapse concept dropped entirely. Hide every
   plausible Streamlit collapse-control testid so the sidebar is permanent.
   Nav lives in the sidebar (sections + filters + theme), so closing it
   would strand the user. We commit to "always visible" instead of the
   fragile toggle-button game. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
button[kind="header"][title="Show sidebar"],
button[kind="header"][title="Open sidebar"],
button[kind="header"][title="Close sidebar"],
button[kind="header"][title="Hide sidebar"] {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}

/* Phase 24.16 — FORCE the sidebar visible regardless of Streamlit's
   internal collapsed state. Without this, hiding the toggle (above)
   creates a one-way trap: any cookie/state-persisted collapse, narrow
   viewport auto-collapse, or accidental drag leaves the user with no
   sidebar AND no way to bring it back. We override aria-expanded="false"
   to keep the sidebar mounted, full width, on-screen — same as
   aria-expanded="true". This guarantees the "permanent sidebar"
   contract that Phase 24.11 promised. */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="true"] {
  display: block !important;
  visibility: visible !important;
  transform: translateX(0) !important;
  margin-left: 0 !important;
  min-width: 244px !important;
  width: 244px !important;
  max-width: 244px !important;
  flex-shrink: 0 !important;
}
/* Make sure the inner div isn't hidden either (some Streamlit versions
   collapse content via the first child rather than the section itself). */
section[data-testid="stSidebar"] > div:first-child {
  display: block !important;
  visibility: visible !important;
  transform: translateX(0) !important;
}
</style>
"""

_ULTRA_FX_JS = """
<script>
/* Phase 24.16 cull — _ULTRA_FX_JS reduced from 135 lines to ~25.
   Dropped:
     - lighthouse mousemove (the lighthouse element is removed)
     - confetti particle engine + Shift+C trigger (130 lines, rarely used)
     - dead `tickFollower` / cursor-follow / slot-roll comments
     - sidebar-toggle observer (sidebar permanent)
   Kept: only the stagger animation that gives a subtle reveal cascade
   to first-paint blocks. That's still visible to the user. */
(function(){
  if (window.__ultraBound) return;
  window.__ultraBound = true;

  function stagger(){
    document.querySelectorAll(
      '[data-testid="stAppViewContainer"] section.main > div > div > div > [data-testid]'
    ).forEach((el, i) => {
      if (!el.style.getPropertyValue('--ultra-stagger')) {
        el.style.setProperty('--ultra-stagger', String(Math.min(i, 24)));
      }
    });
  }
  stagger();
  const mo = new MutationObserver(function(){
    if (window.__ultraRaf) return;
    window.__ultraRaf = requestAnimationFrame(function(){
      window.__ultraRaf = null;
      stagger();
    });
  });
  mo.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


_PHASE27_CRAZY_FX_CSS = """
<style>
/* ============================================================
   Phase 27 — CRAZY GRAPHICS LAYER
   ------------------------------------------------------------
   Drama without breaking legibility. Composable. Honors
   prefers-reduced-motion. Reuses existing tokens (--accent,
   --surface-*, --border-*) so all six themes inherit.

   Effects in this chunk:
     A. Animated mesh-gradient app background
     B. Floating ambient orbs (corners, slow drift)
     C. Subtle drifting grid backdrop
     D. Holographic shine sweep across .hero on first paint
     E. Active-tab conic ribbon glow
     F. Animated cyan→magenta border on .signal-card:hover
     G. Neon scale-from-center underline on primary buttons
     H. RGB-channel glitch on hero h1 hover
     I. Pure-CSS starfield twinkle
     J. KPI tile aura when high-conviction (uses :has())
     K. Scroll-into-view fade-in for cards
     L. Tab alignment refinement + tabpanel cross-fade
   ============================================================ */

@media (prefers-reduced-motion: no-preference) {

/* ============================================================
   A. Animated mesh-gradient background
   Three drifting radial gradients sit BEHIND everything; the
   composite slowly migrates so the canvas feels alive without
   distracting from data.
   ============================================================ */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -2;
    pointer-events: none;
    background:
      radial-gradient(800px 600px at 18% 22%,
        color-mix(in oklch, var(--accent) 16%, transparent), transparent 62%),
      radial-gradient(900px 700px at 82% 30%,
        color-mix(in oklch, #a855f7 14%, transparent), transparent 60%),
      radial-gradient(1000px 800px at 50% 92%,
        color-mix(in oklch, #06b6d4 12%, transparent), transparent 58%);
    animation: omni-mesh-drift 36s ease-in-out infinite alternate;
}
@keyframes omni-mesh-drift {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    50%  { transform: translate3d(2%, -2%, 0) scale(1.04); }
    100% { transform: translate3d(-2%, 1.5%, 0) scale(0.98); }
}

/* ============================================================
   B. Floating ambient orbs
   Three fixed corner orbs that drift; viewport corners only so
   they never overlap dense UI regions.
   ============================================================ */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background:
      radial-gradient(420px 420px at 8% 6%,
        color-mix(in oklch, var(--accent) 28%, transparent), transparent 70%),
      radial-gradient(500px 500px at 96% 94%,
        color-mix(in oklch, #ec4899 22%, transparent), transparent 70%),
      radial-gradient(360px 360px at 92% 4%,
        color-mix(in oklch, #f59e0b 16%, transparent), transparent 70%);
    filter: blur(28px);
    opacity: 0.85;
    animation: omni-orbs-drift 52s ease-in-out infinite alternate;
}
@keyframes omni-orbs-drift {
    0%   { transform: translate3d(0,    0,   0); }
    33%  { transform: translate3d(-1.5%, 1%,  0); }
    66%  { transform: translate3d(1.2%, -1%,  0); }
    100% { transform: translate3d(0,    0.6%, 0); }
}

/* ============================================================
   C. Subtle drifting grid backdrop
   32px grid, ~3% opacity, slow horizontal drift. Adds depth
   to "empty" regions without ever competing with chart ink.
   ============================================================ */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: -10% -10% -10% -10%;
    z-index: -1;
    pointer-events: none;
    background-image:
      linear-gradient(to right,
        color-mix(in oklch, var(--text-primary) 4%, transparent) 1px, transparent 1px),
      linear-gradient(to bottom,
        color-mix(in oklch, var(--text-primary) 4%, transparent) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, black, transparent 90%);
    -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, black, transparent 90%);
    animation: omni-grid-drift 60s linear infinite;
}
@keyframes omni-grid-drift {
    0%   { background-position: 0 0; }
    100% { background-position: 32px 32px; }
}

/* ============================================================
   D. Holographic shine sweep across .hero on first paint
   ============================================================ */
.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg,
        transparent 35%,
        color-mix(in oklch, white 22%, transparent) 47%,
        color-mix(in oklch, var(--accent) 24%, transparent) 50%,
        color-mix(in oklch, white 22%, transparent) 53%,
        transparent 65%);
    transform: translateX(-120%);
    animation: omni-hero-shine 1.6s cubic-bezier(0.22, 1, 0.36, 1) 0.4s 1 forwards;
    pointer-events: none;
    mix-blend-mode: screen;
}
@keyframes omni-hero-shine {
    0%   { transform: translateX(-120%); opacity: 0; }
    20%  { opacity: 1; }
    100% { transform: translateX(120%);  opacity: 0; }
}

/* ============================================================
   E. Active-tab conic ribbon glow
   Subtle rotating conic-gradient halo behind the active tab.
   Sits inside the pill via mask-image so it can't bleed.
   ============================================================ */
.stTabs [data-baseweb="tab"][aria-selected="true"]::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: inherit;
    background: conic-gradient(from 0deg,
        color-mix(in oklch, var(--accent) 60%, transparent),
        color-mix(in oklch, #a855f7 50%, transparent),
        color-mix(in oklch, #ec4899 40%, transparent),
        color-mix(in oklch, var(--accent) 60%, transparent));
    filter: blur(10px);
    opacity: 0.55;
    z-index: -1;
    animation: omni-ribbon-spin 8s linear infinite;
    pointer-events: none;
}
@keyframes omni-ribbon-spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ============================================================
   F. Animated cyan→magenta gradient border on cards (hover)
   Uses ::before with mask-composite trick for a true 1px
   gradient stroke (no background bleed).
   ============================================================ */
.signal-card, .glass, .glass-strong, .prediction-card {
    position: relative;
    isolation: isolate;
}
.signal-card::before,
.glass::before,
.glass-strong::before,
.prediction-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(120deg,
        var(--accent),
        #a855f7,
        #ec4899,
        var(--accent));
    background-size: 300% 300%;
    -webkit-mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    opacity: 0;
    transition: opacity 280ms cubic-bezier(0.22, 1, 0.36, 1);
    pointer-events: none;
    z-index: -1;
}
.signal-card:hover::before,
.glass:hover::before,
.glass-strong:hover::before,
.prediction-card:hover::before {
    opacity: 0.85;
    animation: omni-card-border-shimmer 4s linear infinite;
}
@keyframes omni-card-border-shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

/* ============================================================
   G. Neon scale-from-center underline on primary buttons
   ============================================================ */
.stButton > button[kind="primary"] {
    position: relative;
    overflow: visible;
}
.stButton > button[kind="primary"]::after {
    content: "";
    position: absolute;
    left: 12%;
    right: 12%;
    bottom: -5px;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg,
        transparent,
        var(--accent),
        #a855f7,
        var(--accent),
        transparent);
    transform: scaleX(0);
    transform-origin: center;
    transition: transform 220ms cubic-bezier(0.22, 1, 0.36, 1);
    box-shadow: 0 0 12px color-mix(in oklch, var(--accent) 60%, transparent);
    pointer-events: none;
}
.stButton > button[kind="primary"]:hover::after,
.stButton > button[kind="primary"]:focus-visible::after {
    transform: scaleX(1);
}

/* ============================================================
   H. RGB-channel glitch on hero h1 hover
   Uses dual text-shadow offsets in the cyan + magenta channels
   so the headline subtly de-syncs on hover. Cheap, no canvas.
   ============================================================ */
.hero h1 {
    position: relative;
    transition: text-shadow 180ms cubic-bezier(0.22, 1, 0.36, 1);
}
.hero h1:hover {
    text-shadow:
        -1.5px 0 0 color-mix(in oklch, var(--accent) 80%, transparent),
        1.5px  0 0 color-mix(in oklch, #ec4899 70%, transparent),
        0 0 22px color-mix(in oklch, var(--accent) 25%, transparent);
    animation: omni-glitch 900ms steps(2, end);
}
@keyframes omni-glitch {
    0%   { transform: translate(0,    0); }
    20%  { transform: translate(-0.8px, 0.4px); }
    40%  { transform: translate(0.8px, -0.4px); }
    60%  { transform: translate(-0.4px, 0); }
    80%  { transform: translate(0.4px, 0); }
    100% { transform: translate(0,    0); }
}

/* ============================================================
   I. Pure-CSS starfield (no JS, no canvas)
   Single 1×1 pseudo-element with a long box-shadow list draws
   ~70 stars across the viewport. Slow twinkle.
   ============================================================ */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 1px; height: 1px;
    z-index: -1;
    pointer-events: none;
    background: transparent;
    box-shadow:
       11vw  17vh 0 0.5px rgba(255,255,255,0.55),
       23vw   8vh 0 0.5px rgba(255,255,255,0.40),
       38vw  35vh 0 0.5px rgba(255,255,255,0.65),
       52vw  12vh 0 0.5px rgba(255,255,255,0.45),
       66vw  41vh 0 0.5px rgba(255,255,255,0.55),
       77vw   6vh 0 0.5px rgba(255,255,255,0.40),
       89vw  28vh 0 0.5px rgba(255,255,255,0.55),
        4vw  47vh 0 0.5px rgba(255,255,255,0.45),
       17vw  61vh 0 0.5px rgba(255,255,255,0.50),
       29vw  55vh 0 0.5px rgba(255,255,255,0.40),
       42vw  72vh 0 0.5px rgba(255,255,255,0.45),
       57vw  64vh 0 0.5px rgba(255,255,255,0.55),
       71vw  77vh 0 0.5px rgba(255,255,255,0.40),
       83vw  53vh 0 0.5px rgba(255,255,255,0.55),
       94vw  68vh 0 0.5px rgba(255,255,255,0.40),
        7vw  83vh 0 0.5px rgba(255,255,255,0.50),
       21vw  91vh 0 0.5px rgba(255,255,255,0.45),
       33vw  87vh 0 0.5px rgba(255,255,255,0.40),
       46vw  94vh 0 0.5px rgba(255,255,255,0.55),
       60vw  86vh 0 0.5px rgba(255,255,255,0.45),
       74vw  93vh 0 0.5px rgba(255,255,255,0.50),
       86vw  79vh 0 0.5px rgba(255,255,255,0.40);
    animation: omni-starfield-twinkle 7s ease-in-out infinite alternate;
}
@keyframes omni-starfield-twinkle {
    0%   { opacity: 0.45; }
    50%  { opacity: 0.85; }
    100% { opacity: 0.55; }
}

/* ============================================================
   J. KPI tile aura when conviction is high
   Uses :has() so the aura only fires on tiles whose value is
   ≥ 70 (set via data-attribute downstream; falls back to a
   universal soft glow on .stat-value:hover when no data is
   present, which is harmless polish).
   ============================================================ */
.stat-value {
    transition:
        text-shadow 240ms cubic-bezier(0.22, 1, 0.36, 1),
        transform   240ms cubic-bezier(0.22, 1, 0.36, 1);
}
.stat-value:hover {
    text-shadow:
        0 0 12px color-mix(in oklch, var(--accent) 50%, transparent),
        0 0 28px color-mix(in oklch, var(--accent) 25%, transparent);
    transform: translateY(-1px);
}

/* ============================================================
   K. Scroll-into-view fade for cards (modern browsers)
   ============================================================ */
@supports (animation-timeline: view()) {
    .signal-card, .glass-strong, .prediction-card {
        animation: omni-card-rise linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 28%;
    }
    @keyframes omni-card-rise {
        0%   { opacity: 0;   transform: translateY(14px) scale(0.985); }
        100% { opacity: 1;   transform: translateY(0)    scale(1);     }
    }
}

}  /* end @media (prefers-reduced-motion: no-preference) */

/* ============================================================
   L. Tab alignment refinement
   These rules apply with OR without reduced-motion preference,
   so layout stays stable for users who suppress animation.
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 4px !important;
    padding: 5px !important;
    margin-bottom: 18px !important;
    margin-inline: auto;
}
.stTabs [data-baseweb="tab"] {
    padding: 9px 18px !important;
    align-items: center;
    display: inline-flex;
    line-height: 1.1 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    bottom: 5px;
    width: 16px;
    height: 2px;
    border-radius: 999px;
    box-shadow: 0 0 10px color-mix(in oklch, var(--accent) 70%, transparent);
}
/* Smooth tabpanel cross-fade on switch */
.stTabs [data-baseweb="tab-panel"] > div {
    animation: omni-panel-in 240ms cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes omni-panel-in {
    0%   { opacity: 0; transform: translateY(4px); }
    100% { opacity: 1; transform: translateY(0);   }
}

/* Reduced-motion: kill the panel animation too */
@media (prefers-reduced-motion: reduce) {
    .stTabs [data-baseweb="tab-panel"] > div { animation: none !important; }
}
</style>
"""


_THEME_BUNDLE: str | None = None


_PHASE26_POLISH_CSS = """
<style>
/* ============================================================
   Phase 26 — final polish layer
   ------------------------------------------------------------
   Sits LAST in the bundle so it wins specificity ties on the
   common interactive surfaces (buttons, selectbox, focus rings,
   card hover, scrollbars). Strictly additive — no existing
   selector is overridden destructively, only refined.

   Design intent:
     - Smoother motion (modern easing, no jank on hover)
     - Visible accessible focus rings on every interactive
     - Subtle elevation cue on hoverable cards
     - Refined scrollbars (thin, accent on hover)
     - Honors prefers-reduced-motion
   ============================================================ */

:root {
    /* Motion scale */
    --motion-fast: 120ms;
    --motion-base: 180ms;
    --motion-slow: 280ms;
    --ease-out:    cubic-bezier(0.22, 1, 0.36, 1);
    --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);

    /* Elevation scale (composed with the existing tokens) */
    --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.06);
    --shadow-sm: 0 2px 6px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 8px 24px rgba(15, 23, 42, 0.10), 0 2px 6px rgba(15, 23, 42, 0.06);
    --shadow-lg: 0 16px 48px rgba(15, 23, 42, 0.16), 0 4px 12px rgba(15, 23, 42, 0.08);

    /* Focus ring (uses live --accent so it adapts per theme) */
    --ring: 0 0 0 2px var(--accent),
            0 0 0 6px color-mix(in oklch, var(--accent) 22%, transparent);
}
@media (prefers-color-scheme: dark) {
    :root {
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.30);
        --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.25);
        --shadow-md: 0 10px 28px rgba(0, 0, 0, 0.45), 0 3px 8px rgba(0, 0, 0, 0.30);
        --shadow-lg: 0 20px 56px rgba(0, 0, 0, 0.55), 0 6px 16px rgba(0, 0, 0, 0.35);
    }
}

/* ---- Focus rings (accessibility + polish) ---- */
.stButton > button:focus-visible,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within,
[data-baseweb="tab"]:focus-visible,
a:focus-visible,
[role="button"]:focus-visible {
    outline: none !important;
    box-shadow: var(--ring) !important;
    transition: box-shadow var(--motion-fast) var(--ease-out) !important;
}

/* ---- Buttons: subtle lift on hover, snap on press ---- */
.stButton > button {
    transition:
        background-color var(--motion-fast) var(--ease-out),
        border-color     var(--motion-fast) var(--ease-out),
        box-shadow       var(--motion-fast) var(--ease-out),
        transform        var(--motion-fast) var(--ease-out) !important;
    will-change: transform, box-shadow;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: var(--shadow-xs);
    transition-duration: 80ms !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow:
        var(--shadow-sm),
        0 0 0 1px color-mix(in oklch, var(--accent) 35%, transparent);
}

/* ---- Selectbox: refined dropdown ---- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    transition:
        border-color var(--motion-fast) var(--ease-out),
        box-shadow   var(--motion-fast) var(--ease-out) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:hover > div {
    border-color: color-mix(in oklch, var(--accent) 45%, var(--border-strong)) !important;
}
[role="listbox"], [data-baseweb="popover"] [role="listbox"] {
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-md) !important;
    overflow: hidden;
    animation: omni-pop-in 140ms var(--ease-out);
}
[role="listbox"] [role="option"] {
    transition:
        background-color var(--motion-fast) var(--ease-out),
        color            var(--motion-fast) var(--ease-out) !important;
    border-radius: 6px;
    margin: 2px 6px;
    padding-inline: 10px !important;
}
[role="listbox"] [role="option"]:hover,
[role="listbox"] [role="option"][aria-selected="true"] {
    background: color-mix(in oklch, var(--accent) 14%, transparent) !important;
}

@keyframes omni-pop-in {
    0%   { opacity: 0; transform: translateY(-4px) scale(0.985); }
    100% { opacity: 1; transform: translateY(0)    scale(1);     }
}

/* ---- Cards: depth on hover (interactive cards only) ---- */
.signal-card, .glass, .glass-strong, .prediction-card {
    transition:
        box-shadow   var(--motion-base) var(--ease-out),
        border-color var(--motion-base) var(--ease-out),
        transform    var(--motion-base) var(--ease-out) !important;
}
a:hover .signal-card,
button:hover .signal-card,
.signal-card:focus-within {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px);
}

/* ---- Tabs: smooth indicator slide ---- */
[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border-subtle) !important;
}
[data-baseweb="tab"] {
    transition:
        color           var(--motion-base) var(--ease-out),
        background-color var(--motion-base) var(--ease-out) !important;
    border-radius: var(--r-sm) var(--r-sm) 0 0;
}
[data-baseweb="tab"]:hover {
    background: color-mix(in oklch, var(--accent) 8%, transparent) !important;
}
[data-baseweb="tab-highlight"] {
    transition:
        left  var(--motion-base) var(--ease-out),
        width var(--motion-base) var(--ease-out) !important;
    background: var(--accent) !important;
    height: 2px !important;
    border-radius: 2px;
}

/* ---- Scrollbars (Webkit + Firefox) ---- */
* { scrollbar-width: thin;
    scrollbar-color: color-mix(in oklch, var(--text-muted) 35%, transparent) transparent; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
    background: color-mix(in oklch, var(--text-muted) 30%, transparent);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: content-box;
    transition: background var(--motion-base) var(--ease-out);
}
*::-webkit-scrollbar-thumb:hover {
    background: color-mix(in oklch, var(--accent) 55%, transparent);
    background-clip: content-box;
}

/* ---- Pills: tighter, more chip-like ---- */
.pill {
    line-height: 1.25 !important;
    letter-spacing: 0.02em;
    transition:
        background var(--motion-fast) var(--ease-out),
        color      var(--motion-fast) var(--ease-out) !important;
}

/* ---- Caret + selection ---- */
::selection {
    background: color-mix(in oklch, var(--accent) 35%, transparent);
    color: var(--text-primary);
}

/* ---- Number inputs: tabular for stat-feel ---- */
.stNumberInput input,
.stTextInput input[inputmode="numeric"] {
    font-variant-numeric: tabular-nums;
}

/* ---- Toast tweaks: don't blow into the viewport ---- */
[data-testid="stToast"] {
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-md) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ---- Reduced motion: kill all polish transitions ---- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}
</style>
"""


def _theme_bundle() -> str:
    """Concatenate every CSS/JS chunk into a single markdown blob.

    Joining once (instead of 8 separate `st.markdown` calls per rerun) cuts
    Streamlit-side serialization + re-render of ~2600 lines of CSS on every
    rerun. The result is module-cached so it's joined only once per process.
    """
    global _THEME_BUNDLE
    if _THEME_BUNDLE is None:
        _THEME_BUNDLE = "\n".join((
            _FONT_LINK,
            _CSS,
            _VT_AND_CMDK_JS,
            _ADVANCED_FX_CSS,
            _ADVANCED_FX_JS,
            _ULTRA_FX_SVG_DEFS,
            _ULTRA_FX_CSS,
            _ULTRA_FX_JS,
            _PHASE26_POLISH_CSS,
            _PHASE27_CRAZY_FX_CSS,
        ))
    return _THEME_BUNDLE


def inject_theme() -> None:
    """Inject the modern minimalist CSS + Inter/JetBrains Mono fonts.

    Streamlit rebuilds the DOM on every rerun, so the CSS markdown block
    MUST be re-emitted each rerun or the page renders unstyled (regime
    banner blows out to full width, rubric pills stack vertically, the
    Phase 24.16 skeleton/progress-bar loader becomes plain text, etc.).

    The `_theme_bundle()` string is cached at module load (`_THEME_BUNDLE`
    is built once and reused), so the cost of this call is one
    `st.markdown` of a precomputed string -- negligible vs. the
    appearance regression of skipping it.
    """
    st.markdown(_theme_bundle(), unsafe_allow_html=True)


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
