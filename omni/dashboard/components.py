"""Reusable dashboard widgets — modern glassmorphism build.

Renders into the current Streamlit container using cyan/violet accents and
glass surfaces. All widgets degrade gracefully when data is missing.
"""
from __future__ import annotations

import html as _html  # Phase 24.16 — escape user-derived strings (XSS fix)
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

from omni.dashboard.theme import pill, regime_gradient


def _esc(s: object) -> str:
    """HTML-escape a value for safe interpolation into st.markdown(unsafe_allow_html=True).

    Phase 24.16 — added because news titles, URLs, and ticker symbols were
    being interpolated raw into HTML across components.py. A news source
    pushing a `<script>` tag in a headline would have executed in users'
    browsers. This escapes &, <, >, ", and ' which is what html.escape
    does with quote=True.
    """
    return _html.escape(str(s if s is not None else ""), quote=True)


def _safe_url(url: object) -> str:
    """Validate a URL is http(s) and escape it; replace anything suspicious
    with '#'. Defangs javascript:, data:, file: URIs in news links."""
    s = str(url or "").strip()
    if not s:
        return "#"
    try:
        p = urlparse(s)
    except Exception:  # noqa: BLE001
        return "#"
    if p.scheme not in ("http", "https"):
        return "#"
    return _html.escape(s, quote=True)

try:
    import plotly.graph_objects as go  # type: ignore
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    go = None  # type: ignore
    HAS_PLOTLY = False


_REGIME_COLORS = {
    "bull":   "#16a34a",
    "chop":   "#ca8a04",
    "bear":   "#dc2626",
    "crisis": "#7c3aed",
}
_DIR_LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}
_DIR_KIND = {1: "long", -1: "short", 0: "flat"}


# ---------- plotly theme helper ----------

def _plotly_layout(*, height: int = 260, **kw) -> dict:
    """Transparent plotly layout that lets the page gradient show through."""
    base = dict(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    base.update(kw)
    return base


# ---------- regime banner ----------

def regime_banner(regime_df: pd.DataFrame) -> None:
    if regime_df is None or regime_df.empty:
        st.info("No regime data yet.")
        return
    df = regime_df.sort_values("date").copy()
    last = df.iloc[-1]
    rg = str(last["regime"]).lower()
    grad = regime_gradient(rg)

    # ---- compute regime intelligence ----
    # 1. Days in current regime (consecutive same-regime tail)
    rev = df["regime"].astype(str).str.lower().values[::-1]
    days_in_regime = 1
    for r in rev[1:]:
        if r == rg:
            days_in_regime += 1
        else:
            break

    # 2. Transition matrix (empirical) over the visible window
    seq = df["regime"].astype(str).str.lower().tolist()
    transitions: dict[str, dict[str, int]] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        transitions.setdefault(a, {}).setdefault(b, 0)
        transitions[a][b] += 1
    next_probs = transitions.get(rg, {})
    total_next = sum(next_probs.values()) or 1
    # Top-2 most likely next states
    next_sorted = sorted(next_probs.items(), key=lambda kv: -kv[1])[:2]
    transition_str = "  ".join(
        f"<span style='opacity:0.85'>{nb.upper()} <b>{(c/total_next)*100:.0f}%</b></span>"
        for nb, c in next_sorted
    ) or "<span style='opacity:0.7'>insufficient history</span>"

    # 3. Window stats (how often current regime appeared)
    window_share = (df["regime"].astype(str).str.lower() == rg).mean() * 100

    # 4. What this regime means + suggested posture
    regime_meaning = {
        "bull":   ("Risk-on. Trend-following + momentum signals work well.",
                     "favour LONG conviction signals; tighten stops below 20-DMA"),
        "chop":   ("Range-bound. Mean-reversion edges out trend-following.",
                     "halve position sizes; avoid breakout chasers; trade ranges"),
        "bear":   ("Risk-off. Quality + low-vol names outperform; correlations rise.",
                     "raise the conviction gate to 75; prefer SHORT or cash"),
        "crisis": ("Stress regime. Liquidity premia spike; correlations -> 1.",
                     "consider de-risking entirely; only trade highest-conviction signals"),
    }.get(rg, ("Unknown regime.", "use default rubric weights"))

    # short history strip — last 30 sessions as colored squares
    hist = df.tail(30)
    blocks = "".join(
        f"<span title='{pd.to_datetime(r['date']).date()} · {r['regime']}' "
        f"style='display:inline-block;width:14px;height:22px;margin-right:2px;"
        f"border-radius:3px;background:{_REGIME_COLORS.get(str(r['regime']).lower(),'#475569')};"
        f"opacity:0.9'></span>"
        for _, r in hist.iterrows()
    )
    st.markdown(
        f"""
        <div class='regime-banner' style='background:{grad}'>
            <div style='display:flex;flex-direction:column;gap:4px'>
                <div style='font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;
                            opacity:0.85;font-weight:600'>Current regime</div>
                <div class='label-big'>{rg.upper()}</div>
                <div class='label-meta' style='opacity:0.92'>
                    <b>{days_in_regime}</b> day{'s' if days_in_regime != 1 else ''} in regime  ·
                    {window_share:.0f}% of last {len(df)} sessions  ·
                    as of {pd.to_datetime(last['date']).date()}
                </div>
                <div class='label-meta' style='font-size:0.78rem;opacity:0.92;margin-top:4px;
                            font-style:italic'>{regime_meaning[0]}</div>
                <div class='label-meta' style='font-size:0.78rem;opacity:0.95;
                            margin-top:2px'><b>Posture:</b> {regime_meaning[1]}</div>
            </div>
            <div style='text-align:right;display:flex;flex-direction:column;gap:6px'>
                <div style='font-size:0.7rem;letter-spacing:0.14em;opacity:0.85;
                            text-transform:uppercase'>last 30 sessions</div>
                <div>{blocks}</div>
                <div style='font-size:0.72rem;opacity:0.92;margin-top:4px;
                            letter-spacing:0.06em'>
                    <span style='opacity:0.75'>likely next:</span> {transition_str}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- sparkline helper (UI8) ----------

def sparkline_svg(values: list[float] | "pd.Series", *, width: int = 80,
                  height: int = 18) -> str:
    """Inline SVG polyline of a time series. Returns HTML string.

    Direction (cyan up / red down) is inferred from the first vs last value.
    Safe against empty / NaN-only inputs.
    """
    try:
        if hasattr(values, "tolist"):
            vs = [v for v in values.tolist() if v is not None]
        else:
            vs = [v for v in (values or []) if v is not None]
        vs = [float(v) for v in vs if v == v]  # drop NaN
    except Exception:
        return ""
    if len(vs) < 2:
        return ""
    lo, hi = min(vs), max(vs)
    rng = hi - lo if hi > lo else 1.0
    n = len(vs)
    pts = []
    for i, v in enumerate(vs):
        x = (i / (n - 1)) * (width - 2) + 1
        y = height - 1 - ((v - lo) / rng) * (height - 2)
        pts.append(f"{x:.1f},{y:.1f}")
    cls = "up" if vs[-1] >= vs[0] else "down"
    return (
        f"<svg class='sparkline {cls}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline points='{' '.join(pts)}' /></svg>"
    )


# ---------- KPI row ----------

def kpi_row(summary: dict) -> None:
    if not summary:
        st.warning("No backtest summary available.")
        return
    top = summary.get("top_decile", {}) or {}
    gated = summary.get("gated", {}) or {}
    edge = summary.get("edge", {}) or {}

    cols = st.columns(5)
    with cols[0]:
        st.metric("Hit rate", f"{summary.get('hit_rate', 0):.1%}",
                  f"{edge.get('edge_hit_rate_pp', 0):+.1f}pp vs shuffled")
    with cols[1]:
        st.metric("AUC", f"{summary.get('auc', 0):.3f}",
                  f"{edge.get('edge_auc_pp', 0):+.1f}pp")
    with cols[2]:
        st.metric("Top decile", f"{top.get('precision', 0):.1%}",
                  f"n={top.get('n', 0)}")
    with cols[3]:
        st.metric("Gated hit rate", f"{gated.get('directional_hit_rate', 0):.1%}",
                  f"{gated.get('n_emitted', 0)} emitted")
    with cols[4]:
        st.metric("Brier", f"{summary.get('brier', 0):.3f}",
                  f"logloss {summary.get('log_loss', 0):.2f}")


# ---------- universe heatmap ----------

def universe_heatmap(signals: pd.DataFrame) -> None:
    if signals is None or signals.empty:
        st.info("No signals yet.")
        return
    df = signals.copy()
    if "direction" not in df.columns or "conviction" not in df.columns:
        st.info("Signals frame missing direction/conviction.")
        return
    df["score"] = df["direction"].astype(float) * df["conviction"].astype(float)
    pivot = df.pivot_table(index="asset_class", columns="ticker",
                           values="score", aggfunc="first")
    try:
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            zmin=-100, zmax=100,
            colorscale=[
                [0.0, "#dc2626"], [0.25, "#f87171"], [0.5, "rgba(148,163,184,0.25)"],
                [0.75, "#22d3ee"], [1.0, "#06b6d4"],
            ],
            colorbar=dict(title="signed conviction", thickness=10, len=0.7),
            hovertemplate="<b>%{x}</b><br>%{y}<br>score: %{z:.0f}<extra></extra>",
        ))
        fig.update_layout(**_plotly_layout(height=260))
        st.plotly_chart(fig, width='stretch')
    except ImportError:
        styled = pivot.style.background_gradient(cmap="RdYlGn", vmin=-100, vmax=100) \
            .format("{:.0f}", na_rep="—")
        st.dataframe(styled, width='stretch')


# ---------- signal card ----------

def signal_card(row: pd.Series) -> None:
    direction = int(row.get("direction", 0))
    label = _DIR_LABEL[direction]
    kind = _DIR_KIND[direction]
    conviction = float(row.get("conviction", 0) or 0)
    blended = float(row.get("blended_p", 0) or 0)
    ticker = str(row.get("ticker", "?"))
    ac = str(row.get("asset_class", ""))
    rg = str(row.get("regime", ""))
    passes = bool(row.get("passes_gate", False))
    reasons = str(row.get("reasons", "") or "")

    pillars = {p: row.get(f"pillar_{p}", np.nan)
               for p in ("technical", "fundamental", "sentiment", "macro")}
    pdf = pd.DataFrame({"pillar": list(pillars.keys()),
                        "score": list(pillars.values())}).dropna()

    # Emit a single glass card with HTML rather than Streamlit container so the
    # gradient text on the ticker symbol can render consistently.
    pills_html = (
        pill(label, kind) + " " +
        pill(rg, rg) + " " +
        pill(ac.upper() or "?", "flat")
    )
    body = f"""
    <div class='signal-card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;
                  gap:10px'>
        <div>
          <div class='ticker'>{ticker}</div>
          <div class='meta'>{ac} · regime {rg}</div>
        </div>
        <div style='text-align:right'>
          <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:700'>
            {conviction:.0f}
          </div>
          <div class='meta'>p = {blended:.3f}</div>
        </div>
      </div>
      <div class='conviction-bar'>
        <div class='conviction-fill' style='width:{min(100,max(0,conviction)):.0f}%'></div>
      </div>
      <div style='margin-top:10px'>{pills_html}</div>
      {("<div class='meta' style='margin-top:8px'>gated out: "+reasons+"</div>") if not passes else ""}
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)

    # Phase 25 — plain-English verdict line directly under the card so the
    # Top-signals grid is readable without expanding the detail view.
    try:
        from omni.dashboard.verdict import verdict as _verdict_fn
        _v = _verdict_fn(row.to_dict() if hasattr(row, "to_dict") else dict(row))
        _color = {
            "long": "#16a34a", "short": "#dc2626",
            "warn": "#d97706", "flat": "var(--text-muted)",
        }.get(_v.tone, "var(--text-muted)")
        st.markdown(
            f"<div style='margin-top:6px;font-size:0.85rem;"
            f"color:{_color};font-weight:600'>{_v.headline}</div>"
            f"<div style='font-size:0.78rem;color:var(--text-secondary);"
            f"line-height:1.35'>{_v.summary}</div>",
            unsafe_allow_html=True,
        )
    except Exception:  # noqa: BLE001
        pass

    if not pdf.empty:
        try:
            colors = ["#06b6d4", "#8b5cf6", "#ec4899", "#22d3ee"]
            fig = go.Figure(go.Bar(
                x=pdf["pillar"], y=pdf["score"],
                marker=dict(color=colors[: len(pdf)],
                            line=dict(color="rgba(255,255,255,0.15)", width=1)),
                hovertemplate="%{x}: %{y:.0f}<extra></extra>",
            ))
            fig.update_layout(**_plotly_layout(height=160,
                              yaxis=dict(range=[0, 100],
                                         gridcolor="rgba(148,163,184,0.15)",
                                         zeroline=False),
                              xaxis=dict(zeroline=False)))
            st.plotly_chart(fig, width='stretch',
                            config={"displayModeBar": False})
        except ImportError:
            st.bar_chart(pdf, x="pillar", y="score", height=160)


# ---------- calibration panel ----------

def calibration_panel(preds: pd.DataFrame) -> None:
    if preds is None or preds.empty:
        st.info("No predictions for calibration.")
        return
    from omni.backtest.metrics import calibration_table, decile_lift

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Reliability**")
            tab = calibration_table(preds, n_bins=10)
            if not tab.empty:
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tab["bin_mean_p"], y=tab["bin_mean_y"],
                        mode="lines+markers", name="model",
                        line=dict(color="#06b6d4", width=3),
                        marker=dict(size=9, color="#8b5cf6",
                                    line=dict(color="white", width=1)),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                        line=dict(color="rgba(148,163,184,0.5)", dash="dash"),
                    ))
                    fig.update_layout(**_plotly_layout(
                        height=260,
                        xaxis=dict(title="predicted p", range=[0, 1],
                                   gridcolor="rgba(148,163,184,0.15)"),
                        yaxis=dict(title="empirical hit rate", range=[0, 1],
                                   gridcolor="rgba(148,163,184,0.15)"),
                    ))
                    st.plotly_chart(fig, width='stretch',
                                    config={"displayModeBar": False})
                except ImportError:
                    tab["perfect"] = tab["bin_mean_p"]
                    st.line_chart(tab.set_index("bin_mean_p")[["bin_mean_y", "perfect"]],
                                  height=240)
    with c2:
        with st.container(border=True):
            st.markdown("**Decile lift (mean realized 5d return)**")
            lift = decile_lift(preds, n_bins=10)
            if not lift.empty:
                try:
                    colors = ["#06b6d4" if v >= 0 else "#dc2626"
                              for v in lift["mean_fwd_ret"]]
                    fig = go.Figure(go.Bar(
                        x=lift["decile"], y=lift["mean_fwd_ret"],
                        marker=dict(color=colors,
                                    line=dict(color="rgba(255,255,255,0.15)", width=1)),
                        hovertemplate="decile %{x}: %{y:.3%}<extra></extra>",
                    ))
                    fig.update_layout(**_plotly_layout(
                        height=260,
                        xaxis=dict(title="proba decile",
                                   gridcolor="rgba(148,163,184,0.15)"),
                        yaxis=dict(title="mean fwd 5d return",
                                   gridcolor="rgba(148,163,184,0.15)",
                                   tickformat=".1%"),
                    ))
                    st.plotly_chart(fig, width='stretch',
                                    config={"displayModeBar": False})
                except ImportError:
                    st.bar_chart(lift.set_index("decile")[["mean_fwd_ret"]], height=240)


# ---------- history ----------

def signal_history_table(signals: pd.DataFrame, *, max_rows: int = 200) -> None:
    if signals is None or signals.empty:
        st.info("No signal history.")
        return
    cols = [c for c in ["date", "ticker", "asset_class", "regime", "direction",
                        "conviction", "blended_p", "passes_gate", "fwd_ret_5d", "reasons"]
            if c in signals.columns]
    show = signals.sort_values(["date", "conviction"], ascending=[False, False])[cols].head(max_rows)

    column_config = {}
    if "conviction" in show.columns:
        column_config["conviction"] = st.column_config.ProgressColumn(
            "Conviction", min_value=0, max_value=100, format="%.0f")
    if "blended_p" in show.columns:
        column_config["blended_p"] = st.column_config.NumberColumn(
            "p", format="%.3f")
    if "fwd_ret_5d" in show.columns:
        column_config["fwd_ret_5d"] = st.column_config.NumberColumn(
            "fwd 5d", format="%.2f%%")

    st.dataframe(show, width='stretch', hide_index=True,
                 column_config=column_config)


# ---------- on-demand prediction card ----------

def provenance_strip(prov: dict) -> None:
    """Render a horizontal strip with per-source check/cross + DQ score pill."""
    if not prov:
        return
    score = int(prov.get("data_quality_score", 0))
    score_color = "#22c55e" if score >= 70 else ("#eab308" if score >= 50 else "#ef4444")

    def _badge(label: str, items: list[str], expected: list[str] | None = None) -> str:
        if expected is None:
            expected = items
        chips = []
        for src in expected:
            ok = src in items
            mark = "[+]" if ok else "[-]"
            color = "#16a34a" if ok else "#475569"
            chips.append(
                f"<span style='display:inline-block;padding:2px 8px;margin:2px;"
                f"border-radius:8px;background:rgba(148,163,184,0.12);"
                f"color:{color};font-family:JetBrains Mono,monospace;"
                f"font-size:0.72rem'>{mark} {src}</span>"
            )
        return (f"<div style='display:inline-block;margin-right:18px'>"
                f"<span style='font-size:0.7rem;letter-spacing:0.14em;"
                f"text-transform:uppercase;color:var(--text-muted);"
                f"margin-right:6px'>{label}</span>{''.join(chips)}</div>")

    prices_expected = ["yfinance", "polygon"]
    fund_expected = ["yfinance", "fmp"]
    news_expected = ["newsapi", "marketaux", "gdelt", "stocktwits"]

    html = "<div class='glass-card' style='padding:10px 14px;margin:10px 0'>"
    html += (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:10px;"
        f"background:{score_color}22;color:{score_color};"
        f"font-family:JetBrains Mono,monospace;font-size:0.8rem;font-weight:700;"
        f"margin-right:14px'>DQ {score}/100</span>"
    )
    html += _badge("prices", prov.get("prices", []), prices_expected)
    html += _badge("fundamentals", prov.get("fundamentals", []), fund_expected)
    html += _badge("news", prov.get("news_sources", []), news_expected)
    macro = bool(prov.get("macro_fresh"))
    regime = bool(prov.get("regime_fresh"))
    html += (
        f"<span style='font-size:0.72rem;color:"
        f"{'#22c55e' if macro else '#475569'}'>macro {'OK' if macro else 'OFF'}</span>"
        f" · "
        f"<span style='font-size:0.72rem;color:"
        f"{'#22c55e' if regime else '#475569'}'>regime {'OK' if regime else 'OFF'}</span>"
    )
    if prov.get("warnings"):
        html += f"<div style='margin-top:6px;color:#eab308;font-size:0.75rem'>! {' / '.join(prov['warnings'][:3])}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _qualitative_sharpe(s: float) -> str:
    if s >= 1.0:
        return "strong"
    if s >= 0.0:
        return "moderate"
    return "weak"


def _direction_arrow(direction: int) -> str:
    return "↑" if direction == 1 else ("↓" if direction == -1 else "→")


def _pillar_bars_html(pillars: dict) -> str:
    """Horizontal stacked pillar list — one row per pillar, accent on the top."""
    if not pillars:
        return ""
    items = [(k, float(v)) for k, v in pillars.items()
             if isinstance(v, (int, float)) and not pd.isna(v)]
    if not items:
        return ""
    top_label = max(items, key=lambda kv: kv[1])[0]
    rows = []
    for label, score in items:
        pct = max(0.0, min(100.0, score))
        is_top = "is-top" if label == top_label else ""
        rows.append(
            f"<div class='pillar-row'>"
            f"<div class='pillar-label'>{label.title()}</div>"
            f"<div class='pillar-track'>"
            f"<div class='pillar-fill {is_top}' style='width:{pct:.1f}%'></div>"
            f"</div>"
            f"<div class='pillar-score'>{score:.0f}</div>"
            f"</div>"
        )
    return "".join(rows)


def prediction_card(card_dict: dict) -> None:
    """Render the on-demand PredictionCard with a clean, minimalist layout.

    Sections (all inside ONE outer card, separated by hairline dividers):
        1. Header — ticker · name · pills (regime/gate/model)
        2. Results — 4 stat tiles
        3. Horizon table — tomorrow / week / month
        4. Price strip — last close + intraday + proba bar
        5. Why text
        6. Pillars (horizontal bar list)
        7. Side-by-side details (Data collected · News)
    """
    # ---- pull fields with safe defaults ----
    direction = int(card_dict.get("direction", 0))
    label = _DIR_LABEL[direction]
    kind = _DIR_KIND[direction]
    arrow = _direction_arrow(direction)
    conviction = float(card_dict.get("conviction", 0) or 0)
    blended = float(card_dict.get("blended_p", 0) or 0)
    proba_up = float(card_dict.get("proba_up", 0.5) or 0.5)
    ticker = str(card_dict.get("ticker", "?"))
    name = str(card_dict.get("display_name", ticker))
    ac = str(card_dict.get("asset_class", ""))
    rg = str(card_dict.get("regime", ""))
    asof = str(card_dict.get("asof", ""))[:10]
    passes = bool(card_dict.get("passes_gate", False))
    reasons = str(card_dict.get("reasons", "") or "")
    why = str(card_dict.get("why_text", ""))
    model_kind = str(card_dict.get("model_kind", "universal_5d"))
    fold_aucs = card_dict.get("fold_aucs")

    m1d = float(card_dict.get("move_1d_pct", 0.0) or 0.0)
    m1d_lo = float(card_dict.get("move_1d_low_pct", 0.0) or 0.0)
    m1d_hi = float(card_dict.get("move_1d_high_pct", 0.0) or 0.0)
    m1w = float(card_dict.get("move_1w_pct",
                                card_dict.get("expected_move_pct", 0.0)) or 0.0)
    m1w_lo = float(card_dict.get("move_1w_low_pct",
                                   card_dict.get("expected_low_pct", 0.0)) or 0.0)
    m1w_hi = float(card_dict.get("move_1w_high_pct",
                                   card_dict.get("expected_high_pct", 0.0)) or 0.0)
    m1m = float(card_dict.get("move_1m_pct", 0.0) or 0.0)
    m1m_lo = float(card_dict.get("move_1m_low_pct", 0.0) or 0.0)
    m1m_hi = float(card_dict.get("move_1m_high_pct", 0.0) or 0.0)
    sharpe = float(card_dict.get("realized_sharpe_1y", 0.0) or 0.0)
    last_close = float(card_dict.get("last_close", 0.0) or 0.0)
    last_close_date = str(card_dict.get("last_close_date", "") or "")
    last_intra = float(card_dict.get("last_intraday_price", 0.0) or 0.0)
    last_intra_at = str(card_dict.get("last_intraday_at", "") or "")
    coverage = card_dict.get("data_coverage", {}) or {}
    currency = "₹" if ac == "india" else ("$" if ac in ("us", "crypto") else "")

    # ---- section 1: header ----
    gate_html = (pill("gate: pass", "long") if passes
                  else pill("gate: blocked", "short"))
    pills_html = (
        f"{pill('regime: ' + rg, rg)} {gate_html} "
        f"{pill('model: ' + model_kind.replace('_', ' '), 'flat')}"
    )
    header = (
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"gap:var(--space-4);flex-wrap:wrap;'>"
        f"<div>"
        f"<div style='font-size:var(--text-3xl);font-weight:800;letter-spacing:-0.02em;"
        f"color:var(--text-primary);line-height:1.05'>{ticker}</div>"
        f"<div style='font-size:var(--text-xs);color:var(--text-muted);margin-top:var(--space-1);"
        f"font-family:JetBrains Mono,monospace'>"
        f"{name} · {ac.upper()} · as of {asof}"
        f"</div>"
        f"</div>"
        f"<div style='display:flex;gap:var(--space-2);flex-wrap:wrap;align-items:center'>"
        f"{pills_html}"
        f"</div>"
        f"</div>"
    )
    if not passes and reasons:
        header += (
            f"<div style='font-size:var(--text-xs);color:var(--text-muted);"
            f"margin-top:var(--space-2);font-family:JetBrains Mono,monospace'>"
            f"gated out: {reasons}</div>"
        )

    # ---- open the outer card ----
    st.markdown(f"<div class='card'>{header}<hr class='hairline'/>",
                unsafe_allow_html=True)

    # ---- plain-English verdict banner (Phase 25) ----
    # Surfaces a 1-sentence trader-facing summary BEFORE the stat tiles,
    # because direction=FLAT + conviction=12 is opaque without context.
    # The CIANAGRO May 8 backtest (model bearish, rubric still bullish,
    # gate blocked, realised h+5 = -22.6%) was the motivating case:
    # without this banner that disagreement is invisible to the user.
    try:
        from omni.dashboard.verdict import verdict as _verdict_fn
        _v = _verdict_fn(card_dict)
        _tone_color = {
            "long":  ("var(--positive)", "rgba(22,163,74,0.10)"),
            "short": ("var(--negative)", "rgba(220,38,38,0.10)"),
            "warn":  ("#d97706",         "rgba(217,119,6,0.10)"),
            "flat":  ("var(--text-muted)", "var(--surface-2)"),
        }.get(_v.tone, ("var(--text-muted)", "var(--surface-2)"))
        _disagreement_html = (
            f"<div style='margin-top:var(--space-2);font-size:var(--text-sm);"
            f"color:var(--text-secondary);font-style:italic'>{_v.disagreement}</div>"
            if _v.disagreement else ""
        )
        st.markdown(
            f"<div style='margin:var(--space-3) 0 var(--space-4) 0;"
            f"padding:var(--space-3) var(--space-4);"
            f"background:{_tone_color[1]};border-left:3px solid {_tone_color[0]};"
            f"border-radius:6px'>"
            f"<div style='font-size:var(--text-lg);font-weight:700;"
            f"color:{_tone_color[0]};letter-spacing:-0.01em'>{_v.headline}</div>"
            f"<div style='font-size:var(--text-sm);color:var(--text-primary);"
            f"margin-top:var(--space-1)'>{_v.summary}</div>"
            f"{_disagreement_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception as _exc:  # noqa: BLE001
        # Never let the verdict banner break the card render
        pass

    # ---- section 2: results — 4 stat tiles ----
    st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)

    dir_color = "positive" if direction == 1 else ("negative" if direction == -1 else "")
    move_color = "positive" if m1w >= 0.05 else ("negative" if m1w <= -0.05 else "")
    sign_w = "+" if m1w >= 0 else ""

    sharpe_color = ("positive" if sharpe >= 1.0
                     else "negative" if sharpe < 0
                     else "")

    stats_html = (
        f"<div style='display:grid;grid-template-columns:repeat(4, minmax(0,1fr));"
        f"gap:var(--space-5);'>"
        f"<div class='stat'>"
        f"<div class='stat-label'>Direction · 5d</div>"
        f"<div class='stat-value {dir_color}'>{arrow}&nbsp;{label}</div>"
        f"<div class='stat-delta'>proba_up {proba_up:.3f}</div>"
        f"</div>"
        f"<div class='stat'>"
        f"<div class='stat-label'>Expected · 5d</div>"
        f"<div class='stat-value {move_color}'>{sign_w}{m1w:.2f}%</div>"
        f"<div class='stat-delta'>{m1w_lo:+.2f}% / {m1w_hi:+.2f}% (±1σ)</div>"
        f"</div>"
        f"<div class='stat'>"
        f"<div class='stat-label'>Conviction</div>"
        f"<div class='stat-value'>{conviction:.0f}<span style='font-size:var(--text-base);"
        f"color:var(--text-muted);font-weight:500'>&nbsp;/ 100</span></div>"
        f"<div style='height:4px;background:var(--surface-2);border-radius:999px;"
        f"overflow:hidden;margin-top:var(--space-1)'>"
        f"<div style='height:100%;width:{min(100,max(0,conviction)):.0f}%;"
        f"background:var(--accent);border-radius:999px'></div></div>"
        f"<div class='stat-delta' style='margin-top:var(--space-1)'>"
        f"blended p {blended:.3f}</div>"
        f"</div>"
        f"<div class='stat'>"
        f"<div class='stat-label'>1y Sharpe</div>"
        f"<div class='stat-value {sharpe_color}'>{sharpe:.2f}</div>"
        f"<div class='stat-delta'>{_qualitative_sharpe(sharpe)} · annualised</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(stats_html, unsafe_allow_html=True)

    # ---- section 3: horizon projections ----
    if last_close > 0:
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Horizon projections</div>",
                    unsafe_allow_html=True)
        rows_h = [
            ("Tomorrow",   "1 trading day",   m1d, m1d_lo, m1d_hi),
            ("Next week",  "5 trading days",  m1w, m1w_lo, m1w_hi),
            ("Next month", "21 trading days", m1m, m1m_lo, m1m_hi),
        ]
        h_html = (
            "<table class='horizon-table'>"
            "<thead><tr>"
            "<th>Horizon</th><th>Move</th><th>Target</th><th>Range (±1σ)</th>"
            "</tr></thead><tbody>"
        )
        for lbl, sub, mu, lo, hi in rows_h:
            tgt = last_close * (1 + mu / 100)
            tgt_lo = last_close * (1 + lo / 100)
            tgt_hi = last_close * (1 + hi / 100)
            mu_color = ("var(--kpi-positive)" if mu > 0.05
                         else "var(--kpi-negative)" if mu < -0.05
                         else "var(--text-muted)")
            sign = "+" if mu >= 0 else ""
            h_html += (
                f"<tr>"
                f"<td><div class='h-label'>{lbl}</div>"
                f"<div class='h-sub'>{sub}</div></td>"
                f"<td class='h-num' style='color:{mu_color}'>{sign}{mu:.2f}%</td>"
                f"<td class='h-num'>{currency}{tgt:,.2f}</td>"
                f"<td class='h-range'>{currency}{tgt_lo:,.2f} – {currency}{tgt_hi:,.2f}"
                f"<br>{lo:+.2f}% / {hi:+.2f}%</td>"
                f"</tr>"
            )
        h_html += "</tbody></table>"
        st.markdown(h_html, unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:var(--text-xs);color:var(--text-muted);"
            "margin-top:var(--space-2)'>"
            "Drift extrapolates linearly from the 5-day model edge; "
            "±1σ band uses realised daily vol. 1d and 1m are first-order estimates.</div>",
            unsafe_allow_html=True,
        )

    # ---- section 4: price + proba bar ----
    if last_close > 0:
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        intraday_html = ""
        if last_intra > 0 and last_intra_at:
            intraday_html = (
                f"<span style='color:var(--text-muted);font-size:var(--text-xs)'>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;intraday </span>"
                f"<span style='font-family:JetBrains Mono,monospace;font-weight:600'>"
                f"{currency}{last_intra:,.2f}</span>"
                f"<span style='color:var(--text-muted);font-size:var(--text-xs)'>"
                f"&nbsp;@ {last_intra_at[:16]}</span>"
            )
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"gap:var(--space-4);flex-wrap:wrap;margin-bottom:var(--space-3)'>"
            f"<div>"
            f"<span style='font-size:var(--text-micro);font-weight:600;color:var(--text-muted);"
            f"text-transform:uppercase;letter-spacing:0.10em'>Last close</span>"
            f"&nbsp;&nbsp;<span style='font-family:JetBrains Mono,monospace;"
            f"font-size:var(--text-lg);font-weight:700'>{currency}{last_close:,.2f}</span>"
            f"&nbsp;<span style='color:var(--text-muted);font-size:var(--text-xs);"
            f"font-family:JetBrains Mono,monospace'>({last_close_date})</span>"
            f"{intraday_html}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        # proba bar
        pu_pct = max(0.0, min(100.0, proba_up * 100))
        bp_pct = max(0.0, min(100.0, blended * 100))
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:var(--text-micro);"
            f"color:var(--text-muted);margin-bottom:var(--space-1);"
            f"text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>"
            f"<span>bearish</span><span>0.50</span><span>bullish</span></div>"
            f"<div class='proba-bar'>"
            f"<div class='midline'></div>"
            f"<div class='fill' style='width:{pu_pct:.1f}%'></div>"
            f"<div class='blended-marker' style='left:{bp_pct:.1f}%'></div>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:var(--text-xs);"
            f"color:var(--text-muted);margin-top:var(--space-1);"
            f"font-family:JetBrains Mono,monospace'>"
            f"<span>proba_up {proba_up:.3f}</span>"
            f"<span>blended {blended:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- section 5: why text ----
    if why:
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Why</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='line-height:1.55;font-size:var(--text-sm);"
            f"color:var(--text-secondary);max-width:78ch'>{why}</div>",
            unsafe_allow_html=True,
        )

    # ---- Phase 24.18 / 24.22 — circuit-risk badge (India equities only).
    # Was previously only rendered when the band was elevated/high. We now
    # render at every level (low/elevated/high) so the user always sees the
    # number on India names: "Lower-circuit risk: low (4%)" tells them the
    # crash-tail model has been consulted and the number is small. Below
    # 0.05 we still soft-style as a one-line caption to avoid a chunky
    # green-banner for normal names.
    crisk_pct = card_dict.get("circuit_risk_pct")
    crisk_band = card_dict.get("circuit_risk_band")
    if ac == "india" and crisk_pct is not None:
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        if crisk_band == "high":
            bg, border, icon, headline = (
                "rgba(239, 68, 68, 0.16)", "rgba(239, 68, 68, 0.55)",
                "⚠", "Lower-circuit risk: HIGH",
            )
        elif crisk_band == "elevated":
            bg, border, icon, headline = (
                "rgba(234, 179, 8, 0.16)", "rgba(234, 179, 8, 0.50)",
                "⚡", "Lower-circuit risk: elevated",
            )
        else:  # low
            bg, border, icon, headline = (
                "rgba(34, 197, 94, 0.10)", "rgba(34, 197, 94, 0.30)",
                "○", "Lower-circuit risk: low",
            )
        st.markdown(
            f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:10px;padding:10px 14px;display:flex;"
            f"justify-content:space-between;align-items:center;'>"
            f"<div style='font-weight:600;font-size:var(--text-sm);'>"
            f"{icon} {headline}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:var(--text-sm);"
            f"opacity:0.9;'>p ≈ {float(crisk_pct):.1%}</div>"
            f"</div>"
            f"<div style='font-size:var(--text-xs);color:var(--text-muted);"
            f"margin-top:6px;'>Probability of a ≥9.5% drawdown over the next "
            f"5 trading days. Sparse-positive XGBoost on price + microstructure "
            f"+ news-narrative features (PR-AUC measured OOF).</div>",
            unsafe_allow_html=True,
        )

        # Phase 24.22 — surface the underlying circuit-history evidence so
        # the user can sanity-check the badge. Without this, a chronic-LC
        # name silently overridden up to "high" looks identical to a model
        # genuinely-confident "high" — and conversely a clean name dampened
        # back to "low" needs to show why.
        chist = card_dict.get("circuit_history") or {}
        if chist:
            hits_5 = int(chist.get("hits_5d", 0) or 0)
            hits_20 = int(chist.get("hits_20d", 0) or 0)
            hits_60 = int(chist.get("hits_60d", 0) or 0)
            locked_now = bool(chist.get("currently_locked", False))
            last_dir = chist.get("last_lock_direction")
            streak = int(chist.get("consecutive_lock_days", 0) or 0)
            days_since = int(chist.get("days_since_last_lock", 10**6) or 0)
            parts = []
            if locked_now:
                parts.append(
                    f"<b>currently locked at {last_dir or 'unknown'} circuit</b>"
                )
            if streak > 1:
                parts.append(f"streak: {streak} consecutive locked sessions")
            if hits_5:
                parts.append(f"{hits_5}× in last 5d")
            if hits_20:
                parts.append(f"{hits_20}× in last 20d")
            if hits_60 == 0 and chist.get("n_bars", 0) >= 60:
                parts.append("0 circuits in last 60d")
            if not locked_now and hits_60 and days_since < 10**5:
                parts.append(f"last lock: {days_since}d ago ({last_dir})")
            if parts:
                st.markdown(
                    "<div style='font-size:var(--text-xs);"
                    "color:var(--text-muted);margin-top:4px;"
                    "font-family:JetBrains Mono,monospace;'>"
                    "circuit history: "
                    + " · ".join(parts)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ---- section 6: pillars ----
    pillars = card_dict.get("pillars", {}) or {}
    if pillars:
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Rubric pillars</div>",
                    unsafe_allow_html=True)
        st.markdown(_pillar_bars_html(pillars), unsafe_allow_html=True)

    if fold_aucs:
        avg = sum(fold_aucs) / len(fold_aucs)
        st.markdown(
            "<div style='font-size:var(--text-xs);color:var(--text-muted);"
            f"margin-top:var(--space-3);font-family:JetBrains Mono,monospace'>"
            f"deep-analysis fold AUCs: "
            + " · ".join(f"{a:.3f}" for a in fold_aucs)
            + f"  (mean {avg:.3f})</div>",
            unsafe_allow_html=True,
        )

    # ---- section 6b: intraday microstructure ----
    iret = float(card_dict.get("intraday_return_pct", 0.0) or 0.0)
    ibars = float(card_dict.get("intraday_bar_count", 0.0) or 0.0)
    if ibars >= 3:
        gap = float(card_dict.get("gap_open_pct", 0.0) or 0.0)
        vwd = float(card_dict.get("vwap_dev_bps", 0.0) or 0.0)
        orb = float(card_dict.get("opening_range_breakout", 0.0) or 0.0)
        ivol = float(card_dict.get("intraday_realized_vol", 0.0) or 0.0)
        lhr = float(card_dict.get("last_hour_return_pct", 0.0) or 0.0)
        vz = float(card_dict.get("volume_z_today", 0.0) or 0.0)
        rc = float(card_dict.get("range_compression", 0.0) or 0.0)
        tilt = bool(card_dict.get("intraday_tilt_applied", False))
        ir_color = "positive" if iret > 0.05 else ("negative" if iret < -0.05 else "")
        orb_label = ("breakout up" if orb > 0 else "breakdown" if orb < 0 else "inside range")
        tilt_pill = pill("1d tilt: ON", "long") if tilt else pill("1d tilt: off", "flat")
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:var(--space-2)'>"
            f"<div class='section-label' style='margin:0'>Intraday microstructure</div>"
            f"<div>{tilt_pill}</div></div>",
            unsafe_allow_html=True,
        )
        intra_html = (
            f"<div style='display:grid;grid-template-columns:repeat(4, minmax(0,1fr));"
            f"gap:var(--space-3);'>"
            f"<div class='stat'><div class='stat-label'>Intraday return</div>"
            f"<div class='stat-value {ir_color}'>{iret:+.2f}%</div>"
            f"<div class='stat-delta'>{ibars:.0f} bars · gap {gap:+.2f}%</div></div>"
            f"<div class='stat'><div class='stat-label'>VWAP deviation</div>"
            f"<div class='stat-value'>{vwd:+.0f} bps</div>"
            f"<div class='stat-delta'>last hour {lhr:+.2f}%</div></div>"
            f"<div class='stat'><div class='stat-label'>Opening range</div>"
            f"<div class='stat-value'>{orb_label}</div>"
            f"<div class='stat-delta'>compression {rc:.2f}×</div></div>"
            f"<div class='stat'><div class='stat-label'>Volume z · realised vol</div>"
            f"<div class='stat-value'>{vz:+.2f}σ</div>"
            f"<div class='stat-delta'>session vol {ivol:.2f}%</div></div>"
            f"</div>"
        )
        st.markdown(intra_html, unsafe_allow_html=True)

    # ---- section 6c: pair / cross-asset hedge ----
    ps = card_dict.get("pair_signal") or {}
    if ps and ps.get("benchmark"):
        suggestion = str(ps.get("suggestion", "no edge"))
        is_edge = "no edge" not in suggestion
        sug_kind = "long" if "long " + ticker in suggestion else ("short" if is_edge else "flat")
        alpha = float(ps.get("predicted_alpha_pct", 0.0))
        sharpe_p = float(ps.get("pair_sharpe", 0.0))
        beta = float(ps.get("beta_60d", 1.0))
        corr = float(ps.get("correlation_60d", 0.0))
        rvol = float(ps.get("residual_vol_20d_pct", 0.0))
        a_color = "positive" if alpha > 0.05 else ("negative" if alpha < -0.05 else "")
        st.markdown("<hr class='hairline'/>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:var(--space-2)'>"
            f"<div class='section-label' style='margin:0'>Pair trade · cross-asset</div>"
            f"<div>{pill(suggestion, sug_kind)}</div></div>",
            unsafe_allow_html=True,
        )
        pair_html = (
            f"<div style='display:grid;grid-template-columns:repeat(4, minmax(0,1fr));"
            f"gap:var(--space-3);'>"
            f"<div class='stat'><div class='stat-label'>Benchmark</div>"
            f"<div class='stat-value' style='font-size:var(--text-lg)'>{ps['benchmark']}</div>"
            f"<div class='stat-delta'>{ps.get('benchmark_asset_class','')}</div></div>"
            f"<div class='stat'><div class='stat-label'>β₆₀ · ρ₆₀</div>"
            f"<div class='stat-value'>{beta:+.2f}</div>"
            f"<div class='stat-delta'>correlation {corr:+.2f}</div></div>"
            f"<div class='stat'><div class='stat-label'>Predicted α · 5d</div>"
            f"<div class='stat-value {a_color}'>{alpha:+.2f}%</div>"
            f"<div class='stat-delta'>residual vol {rvol:.2f}%</div></div>"
            f"<div class='stat'><div class='stat-label'>Pair Sharpe</div>"
            f"<div class='stat-value'>{sharpe_p:+.2f}</div>"
            f"<div class='stat-delta'>edge {'yes' if is_edge else 'no'}</div></div>"
            f"</div>"
        )
        st.markdown(pair_html, unsafe_allow_html=True)

    # ---- close outer card ----
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- section 7: details (separate cards below the main one) ----
    e_data, e_news = st.columns(2)
    with e_data:
        n_news = int(coverage.get("news_count", len(card_dict.get("news_top5") or [])))
        with st.expander("Data collected", expanded=False):
            rows = [
                ("price rows",       f"{int(coverage.get('price_rows', 0)):,}"),
                ("price range",      f"{coverage.get('price_first_date','—')}  →  "
                                      f"{coverage.get('price_last_date','—')}"),
                ("history",          f"{coverage.get('history_years', 0):.1f} years"),
                ("price sources",    ", ".join(coverage.get("prices_sources") or []) or "—"),
                ("fundamentals",     f"{int(coverage.get('fundamentals_fields', 0))} fields · "
                                      + (", ".join(coverage.get("fundamentals_sources") or []) or "—")),
                ("news",             f"{n_news} headlines · "
                                      + (", ".join(coverage.get("news_sources") or []) or "—")),
                ("macro fresh",      "yes" if coverage.get("macro_fresh") else "no"),
                ("regime fresh",     "yes" if coverage.get("regime_fresh") else "no"),
                ("realised vol 5d",  f"{float(card_dict.get('recent_vol_5d_pct',0.0)):.2f}%"),
                ("data quality",     f"{int(coverage.get('data_quality_score', 0))} / 100"),
            ]
            tbl = ("<table style='width:100%;border-collapse:collapse;"
                   "font-size:var(--text-sm)'>")
            for k, v in rows:
                tbl += (
                    f"<tr><td style='padding:var(--space-2) 0;color:var(--text-muted);"
                    f"width:42%;font-size:var(--text-xs);text-transform:uppercase;"
                    f"letter-spacing:0.08em;font-weight:600'>{k}</td>"
                    f"<td style='padding:var(--space-2) 0;font-family:JetBrains Mono,monospace;"
                    f"font-variant-numeric:tabular-nums'>{v}</td></tr>"
                )
            tbl += "</table>"
            st.markdown(tbl, unsafe_allow_html=True)

    with e_news:
        news = card_dict.get("news_top5") or []
        # Phase 24.22 — Grow-style compact news cards. Was a flat list of
        # `<a>{title}</a> ({score}) {source}` lines; now each headline is a
        # tile with a colored sentiment-rail down the left, source + relative
        # timestamp on the same row, and the headline as the primary title.
        # Closer to the per-ticker news strip that Groww / Moneycontrol show.
        with st.expander(f"News · {len(news)} headlines", expanded=False):
            if not news:
                st.caption("No news pulled this round.")
            # Prepare a "X hours ago" formatter once
            now = pd.Timestamp.utcnow().tz_localize(None)

            def _rel_time(iso_str: str) -> str:
                if not iso_str:
                    return ""
                try:
                    t = pd.to_datetime(iso_str, errors="coerce", utc=True)
                    if pd.isna(t):
                        return ""
                    delta = now - t.tz_localize(None)
                    secs = int(delta.total_seconds())
                    if secs < 60:
                        return "just now"
                    if secs < 3600:
                        return f"{secs // 60}m ago"
                    if secs < 86400:
                        return f"{secs // 3600}h ago"
                    return f"{secs // 86400}d ago"
                except Exception:  # noqa: BLE001
                    return ""

            for n in news:
                s = float(n.get("sent_score", 0) or 0)
                rail_color = ("#22c55e" if s > 0.2
                              else "#ef4444" if s < -0.2
                              else "#94a3b8")
                badge_bg = ("rgba(34, 197, 94, 0.15)" if s > 0.2
                            else "rgba(239, 68, 68, 0.15)" if s < -0.2
                            else "rgba(148, 163, 184, 0.15)")
                badge_label = ("BULLISH" if s > 0.2
                               else "BEARISH" if s < -0.2
                               else "NEUTRAL")
                # Phase 24.16 — escape title/source and validate URL.
                # News providers are upstream input; do not trust them.
                _t = _esc(n.get("title", ""))
                _u = _safe_url(n.get("url", ""))
                _src = _esc(n.get("source", ""))
                _when = _rel_time(n.get("published_at", "") or n.get("date", ""))
                meta_bits = [b for b in (_src, _when) if b]
                meta_line = " · ".join(meta_bits)
                st.markdown(
                    f"<div style='display:flex;gap:12px;padding:12px 14px;"
                    f"margin-bottom:8px;border:1px solid var(--border-subtle);"
                    f"border-left:3px solid {rail_color};border-radius:8px;"
                    f"background:var(--surface-1);'>"
                    f"<div style='flex:1;min-width:0;'>"
                    f"<a href='{_u}' style='color:var(--text-primary);"
                    f"text-decoration:none;font-weight:600;font-size:var(--text-sm);"
                    f"line-height:1.4;display:block;' target='_blank' rel='noopener noreferrer'>{_t}</a>"
                    f"<div style='display:flex;gap:10px;align-items:center;margin-top:6px;'>"
                    f"<span style='font-size:9px;font-weight:700;letter-spacing:0.08em;"
                    f"padding:2px 8px;border-radius:999px;background:{badge_bg};"
                    f"color:{rail_color};'>{badge_label}</span>"
                    f"<span style='font-size:var(--text-xs);color:var(--text-muted);'>"
                    f"{meta_line}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;"
                    f"font-size:11px;color:{rail_color};margin-left:auto;'>"
                    f"{s:+.2f}</span>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

