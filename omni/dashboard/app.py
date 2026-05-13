"""Omni_Stocks dashboard entry point — modern glassmorphism build.

Run with:
    streamlit run omni/dashboard/app.py

Layout:
  Hero band (gradient cyan/violet, page title + subtitle)
  Sidebar: theme toggle, asset-class filter, conviction floor, only-emitted toggle, refresh
  Main:
    1. Regime banner (gradient + 30-session strip)
    2. KPI metric row (5 metrics in glass tiles)
    3. Tabs:
       - Universe heatmap (Plotly, transparent)
       - Top signal cards (glass cards with rubric pillar bars)
       - Calibration & decile lift (Plotly)
       - Signal history (dataframe with progress columns)

Read-only: never refits a model. All data is loaded from artifacts written
by the daily orchestrator (Phase 9). Falls back to demo data for first-run.
"""
from __future__ import annotations

# ---- sys.path bootstrap ----
# When Streamlit launches this file directly (`streamlit run omni/dashboard/app.py`),
# Python only adds `omni/dashboard/` to sys.path, not the project root, so
# `from omni.dashboard import ...` fails with ModuleNotFoundError. Walk up
# until we find the project root (the dir containing `omni/`) and prepend it.
import sys as _sys
from pathlib import Path as _Path
_here = _Path(__file__).resolve()
_root = None
# Phase 24.13 — was `(_here.parent, *_here.parents)` which duplicates the
# first element (parents[0] IS parent). And without `_root = None` first,
# a `del _root` on a copied app.py outside the omni tree would NameError.
for _candidate in _here.parents:
    if (_candidate / "omni" / "__init__.py").exists():
        _root = str(_candidate)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        break
del _sys, _Path, _here, _candidate, _root

from datetime import datetime, timezone
import logging

import pandas as pd
import streamlit as st
from html import escape as _esc

_log = logging.getLogger("omni.dashboard")

from omni.dashboard import cache as _c
from omni.dashboard import components as _u
from omni.dashboard.theme import inject_theme, hero, pill
from omni.ondemand import model_registry
from omni.ondemand.universe import Resolution, resolve, suggestions


def _theme_override(choice: str) -> None:
    """Inject CSS variables to re-skin the page.

    Phase 24.7 — overrides the new `--theme-c{1..4}`, `--theme-bg-stop{1..3}`,
    `--theme-accent{,-2,-3}` vars used by the Phase 23.5 aurora mesh, so the
    user's pick actually changes the visual theme end-to-end.

    Phase 24.16 — additionally overrides the BASE surface/text/border tokens
    (--surface-0/1/2, --text-primary/secondary/muted, --border-subtle/strong)
    so Light theme actually produces a light page. Previously the picker only
    swapped the aurora background; everything else stayed dark because the
    base CSS keys those tokens off `@media (prefers-color-scheme: dark)` which
    follows the OS, not the user's in-app choice. We force the tokens with
    !important on `html`/`body`/`.stApp` to beat the media query.
    """
    # ------------------------------------------------------------------ tokens
    DARK_BASE = {
        "--surface-0": "#0a0e1a",
        "--surface-1": "#11172a",
        "--surface-2": "#182037",
        "--border-subtle": "rgba(148, 163, 184, 0.10)",
        "--border-strong": "rgba(148, 163, 184, 0.22)",
        "--text-primary": "#e6ecf5",
        "--text-secondary": "#94a3b8",
        "--text-muted": "#64748b",
        "--kpi-positive": "#34d399",
        "--kpi-negative": "#f87171",
    }
    LIGHT_BASE = {
        "--surface-0": "#f8fafc",
        "--surface-1": "#ffffff",
        "--surface-2": "#f1f5f9",
        "--border-subtle": "rgba(15, 23, 42, 0.10)",
        "--border-strong": "rgba(15, 23, 42, 0.20)",
        "--text-primary": "#0f172a",
        "--text-secondary": "#334155",
        "--text-muted": "#64748b",
        "--kpi-positive": "#059669",
        "--kpi-negative": "#dc2626",
    }

    # Each theme is a tuple of CSS variable assignments
    THEMES: dict[str, dict[str, str]] = {
        # Default cosmic dark — keep as-is so "Auto" matches the baseline
        "Auto": {},
        # Pure dark — minimal aurora, near-black, cool blue accent
        "Dark": {
            **DARK_BASE,
            "--theme-c1": "oklch(0.30 0.10 240 / 0.45)",
            "--theme-c2": "oklch(0.30 0.10 220 / 0.35)",
            "--theme-c3": "oklch(0.28 0.08 200 / 0.30)",
            "--theme-c4": "oklch(0.30 0.08 260 / 0.25)",
            "--theme-bg-stop1": "#020409",
            "--theme-bg-stop2": "#04060d",
            "--theme-bg-stop3": "#020409",
            "--theme-accent": "#3b82f6",
            "--theme-accent-2": "#06b6d4",
            "--theme-accent-3": "#0ea5e9",
            "color-scheme": "dark",
        },
        # Light — clean, nearly white, slate text
        "Light": {
            **LIGHT_BASE,
            "--theme-c1": "oklch(0.95 0.05 240 / 0.55)",
            "--theme-c2": "oklch(0.94 0.06 280 / 0.45)",
            "--theme-c3": "oklch(0.96 0.04 200 / 0.40)",
            "--theme-c4": "oklch(0.95 0.06 30 / 0.30)",
            "--theme-bg-stop1": "#f8fafc",
            "--theme-bg-stop2": "#eef2f7",
            "--theme-bg-stop3": "#f1f5f9",
            "--theme-accent": "#0284c7",
            "--theme-accent-2": "#7c3aed",
            "--theme-accent-3": "#0891b2",
            "color-scheme": "light",
        },
        # Cyberpunk — hot magenta + neon cyan + acid yellow over near-black
        "Cyberpunk": {
            **DARK_BASE,
            "--theme-c1": "oklch(0.55 0.30 340 / 0.70)",  # magenta/pink
            "--theme-c2": "oklch(0.65 0.25 200 / 0.60)",  # neon cyan
            "--theme-c3": "oklch(0.70 0.25 90 / 0.45)",   # acid yellow
            "--theme-c4": "oklch(0.45 0.30 320 / 0.55)",  # deep magenta
            "--theme-bg-stop1": "#0a0014",
            "--theme-bg-stop2": "#10001f",
            "--theme-bg-stop3": "#06000c",
            "--theme-accent": "#ff2bd6",
            "--theme-accent-2": "#00f0ff",
            "--theme-accent-3": "#fff200",
            "color-scheme": "dark",
        },
        # Synthwave — purple/pink/teal sunset, retro feel
        "Synthwave": {
            **DARK_BASE,
            "--theme-c1": "oklch(0.50 0.25 320 / 0.65)",
            "--theme-c2": "oklch(0.55 0.22 30 / 0.55)",
            "--theme-c3": "oklch(0.55 0.20 200 / 0.45)",
            "--theme-c4": "oklch(0.45 0.25 270 / 0.55)",
            "--theme-bg-stop1": "#0d0524",
            "--theme-bg-stop2": "#1a0a3a",
            "--theme-bg-stop3": "#0a0420",
            "--theme-accent": "#ff6ec7",
            "--theme-accent-2": "#a855f7",
            "--theme-accent-3": "#22d3ee",
            "color-scheme": "dark",
        },
        # Matrix — pure green-on-black hacker terminal
        "Matrix": {
            **DARK_BASE,
            "--theme-c1": "oklch(0.55 0.20 145 / 0.50)",
            "--theme-c2": "oklch(0.50 0.22 140 / 0.40)",
            "--theme-c3": "oklch(0.45 0.18 150 / 0.35)",
            "--theme-c4": "oklch(0.50 0.20 130 / 0.30)",
            "--theme-bg-stop1": "#000600",
            "--theme-bg-stop2": "#020a02",
            "--theme-bg-stop3": "#000400",
            "--theme-accent": "#00ff9c",
            "--theme-accent-2": "#22c55e",
            "--theme-accent-3": "#84cc16",
            "color-scheme": "dark",
        },
        # Nord — calm muted polar palette
        "Nord": {
            **DARK_BASE,
            "--theme-c1": "oklch(0.50 0.10 240 / 0.50)",
            "--theme-c2": "oklch(0.55 0.10 200 / 0.40)",
            "--theme-c3": "oklch(0.50 0.08 220 / 0.35)",
            "--theme-c4": "oklch(0.55 0.10 160 / 0.30)",
            "--theme-bg-stop1": "#1a212c",
            "--theme-bg-stop2": "#222933",
            "--theme-bg-stop3": "#181d26",
            "--theme-accent": "#88c0d0",
            "--theme-accent-2": "#81a1c1",
            "--theme-accent-3": "#a3be8c",
            "color-scheme": "dark",
        },
    }

    if choice not in THEMES or not THEMES[choice]:
        return  # Auto / unknown — let base CSS rule

    css_vars = "; ".join(f"{k}: {v}" for k, v in THEMES[choice].items())
    # Inject with !important so the user choice beats the @property defaults.
    extras = ""
    if choice == "Cyberpunk":
        # Cyberpunk needs extra punch: glow on text, cyan/magenta KPI delta chips
        extras = """
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
          color: #ff2bd6 !important;
          text-shadow: 0 0 18px rgba(255, 43, 214, 0.55), 0 0 36px rgba(0, 240, 255, 0.25);
        }
        h1, h2, h3 {
          text-shadow: 0 0 16px rgba(255, 43, 214, 0.35);
        }
        .ultra-marquee__track > b.up { color: #00f0ff !important;
          text-shadow: 0 0 12px rgba(0, 240, 255, 0.6); }
        .ultra-marquee__track > b.dn { color: #ff2bd6 !important;
          text-shadow: 0 0 12px rgba(255, 43, 214, 0.6); }
        """
    elif choice == "Matrix":
        extras = """
        body, .stApp, [data-testid="stMetric"] [data-testid="stMetricValue"],
        h1, h2, h3, p, span, div {
          font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
          color: #00ff9c !important;
          text-shadow: 0 0 14px rgba(0, 255, 156, 0.6);
        }
        """
    elif choice == "Light":
        # Phase 24.16 — Light needs full surface inversion. Beat the
        # `_ULTRA_FX_CSS` aurora gradient with !important and re-key all
        # hard-coded dark colour-mix() values that show up in stMetric tiles.
        extras = """
        html, body, .stApp, [data-testid="stAppViewContainer"] {
          color: #0f172a !important;
        }
        /* Aurora mesh — light-mode override of the dark gradient stack */
        html, body, .stApp, [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(at 20% 12%, rgba(125, 211, 252, 0.20) 0%, transparent 55%),
            radial-gradient(at 80% 18%, rgba(196, 181, 253, 0.18) 0%, transparent 55%),
            radial-gradient(at 65% 85%, rgba(165, 243, 252, 0.15) 0%, transparent 55%),
            linear-gradient(180deg, #f8fafc 0%, #eef2f7 50%, #f1f5f9 100%) !important;
        }
        section[data-testid="stSidebar"] {
          background: #ffffff !important;
          border-right: 1px solid rgba(15,23,42,0.08) !important;
        }
        section[data-testid="stSidebar"] * { color: #0f172a !important; }
        [data-testid="stMetric"] {
          background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(248,250,252,0.95)) !important;
          border-color: rgba(15,23,42,0.10) !important;
          color: #0f172a !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0f172a !important; text-shadow: none !important; }
        [data-testid="stMetric"] [data-testid="stMetricLabel"] { color: rgba(15,23,42,0.6) !important; }
        .signal-card { background: #ffffff !important; border-color: rgba(15,23,42,0.08) !important; color: #0f172a !important; }
        .stat, .stat-value, .stat-label, .stat-delta, .meta { color: #0f172a !important; }
        .stat-label, .stat-delta, .meta { color: rgba(15,23,42,0.6) !important; }
        /* baseweb selectbox menu */
        [data-baseweb="popover"] [role="listbox"] { background: #ffffff !important; color: #0f172a !important; }
        """

    # Wrap in :root + html + body selectors to ensure tokens beat the
    # `@media (prefers-color-scheme: dark)` block in _CSS. We append
    # `!important` to each CSS-variable assignment via an inline rewrite.
    forced = "; ".join(f"{k}: {v} !important" for k, v in THEMES[choice].items())
    # Sync the base `--accent` (used by buttons, sliders, focus rings) with
    # the picked theme's `--theme-accent`, so a Cyberpunk pick actually shows
    # magenta buttons rather than cyan ones.
    accent = THEMES[choice].get("--theme-accent")
    accent_strong = THEMES[choice].get("--theme-accent-2", accent)
    if accent:
        forced += f"; --accent: {accent} !important; --accent-strong: {accent_strong} !important"

    st.markdown(
        f"<style>"
        f":root, html, body, .stApp {{ {forced} }}"
        f"{extras}"
        f"</style>",
        unsafe_allow_html=True,
    )


def _suggest_options(query: str) -> list[str]:
    """Return labels for the autocomplete: 'TICKER  —  Display name'.

    Phase 24.16 — dropped the middle-dot `·` separator and the redundant
    exchange tag. The ticker suffix (.NS, -USD, etc.) already encodes the
    venue, and the dot rendered as garbage on some Windows fonts. Now uses
    a single em-dash separator and a left-aligned two-column layout.

    Uses multi-source live search (Yahoo + NSE + CoinGecko) when query is
    long enough; falls back to curated list for empty / very-short queries.
    """
    sugs = suggestions(query or "", k=15)
    out = []
    for s in sugs:
        # Trim long names
        name = s.display_name[:60] + "…" if len(s.display_name) > 60 else s.display_name
        # Two-column layout: ticker pad-right to 16 chars, em dash, name.
        out.append(f"{s.ticker:<16}  —  {name}")
    return out


def _parse_label(label: str) -> str:
    """Extract the ticker from a 'TICKER  —  name' label, or return as-is.
    Tolerates the legacy '·' separator for any cached labels still in
    session_state from the pre-24.16 format.
    """
    sep = "—" if "—" in label else ("·" if "·" in label else "")
    if sep:
        return label.split(sep, 1)[0].strip()
    return label.strip()


def _render_history_pills(entries: list[dict], *, prefix: str,
                            show_count: bool = False) -> None:
    """Render a row of clickable pills for search-history entries.

    Click → write the entry into the selectbox session_state and rerun
    so the user lands on it ready to predict.
    """
    if not entries:
        st.caption("No entries yet.")
        return
    arrow_map = {1: "▲", -1: "▼", 0: "●"}
    cols = st.columns(min(len(entries), 4))
    exch_map = {"india": "NSE", "us": "US", "crypto": "CRYPTO",
                 "fx": "FX", "etf": "ETF", "index": "IDX"}
    for i, e in enumerate(entries):
        with cols[i % len(cols)]:
            tk = e.get("ticker", "?")
            ac = e.get("asset_class", "us")
            ld = e.get("last_direction")
            arrow = arrow_map.get(ld, "·") if ld is not None else "·"
            cnt = f"  ×{e.get('hit_count', 1)}" if show_count else ""
            label = f"{arrow} {tk}{cnt}"
            if st.button(label, key=f"sh_{prefix}_{i}", use_container_width=True,
                            help=(
                                f"{e.get('display_name', tk)}\n"
                                f"asset: {ac}  ·  exchange: {e.get('exchange') or '?'}\n"
                                f"hits: {e.get('hit_count', 1)}\n"
                                + (f"last p(up): {e.get('last_proba_up'):.3f}\n"
                                    if e.get("last_proba_up") is not None else "")
                                + (f"last conviction: {e.get('last_conviction'):.1f}\n"
                                    if e.get("last_conviction") is not None else "")
                                + (f"regime: {e.get('last_regime')}\n"
                                    if e.get("last_regime") else "")
                            )):
                exch = exch_map.get(ac, "?")
                st.session_state["predict_ticker_select"] = (
                    f"{tk:<16}  —  {e.get('display_name', tk)}"
                )
                st.rerun()


def _render_watchlist_tab() -> None:
    """User's saved favorites — predicted on demand, with a remove button.

    Persists to data/watchlist/favorites.json (survives Streamlit restarts).
    """
    from omni.dashboard import watchlist as _wl
    from omni.ondemand.predict import predict_ticker

    favs = _wl.list_favorites()
    st.caption(
        "Your favorite tickers — survives browser refresh and app restart. "
        "Add tickers from the **Predict** tab using the ☆ button."
    )

    if not favs:
        st.info(
            "Watchlist is empty. Go to the **Predict** tab, search for a "
            "ticker (e.g. *RELIANCE*, *cian agro*), then click "
            "**☆ Add to watchlist**."
        )
        return

    cols = st.columns([3, 1, 1])
    cols[0].markdown(f"**{len(favs)} ticker{'s' if len(favs) != 1 else ''} on your watchlist**")
    if cols[1].button("Refresh all", use_container_width=True, key="wl_refresh"):
        st.session_state["_wl_refresh_nonce"] = st.session_state.get("_wl_refresh_nonce", 0) + 1
    if cols[2].button("Clear all", use_container_width=True, key="wl_clear"):
        n = _wl.clear_all()
        st.warning(f"Removed {n} ticker(s).")
        st.rerun()

    rows = []
    for fav in favs:
        try:
            card = predict_ticker(
                fav["ticker"], fav["asset_class"],
                display_name=fav.get("display_name", fav["ticker"]),
            )
            if card is None:
                rows.append({
                    "ticker": fav["ticker"],
                    "name": fav.get("display_name", fav["ticker"]),
                    "asset_class": fav["asset_class"],
                    "direction": "—",
                    "prob_up": None,
                    "conviction": None,
                    "regime": "—",
                    "passes_gate": False,
                    "added_at": fav.get("added_at", "")[:10],
                })
                continue
            d = card.to_dict()
            arrow = {1: "▲ long", -1: "▼ short", 0: "● flat"}.get(d.get("direction", 0), "?")
            rows.append({
                "ticker": d["ticker"],
                "name": fav.get("display_name", d["ticker"]),
                "asset_class": d.get("asset_class", fav["asset_class"]),
                "direction": arrow,
                "prob_up": round(float(d.get("proba_up", 0.5)), 3),
                "conviction": round(float(d.get("conviction", 0)), 1),
                "regime": d.get("regime", "—"),
                "passes_gate": bool(d.get("passes_gate", False)),
                "added_at": fav.get("added_at", "")[:10],
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "ticker": fav["ticker"],
                "name": fav.get("display_name", fav["ticker"]),
                "asset_class": fav["asset_class"],
                "direction": f"err: {type(exc).__name__}",
                "prob_up": None, "conviction": None, "regime": "—",
                "passes_gate": False, "added_at": fav.get("added_at", "")[:10],
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-ticker remove buttons
    st.markdown("##### Manage")
    rcols = st.columns(min(4, max(1, len(favs))))
    for i, fav in enumerate(favs):
        with rcols[i % len(rcols)]:
            if st.button(f"× {fav['ticker']}", key=f"wl_rm_{fav['ticker']}",
                          use_container_width=True):
                _wl.remove_favorite(fav["ticker"])
                st.rerun()


def _render_movers_tab() -> None:
    from omni.dashboard.movers import top_movers

    st.caption(
        "Top 10 gainers and losers across the curated universe, recomputed from "
        "live OHLCV (cached 4h). Choose the scope and horizon below."
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    scope = c1.selectbox(
        "Scope", ["india", "us", "crypto", "all"], index=0,
        key="movers_scope",
        help="india = top-50 NSE/BSE; us = US large caps; crypto = majors; all = everything",
    )
    horizon_label = c2.selectbox(
        "Horizon", ["Today (1d)", "This week (5d)", "This month (21d)"],
        index=0, key="movers_horizon",
    )
    top_k = c3.number_input("Top N", min_value=5, max_value=25, value=10,
                              step=5, key="movers_topk")
    horizon_map = {"Today (1d)": "ret_1d", "This week (5d)": "ret_5d",
                    "This month (21d)": "ret_21d"}
    horizon = horizon_map[horizon_label]

    with st.spinner(f"Computing returns for {scope} universe…"):
        gainers, losers = top_movers(scope=scope, horizon=horizon, top_k=int(top_k))

    if gainers.empty and losers.empty:
        st.warning(
            "No data available — yfinance may be rate-limited or the cache is empty. "
            "Try again in a few minutes."
        )
        return

    pct_label = horizon.replace("ret_", "").replace("d", "-day return %")
    g_show = gainers.rename(columns={horizon: f"return % ({pct_label})",
                                       "last_close": "close",
                                       "last_date": "as of"})
    l_show = losers.rename(columns={horizon: f"return % ({pct_label})",
                                      "last_close": "close",
                                      "last_date": "as of"})

    cg, cl = st.columns(2)
    with cg:
        st.markdown(f"### Top {len(gainers)} gainers")
        st.dataframe(g_show, use_container_width=True, hide_index=True)
    with cl:
        st.markdown(f"### Top {len(losers)} losers")
        st.dataframe(l_show, use_container_width=True, hide_index=True)


def _build_tracked_universe(*, include_history: bool = True,
                            history_limit: int = 20
                            ) -> list[tuple[str, str, str]]:
    """Deduped (ticker, asset_class, display_name) list from watchlist
    plus (optionally) recent searches. Used by Portfolio + Earnings tabs.
    """
    from omni.dashboard import watchlist as _wl
    from omni.dashboard import searchhistory as _sh

    universe: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for f in _wl.list_favorites():
        key = (f["ticker"].upper(), f.get("asset_class", "india"))
        if key in seen:
            continue
        seen.add(key)
        universe.append((f["ticker"], f.get("asset_class", "india"),
                          f.get("display_name", f["ticker"])))
    if include_history:
        for h in _sh.list_recent(history_limit):
            key = (h["ticker"].upper(), h.get("asset_class", "india"))
            if key in seen:
                continue
            seen.add(key)
            universe.append((h["ticker"], h.get("asset_class", "india"),
                              h.get("display_name", h["ticker"])))
    return universe


def _record_and_update_lru(res: Resolution, card_or_dict) -> None:
    """Persist the search to history.json + update the in-session LRU pills.
    Idempotent — safe to call after every successful prediction.
    Also fires a Streamlit toast (UI7) for live feedback.

    Phase 24.16 — accepts either a PredictionCard object (legacy in-process
    path) or a dict (FastAPI client path). Treats both the same.
    """
    from omni.dashboard import searchhistory as _sh

    if card_or_dict is not None:
        d = card_or_dict.to_dict() if hasattr(card_or_dict, "to_dict") \
            else dict(card_or_dict)
        _sh.record_search(
            res.ticker, res.asset_class,
            display_name=res.display_name,
            proba_up=d.get("proba_up"),
            direction=d.get("direction"),
            conviction=d.get("conviction"),
            regime=d.get("regime"),
        )
        # UI7 — toast notification (Streamlit ≥ 1.26)
        try:
            dir_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(d.get("direction", 0)), "FLAT")
            conv = float(d.get("conviction", 0))
            icon = "🟢" if conv >= 70 else ("🟡" if conv >= 40 else "⚪")
            st.toast(f"{res.ticker}: {dir_label} · conviction {conv:.0f}/100", icon=icon)
        except Exception:
            _log.debug("toast failed", exc_info=True)
    rec_list = st.session_state.get("_recent_tickers", [])
    rec_list = [r for r in rec_list if r.get("ticker") != res.ticker]
    rec_list.insert(0, {"ticker": res.ticker, "ac": res.asset_class,
                          "name": res.display_name})
    st.session_state["_recent_tickers"] = rec_list[:6]


def _render_portfolio_tab() -> None:
    """Portfolio mode: combine Kelly fractions across watchlist + recent searches.

    Filters to conviction >= threshold (slider, default 70), sums Kelly
    fractions, applies a per-sector cap, displays as a sized table plus a
    summary KPI row.
    """
    from omni.ondemand.predict import predict_ticker
    from omni.ondemand.kelly import kelly_from_card

    st.markdown("### Portfolio mode")
    st.caption(
        "Kelly-sized recommendations across your watchlist and recent "
        "searches. Total exposure is capped at 100 % of capital."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    conv_min = c1.slider("Min conviction", 0, 100, 70, step=5,
                          help="Only names above this conviction enter the book.")
    sector_cap = c2.slider("Per-sector cap (%)", 5, 100, 30, step=5,
                            help="No single sector exceeds this fraction of capital.")
    include_history = c3.checkbox("Include recent searches", value=True)

    universe = _build_tracked_universe(include_history=include_history)

    if not universe:
        st.info(
            "No tickers in watchlist or recent searches yet. "
            "Star a few names from the Predict tab to build a portfolio."
        )
        return

    rows: list[dict] = []
    progress = st.progress(0.0, text="scoring portfolio …")

    def _score_one(tk: str, ac: str, dn: str) -> dict | None:
        try:
            card = predict_ticker(tk, ac)
            if card is None:
                return {"_err": f"{tk}: no prediction available "
                                f"(ticker may be delisted/illiquid or has "
                                f"<60d of price history)"}
            cd = card.to_dict()
        except Exception as e:  # noqa: BLE001
            return {"_err": f"{tk}: {type(e).__name__}: {e}"}
        kr = kelly_from_card(cd)
        return {
            "ticker": tk,
            "name": dn,
            "asset_class": ac,
            "side": kr.side.upper(),
            "conviction": round(cd.get("conviction", 0), 1),
            "blended_p": round(cd.get("blended_p", 0.5), 3),
            "regime": cd.get("regime", "?"),
            "expected_high_pct": round(cd.get("expected_high_pct", 0), 2),
            "expected_low_pct": round(cd.get("expected_low_pct", 0), 2),
            "kelly_fraction": kr.fraction,
            "kelly_edge": round(kr.edge, 3),
        }

    # Parallel fan-out (3-10x speedup on cold-cache portfolios)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(universe)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_score_one, tk, ac, dn): tk
                        for tk, ac, dn in universe}
            for fut in as_completed(futures):
                # Phase 24.13 — bound per-ticker latency. Without this a
                # single hung yfinance call would block the whole portfolio
                # tab. 60s is generous; predict_ticker has its own internal
                # timeouts so this is a safety net.
                try:
                    r = fut.result(timeout=60)
                except Exception as _e:
                    tk_hung = futures.get(fut, "?")
                    _log.warning("portfolio: %s timed out / failed: %s", tk_hung, _e)
                    r = {"_err": f"{tk_hung}: timed out (>60s)"}
                if r is None:
                    pass
                elif "_err" in r:
                    st.warning(r["_err"])
                else:
                    rows.append(r)
                completed += 1
                progress.progress(completed / n)
    except Exception:
        _log.warning("portfolio threadpool failed; falling back to serial", exc_info=True)
        # Serial fallback for environments where threading misbehaves
        rows = []
        for i, (tk, ac, dn) in enumerate(universe):
            r = _score_one(tk, ac, dn)
            if r and "_err" not in r:
                rows.append(r)
            elif r and "_err" in r:
                st.warning(r["_err"])
            progress.progress((i + 1) / n)
    progress.empty()

    if not rows:
        st.warning("No predictions could be generated.")
        return

    pf = pd.DataFrame(rows)
    in_book = pf[(pf["conviction"] >= conv_min) & (pf["kelly_fraction"] > 0)].copy()
    if in_book.empty:
        st.info(
            f"None of {len(pf)} candidates pass the {conv_min}-conviction "
            f"gate today. Lower the slider or wait for the next regime."
        )
        st.dataframe(pf.sort_values("conviction", ascending=False),
                      use_container_width=True, hide_index=True)
        return

    # Per-sector cap (proxy: asset_class until we have a sector field)
    in_book = in_book.sort_values("kelly_edge", ascending=False).reset_index(drop=True)
    sector_running = {}
    cap = sector_cap / 100.0
    capped: list[float] = []
    for _, r in in_book.iterrows():
        sec = r["asset_class"]
        used = sector_running.get(sec, 0.0)
        room = max(0.0, cap - used)
        f = min(float(r["kelly_fraction"]), room)
        capped.append(f)
        sector_running[sec] = used + f
    in_book["allocated_fraction"] = capped

    # Total-exposure cap (100 %)
    tot = in_book["allocated_fraction"].sum()
    if tot > 1.0:
        in_book["allocated_fraction"] = in_book["allocated_fraction"] / tot

    in_book["allocated_pct"] = (in_book["allocated_fraction"] * 100).round(2)
    in_book = in_book.drop(columns=["kelly_fraction", "allocated_fraction"])

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Names in book", f"{len(in_book)} / {len(pf)}")
    k2.metric("Total exposure", f"{in_book['allocated_pct'].sum():.1f}%")
    longs = (in_book["side"] == "LONG").sum()
    shorts = (in_book["side"] == "SHORT").sum()
    k3.metric("Long / Short", f"{longs} / {shorts}")
    avg_edge = in_book["kelly_edge"].mean()
    k4.metric("Avg Kelly edge", f"{avg_edge:+.3f}")

    st.markdown("#### Recommended book")
    st.dataframe(
        in_book[["ticker", "name", "side", "conviction", "blended_p",
                  "regime", "expected_high_pct", "expected_low_pct",
                  "kelly_edge", "allocated_pct"]],
        use_container_width=True, hide_index=True,
    )

    with st.expander("Below conviction gate (not in book)"):
        rest = pf[~pf["ticker"].isin(in_book["ticker"])].sort_values(
            "conviction", ascending=False)
        if rest.empty:
            st.caption("none")
        else:
            st.dataframe(rest, use_container_width=True, hide_index=True)


def _render_replay_section() -> None:
    """Drag a date slider over the logged-prediction window and see the
    cards as they were emitted, with realised forward returns attached."""
    from datetime import timedelta
    try:
        from omni.monitoring.replay import replay_window, replay_summary
    except Exception as exc:  # noqa: BLE001
        st.caption(f"_replay unavailable: {exc}_")
        return

    st.markdown("### Backtest replay")
    st.caption(
        "Logged-prediction archive joined to realised forward returns. "
        "Use the slider to step through historical days."
    )

    today = pd.Timestamp.now().normalize()
    earliest = (today - timedelta(days=180)).date()
    latest = today.date()

    rng = st.slider(
        "Date range",
        min_value=earliest, max_value=latest,
        value=(earliest, latest), key="replay_range",
    )
    only_active = st.checkbox("Only direction != FLAT", value=True,
                              key="replay_active")

    rdf = replay_window(rng[0].isoformat(), rng[1].isoformat())
    if rdf is None or rdf.empty:
        st.info(
            "No predictions logged in this window yet. The replay populates "
            "as the daily orchestrator runs and `predict_ticker` calls log."
        )
        return

    if only_active:
        rdf = rdf[rdf["direction"] != 0]

    summary = replay_summary(rdf)
    k1, k2, k3 = st.columns(3)
    k1.metric("Predictions", summary.get("n_active", len(rdf)))
    hr = summary.get("hit_rate")
    k2.metric("Hit rate", f"{hr*100:.1f} %" if hr is not None else "—")
    mr = summary.get("mean_realised_pct")
    k3.metric("Mean realised", f"{mr:+.2f} %" if mr is not None else "—")

    st.dataframe(
        rdf[["asof", "ticker", "asset_class", "horizon_days",
              "direction", "conviction", "proba_up", "expected_move_pct",
              "regime", "realised_ret", "hit"]]
        .sort_values("asof", ascending=False)
        .head(300),
        use_container_width=True, hide_index=True,
    )


def _render_earnings_tab() -> None:
    """Earnings-day specials: who's reporting next 7 days + per-ticker history."""
    try:
        from omni.ingest.earnings_calendar import (
            next_earnings, earnings_history,
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"_earnings calendar unavailable: {exc}_")
        return

    st.markdown("### Earnings calendar")
    st.caption(
        "Next reporting dates and historical surprise % for tracked tickers."
    )

    # ---- Section A: who's reporting in the next 7 days
    universe = _build_tracked_universe(history_limit=15)

    today = pd.Timestamp.now().normalize()
    rows: list[dict] = []
    if universe:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with st.spinner(f"Loading earnings calendar for {len(universe)} tickers…"):
            def _safe_next(tk, ac):
                try:
                    return tk, ac, next_earnings(tk, ac)
                except Exception:
                    return tk, ac, None
            ed_map: dict[tuple[str, str], object] = {}
            try:
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = [ex.submit(_safe_next, tk, ac)
                                for tk, ac, _ in universe]
                    for fut in as_completed(futures):
                        # Phase 24.13 — bounded result wait; same rationale
                        # as portfolio. Earnings calls hit yfinance.calendar
                        # which is more flaky than fast_info.
                        try:
                            tk, ac, ed = fut.result(timeout=30)
                        except Exception as _e:
                            _log.warning("earnings future timed out: %s", _e)
                            continue
                        ed_map[(tk, ac)] = ed
            except Exception:
                _log.warning("earnings threadpool failed; falling back to serial", exc_info=True)
                # Fallback to serial (some Streamlit-on-Windows envs misbehave)
                for tk, ac, _ in universe:
                    _, _, ed = _safe_next(tk, ac)
                    ed_map[(tk, ac)] = ed
        for tk, ac, dn in universe:
            ed = ed_map.get((tk, ac))
            if ed is None:
                continue
            days = (pd.Timestamp(ed).normalize() - today).days
            if 0 <= days <= 7:
                rows.append({
                    "ticker": tk,
                    "name": dn,
                    "asset_class": ac,
                    "earnings_date": pd.Timestamp(ed).date().isoformat(),
                    "days_until": days,
                })
    if rows:
        st.markdown("#### Reporting next 7 days")
        st.dataframe(pd.DataFrame(rows).sort_values("days_until"),
                      use_container_width=True, hide_index=True)
    else:
        st.info("No tracked tickers reporting in the next 7 days.")

    # ---- Section B: per-ticker drill-down
    st.markdown("#### Per-ticker history")
    default = universe[0][0] if universe else "INFY"
    tk = st.text_input("Ticker (e.g. INFY, RELIANCE, AAPL)", value=default,
                        key="earnings_drill")
    ac = st.selectbox("Asset class", ["india", "us"], index=0,
                       key="earnings_drill_ac")
    if tk:
        try:
            ed = next_earnings(tk, ac)
            hist = earnings_history(tk, ac)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"could not load: {exc}")
            return
        if ed is not None:
            days = (pd.Timestamp(ed).normalize() - today).days
            st.caption(
                f"Next earnings: **{pd.Timestamp(ed).date().isoformat()}** "
                f"({'in ' + str(days) + ' days' if days >= 0 else str(-days) + ' days ago'})"
            )
        else:
            st.caption("Next earnings date not available.")
        if hist is None or hist.empty:
            st.info("No historical earnings rows available.")
        else:
            st.dataframe(hist.head(8), use_container_width=True, hide_index=True)
            if "surprise_pct" in hist.columns:
                avg = hist["surprise_pct"].dropna().mean()
                st.metric("Avg surprise (last 4 quarters)",
                          f"{avg:+.2f} %" if pd.notna(avg) else "—")


def _render_dev_tab() -> None:
    """Developer admin panel — manual triggers for the maintenance jobs that
    normally run via cron / orchestrator. Useful for one-off refreshes.

    All actions are isolated and idempotent. Long-running jobs (model retrain)
    show a spinner; short ones return immediately. Cache stats are read live.

    Phase 24.14 — gated behind a password. Set OMNI_DEV_PASSWORD in `.env`.
    Without that env var the tab is locked and shows an explanatory message.
    """
    import hmac
    import os
    import subprocess
    import sys
    import time as _time
    from pathlib import Path

    # ---------- Phase 24.14 — password gate ----------
    expected = os.getenv("OMNI_DEV_PASSWORD", "").strip()

    if not expected:
        st.warning(
            "**Dev tab is disabled.** Set `OMNI_DEV_PASSWORD` in your `.env` "
            "to enable it. The tab can run subprocesses (model retraining, "
            "data ingest, cache purge) so it's gated by default."
        )
        return

    if not st.session_state.get("__dev_unlocked", False):
        st.markdown(
            "<div style='font-family:Inter,sans-serif;'>"
            "<div style='font-size:1.4rem;font-weight:800;background:var(--accent-glow);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
            "🔒 Developer Console</div>"
            "<div style='color:var(--text-muted);font-size:0.85rem;'>"
            "This area runs subprocesses and modifies on-disk state. "
            "Enter the dev password to continue.</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()
        col_pw, col_btn = st.columns([4, 1])
        with col_pw:
            pw = st.text_input(
                "Password", type="password", key="__dev_pw_input",
                label_visibility="collapsed", placeholder="Dev password",
            )
        with col_btn:
            unlock = st.button("Unlock", use_container_width=True, type="primary")
        if unlock:
            # Timing-safe comparison so an attacker can't infer length
            if hmac.compare_digest(pw or "", expected):
                st.session_state["__dev_unlocked"] = True
                st.session_state.pop("__dev_pw_input", None)
                st.rerun()
            else:
                st.error("Wrong password.")
                _log.warning("dev tab failed unlock attempt")
        return

    # ---------- unlocked — show full panel ----------

    # Phase 24.13 — cross-platform helpers for the dev-tab subprocess
    # buttons. Earlier hardcoded `.venv\\Scripts\\python.exe` only works on
    # Windows. `sys.executable` is the path of the *currently running*
    # interpreter regardless of OS, so it picks up the venv automatically.
    _PY = sys.executable
    def _script(*parts: str) -> str:
        """Build a script path that works on Windows + macOS + Linux."""
        return str(Path("scripts").joinpath(*parts))

    # Lock button at the top so it's always reachable
    col_h1, col_h2 = st.columns([5, 1])
    with col_h1:
        st.markdown(
            "<div style='font-family:Inter,sans-serif;'>"
            "<div style='font-size:1.4rem;font-weight:800;background:var(--accent-glow);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Developer Console</div>"
            "<div style='color:var(--text-muted);font-size:0.85rem;'>"
            "Manual maintenance triggers. These actions modify on-disk state. Use with care.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_h2:
        if st.button("🔒 Lock", use_container_width=True,
                     help="Re-lock the dev tab. You'll need to enter the password again."):
            st.session_state["__dev_unlocked"] = False
            st.rerun()
    st.divider()

    # --------- Cache stats ---------
    st.markdown("### Cache")
    try:
        from omni.ondemand.cache import get_cache, invalidate
        cache = get_cache()
        n_entries = len(cache)
        size_mb = cache.volume() / (1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        n_entries = 0
        size_mb = 0.0
        st.caption(f"cache unavailable: {exc}")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Cache entries", f"{n_entries:,}")
    with cols[1]:
        st.metric("Cache size", f"{size_mb:.1f} MB")
    with cols[2]:
        if st.button("🗑 Clear prediction cache", use_container_width=True,
                     help="Drops every cached PredictionCard and bench card. "
                          "Next predict_ticker() call will rebuild from live APIs."):
            n1 = invalidate("predict_card:")
            n2 = invalidate("deep_card:")
            n3 = invalidate("bench_card:")
            st.success(f"cleared {n1 + n2 + n3} cached prediction entries")
    with cols[3]:
        if st.button("🗑 Clear ALL caches", use_container_width=True,
                     help="Nuclear option — drops everything. "
                          "Next page load will re-fetch prices, news, fundamentals, macros."):
            cache.clear()
            st.success(f"cleared all {n_entries:,} cache entries")

    st.divider()

    # --------- Models ---------
    st.markdown("### Models")
    model_dir = Path("data/models")
    artifacts = []
    if model_dir.exists():
        for f in sorted(model_dir.glob("*.joblib")) + sorted(model_dir.glob("*.pkl")):
            try:
                mtime = _time.ctime(f.stat().st_mtime)
                size_mb = f.stat().st_size / (1024 * 1024)
                artifacts.append({"file": f.name, "size_MB": round(size_mb, 2),
                                  "modified": mtime})
            except Exception as exc:  # noqa: BLE001
                # File can disappear between the glob and the stat (e.g.
                # if a retrain replaces it mid-listing). Skip the row but
                # keep a debug breadcrumb so genuine permission/io errors
                # don't vanish silently.
                _log.debug("dev: skip artifact %s: %s", f.name, exc)
    if artifacts:
        st.dataframe(pd.DataFrame(artifacts), use_container_width=True, hide_index=True)
    else:
        st.caption("no model artifacts found in data/models/")

    cols = st.columns(3)
    with cols[0]:
        if st.button("🔁 Refit regime (HMM)", use_container_width=True,
                     help="Re-run scripts/refit_regime.py — pulls latest macro data "
                          "and refits the 4-state HMM. Updates 'Last regime point'."):
            with st.spinner("running refit_regime…"):
                try:
                    rc = subprocess.run(
                        [_PY, _script("retrain_all.py"), "--regime-only"],
                        capture_output=True, text=True, timeout=300,
                    )
                    if rc.returncode == 0:
                        st.success("regime refit done")
                    else:
                        st.error(f"failed: {rc.stderr[-500:]}")
                except subprocess.TimeoutExpired:
                    st.error("refit timed out (300s)")
                except FileNotFoundError:
                    st.warning("scripts/retrain_all.py --regime-only not wired; "
                                "run manually: python scripts/refit_regime.py")
    with cols[1]:
        if st.button("🔁 Retrain universal", use_container_width=True,
                     help="Walk-forward retrain of XGB+isotonic for h=1/5/21. "
                          "Takes 30-90s on GPU."):
            with st.spinner("retraining universal…"):
                try:
                    env = os.environ.copy()
                    env["OMNI_XGB_DEVICE"] = "cuda"
                    rc = subprocess.run(
                        [_PY, _script("train_universal_model.py"),
                         "--horizons", "1,5,21", "--force"],
                        capture_output=True, text=True, timeout=600, env=env,
                    )
                    if rc.returncode == 0:
                        st.success("universal retrain done")
                        st.code(rc.stdout[-1000:])
                    else:
                        st.error(f"failed: {rc.stderr[-500:]}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"retrain error: {exc}")
    with cols[2]:
        if st.button("🔁 Retrain stack ensemble", use_container_width=True,
                     help="Stack retrain h=5 with XGB+HGB+LGB+NHITS bases. "
                          "Takes 5-15 min on GPU."):
            with st.spinner("retraining stack…"):
                try:
                    env = os.environ.copy()
                    env["OMNI_XGB_DEVICE"] = "cuda"
                    env["OMNI_LGB_DEVICE"] = "cuda"
                    env["OMNI_NHITS_DEVICE"] = "cuda"
                    rc = subprocess.run(
                        [_PY, _script("train_universal_model.py"),
                         "--horizons", "5", "--ensemble", "stack", "--force"],
                        capture_output=True, text=True, timeout=1800, env=env,
                    )
                    if rc.returncode == 0:
                        st.success("stack retrain done")
                        st.code(rc.stdout[-1000:])
                    else:
                        st.error(f"failed: {rc.stderr[-500:]}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"retrain error: {exc}")

    st.divider()

    # --------- Data lake ---------
    st.markdown("### Data Lake")
    lake_root = Path("data/lake")
    lake_stats = {}
    try:
        prices_dir = lake_root / "prices"
        if prices_dir.exists():
            for ac_dir in prices_dir.iterdir():
                if ac_dir.is_dir():
                    lake_stats[f"prices/{ac_dir.name}"] = len(list(ac_dir.glob("*.parquet")))
        ts = lake_root / "training" / "training_set.parquet"
        if ts.exists():
            df = pd.read_parquet(ts)
            lake_stats["training_set rows"] = len(df)
            lake_stats["training_set cols"] = len(df.columns)
            lake_stats["training_set tickers"] = df["ticker"].nunique() if "ticker" in df.columns else 0
            lake_stats["training_set range"] = f"{df['date'].min().date()} → {df['date'].max().date()}"
    except Exception as exc:  # noqa: BLE001
        st.caption(f"lake stats unavailable: {exc}")

    if lake_stats:
        st.json(lake_stats)

    cols = st.columns(3)
    with cols[0]:
        if st.button("🌐 Refresh marquee", use_container_width=True,
                     help="Force a fresh yfinance fetch for the top ticker tape."):
            try:
                from omni.dashboard.marquee import fetch_marquee_data
                fetch_marquee_data.clear()         # st.cache_data.clear()
                items = fetch_marquee_data()
                up = sum(1 for i in items if i.direction > 0)
                dn = sum(1 for i in items if i.direction < 0)
                st.success(f"refreshed {len(items)} tickers · ▲{up} ▼{dn}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"refresh failed: {exc}")
    with cols[1]:
        if st.button("📊 Run SBI benchmark", use_container_width=True,
                     help="Sanity check: predict SBIN.NS end-to-end and compare "
                          "live vs offline path (parity test)."):
            with st.spinner("running benchmark_sbi.py…"):
                try:
                    rc = subprocess.run(
                        [_PY, _script("benchmark_sbi.py")],
                        capture_output=True, text=True, timeout=180,
                    )
                    st.code((rc.stdout + "\n" + rc.stderr)[-2500:], language="text")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"benchmark error: {exc}")
    with cols[2]:
        if st.button("🧹 Rebuild training set", use_container_width=True,
                     help="Re-run scripts/build_training_set.py — pulls all "
                          "tickers from lake and rebuilds the pooled training "
                          "frame. ~10-20 min."):
            with st.spinner("rebuilding training set… (~15 min)"):
                try:
                    rc = subprocess.run(
                        [_PY, _script("build_training_set.py"),
                         "--no-news", "--asset-classes", "us,india,crypto"],
                        capture_output=True, text=True, timeout=1800,
                    )
                    if rc.returncode == 0:
                        st.success("training set rebuilt")
                        st.code(rc.stdout[-1000:])
                    else:
                        st.error(f"failed: {rc.stderr[-500:]}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"build error: {exc}")

    st.divider()

    # --------- Phase 24.14 — Daily orchestrator + Drive backup ---------
    st.markdown("### Daily orchestrator & Drive backup")
    st.caption(
        "Run the full daily pipeline (mount Drive → ingest → features → "
        "regime → predict → rubric → gate → persist) and back up the data "
        "lake to your mounted Google Drive. None of this happens on a "
        "schedule by default — you have to wire that up via Windows Task "
        "Scheduler or `cron` (instructions below)."
    )

    # Live status of the Drive mount
    try:
        from omni.storage import drive as _drv
        drv_mounted = _drv.is_mounted()
        drv_root = str(_drv.drive_path()).split("Omni_Stocks")[0] + "Omni_Stocks"
    except Exception:
        drv_mounted = False
        drv_root = "(unknown)"

    drv_chip = (
        "<span style='color:#4ade80;background:rgba(74,222,128,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>MOUNTED</span>"
        if drv_mounted else
        "<span style='color:#f87171;background:rgba(248,113,113,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>NOT MOUNTED</span>"
    )
    st.markdown(
        f"<div style='font-size:13px;color:#cbd5e1;margin-bottom:8px;'>"
        f"Drive mount status: {drv_chip} "
        f"<code style='color:rgba(148,163,184,0.7);font-size:11px;'>{_esc(drv_root)}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        if st.button("⚡ Run daily orchestrator now", use_container_width=True,
                     help="Execute scripts/run_daily.py — full ingest + features + "
                          "predict + persist. ~5-15 min depending on universe size. "
                          "Output appears below when done."):
            with st.spinner("running daily orchestrator…"):
                try:
                    rc = subprocess.run(
                        [_PY, _script("run_daily.py")],
                        capture_output=True, text=True, timeout=1800,
                    )
                    tail = (rc.stdout + "\n" + rc.stderr)[-3000:]
                    if rc.returncode == 0:
                        st.success("daily run complete")
                        # Force-clear dashboard caches so new artifacts surface
                        _c.load_predictions.clear()
                        _c.load_signals.clear()
                        _c.load_regime.clear()
                        _c.load_backtest_summary.clear()
                    else:
                        st.error(f"failed (exit {rc.returncode})")
                    st.code(tail, language="text")
                except FileNotFoundError:
                    st.warning(
                        "`scripts/run_daily.py` not found. Use "
                        "`python -m omni.orchestrator.daily` directly, "
                        "or create the wrapper script."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"run failed: {exc}")
    with cols[1]:
        if st.button("☁️ Sync local lake → Drive", use_container_width=True,
                     disabled=not drv_mounted,
                     help="Copy data/lake/** to Drive root. Idempotent — only "
                          "new/changed files transfer. Mount Drive first."):
            with st.spinner("syncing to Drive…"):
                try:
                    src = Path("data/lake")
                    if not src.exists():
                        st.warning("`data/lake/` is empty — nothing to sync.")
                    else:
                        # Use robocopy on Windows / rsync on Unix for efficient
                        # incremental sync. Both are stdlib-free wrappers.
                        if sys.platform == "win32":
                            dst = Path(drv_root) / "lake"
                            cmd = ["robocopy", str(src), str(dst), "/E", "/MIR",
                                   "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/NS"]
                        else:
                            dst = Path(drv_root) / "lake"
                            dst.mkdir(parents=True, exist_ok=True)
                            cmd = ["rsync", "-a", "--delete",
                                   str(src) + "/", str(dst) + "/"]
                        rc = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=1800)
                        # robocopy success codes: 0-7 are non-fatal
                        ok = (rc.returncode <= 7) if sys.platform == "win32" \
                            else (rc.returncode == 0)
                        if ok:
                            st.success("sync complete")
                        else:
                            st.error(f"sync failed (exit {rc.returncode})")
                        st.code((rc.stdout + "\n" + rc.stderr)[-1500:], language="text")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"sync error: {exc}")
    with cols[2]:
        with st.expander("📅 How to schedule daily run", expanded=False):
            st.markdown(
                "**Windows (Task Scheduler):**\n"
                "```\n"
                "schtasks /create /tn OmniDaily /tr "
                "\"C:\\Omni_Stocks\\.venv\\Scripts\\python.exe "
                "C:\\Omni_Stocks\\scripts\\run_daily.py\" /sc daily /st 18:30\n"
                "```\n"
                "Runs every day at 6:30pm IST after market close.\n\n"
                "**macOS / Linux (cron):**\n"
                "```\n"
                "30 18 * * 1-5 cd ~/Omni_Stocks && "
                ".venv/bin/python scripts/run_daily.py >> data/daily.log 2>&1\n"
                "```\n"
                "Weekdays only; logs to `data/daily.log`."
            )

    st.divider()

    # --------- Phase 24.18 — Windows Task Scheduler integration ---------
    st.markdown("### ⏰ Scheduled tasks (Windows Task Scheduler)")
    st.caption(
        "Manage the daily orchestrator schedule directly from the UI. "
        "Install creates a daily 18:30 IST trigger for "
        "`scripts/run_daily.bat`. Install/Uninstall require admin "
        "(re-launch Streamlit as administrator); Run-now and Refresh "
        "work for any user."
    )

    try:
        from omni.ops import scheduler as _sched
        _sched_admin = _sched.is_admin()
        _sched_status = _sched.status()
        _sched_installed = bool(_sched_status.get("installed"))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"scheduler module unavailable: {exc}")
        _sched_admin = False
        _sched_status = {"installed": False}
        _sched_installed = False

    inst_chip = (
        "<span style='color:#4ade80;background:rgba(74,222,128,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>INSTALLED</span>"
        if _sched_installed else
        "<span style='color:#f87171;background:rgba(248,113,113,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>NOT INSTALLED</span>"
    )
    admin_chip = (
        "<span style='color:#4ade80;background:rgba(74,222,128,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>ADMIN</span>"
        if _sched_admin else
        "<span style='color:#fbbf24;background:rgba(251,191,36,0.12);"
        "padding:3px 10px;border-radius:6px;font-size:11px;font-weight:600;'>USER</span>"
    )
    last_run = _sched_status.get("last_run") or "—"
    last_result = _sched_status.get("last_result") or "—"
    next_run = _sched_status.get("next_run") or "—"
    state = _sched_status.get("state") or "—"
    st.markdown(
        f"<div style='font-size:13px;color:#cbd5e1;margin-bottom:8px;'>"
        f"task: <code style='color:#94a3b8;'>Omni_Stocks_Daily</code> &nbsp;{inst_chip}"
        f" &nbsp;process: {admin_chip}<br>"
        f"<span style='color:#94a3b8;font-size:11px;'>"
        f"state={_esc(state)} &nbsp; last_run={_esc(last_run)} &nbsp; "
        f"last_result={_esc(last_result)} &nbsp; next_run={_esc(next_run)}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    if not _sched_admin:
        st.caption(
            "💡 To install/uninstall: close Streamlit, right-click → "
            "**Run as administrator**, then come back here."
        )

    s_cols = st.columns(4)
    with s_cols[0]:
        if st.button(
            "📥 Install daily task", use_container_width=True,
            disabled=(_sched_installed or not _sched_admin),
            help="Create scheduled task that runs scripts/run_daily.bat "
                 "every day at 18:30 local time. Requires admin.",
        ):
            try:
                from omni.ops import scheduler as _sched
                ok, msg = _sched.install()
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"install failed: {exc}")
    with s_cols[1]:
        if st.button(
            "▶ Run task now", use_container_width=True,
            disabled=not _sched_installed,
            help="Trigger the scheduled task immediately. Runs in the "
                 "background — check `logs/daily_*.log` for output.",
        ):
            try:
                from omni.ops import scheduler as _sched
                ok, msg = _sched.run_now()
                (st.success if ok else st.error)(msg)
            except Exception as exc:  # noqa: BLE001
                st.error(f"run_now failed: {exc}")
    with s_cols[2]:
        if st.button(
            "📊 Refresh status", use_container_width=True,
            help="Re-query schtasks for current task state.",
        ):
            st.rerun()
    with s_cols[3]:
        if st.button(
            "🗑 Uninstall task", use_container_width=True,
            disabled=(not _sched_installed or not _sched_admin),
            help="Remove the scheduled task. Requires admin.",
        ):
            try:
                from omni.ops import scheduler as _sched
                ok, msg = _sched.uninstall()
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"uninstall failed: {exc}")

    st.divider()

    # --------- Phase 24.15 — RAG store maintenance ---------
    st.markdown("### RAG store (news embeddings for grounded why-text)")
    st.caption(
        "Local Qdrant store at `data/rag/qdrant/`. Holds chunked news "
        "(Nemotron 1024-d embeddings) used by the grounded why-text "
        "pipeline. Ingest is idempotent — re-running just refreshes any "
        "new articles. Predictions auto-ingest the news they pull, so "
        "the store grows organically; this button is for bulk warm-ups."
    )

    rag_cols = st.columns([2, 2, 2])
    with rag_cols[0]:
        try:
            from omni.llm.rag.store import RAGStore as _RAGStore
            _stats = _RAGStore().stats()
            stats_json = {
                "total_points": _stats.total_points,
                "collections": _stats.collections,
                "path": _stats.storage_path,
            }
            st.json(stats_json, expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"rag stats unavailable: {exc}")

    with rag_cols[1]:
        if st.button("🔍 Re-index news → RAG", use_container_width=True,
                     help="Walk all data/raw_fallback/news_* parquet files, "
                          "chunk them, embed via Nemotron, and upsert to "
                          "Qdrant. Idempotent. ~10 min for 30 days × full "
                          "universe."):
            with st.spinner("re-indexing news…"):
                try:
                    from omni.llm.rag.pipeline import get_pipeline
                    pipe = get_pipeline()
                    n_files = 0
                    n_articles = 0
                    n_chunks = 0
                    news_dirs = [
                        Path("data/raw_fallback/news_newsapi/all"),
                        Path("data/raw_fallback/news_rss/all"),
                    ]
                    for nd in news_dirs:
                        if not nd.exists():
                            continue
                        for parq in nd.glob("*.parquet"):
                            n_files += 1
                            try:
                                df = pd.read_parquet(parq)
                            except Exception:
                                continue
                            if df.empty or "ticker_hint" not in df.columns:
                                continue
                            for tk, grp in df.groupby("ticker_hint"):
                                if not tk:
                                    continue
                                tk_up = str(tk).upper()
                                # Heuristic: NSE tickers typically have .NS
                                # suffix, crypto has -USD, US tickers are bare.
                                if tk_up.endswith(".NS") or tk_up.endswith(".BO"):
                                    ac = "india"
                                elif tk_up.endswith("-USD") or tk_up.endswith("USD"):
                                    ac = "crypto"
                                else:
                                    ac = "us"
                                pipe.ingest_news_for_ticker(grp, ticker=tk_up,
                                                              asset_class=ac)
                                n_articles += len(grp)
                                n_chunks += len(grp)  # ~1 chunk per article
                    st.success(f"re-indexed {n_articles:,} articles "
                                f"({n_chunks:,} chunks) from {n_files} files")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"re-index failed: {exc}")

    with rag_cols[2]:
        purge_days = st.number_input(
            "Purge older than (days)", min_value=30, max_value=365,
            value=90, step=15,
            help="Drop chunks with date < (today - N days). Use to keep "
                 "the store small after long uptime.",
        )
        if st.button("🗑 Purge old chunks", use_container_width=True):
            try:
                from omni.llm.rag.store import RAGStore as _RAGStore
                _RAGStore().purge_older_than(int(purge_days))
                st.success(f"purged chunks older than {purge_days}d")
            except Exception as exc:  # noqa: BLE001
                st.error(f"purge failed: {exc}")

    st.divider()

    # --------- Phase 24.14 — Indian live data smoke test ---------
    st.markdown("### Indian live data (NSE / BSE)")
    st.caption(
        "Free real-time quotes for NSE/BSE via the `nsepython` library "
        "(scrapes nseindia.com directly — ~1s latency, no API key needed). "
        "Polygon's free tier doesn't cover Indian markets; yfinance is "
        "15-min delayed and IP-blocks aggressively."
    )
    try:
        import importlib
        importlib.import_module("nsepython")
        nsepy_installed = True
    except ImportError:
        nsepy_installed = False

    cols = st.columns([2, 1, 2])
    with cols[0]:
        nse_ticker = st.text_input(
            "Ticker", value="RELIANCE", key="nse_test_ticker",
            help="NSE symbol without .NS suffix (e.g. RELIANCE, TCS, SBIN, INFY)",
        )
    with cols[1]:
        if not nsepy_installed:
            if st.button("📦 Install nsepython", use_container_width=True,
                         help="Runs `pip install nsepython`. ~30s."):
                with st.spinner("installing…"):
                    try:
                        rc = subprocess.run(
                            [_PY, "-m", "pip", "install", "nsepython"],
                            capture_output=True, text=True, timeout=180,
                        )
                        if rc.returncode == 0:
                            st.success("installed — refresh page to enable")
                        else:
                            st.error(f"install failed: {rc.stderr[-500:]}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"install error: {exc}")
        else:
            if st.button("📡 Fetch live quote", use_container_width=True,
                         type="primary"):
                with st.spinner("fetching from nseindia.com…"):
                    try:
                        from nsepython import nse_eq  # type: ignore
                        q = nse_eq(nse_ticker.strip().upper())
                        info = q.get("priceInfo", {}) or {}
                        st.session_state["__nse_quote"] = {
                            "symbol": nse_ticker.upper(),
                            "ltp": info.get("lastPrice"),
                            "change": info.get("change"),
                            "pct": info.get("pChange"),
                            "vwap": info.get("vwap"),
                            "open": info.get("open"),
                            "high": info.get("intraDayHighLow", {}).get("max"),
                            "low": info.get("intraDayHighLow", {}).get("min"),
                            "prev_close": info.get("previousClose"),
                            "fetched_at": datetime.now(timezone.utc)
                                .strftime("%Y-%m-%d %H:%M:%S UTC"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"nsepython error: {exc}")
    with cols[2]:
        q = st.session_state.get("__nse_quote")
        if q:
            ltp = q.get("ltp")
            pct = q.get("pct")
            color = "#4ade80" if (pct or 0) >= 0 else "#f87171"
            ltp_s = f"₹{ltp:,.2f}" if ltp else "₹—"
            pct_s = f"({pct:+.2f}%)" if pct is not None else ""
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:13px;'>"
                f"<div><b style='font-size:16px;'>{_esc(q['symbol'])}</b> "
                f"<span style='color:{color};font-weight:700;'>{ltp_s}</span> "
                f"<span style='color:{color};'>{pct_s}</span></div>"
                f"<div style='color:rgba(148,163,184,0.85);font-size:11px;margin-top:4px;'>"
                f"O {q.get('open','—')} · H {q.get('high','—')} · "
                f"L {q.get('low','—')} · prev {q.get('prev_close','—')}"
                f"</div>"
                f"<div style='color:rgba(148,163,184,0.55);font-size:10px;margin-top:2px;'>"
                f"fetched {_esc(str(q.get('fetched_at','')))}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # --------- Environment / version info ---------
    st.markdown("### Environment")
    try:
        import sys as _sys
        env_info = {
            "python": _sys.version.split()[0],
        }
        for pkg in ("xgboost", "lightgbm", "sklearn", "torch", "neuralforecast", "streamlit"):
            try:
                mod = __import__(pkg)
                env_info[pkg] = getattr(mod, "__version__", "?")
            except Exception:
                env_info[pkg] = "(not installed)"
        # GPU check — torch may be missing on CPU-only deploys; that's
        # expected, just don't spam the env panel with the absence.
        try:
            import torch as _t
            env_info["cuda_available"] = _t.cuda.is_available()
            if _t.cuda.is_available():
                env_info["cuda_device"] = _t.cuda.get_device_name(0)
        except ImportError:
            env_info["cuda_available"] = "(torch not installed)"
        except Exception as exc:  # noqa: BLE001
            env_info["cuda_available"] = f"(error: {exc})"
        st.json(env_info)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"env info unavailable: {exc}")


def _render_predict_tab() -> None:
    """On-demand single-ticker prediction tab.

    Free-form ticker entry with universe-aware suggestions. One click runs
    the cached prediction (~5-20s first hit, instant on cache); a second
    button triggers `deep_analysis` which fits a per-ticker walk-forward
    XGBoost model on the live bundle.
    """
    # Lazy imports — avoid pulling pandas/xgboost into the sidebar render.
    # `predict_via_api` is the public entry point; it routes through the
    # FastAPI service when available and falls back to in-process otherwise.
    from omni.ondemand.predict import deep_analysis, predict_ticker  # noqa: F401  (used by fallback)
    from omni.api.client import predict_via_api

    # Phase 24.8 — concise primary instruction with a tooltip for details.
    # Replaces the long "Type any ticker — searches NSE, NASDAQ, NYSE, ..."
    # banner with a tight headline + ⓘ help icon (delivered via the
    # text_input's `help=` arg below).
    st.markdown(
        "<div style='display:flex;align-items:baseline;gap:10px;"
        "margin-bottom:6px;font-family:Inter,sans-serif;'>"
        "<span style='font-size:13px;font-weight:600;letter-spacing:0.04em;"
        "text-transform:uppercase;color:#cbd5e1;'>Search</span>"
        "<span style='font-size:12px;color:rgba(203,213,225,0.65);'>"
        "any ticker, name, or alias — resolves NSE / NASDAQ / NYSE / crypto / FX"
        "</span></div>",
        unsafe_allow_html=True,
    )

    # ---- Search history (Google-style, persistent across restarts) ----
    from omni.dashboard import searchhistory as _sh
    recent_persist = _sh.list_recent(limit=8)
    frequent = _sh.list_frequent(limit=5)
    sh_stats = _sh.stats()

    # Show only if there is anything to show
    if recent_persist or frequent:
        with st.expander(
            f"🕒 Search history  ·  "
            f"{sh_stats.get('total_searches', 0)} searches  ·  "
            f"{sh_stats.get('unique_tickers', 0)} tickers"
            + (f"  ·  most: {sh_stats.get('most_searched')}"
                f" ({sh_stats.get('most_searched_count')}×)"
                if sh_stats.get("most_searched") else ""),
            expanded=False,
        ):
            tab_recent, tab_freq = st.tabs(["Recent", "Most searched"])
            with tab_recent:
                _render_history_pills(recent_persist, prefix="recent")
            with tab_freq:
                _render_history_pills(frequent, prefix="freq", show_count=True)
            cc1, cc2 = st.columns([1, 1])
            if cc1.button("Clear history", key="sh_clear", use_container_width=True,
                            help="Permanently drop all stored searches"):
                n = _sh.clear_all()
                st.toast(f"Cleared {n} entries from history.", icon="🗑️")
                st.rerun()
            cc2.caption(f"Stored at `data/watchlist/search_history.json`")

    # Initial suggestion list (popular names) — refreshed once query typed
    initial = _suggest_options("")

    col_q, col_run, col_deep, col_refresh = st.columns([5, 2, 2, 2])
    # Wrap each button in a class that lifts it to align with the selectbox
    # input baseline — replaces the brittle `<div height:1.85rem>` spacers.
    _btn_align = "<div class='predict-action-btn'></div>"
    with col_q:
        try:
            choice = st.selectbox(
                "Ticker",
                options=initial,
                index=None,
                placeholder="e.g. AAPL, RELIANCE.NS, BTC-USD",
                accept_new_options=True,
                key="predict_ticker_select",
            )
        except TypeError:
            # Older Streamlit without accept_new_options
            choice = st.text_input(
                "Ticker",
                placeholder="e.g. AAPL, RELIANCE.NS, BTC-USD",
                key="predict_ticker_text",
            )
    with col_run:
        st.markdown(_btn_align, unsafe_allow_html=True)
        run_clicked = st.button("Run prediction", use_container_width=True, type="primary")
    with col_deep:
        st.markdown(_btn_align, unsafe_allow_html=True)
        deep_clicked = st.button("Deep analysis (~15s)", use_container_width=True)
    with col_refresh:
        st.markdown(_btn_align, unsafe_allow_html=True)
        # Phase 24.8 — was showing "🔄 Force refresh" — the emoji rendered as
        # an ambiguous checkbox-looking glyph on some Windows fonts. Replaced
        # with a clean text label; intent is clear, no glyph confusion.
        refresh_clicked = st.button(
            "Force refresh", use_container_width=True,
            help="Drop cached prediction + prices + news for this ticker "
                  "and re-fetch live. Use when you want intraday-fresh data.",
        )

    raw = _parse_label(choice) if choice else ""
    if not raw:
        # Phase 24.10 — was a giant st.info "Enter a ticker to begin." box
        # that violated padding < radius rules and added unjustified visual
        # weight to a 5-word helper. The text_input placeholder already says
        # "e.g. AAPL, RELIANCE.NS, BTC-USD" — that's enough. Just show the
        # universal-model status if it's not ready.
        avail = model_registry.is_available()
        if not avail:
            st.caption(
                "Universal model not yet trained — predictions will use the "
                "rubric blend until `python scripts/train_universal_model.py` is run."
            )
        return

    res: Resolution = resolve(raw)
    if res is None:
        st.warning(f"Could not resolve `{raw}` to a known ticker.")
        return
    confidence_label = "match: curated" if res.confident else "match: parsed"
    pill_row = (
        f"<div style='display:flex;gap:8px;margin:4px 0 12px 0;flex-wrap:wrap;'>"
        f"{pill(_esc(str(res.ticker)), kind='long')}"
        f"{pill(_esc(str(res.asset_class)), kind='flat')}"
        f"{pill(_esc(str(res.display_name)), kind='flat')}"
        f"{pill(_esc(confidence_label), kind='flat')}"
        f"</div>"
    )
    st.markdown(pill_row, unsafe_allow_html=True)

    # ★ Favorite toggle — saves to data/watchlist/favorites.json
    from omni.dashboard import watchlist as _wl
    fav_col1, fav_col2 = st.columns([1, 6])
    is_fav = _wl.is_favorite(res.ticker)
    fav_label = "★ Remove from watchlist" if is_fav else "☆ Add to watchlist"
    if fav_col1.button(fav_label, key=f"fav_btn_{res.ticker}",
                         use_container_width=True):
        if is_fav:
            _wl.remove_favorite(res.ticker)
            fav_col2.success(f"Removed **{res.ticker}** from watchlist.")
        else:
            _wl.add_favorite(res.ticker, res.asset_class, res.display_name)
            fav_col2.success(f"Added **{res.ticker}** to watchlist.")
    st.divider()

    # Action dispatch — deep takes priority if both clicked
    card = None
    # Force-refresh: drop every cache entry referencing this ticker so the
    # next predict_ticker call rebuilds prices, news, fundamentals and the
    # prediction card from live APIs. The 30-min default TTL still applies
    # afterwards.
    if refresh_clicked:
        from omni.ondemand.cache import purge_ticker
        n = purge_ticker(res.ticker)
        st.session_state["predict_last_card"] = None
        st.toast(f"Cleared {n} cache entries for {res.ticker} - rebuilding...",
                  icon="🔄")
        # Treat refresh as an implicit Run
        run_clicked = True

    # ------------------------------------------------------------------
    # Production loading UX (Phase 24.16)
    # While predict_ticker / deep_analysis blocks (5-15s on cold cache),
    # show a slim top progress bar + a skeleton card so the page never
    # looks frozen. Both are CSS-animated; we just reserve `st.empty()`
    # slots and clear them once the call returns.
    # ------------------------------------------------------------------
    def _show_loading(label: str):
        """Render top progress bar + skeleton placeholder card.
        Returns (progress_slot, skeleton_slot) — caller .empty()s them
        when the real card is ready.
        """
        progress_slot = st.empty()
        skeleton_slot = st.empty()
        progress_slot.markdown(
            "<div class='top-progress' role='progressbar' "
            f"aria-label='{label}'></div>",
            unsafe_allow_html=True,
        )
        skeleton_slot.markdown(
            "<div class='card' style='padding:20px 22px'>"
            "<div style='display:flex;justify-content:space-between;"
            "align-items:flex-start;gap:16px;margin-bottom:14px'>"
            "<div style='flex:1'>"
            "<div class='skel skel-line-lg'></div>"
            "<div class='skel skel-line-sm'></div>"
            "</div>"
            "<div style='display:flex;gap:6px;'>"
            "<span class='skel skel-pill'></span>"
            "<span class='skel skel-pill'></span>"
            "<span class='skel skel-pill'></span>"
            "</div></div>"
            "<hr class='hairline'/>"
            "<div style='display:grid;grid-template-columns:repeat(4, 1fr);"
            "gap:14px;margin:10px 0'>"
            "<div class='skel skel-tile'></div>"
            "<div class='skel skel-tile'></div>"
            "<div class='skel skel-tile'></div>"
            "<div class='skel skel-tile'></div>"
            "</div>"
            "<hr class='hairline'/>"
            "<div class='skel skel-line'></div>"
            "<div class='skel skel-line'></div>"
            "<div class='skel skel-line-sm'></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return progress_slot, skeleton_slot

    if deep_clicked:
        progress_slot, skeleton_slot = _show_loading(f"Deep analysis · {res.ticker}")
        try:
            # Routes through the API client; falls back to in-process
            # automatically when the API isn't reachable.
            card_dict = predict_via_api(res.ticker, res.asset_class,
                                         display_name=res.display_name, deep=True)
        except Exception as exc:  # noqa: BLE001
            progress_slot.empty(); skeleton_slot.empty()
            st.error(f"Deep analysis failed: {exc}")
            return
        progress_slot.empty(); skeleton_slot.empty()
        st.session_state["predict_last_card"] = card_dict
        st.session_state[f"_predict_generated_at_{res.ticker}"] = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        _record_and_update_lru(res, card_dict)
    elif run_clicked:
        progress_slot, skeleton_slot = _show_loading(f"Predicting {res.ticker}")
        try:
            card_dict = predict_via_api(res.ticker, res.asset_class,
                                         display_name=res.display_name, deep=False)
        except Exception as exc:  # noqa: BLE001
            progress_slot.empty(); skeleton_slot.empty()
            st.error(f"Prediction failed: {exc}")
            return
        progress_slot.empty(); skeleton_slot.empty()
        if card_dict is None:
            st.error(
                f"Could not generate a prediction for **{res.ticker}** "
                f"({res.asset_class}). This usually means:\n"
                f"- yfinance returned no price history for this ticker\n"
                f"- the ticker is delisted, suspended, or too illiquid\n"
                f"- not enough trading days to build features (need ~60+ rows)\n\n"
                f"Tip: try the BSE form (e.g. `{res.ticker}.BO`) or pick from the dropdown above."
            )
            return
        st.session_state["predict_last_card"] = card_dict
        st.session_state[f"_predict_generated_at_{res.ticker}"] = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        _record_and_update_lru(res, card_dict)

    # Render last card from session_state (survives reruns)
    cd = st.session_state.get("predict_last_card")
    if cd is None:
        st.caption("Click **Run prediction** to fetch live data and score this ticker.")
        return
    if cd.get("ticker") and cd["ticker"].upper() != res.ticker.upper():
        st.caption(
            f"Showing cached card for **{cd['ticker']}**. Click Run to refresh for {res.ticker}."
        )

    # As-of indicator: the prediction card's `asof` is the trading day of the
    # last data point; we also show the wall-clock time it was generated so
    # the user knows whether to hit Force refresh.
    asof_str = cd.get("asof", "")
    generated_at = st.session_state.get(f"_predict_generated_at_{res.ticker}", "")
    if asof_str or generated_at:
        bits = []
        if asof_str:
            bits.append(f"data as-of **{str(asof_str)[:10]}**")
        if generated_at:
            bits.append(f"generated **{generated_at}**")
        st.caption(
            "  ·  ".join(bits) +
            "  ·  cache TTL 30 min  ·  click 🔄 Force refresh for live data"
        )

    _u.prediction_card(cd)
    _u.provenance_strip(cd.get("provenance", {}))

    # ---- Kelly position sizing (A5) ---------------------------------------
    try:
        from omni.ondemand.kelly import kelly_from_card
        kr = kelly_from_card(cd)
        k1, k2, k3 = st.columns([1, 1, 2])
        k1.metric("Recommended size", f"{kr.fraction*100:.1f}% of capital",
                  help="Half-Kelly, capped at 10%. 0% means no edge / FLAT.")
        k2.metric("Kelly edge", f"{kr.edge:+.3f}",
                  help="p*b - q. Positive = bet, ≤0 = skip.")
        k3.caption(f"_{kr.side.upper()}_ · {kr.rationale}")
    except Exception as _ke:
        st.caption(f"_Kelly sizing unavailable: {_ke}_")

    # ---- Cutting-edge: TradingView Lightweight-Charts (Part C) ----
    try:
        from omni.dashboard.charts.lwchart import render_lwchart
        from omni.ondemand.lookup import lookup_ticker as _lt
        b = _lt(res.ticker, res.asset_class, display_name=res.display_name)
        if b is not None and b.prices is not None and not b.prices.empty:
            render_lwchart(
                b.prices.tail(750),
                height=380,
                title=f"{res.ticker} · candles + volume (last 3y)",
            )
    except Exception as _e:  # noqa: BLE001
        st.caption(f"chart unavailable: {_e}")


@st.cache_data(ttl=60)
def _lake_freshness() -> dict:
    """Return dict with `last_backfill_iso`, `age_days`, `n_tickers` from the
    lake manifest. Cached 60s so we don't re-read JSON on every rerun.
    Falls back gracefully when the manifest is absent (first-run)."""
    import json
    from datetime import datetime, timezone
    try:
        from omni.lake import paths as _lake_paths
        mf = _lake_paths.lake_root() / "manifest.json"
        if not mf.exists():
            return {"last_backfill_iso": None, "age_days": None, "n_tickers": None}
        with mf.open() as f:
            data = json.load(f)
        bf = data.get("last_backfill") or {}
        iso = bf.get("completed_at")
        if not iso:
            return {"last_backfill_iso": None, "age_days": None, "n_tickers": None}
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        return {
            "last_backfill_iso": iso,
            "age_days": age,
            "n_tickers": bf.get("n_tickers"),
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("lake freshness probe failed: %s", exc)
        return {"last_backfill_iso": None, "age_days": None, "n_tickers": None}


def _render_freshness_banner() -> None:
    """Top-of-app data-freshness chip. Phase 24.18.

    Reads lake manifest + computes (now - last_backfill).days and renders a
    colour-coded chip:
      green  if age < 2d        (lake is fresh)
      amber  if 2 <= age < 5d   (data getting stale)
      red    if age >= 5d       (run scripts/backfill_universe.py)
    """
    info = _lake_freshness()
    age = info.get("age_days")
    if age is None:
        st.warning(
            "Lake manifest not found — run `python scripts/backfill_universe.py` "
            "to populate price/fundamentals/news data."
        )
        return
    iso = info.get("last_backfill_iso", "?")
    ts_short = iso[:16].replace("T", " ") if iso else "?"
    n_t = info.get("n_tickers") or 0
    if age < 2:
        bg = "rgba(34, 197, 94, 0.15)"   # green
        border = "rgba(34, 197, 94, 0.45)"
        icon = "🟢"
        label = "fresh"
    elif age < 5:
        bg = "rgba(234, 179, 8, 0.18)"   # amber
        border = "rgba(234, 179, 8, 0.50)"
        icon = "🟡"
        label = "stale"
    else:
        bg = "rgba(239, 68, 68, 0.18)"   # red
        border = "rgba(239, 68, 68, 0.55)"
        icon = "🔴"
        label = "run backfill"
    st.markdown(
        f"<div style='display:flex;justify-content:flex-end;align-items:center;"
        f"gap:8px;padding:6px 12px;margin:-4px 0 6px 0;'>"
        f"<span style='font-size:0.78rem;color:var(--text-muted);"
        f"letter-spacing:0.04em;'>lake snapshot</span>"
        f"<span style='font-size:0.78rem;background:{bg};border:1px solid {border};"
        f"border-radius:999px;padding:3px 12px;font-family:ui-monospace,monospace;'>"
        f"{icon} {ts_short} UTC · {age:.1f}d ago · {n_t} tickers · {label}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Omni_Stocks · Multi-asset signals",
        layout="wide",
        page_icon=":zap:",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # Phase 24.2 — live market data marquee at the very top.
    # Cached for 5 minutes via st.cache_data inside the module; falls back
    # gracefully to "—" tiles if yfinance is unreachable.
    try:
        from omni.dashboard.marquee import render_marquee
        render_marquee(height_px=36)
    except Exception:  # noqa: BLE001
        _log.warning("marquee render failed", exc_info=True)

    # Phase 24.18 — lake-freshness chip, right-aligned, under the marquee.
    # Tells the user at a glance whether the daily backfill ran today.
    _render_freshness_banner()

    # ---- sidebar
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Inter,sans-serif;'>"
            "<div style='font-size:1.4rem;font-weight:800;background:var(--accent-glow);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>OMNI_STOCKS</div>"
            "<div style='color:var(--text-muted);font-size:0.78rem;letter-spacing:0.18em;"
            "text-transform:uppercase;'>signal console</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Phase 24.10 — primary section navigation. Replaces the 12-tab
        # horizontal strip that used to live above the page content.
        section_options = [
            "Predict",
            "Watchlist",
            "Portfolio",
            "Top movers",
            "Top signals",
            "Watchlist heatmap",
            "Cosmos 3D",
            "Correlations",
            "Calibration",
            "Earnings",
            "History",
            "Dev",
        ]
        current = st.session_state.get("__section", "Predict")
        if current not in section_options:
            current = "Predict"
        st.radio(
            "Section",
            section_options,
            index=section_options.index(current),
            key="__section",
            label_visibility="collapsed",
        )
        st.divider()

        theme_choice = st.selectbox(
            "Theme",
            ["Auto", "Dark", "Light", "Cyberpunk", "Synthwave", "Matrix", "Nord"],
            index=0,
            help=(
                "**Auto** — default cosmic dark · "
                "**Dark** — minimal near-black · "
                "**Light** — clean slate · "
                "**Cyberpunk** — neon magenta + cyan · "
                "**Synthwave** — purple/pink retro sunset · "
                "**Matrix** — green-on-black hacker · "
                "**Nord** — calm muted polar"
            ),
        )
        _theme_override(theme_choice)

        st.markdown("**Filters**")
        if st.button("⟳ Reload artifacts", use_container_width=True):
            _c.load_predictions.clear()
            _c.load_signals.clear()
            _c.load_regime.clear()
            _c.load_backtest_summary.clear()
            st.rerun()

        keys = _c.reload_keys()
        signals = _c.load_signals(keys["signals"])
        regime = _c.load_regime(keys["regime"])
        preds = _c.load_predictions(keys["pred"])
        summary = _c.load_backtest_summary(keys["backtest"])
        if "ticker" in signals.columns:
            tset = {str(t).upper() for t in signals["ticker"].dropna().unique().tolist()}
            synth = {"AAA", "BBB", "CCC", "DDD", "EEE"}
            if tset and tset.issubset(synth):
                st.warning(
                    "Synthetic test artifacts detected (AAA/BBB/CCC/DDD/EEE). "
                    "Run live backfill/pipeline to populate real tickers."
                )

        ac_options = sorted(signals["asset_class"].dropna().unique().tolist()) \
            if "asset_class" in signals.columns else []
        india_focus = st.toggle(
            "🇮🇳 India focus",
            value=False,
            help="Restrict the universe to NSE/BSE only. Smaller universe = sector-specific calibration, less noise from US/crypto regimes.",
        )
        if india_focus:
            ac_default = [a for a in ac_options if a == "india"] or ac_options
        else:
            ac_default = ac_options
        ac_pick = st.multiselect("Asset class", ac_options, default=ac_default)
        min_conv = st.slider("Min conviction", 0, 100, 0, step=5)
        only_emitted = st.toggle("Only emitted (passes gate)", value=False)

        st.divider()
        last_run = (regime["date"].max()
                    if regime is not None and not regime.empty and "date" in regime.columns
                    else None)
        # Phase 24.5 — staleness indicator (warn if the regime model is >7d old)
        if last_run is not None:
            try:
                age_days = (datetime.now(timezone.utc).date() - last_run.date()).days
            except Exception:
                age_days = 0
            if age_days > 30:
                stale_chip = (f"<span style='color:#f87171;font-weight:700;"
                              f"background:rgba(248,113,113,0.12);padding:2px 8px;"
                              f"border-radius:6px;font-size:10px;letter-spacing:0.05em;"
                              f"text-transform:uppercase;margin-left:6px;'>"
                              f"stale · {age_days}d</span>")
            elif age_days > 7:
                stale_chip = (f"<span style='color:#fbbf24;font-weight:700;"
                              f"background:rgba(251,191,36,0.12);padding:2px 8px;"
                              f"border-radius:6px;font-size:10px;letter-spacing:0.05em;"
                              f"text-transform:uppercase;margin-left:6px;'>"
                              f"{age_days}d old</span>")
            else:
                stale_chip = (f"<span style='color:#4ade80;font-weight:700;"
                              f"background:rgba(74,222,128,0.12);padding:2px 8px;"
                              f"border-radius:6px;font-size:10px;letter-spacing:0.05em;"
                              f"text-transform:uppercase;margin-left:6px;'>"
                              f"fresh</span>")
            date_str = last_run.date().isoformat()
        else:
            stale_chip = ""
            date_str = "n/a"
        st.markdown(
            f"<div style='font-size:11px;color:rgba(148,163,184,0.7);"
            f"letter-spacing:0.04em;'>Last regime point</div>"
            f"<div style='font-size:13px;font-weight:600;color:#e2e8f0;"
            f"font-family:JetBrains Mono,monospace;margin-top:2px;'>"
            f"{date_str}{stale_chip}</div>"
            f"<div style='font-size:10px;color:rgba(148,163,184,0.5);"
            f"margin-top:8px;letter-spacing:0.04em;'>"
            f"Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>",
            unsafe_allow_html=True,
        )

    # ---- in-memory filter
    f = signals.copy()
    if ac_pick and "asset_class" in f.columns:
        f = f[f["asset_class"].isin(ac_pick)]
    if "conviction" in f.columns:
        f = f[f["conviction"] >= min_conv]
    if only_emitted and "passes_gate" in f.columns:
        f = f[f["passes_gate"]]

    # ---- hero
    n_emitted = int(f["passes_gate"].sum()) if "passes_gate" in f.columns else 0
    n_total = len(f)
    hero(
        title="Multi-asset signal console",
        subtitle=(
            f"<b>{n_total}</b> tracked instruments · "
            f"<b>{n_emitted}</b> high-conviction emissions today · "
            f"hit rate <b>{summary.get('hit_rate', 0):.1%}</b>"
        ),
        kicker="Omni_Stocks · daily run",
    )

    # ---- 1. regime banner
    _u.regime_banner(regime)

    # ---- 2. KPI row
    _u.kpi_row(summary)

    # ---- 3. section nav (was a 12-tab horizontal strip; moved to sidebar
    #         in Phase 24.10 — too much horizontal real estate eaten by tabs,
    #         and tab labels were cramped on smaller windows. Sidebar radio
    #         is cleaner and lets the main canvas breathe.)
    section = st.session_state.get("__section", "Predict")

    if section == "Predict":
        _render_predict_tab()

    elif section == "Watchlist":
        _render_watchlist_tab()

    elif section == "Portfolio":
        _render_portfolio_tab()

    elif section == "Top movers":
        _render_movers_tab()

    elif section == "Watchlist heatmap":
        st.caption("Color = direction × conviction. Cyan = high-conviction long, red = high-conviction short.")
        if f.empty:
            st.info("No tickers match the current filters. Loosen filters in the sidebar to see the heatmap.")
        else:
            _u.universe_heatmap(f)

    elif section == "Top signals":
        if f.empty:
            st.info("No signals match the current filters.")
        else:
            emitted = f[f["passes_gate"]] if "passes_gate" in f.columns else f
            if emitted.empty:
                emitted = f
            emitted = emitted.sort_values("conviction", ascending=False).head(8)
            cols = st.columns(2)
            for i, (_, row) in enumerate(emitted.iterrows()):
                with cols[i % 2]:
                    _u.signal_card(row)

    elif section == "Cosmos 3D":
        st.caption("Each sphere = a ticker. Radius ∝ conviction · colour = direction · cluster = asset class · orbit speed ∝ recent vol. Drag to rotate, hover for details.")
        try:
            from omni.dashboard.charts.cosmos3d import render_cosmos
            cards = []
            if not f.empty:
                for _, r in f.iterrows():
                    cards.append({
                        "ticker": r.get("ticker", "?"),
                        "asset_class": r.get("asset_class", "us"),
                        "direction": int(r.get("direction", 0) or 0),
                        "conviction": float(r.get("conviction", 0.0) or 0.0),
                        "blended_p": float(r.get("blended_p", 0.5) or 0.5),
                        "regime": str(r.get("regime", "chop")),
                        "recent_vol_5d_pct": float(r.get("recent_vol_5d_pct", 5.0) or 5.0),
                        "move_1w_pct": float(r.get("expected_move_pct", 0.0) or 0.0),
                        "rubric_total": float(r.get("rubric_total", 50.0) or 50.0),
                    })
            if not cards:
                st.info("No signals available — run the daily orchestrator to populate the cosmos.")
            else:
                render_cosmos(cards, height=560, title="Signal Cosmos")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"cosmos unavailable: {exc}")

    elif section == "Correlations":
        st.caption("D3 force-directed graph. Edge thickness ∝ |60-day correlation|. Drag nodes to rearrange.")
        try:
            from omni.dashboard.charts.corrgraph import render_corrgraph
            from omni.lake.io import read_prices
            from concurrent.futures import ThreadPoolExecutor
            tickers_pool = list(f["ticker"].unique())[:40] if not f.empty else []
            if not tickers_pool:
                st.info("No tickers in the current filtered universe — adjust filters in the sidebar.")
            else:
                # Phase 24.12 — was a serial loop calling read_prices() up to
                # 40 times per render (~3-6s on cold parquet cache). Each read
                # is independent so we fan out with a small thread pool.
                ac_map = {tk: f[f["ticker"] == tk]["asset_class"].iloc[0]
                          for tk in tickers_pool}

                def _load(tk: str):
                    try:
                        p = read_prices(ac_map[tk], tk)
                        if p is not None and not p.empty:
                            return tk, p.set_index("date")["adj_close"].tail(120)
                    except Exception:
                        return tk, None
                    return tk, None

                cols_panel: dict = {}
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for tk, series in ex.map(_load, tickers_pool):
                        if series is not None:
                            cols_panel[tk] = series

                panel = pd.DataFrame(cols_panel) if cols_panel else pd.DataFrame()
                if panel.empty:
                    st.info("No price history available locally for these tickers — run `omni-orch run-daily` to backfill.")
                else:
                    render_corrgraph(panel, height=460, threshold=0.4, title="60-day Correlation Graph")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"correlation graph unavailable: {exc}")

    elif section == "Calibration":
        if preds is None or preds.empty:
            st.info("No predictions available yet — run `omni-orch run-daily` to populate calibration data.")
        else:
            _u.calibration_panel(preds)

    elif section == "Earnings":
        _render_earnings_tab()

    elif section == "History":
        if signals is None or signals.empty:
            st.info("No signal history yet. The daily orchestrator writes to `data/signals/` — run it to populate this view.")
        else:
            _u.signal_history_table(signals, max_rows=300)
            st.divider()
            _render_replay_section()

    elif section == "Dev":
        _render_dev_tab()

    st.markdown(
        "<div style='text-align:center;color:var(--text-muted);font-size:0.78rem;"
        "margin-top:32px;font-family:Inter,sans-serif;'>"
        "Read-only view · daily orchestrator writes to "
        "<code>data/{predictions,signals,regime,backtest}/</code></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
