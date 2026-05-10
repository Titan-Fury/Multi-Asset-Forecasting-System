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
for _candidate in (_here.parent, *_here.parents):
    if (_candidate / "omni" / "__init__.py").exists():
        _root = str(_candidate)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        break
del _sys, _Path, _here, _candidate

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from omni.dashboard import cache as _c
from omni.dashboard import components as _u
from omni.dashboard.theme import inject_theme, hero, pill
from omni.ondemand import model_registry
from omni.ondemand.universe import Resolution, resolve, suggestions


def _theme_override(choice: str) -> None:
    """Inject a small CSS override to force light or dark when user picks
    a manual choice. The base CSS already handles 'auto'."""
    if choice == "Dark":
        st.markdown(
            "<style>:root{color-scheme:dark} html{color-scheme:dark}</style>"
            "<style>:root{"
            "--bg-grad: radial-gradient(circle at 12% 0%, #082f49 0%, transparent 45%),"
            "           radial-gradient(circle at 90% 100%, #2e1065 0%, transparent 50%),"
            "           linear-gradient(180deg, #020617 0%, #0b1224 100%);"
            "--glass-bg: rgba(15,23,42,0.55);"
            "--glass-bg-strong: rgba(15,23,42,0.78);"
            "--glass-border: rgba(148,163,184,0.18);"
            "--glass-shadow: 0 8px 32px rgba(0,0,0,0.45);"
            "--text-primary:#e2e8f0;--text-secondary:#94a3b8;--text-muted:#64748b;"
            "--kpi-positive:#34d399;--kpi-negative:#f87171;"
            "--regime-bull:#22c55e;--regime-chop:#eab308;--regime-bear:#ef4444;--regime-crisis:#a855f7;"
            "}</style>",
            unsafe_allow_html=True,
        )
    elif choice == "Light":
        st.markdown(
            "<style>:root{color-scheme:light} html{color-scheme:light}</style>"
            "<style>:root{"
            "--bg-grad: radial-gradient(circle at 15% 0%, #e0f2fe 0%, transparent 45%),"
            "           radial-gradient(circle at 85% 90%, #ede9fe 0%, transparent 45%),"
            "           linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);"
            "--glass-bg: rgba(255,255,255,0.55);"
            "--glass-bg-strong: rgba(255,255,255,0.72);"
            "--glass-border: rgba(255,255,255,0.7);"
            "--glass-shadow: 0 8px 32px rgba(15,23,42,0.08);"
            "--text-primary:#0f172a;--text-secondary:#475569;--text-muted:#64748b;"
            "--kpi-positive:#059669;--kpi-negative:#dc2626;"
            "--regime-bull:#16a34a;--regime-chop:#ca8a04;--regime-bear:#dc2626;--regime-crisis:#7c3aed;"
            "}</style>",
            unsafe_allow_html=True,
        )


def _suggest_options(query: str) -> list[str]:
    """Return labels for the autocomplete: 'TICKER · EXCH · Display name'.

    Uses multi-source live search (Yahoo + NSE + CoinGecko) when query is
    long enough; falls back to curated list for empty / very-short queries.
    """
    sugs = suggestions(query or "", k=15)
    out = []
    for s in sugs:
        # Exchange tag: try to derive from asset_class
        exch_tag = {
            "india": "NSE", "us": "US", "crypto": "CRYPTO",
            "fx": "FX", "etf": "ETF", "index": "IDX",
        }.get(s.asset_class, s.asset_class.upper())
        # Trim long names
        name = s.display_name[:55] + "…" if len(s.display_name) > 55 else s.display_name
        out.append(f"{s.ticker:<14}  ·  {exch_tag:<6}  ·  {name}")
    return out


def _parse_label(label: str) -> str:
    """Extract the ticker from a 'TICKER · EXCH · name' label, or return as-is."""
    if "·" in label:
        return label.split("·", 1)[0].strip()
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
            if st.button(label, key=f"sh_{prefix}_{i}", width='stretch',
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
                    f"{tk:<14}  ·  {exch:<6}  ·  {e.get('display_name', tk)}"
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
    if cols[1].button("Refresh all", width='stretch', key="wl_refresh"):
        st.session_state["_wl_refresh_nonce"] = st.session_state.get("_wl_refresh_nonce", 0) + 1
    if cols[2].button("Clear all", width='stretch', key="wl_clear"):
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
    st.dataframe(df, width='stretch', hide_index=True)

    # Per-ticker remove buttons
    st.markdown("##### Manage")
    rcols = st.columns(min(4, max(1, len(favs))))
    for i, fav in enumerate(favs):
        with rcols[i % len(rcols)]:
            if st.button(f"× {fav['ticker']}", key=f"wl_rm_{fav['ticker']}",
                          width='stretch'):
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
        st.dataframe(g_show, width='stretch', hide_index=True)
    with cl:
        st.markdown(f"### Top {len(losers)} losers")
        st.dataframe(l_show, width='stretch', hide_index=True)


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


def _record_and_update_lru(res: Resolution, card) -> None:
    """Persist the search to history.json + update the in-session LRU pills.
    Idempotent — safe to call after every successful prediction.
    Also fires a Streamlit toast (UI7) for live feedback.
    """
    from omni.dashboard import searchhistory as _sh

    if card is not None:
        d = card.to_dict()
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
            pass
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
            cd = card.to_dict()
        except Exception as e:  # noqa: BLE001
            return {"_err": f"{tk}: {e}"}
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
                r = fut.result()
                if r is None:
                    pass
                elif "_err" in r:
                    st.warning(r["_err"])
                else:
                    rows.append(r)
                completed += 1
                progress.progress(completed / n)
    except Exception:
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
                      width='stretch', hide_index=True)
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
        width='stretch', hide_index=True,
    )

    with st.expander("Below conviction gate (not in book)"):
        rest = pf[~pf["ticker"].isin(in_book["ticker"])].sort_values(
            "conviction", ascending=False)
        if rest.empty:
            st.caption("none")
        else:
            st.dataframe(rest, width='stretch', hide_index=True)


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
        width='stretch', hide_index=True,
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
                        tk, ac, ed = fut.result()
                        ed_map[(tk, ac)] = ed
            except Exception:
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
                      width='stretch', hide_index=True)
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
            st.dataframe(hist.head(8), width='stretch', hide_index=True)
            if "surprise_pct" in hist.columns:
                avg = hist["surprise_pct"].dropna().mean()
                st.metric("Avg surprise (last 4 quarters)",
                          f"{avg:+.2f} %" if pd.notna(avg) else "—")


def _render_predict_tab() -> None:
    """On-demand single-ticker prediction tab.

    Free-form ticker entry with universe-aware suggestions. One click runs
    the cached prediction (~5-20s first hit, instant on cache); a second
    button triggers `deep_analysis` which fits a per-ticker walk-forward
    XGBoost model on the live bundle.
    """
    # Lazy imports — avoid pulling pandas/xgboost into the sidebar render
    from omni.ondemand.predict import deep_analysis, predict_ticker

    st.markdown(
        "<div style='font-family:Inter,sans-serif;color:var(--text-secondary);"
        "font-size:0.92rem;margin-bottom:8px;'>"
        "Type any ticker — searches <b>NSE</b>, <b>NASDAQ</b>, <b>NYSE</b>, "
        "<b>crypto</b>, <b>FX</b> live across Yahoo Finance + NSE + CoinGecko. "
        "Free-form entries are resolved."
        "</div>",
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
            if cc1.button("Clear history", key="sh_clear", width='stretch',
                            help="Permanently drop all stored searches"):
                n = _sh.clear_all()
                st.toast(f"Cleared {n} entries from history.", icon="🗑️")
                st.rerun()
            cc2.caption(f"Stored at `data/watchlist/search_history.json`")

    # Initial suggestion list (popular names) — refreshed once query typed
    initial = _suggest_options("")

    col_q, col_run, col_deep, col_refresh = st.columns([5, 2, 2, 2])
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
        st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Run prediction", width='stretch', type="primary")
    with col_deep:
        st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
        deep_clicked = st.button("Deep analysis (~15s)", width='stretch')
    with col_refresh:
        st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button(
            "🔄 Force refresh", width='stretch',
            help="Drop the cached prediction + prices + news for this "
                  "ticker and re-fetch live. Use when you want intraday-fresh "
                  "data (default cache is 30 min for predictions, 30 min for "
                  "news, 4 h for prices).",
        )

    raw = _parse_label(choice) if choice else ""
    if not raw:
        st.info("Enter a ticker to begin.")
        # Show universal model status as a small pill row
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
        f"{pill(res.ticker, kind='long')}"
        f"{pill(res.asset_class, kind='flat')}"
        f"{pill(res.display_name, kind='flat')}"
        f"{pill(confidence_label, kind='flat')}"
        f"</div>"
    )
    st.markdown(pill_row, unsafe_allow_html=True)

    # ★ Favorite toggle — saves to data/watchlist/favorites.json
    from omni.dashboard import watchlist as _wl
    fav_col1, fav_col2 = st.columns([1, 6])
    is_fav = _wl.is_favorite(res.ticker)
    fav_label = "★ Remove from watchlist" if is_fav else "☆ Add to watchlist"
    if fav_col1.button(fav_label, key=f"fav_btn_{res.ticker}",
                         width='stretch'):
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

    if deep_clicked:
        with st.spinner(f"Fitting walk-forward model for {res.ticker}…"):
            try:
                card = deep_analysis(res.ticker, res.asset_class, display_name=res.display_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Deep analysis failed: {exc}")
                return
        st.session_state["predict_last_card"] = card.to_dict() if card else None
        st.session_state[f"_predict_generated_at_{res.ticker}"] = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        _record_and_update_lru(res, card)
    elif run_clicked:
        with st.spinner(f"Fetching live data for {res.ticker}…"):
            try:
                card = predict_ticker(res.ticker, res.asset_class, display_name=res.display_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                return
        if card is None:
            st.error(
                f"Could not generate a prediction for **{res.ticker}** "
                f"({res.asset_class}). This usually means:\n"
                f"- yfinance returned no price history for this ticker\n"
                f"- the ticker is delisted, suspended, or too illiquid\n"
                f"- not enough trading days to build features (need ~60+ rows)\n\n"
                f"Tip: try the BSE form (e.g. `{res.ticker}.BO`) or pick from the dropdown above."
            )
            return
        st.session_state["predict_last_card"] = card.to_dict() if card else None
        st.session_state[f"_predict_generated_at_{res.ticker}"] = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        _record_and_update_lru(res, card)

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
        pass

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

        theme_choice = st.radio(
            "Theme", ["Auto", "Light", "Dark"], horizontal=True,
            help="Auto follows your OS preference.",
        )
        _theme_override(theme_choice)

        st.markdown("**Filters**")
        if st.button("⟳ Reload artifacts", width='stretch'):
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
                from datetime import datetime as _dt
                age_days = (_dt.now(timezone.utc).date() - last_run.date()).days
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

    # ---- 3. tabs
    (tab_predict, tab_watch, tab_portfolio, tab_movers, tab_heat,
     tab_signals, tab_cosmos, tab_corr, tab_calib, tab_earnings,
     tab_history) = st.tabs([
        "Predict", "★ Watchlist", "Portfolio", "Top movers",
        "Watchlist heatmap", "Top signals", "Cosmos 3D", "Correlations",
        "Calibration", "Earnings", "History",
    ])

    with tab_predict:
        _render_predict_tab()

    with tab_watch:
        _render_watchlist_tab()

    with tab_portfolio:
        _render_portfolio_tab()

    with tab_movers:
        _render_movers_tab()

    with tab_heat:
        st.caption("Color = direction × conviction. Cyan = high-conviction long, red = high-conviction short.")
        _u.universe_heatmap(f)

    with tab_signals:
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

    with tab_cosmos:
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
            render_cosmos(cards, height=560, title="Signal Cosmos")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"cosmos unavailable: {exc}")

    with tab_corr:
        st.caption("D3 force-directed graph. Edge thickness ∝ |60-day correlation|. Drag nodes to rearrange.")
        try:
            from omni.dashboard.charts.corrgraph import render_corrgraph
            from omni.lake.io import read_prices
            from omni.lake.paths import lake_root
            import pandas as pd
            tickers_pool = list(f["ticker"].unique())[:40] if not f.empty else []
            cols_panel = {}
            for tk in tickers_pool:
                try:
                    ac = f[f["ticker"] == tk]["asset_class"].iloc[0]
                    p = read_prices(ac, tk)
                    if p is not None and not p.empty:
                        s = p.set_index("date")["adj_close"].tail(120)
                        cols_panel[tk] = s
                except Exception:
                    continue
            panel = pd.DataFrame(cols_panel) if cols_panel else pd.DataFrame()
            render_corrgraph(panel, height=460, threshold=0.4, title="60-day Correlation Graph")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"correlation graph unavailable: {exc}")

    with tab_calib:
        _u.calibration_panel(preds)

    with tab_earnings:
        _render_earnings_tab()

    with tab_history:
        _u.signal_history_table(signals, max_rows=300)
        st.divider()
        _render_replay_section()

    st.markdown(
        "<div style='text-align:center;color:var(--text-muted);font-size:0.78rem;"
        "margin-top:32px;font-family:Inter,sans-serif;'>"
        "Read-only view · daily orchestrator writes to "
        "<code>data/{predictions,signals,regime,backtest}/</code></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
