"""Live market data feed for the top-of-page marquee ticker tape.

Pulls a small set of indices/FX/crypto/commodities via yfinance, caches
for 5 minutes (st.cache_data), and renders to HTML the marquee strip
already styled in `theme.py` (.ultra-marquee).

Defensive: every fetch is wrapped — if yfinance fails for any symbol we
fall back to a "—" tile rather than crashing the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import streamlit as st

# Symbols ordered by importance for an India-focused multi-asset desk.
# Format: (yfinance_symbol, display_label, fmt, multiplier)
#   fmt: "pct" -> show signed % change, "abs" -> show last price
#   multiplier: post-process the raw value (e.g. ×100 for percent)
_TICKERS: list[tuple[str, str, str]] = [
    ("^NSEI",    "NIFTY 50",   "pct"),
    ("^BSESN",   "SENSEX",     "pct"),
    ("^NSEBANK", "BANKNIFTY",  "pct"),
    ("^GSPC",    "S&P 500",    "pct"),
    ("^IXIC",    "NASDAQ",     "pct"),
    ("^DJI",     "DOW",        "pct"),
    ("^VIX",     "VIX",        "pct"),
    ("BTC-USD",  "BTC",        "pct"),
    ("ETH-USD",  "ETH",        "pct"),
    ("GC=F",     "GOLD",       "pct"),
    ("CL=F",     "BRENT",      "pct"),
    ("INR=X",    "USDINR",     "abs"),
    ("^TNX",     "10Y UST",    "abs"),
]


@dataclass(slots=True)
class MarqueeItem:
    label: str
    value: str        # already-formatted display string
    direction: int    # -1 / 0 / +1

    def html(self) -> str:
        cls = "up" if self.direction > 0 else ("dn" if self.direction < 0 else "")
        return f'<span>{self.label}</span><b class="{cls}">{self.value}</b>'


# Phase 24.12 — last-known-good cache moved to st.session_state. Was a
# module-level mutable list mutated inside @st.cache_data, which created
# subtle cross-session races (multiple Streamlit threads share one process)
# AND was unreachable when the cache hit (the body never ran on hits, so
# the module-level fallback could be stale indefinitely). Session-state
# storage is per-user-session and updates on every cache miss/refresh.
_LAST_GOOD_KEY = "__marquee_last_good"


def _safe_quote(symbol: str) -> tuple[float, float] | None:
    """Return (last, prev_close). None on any failure."""
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        t = yf.Ticker(symbol)
        # fast_info is dramatically cheaper than .info — uses /v8/finance/chart
        info = t.fast_info
        last = float(info["last_price"])
        prev = float(info.get("previous_close") or info.get("regular_market_previous_close") or 0.0)
        if last > 0 and prev > 0:
            return last, prev
    except Exception:  # noqa: BLE001
        # yfinance fast_info is best-effort; fall through to history path.
        # Logging at WARNING here would spam since it fails on weekends /
        # for delisted tickers as a matter of course.
        pass
    # fallback: 2d history
    try:
        h = yf.Ticker(symbol).history(period="2d", interval="1d", auto_adjust=False)
        if len(h) >= 2:
            return float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
        if len(h) == 1:
            return float(h["Close"].iloc[-1]), float(h["Open"].iloc[-1])
    except Exception:  # noqa: BLE001
        # Final fallback path; caller treats None as "no data" and skips
        # the symbol in the marquee. Logging would spam during yfinance
        # outages so we stay silent by design.
        pass
    return None


def _fmt_pct(last: float, prev: float) -> tuple[str, int]:
    if prev == 0:
        return "—", 0
    pct = (last - prev) / prev * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%", (1 if pct > 0 else -1 if pct < 0 else 0)


def _fmt_abs(last: float, prev: float) -> tuple[str, int]:
    direction = 1 if last > prev else (-1 if last < prev else 0)
    # Smart precision: yields like 4.18, FX like 83.42, indices like 22300
    if last < 10:
        s = f"{last:.3f}"
    elif last < 1000:
        s = f"{last:.2f}"
    else:
        s = f"{last:,.0f}"
    return s, direction


@st.cache_data(ttl=300, show_spinner=False)  # refresh every 5 minutes
def fetch_marquee_data() -> list[MarqueeItem]:
    """Fetch live quotes for the marquee. Cached 5 min; batched 1 round-trip.

    Phase 24.20 — was issuing 13 sequential `yf.Ticker(sym).fast_info` calls
    on cache miss (~3-5s wall). Now uses a single `yf.download(group_by=...)`
    request which fetches all symbols in one HTTP round-trip (~0.5-1s) and
    falls back to the per-symbol path only for symbols missing from the
    batch payload (rare, e.g. delisted).
    """
    out: list[MarqueeItem] = []
    failed = 0

    # ---- single batched fetch ----
    batch: dict[str, tuple[float, float]] = {}
    try:
        import yfinance as yf
        symbols = [s for s, _, _ in _TICKERS]
        df = yf.download(
            tickers=" ".join(symbols),
            period="2d",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
            prepost=False,
        )
        # df shape: MultiIndex columns when multiple tickers requested.
        for sym in symbols:
            try:
                if (sym,) in df.columns or sym in getattr(df.columns, "levels", [[]])[0]:
                    sub = df[sym]["Close"].dropna()
                    if len(sub) >= 2:
                        batch[sym] = (float(sub.iloc[-1]), float(sub.iloc[-2]))
                    elif len(sub) == 1:
                        # single bar — use Open as prev proxy
                        op = df[sym]["Open"].dropna()
                        if len(op) >= 1:
                            batch[sym] = (float(sub.iloc[-1]), float(op.iloc[-1]))
            except Exception:  # noqa: BLE001
                # per-symbol decoding failure → fall back to _safe_quote below
                pass
    except Exception:  # noqa: BLE001
        # Total batch failure (network / yfinance change) → per-symbol path.
        pass

    for sym, label, fmt in _TICKERS:
        q = batch.get(sym) or _safe_quote(sym)
        if q is None:
            failed += 1
            out.append(MarqueeItem(label=label, value="—", direction=0))
            continue
        last, prev = q
        if fmt == "pct":
            v, d = _fmt_pct(last, prev)
        else:
            v, d = _fmt_abs(last, prev)
        out.append(MarqueeItem(label=label, value=v, direction=d))

    # If MORE THAN HALF failed, prefer last-known-good if we have it.
    last_good = st.session_state.get(_LAST_GOOD_KEY)
    if failed > len(_TICKERS) // 2:
        if last_good:
            return last_good
        # else: return whatever we got (mostly em-dashes is still visible)
    else:
        st.session_state[_LAST_GOOD_KEY] = out  # remember good fetch
    return out


def _demo_items() -> list[MarqueeItem]:
    """Visible-but-clearly-static fallback if yfinance is completely unavailable."""
    return [
        MarqueeItem("NIFTY 50",   "demo", 0),
        MarqueeItem("SENSEX",     "demo", 0),
        MarqueeItem("BANKNIFTY",  "demo", 0),
        MarqueeItem("S&P 500",    "demo", 0),
        MarqueeItem("NASDAQ",     "demo", 0),
        MarqueeItem("BTC",        "demo", 0),
        MarqueeItem("GOLD",       "demo", 0),
        MarqueeItem("USDINR",     "demo", 0),
    ]


def marquee_html(items: Iterable[MarqueeItem] | None = None,
                  *, height_px: int = 36) -> str:
    """Render the marquee strip. Pass `items=None` to fetch live (cached)."""
    if items is None:
        try:
            items = fetch_marquee_data()
        except Exception:
            items = []
    items = list(items)
    if not items:
        items = _demo_items()       # never go fully blank
    # Duplicate the list so the seamless animation loop has zero gap.
    track = "".join(it.html() for it in items) * 2
    style = f'style="height: {height_px}px;"' if height_px != 28 else ""
    track_style = f'style="height: {height_px}px; padding: {(height_px - 16) // 2}px 0;"' if height_px != 28 else ""
    return (
        f'<div class="ultra-marquee" {style} aria-hidden="true">'
        f'<div class="ultra-marquee__track" {track_style}>{track}</div>'
        f'</div>'
    )


def render_marquee(*, height_px: int = 36) -> None:
    """Streamlit helper: drop the live marquee at the top of the page."""
    st.markdown(marquee_html(height_px=height_px), unsafe_allow_html=True)
