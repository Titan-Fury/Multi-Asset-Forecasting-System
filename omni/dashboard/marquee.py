"""Live market data feed for the top-of-page marquee ticker tape.

Pulls a small set of indices/FX/crypto/commodities via yfinance, caches
for 5 minutes (st.cache_data), and renders to HTML the marquee strip
already styled in `theme.py` (.ultra-marquee).

Defensive: every fetch is wrapped — if yfinance fails for any symbol we
fall back to a "—" tile rather than crashing the dashboard.
"""
from __future__ import annotations

import time
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
    except Exception:
        pass
    # fallback: 2d history
    try:
        h = yf.Ticker(symbol).history(period="2d", interval="1d", auto_adjust=False)
        if len(h) >= 2:
            return float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
        if len(h) == 1:
            return float(h["Close"].iloc[-1]), float(h["Open"].iloc[-1])
    except Exception:
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
    """Fetch live quotes for the marquee. Cached 5 min; 13 symbols ≈ 1-2s."""
    out: list[MarqueeItem] = []
    for sym, label, fmt in _TICKERS:
        q = _safe_quote(sym)
        if q is None:
            out.append(MarqueeItem(label=label, value="—", direction=0))
            continue
        last, prev = q
        if fmt == "pct":
            v, d = _fmt_pct(last, prev)
        else:
            v, d = _fmt_abs(last, prev)
        out.append(MarqueeItem(label=label, value=v, direction=d))
    return out


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
        return ""
    # Duplicate the list so the seamless animation loop has zero gap.
    track = "".join(it.html() for it in items) * 2
    # The CSS in theme.py already styles .ultra-marquee. We just need to
    # override the height inline if requested.
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
