"""NSE & BSE institutional-flow data.

Three free, public, no-auth feeds:

  1. FII/DII activity        — daily net buy/sell by Foreign Institutional
                                 Investors and Domestic Institutional Investors.
                                 Source: NSE & BSE provisional CSVs.
  2. Bulk deals              — single trades >= 0.5% of equity capital.
  3. Block deals             — large negotiated trades on the block window.

Used as MARKET-LEVEL features (not per-ticker price features). They show:
  - Where institutional money is flowing (cash market vs futures)
  - Insider/HNI conviction (bulk / block deals)

Cached at the news TTL (30 min). Stateless and best-effort — empty
DataFrame on any error. Format aligned to the lake schema for joining.
"""
from __future__ import annotations

import pandas as pd
import requests

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS_HTML = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"}
_HEADERS_API = {
    "User-Agent": _UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
_TIMEOUT = 6.0


def _nse_session() -> requests.Session:
    """NSE blocks bare GETs — must visit a page first to set cookies."""
    s = requests.Session()
    s.headers.update(_HEADERS_HTML)
    try:
        s.get("https://www.nseindia.com/", timeout=_TIMEOUT)
        s.get("https://www.nseindia.com/market-data/bulk-block-deals", timeout=_TIMEOUT)
    except Exception as e:
        log.debug(f"nse warmup failed: {e}")
    return s


@cached(ttl=TTL.news, key="nse_fii_dii:v1")
def fii_dii_activity() -> pd.DataFrame:
    """Daily FII + DII cash market net buy/sell (last ~30 sessions).

    Schema: [date, fii_net_cr, dii_net_cr, fii_buy_cr, fii_sell_cr,
             dii_buy_cr, dii_sell_cr]. All amounts in INR crores.
    """
    s = _nse_session()
    s.headers.update(_HEADERS_API)
    try:
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=_TIMEOUT)
        if r.status_code != 200:
            log.debug(f"NSE fii_dii: status={r.status_code}")
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        rows = []
        for grp in data:
            cat = str(grp.get("category", "")).strip().upper().rstrip("*").strip()
            d = grp.get("date", "")
            if cat in ("FII/FPI", "FII"):
                rows.append({
                    "date": d,
                    "fii_buy_cr": float(grp.get("buyValue", 0) or 0),
                    "fii_sell_cr": float(grp.get("sellValue", 0) or 0),
                    "fii_net_cr": float(grp.get("netValue", 0) or 0),
                })
            elif cat == "DII":
                rows.append({
                    "date": d,
                    "dii_buy_cr": float(grp.get("buyValue", 0) or 0),
                    "dii_sell_cr": float(grp.get("sellValue", 0) or 0),
                    "dii_net_cr": float(grp.get("netValue", 0) or 0),
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.groupby("date", as_index=False).first()
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
        return df.reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        log.warning(f"nse fii_dii fetch failed: {e}")
        return pd.DataFrame()


@cached(ttl=TTL.news, key="nse_largedeal_snapshot:v1")
def _largedeal_snapshot() -> dict:
    """Return raw NSE large-deal snapshot. Contains BULK_DEALS_DATA,
    BLOCK_DEALS_DATA, SHORT_DEALS_DATA arrays.
    """
    s = _nse_session()
    s.headers.update(_HEADERS_API)
    try:
        r = s.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal",
                   timeout=_TIMEOUT * 2)
        if r.status_code != 200:
            log.debug(f"NSE largedeal: status={r.status_code}")
            return {}
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.warning(f"nse largedeal fetch failed: {e}")
        return {}


def _normalise_deals(arr: list[dict]) -> pd.DataFrame:
    if not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr)
    rename = {
        "date": "date", "symbol": "ticker", "name": "company",
        "clientName": "client_name", "buySell": "side",
        "qty": "qty", "watp": "avg_price", "remarks": "remarks",
    }
    df = df.rename(columns=rename)
    keep = [c for c in ("date", "ticker", "company", "client_name", "side", "qty", "avg_price", "remarks")
             if c in df.columns]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%d-%b-%Y")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
    for c in ("qty", "avg_price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper().str.strip()
    return df.reset_index(drop=True)


def bulk_deals(days: int = 30) -> pd.DataFrame:
    """Bulk deals from NSE's daily snapshot. `days` arg kept for API compatibility
    — the snapshot endpoint returns the most recent trading day only.
    """
    snap = _largedeal_snapshot()
    return _normalise_deals(snap.get("BULK_DEALS_DATA", []) if isinstance(snap, dict) else [])


def block_deals(days: int = 30) -> pd.DataFrame:
    """Block deals from NSE's daily snapshot."""
    snap = _largedeal_snapshot()
    return _normalise_deals(snap.get("BLOCK_DEALS_DATA", []) if isinstance(snap, dict) else [])


def short_deals(days: int = 30) -> pd.DataFrame:
    """Short-deal disclosures from NSE's daily snapshot."""
    snap = _largedeal_snapshot()
    return _normalise_deals(snap.get("SHORT_DEALS_DATA", []) if isinstance(snap, dict) else [])


def insider_intensity(ticker: str, *, days: int = 30) -> dict:
    """Lightweight aggregate per-ticker: how many bulk/block deals + net side bias.

    Returns {"bulk_n", "block_n", "buy_minus_sell", "net_qty", "last_deal_date"}.
    Empty values when data missing.
    """
    out = {"bulk_n": 0, "block_n": 0, "buy_minus_sell": 0,
            "net_qty": 0.0, "last_deal_date": None}
    try:
        bd = bulk_deals(days=days)
        bk = block_deals(days=days)
        for df, key in ((bd, "bulk_n"), (bk, "block_n")):
            if df is None or df.empty or "ticker" not in df.columns:
                continue
            sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
            out[key] = int(len(sub))
            if not sub.empty:
                if "side" in sub.columns:
                    sides = sub["side"].astype(str).str.upper().fillna("")
                    buys = int((sides == "BUY").sum())
                    sells = int((sides == "SELL").sum())
                    out["buy_minus_sell"] += buys - sells
                if "qty" in sub.columns and "side" in sub.columns:
                    qsigned = sub.apply(
                        lambda r: float(r.get("qty", 0) or 0)
                                    * (1 if str(r.get("side", "")).upper() == "BUY" else -1),
                        axis=1,
                    )
                    out["net_qty"] += float(qsigned.sum())
                if "date" in sub.columns:
                    last = pd.to_datetime(sub["date"], errors="coerce").max()
                    if pd.notna(last) and (
                        out["last_deal_date"] is None
                        or pd.Timestamp(out["last_deal_date"]) < last
                    ):
                        out["last_deal_date"] = str(pd.Timestamp(last).date())
    except Exception as e:  # noqa: BLE001
        log.debug(f"insider_intensity({ticker}) failed: {e}")
    return out
