"""SEC EDGAR fetcher (Form 4 insider trades + 8-K material events).

EDGAR is free and requires no API key — only a polite ``User-Agent`` header
identifying the requester per SEC fair-access rules.

Endpoints used:
- ``https://www.sec.gov/files/company_tickers.json`` (ticker -> CIK)
- ``https://data.sec.gov/submissions/CIK{cik:010d}.json`` (recent filings)

Both are cached aggressively. Form 4 = insider buys/sells (often leading);
8-K = material events (M&A, departures, guidance updates) filed within
4 business days of the event — typically hours before the press release.

Phase 25.1 — `insider_score()` now does a real signed XBRL parse of each
Form 4 (transaction code, shares, price-per-share) so we report SIGNED
dollar value of insider activity, not just filing counts. Adds the
Cohen-Malloy-Pomorski (2012) cluster-buyer count as a documented alpha
signal (≥3 distinct insiders buying within 30d = strong positive).
"""
from __future__ import annotations

import os
import re

# defusedxml hardens parsing of untrusted remote XML (Form 4 filings) against
# entity-expansion / XXE attacks. It is a declared dependency, so we require it
# rather than falling back to the vulnerable stdlib parser.
import defusedxml.ElementTree as _ET
import pandas as pd
import requests

from omni.ondemand.cache import cached, get_cache
from omni.utils.logger import log

_UA = os.environ.get("SEC_USER_AGENT", "Omni_Stocks research-noreply@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}
_TIMEOUT = 20


def _get_cik_map() -> dict[str, str]:
    """Cache the full ticker->CIK map for 7 days (SEC updates daily)."""
    cache = get_cache()
    key = "edgar:cik_map_v1"
    cached_val = cache.get(key)
    if cached_val is not None:
        return cached_val
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out: dict[str, str] = {}
        for _, rec in data.items():
            t = str(rec.get("ticker", "")).upper().strip()
            cik = str(rec.get("cik_str", "")).zfill(10)
            if t and cik:
                out[t] = cik
        cache.set(key, out, expire=7 * 86400)
        log.info(f"edgar: cik map cached ({len(out)} tickers)")
        return out
    except Exception as e:
        log.warning(f"edgar cik map failed: {e}")
        return {}


def _ticker_to_cik(ticker: str) -> str | None:
    """Resolve a US ticker (e.g. 'AAPL') to a 10-digit CIK string."""
    return _get_cik_map().get(ticker.upper().replace("-", "."))


@cached(ttl=1800, key="edgar:filings:{ticker}:{days}")
def recent_filings(ticker: str, *, days: int = 180,
                   forms: tuple[str, ...] = ("4", "8-K")) -> pd.DataFrame:
    """Recent EDGAR filings for ``ticker`` within the last ``days`` days.

    Returns a DataFrame with columns:
        ``filing_date``, ``form``, ``primary_doc``, ``description``,
        ``accession``, ``url``, ``ticker``, ``source``, ``title``,
        ``published_at``, ``sent_score``.

    The last four columns let the result be unioned with the news frame
    in ``lookup._news_combined`` without further reshaping.
    """
    cik = _ticker_to_cik(ticker)
    if not cik:
        return pd.DataFrame()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"edgar submissions failed {ticker}: {e}")
        return pd.DataFrame()

    rec = data.get("filings", {}).get("recent", {})
    if not rec or not rec.get("form"):
        return pd.DataFrame()

    df = pd.DataFrame({
        "form": rec.get("form", []),
        "filing_date": rec.get("filingDate", []),
        "accession": rec.get("accessionNumber", []),
        "primary_doc": rec.get("primaryDocument", []),
        "description": rec.get("primaryDocDescription", []),
    })
    if df.empty:
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = df[df["filing_date"] >= cutoff]
    df = df[df["form"].isin(list(forms))]
    if df.empty:
        return df

    df["ticker"] = ticker.upper()
    df["source"] = "sec_edgar"
    df["title"] = df.apply(
        lambda r: (f"[{r['form']}] "
                    f"{r['description'] or 'EDGAR filing'} "
                    f"({pd.Timestamp(r['filing_date']).date()} · {r['accession']})"),
        axis=1)
    df["published_at"] = df["filing_date"]
    # Heuristic sentiment: 8-K is event-tagged elsewhere; Form 4 polarity needs
    # the actual XBRL transaction parse. Default to 0; refined by `insider_score`.
    df["sent_score"] = 0.0
    df["url"] = df.apply(
        lambda r: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                  f"{str(r['accession']).replace('-', '')}/{r['primary_doc']}",
        axis=1,
    )
    df = df.sort_values("filing_date", ascending=False).reset_index(drop=True)
    return df[["filing_date", "form", "primary_doc", "description", "accession",
               "url", "ticker", "source", "title", "published_at", "sent_score"]]


@cached(ttl=3600, key="edgar:insider_score:{ticker}")
def insider_score(ticker: str) -> dict:
    """Signed insider-activity rollup over the last 30/90 days.

    Phase 25.1 — actual XBRL parse. Pulls each Form 4's primary doc,
    extracts the per-transaction code (P=purchase, S=sale, A=grant,
    F=tax-withhold) plus shares + price-per-share, computes signed
    dollar value, and aggregates per ticker.

    Returns:
        {
          form4_30d:   number of Form 4 filings in last 30d,
          form4_90d:   number in last 90d,
          insider_net_dollar_30d:  signed dollar value, last 30d,
          insider_net_dollar_90d:  signed dollar value, last 90d,
          cluster_buyers_30d:  distinct insiders with >$0 net buy in 30d
                                 (Cohen-Malloy-Pomorski 2012 signal),
          last_form4:  ISO date of most recent Form 4,
        }
    Graceful degradation: per-filing parse failures are skipped silently;
    counts still populate even when XBRL parse fails.
    """
    df = recent_filings(ticker, days=90, forms=("4",))
    if df is None or df.empty:
        return {
            "form4_30d": 0, "form4_90d": 0,
            "insider_net_dollar_30d": 0.0, "insider_net_dollar_90d": 0.0,
            "cluster_buyers_30d": 0,
            "last_form4": None,
        }

    today = pd.Timestamp.utcnow().tz_localize(None)
    cutoff_30 = today - pd.Timedelta(days=30)

    d30 = (df["filing_date"] >= cutoff_30).sum()
    d90 = len(df)
    last = df["filing_date"].max()

    net_30 = 0.0
    net_90 = 0.0
    buyers_30: set[str] = set()

    for _, row in df.iterrows():
        url = row.get("url")
        fd = row.get("filing_date")
        if not url:
            continue
        try:
            txns = _fetch_and_parse_form4(url)
        except Exception as exc:  # noqa: BLE001
            log.debug(f"form4 parse failed {url}: {exc}")
            continue
        for t in txns:
            value = float(t.get("signed_value", 0.0) or 0.0)
            if not value:
                continue
            net_90 += value
            if pd.notna(fd) and fd >= cutoff_30:
                net_30 += value
                # Cohen-Malloy-Pomorski cluster: distinct insiders buying.
                if value > 0 and t.get("reporter"):
                    buyers_30.add(str(t["reporter"]).upper().strip())

    return {
        "form4_30d": int(d30),
        "form4_90d": int(d90),
        "insider_net_dollar_30d": float(net_30),
        "insider_net_dollar_90d": float(net_90),
        "cluster_buyers_30d": int(len(buyers_30)),
        "last_form4": last.date().isoformat() if pd.notna(last) else None,
    }


# ----- Form 4 XBRL parser ----------------------------------------------------

# Transaction codes per SEC Form 4 instructions:
#   P = open-market purchase (BUY)
#   S = open-market sale     (SELL)
#   A = grant/award          (often informational, not a market signal)
#   F = tax-withhold sale    (treat as SELL for net but excluded from
#                              cluster signal because it's automatic)
#   M = exercise of derivative (informational; exclude from net)
#   D = disposition by gift  (exclude)
_BUY_CODES = frozenset(("P",))
_SELL_CODES = frozenset(("S", "F"))
_INFORMATIONAL_CODES = frozenset(("A", "M", "D", "G", "X"))


def _strip_ns(tag: str) -> str:
    """Drop the namespace from an ElementTree tag for easier matching."""
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _findtext_localname(elem, name: str) -> str | None:
    """Find a child element by local-name and return its text value.

    SEC Form 4 wraps numeric fields in a `<value>` sub-element:
        <transactionShares>
            <value>1000</value>
            <footnoteId id="F1"/>
        </transactionShares>
    so we look for an inner `<value>` first and fall back to the
    element's own text.
    """
    if elem is None:
        return None
    for child in elem.iter():
        if _strip_ns(child.tag) != name:
            continue
        # Prefer nested <value> child (common in Form 4 XBRL)
        for inner in child.iter():
            if inner is child:
                continue
            if _strip_ns(inner.tag) == "value":
                text = (inner.text or "").strip()
                if text:
                    return text
        # Fallback to element's own text
        text = (child.text or "").strip()
        if text:
            return text
    return None


def _form4_xml_url(url: str) -> str:
    """Normalize a Form 4 doc URL to its raw XBRL XML.

    SEC's `primary_doc` field for Form 4 typically points to the
    XSLT-rendered HTML version (path includes `xslF345X06/`). The raw
    XBRL XML lives in the same accession folder without that prefix.
    Strip the prefix so we always parse the source XML, not the rendered
    HTML.
    """
    # Common XSLT renderers for Form 3/4/5
    for prefix in ("xslF345X06/", "xslF345X05/", "xslF345X04/",
                    "xslF345X03/", "xslF345X02/", "xslF345X01/"):
        if prefix in url:
            return url.replace(prefix, "")
    return url


def _fetch_and_parse_form4(url: str) -> list[dict]:
    """Fetch a Form 4 primary doc and extract per-transaction signed values.

    SEC Form 4 has both `nonDerivativeTransaction` and
    `derivativeTransaction` blocks. We focus on non-derivative (the
    cleanest insider-buy/sell signal). Returns one dict per transaction:

        {
          code:         transaction code letter (P / S / F / A / ...),
          shares:       float, signed by sign of code,
          price:        float pricePerShare (0 for grants),
          signed_value: float = shares × price × side_sign,
          reporter:     reporting-owner name (for cluster signal),
          tx_date:      transaction date (string),
        }

    Returns empty list on any parse failure.
    """
    # Normalize the URL to raw XBRL XML (strip XSLT-renderer prefix)
    xml_url = _form4_xml_url(url)
    try:
        r = requests.get(xml_url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        body = r.content
    except Exception as exc:  # noqa: BLE001
        log.debug(f"form4 fetch failed {xml_url}: {exc}")
        return []

    # Form 4 raw XML starts with <?xml ...?> followed by <ownershipDocument>.
    # Some legacy filings ship as a wrapping HTML doc with embedded XML,
    # so we try direct parse first, then regex-extract on failure.
    try:
        root = _ET.fromstring(body)
    except _ET.ParseError:
        m = re.search(rb"<\?xml[^?]*\?>.*?</ownershipDocument>", body, re.DOTALL)
        if not m:
            return []
        try:
            root = _ET.fromstring(m.group(0))
        except _ET.ParseError:
            return []

    # Reporter name (used for cluster-buyer cardinality)
    reporter = None
    for el in root.iter():
        if _strip_ns(el.tag) == "rptOwnerName":
            reporter = (el.text or "").strip()
            break

    out: list[dict] = []
    for tx in root.iter():
        if _strip_ns(tx.tag) != "nonDerivativeTransaction":
            continue

        code = _findtext_localname(tx, "transactionCode")
        if not code:
            continue
        code = code.upper().strip()
        if code in _INFORMATIONAL_CODES:
            continue

        shares_str = _findtext_localname(tx, "transactionShares") \
                       or _findtext_localname(tx, "shares")
        price_str = _findtext_localname(tx, "transactionPricePerShare") \
                       or _findtext_localname(tx, "pricePerShare")
        tx_date = _findtext_localname(tx, "transactionDate") \
                       or _findtext_localname(tx, "date")
        try:
            shares = float(shares_str) if shares_str else 0.0
            price = float(price_str) if price_str else 0.0
        except ValueError:
            continue

        if code in _BUY_CODES:
            side = 1.0
        elif code in _SELL_CODES:
            side = -1.0
        else:
            # Unknown code — treat as informational, skip
            continue

        signed_value = side * shares * price
        out.append({
            "code": code,
            "shares": shares,
            "price": price,
            "signed_value": signed_value,
            "reporter": reporter,
            "tx_date": tx_date,
        })

    return out


def event_filings(ticker: str, *, days: int = 30) -> pd.DataFrame:
    """Just the 8-K material events in the last ``days``."""
    return recent_filings(ticker, days=days, forms=("8-K",))
