"""Universe resolver + autocomplete for the on-demand prediction tab.

The curated list below covers ~250 widely-traded instruments across
US large-cap, NSE F&O, top crypto, and major FX pairs. Free-form input
is also accepted: explicit suffixes (`.NS`, `-USD`, `=X`) force the
asset class; otherwise we default to "us" with a warning.

Public API:
    resolve(query) -> Resolution
    suggestions(query, k=10) -> list[Resolution]
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (ticker_id, asset_class, display_name)
# `ticker_id` is what we pass to omni.ingest fetchers (raw symbol, no suffix).
_CURATED: tuple[tuple[str, str, str], ...] = (
    # ---------- US large-cap (~80) ----------
    *((s, "us", n) for s, n in [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet A"),
        ("GOOG", "Alphabet C"), ("AMZN", "Amazon"), ("META", "Meta Platforms"),
        ("NVDA", "NVIDIA"), ("TSLA", "Tesla"), ("BRK-B", "Berkshire Hathaway B"),
        ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("MA", "Mastercard"),
        ("JNJ", "Johnson & Johnson"), ("UNH", "UnitedHealth"), ("XOM", "Exxon Mobil"),
        ("CVX", "Chevron"), ("WMT", "Walmart"), ("HD", "Home Depot"),
        ("PG", "Procter & Gamble"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
        ("DIS", "Disney"), ("BAC", "Bank of America"), ("WFC", "Wells Fargo"),
        ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"), ("C", "Citigroup"),
        ("AXP", "American Express"), ("PFE", "Pfizer"), ("MRK", "Merck"),
        ("ABBV", "AbbVie"), ("LLY", "Eli Lilly"), ("TMO", "Thermo Fisher"),
        ("ABT", "Abbott Labs"), ("DHR", "Danaher"), ("BMY", "Bristol Myers"),
        ("AMGN", "Amgen"), ("GILD", "Gilead"), ("MDT", "Medtronic"),
        ("NFLX", "Netflix"), ("CMCSA", "Comcast"), ("T", "AT&T"),
        ("VZ", "Verizon"), ("AMD", "AMD"), ("INTC", "Intel"),
        ("ORCL", "Oracle"), ("CRM", "Salesforce"), ("ADBE", "Adobe"),
        ("CSCO", "Cisco"), ("QCOM", "Qualcomm"), ("AVGO", "Broadcom"),
        ("TXN", "Texas Instruments"), ("MU", "Micron"), ("AMAT", "Applied Materials"),
        ("LRCX", "Lam Research"), ("KLAC", "KLA Corp"), ("ASML", "ASML"),
        ("PYPL", "PayPal"), ("SQ", "Block"), ("SHOP", "Shopify"),
        ("UBER", "Uber"), ("LYFT", "Lyft"), ("ABNB", "Airbnb"),
        ("DASH", "DoorDash"), ("RBLX", "Roblox"), ("SNAP", "Snap"),
        ("PINS", "Pinterest"), ("SPOT", "Spotify"), ("TWLO", "Twilio"),
        ("ZM", "Zoom"), ("DOCU", "DocuSign"), ("CRWD", "CrowdStrike"),
        ("ZS", "Zscaler"), ("PANW", "Palo Alto Networks"), ("NET", "Cloudflare"),
        ("DDOG", "Datadog"), ("SNOW", "Snowflake"), ("MDB", "MongoDB"),
        ("NKE", "Nike"), ("MCD", "McDonald's"), ("SBUX", "Starbucks"),
        ("COST", "Costco"), ("TGT", "Target"), ("LOW", "Lowe's"),
        ("BA", "Boeing"), ("CAT", "Caterpillar"), ("GE", "General Electric"),
        ("LMT", "Lockheed Martin"), ("RTX", "RTX Corp"), ("NOC", "Northrop Grumman"),
        ("F", "Ford"), ("GM", "General Motors"), ("RIVN", "Rivian"),
        ("LCID", "Lucid"), ("NIO", "NIO"), ("XPEV", "XPeng"),
        ("BABA", "Alibaba"), ("JD", "JD.com"), ("PDD", "PDD Holdings"),
        ("BIDU", "Baidu"), ("TSM", "TSMC"), ("SE", "Sea Ltd"),
        ("PLTR", "Palantir"), ("COIN", "Coinbase"), ("HOOD", "Robinhood"),
        ("MARA", "Marathon Digital"), ("RIOT", "Riot Platforms"),
    ]),
    # ---------- NSE F&O (~50) ----------
    *((s, "india", n) for s, n in [
        ("RELIANCE", "Reliance Industries"), ("TCS", "Tata Consultancy"),
        ("HDFCBANK", "HDFC Bank"), ("INFY", "Infosys"), ("ICICIBANK", "ICICI Bank"),
        ("HINDUNILVR", "Hindustan Unilever"), ("SBIN", "State Bank of India"),
        ("BHARTIARTL", "Bharti Airtel"), ("ITC", "ITC Ltd"), ("LT", "Larsen & Toubro"),
        ("KOTAKBANK", "Kotak Mahindra Bank"), ("AXISBANK", "Axis Bank"),
        ("ASIANPAINT", "Asian Paints"), ("MARUTI", "Maruti Suzuki"),
        ("SUNPHARMA", "Sun Pharma"), ("TITAN", "Titan Company"),
        ("BAJFINANCE", "Bajaj Finance"), ("WIPRO", "Wipro"), ("HCLTECH", "HCL Tech"),
        ("ULTRACEMCO", "UltraTech Cement"), ("ADANIENT", "Adani Enterprises"),
        ("ADANIPORTS", "Adani Ports"), ("ONGC", "ONGC"), ("NTPC", "NTPC"),
        ("POWERGRID", "Power Grid"), ("COALINDIA", "Coal India"),
        ("TATAMOTORS", "Tata Motors"), ("TATASTEEL", "Tata Steel"),
        ("JSWSTEEL", "JSW Steel"), ("HINDALCO", "Hindalco"),
        ("BAJAJFINSV", "Bajaj Finserv"), ("HDFCLIFE", "HDFC Life"),
        ("SBILIFE", "SBI Life"), ("ICICIPRULI", "ICICI Pru Life"),
        ("DIVISLAB", "Divis Labs"), ("DRREDDY", "Dr Reddy's"), ("CIPLA", "Cipla"),
        ("APOLLOHOSP", "Apollo Hospitals"), ("BRITANNIA", "Britannia"),
        ("NESTLEIND", "Nestle India"), ("DABUR", "Dabur"),
        ("GODREJCP", "Godrej Consumer"), ("BAJAJ-AUTO", "Bajaj Auto"),
        ("HEROMOTOCO", "Hero MotoCorp"), ("EICHERMOT", "Eicher Motors"),
        ("M&M", "Mahindra & Mahindra"), ("GRASIM", "Grasim Industries"),
        ("SHRIRAMFIN", "Shriram Finance"), ("CHOLAFIN", "Cholamandalam"),
        ("INDUSINDBK", "IndusInd Bank"), ("BPCL", "BPCL"), ("IOC", "Indian Oil"),
        ("TECHM", "Tech Mahindra"), ("LTIM", "LTIMindtree"),
        # ---- Defense PSUs (often retail-favorite small/mid caps) ----
        ("GRSE", "Garden Reach Shipbuilders"),
        ("MAZDOCK", "Mazagon Dock Shipbuilders"),
        ("COCHINSHIP", "Cochin Shipyard"),
        ("HAL", "Hindustan Aeronautics"),
        ("BEL", "Bharat Electronics"),
        ("BDL", "Bharat Dynamics"),
        ("BEML", "BEML Ltd"),
    ]),
    # ---------- Crypto top-30 (~30) ----------
    *((s, "crypto", n) for s, n in [
        ("BTC", "Bitcoin"), ("ETH", "Ethereum"), ("BNB", "BNB"),
        ("SOL", "Solana"), ("XRP", "XRP"), ("ADA", "Cardano"),
        ("DOGE", "Dogecoin"), ("AVAX", "Avalanche"), ("DOT", "Polkadot"),
        ("LINK", "Chainlink"), ("TRX", "TRON"), ("MATIC", "Polygon"),
        ("LTC", "Litecoin"), ("BCH", "Bitcoin Cash"), ("ATOM", "Cosmos"),
        ("XLM", "Stellar"), ("UNI", "Uniswap"), ("ETC", "Ethereum Classic"),
        ("XMR", "Monero"), ("FIL", "Filecoin"), ("APT", "Aptos"),
        ("NEAR", "NEAR Protocol"), ("ICP", "Internet Computer"),
        ("HBAR", "Hedera"), ("VET", "VeChain"), ("ARB", "Arbitrum"),
        ("OP", "Optimism"), ("INJ", "Injective"), ("SUI", "Sui"),
        ("SHIB", "Shiba Inu"),
    ]),
    # ---------- FX majors + crosses (~12) ----------
    *((s, "fx", n) for s, n in [
        ("EURUSD", "EUR/USD"), ("GBPUSD", "GBP/USD"), ("USDJPY", "USD/JPY"),
        ("USDINR", "USD/INR"), ("AUDUSD", "AUD/USD"), ("USDCAD", "USD/CAD"),
        ("USDCHF", "USD/CHF"), ("NZDUSD", "NZD/USD"), ("EURGBP", "EUR/GBP"),
        ("EURJPY", "EUR/JPY"), ("GBPJPY", "GBP/JPY"), ("AUDJPY", "AUD/JPY"),
    ]),
)


_BY_TICKER = {(t, ac): name for t, ac, name in _CURATED}
_LOOKUP = [(t, ac, name, t.lower(), name.lower()) for t, ac, name in _CURATED]
_ALIAS_RAW: tuple[tuple[str, str, str, str], ...] = (
    # Small/midcap aliases commonly searched by name. Format:
    # (alias_query, ticker_with_optional_suffix, asset_class, display_name)
    # `.BO` is BSE-only on Yahoo; we let fetch_prices() try `.NS` first
    # then fall back to `.BO` automatically. Don't pin `.BO` in the resolver
    # because some downstream modules naively append `.NS` on top.
    ("CIAN AGRO", "CIANAGRO", "india", "CIAN Agro"),
    ("CIANAGRO", "CIANAGRO", "india", "CIAN Agro"),
    ("OMAX AUTOS", "OMAXAUTO", "india", "Omax Autos"),
    ("OMAX AUTO", "OMAXAUTO", "india", "Omax Autos"),
    ("OMAXAUTO", "OMAXAUTO", "india", "Omax Autos"),
    # Legacy / BSE-only autos (NSE delisted, still trades on BSE under
    # a shorter symbol; fetch_prices will auto-fall back to .BO).
    ("HINDUSTAN MOTORS", "HINDMOTORS", "india", "Hindustan Motors"),
    ("HINDMOTORS", "HINDMOTORS", "india", "Hindustan Motors"),
    ("HM LTD", "HINDMOTORS", "india", "Hindustan Motors"),
    # Defence / shipbuilding
    ("GARDEN REACH", "GRSE", "india", "Garden Reach Shipbuilders"),
    ("MAZAGON DOCK", "MAZDOCK", "india", "Mazagon Dock"),
    ("COCHIN SHIPYARD", "COCHINSHIP", "india", "Cochin Shipyard"),
    ("HAL", "HAL", "india", "Hindustan Aeronautics"),
    ("BEL", "BEL", "india", "Bharat Electronics"),
    ("BHARAT DYNAMICS", "BDL", "india", "Bharat Dynamics"),
    # Common retail-search aliases
    ("ZOMATO", "ZOMATO", "india", "Zomato"),
    ("PAYTM", "PAYTM", "india", "Paytm / One97"),
    ("NYKAA", "NYKAA", "india", "Nykaa"),
    ("POLICYBAZAAR", "POLICYBZR", "india", "PB Fintech"),
    ("DMART", "DMART", "india", "Avenue Supermarts"),
    ("IRFC", "IRFC", "india", "Indian Railway Finance"),
    ("IRCTC", "IRCTC", "india", "IRCTC"),
    ("RVNL", "RVNL", "india", "Rail Vikas Nigam"),
    ("YES BANK", "YESBANK", "india", "Yes Bank"),
    ("VEDANTA", "VEDL", "india", "Vedanta"),
    ("LIC", "LICI", "india", "LIC India"),
    ("ZEE", "ZEEL", "india", "Zee Entertainment"),
    ("IDEA", "IDEA", "india", "Vodafone Idea"),
    ("SUZLON", "SUZLON", "india", "Suzlon Energy"),
    ("TATA POWER", "TATAPOWER", "india", "Tata Power"),
    ("TATA CONSUMER", "TATACONSUM", "india", "Tata Consumer"),
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_ALIAS = { _norm(alias): (ticker, ac, name) for alias, ticker, ac, name in _ALIAS_RAW }


@dataclass(frozen=True)
class Resolution:
    ticker: str
    asset_class: str
    display_name: str
    confident: bool  # True if curated, False if fallback parse


# ---------- query parsing ----------

_SUFFIX_NS = re.compile(r"^(.+?)\.NS$", re.IGNORECASE)
_SUFFIX_BO = re.compile(r"^(.+?)\.BO$", re.IGNORECASE)
_SUFFIX_USD = re.compile(r"^(.+?)-USD$", re.IGNORECASE)
_SUFFIX_FX = re.compile(r"^(.+?)=X$", re.IGNORECASE)


def _parse_explicit(q: str) -> Resolution | None:
    if m := _SUFFIX_NS.match(q):
        t = m.group(1).upper()
        return Resolution(f"{t}.NS", "india", _BY_TICKER.get((t, "india"), t), confident=False)
    if m := _SUFFIX_BO.match(q):
        t = m.group(1).upper()
        return Resolution(f"{t}.BO", "india", _BY_TICKER.get((t, "india"), t), confident=False)
    if m := _SUFFIX_USD.match(q):
        t = m.group(1).upper()
        return Resolution(t, "crypto", _BY_TICKER.get((t, "crypto"), t), confident=False)
    if m := _SUFFIX_FX.match(q):
        t = m.group(1).upper()
        return Resolution(t, "fx", _BY_TICKER.get((t, "fx"), t), confident=False)
    return None


def resolve(query: str) -> Resolution | None:
    """Return best-guess resolution. None if query is empty/whitespace.

    Resolution order:
        1. Explicit suffix (.NS / -USD / =X)
        2. Exact ticker match in curated list (case-insensitive)
        3. Exact display-name match
        4. Fallback: treat as US ticker, confident=False
    """
    if not query or not query.strip():
        return None
    q = query.strip()

    explicit = _parse_explicit(q)
    if explicit is not None:
        return explicit

    ql = q.lower()
    qn = _norm(q)
    if qn in _ALIAS:
        t, ac, name = _ALIAS[qn]
        return Resolution(t, ac, name, confident=False)

    for t, ac, name, t_low, name_low in _LOOKUP:
        if t_low == ql:
            return Resolution(t, ac, name, confident=True)
    for t, ac, name, t_low, name_low in _LOOKUP:
        if name_low == ql:
            return Resolution(t, ac, name, confident=True)
    for t, ac, name, _, _ in _LOOKUP:
        if _norm(name) == qn:
            return Resolution(t, ac, name, confident=True)
        if _norm(t) == qn:
            return Resolution(t, ac, name, confident=True)

    # ---- Typo-tolerant fuzzy match against curated + aliases ----
    # Uses character-bigram overlap (Sørensen-Dice). Cheap and good enough
    # for "oman autos" -> "omax autos", "relience" -> "reliance", etc.
    fuzzy = _fuzzy_match(qn)
    if fuzzy is not None:
        return fuzzy

    # If user typed a spaced company-like name, prefer India compact ticker first.
    if " " in q and qn:
        return Resolution(qn.upper(), "india", q.strip().title(), confident=False)

    # Generic fallback: assume US equity.
    return Resolution(q.upper(), "us", q.upper(), confident=False)


def _bigrams(s: str) -> set[str]:
    s = f"  {s}  "
    return {s[i:i+2] for i in range(len(s) - 1)}


def _dice(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.0
    return 2 * len(A & B) / (len(A) + len(B))


def _fuzzy_match(qn: str, *, min_score: float = 0.65) -> Resolution | None:
    """Find the best-scoring curated/alias entry by character-bigram overlap."""
    if not qn or len(qn) < 3:
        return None
    best: tuple[float, Resolution] | None = None
    # Score against curated list
    for t, ac, name in _CURATED:
        score = max(_dice(qn, _norm(t)), _dice(qn, _norm(name)))
        if score >= min_score and (best is None or score > best[0]):
            best = (score, Resolution(t, ac, name, confident=False))
    # Score against alias list
    for alias, t, ac, name in _ALIAS_RAW:
        score = max(_dice(qn, _norm(alias)), _dice(qn, _norm(name)),
                     _dice(qn, _norm(t)))
        if score >= min_score and (best is None or score > best[0]):
            best = (score, Resolution(t, ac, name, confident=False))
    return best[1] if best else None


def suggestions(query: str, k: int = 10) -> list[Resolution]:
    """Return up to k autocomplete suggestions ranked by:
        1. ticker prefix match
        2. ticker substring match
        3. name prefix match
        4. name substring match
    """
    if not query or not query.strip():
        # default: top-of-list previews per asset class
        out: list[Resolution] = []
        seen = set()
        for ac in ("us", "india", "crypto", "fx"):
            for t, a, name, _, _ in _LOOKUP:
                if a == ac and (t, a) not in seen:
                    out.append(Resolution(t, a, name, True))
                    seen.add((t, a))
                    break
        return out[:k]

    ql = query.strip().lower()
    qn = _norm(query.strip())
    buckets: list[list[Resolution]] = [[], [], [], []]
    seen: set[tuple[str, str]] = set()
    for t, ac, name, t_low, name_low in _LOOKUP:
        key = (t, ac)
        if key in seen:
            continue
        if t_low.startswith(ql):
            buckets[0].append(Resolution(t, ac, name, True)); seen.add(key)
        elif ql in t_low:
            buckets[1].append(Resolution(t, ac, name, True)); seen.add(key)
        elif name_low.startswith(ql):
            buckets[2].append(Resolution(t, ac, name, True)); seen.add(key)
        elif ql in name_low:
            buckets[3].append(Resolution(t, ac, name, True)); seen.add(key)
        elif qn and (qn in _norm(name) or qn in _norm(t)):
            buckets[3].append(Resolution(t, ac, name, True)); seen.add(key)

    flat = [r for bucket in buckets for r in bucket]
    if qn in _ALIAS:
        t, ac, name = _ALIAS[qn]
        alias_res = Resolution(t, ac, name, confident=False)
        if all((r.ticker, r.asset_class) != (alias_res.ticker, alias_res.asset_class) for r in flat):
            flat = [alias_res] + flat

    # Also surface aliases that *contain* the query (e.g. "zo" → ZOMATO).
    if ql:
        seen_aliases: set[tuple[str, str]] = {(r.ticker, r.asset_class) for r in flat}
        for alias, t, ac, name in _ALIAS_RAW:
            if (t, ac) in seen_aliases:
                continue
            if ql in alias.lower() or ql in name.lower() or ql in t.lower():
                flat.append(Resolution(t, ac, name, confident=False))
                seen_aliases.add((t, ac))
    return flat[:k]


def all_curated() -> list[Resolution]:
    return [Resolution(t, ac, name, True) for t, ac, name in _CURATED]
