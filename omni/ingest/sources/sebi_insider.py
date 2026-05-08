"""SEBI / NSE Insider Trading (PIT-2015) disclosures.

Under SEBI's Prohibition of Insider Trading Regulations 2015, listed
promoters / KMPs / connected persons must disclose every trade in their
own securities. NSE publishes the daily aggregate at:
  https://www.nseindia.com/api/corporates-pit?index=equities&...

We hit that endpoint with the NSE-cookie session and surface:
  - all PIT disclosures from the last N days
  - per-ticker filter
  - aggregate "insider intensity" similar to bulk-deals

Schema: [date, ticker, person, person_category, transaction_type, qty,
         value_lakhs, mode, post_holding_pct]

Cached at TTL.news (30 min).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from omni.ingest.sources.nse_flows import _nse_session, _HEADERS_API
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


def _format(d) -> str:
    return d.strftime("%d-%m-%Y")


@cached(ttl=TTL.news, key="sebi_pit:{days}")
def insider_disclosures(days: int = 30) -> pd.DataFrame:
    """Recent SEBI PIT-2015 insider disclosures across the NSE listed universe."""
    s = _nse_session()
    s.headers.update(_HEADERS_API)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url = (f"https://www.nseindia.com/api/corporates-pit?"
            f"index=equities&from_date={_format(start)}&to_date={_format(end)}")
    try:
        r = s.get(url, timeout=12)
        if r.status_code != 200:
            log.debug(f"NSE PIT: status={r.status_code}")
            return pd.DataFrame()
        data = r.json()
        rows = data.get("data", []) if isinstance(data, dict) else []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # NSE uses descriptive keys — pull the ones we want
        rename = {
            "symbol": "ticker", "company": "company",
            "anex": "person_category",      # Annexure code (eg PIT-2015 reg 7)
            "personal": "person",            # may be the person's name
            "tdpTransactionType": "transaction_type",
            "secAcq": "qty",                  # securities acquired
            "secVal": "value_lakhs",          # value in INR (we'll /1e5)
            "tdpDerivativeContractType": "mode",
            "buyValue": "buy_value",
            "sellValue": "sell_value",
            "afterAcqSharesNo": "post_holding_qty",
            "afterAcqSharesPer": "post_holding_pct",
            "date": "date",
        }
        df = df.rename(columns=rename)
        keep = [c for c in ("date", "ticker", "company", "person", "person_category",
                              "transaction_type", "qty", "value_lakhs", "mode",
                              "post_holding_pct")
                 if c in df.columns]
        df = df[keep]
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
        for c in ("qty", "value_lakhs", "post_holding_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "value_lakhs" in df.columns:
            # NSE returns raw INR; convert to lakhs
            df["value_lakhs"] = (df["value_lakhs"] / 1e5).round(2)
        return df.reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        log.warning(f"sebi PIT fetch failed: {e}")
        return pd.DataFrame()


def insider_intensity(ticker: str, *, days: int = 30) -> dict:
    """Aggregate insider activity for one ticker.

    Returns: {n_disclosures, n_buys, n_sells, net_value_lakhs,
              last_disclosure_date}
    """
    out = {"n_disclosures": 0, "n_buys": 0, "n_sells": 0,
            "net_value_lakhs": 0.0, "last_disclosure_date": None}
    df = insider_disclosures(days=days)
    if df is None or df.empty or "ticker" not in df.columns:
        return out
    sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if sub.empty:
        return out
    out["n_disclosures"] = int(len(sub))
    if "transaction_type" in sub.columns:
        ttype = sub["transaction_type"].astype(str).str.lower()
        out["n_buys"] = int(ttype.str.contains("buy|acqui|purchase").sum())
        out["n_sells"] = int(ttype.str.contains("sell|dispos|sale").sum())
    if "value_lakhs" in sub.columns and "transaction_type" in sub.columns:
        ttype = sub["transaction_type"].astype(str).str.lower()
        signed = sub["value_lakhs"].fillna(0).astype(float) * (
            ttype.str.contains("buy|acqui|purchase").map({True: 1, False: 0})
            - ttype.str.contains("sell|dispos|sale").map({True: 1, False: 0})
        )
        out["net_value_lakhs"] = float(signed.sum())
    if "date" in sub.columns:
        last = sub["date"].max()
        out["last_disclosure_date"] = str(pd.Timestamp(last).date()) if pd.notna(last) else None
    return out
