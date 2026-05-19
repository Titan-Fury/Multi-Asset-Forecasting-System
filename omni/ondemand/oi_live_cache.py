"""Last-good-snapshot fallback for NSE option-chain OI data.

The live `nse_oi.option_chain()` endpoint is intermittently blocked by
NSE anti-bot measures even with the dedicated per-endpoint warmup. When
the live call returns empty (`has_option_chain=False`) we want to
gracefully fall back to the most recent successful snapshot on disk
rather than letting the OI rubric pillar collapse to neutral 50.

Storage layout:
    data/lake/oi_live_cache/<TICKER>.parquet      # 1 row per attempt date

Schema:
    [date, ticker, oi_pcr, max_pain_strike, max_pain_dist_pct,
     oi_total_lakhs, ce_oi_lakhs, pe_oi_lakhs, spot, has_option_chain]

Behavior:
    - On each call, if the live snapshot succeeds, append/replace today's
      row and return live data.
    - If the live snapshot fails (has_option_chain=False), look for the
      most recent row on disk within `max_age_days` (default 5). Return
      that row's values with a `_stale_age_days` flag for diagnostics.
    - If no fallback exists either, return the original empty-shape dict
      so downstream code's `has_option_chain=False` branch still runs.

Idempotent: re-running the same day replaces today's row, never duplicates.
Crash-safe: write failures are logged but never raise.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.utils.logger import log
from omni.utils.paths import local_data_root

# Anchor on local_data_root so the cache works regardless of CWD. Previously
# used Path("data") which silently resolved to a different folder if any
# caller (orchestrator, tests) ran from outside the project root.
CACHE_DIR = local_data_root() / "lake" / "oi_live_cache"
DEFAULT_MAX_AGE_DAYS = 5

_FIELDS = (
    "oi_pcr", "max_pain_strike", "max_pain_dist_pct",
    "oi_total_lakhs", "ce_oi_lakhs", "pe_oi_lakhs",
    "spot", "has_option_chain",
)


def _path(ticker: str) -> Path:
    safe = str(ticker).upper().replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _read_history(ticker: str) -> pd.DataFrame:
    p = _path(ticker)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        return df
    except Exception as exc:  # noqa: BLE001
        log.debug(f"oi_live_cache read failed for {ticker}: {exc}")
        return pd.DataFrame()


def _write_row(ticker: str, row: dict) -> None:
    """Append-or-replace today's row. Best-effort: never raises."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hist = _read_history(ticker)
        today = pd.Timestamp.now().normalize()
        if not hist.empty and (hist["date"] == today).any():
            hist = hist[hist["date"] != today]
        new_row = {"date": today, "ticker": ticker.upper(), **row}
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist = hist.sort_values("date").reset_index(drop=True)
        hist.to_parquet(_path(ticker), index=False)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"oi_live_cache write failed for {ticker}: {exc}")


def get_or_fallback(ticker: str, *,
                     max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> dict:
    """Return live OI features if the endpoint works, else last-good snapshot.

    Output shape always matches `nse_oi.oi_features` so callers can be
    schema-stable. Adds a `_stale_age_days` int when serving from cache
    (0 means today / fresh; >0 means falling back).

    Schema-stable degradation: when both live AND cache are empty,
    returns the original zeros dict from `oi_features` (with
    `has_option_chain=False`). Callers that gate on that flag stay
    correct.
    """
    nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")

    # 1) Try live
    try:
        from omni.ingest.sources.nse_oi import oi_features
        live = oi_features(nse_sym) or {}
    except Exception as exc:  # noqa: BLE001
        log.debug(f"oi_live_cache: live oi_features failed for {nse_sym}: {exc}")
        live = {}

    if live.get("has_option_chain"):
        # Persist the fresh snapshot then return as-is
        try:
            _write_row(nse_sym, {k: live.get(k) for k in _FIELDS})
        except Exception:  # noqa: BLE001
            pass
        live["_stale_age_days"] = 0
        return live

    # 2) Fall back to most recent on-disk snapshot
    hist = _read_history(nse_sym)
    if hist.empty:
        # Return whatever live had (likely zeros + has_option_chain=False)
        return live or {
            "oi_pcr": 0.0, "max_pain_strike": 0.0, "max_pain_dist_pct": 0.0,
            "oi_total_lakhs": 0.0, "ce_oi_lakhs": 0.0, "pe_oi_lakhs": 0.0,
            "spot": 0.0, "has_option_chain": False,
        }

    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=max_age_days)
    fresh = hist[(hist["date"] >= cutoff) & (hist["has_option_chain"] == True)]  # noqa: E712
    if fresh.empty:
        return live or {
            "oi_pcr": 0.0, "max_pain_strike": 0.0, "max_pain_dist_pct": 0.0,
            "oi_total_lakhs": 0.0, "ce_oi_lakhs": 0.0, "pe_oi_lakhs": 0.0,
            "spot": 0.0, "has_option_chain": False,
        }

    last = fresh.sort_values("date").iloc[-1]
    age = int((today - last["date"]).days)
    out = {k: (float(last[k]) if k in last and pd.notna(last[k])
                else (False if k == "has_option_chain" else 0.0))
            for k in _FIELDS}
    # Keep has_option_chain=True so the patcher ingests the values, but
    # tag the staleness so the rubric / monitoring can see it.
    out["has_option_chain"] = bool(last.get("has_option_chain", False))
    out["_stale_age_days"] = age
    log.info(
        f"oi_live_cache: serving stale snapshot for {nse_sym} "
        f"(age={age}d, pcr={out['oi_pcr']:.3f})"
    )
    return out


def snapshot_now(ticker: str) -> dict:
    """Force a live snapshot + append. Used by daily orchestrator warm-up.

    Same return contract as get_or_fallback. Calling this from the
    orchestrator pre-warms the cache so per-ticker predict_ticker calls
    later in the day can fall back to today's row even if NSE rate-limits
    the per-prediction call.
    """
    return get_or_fallback(ticker, max_age_days=0)
