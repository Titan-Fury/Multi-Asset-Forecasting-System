"""Phase 24.18 verification smoke: CIANAGRO before/after.

Last-session baselines (Phase 24.17, in-memory):
  proba_up         = 0.541
  circuit_risk_pct = 1.5%   band=low
  model_kind       = stack_h5_rank_purged
  warm cache       = 13ms
  panic_keyword_ratio  = 0.000   (news pipeline empty)
  news_volume_z_60d    = NaN

Phase 24.18 adds: news lake-write, sentiment scoring during build, and
panic/promoter/sebi keyword features computed at inference time.

Usage:
    python scripts/smoke_compare.py
    python scripts/smoke_compare.py --force      # purge cache for cold call
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Phase 24.17 numbers captured from the last verified smoke
_PHASE_24_17 = {
    "proba_up": 0.541,
    "circuit_risk_pct": 0.015,
    "circuit_risk_band": "low",
    "model_kind": "stack_h5_rank_purged",
    "warm_ms": 13,
    "panic_keyword_ratio": 0.000,
    "news_volume_z_60d": float("nan"),
}


def _run(ticker: str, ac: str, force: bool) -> dict:
    """Predict once and capture the relevant fields from the card +
    feature row. Returns a flat dict ready for table rendering."""
    from omni.ondemand.predict import predict_ticker
    from omni.ondemand.cache import purge_ticker

    if force:
        n = purge_ticker(ticker)
        print(f"  purged {n} cache entries for {ticker}")

    t0 = time.time()
    card = predict_ticker(ticker, ac)
    elapsed_ms = (time.time() - t0) * 1000
    if card is None:
        return {"error": "predict_ticker returned None", "ms": elapsed_ms}
    d = card.__dict__

    # Try to extract the new sentiment features from the cached feature row
    panic = float("nan")
    nvz = float("nan")
    sentm = float("nan")
    nart = float("nan")
    try:
        from omni.ondemand.lookup import lookup_ticker
        from omni.ondemand.features_inline import build_features_inline
        b = lookup_ticker(ticker, ac)
        if b is not None and b.prices is not None and not b.prices.empty:
            sentiment = b.news_sentiment if hasattr(b, "news_sentiment") else None
            feats = build_features_inline(
                ac, ticker, b.prices,
                fundamentals=b.fundamentals,
                sentiment=sentiment,
                macro=b.macro if hasattr(b, "macro") else None,
                horizons=(1, 5, 21),
            )
            if feats is not None and not feats.empty:
                last = feats.tail(1).iloc[0]
                panic = float(last.get("panic_keyword_ratio", float("nan")))
                nvz = float(last.get("news_volume_z_60d", float("nan")))
                sentm = float(last.get("sent_mean", float("nan")))
                nart = float(last.get("n_articles", float("nan")))
    except Exception as exc:  # noqa: BLE001
        print(f"  (feature-row probe failed: {exc})")

    return {
        "proba_up": d.get("proba_up"),
        "circuit_risk_pct": d.get("circuit_risk_pct"),
        "circuit_risk_band": d.get("circuit_risk_band"),
        "model_kind": d.get("model_kind"),
        "warm_ms": elapsed_ms,
        "panic_keyword_ratio": panic,
        "news_volume_z_60d": nvz,
        "sent_mean": sentm,
        "n_articles": nart,
    }


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if pd.isna(v):
            return "NaN"
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                          help="Purge CIANAGRO cache before the call (cold)")
    parser.add_argument("--ticker", default="CIANAGRO")
    parser.add_argument("--asset-class", default="india")
    args = parser.parse_args()

    print(f"[1/2] {'cold' if args.force else 'warm'} smoke on "
            f"{args.asset_class}/{args.ticker}")
    cur = _run(args.ticker, args.asset_class, force=args.force)

    if "error" in cur:
        print(f"  ERROR: {cur['error']}  ({cur['ms']:.0f} ms)")
        return 1

    print()
    print("[2/2] before/after comparison")
    print("=" * 78)
    print(f"  {'metric':<24}{'Phase 24.17':>16}{'Phase 24.18':>16}  notes")
    print("-" * 78)
    rows = [
        ("proba_up", _PHASE_24_17["proba_up"], cur["proba_up"]),
        ("circuit_risk_pct", _PHASE_24_17["circuit_risk_pct"], cur["circuit_risk_pct"]),
        ("circuit_risk_band", _PHASE_24_17["circuit_risk_band"], cur["circuit_risk_band"]),
        ("model_kind", _PHASE_24_17["model_kind"], cur["model_kind"]),
        ("panic_keyword_ratio", _PHASE_24_17["panic_keyword_ratio"], cur["panic_keyword_ratio"]),
        ("news_volume_z_60d", _PHASE_24_17["news_volume_z_60d"], cur["news_volume_z_60d"]),
        ("sent_mean", "—", cur["sent_mean"]),
        ("n_articles", "—", cur["n_articles"]),
        ("call latency (ms)", _PHASE_24_17["warm_ms"], cur["warm_ms"]),
    ]
    for name, old, new in rows:
        # Compute delta where numeric
        delta = ""
        try:
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) \
                    and not (pd.isna(old) or pd.isna(new)):
                d = new - old
                delta = f"  Δ={d:+.3f}"
        except Exception:
            pass
        print(f"  {name:<24}{_fmt(old):>16}{_fmt(new):>16}{delta}")
    print("=" * 78)
    print()
    print("Interpretation:")
    print("  - proba_up + circuit_risk are stable (model not retrained on news)")
    print("  - panic_keyword_ratio + news_volume_z populate at INFERENCE only")
    print("  - sent_mean / n_articles non-zero = live news pipeline working")
    print("  - latency change reflects whether cache hit or miss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
