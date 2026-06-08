"""End-to-end "predict tomorrow" runner with force-fresh data.

Bypasses all on-demand caches (prices 4h, news 30min, fundamentals 1d)
and pulls everything live before scoring. Use this AFTER market close
on the day before, so the prediction reflects the latest closing print
and the freshest news pass.

Usage (cmd.exe):
    set PYTHONPATH=C:\\Omni_Stocks
    python scripts\\predict_tomorrow.py CIANAGRO INDIA
    python scripts\\predict_tomorrow.py FORCEMOT INDIA
    python scripts\\predict_tomorrow.py CIANAGRO INDIA --deep

Optional environment variables that change behaviour (no code change needed):
    OMNI_CAL_METHOD=beta   -> beta calibration on retraining (better tails).
                              Default isotonic preserves the current trained
                              models.
    OMNI_REGIME_STACKS=1   -> route through the regime-conditional stack
                              bundle if it exists in data/models/.
    OMNI_XGB_DEVICE=cuda   -> use NVIDIA GPU for retraining (training only;
                              inference is always CPU).

What this script prints:
    1. Force-fresh purge confirmation (cache rows dropped)
    2. Bundle freshness diagnostics (latest price date, news age, etc.)
    3. Quick prediction card (universal/stack path)
    4. Optional deep prediction card (per-ticker walk-forward)
    5. Circuit-history evidence (so you can sanity-check the band)
    6. Pair-signal context if any
    7. Saves a JSON dump under data/reports/predict_tomorrow_<TICKER>_<TS>.json

Nothing is committed; the report is local and you can email/diff it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _safe(v):
    """JSON-serialisation helper for pandas/numpy types."""
    try:
        import numpy as np
        import pandas as pd
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # noqa: BLE001
        pass
    return v


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Force-fresh predict-tomorrow runner.")
    parser.add_argument("ticker", help="Ticker symbol (no Yahoo suffix).")
    parser.add_argument("asset_class",
                          type=str.lower,
                          choices=["india", "us", "crypto", "fx"],
                          help="Asset class (case-insensitive).")
    parser.add_argument("--deep", action="store_true",
                          help="Also run the deep walk-forward analysis.")
    parser.add_argument("--no-purge", action="store_true",
                          help="Skip the cache purge (use cached data, faster).")
    parser.add_argument("--report-dir", default="data/reports",
                          help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper().strip()
    asset_class = args.asset_class

    _print_section("1. Environment")
    print(f"  ticker:          {ticker}")
    print(f"  asset_class:     {asset_class}")
    print(f"  deep:            {args.deep}")
    print(f"  purge:           {not args.no_purge}")
    import os
    for env_key in ("OMNI_CAL_METHOD", "OMNI_REGIME_STACKS",
                    "OMNI_XGB_DEVICE"):
        v = os.environ.get(env_key, "(unset)")
        print(f"  {env_key:<22} {v}")

    # Lazy imports so a syntax error in any project module is reported
    # cleanly rather than as a generic import failure.
    try:
        from omni.ondemand import lookup as _lookup
        from omni.ondemand import predict as _predict
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR importing project modules: {e}")
        return 2

    _print_section("2. Force-fresh lookup")
    bundle = None
    try:
        bundle = _lookup.lookup_ticker(
            ticker, asset_class,
            display_name=ticker,
            force_fresh=not args.no_purge,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR lookup_ticker: {e}")
        return 3
    if bundle is None:
        print("ERROR: lookup_ticker returned None (ticker not resolvable).")
        return 4

    # Freshness diagnostics
    last_px = None
    if bundle.prices is not None and not bundle.prices.empty:
        try:
            last_px = bundle.prices["date"].max()
        except Exception:  # noqa: BLE001
            pass
    n_news = int(len(bundle.news_raw)) if bundle.news_raw is not None else 0
    print(f"  display_name:    {bundle.display_name}")
    print(f"  asof:            {bundle.asof}")
    print(f"  prices rows:     {len(bundle.prices) if bundle.prices is not None else 0}")
    print(f"  latest price date: {last_px}")
    print(f"  news rows:       {n_news}")
    if bundle.regime is not None:
        print(f"  regime label:    {bundle.regime.label_today}")
    print(f"  provenance:      {bundle.provenance.__dict__ if bundle.provenance else {}}")

    _print_section("3. Quick prediction (universal/stack)")
    quick_card = None
    try:
        quick_card = _predict.predict_ticker(ticker, asset_class)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR predict_ticker: {e}")
    if quick_card is not None:
        d = quick_card.to_dict()
        keys = ["direction", "proba_up", "conviction", "expected_move_pct",
                "expected_low_pct", "expected_high_pct", "regime",
                "model_kind", "circuit_risk_pct", "circuit_risk_band"]
        for k in keys:
            print(f"  {k:<22} {d.get(k)}")
        if d.get("circuit_history"):
            print("  circuit_history:")
            for k, v in d["circuit_history"].items():
                print(f"      {k}: {v}")
        print(f"  why_text: {d.get('why_text', '')[:240]}")

    deep_card = None
    if args.deep:
        _print_section("4. Deep prediction (per-ticker walk-forward)")
        try:
            deep_card = _predict.deep_analysis(ticker, asset_class)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR deep_analysis: {e}")
        if deep_card is not None:
            d = deep_card.to_dict()
            keys = ["direction", "proba_up", "conviction", "expected_move_pct",
                    "regime", "model_kind", "fold_aucs",
                    "circuit_risk_pct", "circuit_risk_band"]
            for k in keys:
                print(f"  {k:<22} {d.get(k)}")
            print(f"  why_text: {d.get('why_text', '')[:240]}")

    _print_section("5. Sanity comparison")
    if quick_card and deep_card:
        try:
            qd = float(quick_card.proba_up)
            dd = float(deep_card.proba_up)
            spread = abs(qd - dd)
            print(f"  quick proba_up:  {qd:.3f}")
            print(f"  deep  proba_up:  {dd:.3f}")
            print(f"  |spread|:        {spread:.3f}")
            if spread >= 0.10:
                print("  WARNING: quick and deep disagree by >= 10pp.")
                print("           Treat the prediction as low-confidence and")
                print("           rely on the rubric pillars + circuit history.")
            qb = quick_card.circuit_risk_band
            db = deep_card.circuit_risk_band
            if qb != db:
                print(f"  WARNING: circuit-risk band differs: quick={qb} deep={db}")
        except Exception as e:  # noqa: BLE001
            print(f"  spread compute skipped: {e}")
    elif quick_card:
        print("  (deep skipped; quick only)")
    else:
        print("  (no cards produced)")

    _print_section("6. Persist report")
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"predict_tomorrow_{ticker}_{ts}.json"
    payload = {
        "ticker": ticker,
        "asset_class": asset_class,
        "asof": str(bundle.asof) if bundle and bundle.asof is not None else None,
        "latest_price_date": str(last_px) if last_px is not None else None,
        "news_rows": n_news,
        "force_fresh": not args.no_purge,
        "quick": quick_card.to_dict() if quick_card else None,
        "deep": deep_card.to_dict() if deep_card else None,
        "env": {
            "OMNI_CAL_METHOD": os.environ.get("OMNI_CAL_METHOD"),
            "OMNI_REGIME_STACKS": os.environ.get("OMNI_REGIME_STACKS"),
            "OMNI_XGB_DEVICE": os.environ.get("OMNI_XGB_DEVICE"),
        },
    }
    # Stringify any non-JSON-serialisable types.
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(x) for x in o]
        return _safe(o)
    try:
        out_path.write_text(json.dumps(_sanitize(payload), indent=2,
                                          default=str), encoding="utf-8")
        print(f"  wrote: {out_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR writing report: {e}")
        return 5

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
