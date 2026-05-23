"""End-to-end prediction probe for MTARTECH (MTAR Technologies) — Friday 2026-05-15."""
from __future__ import annotations

import sys
import warnings

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

from omni.utils.logger import log

log.info("predict_mtartech: probe start")


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    ticker_query = "MTARTECH"

    _section("1. Universe resolution")
    from omni.ondemand.universe import resolve
    res = resolve(ticker_query)
    if res is None:
        print(f"FAIL: could not resolve {ticker_query!r}")
        return 1
    print(f"  ticker:        {res.ticker}")
    print(f"  asset_class:   {res.asset_class}")
    print(f"  display_name:  {res.display_name}")
    print(f"  confident:     {res.confident}")

    _section("2. Live bundle (prices + news + fundamentals)")
    from omni.ondemand.lookup import lookup_ticker
    bundle = lookup_ticker(res.ticker, asset_class=res.asset_class,
                            display_name=res.display_name)
    if not bundle.prices.empty:
        p = bundle.prices.sort_values("date")
        print(f"  prices rows:   {len(p)}")
        print(f"  date range:    {p['date'].min()} -> {p['date'].max()}")
        last_close = float(p["adj_close"].iloc[-1])
        print(f"  last close:    {last_close:.2f}")
        rv5 = float(p["adj_close"].pct_change().tail(5).std() * 100)
        print(f"  5d realised vol (1-sigma): {rv5:.2f}%")
    print(f"  news rows:     {len(bundle.news_raw) if bundle.news_raw is not None else 0}")
    if bundle.news_raw is not None and not bundle.news_raw.empty:
        srcs = bundle.news_raw["source"].value_counts().head(8).to_dict()
        print(f"  news sources:  {srcs}")
    fund_df = bundle.fundamentals
    if fund_df is not None and not fund_df.empty:
        fund = fund_df.iloc[0].to_dict()
    else:
        fund = {}
    print(f"  fundamentals:  {len(fund)} fields")
    for key in ("profit_margin", "revenue_growth", "debt_to_equity",
                "market_cap", "pe_ratio", "earnings_growth"):
        if key in fund and fund[key] is not None and pd.notna(fund[key]):
            print(f"    {key:22s} {fund[key]}")

    _section("3. Run prediction")
    from omni.ondemand.predict import predict_ticker
    card = predict_ticker(res.ticker, asset_class=res.asset_class)
    if card is None:
        print("FAIL: predict_ticker returned None")
        return 1
    d = card.to_dict() if hasattr(card, "to_dict") else card

    _section("4. Prediction results")
    print(f"  ticker:           {d.get('ticker')}")
    print(f"  asof:             {d.get('asof')}")
    print(f"  regime:           {d.get('regime')}")
    print(f"  proba_up:         {d.get('proba_up'):.4f}  (model raw)")
    print(f"  blended_p:        {d.get('blended_p'):.4f}  (blended with rubric)")
    print(f"  conviction:       {d.get('conviction'):.1f} / 100")
    print(f"  direction:        {d.get('direction')}  "
          f"({'LONG' if d.get('direction') == 1 else 'SHORT' if d.get('direction') == -1 else 'FLAT'})")
    print(f"  passes_gate:      {d.get('passes_gate')}")
    if d.get('expected_return_5d') is not None:
        print(f"  expected_move_5d: {100*d.get('expected_return_5d', 0):.2f}%  "
              f"(band: {100*d.get('return_band_lo_5d', 0):.2f}% / "
              f"+{100*d.get('return_band_hi_5d', 0):.2f}%)")
    print(f"  data_quality:     {d.get('data_quality')} / 100")
    print(f"  model_kind:       {d.get('model_kind')}")
    if d.get("why_path"):
        print(f"  why_path:         {d.get('why_path')}")
    if d.get("gate_reasons"):
        print(f"  gate reasons:     {d.get('gate_reasons')}")

    _section("5. Rubric pillar breakdown")
    pillars = d.get("rubric_pillars") or d.get("pillars") or {}
    for k, v in pillars.items():
        bar = "#" * int(float(v) / 5)
        print(f"  {k:<18s} {float(v):.1f} / 100  {bar}")
    print(f"  rubric_total:    {d.get('rubric_total')}")

    _section("6. Per-horizon breakdown (h=1, h=5, h=21)")
    ph = d.get("per_horizon_proba") or {}
    for h in (1, 5, 21):
        if h in ph:
            p = float(ph[h])
            conv = abs(p - 0.5) * 200
            print(f"  h={h:>2}d:  proba_up={p:.4f}  conv={conv:.1f}")

    _section("7. Tomorrow's projection (1 trading day)")
    if d.get("expected_return_1d") is not None:
        last_close = float(bundle.prices["adj_close"].iloc[-1])
        exp_1d = float(d["expected_return_1d"])
        target = last_close * (1 + exp_1d)
        rv = float(bundle.prices["adj_close"].pct_change().tail(20).std())
        lo = last_close * (1 + exp_1d - rv)
        hi = last_close * (1 + exp_1d + rv)
        print(f"  last close:       Rs {last_close:,.2f}")
        print(f"  expected move:    {100*exp_1d:+.2f}%   ({last_close*exp_1d:+.2f} INR)")
        print(f"  target:           Rs {target:,.2f}")
        print(f"  range (1σ):       Rs {lo:,.2f}  -  Rs {hi:,.2f}")

    _section("8. Compare vs latest snapshot")
    from pathlib import Path
    snap_dir = Path("data/lake/predictions")
    found = False
    for fn in sorted(snap_dir.glob("*.parquet"), reverse=True):
        try:
            prev = pd.read_parquet(fn)
        except Exception:
            continue
        match = prev[prev["ticker"] == res.ticker]
        if not match.empty:
            r = match.iloc[-1]
            print(f"  snapshot:         {fn.name}")
            print(f"  prev proba_up:    {float(r.get('proba_up', 0.5)):.4f}")
            print(f"  prev blended_p:   {float(r.get('blended_p', 0.5)):.4f}")
            print(f"  prev conviction:  {float(r.get('conviction', 0)):.1f}")
            print(f"  prev direction:   {int(r.get('direction', 0))}")
            print(f"  delta proba_up:   {d.get('proba_up') - float(r.get('proba_up', 0.5)):+.4f}")
            print(f"  delta conviction: {d.get('conviction') - float(r.get('conviction', 0)):+.1f}")
            found = True
            break
    if not found:
        print("  no prior snapshot for MTARTECH — first prediction")

    _section("9. Verdict")
    pu = float(d.get("proba_up", 0.5))
    conv = float(d.get("conviction", 0))
    direction_lbl = "LONG" if d.get("direction") == 1 else "SHORT" if d.get("direction") == -1 else "FLAT"
    if conv < 15:
        print(f"  LOW CONVICTION ({conv:.0f}/100). Model has no strong edge.")
        print(f"  proba_up={pu:.3f} (essentially neutral).")
    elif d.get("passes_gate"):
        print(f"  PASS — {direction_lbl} at conv={conv:.0f}/100, "
              f"proba_up={pu:.3f}.")
    else:
        gate_reasons = d.get("gate_reasons", "")
        print(f"  BLOCKED ({direction_lbl}, conv={conv:.0f}/100). "
              f"Reasons: {gate_reasons}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
