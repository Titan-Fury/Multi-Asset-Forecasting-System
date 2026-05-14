"""End-to-end prediction probe for CIANAGRO (Phase 24.16 audit).

Runs the full prediction pipeline and inspects every layer:
  1. Resolve the ticker
  2. Fetch the bundle (prices, fundamentals, news)
  3. Score sentiment per article
  4. Score the rubric pillars
  5. Run the universal model
  6. Compare vs the latest snapshot in data/lake/predictions/
  7. Print tomorrow's actionable forecast

Run:
    PYTHONPATH=. .venv\Scripts\python.exe scripts/predict_cianagro.py
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

# Force UTF-8 stdout/stderr on Windows so unicode block chars (█) and emoji
# don't crash with `'charmap' codec can't encode`.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

# Initialize logger early so internal omni.* INFO/WARNING logs are written to
# the rotating log file in logs/ before the first heavy import.
from omni.utils.logger import log  # noqa: E402

log.info("predict_cianagro: probe start")


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    ticker_query = "CIANAGRO"

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
    bundle = lookup_ticker(res.ticker, res.asset_class,
                           display_name=res.display_name)
    if bundle is None:
        print("FAIL: bundle is None — yfinance returned no data")
        return 1
    px = bundle.prices
    print(f"  prices rows:   {len(px) if px is not None else 0}")
    if px is not None and not px.empty:
        print(f"  date range:    {px['date'].min()} -> {px['date'].max()}")
        print(f"  last close:    {float(px['adj_close'].iloc[-1]):.2f}")
        # 5-day vol
        ret = px['adj_close'].pct_change().dropna()
        sigma = float(ret.tail(60).std()) * (5 ** 0.5) * 100.0
        print(f"  5d realised vol (1-sigma): {sigma:.2f}%")
    news = bundle.news_raw
    print(f"  news rows:     {len(news) if news is not None else 0}")
    if news is not None and not news.empty and "source" in news.columns:
        print(f"  news sources:  {news['source'].value_counts().head(5).to_dict()}")
    fund = bundle.fundamentals
    print(f"  fundamentals:  {len(fund) if fund is not None else 0} fields")
    if fund is not None and not fund.empty:
        # Show key fundamentals
        cols = [c for c in ('pe_ratio', 'roe', 'profit_margin', 'revenue_growth',
                             'debt_to_equity', 'market_cap')
                if c in fund.columns]
        if cols:
            print(f"  key fund fields: ")
            for c in cols:
                v = fund[c].iloc[-1] if not fund.empty else None
                print(f"    {c:<22s} {v}")

    _section("3. Sentiment scoring")
    if news is not None and not news.empty:
        from omni.ondemand.lookup import _score_sentiments
        try:
            scored = _score_sentiments(news, top_k=10)
            print(f"  scored {len(scored)} top articles")
            if "sent_score" in scored.columns:
                ss = pd.to_numeric(scored['sent_score'], errors='coerce').dropna()
                print(f"  sent_score: n={len(ss)}, mean={ss.mean():+.3f}, "
                      f"std={ss.std():.3f}, min={ss.min():+.3f}, max={ss.max():+.3f}")
                pos = int((ss > 0.2).sum())
                neg = int((ss < -0.2).sum())
                print(f"  positive (>0.2): {pos}, negative (<-0.2): {neg}, "
                      f"neutral: {len(ss) - pos - neg}")
                # show top 3 most positive / negative headlines
                if "title" in scored.columns:
                    print()
                    top_pos = scored.nlargest(3, "sent_score")[["title", "sent_score"]]
                    print("  most positive:")
                    for _, r in top_pos.iterrows():
                        title = str(r["title"])[:60]
                        print(f"    {float(r['sent_score']):+.2f}  {title}")
                    print("  most negative:")
                    top_neg = scored.nsmallest(3, "sent_score")[["title", "sent_score"]]
                    for _, r in top_neg.iterrows():
                        title = str(r["title"])[:60]
                        print(f"    {float(r['sent_score']):+.2f}  {title}")
        except Exception as e:
            print(f"  sentiment scoring failed: {e}")
    else:
        print("  no news — sentiment will use neutral 50")

    _section("4. Run prediction (universal + rubric blend + gate)")
    from omni.ondemand.predict import predict_ticker
    card = predict_ticker(res.ticker, res.asset_class,
                          display_name=res.display_name)
    if card is None:
        print("FAIL: predict_ticker returned None")
        return 1
    d = card.to_dict()

    _section("5. Prediction results")
    print(f"  ticker:           {d.get('ticker')}")
    print(f"  asof:             {d.get('asof')}")
    print(f"  regime:           {d.get('regime')}")
    print(f"  proba_up:         {d.get('proba_up'):.4f}  "
          f"(model raw — > 0.5 = up, < 0.5 = down)")
    print(f"  blended_p:        {d.get('blended_p'):.4f}  "
          f"(blended with rubric)")
    print(f"  conviction:       {d.get('conviction'):.1f} / 100")
    print(f"  direction:        {d.get('direction')}  "
          f"({'LONG' if d.get('direction') == 1 else 'SHORT' if d.get('direction') == -1 else 'FLAT'})")
    print(f"  passes_gate:      {d.get('passes_gate')}")
    print(f"  expected_move_5d: {d.get('expected_move_pct'):+.2f}%  "
          f"(band: {d.get('expected_low_pct'):+.2f}% / "
          f"{d.get('expected_high_pct'):+.2f}%)")
    print(f"  data_quality:     {d.get('data_quality_score')} / 100")
    print(f"  model_kind:       {d.get('model_kind')}")
    print(f"  why_path:         {d.get('why_path', 'n/a')}")
    if d.get("reasons"):
        print(f"  gate reasons:     {d.get('reasons')}")

    _section("6. Rubric pillar breakdown")
    pillars = d.get("pillars", {})
    if pillars:
        for k in ("technical", "fundamental", "sentiment", "macro", "microstructure"):
            v = pillars.get(k)
            if v is not None:
                bar = "█" * int(v / 5)
                print(f"  {k:<16s} {v:>5.1f} / 100  {bar}")
            else:
                print(f"  {k:<16s} (absent)")
    else:
        print("  pillars dict not found on card")
    print(f"  rubric_total:    {d.get('rubric_total', '?')}")

    _section("7. Tomorrow's projection (1 trading day)")
    last = float(d.get("last_close", 0))
    m1d = float(d.get("move_1d_pct", 0) or 0)
    m1d_lo = float(d.get("move_1d_low_pct", 0) or 0)
    m1d_hi = float(d.get("move_1d_high_pct", 0) or 0)
    if last > 0:
        tgt = last * (1 + m1d / 100)
        tgt_lo = last * (1 + m1d_lo / 100)
        tgt_hi = last * (1 + m1d_hi / 100)
        print(f"  last close:       Rs {last:,.2f}")
        print(f"  expected move:    {m1d:+.2f}%   ({tgt - last:+.2f} INR)")
        print(f"  target:           Rs {tgt:,.2f}")
        print(f"  range (1σ):       Rs {tgt_lo:,.2f}  -  Rs {tgt_hi:,.2f}")
        print(f"  range pct:        {m1d_lo:+.2f}%  /  {m1d_hi:+.2f}%")

    _section("8. Compare vs last published snapshot")
    from pathlib import Path
    snap_dir = Path("data/lake/predictions")
    files = sorted(snap_dir.glob("*.parquet"))
    if files:
        prev = pd.read_parquet(files[-1])
        match = prev[prev["ticker"] == res.ticker]
        if not match.empty:
            r = match.iloc[0]
            print(f"  snapshot:         {files[-1].name}")
            print(f"  prev proba_up:    {float(r.get('proba_up', 0)):.4f}")
            print(f"  prev blended_p:   {float(r.get('blended_p', 0)):.4f}")
            print(f"  prev conviction:  {float(r.get('conviction', 0)):.1f}")
            print(f"  prev direction:   {int(r.get('direction', 0))}")
            print()
            print(f"  delta proba_up:   {d['proba_up'] - float(r.get('proba_up', 0)):+.4f}")
            print(f"  delta conviction: {d['conviction'] - float(r.get('conviction', 0)):+.1f}")
        else:
            print(f"  ticker {res.ticker} not in {files[-1].name}")
    else:
        print("  no previous snapshots in data/lake/predictions/")

    _section("9. Verdict")
    pu = d.get("proba_up", 0.5)
    bp = d.get("blended_p", 0.5)
    conv = d.get("conviction", 0)
    if conv < 30:
        print(f"  LOW CONVICTION ({conv:.0f}/100). Model has no edge on this name.")
        print(f"  proba_up={pu:.3f} (essentially 50/50).")
        print(f"  This is consistent with the global 'model collapsed' finding.")
    elif d.get("passes_gate"):
        d_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(d.get("direction", 0))]
        print(f"  {d_label} signal with conviction {conv:.0f}/100. Gate passed.")
    else:
        print(f"  conviction {conv:.0f}/100, gate blocked: {d.get('reasons', '?')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
