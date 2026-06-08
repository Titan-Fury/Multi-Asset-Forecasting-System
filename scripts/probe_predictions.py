"""Predicted-vs-actual probe for one or more tickers.

Runs the on-demand prediction stack at several historical as-of dates,
then compares each prediction against the realised 5-day forward move
that actually happened. Shows whether the model would have been right
at the moment of prediction. Also prints today's fresh prediction.

Usage:
    python scripts/probe_omnitech.py
    python scripts/probe_omnitech.py --tickers OMNITECH,RELIANCE,TCS
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import yfinance as yf

from omni.dashboard.verdict import verdict
from omni.ensemble.pipeline import assemble_signals
from omni.features.build import is_target_col
from omni.models.xgboost_5d import _DROP_ALWAYS
from omni.ondemand.cache import purge_ticker
from omni.ondemand.lookup import lookup_ticker
from omni.ondemand.predict import (
    _cached_models,
    _last_row_features,
    _model_version_tag,
    _score_one,
)
from omni.ondemand.universe import resolve
from omni.rubric.scorer import score_row

TICKER = "OMNITECH"
ASSET_CLASS = "india"
DEFAULT_TICKERS = ["OMNITECH", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]


def _yf_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    for sfx in (".NS", ".BO"):
        try:
            h = yf.Ticker(ticker + sfx).history(period=period, auto_adjust=False)
            if h.empty:
                continue
            h = h.reset_index()
            h["date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
            return h.sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    return None


def _realised_forward_pct(history: pd.DataFrame, asof: pd.Timestamp,
                            horizon: int = 5) -> float | None:
    base = history[history["date"] == asof]
    after = history[history["date"] > asof]
    if base.empty or len(after) < horizon:
        return None
    p0 = float(base.iloc[0]["Close"])
    pH = float(after.iloc[horizon - 1]["Close"])
    if p0 <= 0:
        return None
    return (pH / p0 - 1.0) * 100.0


def _predict_at(asof: pd.Timestamp, models_tuple,
                  res) -> dict | None:
    universal, stack_rec, _cf, _cm, meta_head, _meta_heads, regime_stacks = models_tuple
    purge_ticker(res.ticker)
    bundle = lookup_ticker(res.ticker, res.asset_class,
                            display_name=res.display_name)
    if bundle is None:
        return None
    feats = _last_row_features(bundle)
    feats["date"] = pd.to_datetime(feats["date"]).dt.tz_localize(None)
    sub = feats[feats["date"] <= asof]
    if sub.empty:
        return None
    last = sub.tail(1)
    feat_only = last[[c for c in last.columns
                      if c not in _DROP_ALWAYS and not is_target_col(c)]]
    feat_only = feat_only.apply(pd.to_numeric, errors="coerce")
    rg_proba = bundle.regime.proba_today if bundle.regime is not None else None
    rg_label = bundle.regime.label_today if bundle.regime is not None else None
    proba, kind = _score_one(
        feat_only, universal=universal, stack_rec=stack_rec,
        asset_class=bundle.asset_class, meta_head=meta_head,
        regime_stacks=regime_stacks,
        regime_proba=rg_proba, regime_label=rg_label,
    )
    rgl = rg_label or "chop"
    asof_ts = pd.to_datetime(last.iloc[0]["date"])
    sig_df = assemble_signals(
        last,
        proba_up=pd.Series([float(proba)], index=last.index),
        regime_override=pd.Series({asof_ts: rgl}),
    )
    if sig_df.empty:
        return None
    row = sig_df.iloc[0]
    rub = score_row(last.iloc[0].copy(), regime=rgl,
                    asset_class=bundle.asset_class)
    pillars = {
        "technical": rub.technical_score,
        "fundamental": rub.fundamental_score,
        "sentiment": rub.sentiment_score,
        "macro": rub.macro_score,
        "microstructure": getattr(rub, "microstructure_score", 50.0),
        "oi": getattr(rub, "oi_score", 50.0),
        "relative_strength": getattr(rub, "relative_strength_score", 50.0),
        "insider": getattr(rub, "insider_score", 50.0),
        "earnings_momentum": getattr(rub, "earnings_momentum_score", 50.0),
    }
    card = {
        "proba_up": float(proba),
        "blended_p": float(row.get("blended_p", 0.5)),
        "conviction": float(row.get("conviction", 0.0)),
        "direction": int(row.get("direction", 0)),
        "passes_gate": bool(row.get("passes_gate", False)),
        "regime": rgl,
        "model_kind": kind,
        "min_conviction": 15.0,
        "pillars": pillars,
    }
    v = verdict(card)
    return {
        "asof": asof_ts,
        "proba_up": float(proba),
        "conviction": card["conviction"],
        "direction": card["direction"],
        "blended_p": card["blended_p"],
        "regime": rgl,
        "kind": kind,
        "verdict": v.headline,
        "tone": v.tone,
        "summary": v.summary,
    }


def _probe_one(ticker_query: str, models_tuple) -> dict | None:
    """Run the full probe for a single ticker. Returns aggregate dict or
    None on resolver/yfinance failure."""
    print(f"\n{'='*80}")
    print(f"=== {ticker_query} ===")
    print(f"{'='*80}")
    res = resolve(ticker_query)
    if res is None:
        print(f"resolver could not map {ticker_query!r}. Skipping.")
        return None
    print(f"resolver: {ticker_query!r} -> ticker={res.ticker!r}  "
          f"display={res.display_name!r}  asset_class={res.asset_class}")
    hist = _yf_history(res.ticker, period="6mo")
    if hist is None or hist.empty:
        print(f"yfinance returned no data for {res.ticker} .NS / .BO. Skipping.")
        return None
    print(f"yfinance OK: {len(hist):,} bars, "
          f"latest {hist['date'].max().date()}, "
          f"latest close {float(hist.iloc[-1]['Close']):.2f}")

    # Pick up to 5 historical asof dates with at least 5 trading days of
    # forward data already realised, spaced ~1 week apart.
    asofs = []
    if len(hist) > 6:
        for k in (30, 25, 20, 15, 10, 5):
            if k <= len(hist) - 6:
                asofs.append(pd.Timestamp(hist.iloc[-k]["date"]))
        asofs = list(dict.fromkeys(asofs))[-5:]

    print("\nas-of date     p_up   conv  dir  realised_5d   right?  verdict")
    print("-" * 100)
    n_right = 0
    n_total = 0
    for asof in asofs:
        try:
            pred = _predict_at(asof, models_tuple, res)
        except Exception as exc:  # noqa: BLE001
            print(f"{asof.date()}  prediction error: {type(exc).__name__}: {exc}")
            continue
        if pred is None:
            print(f"{asof.date()}  no prediction (lookup/feature failure)")
            continue
        realised = _realised_forward_pct(hist, asof, horizon=5)
        if realised is None:
            realised_str = "n/a"
            ok_str = "  - "
        else:
            if pred["direction"] == 0:
                ok = abs(realised) < 1.0
            else:
                ok = (pred["direction"] > 0) == (realised > 0)
            n_total += 1
            n_right += int(ok)
            realised_str = f"{realised:+6.2f}%"
            ok_str = " yes" if ok else " no "
        d = pred["direction"]
        d_str = "UP" if d > 0 else ("DN" if d < 0 else "FL")
        print(f"{asof.date()}  {pred['proba_up']:.3f}  {pred['conviction']:>4.1f} "
              f"{d_str:>3}  {realised_str:>11}    {ok_str}    "
              f"[{pred['tone']}] {pred['verdict'][:40]}")
        print(f"             regime={pred['regime']}  kind={pred['kind']}  "
              f"blended_p={pred['blended_p']:.3f}")

    hit_pct = (100 * n_right / n_total) if n_total > 0 else None
    if hit_pct is not None:
        print(f"\nhistorical hit-rate: {n_right}/{n_total} = {hit_pct:.1f}%")

    # Today's fresh prediction.
    today = pd.Timestamp(hist["date"].max())
    print(f"\n=== Today's fresh {ticker_query} prediction (asof {today.date()}) ===")
    pred_today = _predict_at(today, models_tuple, res)
    today_summary = None
    if pred_today is None:
        print("today's prediction failed.")
    else:
        d = pred_today["direction"]
        d_str = "UP" if d > 0 else ("DN" if d < 0 else "FLAT")
        last_close = float(hist.iloc[-1]["Close"])
        print(f"  direction          {d_str}")
        print(f"  proba_up           {pred_today['proba_up']:.3f}")
        print(f"  blended_p          {pred_today['blended_p']:.3f}")
        print(f"  conviction         {pred_today['conviction']:.1f}")
        print(f"  regime             {pred_today['regime']}")
        print(f"  model_kind         {pred_today['kind']}")
        print(f"  last close         INR {last_close:.2f}")
        print(f"  verdict            [{pred_today['tone']}] {pred_today['verdict']}")
        today_summary = {
            "direction": d_str,
            "proba_up": pred_today["proba_up"],
            "conviction": pred_today["conviction"],
            "regime": pred_today["regime"],
            "verdict": pred_today["verdict"],
            "tone": pred_today["tone"],
            "last_close": last_close,
        }
    return {
        "query": ticker_query,
        "ticker": res.ticker,
        "display": res.display_name,
        "n_right": n_right,
        "n_total": n_total,
        "hit_pct": hit_pct,
        "today": today_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-ticker predicted-vs-actual probe")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                          help="comma-separated list of ticker queries")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers given.")
        return 1

    print(f"\n=== Predicted-vs-actual probe across {len(tickers)} tickers ===")
    print(f"Tickers: {', '.join(tickers)}\n")
    models_tuple = _cached_models(_model_version_tag())

    results: list[dict] = []
    for tk in tickers:
        try:
            r = _probe_one(tk, models_tuple)
            if r is not None:
                results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{tk}] FATAL: {type(exc).__name__}: {exc}")

    # Aggregate summary table.
    print(f"\n\n{'='*80}")
    print("=== AGGREGATE SUMMARY ===")
    print(f"{'='*80}")
    print(f"{'ticker':<12}{'->':<3}{'symbol':<14}{'hist hit-rate':<16}"
          f"{'today dir':<10}{'p_up':<7}{'conv':<6}{'verdict':<28}")
    print("-" * 100)
    for r in results:
        hit_str = f"{r['n_right']}/{r['n_total']} = {r['hit_pct']:.0f}%" \
                  if r["hit_pct"] is not None else "n/a"
        if r["today"] is not None:
            t = r["today"]
            print(f"{r['query']:<12}{'->':<3}{r['ticker']:<14}{hit_str:<16}"
                  f"{t['direction']:<10}{t['proba_up']:<7.3f}"
                  f"{t['conviction']:<6.1f}"
                  f"[{t['tone']}] {t['verdict'][:20]:<22}")
        else:
            print(f"{r['query']:<12}{'->':<3}{r['ticker']:<14}{hit_str:<16}"
                  f"(today's prediction failed)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
