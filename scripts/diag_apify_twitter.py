"""Standalone diagnostic for the Apify Twitter ingest.

Run before flipping `social.twitter.enabled` in config/sources.yaml so
you can verify the actor + token + universe pull together cleanly.

Usage (cmd.exe):
    set APIFY_TOKEN=apify_api_xxx
    python scripts/diag_apify_twitter.py RELIANCE TCS INFY

If no tickers are passed, falls back to a tiny built-in sample.
The script prints:
  1. Token + actor reachability
  2. Per-ticker tweet counts
  3. First 3 tweets per ticker (text + engagement + signed lex score)
  4. A FinBERT score on the merged frame (if FinBERT available)
  5. Estimated $ cost for the run

Nothing is persisted; this is dry-run only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python scripts/diag_apify_twitter.py` from project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _print_header(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def main(argv: list[str]) -> int:
    tickers = [a.upper() for a in argv if a and not a.startswith("-")]
    if not tickers:
        tickers = ["RELIANCE", "TCS", "INFY"]

    _print_header("1. Environment check")
    token = os.environ.get("APIFY_TOKEN", "").strip()
    print(f"APIFY_TOKEN set: {bool(token)}  (len={len(token)})")
    if not token:
        print("ERROR: set APIFY_TOKEN before running. Aborting.")
        return 2

    try:
        import apify_client  # noqa: F401
        print("apify_client import: ok")
    except ImportError:
        print("ERROR: pip install apify-client")
        return 3

    from omni.ingest.sources import apify_twitter as tw

    _print_header("2. Actor invocation")
    print(f"actor: {tw.DEFAULT_ACTOR_ID}")
    print(f"tickers: {tickers}")
    print(f"max_items_per_ticker: {tw.DEFAULT_MAX_ITEMS_PER_TICKER}")
    print(f"lookback_hours: {tw.DEFAULT_LOOKBACK_HOURS}")
    df = tw.fetch_for_tickers(
        tickers,
        max_items_per_ticker=tw.DEFAULT_MAX_ITEMS_PER_TICKER,
        lookback_hours=tw.DEFAULT_LOOKBACK_HOURS,
        batch_size=min(8, len(tickers)),
    )
    print(f"rows returned: {len(df)}")

    _print_header("3. Per-ticker counts")
    if df.empty:
        print("EMPTY frame; check actor permissions or that the search terms")
        print("actually match recent activity. (`since:` operator can return 0")
        print("results outside Indian-market hours.)")
        return 1
    counts = (df.groupby("ticker_hint").size().sort_values(ascending=False))
    for t, n in counts.items():
        print(f"  {t}: {n}")

    _print_header("4. Sample tweets (first 3 per ticker)")
    for t in counts.index:
        sub = df[df["ticker_hint"] == t].head(3)
        print(f"\n-- {t} ({len(df[df['ticker_hint']==t])} total) --")
        for _, row in sub.iterrows():
            text = (row.get("title") or "")[:140].replace("\n", " ")
            score = row.get("sent_score", 0.0) or 0.0
            print(f"  [{score:+.2f}] {row.get('published_at')!s:.19s}  {text}")
            print(f"        url: {row.get('url')}")

    _print_header("5. FinBERT batch (optional)")
    try:
        from omni.features.sentiment import score_articles
        scored = score_articles(df.drop(columns=["sent_score"], errors="ignore"))
        if "sent_score" in scored.columns and not scored.empty:
            mean_per_ticker = (
                scored.groupby("ticker_hint")["sent_score"]
                .mean().sort_values(ascending=False)
            )
            print("Mean FinBERT sent_score per ticker:")
            for t, v in mean_per_ticker.items():
                print(f"  {t}: {v:+.3f}")
        else:
            print("FinBERT scoring returned no scores (backend probably absent).")
    except Exception as e:  # noqa: BLE001
        print(f"FinBERT step skipped: {e}")

    _print_header("6. Estimated cost")
    # apidojo/tweet-scraper free-tier billing is roughly USD 0.25-0.40 per
    # 1k tweets returned. Round up to be conservative.
    est_low = len(df) * 0.00025
    est_high = len(df) * 0.00040
    print(f"this run returned {len(df)} tweets")
    print(f"  -> approx USD {est_low:.4f} - {est_high:.4f} consumed")

    print("\nDONE. If counts look reasonable, flip:")
    print("  config/sources.yaml -> social.twitter.enabled: true")
    print("and run a daily orchestrator pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
