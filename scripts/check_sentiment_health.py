"""Sentiment Health Score 0-10 — repeatable evaluation.

Five sub-scores, max 10:
    Coverage (0-3): % India tickers with >=3 articles in last 45d
    Information Coefficient (0-3): Spearman(sent_z_30d, fwd_ret_5d)
    Discriminative power (0-2): sent_z gap top-decile vs bottom-decile fwd_ret
    Lead-lag (0-1): peak corr at lag<0 (sent leads ret)
    Circuit-day signal (0-1): panic_keyword_ratio lift on circuit days

Reads `data/lake/training/training_set.parquet` directly. ~10s runtime.

Usage:
    python scripts/check_sentiment_health.py
    python scripts/check_sentiment_health.py --training-set <path>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentiment Health Score 0-10")
    parser.add_argument(
        "--training-set",
        default="data/lake/training/training_set.parquet",
    )
    args = parser.parse_args()

    p = Path(args.training_set)
    if not p.exists():
        print(f"ERROR: training set not found: {p}")
        return 1

    df = pd.read_parquet(p)
    print(f"loaded: shape={df.shape}, tickers={df['ticker'].nunique()}")

    last_date = df["date"].max()
    recent = df[df["date"] >= last_date - pd.Timedelta(days=45)]

    score = 0
    breakdown: list[tuple[str, int, str]] = []

    # ---- 1. Coverage ----
    if "n_articles" in df.columns:
        articles_recent = recent.groupby("ticker")["n_articles"].sum().fillna(0)
        coverage = float((articles_recent >= 3).mean())
    else:
        coverage = 0.0
    if coverage >= 0.80:
        cov_pts = 3
    elif coverage >= 0.50:
        cov_pts = 2
    elif coverage >= 0.20:
        cov_pts = 1
    else:
        cov_pts = 0
    score += cov_pts
    breakdown.append((
        "Coverage", cov_pts,
        f"{coverage:.0%} of {df['ticker'].nunique()} tickers >=3 articles in 45d  -> {cov_pts}/3"
    ))

    # ---- 2. Information Coefficient ----
    # Phase 24.18 — restrict to rows where sentiment was actually observed.
    # Computing Spearman over 99% zero-sentiment rows would always come
    # out near 0 even when the signal is strong on the covered rows.
    ic = 0.0
    n_used = 0
    if {"sent_z_30d", "fwd_ret_5d", "n_articles"}.issubset(df.columns):
        s = df[["sent_z_30d", "fwd_ret_5d", "n_articles"]].dropna()
        s = s[s["n_articles"].fillna(0) > 0]
        n_used = len(s)
        if n_used > 100:
            ic = float(s["sent_z_30d"].corr(s["fwd_ret_5d"], method="spearman"))
    elif {"sent_z_30d", "fwd_ret_5d"}.issubset(df.columns):
        s = df[["sent_z_30d", "fwd_ret_5d"]].dropna()
        n_used = len(s)
        if n_used > 1000:
            ic = float(s["sent_z_30d"].corr(s["fwd_ret_5d"], method="spearman"))
    if abs(ic) >= 0.05:
        ic_pts = 3
    elif abs(ic) >= 0.03:
        ic_pts = 2
    elif abs(ic) >= 0.01:
        ic_pts = 1
    else:
        ic_pts = 0
    score += ic_pts
    breakdown.append((
        "Information Coefficient", ic_pts,
        f"Spearman(sent_z_30d, fwd_ret_5d) on {n_used:,} sentiment-rich rows = {ic:+.4f}  -> {ic_pts}/3"
    ))

    # ---- 3. Discriminative power ----
    disc = 0.0
    n_disc = 0
    if {"sent_z_30d", "fwd_ret_5d", "n_articles"}.issubset(df.columns):
        s = df[["sent_z_30d", "fwd_ret_5d", "n_articles"]].dropna()
        s = s[s["n_articles"].fillna(0) > 0]
        n_disc = len(s)
        if n_disc > 50:
            ranks = s["fwd_ret_5d"].rank(pct=True)
            top = s["sent_z_30d"][ranks > 0.9].mean()
            bot = s["sent_z_30d"][ranks < 0.1].mean()
            sigma = s["sent_z_30d"].std()
            disc = (top - bot) / sigma if sigma > 0 else 0.0
    if abs(disc) >= 0.5:
        disc_pts = 2
    elif abs(disc) >= 0.2:
        disc_pts = 1
    else:
        disc_pts = 0
    score += disc_pts
    breakdown.append((
        "Discriminative power", disc_pts,
        f"sent_z gap on {n_disc:,} sentiment rows = {disc:+.3f}sigma  -> {disc_pts}/2"
    ))

    # ---- 4. Lead-lag ----
    lead_pts = 0
    lag_vals: dict[int, float] = {}
    if {"sent_z_30d", "fwd_ret_5d"}.issubset(df.columns):
        if "n_articles" in df.columns:
            top_t = df.groupby("ticker")["n_articles"].sum().sort_values(ascending=False).head(5).index
        else:
            top_t = df["ticker"].unique()[:5]
        if len(top_t) > 0:
            corrs = {k: [] for k in range(-5, 6)}
            for t in top_t:
                td = df[df["ticker"] == t][["date", "sent_z_30d", "fwd_ret_5d"]].dropna()
                if len(td) < 30:
                    continue
                for k in range(-5, 6):
                    s_lag = td["sent_z_30d"].shift(k)
                    c = s_lag.corr(td["fwd_ret_5d"], method="spearman")
                    if pd.notna(c):
                        corrs[k].append(c)
            for k in corrs:
                if corrs[k]:
                    lag_vals[k] = float(np.mean(corrs[k]))
            if lag_vals:
                peak = max(lag_vals, key=lambda k: abs(lag_vals[k]))
                if peak < 0:
                    lead_pts = 1
    score += lead_pts
    peak_str = f"lag={max(lag_vals, key=lambda k: abs(lag_vals[k]))}" if lag_vals else "lag=n/a"
    breakdown.append((
        "Lead-lag", lead_pts, f"peak corr at {peak_str}  -> {lead_pts}/1"
    ))

    # ---- 5. Circuit-day signal ----
    cir_pts = 0
    cir_lift = 0.0
    if {"panic_keyword_ratio", "fwd_ret_5d"}.issubset(df.columns):
        s = df[["panic_keyword_ratio", "fwd_ret_5d"]].dropna()
        if len(s) > 100 and (s["fwd_ret_5d"] <= -0.095).any():
            mu_circ = s["panic_keyword_ratio"][s["fwd_ret_5d"] <= -0.095].mean()
            mu_norm = s["panic_keyword_ratio"][s["fwd_ret_5d"] > -0.095].mean()
            cir_lift = mu_circ / max(mu_norm, 1e-9)
            if cir_lift >= 3.0:
                cir_pts = 1
    score += cir_pts
    breakdown.append((
        "Circuit-day signal", cir_pts,
        f"panic_keyword lift on circuit days = {cir_lift:.2f}x normal  -> {cir_pts}/1"
    ))

    # ---- Print report ----
    print()
    print("=" * 78)
    print(f"Sentiment Health Score: {score}/10")
    print("=" * 78)
    for name, pts, detail in breakdown:
        print(f"  {name:<25}  {pts:>2} pts   {detail}")
    print()

    if score >= 7:
        verdict = "STRONG"
    elif score >= 4:
        verdict = "MODEST"
    elif score >= 2:
        verdict = "WEAK"
    else:
        verdict = "MISSING"
    print(f"Verdict: {verdict}")
    print()

    print("Diagnosis:")
    if "n_articles" in df.columns:
        total_articles = df["n_articles"].fillna(0).sum()
        per_t_per_d = total_articles / max(
            df["ticker"].nunique() * df["date"].nunique(), 1
        )
        print(f"  total articles in training set:    {int(total_articles):,}")
        print(f"  avg articles per ticker per day:   {per_t_per_d:.4f}")
    if "panic_keyword_ratio" in df.columns:
        non_zero = (df["panic_keyword_ratio"].fillna(0) > 0).sum()
        print(f"  rows with panic_keyword_ratio > 0: {non_zero:,} ({non_zero/len(df):.2%})")
    if "promoter_news_flag" in df.columns:
        flag = (df["promoter_news_flag"].fillna(0) > 0).sum()
        print(f"  rows with promoter_news_flag = 1:  {flag:,} ({flag/len(df):.2%})")

    print()
    print("Recommendations:")
    if cov_pts < 2:
        print("  - Run scripts\\backfill_news.py to populate the lake's news dir.")
        print("  - Confirm config\\sources.yaml: news.newsapi.enabled: true")
        print("  - With Reddit enabled (Phase 24.18 default), retail sentiment is")
        print("    also pulled at inference time but isn't backfilled.")
    if ic_pts < 2 and "sent_mean" in df.columns:
        nz = (df["sent_mean"].fillna(0).abs() > 0.01).sum()
        if nz < 100:
            print(f"  - sent_mean has only {nz} non-zero rows in training. Same root")
            print("    cause as coverage above — backfill news first.")
    if cir_pts == 0:
        print("  - Circuit-day signal absent until news pipeline has weeks of history.")
        print("    Re-evaluate after running daily orchestrator for 2-3 weeks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
