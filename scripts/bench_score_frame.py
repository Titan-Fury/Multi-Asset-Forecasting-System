"""Verify score_frame correctness + measure speedup on a synthetic load."""
import time

import numpy as np
import pandas as pd

from omni.rubric.scorer import score_frame

rng = np.random.default_rng(0)
n = 5000
df = pd.DataFrame({
    "ticker": [f"T{i % 50}" for i in range(n)],
    "asset_class": rng.choice(["us", "india", "crypto"], n),
    "ret_1d": rng.normal(0, 0.02, n),
    "ret_5d": rng.normal(0, 0.04, n),
    "ret_20d": rng.normal(0, 0.07, n),
    "rsi_14": rng.uniform(20, 80, n),
    "macd_hist_pct": rng.normal(0, 0.01, n),
    "trend_strength": rng.normal(0, 0.05, n),
    "vol_ratio": rng.uniform(0.5, 2.0, n),
    "dist_sma_50": rng.normal(0, 0.05, n),
    "dist_sma_200": rng.normal(0, 0.10, n),
    "dist_from_52w_high": rng.uniform(-0.5, 0, n),
    "dist_from_52w_low": rng.uniform(0, 0.5, n),
    "atr_pct_14": rng.uniform(0.005, 0.05, n),
    "pe_ratio": rng.uniform(10, 40, n),
    "roe": rng.uniform(0.05, 0.30, n),
    "debt_to_equity": rng.uniform(0, 2.5, n),
    "revenue_growth": rng.normal(0.10, 0.20, n),
    "n_articles": rng.integers(0, 30, n),
    "sent_mean": rng.normal(0, 0.3, n),
    "vix_level": rng.uniform(12, 35, n),
})
regimes = pd.Series(rng.choice(["bull", "chop", "bear", "crisis"], n), index=df.index)

t0 = time.perf_counter()
scored = score_frame(df, regimes=regimes)
t = time.perf_counter() - t0
print(f"score_frame n={n}: {t*1000:.0f}ms ({n/t:.0f} rows/s)")

# Sanity: 11 score columns added
new_cols = [c for c in scored.columns if c not in df.columns]
print("new cols:", new_cols)
assert "rubric_total" in scored.columns
assert len(scored) == n
assert scored["rubric_total"].between(0, 100).all(), "totals out of range"
assert scored["rubric_direction"].isin([-1, 0, 1]).all()
print("sanity: OK")
print(f"sample dirs: {dict(scored['rubric_direction'].value_counts().to_dict())}")
