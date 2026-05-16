"""Verify learned-rubric blend math + scorer integration."""
import os
import json
import tempfile
import pandas as pd

from omni.rubric.learned import blend_weights, save_learned_weights, load_learned_weights, clear_cache
from omni.rubric.scorer import score_row

# --- 1. blend math
yaml_w = {"technical": 0.20, "fundamental": 0.10, "sentiment": 0.10,
          "macro": 0.10, "microstructure": 0.10, "oi": 0.10,
          "relative_strength": 0.10, "insider": 0.10, "earnings_momentum": 0.10}
learned_w = {"technical": 0.40, "fundamental": 0.05, "sentiment": 0.20,
             "macro": 0.05, "microstructure": 0.05, "oi": 0.05,
             "relative_strength": 0.10, "insider": 0.05, "earnings_momentum": 0.05}
print("YAML sum:", sum(yaml_w.values()))
print("learned sum:", sum(learned_w.values()))
b = blend_weights(yaml_w, learned_w, alpha=0.5)
print("blended (alpha=0.5):", {k: round(v, 3) for k, v in b.items()})
print("blended sum:", round(sum(b.values()), 6))

# --- 2. wire-through: opt-in flag flips weights
clear_cache()
fake = {"bull":   learned_w,
         "bear":   learned_w,
         "chop":   learned_w,
         "crisis": learned_w}
import pathlib
p = pathlib.Path("data/models/learned_rubric_weights.json")
p.parent.mkdir(parents=True, exist_ok=True)
existed = p.exists()
if existed:
    backup = p.read_text()
save_learned_weights(fake, p)

row = pd.Series({
    "ret_1d": 0.01, "ret_5d": 0.05, "ret_20d": 0.15,
    "rsi_14": 70.0, "macd_hist_pct": 0.01,
    "trend_strength": 0.05, "vol_ratio": 1.2,
    "dist_sma_20": 0.03, "dist_sma_50": 0.06, "dist_sma_200": 0.10,
    "dist_from_52w_high": -0.02, "dist_from_52w_low": 0.40,
    "atr_pct_14": 0.018, "asset_class": "india",
})

os.environ.pop("OMNI_LEARNED_RUBRIC", None)
clear_cache()
r1 = score_row(row, regime="bull", asset_class="india")
print(f"\n[OFF] rubric_total = {r1.rubric_total:.2f}")

os.environ["OMNI_LEARNED_RUBRIC"] = "1"
clear_cache()
r2 = score_row(row, regime="bull", asset_class="india")
print(f"[ON ] rubric_total = {r2.rubric_total:.2f}")

# Cleanup
if existed:
    p.write_text(backup)
else:
    p.unlink(missing_ok=True)
os.environ.pop("OMNI_LEARNED_RUBRIC", None)
clear_cache()
