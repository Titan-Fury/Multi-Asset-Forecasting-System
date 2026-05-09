# Revolutionary Ideas — How to Actually Boost Accuracy

> The current universal AUC is **~0.508**. The stack lifts it to
> **~0.518**. To get to a regime where this is actually tradable
> at scale (target: AUC ≥ 0.55, top-decile precision ≥ 65 %), you
> need ideas from outside the standard "more features, deeper
> trees" playbook.
>
> This document ranks 25 high-leverage ideas by **expected impact ×
> implementation cost**. Tier S = revolutionary, Tier A = high
> impact, Tier B = solid improvements. Each has a realistic AUC
> uplift estimate (where measurable) and a code-cost estimate.

---

## Tier S — revolutionary, paradigm-shift ideas

### S1. Order-flow / market-microstructure features
**Idea.** Replace daily OHLCV with intraday Level-1 (best bid/ask) or
Level-2 (full order book) snapshots. Compute:
- **VPIN** (Volume-synchronised Probability of Informed trading)
- **Order-flow imbalance** (OFI): signed volume in the top of book
- **Kyle's lambda** (price impact per unit volume)
- **Realised spread** vs effective spread
- **Trade-size distribution** (institutional vs retail)

**Why it's revolutionary.** Daily features are exhausted as alpha
sources — every quant in the world has them. Microstructure is
where real edge still exists at higher frequency.

**Expected uplift.** AUC 0.51 → 0.56 for 1-day horizon.
**Cost.** High. Need a Polygon.io / Databento subscription
(~$200/mo) or NSE direct feed. Storage ~50 GB/month. Different
modelling approach (LSTM / Transformer for sequence).

---

### S2. Replace XGBoost with a **Temporal Fusion Transformer** (TFT)
**Idea.** TFTs (Lim et al. 2021) handle multi-horizon, multi-asset
time-series natively with attention over both static covariates
(asset_class, sector) and dynamic ones (price history, news
sentiment). They also produce calibrated quantile forecasts.

**Why it's revolutionary.** The current model treats each row
independently. TFT models the *sequence*, so it can pick up regime
transitions, post-earnings drift, momentum exhaustion, etc.

**Expected uplift.** AUC 0.518 (stack) → 0.55–0.58.
**Cost.** Medium. PyTorch Forecasting library has a ready
implementation. Need GPU (RTX 4070 Super is fine). Training time
~30 min per epoch on 623k rows.

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
# replace stack.train_horizon() entirely with:
tft = TemporalFusionTransformer.from_dataset(
    training, hidden_size=64, attention_head_size=4,
    dropout=0.1, learning_rate=0.03,
)
trainer.fit(tft, ...)
```

---

### S3. **News embeddings as features**, not just sentiment scores
**Idea.** Right now news → FinBERT → 1 sentiment scalar per article.
Throw the scalar away. Use the **last-hidden-layer 768-dim
embedding** as the article's representation. Aggregate per-day per-
ticker via attention pooling. Concatenate with the existing
features.

**Why it's revolutionary.** Sentiment scalar discards 99 % of the
information. Embeddings preserve topical content (lawsuit vs
earnings beat vs analyst downgrade), which often matters more
than positive/negative tone. This is what hedge funds like Two
Sigma and AQR have been using for ~5 years.

**Expected uplift.** AUC 0.518 → 0.55 on news-rich tickers.
**Cost.** Medium. Already running FinBERT — switch
`sentiment_per_article` to return embeddings, store as a
768-column `article_embeds.parquet`, mean-pool per
(date, ticker). Tree models handle the dimensionality fine if
you use SVD to compress to 32 components first.

---

### S4. Cross-asset **graph attention** (correlation-aware features)
**Idea.** Build a daily correlation graph of all 195 instruments.
At inference, propagate node features (price momentum, sentiment)
through 2 graph-attention layers. The prediction for RELIANCE is
informed by what's happening at TCS, ICICIBANK, NIFTY, etc.

**Why it's revolutionary.** Stocks don't move in isolation. A
sudden +5 % move in HDFC Bank leaks into RELIANCE through the
NIFTY. A graph-attention network captures this as a learnable
relation, not a hand-engineered "spread vs benchmark" feature.

**Expected uplift.** AUC ~0.52 → 0.55 in stress regimes
(when correlations spike).
**Cost.** Medium. PyTorch Geometric or DGL. ~200 lines of model
code. The existing correlation-graph in the dashboard is the
data structure you already need — just promote it from
visualisation to features.

---

### S5. **Online learning** with concept-drift detection
**Idea.** Replace the weekly batch retrain with a **rolling window
+ ADWIN drift detector**. When ADWIN flags a distribution change
(e.g. regime shift, post-COVID bull → bear), trigger an
immediate retrain on the most recent 6 months instead of waiting
for the next Sunday.

**Why it's revolutionary.** Markets non-stationarity is the #1
killer of quant models. The current weekly schedule gives a
typical signal a 3.5-day average staleness. Adaptive
retraining cuts this to ~1 day during regime shifts and saves
compute during stable periods.

**Expected uplift.** Difficult to measure directly; prevents the
2-week post-regime-shift dead zone. AUC stable around 0.518
instead of dipping to 0.49 after each regime change.
**Cost.** Low. `river.drift.ADWIN` is a 3-line addition in
`omni/orchestrator/daily.py`.

```python
from river.drift import ADWIN
drift = ADWIN()
for x in stream_of_features:
    drift.update(x)
    if drift.drift_detected:
        retrain_now()
```

---

## Tier A — high-impact, established techniques we don't yet use

### A1. **Meta-labelling** (López de Prado AFML §3.6)
**Idea.** Train a *primary* model to call direction (current
universal), then a **secondary** binary model to filter "is the
primary's prediction reliable?". Only emit signals when both
agree at high confidence.

**Why.** Cuts false positives dramatically — at the cost of
trade frequency. Top-decile precision typically jumps 5–10
percentage points.

**Expected uplift.** Top-decile precision 60 % → 68 %.
**Cost.** Low. Extra XGBoost on a residual target. ~150 lines.

---

### A2. **Triple-Barrier labelling** (AFML §3.4)
**Idea.** Replace `up_5d` (will close 5 days from now be higher)
with a path-dependent label: did the price hit `+2 σ profit-take`
before it hit `−1 σ stop-loss` within 5 days? This produces
**asymmetric** labels that better match what a real trader cares
about.

**Why.** Path matters. A stock that goes -10 % → +12 % in 5 days
is a winning trade only if you have the position size + stop to
survive the drawdown.

**Expected uplift.** AUC unchanged but **economic significance**
of top-decile signals 1.5–2× higher in backtest.
**Cost.** Low. Replace `omni/features/targets.py::build_targets`
with `triple_barrier()`. ~40 lines.

---

### A3. **Information-driven bars** (AFML §2.3) instead of time bars
**Idea.** Instead of "1 day", define a bar as "every 50 ticks of
volume" or "every 0.5 % price move". Bars become approximately
homoscedastic (constant volatility), which is exactly what most
ML models want as input.

**Why.** Daily bars during a calm 2017 are nothing like daily
bars during March 2020. Volume bars normalise this.

**Expected uplift.** AUC 0.51 → 0.53 in stress regimes specifically.
**Cost.** Medium. Need to refactor the entire feature pipeline
because indices are no longer dates. Two weeks of work.

---

### A4. **Cross-validation with combinatorial purged K-fold (CPKCV)**
**Idea.** Current purged K-fold gives one OOF curve. CPKCV
generates **multiple** OOF curves by leave-multiple-folds-out,
producing a distribution over OOF AUCs. Use this distribution
to pick model hyper-parameters that are robust, not just lucky.

**Why.** Single-fold OOF is high-variance with our small effective
sample size (~5,000 independent trading days).

**Expected uplift.** Same realised AUC, but ~30 % less
overfitting risk on the model selection step.
**Cost.** Low. Extension of existing `omni/cv/purged_kfold.py`.

---

### A5. **Position sizing via Kelly fraction**
**Idea.** Right now the dashboard says "LONG / FLAT / SHORT".
Add **how much** to bet using Kelly: `f = (p × b - q) / b` where
p = blended_p, q = 1-p, b = expected upside / expected downside
(read off the conformal band).

**Why.** A 0.55 prediction with conformal band [-2 %, +5 %] should
get 4× the position of a 0.55 prediction with band [-8 %, +9 %].
Without this, a portfolio sums every signal at equal size and
gives up most of its edge.

**Expected uplift.** Sharpe 0.6 → 1.1 on backtest.
**Cost.** Low. ~30 lines in `omni/ensemble/pipeline.py`.

---

### A6. **Conditional drift gating**
**Idea.** Build a separate classifier that predicts whether the
universal model will be **right** for this specific row. Use it
as a meta-gate: don't emit a signal when the meta says "this
row is unreliable" (high feature-distance from training data,
in a regime with thin training samples, etc.).

**Why.** Concept-drift is local, not global. Most rows are fine;
some are dangerous. A learned gate catches the dangerous ones.

**Expected uplift.** Top-decile precision 60 % → 67 %.
**Cost.** Medium. Train an isolation-forest / one-class SVM on
the training feature distribution. ~200 lines.

---

### A7. **Feature importance via SHAP — and surgically prune**
**Idea.** Run SHAP on the trained model. The 59 features have
~30 with permutation-importance below noise. Drop them. Add
that headroom back as **interaction features** (e.g.
`rsi × regime`, `news_sent × volume_z`).

**Why.** Tree models are robust to noise features in theory,
but in practice 30 noise features at our sample size eat 1 %
of effective AUC.

**Expected uplift.** AUC 0.508 → 0.515 — small but free.
**Cost.** Low. SHAP is `pip install shap`. One afternoon.

---

### A8. **Snowflake Arctic / Cortex AI for synthetic feature generation**
**Idea.** Use Cortex's `COMPLETE` function with a structured
prompt: *"given these last-30-days OHLCV + news headlines for
RELIANCE, what 3 forward-looking factors should an investor
weigh? Respond as JSON."* Parse the JSON, embed each factor,
use as a feature.

**Why.** LLMs catch idiosyncratic things (lawsuit, regulatory
change, supply-chain shift) that no scalar feature captures.
Snowflake Arctic Search is purpose-built for this and runs at
$0.001/1k tokens.

**Expected uplift.** Hard to measure ex-ante; ~+0.005 AUC in
backtest of similar work in equities.
**Cost.** Low. Already have Cortex backend wired up.

---

## Tier B — solid, lower-risk improvements

### B1. **Earnings-event isolation**
Train a separate model on the 3 trading days surrounding earnings
release. Earnings are a different beast (gap risk, post-earnings
drift); a specialist model on this slice usually beats the
universal there.

### B2. **Sector-relative features** ahead of absolute
Replace `pe_ratio` with `pe_ratio_z_in_sector`, `roe` with
`roe_z_in_sector`. Cancels out sector-wide moves that the
model can't predict anyway.

### B3. **NSE F&O OI as a leading indicator**
We fetch `nse_oi.py` features but only use `oi_pcr` and
`max_pain`. Add: rate of change of OI on top 5 strikes,
implied-volatility surface skew, gamma exposure (GEX).
Options markets often lead spot by 1–2 days.

### B4. **Turnover-adjusted news weighting**
A negative news article on RELIANCE matters more than on a
small-cap nobody trades. Weight news contributions by
`log(20-day average turnover)`.

### B5. **Cross-listing arbitrage as a feature**
For dual-listed names (INFY on NSE + NYSE, RELIANCE on NSE +
NSE-Cash), the spread between their FX-adjusted prices is a
**known-good** mean-reverting signal. ~+5 bps of monthly alpha
in published research.

### B6. **Bayesian credible intervals via MAPIE**
Replace split-conformal with `mapie-conformal-classification`'s
`CrossConformalClassifier` for tighter intervals at the same
coverage. ~10 % narrower bands at the same 80 % coverage.

### B7. **Model ensembling across random seeds**
Train the same XGBoost 5 times with different seeds and average
the calibrated probabilities. The variance reduction alone is
worth ~+0.003 AUC.

### B8. **Distillation: small, fast student model**
Train a tiny LightGBM (8 leaves, 100 trees) to mimic the stack's
predictions. The student is **20× faster** at inference, so the
dashboard can score more tickers in parallel. Useful for the
"Top movers" tab and any future scanning UI.

### B9. **Sentiment by source weight**
Not all sources are equal. Weight Bloomberg / Reuters > MoneyControl >
Telegram. Train these weights from how much they move next-day
returns historically.

### B10. **Market-microstructure proxies from minutely data**
We pull intraday but barely use it. Compute:
- Amihud illiquidity ratio (5-min returns / 5-min dollar volume)
- VWAP deviation
- Closing auction imbalance
These work at daily frequency without needing tick data.

### B11. **Macro nowcasting features**
Ingest the **release calendar** (PMI, CPI, Fed funds rate). A
feature `days_until_next_macro_release` lets the model attenuate
its conviction near data releases (Fed days are unpredictable).

### B12. **Pseudo-labelling for low-coverage tickers**
For small-caps with thin training data (OMAXAUTO etc.), use the
universal model's predictions as **pseudo-labels** to bootstrap
a sub-model for that name specifically. Self-training with high
confidence threshold.

---

## What I'd implement *next*, in order

1. **A5: Kelly position sizing** — 30 lines, biggest realised-Sharpe gain.
2. **A2: Triple-barrier labelling** — 40 lines, fixes economic significance.
3. **A1: Meta-labelling** — 150 lines, big top-decile precision lift.
4. **S3: News embeddings** — 200 lines + a 4 GB embedding store, biggest AUC lift.
5. **A6: Conditional drift gating** — 200 lines, complements meta-labelling.
6. **S5: ADWIN drift detector** — 30 lines, prevents post-regime-shift dead zones.
7. **S2: Temporal Fusion Transformer** — 800 lines + 1 week retraining, biggest *theoretical* ceiling.
8. **S1: Order-flow microstructure** — needs paid data feed; defer until budget exists.

Items 1–3 alone would, at the same OOF AUC of 0.518, take the
expected backtested Sharpe from ~0.6 to ~1.4. That is the
difference between "interesting research" and "actually
deployable on real capital."
