# Phase 28.9 — gated backtest short-bias = blend() 0.5-neutral vs TB base rate (NO production bug)

## Why

28.8 left "run a full `run_backtest` gated-metric evaluation" as the way to
measure the *deployed* signal (the rubric + conviction gate the raw model
analysis excluded). This phase ran it and found the gated metrics are
**not trustworthy as-is** — and pinpointed why.

## What was run

`omni.backtest.runner.run_backtest` on the de-duplicated india training set
(205,283 rows, target `tb_label_5d`, 5 folds, min_train=2520,
run_shuffled_baseline off). Then a conviction-sweep + direction breakdown on
the emitted (gated) signals.

## Result

Gated summary: n_emitted=76,099 (37% emit rate), **directional_hit_rate=0.469**,
**mean_signed_fwd_ret=-0.0044** — i.e. the gated signal, traded as stated,
loses on average. Two diagnostics explain it:

Conviction sweep (5 quantile bins) — **flat, not monotone**:

| conv bin | conv range | n | hit_rate | mean_signed_ret |
| -------- | ---------- | -: | -------: | --------------: |
| 0 | 20.0–21.1 | 15,206 | 0.4688 | -0.00389 |
| 1 | 21.1–22.5 | 15,164 | 0.4655 | -0.00402 |
| 2 | 22.5–24.0 | 15,191 | 0.4736 | -0.00375 |
| 3 | 24.0–26.8 | 15,179 | 0.4689 | -0.00418 |
| 4 | 26.8–62.5 | 15,185 | 0.4753 | **-0.00600** |

Higher conviction does NOT raise the hit-rate (and the top bin is the worst on
signed return). Direction breakdown is the smoking gun:

| direction | n emitted | hit_rate | mean_signed_ret |
| --------- | --------: | -------: | --------------: |
| short (-1) | **75,709 (99.5%)** | 0.4705 | -0.00424 |
| long (+1) | 216 (0.5%) | 0.4306 | -0.04936 |

## Root cause

**CORRECTION (supersedes the first-pass diagnosis below).** The original note
blamed "missing calibration in `run_backtest`". That is WRONG. Reading
`xgboost_5d.walkforward_fit` (lines 394-464) shows it ALREADY calibrates each
fold leak-safely: inner-train fits XGBoost, the last-20% inner-val window fits
an isotonic/beta calibrator, which is applied to the test fold. The per-fold
`mean_proba≈0.33` is therefore *correct* calibration — it faithfully reflects
the 33% "up" base rate of the TB label (calibration preserves the base rate).

The true cause is a **base-rate / neutral-point mismatch in `ensemble/combine.py`
`blend()`**. Direction is set by `blended > 0.5 ± band`, where
`blended = w_model*proba_up + w_rubric*(rubric_total/100)`. The 0.5 neutral
assumes `proba_up` is centered at 0.5. For a 33%-positive TB target the
calibrated `proba_up` centers at 0.33, dragging `blended` below 0.5, so the
gate emits ~99.5% short. The bias comes from the **backtest target choice**
(`target=tb_label_5d`), not from any defect in calibration or in production.

## Decision

**There is no production bug.** Logged production predictions
(`data/lake/predictions/`) show india `proba_up` mean = 0.491 (≈0.5) and a
balanced, long-tilted direction split (+1: 227, 0: 146, -1: 107). The deployed
universal model is effectively centered on a ~balanced target, so `blend()`'s
0.5 neutral is appropriate and direction is healthy. The 99.5%-short was a
**backtest-configuration artifact** (forcing the 33%-positive TB target), not a
harness calibration gap and not a live failure. The 28.9 alarm was a FALSE
ALARM; this correction is the authoritative record.

**Latent fragility (optional hardening, not urgent):** `blend()`'s hardcoded
0.5 neutral is only safe while the universal model's positive base rate is
~50%. If the deployed universal model is ever retrained on a 33%-positive TB
target, the same short bias would appear live. A base-rate-aware neutral
(shift `proba_up` by `0.5 - base_rate`, or pass the training base rate into
`BlendConfig`) would make the blend robust to either target. This touches
production code, so it is offered as a decision, not an autonomous edit.

The harness-independent read of deployed edge on this universe remains the 28.8
finding: the raw model's realized-return decile spread is ~0, so any real edge
comes from the conviction layer. Measuring that properly needs a larger sample
of logged predictions via `scripts/daily_backtest.py` (only 480 rows / 15 dates
exist today — too sparse for a statistically meaningful read).

---

### Original first-pass diagnosis (kept for the record — SUPERSEDED above)

`run_backtest` calls `walkforward_fit` directly with default params — it does
NOT apply the isotonic/beta calibration that the production trainer
(`train_universal_model`) wraps around the model. With TB labels the positive
("up") class is the 33% minority, so the *uncalibrated* `proba_up` is centered
near 0.33. [SUPERSEDED: walkforward_fit does calibrate; the real cause is the
0.5 neutral in blend(), and production is unaffected.]
