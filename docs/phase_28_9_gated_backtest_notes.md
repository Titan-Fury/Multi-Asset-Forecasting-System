# Phase 28.9 — gated backtest is mis-calibrated (harness limitation)

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

`run_backtest` calls `walkforward_fit` directly with default params — it does
NOT apply the isotonic/beta calibration that the production trainer
(`train_universal_model`) wraps around the model. With TB labels the positive
("up") class is the 33% minority, so the *uncalibrated* `proba_up` is centered
near 0.33. The downstream blend/gate read that against a ~0.5 neutral point and
call almost everything "down", so the gate emits a 99.5%-short signal on an
index that rose over 2011–2026 — negative returns by construction.

## Decision

Do NOT conclude the deployed model loses money — this is a **harness
limitation, not a production result**. The production scoring path calibrates
probabilities before the gate, so deployed `proba_up` is correctly centered and
direction is not degenerate. The fix (make `run_backtest` calibrate its OOF
proba before the rubric/blend/gate, leak-safely across folds) is a behavioural
change to evaluation code with real design choices and is left for explicit
sign-off rather than an autonomous edit.

The honest, harness-independent read of deployed edge on this universe remains
the 28.8 finding: the raw model's realized-return decile spread is ~0, so any
real edge must come from the calibrated conviction layer — which this harness
cannot currently measure. Proper next step: evaluate the *logged* production
predictions in `data/lake/predictions/` via `scripts/daily_backtest.py`
(already calibrated, real deployed proba), or add calibration to `run_backtest`.
