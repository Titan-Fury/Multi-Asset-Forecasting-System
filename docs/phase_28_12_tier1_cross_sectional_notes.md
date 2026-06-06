# Phase 28.12 — Tier-1 cross-sectional target + two leakage lessons

## Why

`prediction_efficiency_research.md` Tier-1: the universal model predicts
*absolute* per-ticker direction, but the learnable equity alpha lives in
*relative* (cross-sectional) ranking. This phase tests whether training on a
cross-sectional target and trading a market-neutral long/short book beats the
absolute-direction approach — and, in doing so, surfaced two leakage traps.

## Method

On the fixed india panel (205,283 rows, 59 tickers), build a cross-sectional
median label `fwd_dir_xsmed_5d = (within-date percentile rank of fwd_ret_5d >
0.5)`. Train the universal walk-forward model on (a) the absolute `tb_label_5d`
and (b) the cross-sectional label. Score each with the new market-neutral
`simulate_cross_sectional_ls` (long top-20%, short bottom-20% by model proba,
rebalanced every 5 days, 15 bps round-trip both legs).

## Leakage lesson #1 — a leak in the experiment itself (caught live)

First run reported Sharpe ~21–58 with the model capturing 57% of
perfect-foresight spread — impossible for a 0.53-AUC model. Diagnosis: the
helper label was named `xs_up_5d`, which does NOT match the trainer's target
prefixes (`fwd_ret_/fwd_dir_/fwd_q_/fwd_rank_/tb_label_/tb_first_`), so
`prepare_xy` swept it into the FEATURE matrix — a direct function of the future
return. Classic target leakage, introduced by the experiment harness, not the
model. Renaming it `fwd_dir_xsmed_5d` (a recognized prefix) excluded it from
features. Cross-sectional IC fell from absurd to **0.0254** (a realistic value).
Lesson: any label-derived column must carry a target prefix or it leaks; the
cross-sectional IC / perfect-foresight ratio is a fast leak detector.

## Leakage / methodology lesson #2 — overlapping-return Sharpe inflation

The portfolio simulator amortized each 5-day trade into 5 equal daily legs,
fabricating a smooth daily P&L series and inflating Sharpe (28.11's +1.71;
cross-sectional ~58). Fixed (`67e405c`) by scoring NON-OVERLAPPING 5-day periods
and annualizing by `sqrt(252/5)`. Validated: random signal → ~0 net; perfect →
finite sane upper bound.

## Result (leak-free, non-overlapping, after fixes)

Cross-sectional long/short, scored on realized `fwd_ret_5d`:

| training target | net Sharpe (15bps) | **gross Sharpe** | hit | gross ann |
| --------------- | -----------------: | ---------------: | --: | --------: |
| absolute `tb_label_5d` | -0.14 | 0.92 | 54.3% | +13.1% |
| **cross-sectional median** | **+0.04** | **1.14** | 56.8% | +15.7% |
| RANDOM (1 seed) | -0.73 | 0.58 | 53.0% | +6.7% |
| PERFECT (upper bound) | 25.3 | 26.1 | 100% | +480% |

Directional book for contrast (28.11 corrected): base-rate-ON Sharpe +0.74 but
`avg_trade_ret` negative — beta, not alpha.

## Read

- **Tier-1 thesis holds, modestly.** The cross-sectional target beats absolute
  direction for a market-neutral book (gross Sharpe 1.14 vs 0.92; net +0.04 vs
  -0.14), and there is a small but REAL cross-sectional edge (IC 0.025, gross
  Sharpe ~1.1, clearly above the ~0.58 single-seed random).
- **Not yet net-profitable.** At 5-day full-turnover with 15 bps round-trip on
  both legs, costs (~15%/yr drag) eat almost the entire ~13–16% gross. Net is
  ~breakeven. The edge is real but thin.
- **Levers to make it net-positive (next):** lower turnover (longer hold /
  rebalance less often / only trade when rank changes materially), reduce cost
  assumption for liquid names, raise IC with cross-sectional/factor features
  (Tier-2), and average the random baseline over seeds for a cleaner null.

## Status

`simulate_cross_sectional_ls` shipped in `omni/backtest/portfolio.py`. The
cross-sectional label is an experiment artifact (not yet wired into the
production training target); promoting it to a production target is the logical
follow-up once turnover/cost tuning shows a positive net Sharpe.
