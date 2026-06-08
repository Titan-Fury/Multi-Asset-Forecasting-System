# Phase 28.18 — CSRankNorm (cross-sectional feature ranking) — null result

Tested 28.17 item #1 (qlib CSRankNorm: replace every model feature with its
per-date cross-sectional percentile) on the strict walk-forward cross-sectional
long/short book — the only honest metric (Phase 28.13).

| config | mean IC | gross Sharpe | net @5bps (ann) | hit |
| ------ | ------: | -----------: | --------------: | --: |
| baseline (raw features) | 0.0171 | -0.26 | -3.5% | 53.4% |
| CSRankNorm (all features) | 0.0118 | -0.17 | -2.4% | 53.5% |

## Read

CSRankNorm does **not** improve the signal here. IC actually *drops*
(0.017 → 0.012); gross Sharpe is marginally less-negative but still negative and
within noise; hit-rate is flat. Not wired into production.

Why it doesn't help (honest): this universe already feeds the model
ticker-invariant, unitless features, and a curated subset is *already*
rank-normalised (`add_cross_sectional_ranks`). Re-ranking everything mostly
discards magnitude information that the trees were using, and the binding
constraint is not feature representation — it is the **absence of exploitable
cross-sectional alpha in liquid NIFTY large-cap daily direction** under honest
(past-only) evaluation. Every lever this session converges on the same wall
(28.8, 28.11, 28.12, 28.13).

## Implication for items #2/#3 (28.17)

Label-uniqueness weighting and robust train-only scaling are legitimate and
worth keeping on the roadmap, but the evidence says no preprocessing/weighting
change will manufacture edge on this universe/horizon. The honest path to
"better scores" is a structural change — a less-efficient universe (mid/small
cap), a different horizon/event window (PEAD), or genuinely new data — not more
machinery on the current signal. Documented rather than pursued, to avoid
p-hacking a result that strict WF keeps refuting.

The durable wins this session were correctness/robustness, not alpha:
sanitizer, dedup, base-rate-aware blend, strict-WF measurement, cost-aware
Sharpe, leakage audit, security fixes, dead-code removal, lint/security CI gate.
