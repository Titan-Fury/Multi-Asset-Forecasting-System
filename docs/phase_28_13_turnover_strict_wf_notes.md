# Phase 28.13 — turnover tuning (a) + strict walk-forward (b): the edge was mostly CV optimism

## (a) Turnover / cost tuning — cross-sectional L/S, xs-median model (PURGED CV)

| hold | top_q | gross Sharpe | net @15bps | net @5bps |
| ---- | ----- | -----------: | ---------: | --------: |
| 5d | 0.10 | 0.78 | -0.01 | 0.52 |
| 5d | **0.20** | **1.14** | 0.04 | **0.77** |
| 5d | 0.33 | 1.09 | -0.29 | 0.63 |
| 21d | 0.10 | 0.21 | 0.05 | 0.15 |
| 21d | 0.20 | 0.23 | -0.01 | 0.15 |
| 21d | 0.33 | 0.36 | 0.07 | 0.27 |

Findings: (1) the cross-sectional edge is a **5-day phenomenon** — stretching the
hold to 21 days to cut turnover *destroys* the gross edge (1.14 → ~0.3) far
faster than it saves cost, so "hold longer" is the wrong lever. (2) **Cost is
the binding constraint**: top/bottom-20% at 5-day hold is breakeven at a
conservative 15 bps round-trip-both-legs but reaches net Sharpe ~0.77 at ~5 bps
(plausible for liquid NIFTY names). This looked genuinely promising — until (b).

## (b) Strict walk-forward — the honest measurement

The numbers above use the default **bidirectional purged k-fold**, which trains
an early fold partly on later-dated rows (audit caveat #3). Re-ran the best
config under strict past-only expanding-window walk-forward (`OMNI_CV_MODE=strict`):

| CV mode | gross Sharpe | net @15bps | net @5bps | hit |
| ------- | -----------: | ---------: | --------: | --: |
| purged (bidirectional) | 1.135 | 0.042 | 0.771 | 56.8% |
| **strict walk-forward** | **0.112** | -1.002 | -0.259 | 53.4% |

**The gross Sharpe collapses 1.14 → 0.11.** ~90% of the apparent cross-sectional
edge was the model having "seen" later regimes during training, not real
predictive skill. Under honest past-only training the signal does not beat
costs (net negative even at 5 bps).

## Honest verdict

There is **no robust tradable edge** on this NIFTY-large-cap universe with the
current feature set, once measured the way live trading actually works
(past-only). This is consistent with, and sharper than, the earlier findings:
- 28.8: raw realized decile spread ~0;
- 28.11: directional "Sharpe" was beta + concentration;
- 28.12: a real but tiny cross-sectional IC (0.025) under purged CV;
- 28.13 (here): even that mostly evaporates under strict walk-forward.

This is not a defect to "fix" — it is the correct, sobering read of a hard
problem (cross-sectional daily-horizon prediction on liquid large-caps is close
to efficient). The value delivered is the **measurement honesty**: the harness
now exposes the truth instead of an inflated backtest.

## What would actually move the needle (evidence-based)

1. **Richer cross-sectional signal** (Tier-2): factor exposures, flow/
   microstructure (delivery%, FII/DII, OI), news volume — the IC is the
   bottleneck, and 0.025 is too low to beat costs.
2. **Universe / horizon where edge is less arbitraged**: mid/small-caps,
   intraday→multi-day, or event windows (PEAD) rather than liquid large-cap
   daily direction.
3. **Always validate strategy economics under `OMNI_CV_MODE=strict`** — purged
   k-fold is for AUC, strict WF is for P&L. Use the right tool per question.

## Shipped

- (a) was a parameter sweep (no code) — findings recorded here.
- (b) shipped: `strict_walkforward_splits` + `OMNI_CV_MODE` switch (`f5ab3e4`).
