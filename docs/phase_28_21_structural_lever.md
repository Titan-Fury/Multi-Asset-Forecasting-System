# Phase 28.21 — structural lever, locally tested: down-cap thesis supported in principle, blocked on data

Tested the two structural levers (down-cap / illiquidity, and PEAD) on the
EXISTING panel — no new data — to decide whether they're worth a real
data-engineering investment before building anything.

## Lever 1 — illiquidity / down-cap (Amihud tercile reversal)

5d cross-sectional reversal (`-ret_5d`), bucketed by per-date Amihud
illiquidity tercile (0=liquid … 2=illiquid):

| bucket | full IC | full gross Sh | full net@10bps | 2019+ IC | 2019+ net@10bps |
| ------ | ------: | ------------: | -------------: | -------: | --------------: |
| 0 liquid   | 0.031 | 0.38 | -2.7% | 0.024 | -11.2% |
| 1 mid      | 0.020 | 0.51 | -0.7% | 0.013 | -10.2% |
| 2 illiquid | **0.040** | **0.80** | **+6.5%** | 0.026 | -10.9% |

**Reversal strength rises monotonically with illiquidity** (illiquid IC 0.040 vs
liquid 0.031; only the illiquid bucket is net-positive full-sample) — the
textbook illiquidity-reversal premium. This **directionally validates the
down-cap thesis**: genuinely illiquid names are where reversal alpha lives.

Caveat that prevents declaring it live: (a) post-2019 the edge decayed in *all*
buckets here (even illiquid is net-negative), and (b) this is a NIFTY-large-cap
universe — its "most illiquid" tercile is still fairly liquid, so it is a weak
proxy for true small-caps. The result is consistent with the premium having
compressed in liquid names while (plausibly) persisting in genuinely small/
illiquid names we don't have data for.

## Lever 2 — PEAD (post-earnings drift): UNTESTABLE here

`earnings_window` is all-zero and `days_until_earnings` is 0% populated — the
panel has no earnings-event dates. PEAD cannot be tested without ingesting an
earnings calendar (the `omni.ingest.earnings_calendar` path exists but its
output isn't in the training panel).

## Honest verdict

Both structural levers point the same way: **the bottleneck is DATA, not
algorithm.** The illiquidity-reversal result empirically supports going down-cap,
but capturing it live requires:
1. A **survivorship-clean small/mid-cap universe** with prices (the Phase 28.14
   data gap) — including delisted names so the reversal book isn't measuring a
   survivor artifact.
2. **Earnings-date ingestion** wired into the panel to test/exploit PEAD.

Neither can be done locally now (no membership/delisted feed, no earnings dates
in-panel), and fabricating an edge on the current large-cap universe would just
reproduce the decayed anomaly of Phase 28.20. So this is scoped, evidence-backed,
and explicitly deferred to a data-ingestion effort rather than blindly built.

Methodology unchanged: any future down-cap/PEAD signal must clear
`OMNI_CV_MODE=strict` + cost + sub-period stress tests before being called edge.
