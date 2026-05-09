"""Kelly-criterion position sizing for the on-demand prediction layer.

Given a PredictionCard's blended_p (probability the trade wins) and the
expected upside / downside read off the conformal interval, return a
recommended fraction of capital to allocate.

Why Kelly?
    Equal-sizing every signal throws away the model's edge. A 0.55
    prediction with a tight ±2% conformal band carries far more
    expected utility per unit of risk than a 0.55 prediction with a
    ±9% band — Kelly captures that explicitly.

Math:
    f = (p * b - q) / b
        where p = win probability
              q = 1 - p
              b = win/loss ratio = expected_upside / expected_downside

We use HALF-Kelly (f / 2) to be robust to estimation error in p, and
cap at MAX_FRACTION = 0.10 (10% of capital per trade) to enforce
diversification regardless of what the model says.

Reference:
    Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and
    the Stock Market" (1997).
"""
from __future__ import annotations

from dataclasses import dataclass

# Defaults follow the literature on Kelly application to equities:
#   - Half-Kelly is robust to ~5% error in win-probability estimates.
#   - 10% cap prevents over-concentration in any single signal.
HALF_KELLY = 0.5
MAX_FRACTION = 0.10


@dataclass(frozen=True)
class KellyResult:
    fraction: float          # [0, MAX_FRACTION], recommended allocation
    full_kelly: float        # uncapped, un-halved fraction (for diagnostics)
    edge: float              # p*b - q (positive = bet, 0 or neg = pass)
    win_prob: float
    win_loss_ratio: float
    side: str                # "long" | "short" | "flat"
    rationale: str


def kelly_size(
    blended_p: float,
    expected_upside_pct: float,
    expected_downside_pct: float,
    *,
    direction: int = 0,
    half_kelly: bool = True,
    cap: float = MAX_FRACTION,
) -> KellyResult:
    """Compute the recommended position-size fraction.

    Args:
        blended_p: probability of UP (in [0, 1]).
        expected_upside_pct: positive number, e.g. 5.0 for +5%.
        expected_downside_pct: positive number, e.g. 3.0 for -3%.
        direction: -1 / 0 / 1 from the prediction card; 0 = no-bet.
        half_kelly: if True, halve the raw Kelly fraction.
        cap: maximum fraction returned regardless of math.

    Returns:
        KellyResult; fraction = 0 if no edge / wrong-side / FLAT.
    """
    if direction == 0:
        return KellyResult(0.0, 0.0, 0.0, blended_p, 0.0, "flat",
                            "Direction is FLAT - no bet.")

    # Convert into win-probability for the side we are taking
    p = blended_p if direction > 0 else 1.0 - blended_p
    q = 1.0 - p

    upside = max(float(expected_upside_pct), 1e-6)
    downside = max(float(expected_downside_pct), 1e-6)
    b = upside / downside

    edge = p * b - q                              # numerator of Kelly
    if edge <= 0:
        return KellyResult(0.0, 0.0, edge, p, b,
                            "long" if direction > 0 else "short",
                            f"No Kelly edge (p*b - q = {edge:.3f}); skip.")

    full = edge / b                                # raw Kelly fraction
    f = full * (HALF_KELLY if half_kelly else 1.0)
    f_capped = max(0.0, min(f, cap))

    rationale = (
        f"p={p:.3f}, b={b:.2f}, edge={edge:.3f} -> "
        f"raw Kelly {full:.2%}, "
        f"{'half-' if half_kelly else 'full-'}Kelly {f:.2%}, "
        f"capped at {cap:.0%} -> {f_capped:.2%}"
    )
    return KellyResult(
        fraction=f_capped,
        full_kelly=full,
        edge=edge,
        win_prob=p,
        win_loss_ratio=b,
        side="long" if direction > 0 else "short",
        rationale=rationale,
    )


def kelly_from_card(card_dict: dict) -> KellyResult:
    """Convenience wrapper: take a PredictionCard.to_dict() and compute Kelly.

    Reads:
        blended_p, direction, expected_high_pct, expected_low_pct
    """
    blended_p = float(card_dict.get("blended_p", 0.5))
    direction = int(card_dict.get("direction", 0) or 0)
    # Conformal band: high is positive, low is negative
    upside = abs(float(card_dict.get("expected_high_pct", 0.0)))
    downside = abs(float(card_dict.get("expected_low_pct", 0.0)))
    if upside < 0.1 and downside < 0.1:
        # Card has no conformal band yet; assume symmetric ±5%
        upside = downside = 5.0
    return kelly_size(blended_p, upside, downside, direction=direction)
