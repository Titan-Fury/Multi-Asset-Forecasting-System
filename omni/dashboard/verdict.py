"""Plain-English verdict for prediction cards.

The dashboard already shows direction (LONG/SHORT/FLAT), conviction
0-100 and blended_p — but those are jargon. Traders / casual users
need a single sentence that captures:

  1. How confident the *model* is (proba_up alone, before rubric blend)
  2. Whether the rubric agrees or disagrees with the model
  3. What the gate decided
  4. Whether the user should act, watch, or sit out

The empirical CIANAGRO replay (May 8 close) is the motivating example:

  - rubric pillars said tech=80, sent=75, rs=60 → bullish lean
  - quant model said proba_up=0.323 → strongly bearish
  - blended fell to 0.44 → bearish lean, conviction 12
  - gate (min_conv=15) blocked

Realised h+5 = -22.62%. The quant call was correct but the rubric
disagreement diluted conviction below the gate. The user-facing
display should call this out explicitly: "Model bearish, technicals
still hot — wait for confirmation OR trust the quant".

API:
    v = verdict(card_dict)
    # v.headline      -> "Strongly Bearish" / "Bearish Lean" / ...
    # v.summary       -> 1-line plain-English explanation
    # v.disagreement  -> "" if pillars agree, otherwise a sentence
    # v.tone          -> "long" / "short" / "flat" / "warn"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    headline: str          # 1-3 words: "Strongly Bullish", "Sit Out", ...
    summary: str           # one-sentence plain-English explanation
    disagreement: str      # "" or a sentence describing rubric/quant split
    tone: str              # "long" / "short" / "flat" / "warn"


def _label(proba_up: float, conviction: float) -> tuple[str, str]:
    """Map (proba_up, conviction) to a plain-English headline + tone."""
    p = float(proba_up)
    c = float(conviction)

    # Hard sit-out: conviction below 8 (very low) regardless of direction
    if c < 8:
        return "Sit Out", "flat"

    # Strongly bullish: proba >= 0.65 AND conviction >= 30
    if p >= 0.65 and c >= 30:
        return "Strongly Bullish", "long"
    # Strongly bearish: proba <= 0.35 AND conviction >= 30
    if p <= 0.35 and c >= 30:
        return "Strongly Bearish", "short"

    # Cautious tones — direction set but conviction or proba softer
    if p >= 0.55:
        return ("Cautious Bullish" if c < 25 else "Bullish"), "long"
    if p <= 0.45:
        return ("Cautious Bearish" if c < 25 else "Bearish"), "short"

    # Near 0.5 — no edge
    return "Inconclusive", "flat"


def _rubric_lean(pillars: dict) -> int:
    """Return -1 / 0 / +1 indicating rubric pillars' net lean.

    Uses the simple-average of (technical, sentiment, relative_strength,
    earnings_momentum). >= 60 -> bullish; <= 40 -> bearish; else neutral.
    """
    if not pillars:
        return 0
    keep = ("technical", "sentiment", "relative_strength", "earnings_momentum")
    vals = [float(pillars.get(k, 50.0)) for k in keep
             if pillars.get(k) is not None]
    if not vals:
        return 0
    avg = sum(vals) / len(vals)
    if avg >= 60:
        return 1
    if avg <= 40:
        return -1
    return 0


def _quant_lean(proba_up: float) -> int:
    if proba_up >= 0.55:
        return 1
    if proba_up <= 0.45:
        return -1
    return 0


def _disagreement_phrase(rubric_lean: int, quant_lean: int,
                          pillars: dict | None) -> str:
    """Build a one-sentence note when the rubric and the quant disagree."""
    if rubric_lean == 0 or quant_lean == 0:
        return ""
    if rubric_lean == quant_lean:
        return ""
    pillars = pillars or {}
    # Find the strongest constructive (>= 60) and weakest (<= 40) pillars
    # so the message is concrete instead of generic.
    strong = sorted(
        ((float(v), k) for k, v in pillars.items() if v is not None and float(v) >= 60),
        reverse=True,
    )[:2]
    weak = sorted(
        ((float(v), k) for k, v in pillars.items() if v is not None and float(v) <= 40),
    )[:2]

    def _fmt(items):
        return ", ".join(f"{k.replace('_', ' ')} {int(v)}" for v, k in items)

    if quant_lean < 0 and rubric_lean > 0:
        # CIANAGRO May 8 case: model bearish, fundamentals/technicals still hot
        why = _fmt(strong) if strong else "fundamentals"
        return (f"Model sees a turn lower while {why} still look strong "
                 f"-- classic late-cycle warning, treat any longs with care.")
    if quant_lean > 0 and rubric_lean < 0:
        why = _fmt(weak) if weak else "sentiment"
        return (f"Model sees an upturn while {why} still drag "
                 f"-- early-cycle setup, wait for confirmation before sizing up.")
    return ""


def _summary_phrase(headline: str, tone: str, *,
                     proba_up: float, conviction: float,
                     passes: bool, gate_min: float | None,
                     direction: int, regime: str | None,
                     model_kind: str,
                     quant_lean: int = 0) -> str:
    p = float(proba_up)
    c = float(conviction)
    rg = (regime or "").lower()
    rg_clause = f" ({rg} regime)" if rg else ""

    if not passes:
        gate_str = f" (need {gate_min:.0f}+)" if gate_min else ""
        if c < 8:
            return ("No edge today - sit this one out"
                     f"{rg_clause}. Conviction {c:.0f} is too low to justify a trade{gate_str}.")
        # Use quant_lean (raw model direction) so the warn tone still
        # produces the right bearish/bullish summary.
        if quant_lean < 0 or tone == "short":
            return (f"Model leans bearish (proba_up {p:.2f}) but conviction {c:.0f} "
                     f"is below the gate{gate_str} - watch, don't act.")
        if quant_lean > 0 or tone == "long":
            return (f"Model leans bullish (proba_up {p:.2f}) but conviction {c:.0f} "
                     f"is below the gate{gate_str} - watch, don't act.")
        return (f"Mixed signals; conviction {c:.0f} below gate{gate_str}{rg_clause}.")

    # passes gate
    if direction == 1:
        return (f"BUY signal{rg_clause}. Model probability {p:.2f}, "
                 f"conviction {c:.0f}/100. Treat the expected-move band as your "
                 f"working stop / target range.")
    if direction == -1:
        return (f"SHORT signal{rg_clause}. Model probability {p:.2f} (down), "
                 f"conviction {c:.0f}/100. Use the expected-move band for stop / target.")
    return (f"Direction flat despite passing the gate - unusual; check the "
             f"reasons string.")


def verdict(card_dict: dict) -> Verdict:
    """Build a plain-English verdict from a prediction card dict."""
    proba_up = float(card_dict.get("proba_up", 0.5) or 0.5)
    conviction = float(card_dict.get("conviction", 0.0) or 0.0)
    direction = int(card_dict.get("direction", 0) or 0)
    passes = bool(card_dict.get("passes_gate", False))
    regime = card_dict.get("regime")
    model_kind = str(card_dict.get("model_kind", ""))
    pillars = card_dict.get("pillars") or {}
    gate_min = card_dict.get("min_conviction")  # may be present in newer cards

    headline, tone = _label(proba_up, conviction)
    rubric_lean = _rubric_lean(pillars)
    quant_lean = _quant_lean(proba_up)
    disagreement = _disagreement_phrase(rubric_lean, quant_lean, pillars)

    # Override headline if there's a meaningful disagreement and gate blocked
    if disagreement and not passes and tone != "flat":
        # Promote to a "warn" tone so the UI can highlight it differently
        tone = "warn"

    summary = _summary_phrase(
        headline, tone,
        proba_up=proba_up, conviction=conviction,
        passes=passes, gate_min=gate_min,
        direction=direction, regime=regime,
        model_kind=model_kind,
        quant_lean=quant_lean,
    )

    return Verdict(headline=headline, summary=summary,
                   disagreement=disagreement, tone=tone)
