"""Verify the verdict banner against the real CIANAGRO replay outputs."""
from omni.dashboard.verdict import verdict

# May 8 card (model bearish, rubric still bullish, gate blocked)
may8 = {
    "ticker": "CIANAGRO",
    "proba_up": 0.323,
    "blended_p": 0.440,
    "conviction": 12.07,
    "direction": -1,
    "passes_gate": False,
    "regime": "bull",
    "model_kind": "stack_h5_rank_purged+meta(p=0.66)",
    "min_conviction": 15.0,
    "pillars": {
        "technical": 80.0, "fundamental": 51.0, "sentiment": 75.0,
        "macro": 56.0, "microstructure": 50.0, "oi": 50.0,
        "relative_strength": 60.0, "insider": 50.0, "earnings_momentum": 50.0,
    },
}

# May 15 card (gate blocked, low conviction)
may15 = {
    "ticker": "CIANAGRO",
    "proba_up": 0.401,
    "blended_p": 0.483,
    "conviction": 3.34,
    "direction": 0,
    "passes_gate": False,
    "regime": "bull",
    "model_kind": "stack_h5_rank_purged+meta(p=0.31)",
    "min_conviction": 15.0,
    "pillars": {
        "technical": 95.0, "fundamental": 51.0, "sentiment": 35.0,
        "macro": 56.0, "microstructure": 50.0, "oi": 50.0,
        "relative_strength": 60.0, "insider": 50.0, "earnings_momentum": 82.0,
    },
}

# Hypothetical clean BUY (e.g. a high-conviction long that passes gate)
buy = {
    "ticker": "TCS",
    "proba_up": 0.71,
    "blended_p": 0.68,
    "conviction": 36.0,
    "direction": 1,
    "passes_gate": True,
    "regime": "bull",
    "model_kind": "stack_h5_rank_purged",
    "min_conviction": 15.0,
    "pillars": {
        "technical": 70, "fundamental": 65, "sentiment": 60, "macro": 55,
        "relative_strength": 70, "earnings_momentum": 75,
    },
}

for label, c in [("CIANAGRO @ May 8", may8),
                   ("CIANAGRO @ May 15", may15),
                   ("TCS clean BUY", buy)]:
    v = verdict(c)
    print(f"=== {label} ===")
    print(f"  headline:      {v.headline}    [tone={v.tone}]")
    print(f"  summary:       {v.summary}")
    if v.disagreement:
        print(f"  disagreement:  {v.disagreement}")
    print()
