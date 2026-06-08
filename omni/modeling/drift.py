"""ADWIN-based drift detector for prediction-error streams.

Uses river's ADWIN (Adaptive WINdow) — a parameter-free drift detector
that maintains an adaptive window over a stream and flags when the
recent slice's mean significantly differs from the older slice.

Why ADWIN here:
    Our universal models are retrained weekly. Between retrains, regime
    shifts (e.g. Aug-2024 yen-carry unwind, Apr-2025 tariff shock) can
    blow up live prediction error long before the calendar trigger.
    ADWIN catches it within ~30-100 observations and we can then
    schedule an emergency retrain.

Public API:
    DriftMonitor(delta=0.002).update(brier_score) -> bool   # True == drift
    state JSON-persisted at data/drift/<scope>.json

Falls back to a no-op if river is not installed (so the dashboard still
runs in minimal-deps environments).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from river.drift import ADWIN
    HAS_RIVER = True
except Exception:
    HAS_RIVER = False
    ADWIN = None  # type: ignore


STATE_DIR = Path("data") / "drift"


@dataclass
class DriftEvent:
    detected_at: str          # ISO date
    n_obs: int
    metric_value: float
    note: str = ""


@dataclass
class DriftMonitor:
    """Wraps river.drift.ADWIN with optional state persistence.

    Args:
        scope: filename-safe identifier ("h1", "h5", "h21", ...).
        delta: ADWIN confidence parameter (smaller = stricter).
    """
    scope: str
    delta: float = 0.002
    detector: object = field(default=None, init=False, repr=False)
    n_seen: int = 0
    last_drift: str | None = None

    def __post_init__(self):
        if HAS_RIVER:
            self.detector = ADWIN(delta=self.delta)
        else:
            self.detector = None

    def update(self, value: float) -> bool:
        """Push a new error/loss value. Returns True if drift detected."""
        if self.detector is None:
            return False
        self.n_seen += 1
        self.detector.update(float(value))
        drift = bool(getattr(self.detector, "drift_detected", False))
        return drift

    def reset(self) -> None:
        if HAS_RIVER:
            self.detector = ADWIN(delta=self.delta)
        self.n_seen = 0

    # ------------------------------------------------------------------
    # State persistence (lightweight — we only persist counters; the
    # ADWIN window is small enough that we just rebuild on load).
    # ------------------------------------------------------------------
    def state_path(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_DIR / f"{self.scope}.json"

    def save(self) -> None:
        self.state_path().write_text(json.dumps({
            "scope": self.scope,
            "delta": self.delta,
            "n_seen": self.n_seen,
            "last_drift": self.last_drift,
        }))

    @classmethod
    def load(cls, scope: str, *, delta: float = 0.002) -> DriftMonitor:
        path = STATE_DIR / f"{scope}.json"
        if not path.exists():
            return cls(scope=scope, delta=delta)
        try:
            obj = json.loads(path.read_text())
            m = cls(scope=scope, delta=float(obj.get("delta", delta)))
            m.n_seen = int(obj.get("n_seen", 0))
            m.last_drift = obj.get("last_drift")
            return m
        except Exception:
            return cls(scope=scope, delta=delta)
