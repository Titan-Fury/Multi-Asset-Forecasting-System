"""Universal model registry — load/save bootstrapped XGBoost+isotonic models.

Supports a SINGLE 5d model (legacy) or a MULTI-HORIZON set (1d / 5d / 21d).

File layout (under data/models/):
    universal_5d.joblib        -- {model, calibrator, feature_cols, medians}
    universal_5d.meta.json     -- training metadata (timestamp, tickers, AUC)
    universal_1d.joblib        -- (Step 5) optional 1-day model
    universal_21d.joblib       -- (Step 5) optional 21-day model

Public API:
    save(um)              -- save the legacy 5d model
    load()                -- load the legacy 5d model (raises if missing)
    save_horizon(um, h)   -- save a horizon-specific model (h in {1, 5, 21})
    load_horizon(h)       -- load a horizon-specific model (None if missing)
    available_horizons()  -- list[int] of horizons currently on disk
    is_available()        -- bool: at least the 5d model exists
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from omni.utils.logger import log
from omni.utils.paths import models_dir


_MODEL_NAME = "universal_5d"
_LEGACY_HORIZON = 5
_VALID_HORIZONS = (1, 5, 21)


@dataclass
class UniversalModel:
    model: object              # xgb.XGBClassifier
    calibrator: object         # CalibratedClassifierCV / FrozenEstimator wrapper
    feature_cols: list[str]
    medians: pd.Series         # per-feature median for NaN imputation
    trained_at: str
    tickers_used: list[str]
    oof_auc: float | None
    horizon_days: int = _LEGACY_HORIZON
    n_train_rows: int | None = None


def _paths_for(horizon: int = _LEGACY_HORIZON) -> tuple[Path, Path]:
    base = models_dir()
    base.mkdir(parents=True, exist_ok=True)
    name = f"universal_{horizon}d"
    return base / f"{name}.joblib", base / f"{name}.meta.json"


def _paths() -> tuple[Path, Path]:
    """Legacy 5-day paths (kept for backwards compat)."""
    return _paths_for(_LEGACY_HORIZON)


def save(um: UniversalModel) -> Path:
    """Save the legacy 5d model (h=5)."""
    return save_horizon(um, _LEGACY_HORIZON)


def save_horizon(um: UniversalModel, horizon: int) -> Path:
    """Save a horizon-specific model. ``horizon`` must be in {1, 5, 21}.

    Persists training metadata in two places for robustness:
      1. Inside the joblib bundle itself (so any caller doing
         ``joblib.load(path)`` directly gets `trained_at`, `oof_auc`,
         `tickers_used`, `n_train_rows`, `horizon_days` without needing
         to read a sidecar file).
      2. A `<name>.meta.json` sidecar (back-compat with older readers
         and convenient for quick CLI inspection).
    """
    if horizon not in _VALID_HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}; valid: {_VALID_HORIZONS}")
    blob, meta = _paths_for(horizon)
    metadata = {
        "trained_at": um.trained_at,
        "tickers_used": list(um.tickers_used or []),
        "oof_auc": um.oof_auc,
        "n_features": len(um.feature_cols),
        "n_train_rows": getattr(um, "n_train_rows", None),
        "horizon_days": horizon,
    }
    joblib.dump({
        "model": um.model,
        "calibrator": um.calibrator,
        "feature_cols": um.feature_cols,
        "medians": um.medians.to_dict(),
        # Embedded metadata (Phase 24.18 — was previously sidecar-only).
        "metadata": metadata,
    }, blob)
    meta.write_text(json.dumps(metadata, indent=2))
    # Phase 28.4 — feature contract sidecar so inference can detect schema
    # drift loudly (raises FeatureContractMismatch) instead of silently
    # median-filling missing columns and depressing live AUC.
    try:
        from omni.features.contract import write_contract
        write_contract(blob, um.feature_cols)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"feature contract write skipped: {exc}")
    log.info(f"universal model saved (h={horizon}d) -> {blob} "
             f"(oof_auc={um.oof_auc}, n_train_rows={metadata['n_train_rows']})")
    return blob


_CACHE: dict[int, UniversalModel] = {}
# Phase 28.4 — guard `_CACHE` mutations against concurrent loads /
# invalidations under FastAPI's threadpool + Streamlit reruns. Reads
# of an existing entry stay lockless because dict .get() is atomic.
import threading as _mr_threading
_CACHE_LOCK = _mr_threading.Lock()


def load() -> UniversalModel:
    """Load the legacy 5d model. Cached in-memory after first call.

    Raises FileNotFoundError with actionable message if the file is missing.
    Equivalent to ``load_horizon(5)`` but raises on missing for back-compat.
    """
    um = load_horizon(_LEGACY_HORIZON)
    if um is None:
        blob, _ = _paths_for(_LEGACY_HORIZON)
        raise FileNotFoundError(
            f"Universal model not found at {blob}.\n"
            f"Run: python scripts/train_universal_model.py"
        )
    return um


def load_horizon(horizon: int) -> UniversalModel | None:
    """Load a horizon-specific model. Returns None if file is missing.

    Reads metadata from the joblib bundle when present (new format), and
    falls back to the `<name>.meta.json` sidecar for older artifacts.
    """
    # Fast path: atomic dict lookup, no lock needed.
    cached = _CACHE.get(horizon)
    if cached is not None:
        return cached
    blob, meta = _paths_for(horizon)
    if not blob.exists():
        return None
    bundle = joblib.load(blob)
    embedded = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    sidecar = json.loads(meta.read_text()) if meta.exists() else {}
    # Prefer embedded metadata; fall back to sidecar; then to defaults.
    md = {**sidecar, **embedded}  # embedded wins on key overlap
    um = UniversalModel(
        model=bundle["model"],
        calibrator=bundle["calibrator"],
        feature_cols=bundle["feature_cols"],
        medians=pd.Series(bundle["medians"]),
        trained_at=md.get("trained_at", "?"),
        tickers_used=md.get("tickers_used", []) or [],
        oof_auc=md.get("oof_auc"),
        horizon_days=int(md.get("horizon_days", horizon)),
        n_train_rows=md.get("n_train_rows"),
    )
    # Phase 28.4 — atomic publish under the lock. Re-check inside the
    # lock in case another thread populated the entry while we were
    # joblib-loading; the second result wins and the duplicate UM
    # we built gets garbage-collected.
    with _CACHE_LOCK:
        existing = _CACHE.get(horizon)
        if existing is not None:
            return existing
        _CACHE[horizon] = um
    log.info(f"universal model loaded (h={horizon}d): "
              f"{len(um.feature_cols)} features, trained {um.trained_at}, "
              f"oof_auc={um.oof_auc}, n_train_rows={um.n_train_rows}")
    return um


def available_horizons() -> list[int]:
    """Return horizons in {1, 5, 21} that currently exist on disk."""
    return [h for h in _VALID_HORIZONS if _paths_for(h)[0].exists()]


def invalidate() -> None:
    # Phase 28.4 — atomic replace under the lock so concurrent readers
    # don't observe a half-emptied dict.
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = {}


def is_available() -> bool:
    """True iff the legacy 5d model exists (the minimum needed to predict)."""
    blob, _ = _paths_for(_LEGACY_HORIZON)
    return blob.exists()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
