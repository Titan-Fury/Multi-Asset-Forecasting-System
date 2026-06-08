"""Feature contract — single source of truth for the column set a model
expects at inference, written alongside each model joblib at training time.

Phase 28.4 — addresses the silent feature-schema drift between the training
path (`omni.features.build.build_for_ticker`) and the live path
(`omni.ondemand.features_inline.build_features_inline`). Previously, when
inference produced fewer columns than training had used, missing columns
were silently filled with the training-time median. A model that learned to
split on `quality_sloan_accruals` would then score every live row at the
same constant value, depressing live AUC while leaving OOF AUC unchanged.

The contract closes this loophole:
  - `write_contract(model_path, feature_cols)` is called by
    `omni.ondemand.model_registry.save_horizon` after every retrain so the
    contract file always matches the latest artifact.
  - `assert_contract(model_path, produced_cols)` is called by the prediction
    path before scoring; it raises `FeatureContractMismatch` (with a list of
    the missing columns) if any required column is absent. Callers that want
    the legacy median-fill behaviour can catch the exception and warn — the
    important property is that the mismatch is *visible* instead of silent.

The contract is JSON for hand-inspection, sibling to the model joblib so
its mtime tracks the model's mtime via the registry's normal save_horizon
flow.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from omni.utils.logger import log


class FeatureContractMismatch(Exception):
    """Raised when the produced feature set is missing columns the model
    was trained on. Carries a `missing` list so callers can build a
    diagnostic message or fall back to median-fill with an explicit warning.
    """

    def __init__(self, missing: list[str], model_path: str | Path):
        self.missing = list(missing)
        self.model_path = str(model_path)
        super().__init__(
            f"feature contract mismatch for {self.model_path}: "
            f"{len(self.missing)} columns missing at inference "
            f"(first 8: {self.missing[:8]})"
        )


def _contract_path(model_path: str | Path) -> Path:
    """Return the sibling .contract.json path for a given joblib artifact."""
    p = Path(model_path)
    return p.with_suffix(".contract.json")


def write_contract(model_path: str | Path,
                    feature_cols: Iterable[str]) -> Path:
    """Persist the feature contract beside the model joblib.

    Idempotent — overwrites the existing contract on each call. Uses
    sorted column names for deterministic diffing across retrains.
    """
    cols = sorted({str(c) for c in feature_cols})
    cp = _contract_path(model_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"feature_cols": cols, "n": len(cols)},
                              indent=2, sort_keys=True))
    log.debug(f"feature contract written: {cp} ({len(cols)} cols)")
    return cp


def read_contract(model_path: str | Path) -> list[str] | None:
    """Return the sorted feature_cols list from the sibling contract,
    or None when no contract file exists (older models pre-Phase 28.4).
    """
    cp = _contract_path(model_path)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text())
        cols = data.get("feature_cols") or []
        return [str(c) for c in cols]
    except Exception as exc:  # noqa: BLE001
        log.warning(f"feature contract unreadable at {cp}: {exc}")
        return None


def assert_contract(model_path: str | Path,
                    produced_cols: Iterable[str]) -> None:
    """Raise `FeatureContractMismatch` if any contracted column is missing
    from the produced feature set. No-op when the contract file is absent
    (legacy models keep working — a one-time warning is logged elsewhere).

    Note: produced_cols may be a SUPERSET of the contract; that's fine
    because the model selects the columns it cares about by name. The
    failure case is strict subset / missing-column.
    """
    contract = read_contract(model_path)
    if contract is None:
        return
    have = {str(c) for c in produced_cols}
    missing = [c for c in contract if c not in have]
    if missing:
        raise FeatureContractMismatch(missing, model_path)
