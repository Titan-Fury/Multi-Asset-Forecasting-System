"""Canonical filesystem layout for the data lake.

Every path is computed from `LAKE_ROOT` (resolves to ``data/lake/`` under the
project root by default; can be overridden with the ``OMNI_LAKE_ROOT`` env var).
Functions are pure — they only return Path objects, never create directories.
Use ``ensure_parent(p)`` before writing.
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    # this file is omni/lake/paths.py -> parents[2] is the project root
    return Path(__file__).resolve().parents[2]


def lake_root() -> Path:
    override = os.environ.get("OMNI_LAKE_ROOT")
    if override:
        return Path(override)
    return _project_root() / "data" / "lake"


def prices_path(asset_class: str, ticker: str) -> Path:
    return lake_root() / "prices" / asset_class.lower() / f"{ticker.upper()}.parquet"


def prices_dir(asset_class: str) -> Path:
    return lake_root() / "prices" / asset_class.lower()


def fundamentals_path(date_iso: str) -> Path:
    return lake_root() / "fundamentals" / f"{date_iso}.parquet"


def fundamentals_dir() -> Path:
    return lake_root() / "fundamentals"


def news_path(date_iso: str) -> Path:
    return lake_root() / "news" / f"{date_iso}.parquet"


def news_dir() -> Path:
    return lake_root() / "news"


def edgar_form4_path() -> Path:
    return lake_root() / "edgar" / "form4.parquet"


def macro_fred_path() -> Path:
    return lake_root() / "macro" / "fred.parquet"


def predictions_path(date_iso: str) -> Path:
    return lake_root() / "predictions" / f"{date_iso}.parquet"


def predictions_dir() -> Path:
    return lake_root() / "predictions"


def monitoring_path() -> Path:
    return lake_root() / "monitoring" / "realized_metrics.parquet"


def manifest_path() -> Path:
    return lake_root() / "manifest.json"


def training_set_path(target: str = "fwd_dir_5d") -> Path:
    return lake_root() / "training" / f"training_set_{target}.parquet"


def ensure_parent(p: Path) -> Path:
    """Create parent directories for ``p`` if needed; returns ``p`` for chaining."""
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def list_known_price_tickers(asset_class: str | None = None) -> list[tuple[str, str]]:
    """Return [(asset_class, ticker)] tuples already present in the lake."""
    out: list[tuple[str, str]] = []
    pdir = lake_root() / "prices"
    if not pdir.exists():
        return out
    classes = [asset_class] if asset_class else [d.name for d in pdir.iterdir()
                                                    if d.is_dir()]
    for ac in classes:
        d = prices_dir(ac)
        if not d.exists():
            continue
        for f in d.glob("*.parquet"):
            out.append((ac, f.stem))
    return sorted(out)
