"""Environment variable helpers + YAML config loading."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from omni.utils.paths import config_dir

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Required env var '{name}' is not set")
    return val


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} placeholders inside YAML values."""
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            return os.getenv(m.group(1), m.group(0))
        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


# mtime-keyed cache: re-reads the file when it's modified on disk so an
# operator editing config/llm_routes.yaml during a long-running Streamlit
# session sees the change without restarting. The previous lru_cache made
# every config effectively immutable per-process.
_YAML_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_YAML_LOCK = threading.Lock()


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from config/. Argument is base name without extension."""
    path: Path = config_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    mtime = path.stat().st_mtime
    cached = _YAML_CACHE.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with _YAML_LOCK:
        cached = _YAML_CACHE.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        result = _expand_env(data)
        _YAML_CACHE[name] = (mtime, result)
        return result


def universe() -> dict[str, Any]:
    return load_yaml("universe")


def sources() -> dict[str, Any]:
    return load_yaml("sources")


def rubric() -> dict[str, Any]:
    return load_yaml("rubric")


def llm_routes() -> dict[str, Any]:
    return load_yaml("llm_routes")
