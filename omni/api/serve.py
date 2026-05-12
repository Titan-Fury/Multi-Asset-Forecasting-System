"""Entry point for `python -m omni.api.serve`.

Defaults are tuned for a development workstation. Override via env vars
for staging / production:

    OMNI_API_HOST=0.0.0.0
    OMNI_API_PORT=8000
    OMNI_API_WORKERS=2          (only honoured outside reload mode)
    OMNI_API_RELOAD=0           (1 enables reload-on-edit for dev)
    OMNI_API_LOG_LEVEL=info
    OMNI_API_WARMUP_RERANKER=1  (load bge-reranker on startup)
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("OMNI_API_HOST", "127.0.0.1")
    port = int(os.environ.get("OMNI_API_PORT", "8000"))
    reload = os.environ.get("OMNI_API_RELOAD", "0") == "1"
    workers = int(os.environ.get("OMNI_API_WORKERS", "1")) if not reload else 1
    log_level = os.environ.get("OMNI_API_LOG_LEVEL", "info")

    uvicorn.run(
        "omni.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
