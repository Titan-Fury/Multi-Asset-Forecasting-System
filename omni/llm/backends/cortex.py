"""Snowflake Cortex backend (CORTEX.SENTIMENT and CORTEX.COMPLETE).

Uses snowflake.connector with a named connection from
~/.snowflake/connections.toml (env var SNOWFLAKE_CONNECTION_NAME).
"""
from __future__ import annotations

import threading
import time
from typing import Any

from omni.llm.backends.base import Backend, BackendError, LLMResult
from omni.utils.env import get_env
from omni.utils.logger import log


class CortexBackend(Backend):
    _conn = None  # class-level lazy connection
    _conn_lock = threading.Lock()

    def _connect(self):
        # Fast path: connection already alive.
        c = CortexBackend._conn
        if c is not None and not c.is_closed():
            return c
        # Slow path: serialize creation so two threads don't both connect.
        with CortexBackend._conn_lock:
            c = CortexBackend._conn
            if c is not None and not c.is_closed():
                return c
            try:
                import snowflake.connector
            except ImportError as e:
                raise BackendError(
                    f"snowflake-connector-python not installed: {e}"
                ) from e
            conn_name = get_env(self.config.get(
                "connection_env", "SNOWFLAKE_CONNECTION_NAME"))
            if not conn_name:
                raise BackendError("SNOWFLAKE_CONNECTION_NAME not set")
            try:
                CortexBackend._conn = snowflake.connector.connect(
                    connection_name=conn_name)
            except Exception as e:
                raise BackendError(f"snowflake connect failed: {e}") from e
            log.info(f"Cortex connected via '{conn_name}'")
            return CortexBackend._conn

    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        fn = self.config.get("function", "SNOWFLAKE.CORTEX.COMPLETE").upper()
        t0 = time.monotonic()
        cur = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            if fn.endswith(".SENTIMENT"):
                cur.execute(f"SELECT {fn}(%s)", (prompt,))
                row = cur.fetchone()
                score = float(row[0]) if row and row[0] is not None else 0.0
                return LLMResult(
                    backend=self.name,
                    text=f"{score:.4f}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                    extra={"sentiment_score": score, "function": fn},
                )
            # COMPLETE path
            model = self.config.get("model", "llama3.1-70b")
            cur.execute(f"SELECT {fn}(%s, %s)", (model, prompt))
            row = cur.fetchone()
            text = row[0] if row else ""
            return LLMResult(
                backend=self.name,
                text=str(text),
                latency_ms=(time.monotonic() - t0) * 1000,
                extra={"function": fn, "model": model},
            )
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"cortex {fn} failed: {e}") from e
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    pass
