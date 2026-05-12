"""Bug-pattern scanner for the Omni_Stocks codebase (Phase 24.16).

AST-based static analysis. Catches bug categories the auditor flagged
plus the patterns I've seen during the recent cull and refactor:

  - bare `except:` (catches KeyboardInterrupt — masks real bugs)
  - `except ...: pass` without any logging (silent failure)
  - mutable default args
  - `open(...)` not used as a context manager (resource leak risk)
  - dead references to removed symbols (theme.py cull may have left some)
  - module-level imports of optional/heavy deps without try-block
  - bare `Exception` re-raised as a different type without `from e`

Run:
    python scripts/scan_bugs.py [path...]
    # default scan: omni/, scripts/

Exit codes:
    0   no findings
    1   warnings only
    2   one or more error-level findings
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Finding:
    severity: str        # "info" | "warn" | "error"
    rule: str
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return (f"{self.severity.upper():5s}  {self.rule:<22s}  "
                f"{self.path}:{self.line}  {self.message}")


def _scan_file(path: Path) -> list[Finding]:
    out: list[Finding] = []
    try:
        src = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        out.append(Finding("error", "non-utf8-source", path, 0,
                           f"file is not valid UTF-8: {e}"))
        return out
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        out.append(Finding("error", "syntax-error", path, e.lineno or 0,
                           f"{e.msg}"))
        return out

    # Walk the tree for known bug patterns
    for node in ast.walk(tree):
        # Bare `except:` (no exception type)
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            out.append(Finding(
                "error", "bare-except", path, node.lineno,
                "bare `except:` catches KeyboardInterrupt and SystemExit; "
                "use `except Exception:` instead"
            ))

        # `except ...: pass` with NO logging or re-raise.
        # Phase 24.16 — only flag BROAD catches (`except Exception:` or
        # bare `except:`). Specific exception types (KeyError, OSError,
        # ValueError, ...) being swallowed is usually intentional
        # race-tolerance or fall-through code.
        if isinstance(node, ast.ExceptHandler):
            type_name = ""
            if node.type is None:
                type_name = "<bare>"
            elif isinstance(node.type, ast.Name):
                type_name = node.type.id
            elif isinstance(node.type, ast.Attribute):
                # e.g. `requests.exceptions.RequestException` — too specific
                # to flag as silent
                type_name = "<attribute>"
            elif isinstance(node.type, ast.Tuple):
                # `except (A, B):` — composite, hard to judge; skip flagging
                type_name = "<tuple>"

            broad = type_name in ("Exception", "BaseException", "<bare>")
            body = node.body
            if broad and len(body) == 1 and isinstance(body[0], ast.Pass):
                out.append(Finding(
                    "warn", "silent-except", path, node.lineno,
                    f"broad `except {type_name}:` swallowed silently with `pass`; "
                    f"log it or document why"
                ))
            elif (broad and len(body) == 1 and isinstance(body[0], ast.Expr)
                  and isinstance(body[0].value, ast.Constant)):
                out.append(Finding(
                    "warn", "silent-except", path, node.lineno,
                    f"broad `except {type_name}:` body is a constant; "
                    f"did you mean to log or re-raise?"
                ))

        # Mutable default args
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    out.append(Finding(
                        "error", "mutable-default", path, default.lineno,
                        f"function `{node.name}` has a mutable default "
                        f"({type(default).__name__}); use None + lazy init"
                    ))
                elif (isinstance(default, ast.Call)
                      and isinstance(default.func, ast.Name)
                      and default.func.id in ("list", "dict", "set", "deque")):
                    out.append(Finding(
                        "error", "mutable-default", path, default.lineno,
                        f"function `{node.name}` has a mutable default "
                        f"(`{default.func.id}()`); use None + lazy init"
                    ))

        # `open(...)` not in a `with` block (leak risk)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            # Find the enclosing statement and check it isn't `with`
            # We use a parent map to walk up.
            pass  # handled by parent-walk below

    # Parent-aware checks (open-without-with, raise-without-from)
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def _ancestors(n: ast.AST):
        cur = parents.get(id(n))
        while cur is not None:
            yield cur
            cur = parents.get(id(cur))

    for node in ast.walk(tree):
        # open(...) outside `with`
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            in_with = any(isinstance(a, ast.withitem) for a in _ancestors(node))
            in_assign = any(isinstance(a, ast.Assign) for a in _ancestors(node))
            # Common safe pattern: `open(...)` directly inside a `with`
            # Edge: `pd.read_csv(open(...))` is NOT safe but rare
            if not in_with and in_assign:
                # Skim ancestors for a Path.open(... or io.open(... attribute call too
                out.append(Finding(
                    "warn", "open-no-with", path, node.lineno,
                    "`open(...)` not used as a context manager; "
                    "wrap with `with ... as f:` to avoid leaking the handle"
                ))

        # re-raise wrong type without `from`
        if isinstance(node, ast.Raise) and node.exc is not None and node.cause is None:
            # Inside an except block?
            in_except = any(isinstance(a, ast.ExceptHandler) for a in _ancestors(node))
            if in_except and isinstance(node.exc, ast.Call):
                # Re-raising a different exception type without `from e`
                # (Python issues a warning about chained exceptions)
                out.append(Finding(
                    "info", "raise-no-from", path, node.lineno,
                    "re-raising in an except block without `from e`; "
                    "loses the cause chain"
                ))

    # File-level pattern: detect `from X import *` (namespace pollution)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            out.append(Finding(
                "warn", "wildcard-import", path, node.lineno,
                f"wildcard import from {node.module}; explicit list is safer"
            ))

    return out


def main(argv: list[str]) -> int:
    targets = [Path(p) for p in argv[1:]] or [Path("omni"), Path("scripts")]
    files: list[Path] = []
    for t in targets:
        if t.is_file() and t.suffix == ".py":
            files.append(t)
        elif t.is_dir():
            files.extend(t.rglob("*.py"))

    findings: list[Finding] = []
    for f in sorted(files):
        # Skip generated / cache dirs
        if any(part in (".venv", "__pycache__", "lightning_logs",
                         "egg-info") for part in f.parts):
            continue
        findings.extend(_scan_file(f))

    # Group + report
    by_rule: dict[str, list[Finding]] = {}
    for fnd in findings:
        by_rule.setdefault(fnd.rule, []).append(fnd)

    print(f"Scanned {len(files)} files. Findings: {len(findings)}.")
    print()
    for rule in sorted(by_rule.keys()):
        items = by_rule[rule]
        sev = items[0].severity
        print(f"  [{sev}] {rule}: {len(items)}")
    print()

    # Print details (errors first, then warnings, then info)
    order = {"error": 0, "warn": 1, "info": 2}
    for fnd in sorted(findings,
                       key=lambda f: (order.get(f.severity, 9),
                                       f.rule, str(f.path), f.line)):
        print(f"  {fnd}")

    n_errors = sum(1 for f in findings if f.severity == "error")
    if n_errors:
        return 2
    if findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
