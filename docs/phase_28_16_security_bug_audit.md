# Phase 28.16 — codebase security / bug / optimization audit

Full defensive sweep over `omni/` + `scripts/` (30,102 LOC). Tools: bandit
(security), pyflakes (bugs), vulture (dead code, Phase 28.15), targeted regex
scans. Issues handled one by one; the rest verified clean.

## Security — FIXED (commit `d5f4b32`)

bandit `-ll` found 3 High + 3 Medium; all resolved (post-fix: **0 high, 0 medium**):

| sev | id | what | fix |
| --- | -- | ---- | --- |
| High×3 | B324 | MD5/SHA1 for non-crypto IDs (RSS slug, Qdrant id, embed cache key) | `usedforsecurity=False` |
| Med×3 | B314 | untrusted remote XML (RSS feeds, SEC Form 4) parsed with stdlib `xml.etree` → entity-expansion / XXE | `defusedxml` (now a core dep), required unconditionally |

## Security — VERIFIED CLEAN (no change needed)

- **No hardcoded secrets** — regex scans for `api_key/token/secret/password=...`
  literals and provider prefixes (`sk-`, `ghp_`, `AKIA`, `xox`, PEM private
  keys) found nothing; all credentials come from env / config.
- **All `requests` calls carry a `timeout=`** (bandit B113 = 0) — no hang/DoS.
- **No `shell=True`** anywhere (bandit B602 = 0) — no shell-injection surface;
  the 10 `subprocess` calls all use arg-lists (`shell=False`).
- **API is secure-by-default** — `serve.py` binds `127.0.0.1` (0.0.0.0 only via
  opt-in env); no `debug=True`, no `reload=True` default, no wildcard CORS, no
  `trust_remote_code=True`.

## Bugs — VERIFIED CLEAN

- No `== None` / `!= None` (uses `is None`).
- No mutable default arguments.
- No bare `except:` in code (all `except Exception` or specific).
- No removed `DataFrame.append` (pandas≥2.2 would crash) — all `.append()` are
  on Python lists.
- pyflakes clean on touched files (project-wide baseline unchanged).

## Optimization — already done where it matters

- The two historical hot paths are already vectorized away from `iterrows`
  (`rubric/scorer.py`, `llm/backends/finbert.py` batch path — see their inline
  notes). Remaining `iterrows` are over small UI / per-ticker result frames, not
  the 200k-row panel; rewriting them would be premature optimization with real
  regression risk, so left as-is.
- Dead/redundant code removed separately in Phase 28.15.

## Deliberately left as-is

- 44 `try/except/pass` + 4 `try/except/continue` (bandit B110/B112) — this
  codebase's intentional graceful-degradation pattern ("never crash on a flaky
  feed"). Mass-rewriting would add churn/risk without fixing a real bug.
- A handful of `f"..."` strings without placeholders in `scripts/` — cosmetic;
  flagged for a future `ruff` sweep.

## Recommendation

Add a `ruff` + `bandit` pre-commit/CI gate so the cosmetic lint items and any
future High/Medium security regressions are caught automatically.

## Net

One real security class fixed (weak hashes + untrusted XML); everything else
audited and confirmed clean. The codebase is well-written and secure-by-default.
