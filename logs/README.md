# logs/

**Not committed to git.** Local-only directory for runtime logs and
process stderr/stdout dumps (`*.err`, `*.out`, `*.log`).

The structured logger in `omni/utils/logger.py` writes to
`data/omni.log` by default. This directory is for ad-hoc
captures from `python ... 2> logs/run.err 1> logs/run.out`.
