# Phase 28.19 — UI/app optimization research (web-grounded) + plan

Researched against current official guidance and mapped to this codebase.
Sources consulted this session:
- **Streamlit docs** — Caching overview
  (docs.streamlit.io/develop/concepts/architecture/caching) and Fragments
  (…/architecture/fragments), fetched 2026-06.
- Microsoft **qlib**, de Prado/**mlfinlab**, Numerai (Phase 28.17, for the
  modelling side).

Environment: Streamlit **1.57.0** installed (so `st.fragment`, ≥1.37, is
available). Dashboard boots clean headless. Code is already ruff-swept and
caches idiomatically — so this is targeted optimization, not cleanup.

## Where the dashboard already follows best practice

- `@st.cache_data(ttl=…, show_spinner=False)` for every parquet/JSON/data load
  (`cache.py`, `marquee.py`, `app.py`) with **mtime cache-busting** via
  `reload_keys()` — exactly the documented pattern.
- `st.session_state` for cross-rerun UI state (selections, dev-unlock, last
  card) rather than module globals.
- Section dispatch so only the active tab's body renders.

## Optimizations — prioritized

### 1. `st.fragment` for interaction-local sections (highest leverage)
Streamlit reruns the **entire** 2,500-line `app.py` top-to-bottom on every
widget interaction. Wrapping self-contained, widget-bearing sections in
`@st.fragment` makes a widget trigger only *that* fragment's rerun.
- DONE: live marquee → `@st.fragment(run_every="300s")` (Phase 28.19,
  commit fa0c2e4) — self-refreshes without a full rerun.
- NEXT (needs in-browser validation): wrap the **Predict** card flow and each
  heavy **viz tab** (cosmos3d, regime heatmap, waterfall, corrgraph) in their
  own fragments so e.g. changing the conviction slider re-renders one chart, not
  the whole page + marquee + sidebar. Biggest perceived-latency win.

### 2. Replace targeted `st.rerun()` with fragment reruns
There are 13 full-page `st.rerun()` calls. Several (watchlist add/remove,
section switch, refresh buttons) live inside what should be fragments — once a
section is a fragment, use `st.rerun(scope="fragment")` so the click doesn't
re-execute the world.

### 3. `@st.cache_resource` for any in-process heavy object
`cache_data` copies/pickles its return on every call. For unserializable,
expensive singletons (a loaded model, an embedder, a DB/HTTP client) the docs
prescribe `cache_resource`. The dashboard currently routes model use through
`predict.py`'s own process cache (fine); if any future dashboard code loads a
model/embedder directly, use `cache_resource`, not `cache_data`.

### 4. Guarantee data freshness AND cache hits
`load_predictions(_mt=0.0)` etc. only refresh when callers pass
`reload_keys()['pred']`. Audit call sites to ensure the mtime is threaded
through, so a daily run invalidates the 1h TTL immediately instead of serving
stale data for up to an hour.

### 5. Visuals — measured, not blind
Visual polish (spacing, contrast, chart density, mobile layout) must be judged
on a rendered page. The sandbox browser is blocked (ERR_BLOCKED_BY_CLIENT), so
this is deferred to a session where the app can be opened and eyeballed, rather
than risk regressions on a 3.3k-line `theme.py` I cannot see rendered. The CSS
is large but it is *styling*, not dead bloat.

## App-wide engineering practices to adopt next

- **Dashboard render smoke-test in CI**: `streamlit run … --server.headless`
  boots clean in seconds — add a CI job that boots it and asserts no startup
  traceback (catches import/syntax regressions the lint gate can't).
- **Progressive `mypy`**: already a dev dep; enable on a few leaf modules first,
  expand — catches type bugs the runtime-dynamic dashboard hides.
- **Tests for dashboard helpers** (`movers`, `marquee`, `searchhistory`,
  `watchlist`) — pure functions, cheap to cover.
- Modelling side: see Phase 28.17 (cross-sectional features, uniqueness
  weights) — but 28.18 showed preprocessing won't manufacture edge here.

## Honest framing

The two genuinely safe, verifiable wins were taken: the lint/security gate
(Phase 28.16) and the marquee fragment (here). The rest of the UI optimization
(fragmenting the predict/viz sections, visual polish) is real and high-value but
requires a *renderable* Streamlit instance to apply without regressing a working
app — recommended as the next interactive session, not a blind edit.
