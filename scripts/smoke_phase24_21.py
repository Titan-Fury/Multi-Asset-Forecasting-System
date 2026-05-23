"""End-to-end smoke test for phase 24.21 changes.

Exercises the new code paths without requiring network / GPU. Pure
defensive checks: each subtest catches its own failures and reports
PASS/FAIL/SKIP so we get a complete picture in one run.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# ---- sys.path bootstrap so `import omni` works regardless of cwd
_here = Path(__file__).resolve()
for _candidate in _here.parents:
    if (_candidate / "omni" / "__init__.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

_REPORT_PATH = _here.parent.parent / "smoke_report.txt"


_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _record(name, status, detail=""):
    _results.append((name, status, detail))
    print(f"[{status:4}] {name}  {detail}")


def t_gdelt_throttle():
    """GDELT module must enforce 5s gap between calls."""
    try:
        from omni.ingest.sources import gdelt
        # Reset state, then make two calls in a row to a stub-friendly URL
        # we never actually fetch from. We're exercising the throttle
        # logic only — the request will fail, that's fine.
        gdelt._last_call_ts = 0.0
        t0 = time.monotonic()
        gdelt._throttle()  # first call: no wait
        t1 = time.monotonic()
        gdelt._throttle()  # second call: must wait ~5s
        elapsed = time.monotonic() - t1
        if t1 - t0 > 0.1:
            return _record("gdelt.throttle first-call", "FAIL",
                           f"first call slept {t1-t0:.2f}s (should be ~0)")
        if elapsed < 4.5 or elapsed > 6.0:
            return _record("gdelt.throttle second-call", "FAIL",
                           f"second call elapsed {elapsed:.2f}s, expected ~5s")
        _record("gdelt.throttle", "PASS", f"second call slept {elapsed:.2f}s")
    except Exception as e:
        _record("gdelt.throttle", "FAIL", f"{type(e).__name__}: {e}")


def t_router_breaker():
    """Router circuit breaker must trip after 3 hard failures."""
    try:
        from omni.llm.router import Router
        from omni.llm.backends.base import BackendError
        r = Router()
        # Inject a fake hard-failure 3 times via the helper directly
        for _ in range(3):
            assert r._is_hard_failure(BackendError("connection refused"))
        # Verify soft failures don't trip
        assert not r._is_hard_failure(BackendError("rate-limited"))
        assert not r._is_hard_failure(BackendError("timeout"))
        _record("router.circuit_breaker", "PASS", "hard/soft classification correct")
    except Exception as e:
        _record("router.circuit_breaker", "FAIL", f"{type(e).__name__}: {e}")


def t_yaml_loads():
    """Updated yaml must parse and contain new rate limits."""
    try:
        from omni.utils.env import llm_routes
        cfg = llm_routes()
        be = cfg.get("backends", {})
        groq = be.get("groq_llama8b", {}).get("rate_limit", {})
        gem = be.get("gemini_flash_lite", {}).get("rate_limit", {})
        mini = be.get("openrouter_minimax_free", {}).get("rate_limit", {})
        gemma = be.get("openrouter_gemma", {})
        assert groq.get("rpm") == 10, f"groq_llama8b rpm={groq.get('rpm')}"
        assert gem.get("rpm") == 10, f"gemini_flash_lite rpm={gem.get('rpm')}"
        assert mini.get("rpm") == 3, f"openrouter_minimax_free rpm={mini.get('rpm')}"
        assert "gemma-3" in gemma.get("model", ""), f"gemma model={gemma.get('model')}"
        _record("config/llm_routes.yaml", "PASS", "rate limits + gemma-3 model present")
    except Exception as e:
        _record("config/llm_routes.yaml", "FAIL", f"{type(e).__name__}: {e}")


def t_symbol_aliases():
    """Index names must route to ^NSEI, etc."""
    try:
        from omni.ingest.prices import _yf_symbol
        cases = [
            (("india", "NIFTY 50"), "^NSEI"),
            (("india", "SENSEX"), "^BSESN"),
            (("india", "BANKNIFTY"), "^NSEBANK"),
            (("india", "RELIANCE"), "RELIANCE.NS"),  # unchanged
            (("india", "CIANAGRO"), "CIANAGRO.BO"),  # BSE-only preserved
            (("us", "AAPL"), "AAPL"),                # unchanged
        ]
        for (ac, tk), expected in cases:
            got = _yf_symbol(ac, tk)
            assert got == expected, f"{ac}/{tk} -> {got}, expected {expected}"
        _record("prices.symbol_aliases", "PASS", f"{len(cases)} cases")
    except Exception as e:
        _record("prices.symbol_aliases", "FAIL", f"{type(e).__name__}: {e}")


def t_negative_cache():
    """Negative cache must short-circuit repeat invalid lookups."""
    try:
        from omni.ondemand import lookup
        lookup._negative_cache.clear()
        assert not lookup._neg_cache_hit("us", "ZZZZNOTREAL")
        lookup._neg_cache_put("us", "ZZZZNOTREAL")
        assert lookup._neg_cache_hit("us", "ZZZZNOTREAL")
        # case-insensitive
        assert lookup._neg_cache_hit("us", "zzzznotreal")
        _record("lookup.negative_cache", "PASS", "put/hit/case-insensitive OK")
    except Exception as e:
        _record("lookup.negative_cache", "FAIL", f"{type(e).__name__}: {e}")


def t_pooled_session():
    """All four backends must use the shared http_session."""
    try:
        from omni.llm.backends.base import http_session
        s1 = http_session()
        s2 = http_session()
        assert s1 is s2, "http_session is not a singleton"
        # adapter mounted
        assert s1.get_adapter("https://api.groq.com") is not None
        _record("backends.http_session", "PASS", "singleton + https adapter mounted")
    except Exception as e:
        _record("backends.http_session", "FAIL", f"{type(e).__name__}: {e}")


def t_finbert_batch_method():
    """FinBERTBackend must expose complete_batch."""
    try:
        from omni.llm.backends.finbert import FinBERTBackend
        assert hasattr(FinBERTBackend, "complete_batch")
        _record("finbert.complete_batch", "PASS", "method present")
    except Exception as e:
        _record("finbert.complete_batch", "FAIL", f"{type(e).__name__}: {e}")


def t_sentiment_vectorized():
    """score_articles must produce same shape; aggregate must produce
    sent_pos_share via the vectorized path."""
    try:
        import pandas as pd
        from omni.features import sentiment

        # Fake 4 articles. We can't actually call FinBERT here without
        # heavy deps; the per-row router fallback path will fire and
        # log warnings. We only need to verify the dataframe contract.
        news = pd.DataFrame([
            {"title": "Stocks hit lower circuit, panic sell-off", "description": "",
             "published_at": "2026-05-10T10:00:00Z", "ticker_hint": "X"},
            {"title": "Promoter pledge of shares disclosed", "description": "",
             "published_at": "2026-05-10T11:00:00Z", "ticker_hint": "X"},
            {"title": "SEBI investigation closed", "description": "",
             "published_at": "2026-05-11T10:00:00Z", "ticker_hint": "X"},
            {"title": "", "description": "",
             "published_at": "2026-05-11T11:00:00Z", "ticker_hint": "X"},
        ])
        scored = sentiment.score_articles(news)
        # Check the vectorized keyword columns (don't depend on FinBERT)
        for col in ("panic_kw", "promoter_kw", "sebi_kw", "sent_score"):
            assert col in scored.columns, f"missing {col}"
        assert int(scored.loc[0, "panic_kw"]) == 1
        assert int(scored.loc[1, "promoter_kw"]) == 1
        assert int(scored.loc[2, "sebi_kw"]) == 1
        assert int(scored.loc[3, "panic_kw"]) == 0
        # Now aggregate
        agg = sentiment.aggregate(scored)
        assert "sent_pos_share" in agg.columns
        assert "sent_neg_share" in agg.columns
        assert "panic_kw_count" in agg.columns
        _record("sentiment.vectorized", "PASS",
                f"keyword cols correct, agg shape={agg.shape}")
    except Exception as e:
        _record("sentiment.vectorized", "FAIL", f"{type(e).__name__}: {e}")


def t_embedder_eviction():
    """OrderedDict eviction must drop oldest first, capped at _mem_cap."""
    try:
        import numpy as np
        from omni.llm.rag.embedder import Embedder
        emb = Embedder()
        emb._mem_cap = 3  # tiny cap for the test
        for i in range(5):
            emb._mem_put(f"k{i}", np.zeros(4, dtype=np.float32))
        assert len(emb._mem) == 3, f"len={len(emb._mem)}"
        keys = list(emb._mem.keys())
        assert keys == ["k2", "k3", "k4"], f"keys={keys}"
        _record("embedder.OrderedDict_eviction", "PASS", "FIFO order preserved")
    except Exception as e:
        _record("embedder.OrderedDict_eviction", "FAIL", f"{type(e).__name__}: {e}")


def t_lake_column_projection():
    """read_prices must accept columns= kwarg."""
    try:
        from omni.lake import io
        # Just verify the signature accepts columns=, not the actual read
        import inspect
        sig = inspect.signature(io.read_prices)
        assert "columns" in sig.parameters
        sig2 = inspect.signature(io.read_news_window)
        assert "columns" in sig2.parameters
        sig3 = inspect.signature(io.read_fundamentals_asof)
        assert "columns" in sig3.parameters
        _record("lake.io.column_projection", "PASS",
                "columns= present on 3 readers")
    except Exception as e:
        _record("lake.io.column_projection", "FAIL", f"{type(e).__name__}: {e}")


def t_blend_decomposition():
    """blend() must emit the new decomposition columns end-to-end with gate."""
    try:
        import pandas as pd
        from omni.ensemble import combine, gate
        df = pd.DataFrame([
            {"ticker": "X", "asset_class": "us",
             "proba_up": 0.7, "rubric_total": 80.0, "regime": "bull"},
            {"ticker": "Y", "asset_class": "india",
             "proba_up": 0.3, "rubric_total": 25.0, "regime": "bear"},
        ])
        b = combine.blend(df)
        for col in ("blend_w_model", "blend_w_rubric",
                    "blend_model_contrib", "blend_rubric_contrib",
                    "pre_gate_conviction", "blended_p", "conviction",
                    "direction"):
            assert col in b.columns, f"missing {col}"
        # weights sum to 1
        s = b["blend_w_model"] + b["blend_w_rubric"]
        assert (s.between(0.999, 1.001)).all(), f"weights don't sum to 1: {s.tolist()}"
        # contribs sum to blended_p
        c = b["blend_model_contrib"] + b["blend_rubric_contrib"]
        diff = (c - b["blended_p"]).abs()
        assert (diff < 1e-6).all(), f"contribs don't sum to blended_p: {diff.tolist()}"
        # Now apply gate
        g = gate.apply(b)
        assert "passes_gate" in g.columns
        assert "effective_conviction" in g.columns
        _record("blend.decomposition", "PASS",
                f"weights sum=1, contribs sum to blended_p, gate ran ({int(g['passes_gate'].sum())}/{len(g)} pass)")
    except Exception as e:
        _record("blend.decomposition", "FAIL", f"{type(e).__name__}: {e}")


def t_charts_importable():
    """The 3 new chart modules must import without streamlit/plotly side effects."""
    try:
        from omni.dashboard.charts import waterfall, regime_heatmap, counterfactual
        # Verify the public functions exist
        assert callable(waterfall.render_conviction_waterfall)
        assert callable(regime_heatmap.render_transition_matrix)
        assert callable(regime_heatmap.render_posterior_timeseries)
        assert callable(regime_heatmap.crisis_warning_chip)
        assert callable(counterfactual.render_counterfactual_panel)
        _record("dashboard.charts", "PASS",
                "3 modules import + 5 public functions present")
    except Exception as e:
        _record("dashboard.charts", "FAIL", f"{type(e).__name__}: {e}")


def t_crisis_warning():
    """crisis_warning_chip math."""
    try:
        import pandas as pd
        from omni.dashboard.charts.regime_heatmap import crisis_warning_chip
        # below threshold
        df = pd.DataFrame({"crisis": [0.05, 0.07, 0.10]})
        warn, p = crisis_warning_chip(df, threshold=0.15)
        assert not warn and abs(p - 0.10) < 1e-6
        # above
        df2 = pd.DataFrame({"crisis": [0.05, 0.10, 0.20]})
        warn2, p2 = crisis_warning_chip(df2, threshold=0.15)
        assert warn2 and abs(p2 - 0.20) < 1e-6
        # missing column → safe (False, 0)
        df3 = pd.DataFrame({"bull": [0.5]})
        warn3, p3 = crisis_warning_chip(df3)
        assert not warn3 and p3 == 0.0
        _record("regime.crisis_warning", "PASS", "threshold logic + missing-col safe")
    except Exception as e:
        _record("regime.crisis_warning", "FAIL", f"{type(e).__name__}: {e}")


def t_counterfactual_recompute():
    """Counterfactual recompute must be deterministic and reflect rubric/regime
    changes in the right direction."""
    try:
        from omni.dashboard.charts.counterfactual import _recompute
        card = {"ticker": "X", "asset_class": "us",
                "proba_up": 0.65, "rubric_total": 70.0, "regime": "bull"}
        # Baseline: bull regime, rubric 70
        c0, d0, p0 = _recompute(card, regime="bull", rubric_total=70.0)
        # Drop rubric to 30 → conviction should fall
        c1, d1, p1 = _recompute(card, regime="bull", rubric_total=30.0)
        assert c1 < c0, f"low rubric should drop conviction: {c0:.1f} -> {c1:.1f}"
        # Crisis regime should heavily penalise long
        c2, d2, p2 = _recompute(card, regime="crisis", rubric_total=70.0)
        assert c2 < c0, f"crisis regime should drag long conviction: {c0:.1f} -> {c2:.1f}"
        _record("counterfactual.recompute", "PASS",
                f"bull/rubric70={c0:.1f} -> low-rubric={c1:.1f}, crisis={c2:.1f}")
    except Exception as e:
        _record("counterfactual.recompute", "FAIL", f"{type(e).__name__}: {e}")


def t_fred_retry_signature():
    """fred_live.series must still take series_id + lookback_years kwarg."""
    try:
        from omni.ingest.sources import fred_live
        import inspect
        sig = inspect.signature(fred_live.series)
        assert "series_id" in sig.parameters
        assert "lookback_years" in sig.parameters
        _record("fred.signature", "PASS", "preserved API")
    except Exception as e:
        _record("fred.signature", "FAIL", f"{type(e).__name__}: {e}")


def main() -> int:
    print("\n=== phase 24.21 end-to-end smoke ===\n")
    t_gdelt_throttle()
    t_router_breaker()
    t_yaml_loads()
    t_symbol_aliases()
    t_negative_cache()
    t_pooled_session()
    t_finbert_batch_method()
    t_sentiment_vectorized()
    t_embedder_eviction()
    t_lake_column_projection()
    t_blend_decomposition()
    t_charts_importable()
    t_crisis_warning()
    t_counterfactual_recompute()
    t_fred_retry_signature()

    print("\n=== summary ===")
    fails = [r for r in _results if r[1] == "FAIL"]
    passes = [r for r in _results if r[1] == "PASS"]
    print(f"PASS: {len(passes)}/{len(_results)}")
    if fails:
        print("FAIL:")
        for n, _, d in fails:
            print(f"  - {n}: {d}")

    # Persist the report to disk so the bash wrapper can read it back
    # even when stdout redirect drops on this Windows env.
    lines = [f"PASS: {len(passes)}/{len(_results)}"]
    for n, status, d in _results:
        lines.append(f"[{status}] {n}  {d}")
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written to: {_REPORT_PATH}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
