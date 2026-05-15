"""On-demand prediction orchestrator.

`predict_ticker(ticker, asset_class)` runs the full pipeline:
    lookup -> features -> universal model -> rubric -> blend+gate ->
    PredictionCard with provenance + LLM "why" explanation.

`deep_analysis(ticker, asset_class)` is the ~30s heavy variant: instead
of the universal model it runs a per-ticker walk-forward XGBoost +
isotonic calibration on the single ticker's history (min_train=252,
n_folds=4), then uses the LAST fold's calibrator for the live row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import math

import numpy as np
import pandas as pd

from omni.ensemble.pipeline import assemble_signals
from omni.models.xgboost_5d import walkforward_fit
from omni.ondemand import model_registry
from omni.ondemand.cache import TTL, cached
from omni.ondemand.features_inline import build_features_inline
from omni.ondemand.lookup import LookupBundle, lookup_ticker
from omni.utils.logger import log


@dataclass
class PredictionCard:
    ticker: str
    asset_class: str
    display_name: str
    asof: pd.Timestamp
    direction: int                       # -1 / 0 / +1
    conviction: float                    # 0..100
    blended_p: float                     # 0..1
    proba_up: float                      # 0..1
    rubric_total: float
    pillars: dict                        # technical/fundamental/sentiment/macro
    regime: str
    passes_gate: bool
    reasons: str
    why_text: str
    news_top5: list[dict] = field(default_factory=list)
    # Phase 24.15 — RAG-grounded citations from the explanation pipeline.
    # Each entry: {n, ticker, source, date, url, title, score}. Empty when
    # the deterministic fallback is used (no news, RAG offline, or
    # validation failed twice). UI renders these as numbered chips below
    # the why-text.
    why_citations: list[dict] = field(default_factory=list)
    why_grounding_score: float = 0.0
    why_path: str = "deterministic"  # "rag" | "rag_low_grounding" | "deterministic"
    provenance: dict = field(default_factory=dict)
    model_kind: str = "universal_5d"
    deep_analysis_available: bool = True
    fold_aucs: list[float] | None = None  # only set by deep_analysis
    # Phase 24.18 — lower-circuit risk score (India equities only).
    # `circuit_risk_pct`  in [0, 1] — probability of hitting -9.5%+ over 5d.
    # `circuit_risk_band` "low" / "elevated" / "high" or None when n/a.
    circuit_risk_pct: float | None = None
    circuit_risk_band: str | None = None
    # ---- new in UI v2 ----
    expected_move_pct: float = 0.0       # signed point estimate over 5 trading days
    expected_low_pct: float = 0.0        # ~ -1 sigma envelope
    expected_high_pct: float = 0.0       # ~ +1 sigma envelope
    # Per-horizon expected moves (point + ±1σ band, all in percent)
    move_1d_pct: float = 0.0
    move_1d_low_pct: float = 0.0
    move_1d_high_pct: float = 0.0
    move_1w_pct: float = 0.0
    move_1w_low_pct: float = 0.0
    move_1w_high_pct: float = 0.0
    move_1m_pct: float = 0.0
    move_1m_low_pct: float = 0.0
    move_1m_high_pct: float = 0.0
    recent_vol_5d_pct: float = 0.0       # realised 5d std-dev (annualised? no — 5d horizon)
    realized_sharpe_1y: float = 0.0      # last 252 daily returns annualised
    last_close: float = 0.0
    last_close_date: str = ""
    last_intraday_price: float = 0.0
    last_intraday_at: str = ""
    data_coverage: dict = field(default_factory=dict)
    # ---- intraday microstructure (Part A) ----
    intraday_return_pct: float = 0.0
    gap_open_pct: float = 0.0
    vwap_dev_bps: float = 0.0
    opening_range_breakout: float = 0.0
    intraday_realized_vol: float = 0.0
    last_hour_return_pct: float = 0.0
    volume_z_today: float = 0.0
    range_compression: float = 0.0
    intraday_bar_count: float = 0.0
    intraday_tilt_applied: bool = False
    # ---- pair / cross-asset (Part B) ----
    pair_signal: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asof"] = pd.Timestamp(self.asof).isoformat()
        return d


# ---------- shared core ----------

def _last_row_features(bundle: LookupBundle) -> pd.DataFrame:
    """Build the feature frame for this ticker. Returns full history (used for
    deep_analysis); callers pull `.tail(1)` for single-row scoring."""
    feats = build_features_inline(
        bundle.asset_class, bundle.ticker, bundle.prices,
        fundamentals=bundle.fundamentals,
        sentiment=bundle.news_sentiment,
        macro=bundle.macro,
        horizons=(1, 5, 20),
    )
    return feats


def _align_to_universal(X_one: pd.DataFrame, um: model_registry.UniversalModel
                        ) -> np.ndarray:
    """Reindex one row to the universal model's feature_cols, fill NaN with
    training-time medians."""
    aligned = X_one.reindex(columns=um.feature_cols)
    medians = um.medians.reindex(um.feature_cols)
    aligned = aligned.fillna(medians).fillna(0.0)
    return aligned.to_numpy(dtype=float)


# ============================================================================
# Phase 24 — Tier-1 inference optimizations
# ----------------------------------------------------------------------------
# Module-level caches so a multi-ticker screen sweep doesn't re-load the
# 41 MB stack pickle 100 times. The cache invalidates whenever the universal
# model's mtime tag changes (same tag we use in the prediction-card cache key).
# ============================================================================

_MODEL_CACHE: dict[str, tuple[Any, ...]] = {}


def _cached_models(mv_tag: str):
    """Return (universal, stack_record, conformal, circuit_risk) loaded once
    per mv_tag. circuit_risk is the optional Phase 24.18 sparse-positive
    classifier; None when the artifact is absent (training script not run)."""
    if mv_tag in _MODEL_CACHE:
        return _MODEL_CACHE[mv_tag]
    universal = None
    stack_rec = None
    conformal = None
    circuit_model = None
    try:
        universal = model_registry.load()
    except Exception as e:  # noqa: BLE001
        log.warning(f"universal load skipped: {e}")
    try:
        import joblib as _jl
        from pathlib import Path as _P
        sp_new = _P("data") / "models" / "stack_h5.joblib"
        sp_legacy = _P("data") / "models" / "stack_h5.pkl"
        sb = None
        if sp_new.exists():
            sb = _jl.load(sp_new)
        elif sp_legacy.exists():
            # Legacy artifact only; warn and read once. Re-train to migrate.
            import pickle  # nosec - legacy artifact migration only
            with sp_legacy.open("rb") as f:
                sb = pickle.load(f)  # nosec
            log.warning(
                f"stack: loaded legacy pickle {sp_legacy}; "
                "re-run scripts/train_universal_model.py to migrate to joblib."
            )
        if sb is not None:
            stack_rec = sb.get("stack")
    except Exception as e:  # noqa: BLE001
        log.warning(f"stack load skipped: {e}")
    try:
        import joblib as _jl
        from pathlib import Path as _P
        cp = _P("data") / "models" / "conformal_h5.joblib"
        if cp.exists():
            conformal = _jl.load(cp)
    except Exception as e:  # noqa: BLE001
        log.debug(f"conformal load skipped: {e}")
    # Phase 24.18 — circuit-risk model (India only). Loaded the same way
    # the other heads are: pickled, file-mtime keyed via mv_tag so a retrain
    # invalidates the cache automatically.
    try:
        from omni.models import circuit_risk as _crisk
        circuit_model = _crisk.load()
        if circuit_model is not None:
            log.info(
                f"circuit-risk model loaded: pr_auc={circuit_model.pr_auc:.4f} "
                f"base_rate={circuit_model.base_rate:.4f} "
                f"feats={len(circuit_model.feature_cols)}"
            )
    except Exception as e:  # noqa: BLE001
        log.debug(f"circuit-risk load skipped: {e}")
    # Phase 24.18 — meta-labelling heads (de Prado AFML §3.6). Predicts
    # P(primary signal is correct) given primary outputs + risk-state cols.
    # Used as a SECONDARY filter: signals where meta_p < 0.45 are
    # downweighted toward 0.5 (less conviction). Loaded lazily; absence is
    # non-fatal — predict.py just returns the raw stack proba.
    #
    # Phase 24.18b — multi-horizon: separate heads for h={1,5,21}. Stored as
    # `meta_heads[h]` -> (model, MetaResult). The h=5 entry is the legacy
    # `meta_head` exposed via `meta_heads[5]`. Code that wants a specific
    # horizon should look up `meta_heads.get(h)`.
    meta_heads: dict[int, object] = {}
    try:
        from omni.modeling.meta_label import load_meta as _load_meta
        from pathlib import Path as _P
        for h in (1, 5, 21):
            mp = _P("data") / "models" / f"meta_h{h}.joblib"
            if mp.exists():
                obj = _load_meta(mp)
                if obj is not None:
                    meta_heads[h] = obj
                    _, mr = obj
                    log.info(
                        f"meta-labelling head h={h} loaded: "
                        f"oof_auc={mr.auc_oof:.4f} feats={len(mr.feature_cols)}"
                    )
    except Exception as e:  # noqa: BLE001
        log.debug(f"meta-labelling load skipped: {e}")
    # Back-compat: callers previously expected a single `meta_head`. Expose
    # the h=5 head under that name when present.
    meta_head = meta_heads.get(5)
    _MODEL_CACHE.clear()
    _MODEL_CACHE[mv_tag] = (universal, stack_rec, conformal, circuit_model,
                              meta_head, meta_heads)
    return _MODEL_CACHE[mv_tag]


def _score_one(feat_only: pd.DataFrame,
               *, universal, stack_rec, asset_class: str,
               meta_head=None,
               ) -> tuple[float, str]:
    """Score a single live-row feature vector through universal/stack/specialist.

    Returns (proba_up, model_kind). Centralizes the alignment so we don't
    re-reindex/refill for each base learner. If universal is None we return
    (0.5, 'rubric_only').

    Phase 24.18 — when a meta-labelling head is available, the primary's
    raw proba is shrunk toward 0.5 by `meta_p` (the head's estimate of
    signal correctness). This is the de Prado §3.6 bet-sizing pattern:
    final_proba_up = 0.5 + (raw_proba - 0.5) * meta_p
    Effect: confident primaries that the meta agrees with stay sharp;
    primaries the meta distrusts get pulled to 50/50 (low conviction).
    """
    if universal is None:
        return 0.5, "rubric_only"

    X_uni = _align_to_universal(feat_only, universal)
    proba = float(universal.calibrator.predict_proba(X_uni)[0, 1])
    kind = "universal_5d"

    # Stack override — share the reindex (most cols overlap with universal)
    if stack_rec is not None:
        try:
            uni_auc = float(universal.oof_auc or 0.5)
            if stack_rec.oof_meta_auc > uni_auc:
                Xs = feat_only.reindex(columns=stack_rec.feature_cols)
                p_stack = float(stack_rec.predict_proba(Xs)[0])
                proba = p_stack
                kind = "stack_h5_rank_purged"
        except Exception as exc:  # noqa: BLE001
            log.warning(f"stack score failed: {exc}")

    # India specialist (lazy load — cheap; cached by specialists module)
    if asset_class == "india":
        try:
            from omni.specialists import india as _ind
            spec = _ind.load()
            if spec is not None:
                Xs = feat_only.reindex(columns=spec.feature_cols)
                p_spec = float(spec.predict_proba(Xs)[0])
                if spec.oof_auc > 0.51:
                    proba = 0.5 * proba + 0.5 * p_spec
                    kind += "+india"
                else:
                    log.info(f"india specialist auc={spec.oof_auc:.3f} <0.51, skipped")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"india specialist score failed: {exc}")

    # Phase 24.18 — meta-labelling shrinkage. Skip if the meta head's OOF AUC
    # is too weak to be useful (< 0.52 is essentially noise).
    if meta_head is not None:
        try:
            from omni.modeling.meta_label import apply_meta
            m_model, m_result = meta_head
            if m_result.auc_oof and m_result.auc_oof >= 0.52:
                # Build the meta feature row matching the trained schema.
                primary_dir = 1.0 if proba > 0.5 else -1.0
                meta_row = {
                    "primary_p": proba,
                    "primary_conviction": abs(proba - 0.5) * 200.0,
                    "primary_dir": primary_dir,
                }
                # Add risk-state cols (apply_meta will reindex to the trained order)
                for c in ("rsi_14", "atr_pct_14", "vol_ratio",
                          "dist_from_52w_high", "dist_from_52w_low",
                          "macd_hist_pct", "ret_5d", "ret_20d"):
                    if c in feat_only.columns:
                        v = feat_only[c].iloc[0]
                        meta_row[c] = float(v) if pd.notna(v) else 0.0
                meta_df = pd.DataFrame([meta_row])
                meta_p = float(apply_meta(m_model, meta_df, m_result.feature_cols)[0])
                # Shrink raw proba toward 0.5 by meta confidence.
                shrunk = 0.5 + (proba - 0.5) * meta_p
                log.debug(
                    f"meta shrink: raw={proba:.3f} meta_p={meta_p:.3f} "
                    f"-> shrunk={shrunk:.3f}"
                )
                proba = shrunk
                kind += f"+meta(p={meta_p:.2f})"
        except Exception as exc:  # noqa: BLE001
            log.warning(f"meta-labelling apply failed: {exc}")

    return proba, kind


def predict_batch(items: list[tuple[str, str]],
                   *, max_workers: int = 8) -> list["PredictionCard | None"]:
    """Score multiple tickers in one pass. Loads models ONCE, runs feature
    builds in a thread pool, then scores sequentially against the warm models.

    Args:
        items: list of (ticker, asset_class) tuples.
        max_workers: thread-pool size for feature lookups (I/O bound).

    Returns: list of PredictionCard|None aligned with `items`.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Warm the model cache once for the whole batch
    _ = _cached_models(_model_version_tag())

    def _one(t_ac):
        try:
            return predict_ticker(t_ac[0], t_ac[1])
        except Exception as e:  # noqa: BLE001
            log.warning(f"batch predict failed for {t_ac}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, items))


def _build_card(
    bundle: LookupBundle,
    feats_full: pd.DataFrame,
    proba_up: float,
    *,
    model_kind: str,
    fold_aucs: list[float] | None = None,
    per_horizon_proba: dict[int, float] | None = None,
) -> PredictionCard:
    last = feats_full.tail(1).copy()
    if last.empty:
        raise RuntimeError("feature frame is empty after build")
    # Drop forward-target cols on the live row (they will be NaN anyway)
    for c in [c for c in last.columns if c.startswith(("fwd_ret_", "fwd_dir_", "fwd_q_"))]:
        last[c] = np.nan

    regime_label = bundle.regime.label_today if bundle.regime else "chop"
    asof_ts = pd.to_datetime(last.iloc[0]["date"])
    regime_override = pd.Series({asof_ts: regime_label})

    sig_df = assemble_signals(
        last,
        proba_up=pd.Series([float(proba_up)], index=last.index),
        regime_override=regime_override,
    )
    if sig_df.empty:
        raise RuntimeError("assemble_signals produced no rows")
    row = sig_df.iloc[0]

    # Pillars come from the rubric output that lives back on `last` after
    # rubric_score_frame side-effected it. We re-run scorer here to capture them.
    # Phase 24.16 — augment the row with live OI features for India tickers
    # so the new `oi` pillar has data instead of falling back to neutral 50.
    feat_row = last.iloc[0].copy()
    if bundle.asset_class == "india":
        try:
            from omni.ingest.sources.nse_oi import oi_features
            oi_data = oi_features(bundle.ticker)
            if oi_data.get("has_option_chain"):
                if "oi_pcr" not in feat_row or pd.isna(feat_row.get("oi_pcr")):
                    feat_row["oi_pcr"] = float(oi_data.get("oi_pcr", 0))
                if "max_pain_dist_pct" not in feat_row or pd.isna(feat_row.get("max_pain_dist_pct")):
                    feat_row["max_pain_dist_pct"] = float(oi_data.get("max_pain_dist_pct", 0))
                if "oi_total_lakhs" not in feat_row or pd.isna(feat_row.get("oi_total_lakhs")):
                    feat_row["oi_total_lakhs"] = float(oi_data.get("oi_total_lakhs", 0))
        except Exception as exc:  # noqa: BLE001
            log.debug(f"OI augmentation skipped for {bundle.ticker}: {exc}")
    from omni.rubric.scorer import score_row
    rub = score_row(feat_row, regime=regime_label, asset_class=bundle.asset_class)
    pillars = {
        "technical": float(rub.technical_score),
        "fundamental": float(rub.fundamental_score),
        "sentiment": float(rub.sentiment_score),
        "macro": float(rub.macro_score),
        "microstructure": float(getattr(rub, "microstructure_score", 50.0)),
        "oi": float(getattr(rub, "oi_score", 50.0)),
        "relative_strength": float(getattr(rub, "relative_strength_score", 50.0)),
        "insider": float(getattr(rub, "insider_score", 50.0)),
        "earnings_momentum": float(getattr(rub, "earnings_momentum_score", 50.0)),
    }

    # Top 5 news headlines for the card
    news_top: list[dict] = []
    if bundle.news_raw is not None and not bundle.news_raw.empty:
        nh = bundle.news_raw.head(5)
        for _, r in nh.iterrows():
            news_top.append({
                "title": str(r.get("title", ""))[:160],
                "source": str(r.get("source", "")),
                "url": str(r.get("url", "")),
                "sent_score": float(r.get("sent_score", 0.0)) if not pd.isna(r.get("sent_score", np.nan)) else 0.0,
                "published_at": str(r.get("published_at", "")),
            })

    # LLM "why" — best-effort
    why, why_citations, why_grounding, why_path = _explain(
        bundle, pillars, regime_label, float(proba_up),
        direction=int(row.get("direction", 0)),
        conviction=float(row.get("conviction", 0.0)),
    )

    # ---- expected-move estimate + data coverage ----
    px = bundle.prices
    px_sorted = px.sort_values("date") if px is not None and not px.empty else px
    last_close = float(px_sorted["adj_close"].iloc[-1]) if px_sorted is not None and not px_sorted.empty else 0.0
    last_close_date = (str(pd.Timestamp(px_sorted["date"].iloc[-1]).date())
                        if px_sorted is not None and not px_sorted.empty else "")
    # Daily log-returns -> realised vol
    daily_ret = (px_sorted["adj_close"].pct_change().dropna()
                  if px_sorted is not None and not px_sorted.empty else pd.Series(dtype=float))
    sigma_d = float(daily_ret.tail(60).std()) if len(daily_ret) >= 20 else 0.0
    sigma_5d_pct = sigma_d * (5 ** 0.5) * 100.0  # 5-trading-day horizon, in percent
    edge = (float(row.get("blended_p", 0.5)) - 0.5) * 2.0  # in [-1, 1]
    expected_move_pct = float(edge * sigma_5d_pct)

    # Conformal prediction interval (Phase 22 / M4): if a calibrator was
    # persisted alongside the universal model, use its 80% coverage band.
    # Otherwise fall back to the heuristic ±1σ envelope.
    _conformal_used = False
    try:
        from omni.models.conformal import ConformalIntervalCalibrator
        from pathlib import Path as _P
        import joblib as _jl
        _cal_path = _P("data") / "models" / "conformal_h5.joblib"
        if _cal_path.exists():
            _cal: ConformalIntervalCalibrator = _jl.load(_cal_path)
            _lo, _hi = _cal.widen(expected_move_pct, sigma_5d_pct)
            expected_low_pct = float(_lo)
            expected_high_pct = float(_hi)
            _conformal_used = True
    except Exception as exc:  # noqa: BLE001
        # Conformal calibrator is optional; silent fallback to heuristic
        # was masking real loader bugs. Log at debug so they're discoverable
        # without spamming production.
        log.debug("conformal calibrator unavailable, using ±1σ band: %s", exc)
    if not _conformal_used:
        expected_low_pct = float(expected_move_pct - sigma_5d_pct)
        expected_high_pct = float(expected_move_pct + sigma_5d_pct)

    # Per-horizon expected moves. Drift scales LINEARLY with time (h/5),
    # vol scales as SQRT(h/5). The model is calibrated on 5d direction;
    # 1d and 1m are first-order extrapolations from the same edge UNLESS a
    # horizon-specific model is provided in ``per_horizon_proba`` (Step 5).
    def _project(h_days: int, edge_h: float | None = None) -> tuple[float, float, float]:
        sigma_h = sigma_d * (h_days ** 0.5) * 100.0
        if edge_h is not None:
            mu_h = float(edge_h * sigma_h)
        else:
            mu_h = expected_move_pct * (h_days / 5.0)
        return mu_h, mu_h - sigma_h, mu_h + sigma_h

    def _edge_from_proba(p: float | None) -> float | None:
        return None if p is None else (float(p) - 0.5) * 2.0

    p1d_edge = _edge_from_proba((per_horizon_proba or {}).get(1))
    p21_edge = _edge_from_proba((per_horizon_proba or {}).get(21))

    # ---- Intraday microstructure (Part A) ----
    intraday_feats: dict = {}
    intraday_tilt_applied = False
    try:
        from omni.ingest.intraday import fetch_intraday_bars
        from omni.features.intraday import compute_intraday_features
        ibars = fetch_intraday_bars(bundle.asset_class, bundle.ticker, days=5)
        intraday_feats = compute_intraday_features(ibars, last_close)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"intraday features skipped for {bundle.ticker}: {exc}")
        intraday_feats = {}

    # Apply bounded tilt to 1d edge (only if we have intraday data)
    if intraday_feats and intraday_feats.get("intraday_bar_count", 0) >= 3:
        ir = float(intraday_feats.get("intraday_return_pct", 0.0))
        vd = float(intraday_feats.get("vwap_dev_bps", 0.0))
        tilt = 0.5 * math.tanh(ir / 2.0 + vd / 200.0)
        # Convert tilt back to a proba shift bounded in [0.02, 0.98]
        if (per_horizon_proba or {}).get(1) is not None:
            p1 = float(per_horizon_proba[1])
            # tilt is roughly a probability shift in [-0.5, +0.5]; scale to gentle
            p1_adj = max(0.02, min(0.98, p1 + 0.15 * tilt))
            per_horizon_proba[1] = p1_adj
            p1d_edge = _edge_from_proba(p1_adj)
            intraday_tilt_applied = True

    m1d, m1d_lo, m1d_hi = _project(1, edge_h=p1d_edge)
    m1w, m1w_lo, m1w_hi = _project(5)        # 5d uses the primary `proba_up`
    m1m, m1m_lo, m1m_hi = _project(21, edge_h=p21_edge)
    # Annualised Sharpe over last 252 days (zero risk-free assumption)
    if len(daily_ret) >= 60:
        rr = daily_ret.tail(252)
        mu, sd = float(rr.mean()), float(rr.std())
        sharpe_1y = float((mu / sd) * (252 ** 0.5)) if sd > 1e-9 else 0.0
    else:
        sharpe_1y = 0.0

    # Latest intraday spot (best-effort — won't crash on failure)
    intraday: dict = {}
    try:
        from omni.ingest.prices import latest_intraday as _intra
        intraday = _intra(bundle.asset_class, bundle.ticker) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning(f"intraday hint failed for {bundle.ticker}: {exc}")

    fund_fields = 0
    if bundle.fundamentals is not None and not bundle.fundamentals.empty:
        fund_fields = int(bundle.fundamentals.iloc[0].notna().sum())

    coverage = {
        "price_rows": int(len(px_sorted)) if px_sorted is not None else 0,
        "price_first_date": (str(pd.Timestamp(px_sorted["date"].iloc[0]).date())
                              if px_sorted is not None and not px_sorted.empty else ""),
        "price_last_date": last_close_date,
        "history_years": (round(((pd.Timestamp(px_sorted["date"].iloc[-1])
                                  - pd.Timestamp(px_sorted["date"].iloc[0])).days / 365.25), 2)
                          if px_sorted is not None and len(px_sorted) > 1 else 0.0),
        "news_count": int(len(bundle.news_raw)) if bundle.news_raw is not None else 0,
        "news_sources": list(bundle.provenance.news_sources) if bundle.provenance else [],
        "prices_sources": list(bundle.provenance.prices_sources) if bundle.provenance else [],
        "fundamentals_sources": list(bundle.provenance.fundamentals_sources) if bundle.provenance else [],
        "fundamentals_fields": fund_fields,
        "macro_fresh": bool(bundle.provenance.macro_fresh) if bundle.provenance else False,
        "regime_fresh": bool(bundle.provenance.regime_fresh) if bundle.provenance else False,
        "data_quality_score": int(bundle.provenance.data_quality_score) if bundle.provenance else 0,
    }

    return PredictionCard(
        ticker=bundle.ticker,
        asset_class=bundle.asset_class,
        display_name=bundle.display_name,
        asof=asof_ts,
        direction=int(row.get("direction", 0)),
        conviction=float(row.get("conviction", 0.0)),
        blended_p=float(row.get("blended_p", 0.0)),
        proba_up=float(proba_up),
        rubric_total=float(rub.rubric_total),
        pillars=pillars,
        regime=regime_label,
        passes_gate=bool(row.get("passes_gate", False)),
        reasons=str(row.get("reasons", "")),
        why_text=why,
        news_top5=news_top,
        why_citations=why_citations,
        why_grounding_score=why_grounding,
        why_path=why_path,
        provenance=bundle.provenance.summary(),
        model_kind=model_kind,
        deep_analysis_available=(model_kind == "universal_5d"),
        fold_aucs=fold_aucs,
        expected_move_pct=expected_move_pct,
        expected_low_pct=expected_low_pct,
        expected_high_pct=expected_high_pct,
        move_1d_pct=m1d, move_1d_low_pct=m1d_lo, move_1d_high_pct=m1d_hi,
        move_1w_pct=m1w, move_1w_low_pct=m1w_lo, move_1w_high_pct=m1w_hi,
        move_1m_pct=m1m, move_1m_low_pct=m1m_lo, move_1m_high_pct=m1m_hi,
        recent_vol_5d_pct=sigma_5d_pct,
        realized_sharpe_1y=sharpe_1y,
        last_close=last_close,
        last_close_date=last_close_date,
        last_intraday_price=float(intraday.get("last_intraday_price", 0.0) or 0.0),
        last_intraday_at=str(intraday.get("last_intraday_at", "") or ""),
        data_coverage=coverage,
        intraday_return_pct=float(intraday_feats.get("intraday_return_pct", 0.0)),
        gap_open_pct=float(intraday_feats.get("gap_open_pct", 0.0)),
        vwap_dev_bps=float(intraday_feats.get("vwap_dev_bps", 0.0)),
        opening_range_breakout=float(intraday_feats.get("opening_range_breakout", 0.0)),
        intraday_realized_vol=float(intraday_feats.get("intraday_realized_vol", 0.0)),
        last_hour_return_pct=float(intraday_feats.get("last_hour_return_pct", 0.0)),
        volume_z_today=float(intraday_feats.get("volume_z_today", 0.0)),
        range_compression=float(intraday_feats.get("range_compression", 0.0)),
        intraday_bar_count=float(intraday_feats.get("intraday_bar_count", 0.0)),
        intraday_tilt_applied=bool(intraday_tilt_applied),
    )


def _deterministic_why(direction: int, regime: str, pillars: dict,
                          proba_up: float, conviction: float) -> str:
    """Fallback explanation when RAG is unavailable or ungrounded.

    Used when:
      - no news available for the ticker
      - RAG store unreachable / qdrant import failed
      - both LLM attempts came back ungrounded (grounding_score < threshold)
    """
    label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(direction), "FLAT")
    strongest = max(pillars.items(), key=lambda kv: abs(kv[1] - 50))
    return (f"{label} signal in a {regime} regime. Strongest pillar is "
             f"{strongest[0]} ({strongest[1]:.0f}/100). Model 5d up-probability "
             f"is {proba_up:.0%}. Conviction {conviction:.0f}/100.")


def _explain(bundle: LookupBundle, pillars: dict, regime: str,
              proba_up: float, *, direction: int, conviction: float,
              ) -> tuple[str, list[dict], float, str]:
    """Generate a why-text. Returns (text, citations, grounding_score, path).

    Phase 24.15 — wired through the RAG pipeline:
      1. Idempotently ingest bundle.news_raw into the local Qdrant store
      2. Retrieve top-5 most relevant chunks (BM25 + dense + reranker)
      3. Generate via signal_explanation_rag task chain
      4. Validate (citations, numeric grounding, sentence-cosine score)
      5. On any failure path: drop to deterministic explanation

    The path string indicates which branch executed for the UI to surface
    a small "rag-grounded" / "fallback" badge.
    """
    deterministic = _deterministic_why(direction, regime, pillars, proba_up, conviction)

    has_news = (bundle.news_raw is not None
                 and not bundle.news_raw.empty)
    if not has_news:
        return deterministic, [], 0.0, "deterministic"

    # Try the RAG path. Any exception falls through to deterministic.
    try:
        from omni.llm.rag.pipeline import get_pipeline
        pipe = get_pipeline()
        # Idempotent ingest — if we've seen these articles before, this is
        # a fast no-op (cached embeddings + identical chunk ids)
        pipe.ingest_news_for_ticker(
            bundle.news_raw, ticker=bundle.ticker,
            asset_class=bundle.asset_class,
        )
        resp = pipe.explain_signal(
            ticker=bundle.ticker,
            display_name=bundle.display_name,
            asset_class=bundle.asset_class,
            regime=regime,
            pillars=pillars,
            proba_up=float(proba_up),
            direction=int(direction),
            conviction=float(conviction),
        )
        if resp.is_safe and resp.text:
            return (resp.text, resp.citations, resp.grounding_score, "rag")
        if resp.text and resp.fallback_used == "low_grounding":
            # Keep the grounded-ish output but tag it so UI can warn
            return (resp.text, resp.citations, resp.grounding_score,
                     "rag_low_grounding")
        # Any other failure: fall back to deterministic
        log.debug("rag explain fell back: %s; issues=%s",
                   resp.fallback_used, resp.issues)
    except Exception as e:  # noqa: BLE001
        log.debug("rag pipeline error, using deterministic: %s", e)

    return deterministic, [], 0.0, "deterministic"


# ---------- public entry points ----------

def _model_version_tag() -> str:
    """File mtime of the 5d universal model bundle, formatted compactly.
    Including this in the cache key ensures predictions invalidate after a
    retrain (audit-doc B3).
    """
    try:
        blob, _ = model_registry._paths_for(5)
        if blob.exists():
            return f"v{int(blob.stat().st_mtime)}"
    except Exception as exc:  # noqa: BLE001
        # Model registry path discovery shouldn't normally fail; log at
        # debug so a missing _paths_for surface (e.g. registry rewrite)
        # is investigable instead of silently re-using "v0" forever.
        log.debug("model_version_tag fallback to v0: %s", exc)
    return "v0"


@cached(ttl=TTL.deep_analysis, key="predict_card:{ticker}:{asset_class}:{_mv}")
def _predict_ticker_versioned(ticker: str, asset_class: str,
                                *, display_name: str | None = None,
                                _mv: str = "v0") -> "PredictionCard | None":
    return _predict_ticker_impl(ticker, asset_class, display_name=display_name)


def predict_ticker(ticker: str, asset_class: str,
                    *, display_name: str | None = None) -> "PredictionCard | None":
    """Cached entry point. Cache key includes the model bundle's mtime so
    predictions automatically refresh after a retrain.
    """
    return _predict_ticker_versioned(
        ticker, asset_class, display_name=display_name,
        _mv=_model_version_tag(),
    )


def _predict_ticker_impl(ticker: str, asset_class: str,
                           *, display_name: str | None = None) -> "PredictionCard | None":
    bundle = lookup_ticker(ticker, asset_class, display_name=display_name)
    if bundle is None:
        return None

    feats = _last_row_features(bundle)
    if feats.empty:
        log.warning(f"predict: no features for {asset_class}/{ticker}")
        return None

    # Phase 24 — single feature alignment, shared across all model paths.
    from omni.models.xgboost_5d import _DROP_ALWAYS
    from omni.features.build import is_target_col
    last = feats.tail(1)
    feat_only = last[[c for c in last.columns
                       if c not in _DROP_ALWAYS and not is_target_col(c)]]
    feat_only = feat_only.apply(pd.to_numeric, errors="coerce")

    universal, stack_rec, _conformal, circuit_model, meta_head, meta_heads = _cached_models(_model_version_tag())
    proba, kind = _score_one(feat_only, universal=universal,
                              stack_rec=stack_rec, asset_class=asset_class,
                              meta_head=meta_head)

    # Multi-horizon predictions reuse the same feat_only frame.
    # h=5 already has meta-shrinkage applied via _score_one above.
    per_horizon: dict[int, float] = {5: proba}
    if universal is not None:
        for h in (1, 21):
            try:
                um_h = model_registry.load_horizon(h)
            except Exception:  # noqa: BLE001
                um_h = None
            if um_h is None:
                continue
            try:
                X_h = _align_to_universal(feat_only, um_h)
                p_h = float(um_h.calibrator.predict_proba(X_h)[0, 1])
                # Phase 24.18b — apply per-horizon meta-labelling shrinkage
                # to h=1 and h=21 just like _score_one does for h=5. h=1
                # benefits the most because its primary AUC is highest
                # (~0.69 with TB labels).
                meta_h_pair = meta_heads.get(h)
                if meta_h_pair is not None:
                    try:
                        from omni.modeling.meta_label import apply_meta
                        m_model_h, m_result_h = meta_h_pair
                        if m_result_h.auc_oof and m_result_h.auc_oof >= 0.52:
                            primary_dir_h = 1.0 if p_h > 0.5 else -1.0
                            meta_row = {
                                "primary_p": p_h,
                                "primary_conviction": abs(p_h - 0.5) * 200.0,
                                "primary_dir": primary_dir_h,
                            }
                            for c in ("rsi_14", "atr_pct_14", "vol_ratio",
                                      "dist_from_52w_high", "dist_from_52w_low",
                                      "macd_hist_pct", "ret_5d", "ret_20d"):
                                if c in feat_only.columns:
                                    v = feat_only[c].iloc[0]
                                    meta_row[c] = float(v) if pd.notna(v) else 0.0
                            meta_df = pd.DataFrame([meta_row])
                            meta_p_h = float(apply_meta(
                                m_model_h, meta_df, m_result_h.feature_cols
                            )[0])
                            shrunk_h = 0.5 + (p_h - 0.5) * meta_p_h
                            log.debug(
                                f"meta shrink h={h}: raw={p_h:.3f} "
                                f"meta_p={meta_p_h:.3f} -> {shrunk_h:.3f}"
                            )
                            p_h = shrunk_h
                    except Exception as exc:  # noqa: BLE001
                        log.warning(f"meta-labelling h={h} apply failed: {exc}")
                per_horizon[h] = p_h
            except Exception as exc:  # noqa: BLE001
                log.warning(f"horizon model h={h} predict failed: {exc}")

    # Phase 24.18 — circuit-risk score (India equities only). Computed AFTER
    # the main proba so a circuit-model failure can never break the card.
    circuit_pct: float | None = None
    circuit_band_str: str | None = None
    if asset_class == "india" and circuit_model is not None:
        try:
            from omni.models import circuit_risk as _crisk
            p_circ = float(circuit_model.predict_proba(feat_only)[0, 1])
            circuit_pct = p_circ
            circuit_band_str = _crisk.band(p_circ)
            log.debug(
                f"circuit_risk[{ticker}]: p={p_circ:.4f} band={circuit_band_str}"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"circuit_risk score failed [{ticker}]: {exc}")

    card = _build_card(bundle, feats, proba, model_kind=kind,
                        per_horizon_proba=per_horizon)
    if card is not None:
        card.circuit_risk_pct = circuit_pct
        card.circuit_risk_band = circuit_band_str

    # ---- Pair / cross-asset signal (Part B) ----
    try:
        from omni.ondemand.pair_signal import compute_pair_signal
        pair = compute_pair_signal(
            card.to_dict(),
            stock_prices=bundle.prices,
            bench_prices=None,  # let pair_signal fetch the bench prices
        )
        card.pair_signal = pair or {}
    except Exception as exc:  # noqa: BLE001
        log.warning(f"pair_signal failed for {ticker}: {exc}")
        card.pair_signal = {}

    # Log it (best-effort, non-blocking on failure)
    try:
        from omni.monitoring.predict_log import log_prediction
        log_prediction(card.to_dict(), horizon_days=5)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"predict_log failed: {exc}")
    return card


@cached(ttl=TTL.deep_analysis, key="deep_card:{ticker}:{asset_class}")
def deep_analysis(ticker: str, asset_class: str,
                  *, display_name: str | None = None) -> PredictionCard | None:
    bundle = lookup_ticker(ticker, asset_class, display_name=display_name)
    if bundle is None:
        return None

    feats = _last_row_features(bundle)
    feats = feats.dropna(subset=["fwd_dir_5d"])
    if len(feats) < 300:
        log.warning(f"deep_analysis: only {len(feats)} rows; falling back to universal")
        return predict_ticker(ticker, asset_class, display_name=display_name)

    try:
        res = walkforward_fit(feats, target="fwd_dir_5d",
                                n_folds=4, min_train=252, keep_models=True)
    except Exception as e:
        log.warning(f"deep_analysis walkforward failed: {e}")
        return predict_ticker(ticker, asset_class, display_name=display_name)

    if not res.folds:
        return predict_ticker(ticker, asset_class, display_name=display_name)

    last_fold = res.folds[-1]
    # Build the live (last-row) feature vector matching the fold's feature_cols
    feat_cols = res.feature_cols
    full = _last_row_features(bundle)  # re-build to keep the live row
    live = full.tail(1)[feat_cols].apply(pd.to_numeric, errors="coerce")
    medians = feats[feat_cols].apply(pd.to_numeric, errors="coerce").median()
    live = live.fillna(medians).fillna(0.0)
    proba_deep = float(last_fold.calibrator.predict_proba(live.to_numpy(dtype=float))[0, 1])

    # ---- M2 (Phase 22): Bayesian shrinkage toward universal ----------------
    # Small training set -> mostly universal; long history -> mostly deep.
    # Weight w = sqrt(n_deep / (n_deep + 252))  (252 = 1y prior strength).
    proba_universal = None
    shrink_w = 1.0
    try:
        um = model_registry.load()
        # .copy() because tail(1) returns a view; without it the np.nan
        # assignment below trips pandas' SettingWithCopyWarning (and will
        # become a hard error in pandas 3.x).
        feat_only = full.tail(1).copy()
        for c in [c for c in feat_only.columns
                    if c.startswith(("fwd_ret_", "fwd_dir_", "fwd_q_"))]:
            feat_only[c] = np.nan
        X_one = _align_to_universal(feat_only, um)
        proba_universal = float(um.calibrator.predict_proba(X_one)[0, 1])
        n_deep = int(len(feats))
        shrink_w = float(np.sqrt(n_deep / (n_deep + 252.0)))
        proba = shrink_w * proba_deep + (1.0 - shrink_w) * proba_universal
        log.info(
            f"deep_analysis shrinkage: deep={proba_deep:.3f} "
            f"univ={proba_universal:.3f} w={shrink_w:.2f} -> {proba:.3f}"
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deep shrinkage skipped: {exc}")
        proba = proba_deep

    # Per-fold AUC
    from sklearn.metrics import roc_auc_score
    fold_aucs: list[float] = []
    for f in res.folds:
        try:
            if len(set(f.y_true)) == 2:
                fold_aucs.append(float(roc_auc_score(f.y_true, f.proba)))
        except Exception as exc:  # noqa: BLE001
            # Single-class fold or sklearn version issue — skip but log
            # so a regression in the metric pipeline isn't silent.
            log.debug("deep_analysis: skip fold AUC: %s", exc)

    card = _build_card(bundle, full, proba,
                       model_kind="walk_forward_per_ticker", fold_aucs=fold_aucs)
    card.deep_analysis_available = False
    return card
