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

import math

import numpy as np
import pandas as pd

from omni.ensemble.pipeline import assemble_signals
from omni.llm.backends.base import BackendError
from omni.llm.router import route as _llm_route
from omni.models.xgboost_5d import walkforward_fit
from omni.ondemand import model_registry
from omni.ondemand.cache import TTL, cached
from omni.ondemand.features_inline import build_features_inline
from omni.ondemand.lookup import LookupBundle, lookup_ticker
from omni.ondemand.quality import Provenance
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
    provenance: dict = field(default_factory=dict)
    model_kind: str = "universal_5d"
    deep_analysis_available: bool = True
    fold_aucs: list[float] | None = None  # only set by deep_analysis
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
    from omni.rubric.scorer import score_row
    rub = score_row(last.iloc[0], regime=regime_label, asset_class=bundle.asset_class)
    pillars = {
        "technical": float(rub.technical_score),
        "fundamental": float(rub.fundamental_score),
        "sentiment": float(rub.sentiment_score),
        "macro": float(rub.macro_score),
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
    why = _explain(bundle, pillars, regime_label, float(proba_up),
                    direction=int(row.get("direction", 0)),
                    conviction=float(row.get("conviction", 0.0)))

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


def _explain(bundle: LookupBundle, pillars: dict, regime: str,
             proba_up: float, *, direction: int, conviction: float) -> str:
    label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(direction, "FLAT")
    prompt = (
        f"Stock: {bundle.display_name} ({bundle.ticker}, {bundle.asset_class.upper()})\n"
        f"Today: regime={regime}, model proba(up,5d)={proba_up:.3f}, "
        f"rubric pillars: tech={pillars['technical']:.0f}, "
        f"fund={pillars['fundamental']:.0f}, sent={pillars['sentiment']:.0f}, "
        f"macro={pillars['macro']:.0f}. "
        f"Final signal: {label} (conviction {conviction:.0f}/100).\n\n"
        f"Write 3-4 sentences (max 80 words) explaining WHY this signal "
        f"is what it is, in plain language, citing the strongest pillar "
        f"and the regime. No financial advice disclaimer."
    )
    try:
        res = _llm_route("signal_explanation", prompt)
        return (res.text if hasattr(res, "text") else str(res)).strip()
    except (BackendError, Exception) as e:
        log.debug(f"LLM explain skipped: {e}")
        # Deterministic fallback
        strongest = max(pillars.items(), key=lambda kv: abs(kv[1] - 50))
        return (f"{label} signal in a {regime} regime. Strongest pillar is "
                 f"{strongest[0]} ({strongest[1]:.0f}/100). Model 5d up-probability "
                 f"is {proba_up:.0%}. Conviction {conviction:.0f}/100.")


# ---------- public entry points ----------

def _model_version_tag() -> str:
    """File mtime of the 5d universal model bundle, formatted compactly.
    Including this in the cache key ensures predictions invalidate after a
    retrain (audit-doc B3).
    """
    try:
        from pathlib import Path
        blob, _ = model_registry._paths_for(5)
        if blob.exists():
            return f"v{int(blob.stat().st_mtime)}"
    except Exception:
        pass
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

    try:
        um = model_registry.load()
    except FileNotFoundError as e:
        log.warning(f"universal model missing: {e}")
        # Fall back to a deterministic 0.5 proba rather than crashing.
        proba = 0.5
        kind = "rubric_only"
        per_horizon: dict[int, float] = {}
    else:
        # Pull the last row's features
        from omni.models.xgboost_5d import _DROP_ALWAYS
        from omni.features.build import is_target_col
        last = feats.tail(1)
        feat_only = last[[c for c in last.columns
                           if c not in _DROP_ALWAYS and not is_target_col(c)]]
        feat_only = feat_only.apply(pd.to_numeric, errors="coerce")
        X = _align_to_universal(feat_only, um)
        proba = float(um.calibrator.predict_proba(X)[0, 1])
        kind = "universal_5d"

        # ---- Stacked-ensemble override (stack_h5.pkl) ----
        # When data/models/stack_h5.pkl exists and its meta-AUC beats the
        # universal's OOF AUC, use the stack instead. The stack was trained
        # on the rank target with purged K-fold + embargo (de Prado).
        try:
            import pickle
            from pathlib import Path
            stack_path = Path("data/models/stack_h5.pkl")
            if stack_path.exists():
                with stack_path.open("rb") as f:
                    sb = pickle.load(f)
                sr = sb.get("stack")
                if sr is not None and sr.oof_meta_auc > (um.oof_auc or 0.5):
                    Xs = feat_only.reindex(columns=sr.feature_cols)
                    p_stack = float(sr.predict_proba(Xs)[0])
                    proba = p_stack
                    kind = "stack_h5_rank_purged"
        except Exception as exc:  # noqa: BLE001
            log.warning(f"stack override predict failed: {exc}")

        # ---- India specialist blend (only for india tickers) ----
        if asset_class == "india":
            try:
                from omni.specialists import india as _ind
                spec = _ind.load()
                if spec is not None:
                    Xs = feat_only.reindex(columns=spec.feature_cols)
                    p_spec = float(spec.predict_proba(Xs)[0])
                    # Blend: 50/50 with universal, but only if specialist OOF AUC > 0.51
                    if spec.oof_auc > 0.51:
                        proba = 0.5 * proba + 0.5 * p_spec
                        kind = "universal+india_specialist"
                    else:
                        # Specialist underperforms — keep universal but log
                        log.info(f"india specialist auc={spec.oof_auc:.3f} below 0.51, not blended")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"india specialist predict failed: {exc}")

        # Multi-horizon: try to load 1d / 21d models too. Each computes its
        # own X alignment because feature_cols may differ per horizon.
        per_horizon = {5: proba}
        for h in (1, 21):
            try:
                um_h = model_registry.load_horizon(h)
            except Exception:  # noqa: BLE001
                um_h = None
            if um_h is None:
                continue
            try:
                X_h = _align_to_universal(feat_only, um_h)
                per_horizon[h] = float(um_h.calibrator.predict_proba(X_h)[0, 1])
            except Exception as exc:  # noqa: BLE001
                log.warning(f"horizon model h={h} predict failed: {exc}")

    card = _build_card(bundle, feats, proba, model_kind=kind,
                        per_horizon_proba=per_horizon if 'per_horizon' in locals() else None)

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
    proba = float(last_fold.calibrator.predict_proba(live.to_numpy(dtype=float))[0, 1])

    # Per-fold AUC
    from sklearn.metrics import roc_auc_score
    fold_aucs: list[float] = []
    for f in res.folds:
        try:
            if len(set(f.y_true)) == 2:
                fold_aucs.append(float(roc_auc_score(f.y_true, f.proba)))
        except Exception:
            pass

    card = _build_card(bundle, full, proba,
                       model_kind="walk_forward_per_ticker", fold_aucs=fold_aucs)
    card.deep_analysis_available = False
    return card
