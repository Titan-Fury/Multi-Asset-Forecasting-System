"""Walk-forward backtest runner.

Stitches the pieces together:

  features (from omni.features.build)
        |
        v
  walk-forward XGBoost  ->  WalkForwardResult.predictions_frame()
        |
        v
  rubric scores (regime-conditioned)  ->  blend  ->  gate
        |
        v
  per-fold + aggregated metrics + checks

Public API:
  run_backtest(features, regime_labels=None, ...) -> BacktestReport
  BacktestReport.summary  -- dict of aggregate metrics
  BacktestReport.predictions  -- per-row predictions frame
  BacktestReport.signals     -- gated signal frame
  BacktestReport.fold_metrics -- list of per-fold dicts
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from omni.backtest import checks as _checks
from omni.backtest import metrics as _metrics
from omni.ensemble.combine import BlendConfig, blend
from omni.ensemble.gate import GateConfig, apply as gate_apply
from omni.models.xgboost_5d import walkforward_fit
from omni.rubric.scorer import score_frame as rubric_score_frame
from omni.utils.logger import log


@dataclass
class BacktestReport:
    summary: dict = field(default_factory=dict)
    fold_metrics: list[dict] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_cols: list[str] = field(default_factory=list)
    leakage_ok: bool = False
    chronology_ok: bool = False


def run_backtest(features: pd.DataFrame,
                 *,
                 regime_labels: pd.Series | None = None,
                 target: str = "fwd_dir_5d",
                 n_folds: int = 5,
                 min_train: int = 504,
                 blend_cfg: BlendConfig | None = None,
                 gate_cfg: GateConfig | None = None,
                 run_shuffled_baseline: bool = True) -> BacktestReport:
    """Run the full walk-forward backtest.

    Args:
      features: a feature frame from omni.features.build with `date`,
        `ticker`, `asset_class`, all features, and target columns.
      regime_labels: optional Series indexed by date -> regime str. If None,
        all rows treated as 'chop'.
    """
    rep = BacktestReport()

    # --- model walk-forward
    wf = walkforward_fit(features, target=target, n_folds=n_folds, min_train=min_train)
    rep.feature_cols = wf.feature_cols

    # --- leakage / chronology checks
    _checks.assert_no_target_leakage(rep.feature_cols)
    _checks.assert_chronological_splits(wf.folds)
    rep.leakage_ok = True
    rep.chronology_ok = True

    preds = wf.predictions_frame()
    if preds.empty:
        log.warning("walkforward produced no predictions")
        return rep

    # Attach realized fwd_ret + asset_class for downstream metrics
    base = features[["date", "ticker", "asset_class"]].copy()
    if "fwd_ret_5d" in features.columns:
        base["fwd_ret_5d"] = features["fwd_ret_5d"].values
    base["date"] = pd.to_datetime(base["date"])
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.merge(base, on=["date", "ticker"], how="left")

    # Per-fold metrics
    fold_dicts = []
    for fold in sorted(preds["fold"].unique()):
        sub = preds[preds["fold"] == fold]
        fm = _metrics.summary(sub)
        fm["fold"] = int(fold)
        fold_dicts.append(fm)
    rep.fold_metrics = fold_dicts
    rep.summary = _metrics.summary(preds)

    # --- rubric + blend + gate on the union of test rows
    # Use feature row alignment via (date, ticker)
    feat_test = features.merge(preds[["date", "ticker", "proba_up"]],
                               on=["date", "ticker"], how="inner")
    if regime_labels is not None:
        rmap = pd.Series(regime_labels.values, index=pd.to_datetime(regime_labels.index))
        feat_test["regime"] = pd.to_datetime(feat_test["date"]).map(rmap).fillna("chop")
    else:
        feat_test["regime"] = "chop"

    scored = rubric_score_frame(feat_test, regimes=feat_test["regime"])
    blended = blend(scored, cfg=blend_cfg)
    gated = gate_apply(blended, cfg=gate_cfg)

    rep.predictions = preds
    rep.signals = gated[[
        "date", "ticker", "asset_class", "regime", "proba_up", "rubric_total",
        "blended_p", "conviction", "direction", "passes_gate", "reasons",
    ] + (["fwd_ret_5d"] if "fwd_ret_5d" in gated.columns else [])].copy()

    # Gated metrics need fwd_ret_5d + y_true
    if "fwd_ret_5d" in gated.columns:
        gated_with_y = gated.copy()
        gated_with_y["y_true"] = (gated_with_y["fwd_ret_5d"] > 0).astype(int)
        rep.summary["gated"] = _metrics.gated_metrics(gated_with_y)

    # --- shuffled baseline
    if run_shuffled_baseline:
        def _wf(shuffled_features):
            sw = walkforward_fit(shuffled_features, target=target,
                                 n_folds=n_folds, min_train=min_train)
            return sw.predictions_frame()
        try:
            base_metrics = _checks.shuffled_baseline(features, target=target,
                                                    walkforward_fn=_wf)
            rep.summary["shuffled_baseline"] = base_metrics
            rep.summary["edge"] = _checks.edge_over_baseline(rep.summary, base_metrics)
        except Exception as e:
            log.warning(f"shuffled baseline failed: {e}")

    log.info(f"backtest done: hit_rate={rep.summary.get('hit_rate'):.3f} "
             f"auc={rep.summary.get('auc')}")
    return rep
