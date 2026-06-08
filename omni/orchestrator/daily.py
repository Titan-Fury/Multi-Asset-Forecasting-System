"""Daily orchestrator.

Pipeline:
    mount Drive  ->  ingest  ->  features  ->  regime  ->  predict
        ->  rubric  ->  blend  ->  gate  ->  persist  ->  unmount Drive

Each stage is wrapped in try/except so a partial failure (e.g. one bad
data source) doesn't take the whole run down. Stage errors are recorded
in the run manifest.

Two entry points:

  run_daily(...)           -- in-process, returns RunReport
  if __name__ == "__main__": forwards to omni.orchestrator.cli

Keep this file orchestration-only -- no business logic. Each stage
delegates to the corresponding module so unit tests can target stages
in isolation.
"""
from __future__ import annotations

import functools
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from omni.orchestrator import persist as _persist
from omni.utils.logger import log


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    error: str | None = None
    notes: dict = field(default_factory=dict)


@dataclass
class RunReport:
    run_date: str
    stages: list[StageResult] = field(default_factory=list)
    persist: _persist.PersistResult | None = None
    notes: list[str] = field(default_factory=list)

    def add(self, sr: StageResult) -> None:
        self.stages.append(sr)
        emoji = "OK" if sr.ok else "FAIL"
        log.info(f"[{emoji}] {sr.name} in {sr.seconds:.1f}s "
                 f"{('-- ' + sr.error) if sr.error else ''}")

    def note(self, msg: str) -> None:
        """Record a non-stage breadcrumb (best-effort sub-step)."""
        self.notes.append(msg)
        log.info(f"[NOTE] {msg}")

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def manifest(self) -> dict:
        return {
            "run_date": self.run_date,
            "ok": self.ok,
            "stages": [s.__dict__ for s in self.stages],
            "notes": list(self.notes),
        }


def _stage(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(rep: RunReport, *args, **kw):
            t0 = time.perf_counter()
            try:
                out = fn(rep, *args, **kw)
                rep.add(StageResult(name=name, ok=True,
                                    seconds=time.perf_counter() - t0))
                return out
            except Exception as e:
                rep.add(StageResult(
                    name=name, ok=False,
                    seconds=time.perf_counter() - t0,
                    error=f"{type(e).__name__}: {e}",
                    notes={"trace": traceback.format_exc(limit=5)},
                ))
                return None
        return wrapper
    return deco


# ----- stage implementations (lazy imports keep the orchestrator importable
#       even if optional stages aren't configured yet) -----

@_stage("mount_drive")
def _stage_mount(rep: RunReport):
    from omni.storage import mount as _mount
    _mount.mount()


@_stage("ingest")
def _stage_ingest(rep: RunReport, *, run_date: str):
    # Phase 25.1 — snapshot the alt-data archives FIRST so per-ticker
    # predictions later in the run can read accumulated history. These
    # are best-effort: any failure is logged but doesn't abort ingest.
    try:
        from omni.ondemand.fii_dii_archive import snapshot_now as _fii_snap
        hist = _fii_snap()
        if hist is not None and not hist.empty:
            rep.note(f"alt-data: fii_dii archive size = {len(hist)}")
    except Exception as exc:  # noqa: BLE001
        rep.note(f"alt-data fii_dii snapshot failed (best-effort): {exc}")
    try:
        from omni.ondemand.deals_archive import snapshot_now as _deals_snap
        sizes = _deals_snap()
        rep.note(f"alt-data: deals archive sizes = {sizes}")
    except Exception as exc:  # noqa: BLE001
        rep.note(f"alt-data deals snapshot failed (best-effort): {exc}")
    # Pre-warm OI live cache for the F&O universe (best-effort, slow if
    # NSE is rate-limiting; one snapshot per ticker is enough per day).
    try:
        from omni.ingest.sources.nse_oi import _KNOWN_FNO_SAMPLE
        from omni.ondemand.oi_live_cache import snapshot_now as _oi_snap
        n_ok = 0
        for sym in _KNOWN_FNO_SAMPLE:
            try:
                out = _oi_snap(sym)
                if out.get("has_option_chain"):
                    n_ok += 1
            except Exception:  # noqa: BLE001
                pass
        rep.note(f"alt-data: OI cache prewarm hit {n_ok}/{len(_KNOWN_FNO_SAMPLE)}")
    except Exception as exc:  # noqa: BLE001
        rep.note(f"alt-data OI prewarm failed (best-effort): {exc}")

    from omni.ingest import runner as _ing
    return _ing.run_daily_ingest(run_date=run_date)


@_stage("features")
def _stage_features(rep: RunReport, *, run_date: str):
    """Build per-ticker feature frames across the universe and return one
    concatenated frame for downstream stages."""
    from omni.features import macro as _macro
    from omni.features.build import build_for_ticker
    from omni.utils.env import universe as _universe

    cfg = _universe()
    macro = _macro.compute()

    frames = []
    for asset_class, spec in cfg["asset_classes"].items():
        for ticker in spec["tickers"]:
            try:
                df = build_for_ticker(asset_class, ticker, macro=macro)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                log.warning(f"features {asset_class}/{ticker} failed: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Stash macro on the frame for the regime stage to reuse
    out.attrs["macro"] = macro
    return out


@_stage("regime")
def _stage_regime(rep: RunReport, features):
    """Fit the HMM on a macro frame, not on per-ticker features. The macro
    frame is attached to the features frame by _stage_features via .attrs."""
    from omni.regime.hmm import build_regime_features, fit
    macro = features.attrs.get("macro") if features is not None else None
    if macro is None or macro.empty:
        log.warning("regime: no macro frame available, skipping")
        return pd.DataFrame(columns=["date", "regime"])
    rfeat = build_regime_features(macro)
    if rfeat.empty or len(rfeat) < 100:
        log.warning(f"regime: not enough rows to fit HMM ({len(rfeat)})")
        return pd.DataFrame(columns=["date", "regime"])
    model = fit(rfeat)
    labels = model.predict(rfeat)
    return pd.DataFrame({"date": pd.to_datetime(rfeat["date"]), "regime": labels})


@_stage("predict")
def _stage_predict(rep: RunReport, features, *, target: str = "fwd_dir_5d"):
    from omni.models.xgboost_5d import walkforward_fit
    wf = walkforward_fit(features, target=target)
    return wf.predictions_frame()


@_stage("rubric_blend_gate")
def _stage_signals(rep: RunReport, features, predictions, regime_df):
    from omni.ensemble.pipeline import assemble_signals
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    proba = predictions.set_index(["date", "ticker"])["proba_up"] \
        if "ticker" in predictions.columns else predictions.set_index("date")["proba_up"]
    regime_override = None
    if regime_df is not None and not regime_df.empty:
        regime_override = pd.Series(regime_df["regime"].values,
                                     index=pd.to_datetime(regime_df["date"]))
    return assemble_signals(features, proba_up=proba,
                            regime_override=regime_override)


@_stage("backtest")
def _stage_backtest(rep: RunReport, features, regime_df):
    from omni.backtest.runner import run_backtest
    rl = pd.Series(regime_df["regime"].values,
                   index=pd.to_datetime(regime_df["date"])) \
        if regime_df is not None and not regime_df.empty else None
    return run_backtest(features, regime_labels=rl, run_shuffled_baseline=False)


@_stage("persist")
def _stage_persist(rep: RunReport, *, predictions, signals, regime_df,
                   backtest_report, features) -> _persist.PersistResult:
    bt_summary = backtest_report.summary if backtest_report else None
    fold_metrics = backtest_report.fold_metrics if backtest_report else None
    feat_latest = None
    if features is not None and not features.empty and "date" in features.columns:
        last_date = pd.to_datetime(features["date"]).max()
        feat_latest = features[pd.to_datetime(features["date"]) == last_date].copy()
    return _persist.persist_run(
        predictions=predictions,
        signals=signals,
        regime_labels=regime_df,
        backtest_summary=bt_summary,
        fold_metrics=fold_metrics,
        features_latest=feat_latest,
        run_date=rep.run_date,
    )


@_stage("unmount_drive")
def _stage_unmount(rep: RunReport):
    from omni.storage import mount as _mount
    _mount.unmount()


# ----- top-level entry point -----

def run_daily(*,
              run_date: str | None = None,
              skip_backtest: bool = False,
              skip_mount: bool = False) -> RunReport:
    run_date = run_date or datetime.now(UTC).strftime("%Y-%m-%d")
    rep = RunReport(run_date=run_date)
    log.info(f"=== Omni_Stocks daily run {run_date} ===")

    if not skip_mount:
        _stage_mount(rep)

    _stage_ingest(rep, run_date=run_date)
    features = _stage_features(rep, run_date=run_date)

    regime_df = _stage_regime(rep, features) if features is not None else None
    predictions = _stage_predict(rep, features) if features is not None else None
    signals = _stage_signals(rep, features, predictions, regime_df) \
        if features is not None else None

    backtest = None
    if not skip_backtest and features is not None:
        backtest = _stage_backtest(rep, features, regime_df)

    rep.persist = _stage_persist(
        rep,
        predictions=predictions,
        signals=signals,
        regime_df=regime_df,
        backtest_report=backtest,
        features=features,
    )

    if not skip_mount:
        _stage_unmount(rep)

    log.info(f"=== run {run_date} {'OK' if rep.ok else 'FAILED'} "
             f"({sum(s.seconds for s in rep.stages):.1f}s total) ===")
    return rep


if __name__ == "__main__":
    from omni.orchestrator.cli import main as _main
    _main()
