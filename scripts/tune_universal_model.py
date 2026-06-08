"""Optuna hyperparameter tuner for the universal model.

Searches XGBoost hyperparameters via TPE with the existing walk-forward
fold schedule as the objective. Best params get written to
``config/xgb_best_params.yaml`` so the trainer picks them up automatically.

Usage:
  python scripts/tune_universal_model.py --trials 50 --target fwd_dir_5d
  python scripts/tune_universal_model.py --trials 200 --n-folds 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omni.lake import paths as lake_paths  # noqa: E402
from omni.utils.logger import log  # noqa: E402

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for universal_5d")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--target", default="fwd_dir_5d")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2520)
    p.add_argument("--training-set", default=None,
                    help="override path; default: data/lake/training/training_set.parquet")
    p.add_argument("--output", default="config/xgb_best_params.yaml")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _objective_factory(df: pd.DataFrame, target: str, n_folds: int, min_train: int):
    from sklearn.metrics import roc_auc_score

    from omni.models.xgboost_5d import walkforward_fit

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        try:
            res = walkforward_fit(df, target=target, n_folds=n_folds,
                                       min_train=min_train, keep_models=False,
                                       xgb_params=params)
        except TypeError:
            # walkforward_fit doesn't accept xgb_params yet — skip tuning
            log.warning("walkforward_fit doesn't accept xgb_params; "
                          "Optuna tuning will use defaults")
            res = walkforward_fit(df, target=target, n_folds=n_folds,
                                       min_train=min_train, keep_models=False)
        if not res.folds:
            return 0.5
        pred = res.predictions_frame()
        if pred["y_true"].nunique() != 2:
            return 0.5
        return float(roc_auc_score(pred["y_true"], pred["proba_up"]))

    return objective


def main() -> int:
    if not _HAS_OPTUNA:
        print("optuna is not installed. Run: pip install optuna")
        return 2
    args = _parse_args()
    ts_path = (Path(args.training_set) if args.training_set else
               lake_paths.lake_root() / "training" / "training_set.parquet")
    if not ts_path.exists():
        print(f"ERROR: training set not found at {ts_path}\n"
               f"Run: python scripts/build_training_set.py")
        return 1
    print(f"loading training set: {ts_path}")
    df = pd.read_parquet(ts_path)
    df = df.dropna(subset=[args.target])
    print(f"  rows: {len(df):,}, target: {args.target}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    obj = _objective_factory(df, args.target, args.n_folds, args.min_train)
    t0 = time.time()
    study.optimize(obj, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.time() - t0

    print(f"\ntrials done in {elapsed/60:.1f}m")
    print(f"  best AUC:    {study.best_value:.5f}")
    print(f"  best params: {json.dumps(study.best_params, indent=2)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuned_at": pd.Timestamp.utcnow().isoformat(),
        "target": args.target,
        "best_auc": float(study.best_value),
        "n_trials": args.trials,
        "params": study.best_params,
    }
    if _HAS_YAML:
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        out.with_suffix(".json").write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
