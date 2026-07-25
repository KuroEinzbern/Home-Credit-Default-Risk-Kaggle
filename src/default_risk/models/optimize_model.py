
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable
import yaml
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.models import infer_signature
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
import default_risk.config as cfg

from default_risk.scripts.auxiliars_for_modeling import cast_object_into_categoricals
from default_risk.scripts.auxiliars_for_modeling import get_baseline_setup
from default_risk.scripts.auxiliars_for_modeling import prepare_columns
from default_risk.scripts.auxiliars_for_modeling import get_pipeline

STUDY_NAME = "optuna-optimization-1.1"
N_TRIALS = 60
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
SEED = 42

TUNED_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "gamma",
)


@dataclass
class CvResult:
    oof_auc: float
    fold_scores: list[float]
    best_iterations: list[int]

    @property
    def fold_mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def fold_std(self) -> float:
        return float(np.std(self.fold_scores))

    @property
    def rounds_for_full_refit(self) -> int:
        """Median best round, scaled up because the refit sees k/(k-1) more rows."""
        k = len(self.best_iterations)
        return max(1, int(np.median(self.best_iterations) * k / (k - 1)))


def suggest_params(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params = {
        **base,
        "grow_policy": grow_policy,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0, log=True),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 300, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 128, log=True)
        params["max_depth"] = 0
    else:
        params["max_depth"] = trial.suggest_int("max_depth", 4, 10)
    return params


def fit_fold(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    """Fit the preprocessor on the training fold only, then the estimator.

    Slicing a Pipeline returns a view over the *same* step objects, so fitting
    the slices fits `pipeline` itself. Callers must pass a cloned pipeline.
    fit_transform (not fit + transform) is what makes the target encoder use its
    internal cross-fitting on the training rows.
    """
    estimator = pipeline[-1]
    if len(pipeline.steps) > 1:
        preprocessor = pipeline[:-1]
        x_train_t = preprocessor.fit_transform(x_train, y_train)
        x_val_t = preprocessor.transform(x_val)
    else:
        x_train_t, x_val_t = x_train, x_val

    estimator.fit(x_train_t, y_train, eval_set=[(x_val_t, y_val)], verbose=False)
    return estimator, x_val_t


def run_cv(
    pipeline: Pipeline,
    cv: BaseCrossValidator,
    X: pd.DataFrame,
    Y: pd.Series,
    *,
    report_fn: Callable[[int, float], None] | None = None,
    importance_prefix: str | None = None,
) -> CvResult:
    y_np = Y.to_numpy()
    oof_prob = np.zeros(len(Y))
    scored = np.zeros(len(Y), dtype=bool)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    for fold_idx, (train_index, val_index) in enumerate(cv.split(X, Y)):
        estimator, x_val_t = fit_fold(
            clone(pipeline),
            X.iloc[train_index],
            Y.iloc[train_index],
            X.iloc[val_index],
            Y.iloc[val_index],
        )
        default_prob = estimator.predict_proba(x_val_t)[:, 1]
        oof_prob[val_index] = default_prob
        scored[val_index] = True

        fold_scores.append(roc_auc_score(y_np[val_index], default_prob))
        best_iter = getattr(estimator, "best_iteration", None)
        best_iterations.append(
            estimator.n_estimators if best_iter is None else best_iter + 1
        )

        if importance_prefix is not None:
            save_feature_importance(
                estimator, x_val_t, f"{importance_prefix}_fold{fold_idx + 1}"
            )
        if report_fn is not None:
            report_fn(fold_idx, roc_auc_score(y_np[scored], oof_prob[scored]))

    return CvResult(
        oof_auc=float(roc_auc_score(y_np, oof_prob)),
        fold_scores=fold_scores,
        best_iterations=best_iterations,
    )


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    Y: pd.Series,
    cv: BaseCrossValidator,
    base_params: dict[str, Any],
    build_pipeline: Callable[[dict[str, Any]], Pipeline],
) -> float:
    pipeline = build_pipeline(suggest_params(trial, base_params))

    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.log_params(trial.params)

        def report(fold_idx: int, partial_oof: float) -> None:
            mlflow.log_metric("partial_oof_auc", partial_oof, step=fold_idx)
            trial.report(partial_oof, fold_idx)
            if trial.should_prune():
                mlflow.set_tag("pruned_at_fold", fold_idx + 1)
                raise optuna.TrialPruned()

        result = run_cv(pipeline, cv, X, Y, report_fn=report)
        mlflow.log_metrics(
            {
                "oof_auc": result.oof_auc,
                "cv_mean_auc": result.fold_mean,
                "cv_std": result.fold_std,
                "median_best_iteration": float(np.median(result.best_iterations)),
            }
        )

    trial.set_user_attr("cv_std", result.fold_std)
    trial.set_user_attr("rounds_for_full_refit", result.rounds_for_full_refit)
    return result.oof_auc


def main() -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")

    cv, hiperparams = get_baseline_setup()
    dataset = pd.read_parquet(cfg.MASTER_DATA_DIR / "prepared_dataset.parquet")
    X, Y = prepare_columns(dataset)
    X = cast_object_into_categoricals(X)

    features_for_target_encoding: list[str] = []

    base_params: dict[str, Any] = {
        **{k: v for k, v in hiperparams.items() if k not in TUNED_KEYS},
        "n_estimators": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "eval_metric": "auc",
        "random_state": SEED,
    }

    def build_pipeline(params: dict[str, Any]) -> Pipeline:
        return get_pipeline(
            50, features_for_target_encoding, xgb.XGBClassifier(**params)
        )

    mlflow.set_experiment(experiment_name)
    mlflow.xgboost.autolog(disable=True) 
    mlflow.lightgbm.autolog(disable=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=SEED, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        storage=f"sqlite:///{cfg.MODELS_DIR / 'optuna.db'}",
        load_if_exists=True,
    )

    with mlflow.start_run(run_name="optuna_search"):
        study.optimize(
            lambda trial: objective(trial, X, Y, cv, base_params, build_pipeline),
            n_trials=N_TRIALS,
            n_jobs=1,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_oof_auc", study.best_value)
        study.trials_dataframe().to_csv(
            cfg.ARTIFACTS_DIR / f"{STUDY_NAME}_trials.csv", index=False
        )
        mlflow.log_artifact(str(cfg.ARTIFACTS_DIR / f"{STUDY_NAME}_trials.csv"))

    best_params = {**base_params, **study.best_params}
    print(f"best oof_auc={study.best_value:.5f} with {study.best_params}")

    best_params = {**base_params, **study.best_params}
    output_config = {
        "xgboost": {
            k: v for k, v in best_params.items() 
            if k not in ["eval_metric", "random_state"]
        }
    }

    with open(cfg.PROJECT_ROOT / "params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(output_config, f, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    main()