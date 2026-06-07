"""
Train baseline coral-cover models and write metric / diagnostic plots.

Usage:
    python -m src.models.run_baselines
    python -m src.models.run_baselines --n-splits 5 --inner-splits 3 --n-iter 20
    python -m src.models.run_baselines --holdout --test-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src import config
from src.models.baseline_features import (
    baseline_feature_spec,
    make_baseline_pipeline,
    prepare_baseline_fold_frames,
)
from src.models.baseline_models import (
    BASELINE_MODEL_NAMES,
    make_baseline_estimator,
    predict_coral_cover,
)
from src.models.baseline_plots import (
    plot_metrics_comparison,
    plot_observed_vs_predicted,
)
from src.models.baseline_tuning import TuningMethod, tune_baseline_estimator
from src.models.coral_data import coral_cover_target, load_model_ready_data
from src.models.hbb._config import CV_PREDICTORS
from src.models.metrics import regression_metrics


def _site_group_split(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sites = df["site"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(df)), groups=sites))
    return train_idx, test_idx


def _site_groups(df: pd.DataFrame) -> np.ndarray:
    return df["site"].astype(int).to_numpy()


def _safe_inner_splits(groups: np.ndarray, requested: int) -> int:
    n_unique = int(pd.Series(groups).nunique())
    return int(max(2, min(requested, n_unique)))


def _cv_predictions_with_tuning(
    *,
    df: pd.DataFrame,
    name: str,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    inner_splits: int,
    n_iter: int,
    seed: int,
    n_jobs: int,
    tuning_method: TuningMethod = "bayes",
    verbose: int = 0,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, float]]]:
    """Nested grouped CV with beta-aligned per-fold feature preparation."""
    outer = GroupKFold(n_splits=n_splits)
    y_pred = np.full_like(y, np.nan, dtype=float)
    fold_meta: list[dict[str, object]] = []
    fold_metrics: list[dict[str, float]] = []

    for fold_id, (train_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(df)), y, groups=groups), start=1
    ):
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        train_prep, test_prep, _ = prepare_baseline_fold_frames(
            df.iloc[train_idx], df.iloc[test_idx]
        )

        base_est = make_baseline_estimator(name, random_state=seed)
        pipe = make_baseline_pipeline(base_est)

        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"`sklearn\\.utils\\.parallel\\.delayed`.*",
        )
        inner_n = _safe_inner_splits(groups_train, inner_splits)
        inner_cv = GroupKFold(n_splits=inner_n)
        best, best_params, tuning = tune_baseline_estimator(
            pipe,
            name,  # type: ignore[arg-type]
            X_train=train_prep,
            y_train=y_train,
            groups_train=groups_train,
            inner_cv=inner_cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            seed=seed,
            method=tuning_method,
            verbose=verbose,
        )

        y_fold_pred = predict_coral_cover(best, test_prep)
        y_pred[test_idx] = y_fold_pred
        m = regression_metrics(y_test, y_fold_pred)
        fold_metrics.append(m)
        fold_meta.append(
            {
                "model": name,
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "best_params": best_params,
                **tuning,
            }
        )

    if not np.isfinite(y_pred).all():
        raise RuntimeError(
            "CV predictions contain NaNs; check CV splitting and model fitting."
        )
    return y_pred, fold_meta, fold_metrics


def run_baselines(
    *,
    data_path: Path | None = None,
    output_dir: Path | None = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    show: bool = False,
    cv: bool = True,
    n_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = 50,
    n_jobs: int = -1,
    tuning_method: TuningMethod = "bayes",
) -> pd.DataFrame:
    df = load_model_ready_data(data_path)
    y_all = coral_cover_target(df)
    groups = _site_groups(df)
    spec = baseline_feature_spec()
    feature_names = [*spec["fixed_effects"], "site_hier_logit"]

    if output_dir is None:
        output_dir = config.sully_og_dir / "output" / "baselines"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    metrics_rows: list[dict] = []
    scatter_preds: dict = {}
    pred_records = []
    fold_records = []

    if cv:
        print(f"Rows: {len(df)} | grouped CV: {n_splits}-fold (site)")
    else:
        train_idx, test_idx = _site_group_split(
            df, test_fraction=test_fraction, seed=seed
        )
        print(
            f"Rows: {len(df)} | train: {len(train_idx)} | test: {len(test_idx)} (grouped by site)"
        )

    print(
        f"Features ({len(feature_names)}): "
        f"{len(CV_PREDICTORS)} env (beta-aligned) + hierarchical site intercept"
    )

    for name in BASELINE_MODEL_NAMES:
        print(f"Fitting {name}...")
        try:
            if cv:
                y_pred, fold_meta, fold_metrics = _cv_predictions_with_tuning(
                    df=df,
                    name=name,
                    y=y_all,
                    groups=groups,
                    n_splits=n_splits,
                    inner_splits=inner_splits,
                    n_iter=n_iter,
                    seed=seed,
                    n_jobs=n_jobs,
                    tuning_method=tuning_method,
                )

                fold_df = pd.DataFrame(fold_metrics)
                mean_metrics = fold_df.mean(numeric_only=True).to_dict()
                std_metrics = fold_df.std(numeric_only=True, ddof=1).to_dict()
                metrics_rows.append(
                    {
                        "model": name,
                        **{k: float(v) for k, v in mean_metrics.items()},
                        **{f"{k}_std": float(v) for k, v in std_metrics.items()},
                    }
                )
                scatter_preds[name] = (y_all, y_pred)
                pred_records.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "y_obs": y_all,
                            "y_pred": y_pred,
                        }
                    )
                )
                fold_records.append(pd.DataFrame(fold_meta))

                last_tune = fold_meta[-1] if fold_meta else {}
                trials = last_tune.get("n_trials_completed", "?")
                requested = last_tune.get("n_trials_requested", "?")
                elapsed = last_tune.get("elapsed_sec", "?")
                print(
                    f"  CV mean R²={mean_metrics['r2']:.4f}  RMSE={mean_metrics['rmse']:.4f}  "
                    f"MAE={mean_metrics['mae']:.4f}  "
                    f"(tuning {trials}/{requested} trials, {elapsed}s/fold)"
                )
            else:
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)
                train_prep, test_prep, _ = prepare_baseline_fold_frames(
                    train_df, test_df
                )
                y_train = coral_cover_target(train_df)
                y_test = coral_cover_target(test_df)

                pipe = make_baseline_pipeline(
                    make_baseline_estimator(name, random_state=seed)
                )
                pipe.fit(train_prep, y_train)
                y_pred = predict_coral_cover(pipe, test_prep)
                m = regression_metrics(y_test, y_pred)
                metrics_rows.append({"model": name, **m})
                scatter_preds[name] = (y_test, y_pred)
                pred_records.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "y_obs": y_test,
                            "y_pred": y_pred,
                        }
                    )
                )
                print(
                    f"  test R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
                )
        except ImportError as e:
            print(f"  Skipping {name}: {e}")
            continue

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        output_dir / ("baseline_cv_metrics.csv" if cv else "baseline_metrics.csv"),
        index=False,
    )

    plot_metrics_comparison(
        metrics_df,
        output_path=output_dir
        / ("baseline_cv_r2_rmse.png" if cv else "baseline_r2_rmse.png"),
        show=show,
    )
    plot_observed_vs_predicted(
        scatter_preds,
        output_path=output_dir
        / (
            "baseline_cv_observed_vs_predicted.png"
            if cv
            else "baseline_observed_vs_predicted.png"
        ),
        show=show,
    )

    if pred_records:
        pd.concat(pred_records, ignore_index=True).to_csv(
            output_dir
            / (
                "baseline_cv_predictions.csv" if cv else "baseline_test_predictions.csv"
            ),
            index=False,
        )
    if fold_records and cv:
        fold_df = pd.concat(fold_records, ignore_index=True)
        fold_df["best_params"] = fold_df["best_params"].astype(str)
        fold_df.to_csv(output_dir / "baseline_cv_fold_details.csv", index=False)

    print(f"Wrote outputs to {output_dir}")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline coral-cover models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to data_for_maps.csv (default: sully_og/data_for_maps.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSVs and figures",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Hold-out fraction (grouped by site)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Use a single grouped hold-out split instead of cross-validation",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of outer CV folds (when --cv is set)",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
        help="Number of inner CV folds for tuning (ignored with --holdout)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Hyperparameter search trials per outer fold (ignored with --holdout)",
    )
    parser.add_argument(
        "--tuning",
        choices=("bayes", "random"),
        default="bayes",
        help="Inner-CV search: Bayesian optimization (skopt) or random search",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for hyperparameter search (ignored with --holdout)",
    )
    args = parser.parse_args()

    run_baselines(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        seed=args.seed,
        show=args.show,
        cv=not args.holdout,
        n_splits=args.n_splits,
        inner_splits=args.inner_splits,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        tuning_method=args.tuning,
    )


if __name__ == "__main__":
    main()
