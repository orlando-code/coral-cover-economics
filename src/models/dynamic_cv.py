"""Cross-validation for persistence-adjusted dynamic projection models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.coral_data import coral_cover_target
from src.models.cv_methods import FoldSpec, fold_manifest_dataframe
from src.models.dynamic_features import dynamic_feature_spec, prepare_dynamic_fold_frames
from src.models.dynamic_models import (
    DYNAMIC_DISPLAY_NAMES,
    DYNAMIC_MODEL_NAMES,
    make_dynamic_model,
)
from src.models.metrics import projection_metrics, regression_metrics


def run_dynamic_cv(
    *,
    df: pd.DataFrame,
    folds: list[FoldSpec],
    output_dir: Path,
    dynamic_models: list[str],
    seed: int,
    n_jobs: int = -1,
    log_fn=None,
) -> None:
    """Run dynamic projection models across CV folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y = coral_cover_target(df)
    fold_manifest_dataframe(folds).to_csv(output_dir / "fold_manifest.csv", index=False)
    (output_dir / "feature_spec.json").write_text(
        json.dumps(dynamic_feature_spec(), indent=2) + "\n"
    )

    active: list[str] = []
    for name in dynamic_models:
        if name not in DYNAMIC_MODEL_NAMES:
            raise ValueError(f"Unknown dynamic model: {name}")
        if name == "persist_residual_xgboost":
            try:
                make_dynamic_model(name, random_state=seed, n_jobs=n_jobs)
            except ImportError as exc:
                if log_fn:
                    log_fn(f"Skipping {name}: {exc}")
                continue
        active.append(name)

    if not active:
        raise RuntimeError("No dynamic projection models available to run.")

    regimes = sorted({f.name for f in folds})
    metrics_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    for regime in regimes:
        regime_folds = [f for f in folds if f.name == regime]
        for f in regime_folds:
            train_idx, test_idx = f.train_idx, f.test_idx
            y_train, y_test = y[train_idx], y[test_idx]
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            fold_tag = f"{regime}__{f.fold}"

            train_X, test_X, _, meta = prepare_dynamic_fold_frames(
                train_df, test_df, y_train
            )

            for model_name in active:
                model = make_dynamic_model(
                    model_name, random_state=seed, n_jobs=n_jobs
                )
                model.fit(
                    train_X,
                    y_train,
                    y_persist=meta["y_persist_train"],
                    y_lag=meta["y_lag_train"],
                    has_prior=meta["has_prior_train"],
                )
                y_pred = model.predict(
                    test_X,
                    y_persist=meta["y_persist_test"],
                    y_lag=meta["y_lag_test"],
                    has_prior=meta["has_prior_test"],
                )

                level = regression_metrics(y_test, y_pred)
                extra = projection_metrics(
                    y_test,
                    y_pred,
                    y_persist=meta["y_persist_test"],
                    y_lag=meta["y_lag_test"],
                    has_prior=meta["has_prior_test"],
                )

                metrics_rows.append(
                    {
                        "fold_tag": fold_tag,
                        "regime": regime,
                        "fold": int(f.fold),
                        "model": model_name,
                        **level,
                        **extra,
                        "phi": model.phi_,
                        "test_has_prior_frac": meta["test_has_prior_frac"],
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "fit_params": json.dumps(model.fit_params()),
                    }
                )

                pred_rows.append(
                    pd.DataFrame(
                        {
                            "fold_tag": fold_tag,
                            "regime": regime,
                            "fold": int(f.fold),
                            "model": model_name,
                            "row_id": test_idx.astype(int),
                            "site": df.loc[test_idx, "site"].to_numpy(),
                            "region": df.loc[test_idx, "region"].to_numpy(),
                            "y_obs": y_test,
                            "y_pred": y_pred,
                            "y_persist": meta["y_persist_test"],
                            "y_lag": meta["y_lag_test"],
                            "has_prior_survey": meta["has_prior_test"],
                        }
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    summary = (
        metrics_df.groupby(["regime", "model"], as_index=False)
        .agg(
            folds=("fold_tag", "count"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            delta_r2_mean=("delta_r2", "mean"),
            persist_r2_mean=("persist_r2", "mean"),
            prior_only_r2_mean=("prior_only_r2", "mean"),
            test_has_prior_frac_mean=("test_has_prior_frac", "mean"),
        )
        .sort_values(["regime", "r2_mean"], ascending=[True, False])
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    if log_fn:
        log_fn(
            f"Dynamic projection CV complete ({len(active)} models, "
            f"{len(metrics_rows)} fold fits) → {output_dir}"
        )
        for _, row in summary.iterrows():
            log_fn(
                f"  {DYNAMIC_DISPLAY_NAMES.get(row['model'], row['model'])} · "
                f"{row['regime']}: R²={row['r2_mean']:.4f} "
                f"(persist {row['persist_r2_mean']:.4f}, "
                f"ΔR² vs persist {row['delta_r2_mean']:.4f})"
            )
