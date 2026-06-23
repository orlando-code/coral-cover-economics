#!/usr/bin/env python3
"""Unified cross-validation runner for multiple model families.

This script is the single entrypoint for running one or more models through one or
more CV regimes (fold-building is centralized in :mod:`src.models.cv_methods`).

Examples
--------
# Default: all baselines + hierarchical beta-GLMM across all CV regimes
python -m src.models.run_cross_validation

# Baselines only, single regime
python -m src.models.run_cross_validation --models baselines --regimes site_group_kfold

# Single baseline model
python -m src.models.run_cross_validation --models random_forest

# Beta-GLMM only with custom MCMC settings
python -m src.models.run_cross_validation --models beta_glmm \\
    --beta-n-chains 4 --beta-n-tune 500 --beta-n-samples 1000

# Forward forecasting on repeat-visit sites (~80/20 temporal holdout)
python -m src.models.run_cross_validation --regimes forward_repeat_sites

# Compare corrected vs paper beta indexing on the same CV split
python -m src.models.run_cross_validation --models beta_glmm \\
    --regimes forward_repeat_sites --beta-variants reparam,paper_reproduction

# Dynamic persistence-adjusted projection models (longitudinal CV)
python -m src.models.run_cross_validation --models dynamic_projection \\
    --regimes forward_repeat_sites

# All families on forward-repeat sites
python -m src.models.run_cross_validation --models baselines,beta_glmm,dynamic_projection \\
    --regimes forward_repeat_sites --beta-variants reparam

# Quick smoke test (short MCMC, serial PyMC cores)
RCV_SMOKE=1 python -m src.models.run_cross_validation --models beta_glmm \\
    --regimes forward_repeat_sites

# Plot CV splits only
python -m src.models.cv_plots

Progress:
    Rich panels, tables, and progress bars (set ``RCV_PLAIN=1`` for plain output).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    Console = None  # type: ignore[misc, assignment]
    Panel = None
    Progress = None
    Table = None
    box = None
    _RICH_AVAILABLE = False

from src import config

# Beta model
from src.dataloading.build_sully_model_ready_data import to_hbb_frame

# Baselines
from src.models.baseline_features import baseline_feature_spec
from src.models.baseline_models import (
    BASELINE_MODEL_NAMES,
    DISPLAY_NAMES,
    is_persistence_baseline,
    make_baseline_estimator,
    predict_coral_cover,
)
from src.models.baseline_persistence import predict_survey_mean_baseline
from src.models.baseline_plots import plot_metrics_comparison
from src.models.coral_data import coral_cover_target, load_model_ready_data
from src.models.cv_common import (
    NullProgress,
    cv_console,
    cv_log,
    extract_sampler_diagnostics,
    fmt_float,
    format_exc,
    now_tag,
    parse_csv_list,
    write_json,
)
from src.models.cv_methods import (
    ALL_CV_REGIMES,
    FoldSpec,
    build_all_folds,
    fold_manifest_dataframe,
)
from src.models.cv_plots import plot_cv_regimes
from src.models.cv_prediction_plots import (
    plot_cv_observed_vs_predicted,
    save_beta_fold_diagnostics,
    write_combined_cv_plots,
    write_family_cv_residual_plots,
)
from src.models.dynamic_cv import run_dynamic_cv
from src.models.dynamic_features import dynamic_feature_spec
from src.models.dynamic_models import DYNAMIC_DISPLAY_NAMES, DYNAMIC_MODEL_NAMES
from src.models.hbb import (
    HAS_PYMC,
    HierarchicalBetaModel,
    predict_from_posterior_cv,
    prepare_cv_fold_arrays,
)
from src.models.hbb.cv_decomposition import (
    compute_fold_decomposition,
    write_variant_decomposition_summary,
)
from src.models.hbb.mcmc_config import (
    CV_MCMC_DEFAULTS,
    MCMCConfig,
    add_cv_mcmc_arguments,
    apply_cv_smoke_overrides,
    mcmc_config_from_cv_args,
    merge_mcmc_dict,
)
from src.models.hbb.model import resolve_pymc_ncores
from src.models.hbb.variants import (
    VARIANTS,
    Variant,
    apply_variant_options,
    beta_variant_output_dir,
    parse_optional_bool,
    parse_variant_names,
)
from src.models.metrics import regression_metrics

_CONSOLE = cv_console()


def _model_label(name: str) -> str:
    if name in DYNAMIC_DISPLAY_NAMES:
        return DYNAMIC_DISPLAY_NAMES[name]
    return DISPLAY_NAMES.get(name, name)  # type: ignore[arg-type]


def _resolve_models(
    models: list[str],
    baseline_models_filter: Optional[list[str]] = None,
    dynamic_models_filter: Optional[list[str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (model families, baseline model names, dynamic model names)."""
    families: list[str] = []
    baselines: list[str] = []
    dynamic: list[str] = []
    for name in models:
        if name == "beta_glmm":
            families.append("beta_glmm")
        elif name == "dynamic_projection":
            dynamic.extend(DYNAMIC_MODEL_NAMES)
        elif name in DYNAMIC_MODEL_NAMES:
            dynamic.append(name)
        elif name == "baselines":
            baselines.extend(BASELINE_MODEL_NAMES)
        elif name in BASELINE_MODEL_NAMES:
            baselines.append(name)
        else:
            opts = ", ".join(
                [
                    *BASELINE_MODEL_NAMES,
                    "baselines",
                    "beta_glmm",
                    "dynamic_projection",
                    *DYNAMIC_MODEL_NAMES,
                ]
            )
            raise ValueError(f"Unknown model '{name}'. Expected one of: {opts}")

    if baselines:
        if baseline_models_filter:
            allowed = set(baseline_models_filter)
            baselines = [m for m in baselines if m in allowed]
        else:
            baselines = list(dict.fromkeys(baselines))
        if not baselines:
            raise ValueError("No baseline models selected after filtering.")
        if "baselines" not in families:
            families.append("baselines")

    if dynamic:
        if dynamic_models_filter:
            allowed = set(dynamic_models_filter)
            dynamic = [m for m in dynamic if m in allowed]
        else:
            dynamic = list(dict.fromkeys(dynamic))
        if not dynamic:
            raise ValueError("No dynamic projection models selected after filtering.")
        if "dynamic_projection" not in families:
            families.append("dynamic_projection")

    if not families:
        raise ValueError("No models to run.")
    return families, baselines, dynamic


def _safe_inner_splits(groups: np.ndarray, requested: int) -> int:
    n_unique = int(pd.Series(groups).nunique())
    return int(max(2, min(requested, n_unique)))


def _print_run_header(cfg: dict[str, Any]) -> None:
    baselines = cfg["baselines"]
    lines = [
        f"Output: [cyan]{cfg['output_dir']}[/]",
        f"Regimes: {', '.join(cfg['regimes'])}",
        f"Folds per regime: {cfg['k_folds']}  |  seed: {cfg['seed']}",
        f"Model families: {', '.join(cfg['model_families'])}",
    ]
    if "baselines" in cfg["model_families"]:
        lines.append(
            "Baselines: "
            + ", ".join(_model_label(m) for m in baselines["baseline_models"])
        )
        lines.append(
            f"  inner CV: {baselines['inner_splits']} splits  |  "
            f"n_iter={baselines['n_iter']}  |  n_jobs={baselines['n_jobs']}"
        )
    if "beta_glmm" in cfg["model_families"]:
        beta = cfg["beta_glmm"]
        mcmc = beta.get("mcmc") or {}
        variants = beta.get("variants") or ["reparam"]
        lines.append(f"Beta-GLMM min train rows: {beta['min_train_rows']}")
        lines.append(f"  variants: {', '.join(variants)}")
        if mcmc:
            lines.append(
                f"  MCMC: {mcmc.get('n_chains', '?')} chains × "
                f"({mcmc.get('n_tune', '?')} tune + {mcmc.get('n_samples', '?')} draws)  |  "
                f"target_accept={mcmc.get('target_accept', '?')}  |  "
                f"max_treedepth={mcmc.get('max_treedepth', '?')}  |  "
                f"ncores={resolve_pymc_ncores(ncores=mcmc.get('ncores'), n_chains=int(mcmc.get('n_chains', 2)))}  |  "
                f"mp_ctx={mcmc.get('mp_ctx', 'spawn')}"
            )
    if "dynamic_projection" in cfg["model_families"]:
        dyn = cfg.get("dynamic_projection") or {}
        dyn_models = dyn.get("models") or list(DYNAMIC_MODEL_NAMES)
        lines.append(
            "Dynamic projection: " + ", ".join(_model_label(m) for m in dyn_models)
        )
    if os.getenv("RCV_SMOKE") == "1":
        lines.append(
            "[yellow]RCV_SMOKE=1 — shortened MCMC, baseline n_iter capped at 2, "
            "smoke output dir[/]"
        )
    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel("\n".join(lines), title="Cross-validation", border_style="blue")
        )
    else:
        cv_log("=== Cross-validation ===")
        for line in lines:
            cv_log(line.replace("[cyan]", "").replace("[/]", ""))


def _print_dataset_summary(df: pd.DataFrame) -> None:
    if _CONSOLE is None:
        cv_log(
            f"Loaded {len(df):,} rows | {df['site'].nunique():,} sites | "
            f"{df['region'].nunique():,} regions"
        )
        return
    table = Table(title="Dataset", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Observations", f"{len(df):,}")
    table.add_row("Sites", f"{df['site'].nunique():,}")
    table.add_row("Ecoregions", f"{df['region'].nunique():,}")
    _CONSOLE.print(table)


def _print_fold_plan(folds: list[FoldSpec], *, title: str = "Fold plan") -> None:
    if _CONSOLE is None:
        cv_log(
            f"{title}: {len(folds)} fold{'s' if len(folds) > 1 else ''} across {len({f.name for f in folds})} regime{'s' if len(folds) > 1 else ''}"
        )
        return
    table = Table(
        title=f"{title} ({len(folds)} fold{'s' if len(folds) > 1 else ''})",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Fold", style="cyan")
    table.add_column("Regime")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")
    for f in sorted(folds, key=lambda x: (x.name, x.fold)):
        table.add_row(
            f"{f.name}__{f.fold}",
            f.name,
            f"{len(f.train_idx):,}",
            f"{len(f.test_idx):,}",
        )
    _CONSOLE.print(table)


def _print_skipped_regimes(skipped: list[dict[str, Any]]) -> None:
    if not skipped:
        return
    if _CONSOLE is None:
        for row in skipped:
            cv_log(f"Skipped regime {row['regime']}: {row['reason']}")
        return
    table = Table(title="Skipped regimes", box=box.ROUNDED, border_style="yellow")
    table.add_column("Regime", style="yellow")
    table.add_column("Reason")
    for row in skipped:
        table.add_row(str(row["regime"]), str(row["reason"]))
    _CONSOLE.print(table)


def _print_baseline_fold_result(
    *,
    regime: str,
    model_name: str,
    fold: int,
    metrics: dict[str, float],
    n_train: int,
    n_test: int,
    tuning: dict[str, Any] | None = None,
) -> None:
    msg = (
        f"{_model_label(model_name)} · {regime} fold {fold} "
        f"(train={n_train:,}, test={n_test:,})  "
        f"R²={fmt_float(metrics['r2'])}  RMSE={fmt_float(metrics['rmse'])}  "
        f"MAE={fmt_float(metrics['mae'])}"
    )
    if tuning:
        trials = tuning.get("n_trials_completed", "?")
        requested = tuning.get("n_trials_requested", "?")
        best_cv = tuning.get("best_cv_r2")
        elapsed = tuning.get("elapsed_sec", "?")
        method = tuning.get("tuning_method", "?")
        inner = tuning.get("inner_cv_splits", "?")
        cv_part = (
            f"best inner-CV R²={fmt_float(best_cv)}"
            if best_cv is not None and np.isfinite(best_cv)
            else "no inner tuning"
        )
        msg += (
            f"  |  tuning={method} {trials}/{requested} trials"
            f" ({inner}-fold inner CV, {elapsed}s)  {cv_part}"
        )
    if _CONSOLE is not None:
        _CONSOLE.print(f"  [green]✓[/] {msg}")
    else:
        cv_log(f"  Done {msg}")


def _print_baseline_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    if _CONSOLE is None:
        cv_log("\nBaseline summary:\n" + summary.to_string(index=False))
        return
    table = Table(title="Baseline results by regime", box=box.ROUNDED)
    table.add_column("Regime", style="cyan")
    table.add_column("Model")
    table.add_column("Folds", justify="right")
    table.add_column("R² (mean ± sd)", justify="right")
    table.add_column("RMSE (mean ± sd)", justify="right")
    for _, row in summary.iterrows():
        r2_cell = (
            f"{fmt_float(row['r2_mean'])} ± {fmt_float(row['r2_sd'])}"
            if pd.notna(row["r2_sd"])
            else fmt_float(row["r2_mean"])
        )
        rmse_cell = (
            f"{fmt_float(row['rmse_mean'])} ± {fmt_float(row['rmse_sd'])}"
            if pd.notna(row["rmse_sd"])
            else fmt_float(row["rmse_mean"])
        )
        table.add_row(
            str(row["regime"]),
            _model_label(str(row["model"])),
            str(int(row["folds"])),
            r2_cell,
            rmse_cell,
        )
    _CONSOLE.print(table)


def _print_beta_sampling_header(
    fold_tag: str,
    *,
    n_train: int,
    n_test: int,
    mcmc: dict[str, Any],
) -> None:
    """Announce a fold before PyMC sampling (native PyMC progress follows)."""
    total_draws = int(mcmc["n_chains"]) * (int(mcmc["n_tune"]) + int(mcmc["n_samples"]))
    lines = [
        f"[bold]{fold_tag}[/]  train={n_train:,}  test={n_test:,}",
        (
            f"MCMC: {mcmc['n_chains']} chains × "
            f"({mcmc['n_tune']} tune + {mcmc['n_samples']} draws)  "
            f"≈ {total_draws:,} total iterations"
        ),
    ]
    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel("\n".join(lines), title="Beta-GLMM sampling", border_style="cyan")
        )
        _CONSOLE.print(
            "[dim]PyMC sampling progress below "
            "(Rich progress bar disabled during MCMC).[/]\n"
        )
    else:
        cv_log("=== Beta-GLMM sampling ===")
        for line in lines:
            cv_log(line.replace("[bold]", "").replace("[/]", ""))


def _print_beta_fold_result(metrics_row: pd.Series) -> None:
    msg = (
        f"{metrics_row['fold_tag']}  "
        f"R²={fmt_float(metrics_row['r2'])}  "
        f"RMSE={fmt_float(metrics_row['rmse'])}  "
        f"MAE={fmt_float(metrics_row['mae'])}  "
        f"cov95={fmt_float(metrics_row['coverage95'])}  "
        f"R̂_max={fmt_float(metrics_row['max_rhat'])}  "
        f"ESS_min={fmt_float(metrics_row['min_neff'], 0)}"
    )
    if _CONSOLE is not None:
        _CONSOLE.print(f"  [green]✓[/] {msg}")
    else:
        cv_log(f"  Done {msg}")


def _print_beta_summary(metrics_df: pd.DataFrame) -> None:
    summary = metrics_df.groupby("regime", as_index=False).agg(
        folds=("fold_tag", "count"),
        r2_mean=("r2", "mean"),
        r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        coverage95_mean=("coverage95", "mean"),
        max_rhat_mean=("max_rhat", "mean"),
    )
    if _CONSOLE is None:
        cv_log("\nBeta-GLMM summary:\n" + summary.to_string(index=False))
        return
    table = Table(title="Beta-GLMM results by regime", box=box.ROUNDED)
    table.add_column("Regime", style="cyan")
    table.add_column("Folds", justify="right")
    table.add_column("R² (mean ± sd)", justify="right")
    table.add_column("RMSE (mean ± sd)", justify="right")
    table.add_column("Cov95", justify="right")
    table.add_column("R̂ (mean)", justify="right")
    for _, row in summary.iterrows():
        r2_cell = (
            f"{fmt_float(row['r2_mean'])} ± {fmt_float(row['r2_sd'])}"
            if pd.notna(row["r2_sd"])
            else fmt_float(row["r2_mean"])
        )
        rmse_cell = (
            f"{fmt_float(row['rmse_mean'])} ± {fmt_float(row['rmse_sd'])}"
            if pd.notna(row["rmse_sd"])
            else fmt_float(row["rmse_mean"])
        )
        table.add_row(
            str(row["regime"]),
            str(int(row["folds"])),
            r2_cell,
            rmse_cell,
            fmt_float(row["coverage95_mean"]),
            fmt_float(row["max_rhat_mean"]),
        )
    _CONSOLE.print(table)


def run_baselines_cv(
    *,
    df: pd.DataFrame,
    folds: list[FoldSpec],
    output_dir: Path,
    baseline_models: list[str],
    seed: int,
    inner_splits: int,
    n_iter: int,
    n_jobs: int,
    tuning_method: str = "bayes",
) -> None:
    from sklearn.model_selection import GroupKFold

    from src.models.baseline_features import (
        baseline_feature_spec,
        make_baseline_pipeline,
        prepare_baseline_fold_frames,
    )
    from src.models.baseline_tuning import tune_baseline_estimator

    output_dir.mkdir(parents=True, exist_ok=True)

    y = coral_cover_target(df)
    site_groups = df["site"].astype(int).to_numpy()

    # Save fold manifest for this model family.
    fold_manifest_dataframe(folds).to_csv(output_dir / "fold_manifest.csv", index=False)
    spec = baseline_feature_spec()
    (output_dir / "feature_spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    metrics_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    regimes = sorted({f.name for f in folds})
    est_n_jobs = 1 if n_jobs != 1 else -1
    active_models: list[str] = []
    for model_name in baseline_models:
        if model_name not in BASELINE_MODEL_NAMES:
            raise ValueError(f"Unknown baseline model: {model_name}")
        if is_persistence_baseline(model_name):
            active_models.append(model_name)
            continue
        try:
            _ = make_baseline_estimator(
                model_name, n_jobs=est_n_jobs, random_state=seed
            )
            active_models.append(model_name)
        except ImportError as exc:
            if _CONSOLE is not None:
                _CONSOLE.print(f"[yellow]Skipping {_model_label(model_name)}:[/] {exc}")
            else:
                cv_log(f"Skipping baseline model {model_name}: {exc}")

    tasks = [
        (regime, model_name, f)
        for regime in regimes
        for model_name in active_models
        for f in folds
        if f.name == regime
    ]
    if not tasks:
        raise RuntimeError("No baseline CV tasks to run.")

    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel(
                f"Fitting {len(tasks)} folds across {len(regimes)} regime{'' if len(regimes) == 1 else 's'} "
                f"and {len(active_models)} model{'s' if len(active_models) > 1 else ''}",
                title="Baselines",
                border_style="green",
            )
        )

    progress_ctx = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_CONSOLE,
            transient=False,
        )
        if _CONSOLE is not None and Progress is not None
        else None
    )
    progress = progress_ctx or NullProgress()

    with progress:
        task_id = progress.add_task("Baselines", total=len(tasks))
        for regime, model_name, f in tasks:
            progress.update(
                task_id,
                description=(f"{_model_label(model_name)} · {regime} · fold {f.fold}"),
            )
            train_idx, test_idx = f.train_idx, f.test_idx
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = site_groups[train_idx]

            if is_persistence_baseline(model_name):
                y_pred = predict_survey_mean_baseline(
                    df.iloc[train_idx],
                    df.iloc[test_idx],
                    y_train,
                )
                best_params: dict[str, Any] = {}
                tuning: dict[str, Any] = {
                    "tuning_method": "none",
                    "n_trials_requested": 0,
                    "n_trials_completed": 0,
                    "best_cv_r2": np.nan,
                    "elapsed_sec": 0.0,
                }
            else:
                train_prep, test_prep, _ = prepare_baseline_fold_frames(
                    df.iloc[train_idx], df.iloc[test_idx]
                )

                est = make_baseline_estimator(
                    model_name, n_jobs=est_n_jobs, random_state=seed
                )
                pipe = make_baseline_pipeline(est)

                inner_n = _safe_inner_splits(groups_train, inner_splits)
                inner_cv = GroupKFold(n_splits=inner_n)
                best, best_params, tuning = tune_baseline_estimator(
                    pipe,
                    model_name,  # type: ignore[arg-type]
                    X_train=train_prep,
                    y_train=y_train,
                    groups_train=groups_train,
                    inner_cv=inner_cv,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    seed=seed,
                    method=tuning_method,  # type: ignore[arg-type]
                )

                y_pred = predict_coral_cover(best, test_prep)
            m = regression_metrics(y_test, y_pred)

            fold_tag = f"{regime}__{f.fold}"
            _print_baseline_fold_result(
                regime=regime,
                model_name=model_name,
                fold=int(f.fold),
                metrics=m,
                n_train=len(train_idx),
                n_test=len(test_idx),
                tuning=tuning,
            )
            metrics_rows.append(
                {
                    "fold_tag": fold_tag,
                    "regime": regime,
                    "fold": int(f.fold),
                    "model": model_name,
                    **m,
                    "best_params": str(best_params),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **{f"tuning_{k}": v for k, v in tuning.items()},
                }
            )
            tuning_rows.append(
                {
                    "fold_tag": fold_tag,
                    "regime": regime,
                    "fold": int(f.fold),
                    "model": model_name,
                    "best_params": str(best_params),
                    **tuning,
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
                    }
                )
            )
            progress.advance(task_id)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(output_dir / "tuning_by_fold.csv", index=False)

    summary = (
        metrics_df.groupby(["regime", "model"], as_index=False)
        .agg(
            folds=("fold_tag", "count"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
        )
        .sort_values(["regime", "r2_mean"], ascending=[True, False])
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    _print_baseline_summary(summary)

    # One compact plot per regime: baseline mean R² / RMSE.
    for regime in regimes:
        sub = summary.loc[
            summary["regime"] == regime,
            ["model", "r2_mean", "rmse_mean", "r2_sd", "rmse_sd"],
        ].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                "r2_mean": "r2",
                "rmse_mean": "rmse",
                "r2_sd": "r2_std",
                "rmse_sd": "rmse_std",
            }
        )
        plot_metrics_comparison(
            sub,
            output_path=output_dir / f"plot_{regime}_r2_rmse.png",
            show=False,
        )

    if pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_df.to_csv(output_dir / "predictions.csv", index=False)
        plot_cv_observed_vs_predicted(
            pred_df,
            output_dir=output_dir,
            model_col="model",
        )


def run_beta_glmm_cv(
    *,
    df: pd.DataFrame,
    folds: list[FoldSpec],
    output_dir: Path,
    seed: int,
    min_train_rows: int,
    y_eps: float,
    mcmc: dict[str, Any],
    variant: str | Variant = "reparam",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_manifest_dataframe(folds).to_csv(output_dir / "fold_manifest.csv", index=False)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    decomposition_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    eligible = [
        f for f in folds if len(f.train_idx) >= min_train_rows and len(f.test_idx) > 0
    ]
    skipped = len(folds) - len(eligible)
    variant_name = variant.name if isinstance(variant, Variant) else variant
    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel(
                f"Fitting {len(eligible)} fold{'' if len(eligible) == 1 else 's'}"
                + (f" ({skipped} skipped)" if skipped else "")
                + f"  ·  variant={variant_name}",
                title="Beta-GLMM",
                border_style="magenta",
            )
        )
    else:
        cv_log(
            f"Beta-GLMM ({variant_name}): {len(eligible)} folds to fit"
            + (f", {skipped} skipped" if skipped else "")
        )

    for fold_idx, f in enumerate(eligible, start=1):
        fold_tag = f"{f.name}__{f.fold}"
        train_df = df.iloc[f.train_idx].reset_index(drop=True)
        test_df = df.iloc[f.test_idx].reset_index(drop=True)

        if _CONSOLE is not None:
            _CONSOLE.rule(
                f"Beta-GLMM fold {fold_idx}/{len(eligible)}",
                style="cyan",
            )
        _print_beta_sampling_header(
            fold_tag,
            n_train=len(train_df),
            n_test=len(test_df),
            mcmc=mcmc,
        )

        try:
            arrays = prepare_cv_fold_arrays(
                train_df, test_df, y_eps=y_eps, variant=variant
            )
            # save train and test dfs
            train_df.to_csv(output_dir / "train_df.csv")
            test_df.to_csv(output_dir / "test_df.csv")
            print(f"Written train/test to files: {output_dir / 'train_df.csv'}")

            model = HierarchicalBetaModel()
            model.fit(
                X=arrays["X_train"],
                y=arrays["y_train"],
                site_idx=arrays["site_idx"],
                region_idx=arrays["region_idx"],
                site_to_region=arrays["site_to_region"],
                reef_to_site_map=arrays["reef_to_site_map"],
                diversity=arrays["diversity"],
                col_names=arrays["col_names"],
                n_samples=mcmc["n_samples"],
                n_tune=mcmc["n_tune"],
                n_chains=mcmc["n_chains"],
                target_accept=mcmc["target_accept"],
                max_treedepth=mcmc["max_treedepth"],
                spec=arrays["spec"],
                ncores=resolve_pymc_ncores(
                    ncores=mcmc.get("ncores"),
                    n_chains=int(mcmc["n_chains"]),
                ),
                mp_ctx=mcmc.get("mp_ctx"),
                random_seed=seed + sum(ord(c) for c in fold_tag) % 1_000_000,
                progressbar=True,
                use_site_hierarchy=arrays["use_site_hierarchy"],
                use_ecoregion_hierarchy=arrays["use_ecoregion_hierarchy"],
                use_diversity=arrays["use_diversity"],
            )

            pred = predict_from_posterior_cv(
                model=model,
                X_test=arrays["X_test"],
                test_df=arrays["test_df"],
                dense_info=arrays["dense_info"],
                n_train=arrays["n_train"],
                y_eps=y_eps,
            )

            summary_df = model.summary
            max_rhat = (
                float(summary_df["r_hat"].max())
                if "r_hat" in summary_df.columns
                else np.nan
            )
            min_neff = (
                float(summary_df["ess_bulk"].min())
                if "ess_bulk" in summary_df.columns
                else np.nan
            )
            sampler = extract_sampler_diagnostics(model, mcmc["max_treedepth"])

            metrics = pd.DataFrame(
                [
                    {
                        "fold_tag": fold_tag,
                        "regime": f.name,
                        "fold": int(f.fold),
                        "variant": variant_name,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "r2": pred["metrics"]["r2"],
                        "rmse": pred["metrics"]["rmse"],
                        "mae": pred["metrics"]["mae"],
                        "coverage95": pred["metrics"]["coverage95"],
                        "mean_log_score": pred["metrics"]["mean_log_score"],
                        "max_rhat": max_rhat,
                        "min_neff": min_neff,
                        "n_divergences": sampler["n_divergences"],
                        "pct_max_treedepth": sampler["pct_max_treedepth"],
                    }
                ]
            )

            predictions = pred["predictions"].copy()
            predictions.insert(0, "fold_tag", fold_tag)
            predictions.insert(1, "regime", f.name)
            predictions.insert(2, "fold", int(f.fold))

            all_metrics.append(metrics)
            all_predictions.append(predictions)
            summary_df.to_csv(output_dir / f"summary_{fold_tag}.csv")

            metrics_row = metrics.iloc[0].to_dict()
            fold_dir = output_dir / "folds" / fold_tag
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(fold_dir / "train_df.csv", index=False)
            test_df.to_csv(fold_dir / "test_df.csv", index=False)
            if _CONSOLE is not None:
                with _CONSOLE.status("[cyan]Writing fold diagnostics…[/]"):
                    save_beta_fold_diagnostics(
                        model,
                        fold_dir=fold_dir,
                        fold_tag=fold_tag,
                        predictions=pred["predictions"],
                        test_df=test_df,
                        train_df=train_df,
                        summary_df=summary_df,
                        sampler=sampler,
                        mcmc=mcmc,
                        metrics=metrics_row,
                        variant=variant_name,
                    )
            else:
                cv_log("Writing fold diagnostics…")
                save_beta_fold_diagnostics(
                    model,
                    fold_dir=fold_dir,
                    fold_tag=fold_tag,
                    predictions=pred["predictions"],
                    test_df=test_df,
                    train_df=train_df,
                    summary_df=summary_df,
                    sampler=sampler,
                    mcmc=mcmc,
                    metrics=metrics_row,
                    variant=variant_name,
                )
            summary = compute_fold_decomposition(
                predictions=pred["predictions"],
                test_df=test_df,
                train_df=train_df,
                fold_dir=fold_dir,
                variant=variant_name,
                n_train=arrays["n_train"],
                dense_info=arrays["dense_info"],
                X_test=arrays["X_test"],
                use_site_hierarchy=arrays["use_site_hierarchy"],
                use_ecoregion_hierarchy=arrays["use_ecoregion_hierarchy"],
                use_diversity=arrays["use_diversity"],
            )
            summary["fold_tag"] = fold_tag
            decomposition_summaries.append(summary)
            _print_beta_fold_result(metrics.iloc[0])

        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "fold_tag": fold_tag,
                    "regime": f.name,
                    "fold": int(f.fold),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "error": format_exc(exc),
                }
            )
            err_msg = format_exc(exc)
            if _CONSOLE is not None:
                _CONSOLE.print(f"  [red]✗[/] {fold_tag}: {err_msg}")
            else:
                cv_log(f"  Fold failed: {err_msg}")

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    if not all_metrics:
        raise RuntimeError("No folds were successfully fit.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    write_variant_decomposition_summary(decomposition_summaries, output_dir)

    summary = metrics_df.groupby("regime", as_index=False).agg(
        folds=("fold_tag", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
        coverage95_mean=("coverage95", "mean"),
        mean_log_score=("mean_log_score", "mean"),
        max_rhat_mean=("max_rhat", "mean"),
        min_neff_mean=("min_neff", "mean"),
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    plot_cv_observed_vs_predicted(
        pred_df,
        output_dir=output_dir,
        model_col=None,
        single_model_label=variant_name,
    )
    _print_beta_summary(metrics_df)


def run_cross_validation(
    *,
    models: list[str] | str,
    regimes: list[str] | str,
    output_dir: Path,
    seed: int = 42,
    k_folds: int = 5,
    spatial_bins: int = 4,
    baseline_models: Optional[list[str]] = None,
    baseline_inner_splits: int = 5,
    baseline_n_iter: int = 50,
    baseline_n_jobs: int = -1,
    baseline_tuning: str = "bayes",
    beta_min_train_rows: int = 500,
    beta_y_eps: float = 1e-6,
    beta_mcmc: MCMCConfig | dict[str, Any] | None = None,
    beta_variants: Optional[list[str]] = None,
    beta_use_site_hierarchy: bool | None = None,
    beta_use_ecoregion_hierarchy: bool | None = None,
    dynamic_models: Optional[list[str]] = None,
    dynamic_n_jobs: int = -1,
    in_sample_min_site_measurements: int = 1,
) -> Path:
    """Run cross-validation for a given set of models and regimes.

    Args:
        models (list[str]): List of model names to run.
        regimes (list[str]): List of CV regimes to run.
        output_dir (Path): Path to the output directory.
        seed (int): Random seed for CV.
        k_folds (int): Number of folds for CV.
        spatial_bins (int): Number of spatial bins for CV.
        baseline_models (Optional[list[str]], optional): List of baseline model names to run. Defaults to None.
        baseline_inner_splits (int, optional): Inner grouped-CV folds for tuning. Defaults to 5.
        baseline_n_iter (int, optional): Tuning trials per outer fold. Defaults to 50.
        baseline_n_jobs (int, optional): Number of jobs for baseline models. Defaults to -1.
        baseline_tuning (str, optional): ``bayes`` (Gaussian-process search) or ``random``. Defaults to bayes.
        beta_min_train_rows (int, optional): Minimum number of training rows for beta-GLMM. Defaults to 500.
        beta_y_eps (float, optional): Epsilon for beta-GLMM. Defaults to 1e-6.
        beta_mcmc (Optional[dict[str, Any]], optional): MCMC settings for beta-GLMM. Defaults to None.

    Returns:
        Path: Path to the output directory.
    """
    if type(models) == str:
        models = [models]
    if type(regimes) == str:
        regimes = [regimes]

    output_dir = Path(output_dir)
    base_mcmc = (
        beta_mcmc
        if isinstance(beta_mcmc, MCMCConfig)
        else merge_mcmc_dict(CV_MCMC_DEFAULTS, beta_mcmc)
    )
    (
        output_dir,
        regimes,
        models,
        beta_mcmc_resolved,
        beta_min_train_rows,
        baseline_n_iter,
    ) = apply_cv_smoke_overrides(
        output_dir=output_dir,
        regimes=regimes,
        models=models,
        mcmc=base_mcmc,
        beta_min_train_rows=beta_min_train_rows,
        baseline_n_iter=baseline_n_iter,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    families, resolved_baselines, resolved_dynamic = _resolve_models(
        models, baseline_models, dynamic_models
    )
    mcmc = beta_mcmc_resolved.to_dict()
    beta_variant_list = apply_variant_options(
        [
            VARIANTS[name]
            for name in parse_variant_names(",".join(beta_variants or ["reparam"]))
        ],
        exclude_vars=(),
        latitude_transform=None,
        use_site_hierarchy=beta_use_site_hierarchy,
        use_ecoregion_hierarchy=beta_use_ecoregion_hierarchy,
    )

    cfg = {
        "models": models,
        "model_families": families,
        "regimes": regimes,
        "seed": seed,
        "k_folds": k_folds,
        "spatial_bins": spatial_bins,
        "baselines": {
            "baseline_models": resolved_baselines,
            "inner_splits": baseline_inner_splits,
            "n_iter": baseline_n_iter,
            "n_jobs": baseline_n_jobs,
            "tuning": baseline_tuning,
            "feature_spec": baseline_feature_spec(),
        },
        "beta_glmm": {
            "min_train_rows": beta_min_train_rows,
            "y_eps": beta_y_eps,
            "mcmc": mcmc,
            "variants": [v.name for v in beta_variant_list],
        },
        "dynamic_projection": {
            "models": resolved_dynamic,
            "n_jobs": dynamic_n_jobs,
            "feature_spec": dynamic_feature_spec(),
        },
        "in_sample_min_site_measurements": in_sample_min_site_measurements,
        "timestamp": now_tag(),
    }

    cfg["output_dir"] = str(output_dir)
    write_json(output_dir / "cv_config.json", cfg)
    _print_run_header(cfg)

    df = load_model_ready_data()
    _print_dataset_summary(df)

    plot_folds, plot_skipped = build_all_folds(
        df,
        validation_regimes=regimes,
        k_folds=k_folds,
        seed=seed,
        spatial_bins=spatial_bins,
        in_sample_min_site_measurements=in_sample_min_site_measurements,
    )
    _print_fold_plan(plot_folds)
    _print_skipped_regimes(plot_skipped)

    plot_dir = output_dir / "cv_regime_plots"
    if _CONSOLE is not None:
        with _CONSOLE.status("[cyan]Writing CV regime plots…[/]"):
            plot_cv_regimes(df, plot_folds, plot_dir)
    else:
        cv_log("Writing CV regime plots…")
        plot_cv_regimes(df, plot_folds, plot_dir)
    if plot_skipped:
        pd.DataFrame(plot_skipped).to_csv(plot_dir / "skipped_regimes.csv", index=False)
    cv_log(f"CV regime plots → {plot_dir}")

    for model in families:
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        if model == "baselines":
            folds, skipped = plot_folds, plot_skipped
            if skipped:
                pd.DataFrame(skipped).to_csv(
                    model_dir / "skipped_regimes.csv", index=False
                )

            run_baselines_cv(
                df=df,
                folds=folds,
                output_dir=model_dir,
                baseline_models=resolved_baselines,
                seed=seed,
                inner_splits=baseline_inner_splits,
                n_iter=baseline_n_iter,
                n_jobs=baseline_n_jobs,
                tuning_method=baseline_tuning,
            )
            residual_plots = write_family_cv_residual_plots(
                output_dir,
                df,
                folds,
                family="baselines",
            )
            if residual_plots:
                cv_log(
                    f"Baseline residual diagnostics → {model_dir / 'residual_diagnostics'}"
                )

        elif model == "dynamic_projection":
            folds, skipped = plot_folds, plot_skipped
            if skipped:
                pd.DataFrame(skipped).to_csv(
                    model_dir / "skipped_regimes.csv", index=False
                )
            run_dynamic_cv(
                df=df,
                folds=folds,
                output_dir=model_dir,
                dynamic_models=resolved_dynamic,
                seed=seed,
                n_jobs=dynamic_n_jobs,
                log_fn=cv_log,
            )
            residual_plots = write_family_cv_residual_plots(
                output_dir,
                df,
                folds,
                family="dynamic_projection",
            )
            if residual_plots:
                cv_log(
                    f"Dynamic projection residual diagnostics → "
                    f"{model_dir / 'residual_diagnostics'}"
                )

        elif model == "beta_glmm":
            if not HAS_PYMC:
                raise ImportError(
                    "PyMC is required for beta_glmm cross-validation. Install with: pip install pymc arviz"
                )

            beta_df = to_hbb_frame(df)
            folds, skipped = plot_folds, plot_skipped
            n_variants = len(beta_variant_list)
            for variant in beta_variant_list:
                variant_dir = beta_variant_output_dir(output_dir, variant, n_variants)
                variant_dir.mkdir(parents=True, exist_ok=True)
                if skipped:
                    pd.DataFrame(skipped).to_csv(
                        variant_dir / "skipped_regimes.csv", index=False
                    )

                run_beta_glmm_cv(
                    df=beta_df,
                    folds=folds,
                    output_dir=variant_dir,
                    seed=seed,
                    min_train_rows=beta_min_train_rows,
                    y_eps=beta_y_eps,
                    mcmc=mcmc,
                    variant=variant,
                )

            residual_plots = write_family_cv_residual_plots(
                output_dir,
                beta_df,
                folds,
                family="beta_glmm",
            )
            if residual_plots:
                cv_log(f"Beta-GLMM residual diagnostics → {output_dir / 'beta_glmm'}")

        else:
            raise ValueError(f"Unknown model family: {model}")

    combined_plots = write_combined_cv_plots(output_dir, families=families)
    if combined_plots:
        cv_log(f"Combined model comparison → {output_dir}")
    elif "beta_glmm" in families and len(beta_variant_list) > 1:
        cv_log(
            "Combined comparison skipped (need baselines or multiple model families "
            "with ≥2 models in metrics)."
        )

    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel(
                f"Results saved to [cyan]{output_dir}[/]",
                title="Cross-validation complete",
                border_style="green",
            )
        )
    else:
        cv_log(f"Cross-validation complete → {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified cross-validation runner")
    parser.add_argument(
        "--models",
        type=str,
        default="baselines,beta_glmm",
        help=(
            "Comma-separated models: individual baselines "
            f"({','.join(BASELINE_MODEL_NAMES)}), baselines (all), beta_glmm, "
            f"dynamic_projection ({','.join(DYNAMIC_MODEL_NAMES)}) "
            "(default: baselines,beta_glmm)"
        ),
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=",".join(ALL_CV_REGIMES),
        help="Comma-separated CV regimes (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sully_og/output/cross_validation)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument(
        "--in-sample-min-site-measurements",
        type=int,
        default=1,
        help=(
            "For in_sample_multi_visit: keep sites with strictly more than this "
            "many rows (default: 1 → at least 2 visits per site)"
        ),
    )

    # Baselines
    parser.add_argument(
        "--baseline-models",
        type=str,
        default=None,
        help="Filter baselines when --models includes 'baselines' (default: all)",
    )
    parser.add_argument(
        "--baseline-inner-splits",
        type=int,
        default=5,
        help="Inner grouped-CV folds for baseline tuning (default: 5)",
    )
    parser.add_argument(
        "--baseline-n-iter",
        type=int,
        default=50,
        help="Hyperparameter search trials per outer fold (default: 50)",
    )
    parser.add_argument("--baseline-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--baseline-tuning",
        choices=("bayes", "random"),
        default="bayes",
        help="Inner-CV search: Bayesian optimization (skopt) or random search",
    )

    # Beta GLMM
    parser.add_argument(
        "--beta-min-train-rows",
        type=int,
        default=500,
        help="Skip folds with fewer training rows (default: 500)",
    )
    parser.add_argument(
        "--beta-y-eps",
        type=float,
        default=1e-6,
        help="Epsilon for beta-scale y clipping (default: 1e-6)",
    )
    add_cv_mcmc_arguments(parser)
    parser.add_argument(
        "--beta-variants",
        type=str,
        default="reparam",
        help=(
            "Comma-separated hierarchical beta variants "
            f"({', '.join(sorted(VARIANTS))}). "
            "Use e.g. reparam,reparam_site_only,reparam_ecoregion_only,reparam_flat "
            "to compare hierarchy structures."
        ),
    )
    parser.add_argument(
        "--beta-site-hierarchy",
        choices=["true", "false"],
        default=None,
        help="Override site random effects for all selected beta variants.",
    )
    parser.add_argument(
        "--beta-ecoregion-hierarchy",
        choices=["true", "false"],
        default=None,
        help="Override ecoregion random effects for all selected beta variants.",
    )
    parser.add_argument(
        "--dynamic-models",
        type=str,
        default=None,
        help=(
            "Filter dynamic projection models when --models includes "
            f"'dynamic_projection' (default: all: {','.join(DYNAMIC_MODEL_NAMES)})"
        ),
    )
    parser.add_argument("--dynamic-n-jobs", type=int, default=-1)

    args = parser.parse_args()

    models = parse_csv_list(args.models)
    regimes = parse_csv_list(args.regimes)

    out = args.output_dir
    if out is None:
        out = config.sully_og_dir / "output" / "cross_validation"

    baseline_models = (
        parse_csv_list(args.baseline_models) if args.baseline_models else None
    )
    dynamic_models = (
        parse_csv_list(args.dynamic_models) if args.dynamic_models else None
    )

    run_cross_validation(
        models=models,
        regimes=regimes,
        output_dir=Path(out),
        seed=args.seed,
        k_folds=args.k_folds,
        spatial_bins=args.spatial_bins,
        baseline_models=baseline_models,
        baseline_inner_splits=args.baseline_inner_splits,
        baseline_n_iter=args.baseline_n_iter,
        baseline_n_jobs=args.baseline_n_jobs,
        baseline_tuning=args.baseline_tuning,
        beta_min_train_rows=args.beta_min_train_rows,
        beta_y_eps=args.beta_y_eps,
        beta_mcmc=mcmc_config_from_cv_args(args),
        beta_variants=parse_variant_names(args.beta_variants),
        beta_use_site_hierarchy=parse_optional_bool(args.beta_site_hierarchy),
        beta_use_ecoregion_hierarchy=parse_optional_bool(args.beta_ecoregion_hierarchy),
        dynamic_models=dynamic_models,
        dynamic_n_jobs=args.dynamic_n_jobs,
        in_sample_min_site_measurements=args.in_sample_min_site_measurements,
    )


if __name__ == "__main__":
    main()
