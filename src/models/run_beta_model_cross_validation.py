#!/usr/bin/env python3
"""
Cross-validation for the reparameterized hierarchical beta-GLMM.

Validation regimes (fixed seeds, matching R):
- random_kfold
- site_group_kfold
- ecoregion_group_kfold ()
- forward_time_blocks
- spatial_kfold

Usage:
    python -m src.models.run_beta_model_cross_validation
    RCV_SMOKE=1 python -m src.models.run_beta_model_cross_validation

Progress:
    Uses Rich tables and a fold progress bar when ``rich`` is installed.
    Set ``RCV_PLAIN=1`` for plain logging and PyMC's per-chain progress bars.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    Console = None  # type: ignore[misc, assignment]
    Panel = None
    Table = None
    box = None
    _RICH_AVAILABLE = False

from src import config
from src.models.cv_methods import FoldSpec, build_all_folds, fold_manifest_dataframe
from src.models.cv_prediction_plots import (
    plot_cv_observed_vs_predicted,
    plot_cv_residual_diagnostics,
    save_beta_fold_diagnostics,
)
from src.models.hbb import (
    HAS_PYMC,
    HierarchicalBetaModel,
    load_model_data_for_cv,
    predict_from_posterior_cv,
    prepare_cv_fold_arrays,
)
from src.models.cv_common import cv_console, cv_log, extract_sampler_diagnostics, fmt_float
from src.models.hbb.mcmc_config import CV_MCMC_DEFAULTS
from src.models.hbb.model import resolve_pymc_ncores

_CONSOLE = cv_console()


def _print_run_header(cfg: dict[str, Any]) -> None:
    mcmc = cfg["mcmc"]
    smoke = os.getenv("RCV_SMOKE") == "1"
    lines = [
        f"Data: [cyan]{cfg['data_dir']}[/]",
        f"Output: [cyan]{cfg['output_dir']}[/]",
        f"Regimes: {', '.join(cfg['validation_regimes'])}",
        f"Folds per regime: {cfg['k_folds']}  |  min train rows: {cfg['min_train_rows']}",
        (
            f"MCMC: {mcmc['n_chains']} chains × "
            f"({mcmc['n_tune']} tune + {mcmc['n_samples']} draws)  |  "
            f"target_accept={mcmc['target_accept']}"
        ),
    ]
    if smoke:
        lines.append("[yellow]RCV_SMOKE=1 — reduced regimes and shorter sampling[/]")
    if _CONSOLE is not None:
        _CONSOLE.print(
            Panel(
                "\n".join(lines),
                title="Beta-GLMM cross-validation",
                border_style="blue",
            )
        )
    else:
        cv_log("=== Beta-GLMM cross-validation ===")
        for line in lines:
            cv_log(line.replace("[cyan]", "").replace("[/]", "").replace("[yellow]", ""))


def _print_dataset_summary(df: pd.DataFrame) -> None:
    if _CONSOLE is None:
        cv_log(
            f"Loaded {len(df)} rows | {df['site'].nunique()} sites | "
            f"{df['region'].nunique()} regions"
        )
        return
    table = Table(title="Dataset", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Observations", f"{len(df):,}")
    table.add_row("Sites", f"{df['site'].nunique():,}")
    table.add_row("Ecoregions", f"{df['region'].nunique():,}")
    if "Average_coral_cover" in df.columns:
        cover = df["Average_coral_cover"].astype(float)
        if cover.max() > 1.5:
            cover = cover / 100.0
        table.add_row("Coral cover (mean)", fmt_float(float(cover.mean()), 3))
        table.add_row("Coral cover (sd)", fmt_float(float(cover.std()), 3))
    _CONSOLE.print(table)


def _print_fold_plan(folds: list[FoldSpec], cfg: dict[str, Any]) -> tuple[int, int]:
    """Print fold manifest; return (eligible, skipped) counts."""
    eligible = 0
    skipped = 0
    for f in folds:
        n_train = len(f.train_idx)
        n_test = len(f.test_idx)
        if n_train < cfg["min_train_rows"] or n_test == 0:
            skipped += 1
        else:
            eligible += 1

    if _CONSOLE is None:
        cv_log(f"Folds: {len(folds)} total ({eligible} to fit, {skipped} skipped)")
        return eligible, skipped

    table = Table(
        title=f"Fold plan ({eligible} to fit, {skipped} skipped)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
    )
    table.add_column("Fold", style="cyan")
    table.add_column("Regime")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Status")
    for f in folds:
        fold_tag = f"{f.name}__{f.fold}"
        n_train = len(f.train_idx)
        n_test = len(f.test_idx)
        if n_train < cfg["min_train_rows"] or n_test == 0:
            status = "[dim]skip[/]"
        else:
            status = "[green]fit[/]"
        table.add_row(
            fold_tag,
            f.name,
            f"{n_train:,}",
            f"{n_test:,}",
            status,
        )
    _CONSOLE.print(table)
    return eligible, skipped


def _print_fold_result(metrics_row: pd.Series) -> None:
    msg = (
        f"  R²={fmt_float(metrics_row['r2'])}  "
        f"RMSE={fmt_float(metrics_row['rmse'])}  "
        f"MAE={fmt_float(metrics_row['mae'])}  "
        f"cov95={fmt_float(metrics_row['coverage95'])}  "
        f"R̂_max={fmt_float(metrics_row['max_rhat'])}  "
        f"ESS_min={fmt_float(metrics_row['min_neff'], 0)}  "
        f"div={fmt_float(metrics_row['n_divergences'], 0)}"
    )
    line = f"{metrics_row['fold_tag']}{msg}"
    if _CONSOLE is not None:
        _CONSOLE.print(f"[green]✓[/] {line}")
    else:
        cv_log(f"Done {line}")


def _print_regime_summary(metrics_df: pd.DataFrame) -> None:
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
        cv_log("\nSummary by regime:")
        cv_log(summary.to_string(index=False))
        return

    table = Table(title="Results by regime", box=box.ROUNDED)
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


DEFAULT_CFG: dict[str, Any] = {
    "data_dir": config.sully_og_dir,
    "output_dir": config.sully_og_dir / "output" / "cross_validation",
    "seed": 42,
    "k_folds": 5,
    "spatial_bins": 4,
    "min_train_rows": 500,
    "y_eps": 1e-6,
    "mcmc": CV_MCMC_DEFAULTS.to_dict(),
    "validation_regimes": [
        "random_kfold",
        "site_group_kfold",
        "ecoregion_group_kfold",
        "forward_time_blocks",
        # "spatial_kfold",
    ],
}


def apply_smoke_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("RCV_SMOKE") != "1":
        return cfg
    out = dict(cfg)
    out["k_folds"] = 2
    out["validation_regimes"] = ["random_kfold"]
    out["min_train_rows"] = 200
    out["output_dir"] = Path(cfg["output_dir"]) / "smoke"
    out["mcmc"] = dict(cfg["mcmc"])
    out["mcmc"]["n_tune"] = 50
    out["mcmc"]["n_samples"] = 50
    out["mcmc"]["ncores"] = 1
    return out




def fit_fold_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    fold_tag: str,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    arrays = prepare_cv_fold_arrays(train_df, test_df, y_eps=cfg["y_eps"])
    mcmc = cfg["mcmc"]

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
        ncores=resolve_pymc_ncores(
            ncores=mcmc.get("ncores"),
            n_chains=int(mcmc["n_chains"]),
        ),
        mp_ctx=mcmc.get("mp_ctx"),
        random_seed=cfg["seed"] + sum(ord(c) for c in fold_tag) % 1_000_000,
        progressbar=True,
        use_site_hierarchy=arrays.get("use_site_hierarchy", True),
        use_ecoregion_hierarchy=arrays.get("use_ecoregion_hierarchy", True),
        use_diversity=arrays.get("use_diversity", True),
    )

    pred = predict_from_posterior_cv(
        model=model,
        X_test=arrays["X_test"],
        test_df=arrays["test_df"],
        dense_info=arrays["dense_info"],
        n_train=arrays["n_train"],
        y_eps=cfg["y_eps"],
    )

    summary = model.summary
    max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary.columns else np.nan
    min_neff = (
        float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else np.nan
    )
    sampler = extract_sampler_diagnostics(model, mcmc["max_treedepth"])

    metrics = pd.DataFrame(
        [
            {
                "fold_tag": fold_tag,
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

    metrics_row = metrics.iloc[0].to_dict()
    summary.to_csv(output_dir / f"summary_{fold_tag}.csv", index=False)
    save_beta_fold_diagnostics(
        model,
        fold_dir=output_dir / "folds" / fold_tag,
        fold_tag=fold_tag,
        predictions=pred["predictions"],
        test_df=test_df,
        train_df=train_df,
        summary_df=summary,
        sampler=sampler,
        mcmc=mcmc,
        metrics=metrics_row,
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "fit_summary": summary,
    }


def _build_folds(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[list[FoldSpec], list[dict[str, Any]]]:
    return build_all_folds(
        df,
        validation_regimes=list(cfg["validation_regimes"]),
        k_folds=int(cfg["k_folds"]),
        seed=int(cfg["seed"]),
        spatial_bins=int(cfg["spatial_bins"]),
    )


def _run_fold_loop(
    df: pd.DataFrame,
    all_folds: list[FoldSpec],
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]]]:
    """Fit all eligible folds; return metrics frames, predictions, failures."""
    eligible_folds = [
        f
        for f in all_folds
        if len(f.train_idx) >= cfg["min_train_rows"] and len(f.test_idx) > 0
    ]
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_failures: list[dict[str, Any]] = []
    mcmc = cfg["mcmc"]
    total_draws = mcmc["n_chains"] * (mcmc["n_tune"] + mcmc["n_samples"])

    def _process_fold(f: FoldSpec) -> Optional[pd.Series]:
        nonlocal all_metrics, all_predictions, all_failures
        fold_tag = f"{f.name}__{f.fold}"
        train_df = df.iloc[f.train_idx].reset_index(drop=True)
        test_df = df.iloc[f.test_idx].reset_index(drop=True)
        try:
            res = fit_fold_model(
                train_df,
                test_df,
                cfg,
                fold_tag,
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - collect per-fold failures like R
            all_failures.append(
                {
                    "fold_tag": fold_tag,
                    "regime": f.name,
                    "fold": f.fold,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "error": str(exc),
                }
            )
            if _CONSOLE is not None:
                _CONSOLE.print(f"[red]✗[/] {fold_tag}: {exc}")
            else:
                cv_log(f"Fold {fold_tag} failed: {exc}")
            return None

        metrics = res["metrics"].copy()

        metrics["regime"] = f.name
        metrics["fold"] = f.fold
        all_metrics.append(metrics)
        fold_preds = res["predictions"].copy()
        fold_preds.insert(1, "regime", f.name)
        fold_preds.insert(2, "fold", int(f.fold))
        all_predictions.append(fold_preds)
        return metrics.iloc[0]

    for i, f in enumerate(eligible_folds, start=1):
        fold_tag = f"{f.name}__{f.fold}"
        n_train = len(f.train_idx)
        n_test = len(f.test_idx)
        if _CONSOLE is not None:
            _CONSOLE.rule(f"Beta-GLMM fold {i}/{len(eligible_folds)}", style="cyan")
            _CONSOLE.print(
                Panel(
                    "\n".join(
                        [
                            f"[bold]{fold_tag}[/]  train={n_train:,}  test={n_test:,}",
                            (
                                f"MCMC: {mcmc['n_chains']} chains × "
                                f"({mcmc['n_tune']} tune + {mcmc['n_samples']} draws)  "
                                f"≈ {total_draws:,} total iterations"
                            ),
                        ]
                    ),
                    title="Beta-GLMM sampling",
                    border_style="cyan",
                )
            )
            _CONSOLE.print(
                "[dim]PyMC sampling progress below "
                "(Rich progress bar disabled during MCMC).[/]\n"
            )
        else:
            cv_log(
                f"[{i}/{len(eligible_folds)}] Running {fold_tag} "
                f"(train={n_train}, test={n_test})..."
            )
        metrics_row = _process_fold(f)
        if metrics_row is not None:
            _print_fold_result(metrics_row)

    return all_metrics, all_predictions, all_failures


def run_cross_validation(cfg: Optional[dict[str, Any]] = None) -> Path:
    if not HAS_PYMC:
        raise ImportError(
            "PyMC is required for cross-validation. Install with: pip install pymc arviz"
        )

    t0 = time.perf_counter()
    cfg = apply_smoke_cfg(cfg or DEFAULT_CFG)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_run_header(cfg)

    cv_log("Loading data (data.csv + shapefile pipeline)...")
    df = load_model_data_for_cv(Path(cfg["data_dir"]))
    _print_dataset_summary(df)

    all_folds, skipped = _build_folds(df, cfg)
    fold_manifest_dataframe(all_folds).to_csv(
        output_dir / "fold_manifest.csv", index=False
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_regimes.csv", index=False)
    _print_fold_plan(all_folds, cfg)

    all_metrics, all_predictions, all_failures = _run_fold_loop(
        df, all_folds, cfg, output_dir
    )

    if not all_metrics:
        raise RuntimeError("No folds were successfully fit.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    if all_failures:
        pd.DataFrame(all_failures).to_csv(
            output_dir / "validation_failures.csv", index=False
        )
        if _CONSOLE is not None:
            _CONSOLE.print(
                f"[yellow]{len(all_failures)} fold(s) failed[/] — see validation_failures.csv"
            )
        else:
            cv_log(f"{len(all_failures)} fold(s) failed — see validation_failures.csv")

    metrics_df.to_csv(output_dir / "validation_metrics_by_fold.csv", index=False)
    pred_df.to_csv(output_dir / "validation_predictions.csv", index=False)

    metrics_regime = metrics_df.groupby("regime", as_index=False).agg(
        folds=("fold_tag", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
        coverage95_mean=("coverage95", "mean"),
        mean_log_score=("mean_log_score", "mean"),
        max_rhat_mean=("max_rhat", "mean"),
        min_neff_mean=("min_neff", "mean"),
    )
    metrics_regime.to_csv(output_dir / "validation_metrics_by_regime.csv", index=False)

    cv_log("Writing plots and summaries...")
    plot_cv_observed_vs_predicted(
        pred_df,
        output_dir=output_dir,
        model_col=None,
    )
    residual_plots: list[Path] = []
    for regime in sorted(pred_df["regime"].astype(str).unique()):
        regime_sub = pred_df.loc[pred_df["regime"] == regime]
        residual_plots.extend(
            plot_cv_residual_diagnostics(
                regime_sub,
                df,
                all_folds,
                output_dir=output_dir,
                prefix=regime,
            )
        )
    if residual_plots:
        cv_log(f"Residual diagnostics → {output_dir / 'residual_diagnostics'}")
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_df.boxplot(column="rmse", by="regime", ax=ax)
    ax.set_title("RMSE by validation regime")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_by_regime.png", dpi=300)
    plt.close(fig)

    _print_regime_summary(metrics_df)
    elapsed = time.perf_counter() - t0
    done_msg = f"Finished in {elapsed / 60:.1f} min — outputs in {output_dir}"
    if _CONSOLE is not None:
        _CONSOLE.print(Panel(done_msg, border_style="green", title="Complete"))
    else:
        cv_log(done_msg)
    return output_dir


def main() -> None:
    run_cross_validation()


if __name__ == "__main__":
    main()
