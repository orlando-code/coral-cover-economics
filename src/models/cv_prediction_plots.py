"""Prediction diagnostics for cross-validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.models.baseline_plots import (
    MODEL_COLOUR_MAP,
    plot_metrics_comparison,
    plot_observed_vs_predicted,
)

BETA_MODEL_LABEL = "beta_glmm"
COMBINED_DISPLAY_NAMES: dict[str, str] = {
    BETA_MODEL_LABEL: "Beta-GLMM",
}
COMBINED_MODEL_COLOURS: dict[str, str] = {
    **MODEL_COLOUR_MAP,
    BETA_MODEL_LABEL: "tab:purple",
}


def plot_cv_observed_vs_predicted(
    pred_df: pd.DataFrame,
    *,
    output_dir: Path,
    model_col: Optional[str] = "model",
    single_model_label: str = "Beta-GLMM",
    plot_name_suffix: str = "",
) -> list[Path]:
    """Write observed-vs-predicted scatter plots per CV regime."""
    if pred_df.empty:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for regime, sub in pred_df.groupby("regime", sort=True):
        if model_col and model_col in sub.columns:
            preds: dict[str, tuple] = {}
            for model_name in sorted(sub[model_col].astype(str).unique()):
                msub = sub.loc[sub[model_col] == model_name]
                preds[model_name] = (
                    msub["y_obs"].to_numpy(dtype=float),
                    msub["y_pred"].to_numpy(dtype=float),
                )
        else:
            preds = {
                single_model_label: (
                    sub["y_obs"].to_numpy(dtype=float),
                    sub["y_pred"].to_numpy(dtype=float),
                )
            }

        suffix = f"_{plot_name_suffix}" if plot_name_suffix else ""
        out = output_dir / f"plot_{regime}{suffix}_observed_vs_predicted.png"
        plot_observed_vs_predicted(
            preds,
            output_path=out,
            show=False,
            display_names=COMBINED_DISPLAY_NAMES if model_col else None,
        )
        written.append(out)

    return written


def plot_cv_fold_observed_vs_predicted(
    y_obs,
    y_pred,
    *,
    output_path: Path,
    label: str = "Beta-GLMM",
) -> Path:
    """Single-fold observed-vs-predicted scatter."""
    output_path = Path(output_path)
    plot_observed_vs_predicted(
        {label: (y_obs, y_pred)},
        output_path=output_path,
        show=False,
    )
    return output_path


def save_beta_fold_diagnostics(
    model: Any,
    *,
    fold_dir: Path,
    fold_tag: str,
    predictions: pd.DataFrame,
    summary_df: pd.DataFrame,
    sampler: dict[str, float],
    mcmc: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Persist MCMC traces, diagnostic plots, and fit statistics for one fold."""
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(fold_dir / "fit_summary.csv", index=False)

    fit_statistics = {
        "fold_tag": fold_tag,
        "mcmc": mcmc,
        "sampler": sampler,
        "metrics": metrics,
    }
    (fold_dir / "fit_statistics.json").write_text(
        json.dumps(fit_statistics, indent=2, default=str) + "\n"
    )

    sampler_row = {**sampler, **metrics}
    pd.DataFrame([sampler_row]).to_csv(
        fold_dir / "sampler_diagnostics.csv", index=False
    )

    if model.trace is not None:
        model.trace.to_netcdf(fold_dir / "trace.nc")
        model.plot_coefficient_traces_and_posteriors(
            fold_dir / "coefficient_diagnostics"
        )
        model.save_diagnostics(fold_dir / "trace_diagnostics")

    plot_cv_fold_observed_vs_predicted(
        predictions["y_obs"].to_numpy(dtype=float),
        predictions["y_pred"].to_numpy(dtype=float),
        output_path=fold_dir / "observed_vs_predicted.png",
    )
    return fold_dir


def collect_combined_cv_metrics(
    output_dir: Path,
    *,
    families: list[str],
) -> pd.DataFrame:
    """Fold-level metrics for all model families in one table."""
    parts: list[pd.DataFrame] = []
    output_dir = Path(output_dir)

    baselines_path = output_dir / "baselines" / "metrics_by_fold.csv"
    if "baselines" in families and baselines_path.exists():
        parts.append(pd.read_csv(baselines_path))

    beta_path = output_dir / "beta_glmm" / "metrics_by_fold.csv"
    if "beta_glmm" in families and beta_path.exists():
        beta = pd.read_csv(beta_path)
        beta = beta.assign(model=BETA_MODEL_LABEL)
        parts.append(beta)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    keep = [
        c
        for c in [
            "fold_tag",
            "regime",
            "fold",
            "model",
            "n_train",
            "n_test",
            "r2",
            "rmse",
            "mae",
            "coverage95",
            "max_rhat",
        ]
        if c in combined.columns
    ]
    return combined[keep].copy()


def summarize_combined_cv_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sd of fold metrics per regime and model."""
    if metrics_df.empty:
        return metrics_df

    agg_spec: dict[str, tuple[str, str]] = {
        "folds": ("fold_tag", "count"),
        "r2_mean": ("r2", "mean"),
        "r2_sd": ("r2", "std"),
        "rmse_mean": ("rmse", "mean"),
        "rmse_sd": ("rmse", "std"),
        "mae_mean": ("mae", "mean"),
        "mae_sd": ("mae", "std"),
    }
    if "coverage95" in metrics_df.columns:
        agg_spec["coverage95_mean"] = ("coverage95", "mean")
    if "max_rhat" in metrics_df.columns:
        agg_spec["max_rhat_mean"] = ("max_rhat", "mean")

    summary = metrics_df.groupby(["regime", "model"], as_index=False).agg(**agg_spec)
    return summary.sort_values(["regime", "r2_mean"], ascending=[True, False])


def collect_combined_cv_predictions(
    output_dir: Path,
    *,
    families: list[str],
) -> pd.DataFrame:
    """Test-set predictions for all model families in one table."""
    parts: list[pd.DataFrame] = []
    output_dir = Path(output_dir)

    baselines_path = output_dir / "baselines" / "predictions.csv"
    if "baselines" in families and baselines_path.exists():
        parts.append(pd.read_csv(baselines_path))

    beta_path = output_dir / "beta_glmm" / "predictions.csv"
    if "beta_glmm" in families and beta_path.exists():
        beta = pd.read_csv(beta_path)
        beta = beta.assign(model=BETA_MODEL_LABEL)
        parts.append(beta)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def write_combined_cv_plots(
    output_dir: Path,
    *,
    families: list[str],
) -> list[Path]:
    """Write cross-family metric and observed-vs-predicted comparison plots."""
    output_dir = Path(output_dir)
    if len(families) < 2:
        return []

    written: list[Path] = []
    combined_metrics = collect_combined_cv_metrics(output_dir, families=families)
    if not combined_metrics.empty:
        combined_metrics.to_csv(output_dir / "combined_metrics_by_fold.csv", index=False)
        summary = summarize_combined_cv_metrics(combined_metrics)
        summary.to_csv(output_dir / "combined_metrics_summary.csv", index=False)

        for regime in sorted(summary["regime"].unique()):
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
            regime_label = regime.replace("_", " ")
            out = output_dir / f"plot_{regime}_all_models_r2_rmse.png"
            plot_metrics_comparison(
                sub,
                output_path=out,
                show=False,
                title=f"All models — {regime_label}",
                display_names=COMBINED_DISPLAY_NAMES,
                model_colours=COMBINED_MODEL_COLOURS,
            )
            written.append(out)

    combined_preds = collect_combined_cv_predictions(output_dir, families=families)
    if not combined_preds.empty:
        combined_preds.to_csv(output_dir / "combined_predictions.csv", index=False)
        written.extend(
            plot_cv_observed_vs_predicted(
                combined_preds,
                output_dir=output_dir,
                model_col="model",
                single_model_label=COMBINED_DISPLAY_NAMES[BETA_MODEL_LABEL],
                plot_name_suffix="all_models",
            )
        )

    return written
