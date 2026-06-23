"""Prediction diagnostics for cross-validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.baseline_models import DISPLAY_NAMES
from src.models.dynamic_models import DYNAMIC_DISPLAY_NAMES
from src.models.baseline_plots import (
    FIG_DPI,
    MODEL_COLOUR_MAP,
    plot_metrics_comparison,
)
from src.models.residual_plots import (
    DEFAULT_RESIDUAL_PLOT_CONFIG,
    ResidualPlotConfig,
    plot_residual_scatter,
    style_observed_vs_predicted_axes,
    write_cv_residual_diagnostics,
    write_fold_residual_diagnostics,
)
from src.models.hbb.variants import (
    BETA_VARIANT_LABELS,
    VARIANTS,
    variant_plot_title,
)

BETA_MODEL_LABEL = "beta_glmm"
COMBINED_DISPLAY_NAMES: dict[str, str] = {
    **DISPLAY_NAMES,
    **DYNAMIC_DISPLAY_NAMES,
    BETA_MODEL_LABEL: "Beta-GLMM",
    **BETA_VARIANT_LABELS,
}
COMBINED_MODEL_COLOURS: dict[str, str] = {
    **MODEL_COLOUR_MAP,
    BETA_MODEL_LABEL: "tab:purple",
    **{k: "tab:purple" for k in BETA_VARIANT_LABELS},
    "paper_reproduction": "#F21A00",
    "paper_region_fixed": "#E1AF00",
    "reparam": "#3B9AB2",
    "persist_residual_xgboost": "#2A9D8F",
    "persist_ar_env": "#E76F51",
    "persist_residual_linear": "#264653",
    "persist_residual_ridge": "#457B9D",
}


def _variant_meta(name: str) -> Any | None:
    return VARIANTS.get(name)


def beta_variant_plot_title(name: str, *, short: bool = False) -> str:
    """Human-readable beta variant label for plot titles / axis labels."""
    return variant_plot_title(name, short=short)


def model_plot_label(model: str, *, short: bool = False) -> str:
    if model in DISPLAY_NAMES:
        return DISPLAY_NAMES[model]
    if model in DYNAMIC_DISPLAY_NAMES:
        return DYNAMIC_DISPLAY_NAMES[model]
    if model == BETA_MODEL_LABEL:
        return "Beta-GLMM"
    if _variant_meta(model) is not None:
        return beta_variant_plot_title(model, short=short)
    return BETA_VARIANT_LABELS.get(model, str(model))


def build_display_name_map(models: list[str], *, short: bool = False) -> dict[str, str]:
    return {m: model_plot_label(m, short=short) for m in models}


def _iter_beta_variant_dirs(beta_root: Path) -> list[Path]:
    """Variant output directories (preferred over legacy flat beta_glmm/)."""
    if not beta_root.is_dir():
        return []
    return sorted(
        child
        for child in beta_root.iterdir()
        if child.is_dir() and (child / "metrics_by_fold.csv").exists()
    )


def _iter_beta_metrics_paths(output_dir: Path) -> list[tuple[str, Path]]:
    """Return (model_label, metrics_csv) for each beta variant output."""
    beta_root = Path(output_dir) / "beta_glmm"
    variant_dirs = _iter_beta_variant_dirs(beta_root)
    if variant_dirs:
        return [(child.name, child / "metrics_by_fold.csv") for child in variant_dirs]

    direct = beta_root / "metrics_by_fold.csv"
    if direct.exists():
        return [(BETA_MODEL_LABEL, direct)]
    return []


def _iter_beta_prediction_paths(output_dir: Path) -> list[tuple[str, Path]]:
    beta_root = Path(output_dir) / "beta_glmm"
    variant_dirs = _iter_beta_variant_dirs(beta_root)
    if variant_dirs:
        return [
            (child.name, child / "predictions.csv")
            for child in variant_dirs
            if (child / "predictions.csv").exists()
        ]

    direct = beta_root / "predictions.csv"
    if direct.exists():
        return [(BETA_MODEL_LABEL, direct)]
    return []


def _style_observed_vs_predicted_panel(
    ax: plt.Axes,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    config: ResidualPlotConfig,
) -> None:
    style_observed_vs_predicted_axes(
        ax,
        y_obs,
        y_pred,
        title=title,
        config=config,
    )


def _plot_observed_vs_predicted_panels(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path | None = None,
    display_names: Mapping[str, str] | None = None,
    config: ResidualPlotConfig | None = None,
    show: bool = False,
) -> plt.Figure:
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    base_labels = {**COMBINED_DISPLAY_NAMES, **(display_names or {})}
    names = list(predictions.keys())
    label_map = {
        name: base_labels.get(name, model_plot_label(name)) for name in names
    }
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.2 * nrows), dpi=cfg.dpi)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, names):
        y_obs, y_pred = predictions[name]
        _style_observed_vs_predicted_panel(
            ax,
            np.asarray(y_obs, dtype=float),
            np.asarray(y_pred, dtype=float),
            title=label_map.get(name, str(name)),
            config=cfg,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _plot_residuals_vs_predicted_panels(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path | None = None,
    display_names: Mapping[str, str] | None = None,
    config: ResidualPlotConfig | None = None,
    show: bool = False,
) -> plt.Figure:
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    base_labels = {**COMBINED_DISPLAY_NAMES, **(display_names or {})}
    names = list(predictions.keys())
    label_map = {
        name: base_labels.get(name, model_plot_label(name)) for name in names
    }
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.0 * nrows), dpi=cfg.dpi)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, names):
        y_obs, y_pred = predictions[name]
        y_obs = np.asarray(y_obs, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_pred - y_obs
        plot_residual_scatter(
            y_pred,
            residuals,
            ax=ax,
            config=cfg,
            xlabel="Predicted coral cover",
            title=label_map.get(name, str(name)),
            fit_regression=True,
            fit_method="ols",
            use_meta_regression=False,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cv_observed_vs_predicted(
    pred_df: pd.DataFrame,
    *,
    output_dir: Path,
    model_col: Optional[str] = "model",
    single_model_label: str = "Beta-GLMM",
    plot_name_suffix: str = "",
    config: ResidualPlotConfig | None = None,
    display_names: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write observed-vs-predicted and residuals-vs-predicted plots per CV regime."""
    if pred_df.empty:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    extra_labels = dict(display_names or {})

    for regime, sub in pred_df.groupby("regime", sort=True):
        if model_col and model_col in sub.columns:
            preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for model_name in sorted(sub[model_col].astype(str).unique()):
                msub = sub.loc[sub[model_col] == model_name]
                preds[model_name] = (
                    msub["y_obs"].to_numpy(dtype=float),
                    msub["y_pred"].to_numpy(dtype=float),
                )
        else:
            label_key = single_model_label
            preds = {
                label_key: (
                    sub["y_obs"].to_numpy(dtype=float),
                    sub["y_pred"].to_numpy(dtype=float),
                )
            }

        regime_labels = {
            **build_display_name_map(list(preds.keys())),
            **extra_labels,
        }
        suffix = f"_{plot_name_suffix}" if plot_name_suffix else ""
        obs_path = output_dir / f"plot_{regime}{suffix}_observed_vs_predicted.png"
        res_path = output_dir / f"plot_{regime}{suffix}_residuals_vs_predicted.png"
        _plot_observed_vs_predicted_panels(
            preds, output_path=obs_path, config=cfg, display_names=regime_labels
        )
        _plot_residuals_vs_predicted_panels(
            preds, output_path=res_path, config=cfg, display_names=regime_labels
        )
        written.extend([obs_path, res_path])

    return written


def plot_cv_fold_observed_vs_predicted(
    y_obs,
    y_pred,
    *,
    output_path: Path,
    label: str = "Beta-GLMM",
    config: ResidualPlotConfig | None = None,
    display_names: Mapping[str, str] | None = None,
) -> Path:
    """Single-fold observed-vs-predicted scatter."""
    output_path = Path(output_path)
    names = {label: model_plot_label(label), **(display_names or {})}
    _plot_observed_vs_predicted_panels(
        {label: (y_obs, y_pred)},
        output_path=output_path,
        config=config,
        display_names=names,
    )
    return output_path


def save_beta_fold_diagnostics(
    model: Any,
    *,
    fold_dir: Path,
    fold_tag: str,
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sampler: dict[str, float],
    mcmc: dict[str, Any],
    metrics: dict[str, Any],
    residual_plot_config: ResidualPlotConfig | None = None,
    variant: str | None = None,
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
        "col_names": getattr(model, "col_names", None),
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

    fold_label = variant if variant else "beta_glmm"
    plot_cv_fold_observed_vs_predicted(
        predictions["y_obs"].to_numpy(dtype=float),
        predictions["y_pred"].to_numpy(dtype=float),
        output_path=fold_dir / "observed_vs_predicted.png",
        label=fold_label,
    )

    predictions.to_csv(fold_dir / "predictions.csv")

    write_fold_residual_diagnostics(
        predictions,
        test_df=test_df,
        train_df=train_df,
        output_dir=fold_dir / "residual_diagnostics",
        config=residual_plot_config,
    )
    return fold_dir


def plot_cv_residual_diagnostics(
    pred_df: pd.DataFrame,
    full_df: pd.DataFrame,
    folds: list[Any],
    *,
    output_dir: Path,
    prefix: str = "",
    fold_col: str = "fold",
    config: ResidualPlotConfig | None = None,
) -> list[Path]:
    """Write fold-coloured residual diagnostics for one CV prediction table."""
    out_dir = Path(output_dir) / "residual_diagnostics"
    return write_cv_residual_diagnostics(
        pred_df,
        full_df,
        folds,
        out_dir,
        prefix=prefix,
        fold_col=fold_col,
        config=config,
    )


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

    dynamic_metrics_path = output_dir / "dynamic_projection" / "metrics_by_fold.csv"
    if "dynamic_projection" in families and dynamic_metrics_path.exists():
        parts.append(pd.read_csv(dynamic_metrics_path))

    if "beta_glmm" in families:
        for model_label, metrics_path in _iter_beta_metrics_paths(output_dir):
            beta = pd.read_csv(metrics_path)
            beta = beta.assign(model=model_label)
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
            "variant",
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

    dynamic_pred_path = output_dir / "dynamic_projection" / "predictions.csv"
    if "dynamic_projection" in families and dynamic_pred_path.exists():
        parts.append(pd.read_csv(dynamic_pred_path))

    if "beta_glmm" in families:
        for model_label, pred_path in _iter_beta_prediction_paths(output_dir):
            beta = pd.read_csv(pred_path)
            beta = beta.assign(model=model_label)
            parts.append(beta)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def write_combined_cv_plots(
    output_dir: Path,
    *,
    families: list[str],
) -> list[Path]:
    """Write cross-family metric and prediction comparison plots."""
    output_dir = Path(output_dir)
    written: list[Path] = []
    combined_metrics = collect_combined_cv_metrics(output_dir, families=families)
    if combined_metrics.empty or combined_metrics["model"].nunique() < 2:
        return written

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
        model_ids = sub["model"].astype(str).tolist()
        bar_labels = build_display_name_map(model_ids, short=True)
        out = output_dir / f"plot_{regime}_all_models_r2_rmse.png"
        plot_metrics_comparison(
            sub,
            output_path=out,
            show=False,
            title=f"All models — {regime_label}",
            display_names=bar_labels,
            model_colours=COMBINED_MODEL_COLOURS,
            tight_ylim=True,
            horizontal_grid=True,
            x_rotation=45,
            figsize=(max(12.0, 1.5 * len(sub)), 5.0),
        )
        written.append(out)

    combined_preds = collect_combined_cv_predictions(output_dir, families=families)
    if not combined_preds.empty:
        combined_preds.to_csv(output_dir / "combined_predictions.csv", index=False)
        pred_models = sorted(combined_preds["model"].astype(str).unique())
        pred_labels = build_display_name_map(pred_models)
        written.extend(
            plot_cv_observed_vs_predicted(
                combined_preds,
                output_dir=output_dir,
                model_col="model",
                single_model_label=BETA_MODEL_LABEL,
                plot_name_suffix="all_models",
                display_names=pred_labels,
            )
        )

    return written


def refresh_cv_output_plots(
    output_dir: Path,
    *,
    families: list[str] | None = None,
) -> list[Path]:
    """Regenerate comparison and per-family prediction plots from saved CSVs."""
    output_dir = Path(output_dir)
    families = families or []
    if not families:
        if (output_dir / "baselines").exists():
            families.append("baselines")
        if (output_dir / "beta_glmm").exists():
            families.append("beta_glmm")

    written: list[Path] = []
    for model_label, pred_path in _iter_beta_prediction_paths(output_dir):
        pred_df = pd.read_csv(pred_path)
        if pred_df.empty:
            continue
        written.extend(
            plot_cv_observed_vs_predicted(
                pred_df,
                output_dir=pred_path.parent,
                model_col=None,
                single_model_label=model_label,
            )
        )

    baselines_path = output_dir / "baselines" / "predictions.csv"
    if baselines_path.exists():
        pred_df = pd.read_csv(baselines_path)
        if not pred_df.empty:
            written.extend(
                plot_cv_observed_vs_predicted(
                    pred_df,
                    output_dir=output_dir / "baselines",
                    model_col="model",
                )
            )

    written.extend(write_combined_cv_plots(output_dir, families=families))
    return written


def write_family_cv_residual_plots(
    output_dir: Path,
    full_df: pd.DataFrame,
    folds: list[Any],
    *,
    family: str,
    config: ResidualPlotConfig | None = None,
) -> list[Path]:
    """Write regime-level residual diagnostics for one model family."""
    output_dir = Path(output_dir)
    written: list[Path] = []

    if family == "beta_glmm":
        pred_sources = _iter_beta_prediction_paths(output_dir)
    else:
        pred_path = output_dir / family / "predictions.csv"
        pred_sources = [(family, pred_path)] if pred_path.exists() else []

    for model_label, pred_path in pred_sources:
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        if pred_df.empty:
            continue

        family_dir = pred_path.parent

        if family in {"baselines", "dynamic_projection"} and "model" in pred_df.columns:
            for baseline_name in sorted(pred_df["model"].astype(str).unique()):
                sub = pred_df.loc[pred_df["model"] == baseline_name]
                for regime in sorted(sub["regime"].astype(str).unique()):
                    regime_sub = sub.loc[sub["regime"] == regime]
                    written.extend(
                        plot_cv_residual_diagnostics(
                            regime_sub,
                            full_df,
                            folds,
                            output_dir=family_dir,
                            prefix=f"{baseline_name}_{regime}",
                            config=config,
                        )
                    )
        else:
            for regime in sorted(pred_df["regime"].astype(str).unique()):
                regime_sub = pred_df.loc[pred_df["regime"] == regime]
                prefix = (
                    f"{model_label}_{regime}"
                    if model_label != BETA_MODEL_LABEL
                    else regime
                )
                written.extend(
                    plot_cv_residual_diagnostics(
                        regime_sub,
                        full_df,
                        folds,
                        output_dir=family_dir,
                        prefix=prefix,
                        config=config,
                    )
                )
    return written
