"""Plots comparing baseline coral-cover models."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.baseline_models import DISPLAY_NAMES, BaselineName
from src.models.residual_plots import (
    DEFAULT_RESIDUAL_PLOT_CONFIG,
    style_observed_vs_predicted_axes,
)

FIG_DPI = 300

MODEL_COLOUR_MAP = {
    "linear": "grey",
    "random_forest": "#3B9AB2",
    "xgboost": "#78B7C5",
    "neural_network": "#EBCC2A",
    "survey_mean": "#888888",
    "beta_glmm": "#F21A00",
}
# E1AF00


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)  # type: ignore[arg-type]


def _tight_metric_ylim(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    err: np.ndarray | None = None,
    pad_frac: float = 0.12,
    min_span: float | None = None,
) -> None:
    """Zoom y-axis to the data range so bar heights are easier to compare."""
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        return

    y = y[mask]
    if err is not None:
        err = np.asarray(err, dtype=float)[mask]
        err = np.where(np.isfinite(err), err, 0.0)
        lo = float(np.min(y - err))
        hi = float(np.max(y + err))
    else:
        lo = float(np.min(y))
        hi = float(np.max(y))

    span = hi - lo
    if min_span is not None and span < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - min_span / 2, mid + min_span / 2
        span = hi - lo
    pad = max(span * pad_frac, 1e-4)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_ylim(lo - pad, hi + pad)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    show: bool = False,
    title: str | None = None,
    display_names: Mapping[str, str] | None = None,
    model_colours: Mapping[str, str] | None = None,
    tight_ylim: bool = False,
    horizontal_grid: bool = False,
    x_rotation: float = 25,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Bar charts of R² and RMSE per model.

    Expected columns: ``model``, ``r2``, ``rmse``. If ``*_std`` columns are present,
    labels will mention cross-validation.
    """
    label_map = {**DISPLAY_NAMES, **(display_names or {})}
    colour_map = {**MODEL_COLOUR_MAP, **(model_colours or {})}
    plot_df = metrics_df.copy()
    plot_df["label"] = plot_df["model"].map(
        lambda m: label_map.get(m, _display(m) if m in DISPLAY_NAMES else str(m))
    )

    n_models = len(plot_df)
    if figsize is None:
        figsize = (max(10.0, 1.35 * n_models), 4.8)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    order = plot_df["label"].tolist()

    is_cv = any(c.endswith("_std") for c in plot_df.columns)

    sns.barplot(
        data=plot_df,
        x="label",
        y="r2",
        order=order,
        ax=axes[0],
        hue=plot_df["model"].map(lambda x: colour_map.get(x, "tab:gray")),
        legend=False,
    )
    r2_err = None
    if is_cv and "r2_std" in plot_df.columns:
        r2_std = plot_df["r2_std"].to_numpy(dtype=float)
        if np.isfinite(r2_std).any():
            r2_err = np.where(np.isfinite(r2_std), 1.96 * r2_std, 0.0)
            x_pos = np.arange(len(plot_df))
            axes[0].errorbar(
                x_pos, plot_df["r2"], yerr=r2_err, fmt="none", ecolor="k", capsize=4
            )
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title("CV mean $R^2$" if is_cv else "Test-set $R^2$")
    if tight_ylim:
        _tight_metric_ylim(axes[0], plot_df["r2"].to_numpy(), err=r2_err, min_span=0.05)

    sns.barplot(
        data=plot_df,
        x="label",
        y="rmse",
        order=order,
        ax=axes[1],
        hue=plot_df["model"].map(lambda x: colour_map.get(x, "tab:gray")),
        legend=False,
    )
    rmse_err = None
    if is_cv and "rmse_std" in plot_df.columns:
        rmse_std = plot_df["rmse_std"].to_numpy(dtype=float)
        if np.isfinite(rmse_std).any():
            rmse_err = np.where(np.isfinite(rmse_std), 1.96 * rmse_std, 0.0)
            x_pos = np.arange(len(plot_df))
            axes[1].errorbar(
                x_pos, plot_df["rmse"], yerr=rmse_err, fmt="none", ecolor="k", capsize=4
            )
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("CV mean RMSE" if is_cv else "Test-set RMSE")
    if tight_ylim:
        _tight_metric_ylim(axes[1], plot_df["rmse"].to_numpy(), err=rmse_err)

    grid_kw = {"linestyle": "--", "alpha": 0.35}
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=x_rotation, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        if horizontal_grid:
            ax.grid(True, axis="y", **grid_kw)
            ax.grid(False, axis="x")
        else:
            ax.grid(True, **grid_kw)

    fig.suptitle(
        title or "Baseline models — coral cover prediction", fontsize=13, y=1.02
    )
    bottom_pad = 0.32 if x_rotation >= 40 else 0.18
    fig.tight_layout(rect=(0, bottom_pad, 1, 0.98))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_observed_vs_predicted(
    predictions: Mapping[BaselineName, tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path | None = None,
    show: bool = False,
    display_names: Mapping[str, str] | None = None,
) -> plt.Figure:
    """
    Scatter panels: observed vs predicted coral cover per model.

    ``predictions``: model name -> (y_obs, y_pred) on the test split.
    """
    cfg = DEFAULT_RESIDUAL_PLOT_CONFIG
    label_map = {**DISPLAY_NAMES, **(display_names or {})}
    names = list(predictions.keys())
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.2 * nrows), dpi=cfg.dpi)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, names):
        y_obs, y_pred = predictions[name]
        style_observed_vs_predicted_axes(
            ax,
            y_obs,
            y_pred,
            title=label_map.get(name, _display(name) if name in DISPLAY_NAMES else str(name)),
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

