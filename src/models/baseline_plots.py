"""Plots comparing baseline coral-cover models."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.baseline_models import DISPLAY_NAMES, BaselineName

FIG_DPI = 300

MODEL_COLOUR_MAP = {
    "linear": "grey",
    "random_forest": "#3B9AB2",
    "xgboost": "#78B7C5",
    "neural_network": "#EBCC2A",
    "beta_glmm": "#F21A00",
}
# E1AF00


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)  # type: ignore[arg-type]


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    show: bool = False,
    title: str | None = None,
    display_names: Mapping[str, str] | None = None,
    model_colours: Mapping[str, str] | None = None,
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    order = plot_df["label"].tolist()

    is_cv = any(c.endswith("_std") for c in plot_df.columns)

    bar1 = sns.barplot(
        data=plot_df,
        x="label",
        y="r2",
        order=order,
        ax=axes[0],
        hue=plot_df["model"].map(lambda x: colour_map.get(x, "tab:gray")),
        legend=False,
    )
    # add 95% CI error bars if std columns present (assumes r2_std is the std of the estimate)
    if is_cv and "r2_std" in plot_df.columns:
        ci = 1.96 * plot_df["r2_std"]
        x_pos = np.arange(len(plot_df))
        axes[0].errorbar(
            x_pos, plot_df["r2"], yerr=ci, fmt="none", ecolor="k", capsize=4
        )
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title("CV mean $R^2$" if is_cv else "Test-set $R^2$")

    bar2 = sns.barplot(
        data=plot_df,
        x="label",
        y="rmse",
        order=order,
        ax=axes[1],
        hue=plot_df["model"].map(lambda x: colour_map.get(x, "tab:gray")),
        legend=False,
    )
    if is_cv and "rmse_std" in plot_df.columns:
        ci = 1.96 * plot_df["rmse_std"]
        x_pos = np.arange(len(plot_df))
        axes[1].errorbar(
            x_pos, plot_df["rmse"], yerr=ci, fmt="none", ecolor="k", capsize=4
        )
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("CV mean RMSE" if is_cv else "Test-set RMSE")

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        title or "Baseline models — coral cover prediction", fontsize=13, y=1.02
    )
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
    label_map = {**DISPLAY_NAMES, **(display_names or {})}
    names = list(predictions.keys())
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_flat, names):
        y_obs, y_pred = predictions[name]
        ax.scatter(y_obs, y_pred, alpha=0.25, s=12, edgecolors="none")
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed coral cover")
        ax.set_ylabel("Predicted coral cover")
        ax.set_title(
            label_map.get(name, _display(name) if name in DISPLAY_NAMES else str(name))
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
