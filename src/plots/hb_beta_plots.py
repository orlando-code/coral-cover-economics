"""Plots for the hierarchical beta coral-cover model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plots.plot_config import (
    COVARIATE_LABELS,
    COVARIATE_LABELS_DICT,
    HBB_FIG_DPI,
    HYPERPARAM_LABELS,
    HYPERPARAM_TRACE_VARS,
    SPOT_CLASSES,
    SPOT_STYLE,
    TRACE_DIAGNOSTIC_VARS,
)

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


def _az():
    if not HAS_ARVIZ:
        raise ImportError("ArviZ required.")
    return az


def _label(name: str) -> str:
    from src.models.hbb.variants import display_coefficient_label

    return display_coefficient_label(name)


def _save(fig: plt.Figure, path: Optional[Path], show: bool) -> None:
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=HBB_FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    elif path:
        plt.close(fig)


def _beta_dim(trace) -> str:
    dims = [d for d in trace.posterior["beta"].dims if d not in ("chain", "draw")]
    if not dims:
        raise ValueError("No beta dimension in trace.")
    return dims[0]


def _style_trace_row(ax_row: np.ndarray, *, headers: bool) -> None:
    ax_row[0].set_title("")
    ax_row[1].set_title("")
    if headers:
        ax_row[0].set_title("Posterior", fontsize=11, pad=8)
        ax_row[1].set_title("Chain", fontsize=11, pad=8)


def _row_suptitles(fig: plt.Figure, axes: np.ndarray, labels: list[str]) -> None:
    for i, text in enumerate(labels):
        b0, b1 = axes[i, 0].get_position(), axes[i, 1].get_position()
        fig.text(
            (b0.x0 + b1.x1) / 2,
            max(b0.y1, b1.y1) + 0.012,
            text,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )


def _plot_trace_rows(
    trace,
    rows: list[tuple[str, str, Optional[dict]]],
    *,
    fig_w: float = 10,
    row_h: float = 2.8,
    adjust: tuple[float, float, float, float, float, float] = (
        0.14, 0.97, 0.97, 0.05, 0.55, 0.28
    ),
    path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """rows: (var_name, row_title, coords or None)."""
    az = _az()
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(fig_w, row_h * n), squeeze=False)
    for i, (var, title, coords) in enumerate(rows):
        kw: dict[str, Any] = dict(
            var_names=[var], axes=axes[i : i + 1], compact=False
        )
        if coords:
            kw["coords"] = coords
        az.plot_trace(trace, **kw)
        _style_trace_row(axes[i], headers=(i == 0))
    fig.subplots_adjust(
        left=adjust[0],
        right=adjust[1],
        top=adjust[2],
        bottom=adjust[3],
        hspace=adjust[4],
        wspace=adjust[5],
    )
    _row_suptitles(fig, axes, [r[1] for r in rows])
    _save(fig, path, show)
    return fig


def _quantile_row(name: str, samples: np.ndarray) -> dict[str, Any]:
    q = np.quantile(samples, [0.025, 0.25, 0.75, 0.975])
    return {
        "index": name,
        "mean": float(samples.mean()),
        "lower_2.5": float(q[0]),
        "lower_25": float(q[1]),
        "upper_75": float(q[2]),
        "upper_97.5": float(q[3]),
    }


def posterior_coefficient_summary(
    trace, col_names: Optional[list[str]] = None
) -> pd.DataFrame:
    _az()
    from src.models.hbb.variants import coefficient_labels

    names = list(col_names or [])
    if names and names[0] == "Intercept":
        row_labels = ["Intercept", *coefficient_labels(names[1:])]
    elif names:
        row_labels = coefficient_labels(names)
    else:
        row_labels = []

    beta = trace.posterior["beta"].stack(sample=("chain", "draw")).values
    rows = [
        _quantile_row(
            row_labels[i] if i < len(row_labels) else f"beta[{i}]",
            beta[i],
        )
        for i in range(beta.shape[0])
    ]
    if "beta_diversity" in trace.posterior:
        s = trace.posterior["beta_diversity"].stack(sample=("chain", "draw")).values.ravel()
        rows.append(_quantile_row("Diversity", s))
    return pd.DataFrame(rows).set_index("index")


def _significance_colors(
    df: pd.DataFrame, lo: str, hi: str, *, strict_zero: bool
) -> list[str]:
    out = []
    for _, r in df.iterrows():
        if strict_zero:
            if r["mean"] > 0 and r[lo] >= 0:
                out.append("blue")
            elif r["mean"] < 0 and r[hi] <= 0:
                out.append("red")
            else:
                out.append("gray")
        else:
            if r[lo] > 0:
                out.append("blue")
            elif r[hi] < 0:
                out.append("red")
            else:
                out.append("gray")
    return out


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    plt.tight_layout()
    _save(fig, output_path, False)
    return fig


def plot_coefficient_forest_df(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 7),
    show: bool = False,
    label_col: str | None = None,
) -> plt.Figure:
    """Standardised beta-coefficient forest plot from a summary table."""
    work = df.copy()
    if label_col and label_col in work.columns:
        work = work.set_index(label_col)
    elif "variable" in work.columns:
        work = work.set_index("variable")
    elif work.index.name is None and "index" not in work.columns:
        work.index.name = "variable"

    required = {"mean", "lower_2.5", "upper_97.5", "lower_25", "upper_75"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Coefficient summary missing columns: {sorted(missing)}")

    s = work.sort_values("mean")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=figsize)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.hlines(i, r["lower_2.5"], r["upper_97.5"], color="black", lw=0.8, zorder=1)
        ax.hlines(i, r["lower_25"], r["upper_75"], color="black", lw=2.5, zorder=2)
    ax.scatter(
        s["mean"],
        y,
        c=_significance_colors(s, "lower_2.5", "upper_97.5", strict_zero=True),
        s=80,
        zorder=5,
        edgecolors="black",
    )
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(str(i)) for i in s.index])
    ax.set_xlabel(r"Estimated $\gamma$ coefficients")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", ls="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, output_path, show)
    return fig


def plot_posterior_coefficient_forest(
    trace,
    col_names: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    figsize: tuple[float, float] = (9, 7),
    show: bool = False,
    title: str | None = None,
) -> plt.Figure:
    s = posterior_coefficient_summary(trace, col_names)
    return plot_coefficient_forest_df(
        s.reset_index().rename(columns={"index": "variable"}),
        output_path,
        figsize=figsize,
        show=show,
        title=title,
        label_col="variable",
    )


def plot_coefficient_traces_and_posteriors(
    trace,
    col_names: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    *,
    var_names: Optional[list[str]] = None,
    per_coefficient_traces: bool = True,
    include_forest: bool = True,
    show: bool = False,
) -> dict[str, plt.Figure]:
    az = _az()
    out = Path(output_dir) if output_dir else None
    avail = [v for v in (var_names or TRACE_DIAGNOSTIC_VARS) if v in trace.posterior]
    if not avail:
        raise ValueError("No trace variables found.")

    hp = [v for v in HYPERPARAM_TRACE_VARS if v in trace.posterior]
    figures = {
        "trace_hyperparameters": _plot_trace_rows(
            trace,
            [(v, HYPERPARAM_LABELS.get(v, v), None) for v in hp],
            path=out / "coefficient_trace_overview.png" if out else None,
            show=show,
        ),
    }

    if per_coefficient_traces and col_names:
        dim = _beta_dim(trace)
        figures["trace_by_predictor"] = _plot_trace_rows(
            trace,
            [("beta", _label(p), {dim: i}) for i, p in enumerate(col_names)],
            fig_w=11,
            row_h=3.2,
            adjust=(0.16, 0.97, 0.98, 0.04, 0.62, 0.3),
            path=out / "coefficient_traces_by_predictor.png" if out else None,
            show=show,
        )

    ess_vars = hp if hp else [v for v in avail if v != "beta"]
    if not ess_vars:
        ess_vars = avail[:1]
    ess_ax = az.plot_ess(trace, var_names=ess_vars)
    fig_ess = ess_ax.flat[0].figure if isinstance(ess_ax, np.ndarray) else ess_ax.figure
    fig_ess.subplots_adjust(hspace=0.45, wspace=0.25)
    figures["ess"] = fig_ess
    _save(fig_ess, out / "coefficient_ess.png" if out else None, show)

    dim = _beta_dim(trace)
    n_beta = trace.posterior["beta"].sizes[dim]
    names = col_names or [f"beta[{i}]" for i in range(n_beta)]
    panels: list[tuple[str, Optional[dict], str]] = [
        ("beta", {dim: i}, _label(names[i] if i < len(names) else f"beta[{i}]"))
        for i in range(n_beta)
    ]
    if "beta_diversity" in trace.posterior:
        panels.append(("beta_diversity", None, "Beta diversity"))
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig_p, axes_p = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False
    )
    for ax in axes_p.flat[len(panels) :]:
        ax.set_visible(False)
    for ax, (var, coords, title) in zip(axes_p.flat, panels):
        kw = dict(var_names=[var], ax=ax)
        if coords:
            kw["coords"] = coords
        az.plot_posterior(trace, **kw)
        ax.set_title(title, fontsize=11, pad=10)
    fig_p.subplots_adjust(hspace=0.55, wspace=0.35)
    figures["posterior"] = fig_p
    _save(fig_p, out / "coefficient_posterior.png" if out else None, show)

    if include_forest:
        forest_path = out / "coeff_forest.png" if out else None
        figures["posterior_forest"] = plot_posterior_coefficient_forest(
            trace,
            col_names,
            forest_path,
            show=show,
        )
    return figures


def _scalar_hyperparameter_vars(trace) -> list[str]:
    """Scalar posterior variables for compact trace-diagnostic PNGs."""
    return [
        v
        for v in (
            "beta_diversity",
            "mu_global",
            "sigma_site",
            "sigma_ecoregion",
            "sigma",
            "theta",
        )
        if v in trace.posterior
    ]


def save_trace_diagnostics(trace, output_path: Path) -> None:
    az = _az()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # Beta coefficient traces live in coefficient_diagnostics/; keep these PNGs
    # to scalar hyperparameters so ArviZ does not expand beta into 50+ subplots.
    scalar_vars = _scalar_hyperparameter_vars(trace)
    if not scalar_vars:
        return

    plots: list[tuple[str, Any]] = [
        (
            "trace",
            lambda: az.plot_trace(trace, var_names=scalar_vars, compact=True),
        ),
        ("posterior", lambda: az.plot_posterior(trace, var_names=scalar_vars)),
        ("autocorr", lambda: az.plot_autocorr(trace, var_names=scalar_vars)),
        ("ess", lambda: az.plot_ess(trace, var_names=scalar_vars)),
    ]
    if len(scalar_vars) >= 2:
        plots.insert(1, ("pair", lambda: az.plot_pair(trace, var_names=scalar_vars)))

    for name, plot_fn in plots:
        plot_fn()
        plt.savefig(output_path / f"{name}.png", dpi=HBB_FIG_DPI)
        plt.close()


def _scatter_by_class(
    ax, df: pd.DataFrame, classification: np.ndarray, transform=None
) -> None:
    for cls in SPOT_CLASSES:
        m = classification == cls
        kw = {**SPOT_STYLE[cls]}
        label = kw.pop("label")
        ax.scatter(
            df.loc[m, "lon"],
            df.loc[m, "lat"],
            transform=transform,
            label=label,
            **kw,
        )


def plot_observed_vs_expected(
    observed: np.ndarray,
    expected: np.ndarray,
    classification: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if classification is not None:
        for cls in SPOT_CLASSES:
            m = classification == cls
            kw = {k: v for k, v in SPOT_STYLE[cls].items() if k != "label"}
            ax.scatter(
                observed[m] * 100,
                expected[m] * 100,
                label=SPOT_STYLE[cls]["label"],
                **kw,
            )
    else:
        ax.scatter(observed * 100, expected * 100, c="gray", alpha=0.5, s=30)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    sd = np.std(observed) * 100
    ax.plot([0, 100], [1.5 * sd, 100 + 1.5 * sd], "r-", lw=0.5, alpha=0.5)
    ax.plot([0, 100], [-1.5 * sd, 100 - 1.5 * sd], "r-", lw=0.5, alpha=0.5)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Observed % coral cover", ylabel="Expected % coral cover")
    ax.set_title("Observed vs Expected Coral Cover")
    ax.legend(loc="upper left")
    plt.tight_layout()
    _save(fig, output_path, False)
    return fig


def plot_bright_dark_spots_map(
    df: pd.DataFrame,
    classification: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.Robinson(central_longitude=150)},
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgreen", edgecolor="darkgreen")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        _scatter_by_class(ax, df, classification, transform=ccrs.PlateCarree())
        ax.set_global()
    except ImportError:
        fig, ax = plt.subplots(figsize=figsize)
        _scatter_by_class(ax, df, classification)
        ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.legend(loc="lower left")
    ax.set_title("Bright and Dark Spots for Coral Reefs")
    plt.tight_layout()
    _save(fig, output_path, False)
    return fig


def plot_coral_cover_change_histogram(
    change: np.ndarray,
    current_cover: np.ndarray,
    scenario: str,
    year: int,
    relative: bool = True,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if relative:
        values = np.clip(100 * change / current_cover, -100, 100)
        xlabel, bins = "Relative coral cover change (%)", np.linspace(-100, 0, 11)
    else:
        values, xlabel, bins = change * 100, "Absolute coral cover change (% points)", np.linspace(-25, 0, 11)
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set(xlabel=xlabel, ylabel="Count", title=f"{scenario.upper()} year {year}")
    plt.tight_layout()
    _save(fig, output_path, False)
    return fig


def plot_absolute_vs_relative_change(
    current_cover: np.ndarray,
    absolute_change: Any,
    scenarios: list[tuple[str, int]],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 10),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for idx, (ax, (scen, year)) in enumerate(zip(axes.flat, scenarios)):
        key = (scen, year)
        ch = absolute_change(scen, year) if callable(absolute_change) else absolute_change.get(key, np.zeros_like(current_cover))
        ch = np.clip(ch, -1, 0)
        if idx % 2 == 0:
            ax.scatter(current_cover * 100, ch * 100, alpha=0.5, s=10)
            ax.set_ylabel("Absolute change in % coral cover")
            ax.set_ylim(-25, 0)
        else:
            rel = np.clip(ch / current_cover, -1, 0)
            ax.scatter(current_cover * 100, rel * 100, alpha=0.5, s=10)
            ax.set_ylabel("Relative change in % coral cover")
            ax.set_ylim(-100, 0)
        ax.set(xlim=(0, 100), xlabel="Modern observed % coral cover", title=f"{scen.upper()} year {year}")
    plt.tight_layout()
    _save(fig, output_path, False)
    return fig
