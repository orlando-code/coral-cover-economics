"""Residual diagnostic plots for model and cross-validation pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import linregress, t

FitMethod = Literal["ols", "loess", "none"]

DAYS_COL = "days_since_19811231"
REEF_COL_CANDIDATES = ("reef_id", "Reef_ID", "reef")

# Backwards-compatible alias for imports expecting FIG_DPI at module level.
FIG_DPI = 300


@dataclass(frozen=True)
class OLSFit:
    slope: float
    intercept: float
    r2: float
    slope_stderr: float
    intercept_stderr: float
    n: int

    @property
    def slope_se(self) -> float:
        return self.slope_stderr


@dataclass(frozen=True)
class LoessFit:
    frac: float
    degree: int
    r2: float
    n: int
    x_line: np.ndarray
    y_line: np.ndarray


@dataclass
class ResidualPanelConfig:
    """Per-panel behaviour and labels."""

    fit_regression: bool = True
    use_meta_regression: bool | None = None
    integer_xticks: bool = False
    xlabel: str | None = None
    title: str | None = None


@dataclass
class ResidualPlotConfig:
    """Shared formatting for residual diagnostic scatter plots."""

    dpi: int = 300
    scatter_size: float = 2.0
    scatter_alpha: float = 0.5
    scatter_size_cv: float = 4.0
    scatter_alpha_cv: float = 0.55
    regression_color: str = "darkred"
    regression_ci_alpha: float = 0.2
    regression_label: str = "OLS fit"
    meta_regression_label: str = "Meta-regression"
    ci_label: str = "95% CI"
    zero_line_color: str = "grey"
    zero_line_style: str = "--"
    grid_linestyle: str = "--"
    grid_alpha: float = 0.5
    ylabel: str = "Residuals (predicted - observed)"
    annotation_fontsize: int = 11
    annotation_color: str = "darkred"
    annotation_xy: tuple[float, float] = (0.04, 0.96)
    annotation_bbox: dict[str, Any] = field(
        default_factory=lambda: {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "darkgrey",
            "alpha": 0.7,
        }
    )
    legend_fontsize: int = 10
    legend_loc: str = "upper right"
    fold_cmap: str = "tab10"
    use_meta_regression: bool = True
    xlim_pad_fraction: float = 0.05
    integer_xticks_max: int = 40
    panels: dict[str, ResidualPanelConfig] = field(default_factory=dict)

    def copy(self) -> ResidualPlotConfig:
        fields = {k: v for k, v in self.__dict__.items() if k != "panels"}
        fields["panels"] = {
            slug: ResidualPanelConfig(**cfg.__dict__)
            for slug, cfg in self.panels.items()
        }
        return ResidualPlotConfig(**fields)

    def update(self, **kwargs: Any) -> ResidualPlotConfig:
        out = self.copy()
        panel_updates = kwargs.pop("panels", None)
        for key, value in kwargs.items():
            setattr(out, key, value)
        if panel_updates:
            merged = dict(out.panels)
            for slug, panel_cfg in panel_updates.items():
                if isinstance(panel_cfg, ResidualPanelConfig):
                    merged[slug] = panel_cfg
                elif isinstance(panel_cfg, dict):
                    base = merged.get(slug, default_panel(slug))
                    merged[slug] = ResidualPanelConfig(**{**base.__dict__, **panel_cfg})
                else:
                    raise TypeError(
                        "panel overrides must be ResidualPanelConfig or dict"
                    )
            out.panels = merged
        return out

    def panel(self, slug: str) -> ResidualPanelConfig:
        if slug in self.panels:
            return self.panels[slug]
        return default_panel(slug)


def default_panel(slug: str) -> ResidualPanelConfig:
    defaults: dict[str, ResidualPanelConfig] = {
        "test_index": ResidualPanelConfig(
            fit_regression=False,
            xlabel="Test point index",
        ),
        "reef_training_count": ResidualPanelConfig(
            fit_regression=True,
            integer_xticks=True,
            xlabel="Number of training points",
        ),
        "years_since_training": ResidualPanelConfig(
            fit_regression=True,
            xlabel="Years since most recent training date",
        ),
    }
    return defaults.get(slug, ResidualPanelConfig())


DEFAULT_RESIDUAL_PLOT_CONFIG = ResidualPlotConfig()


def _reef_col(df: pd.DataFrame) -> str:
    for col in REEF_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"No reef id column found (tried {REEF_COL_CANDIDATES}).")


def make_line_str(slope: float, intercept: float) -> str:
    if intercept < 0:
        return rf"$y = {slope:.3f}\,x {intercept:.2f}$"
    return rf"$y = {slope:.3f}\,x + {intercept:.2f}$"


def loess_annotation(fit: LoessFit) -> str:
    return (
        rf"LOESS ($\alpha={fit.frac:.2f}$, deg={fit.degree})"
        + "\n"
        + rf"$R^2={fit.r2:.3f}$"
    )


def loess_legend_label(fit: LoessFit) -> str:
    return rf"LOESS fit ($\alpha={fit.frac:.2f}$, deg={fit.degree})"


def apply_residual_plot_formatting(
    ax: Axes,
    config: ResidualPlotConfig,
    *,
    fit: OLSFit | None = None,
    loess_fit: LoessFit | None = None,
    annotation_suffix: str = "",
) -> Axes:
    """Apply shared axis styling to a residual scatter panel."""
    ax.set_ylabel(config.ylabel)
    start_x, end_x = ax.get_xlim()
    x_diff = end_x - start_x
    pad = config.xlim_pad_fraction
    ax.hlines(
        0,
        start_x - x_diff * pad,
        end_x + x_diff * pad,
        color=config.zero_line_color,
        ls=config.zero_line_style,
    )

    if loess_fit is not None:
        annot_str = loess_annotation(loess_fit)
    elif fit is not None:
        annot_str = (
            make_line_str(fit.slope, fit.intercept) + "\n" + rf"$R^2={fit.r2:.3f}$"
        )
    else:
        annot_str = ""

    if annot_str:
        if annotation_suffix:
            annot_str += f"\n{annotation_suffix}"
        ax.annotate(
            annot_str,
            xy=config.annotation_xy,
            xycoords="axes fraction",
            fontsize=config.annotation_fontsize,
            alpha=0.9,
            color=config.annotation_color,
            ha="left",
            va="top",
            bbox=config.annotation_bbox,
        )

    ax.grid("both", linestyle=config.grid_linestyle, alpha=config.grid_alpha)
    ax.set_xlim(start_x - x_diff * pad, end_x + x_diff * pad)
    return ax


def format_residual_axis(
    ax: Axes,
    *,
    slope: float | None = None,
    intercept: float | None = None,
    r2: float | None = None,
    annotation_suffix: str = "",
    config: ResidualPlotConfig | None = None,
) -> Axes:
    """Backwards-compatible wrapper around :func:`apply_residual_plot_formatting`."""
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    fit = None
    if slope is not None and intercept is not None and r2 is not None:
        fit = OLSFit(
            slope=slope,
            intercept=intercept,
            r2=r2,
            slope_stderr=float("nan"),
            intercept_stderr=float("nan"),
            n=0,
        )
    return apply_residual_plot_formatting(
        ax, cfg, fit=fit, annotation_suffix=annotation_suffix
    )


def fit_ols(x_vals: np.ndarray, y_vals: np.ndarray) -> OLSFit | None:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    if np.allclose(x, x[0]):
        return None
    res = linregress(x, y)
    return OLSFit(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r2=float(res.rvalue**2),
        slope_stderr=float(res.stderr),
        intercept_stderr=float(res.intercept_stderr),
        n=int(len(x)),
    )


def fit_loess(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    frac: float = 0.3,
    degree: int = 1,
    it: int = 3,
    n_eval: int = 200,
) -> LoessFit | None:
    """Fit a LOESS smooth using library implementations.

    Degree 1 uses statsmodels LOWESS (robust local linear regression).
    Degrees 0 and 2 use scikit-misc (local constant / quadratic LOESS).
    """
    if degree not in (0, 1, 2):
        raise ValueError("LOESS degree must be 0, 1, or 2")

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    min_points = max(3, degree + 2)
    if len(x) < min_points:
        return None
    if np.allclose(x, x[0]):
        return None

    x_line = np.linspace(float(np.min(x)), float(np.max(x)), n_eval)

    if degree == 1:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        y_hat = np.asarray(
            lowess(y, x, frac=frac, it=it, xvals=x, return_sorted=False),
            dtype=float,
        )
        y_line = np.asarray(
            lowess(y, x, frac=frac, it=it, xvals=x_line, return_sorted=False),
            dtype=float,
        )
    else:
        from skmisc.loess import loess as sk_loess

        model = sk_loess(
            x,
            y,
            span=frac,
            degree=degree,
            family="symmetric",
            iterations=it,
        )
        model.fit()
        y_hat = np.asarray(model.predict(x).values, dtype=float)
        y_line = np.asarray(model.predict(x_line).values, dtype=float)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return LoessFit(
        frac=float(frac),
        degree=int(degree),
        r2=r2,
        n=int(len(x)),
        x_line=x_line,
        y_line=y_line,
    )


def meta_regress_ols(fits: list[OLSFit]) -> OLSFit | None:
    """Inverse-variance weighted combination of per-fold OLS coefficients."""
    valid = [f for f in fits if f is not None and np.isfinite(f.slope_stderr)]
    if not valid:
        valid = [f for f in fits if f is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    slope_weights = np.array(
        [1.0 / max(f.slope_stderr**2, 1e-12) for f in valid], dtype=float
    )
    slope_meta = float(
        np.sum(slope_weights * [f.slope for f in valid]) / slope_weights.sum()
    )

    intercept_weights = np.array(
        [1.0 / max(f.intercept_stderr**2, 1e-12) for f in valid], dtype=float
    )
    intercept_meta = float(
        np.sum(intercept_weights * [f.intercept for f in valid])
        / intercept_weights.sum()
    )

    r2_meta = float(np.mean([f.r2 for f in valid]))
    slope_se = float(np.sqrt(1.0 / slope_weights.sum()))
    intercept_se = float(np.sqrt(1.0 / intercept_weights.sum()))
    return OLSFit(
        slope=slope_meta,
        intercept=intercept_meta,
        r2=r2_meta,
        slope_stderr=slope_se,
        intercept_stderr=intercept_se,
        n=int(sum(f.n for f in valid)),
    )


def _regression_ci_band(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    fit: OLSFit,
    x_line: np.ndarray,
) -> np.ndarray:
    n = len(x_vals)
    dof = max(1, n - 2)
    tval = t.ppf(0.975, dof)
    y_hat = fit.slope * x_vals + fit.intercept
    resid = y_vals - y_hat
    se_reg = np.sqrt(np.sum(resid**2) / dof)
    mean_x = np.mean(x_vals)
    denom = np.sum((x_vals - mean_x) ** 2)
    if denom <= 0:
        return np.zeros_like(x_line)
    se_fit = se_reg * np.sqrt(1 / n + (x_line - mean_x) ** 2 / denom)
    return tval * se_fit


def _draw_regression_line(
    ax: Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    fit: OLSFit,
    config: ResidualPlotConfig,
    *,
    label: str | None = None,
) -> None:
    x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
    y_line = fit.slope * x_line + fit.intercept
    ci = _regression_ci_band(x_vals, y_vals, fit, x_line)
    color = config.regression_color
    ax.fill_between(
        x_line,
        y_line - ci,
        y_line + ci,
        color=color,
        alpha=config.regression_ci_alpha,
        label=config.ci_label,
    )
    ax.plot(x_line, y_line, c=color, label=label or config.regression_label)


def style_observed_vs_predicted_axes(
    ax: Axes,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str | None = None,
    config: ResidualPlotConfig | None = None,
    xlabel: str = "Observed coral cover",
    ylabel: str = "Predicted coral cover",
    lims: tuple[float, float] = (0.0, 1.0),
) -> Axes:
    """Style an observed-vs-predicted panel (1:1 line, OLS fit + 95% CI, grid)."""
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ax.scatter(
        y_obs,
        y_pred,
        alpha=cfg.scatter_alpha,
        s=cfg.scatter_size,
        edgecolors="none",
    )
    ax.plot(lims, lims, color=cfg.zero_line_color, ls="--", lw=1, alpha=0.85, label="1:1")

    fit = fit_ols(y_obs, y_pred)
    if fit is not None:
        _draw_regression_line(ax, y_obs, y_pred, fit, cfg)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)

    residuals = y_pred - y_obs
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if fit is not None:
        annot_str = (
            make_line_str(fit.slope, fit.intercept)
            + "\n"
            + rf"$R^2={fit.r2:.3f}$"
            + "\n"
            + rf"RMSE={rmse:.3f}"
        )
        ax.legend(loc=cfg.legend_loc, fontsize=cfg.legend_fontsize)
    else:
        annot_str = rf"$R^2={r2:.3f}$" + "\n" + rf"RMSE={rmse:.3f}"

    ax.annotate(
        annot_str,
        xy=cfg.annotation_xy,
        xycoords="axes fraction",
        fontsize=cfg.annotation_fontsize,
        alpha=0.9,
        color=cfg.annotation_color,
        ha="left",
        va="top",
        bbox=cfg.annotation_bbox,
    )
    ax.grid("both", linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    return ax


def _draw_loess_line(
    ax: Axes,
    fit: LoessFit,
    config: ResidualPlotConfig,
    *,
    label: str | None = None,
) -> None:
    color = config.regression_color
    ax.plot(
        fit.x_line,
        fit.y_line,
        c=color,
        label=label or loess_legend_label(fit),
    )


def _apply_integer_xticks(
    ax: Axes, x_vals: np.ndarray, config: ResidualPlotConfig
) -> None:
    if len(x_vals) == 0:
        return
    xmax = int(np.nanmax(x_vals))
    if 0 < xmax <= config.integer_xticks_max:
        xticks = np.arange(1, xmax + 1)
        ax.set_xticks(xticks[::2])
        ax.set_xticklabels(xticks[::2])


def enrich_residual_frame(
    frame: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach reef-level training counts and temporal gap columns."""
    out = frame.copy()
    reef_col = _reef_col(test_df)

    merge_cols = ["row_id", reef_col]
    if DAYS_COL in test_df.columns:
        merge_cols.append(DAYS_COL)
    for col in ("site", "region"):
        if col in test_df.columns and col not in out.columns:
            merge_cols.append(col)

    if "row_id" in out.columns and "row_id" in test_df.columns:
        meta = test_df[merge_cols].drop_duplicates(subset=["row_id"])
        out = out.merge(meta, on="row_id", how="left", suffixes=("", "_test"))
    else:
        for col in merge_cols:
            if col != "row_id" and col in test_df.columns:
                out[col] = test_df[col].to_numpy()

    train = train_df.copy()
    reef_id_counts = train[reef_col].value_counts()
    out["reef_id_count"] = out[reef_col].map(reef_id_counts)

    if DAYS_COL in train.columns and DAYS_COL in out.columns:
        train[DAYS_COL] = pd.to_numeric(train[DAYS_COL], errors="coerce")
        out[DAYS_COL] = pd.to_numeric(out[DAYS_COL], errors="coerce")
        most_recent = (
            train.groupby(reef_col)[DAYS_COL].max().rename("most_recent_training_date")
        )
        out = out.drop(columns=["most_recent_training_date"], errors="ignore")
        out = out.merge(
            most_recent.reset_index(),
            on=reef_col,
            how="left",
        )
        out["time_diff"] = out[DAYS_COL] - out["most_recent_training_date"]
        out["time_diff_years"] = out["time_diff"] / 365.25

    if "y_obs" in out.columns and "y_pred" in out.columns:
        out["residuals"] = out["y_pred"] - out["y_obs"]
    return out


def build_residual_frame(
    predictions: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    if "row_id" in predictions.columns and "row_id" in test_df.columns:
        merged = predictions.merge(
            test_df, on="row_id", how="left", suffixes=("", "_test")
        )
    else:
        merged = predictions.copy()
    return enrich_residual_frame(merged, test_df=test_df, train_df=train_df)


def add_fold_specific_features(
    pred_df: pd.DataFrame,
    full_df: pd.DataFrame,
    folds: list[Any],
) -> pd.DataFrame:
    """Add per-fold training-derived features (reef counts, temporal gaps)."""
    if pred_df.empty:
        return pred_df

    out = pred_df.copy()
    reef_col = _reef_col(full_df)
    fold_tags = out["fold_tag"].unique() if "fold_tag" in out.columns else []
    fold_map = {f"{f.name}__{f.fold}": f for f in folds}

    reef_counts = pd.Series(index=out.index, dtype=float)
    time_diff_years = pd.Series(index=out.index, dtype=float)

    for fold_tag in fold_tags:
        fold = fold_map.get(str(fold_tag))
        if fold is None:
            continue
        mask = out["fold_tag"] == fold_tag
        if not mask.any():
            continue
        train_df = full_df.iloc[fold.train_idx].reset_index(drop=True)
        test_rows = out.loc[mask]
        enriched = enrich_residual_frame(
            test_rows,
            test_df=full_df.iloc[fold.test_idx].reset_index(drop=True),
            train_df=train_df,
        )
        reef_counts.loc[mask] = enriched["reef_id_count"].to_numpy()
        if "time_diff_years" in enriched.columns:
            time_diff_years.loc[mask] = enriched["time_diff_years"].to_numpy()

    if reef_col not in out.columns and reef_col in full_df.columns:
        if "row_id" in out.columns:
            out = out.merge(
                full_df[["row_id", reef_col]].drop_duplicates(),
                on="row_id",
                how="left",
            )
    out["reef_id_count"] = reef_counts
    out["time_diff_years"] = time_diff_years
    if "y_obs" in out.columns and "y_pred" in out.columns:
        out["residuals"] = out["y_pred"] - out["y_obs"]
    return out


def plot_residual_scatter(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    xlabel: str | None = None,
    fold_labels: np.ndarray | None = None,
    ax: Axes | None = None,
    output_path: Path | None = None,
    config: ResidualPlotConfig | None = None,
    panel_slug: str | None = None,
    fit_regression: bool | None = None,
    fit_method: FitMethod = "ols",
    loess_frac: float = 0.3,
    loess_degree: int = 1,
    loess_iters: int = 3,
    use_meta_regression: bool | None = None,
    integer_xticks: bool | None = None,
    title: str | None = None,
    show: bool = False,
) -> Axes:
    """Scatter of residuals vs a covariate with optional OLS or LOESS overlay.

    Pass ``fit_method="none"`` for scatter only (no trend overlay), regardless of
    ``fit_regression`` or panel defaults.
    """
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    panel = cfg.panel(panel_slug) if panel_slug else ResidualPanelConfig()

    if fit_method == "none":
        do_fit = False
    else:
        do_fit = panel.fit_regression if fit_regression is None else fit_regression
    do_meta = (
        cfg.use_meta_regression if use_meta_regression is None else use_meta_regression
    )
    if panel.use_meta_regression is not None and use_meta_regression is None:
        do_meta = panel.use_meta_regression
    do_integer_xticks = (
        panel.integer_xticks if integer_xticks is None else integer_xticks
    )
    xlabel = xlabel if xlabel is not None else panel.xlabel
    title = title if title is not None else panel.title

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if fold_labels is not None:
        fold_labels = np.asarray(fold_labels)[mask]

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(dpi=cfg.dpi)

    annotation_suffix = ""
    pooled_fit: OLSFit | None = None
    loess_fit: LoessFit | None = None
    drew_regression = False

    if fold_labels is not None and len(np.unique(fold_labels)) > 1:
        unique_folds = sorted(np.unique(fold_labels), key=lambda v: (str(v), v))
        cmap = plt.get_cmap(cfg.fold_cmap, max(len(unique_folds), 1))
        per_fold_fits: list[OLSFit] = []
        for i, fold in enumerate(unique_folds):
            fmask = fold_labels == fold
            ax.scatter(
                x[fmask],
                y[fmask],
                s=cfg.scatter_size_cv,
                alpha=cfg.scatter_alpha_cv,
                color=cmap(i % 10),
                label=f"Fold {fold}",
                edgecolors="none",
            )
            if do_fit and fit_method == "ols":
                fold_fit = fit_ols(x[fmask], y[fmask])
                if fold_fit is not None:
                    per_fold_fits.append(fold_fit)

        if do_fit:
            if fit_method == "loess":
                loess_fit = fit_loess(
                    x,
                    y,
                    frac=loess_frac,
                    degree=loess_degree,
                    it=loess_iters,
                )
                if loess_fit is not None:
                    _draw_loess_line(ax, loess_fit, cfg)
                    drew_regression = True
            elif fit_method == "ols" and do_meta and per_fold_fits:
                pooled_fit = meta_regress_ols(per_fold_fits)
                if pooled_fit is not None:
                    annotation_suffix = f"meta ({len(per_fold_fits)} folds)"
                    _draw_regression_line(
                        ax,
                        x,
                        y,
                        pooled_fit,
                        cfg,
                        label=cfg.meta_regression_label,
                    )
                    drew_regression = True
            elif fit_method == "ols" and per_fold_fits:
                pooled_fit = fit_ols(x, y)
                if pooled_fit is not None:
                    _draw_regression_line(ax, x, y, pooled_fit, cfg)
                    drew_regression = True
    else:
        ax.scatter(
            x,
            y,
            s=cfg.scatter_size,
            alpha=cfg.scatter_alpha,
            edgecolors="none",
        )
        if do_fit:
            if fit_method == "loess":
                loess_fit = fit_loess(
                    x,
                    y,
                    frac=loess_frac,
                    degree=loess_degree,
                    it=loess_iters,
                )
                if loess_fit is not None:
                    _draw_loess_line(ax, loess_fit, cfg)
                    drew_regression = True
            elif fit_method == "ols":
                pooled_fit = fit_ols(x, y)
                if pooled_fit is not None:
                    _draw_regression_line(ax, x, y, pooled_fit, cfg)
                    drew_regression = True

    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if drew_regression:
        ax.legend(loc=cfg.legend_loc, fontsize=cfg.legend_fontsize)
        apply_residual_plot_formatting(
            ax,
            cfg,
            fit=pooled_fit,
            loess_fit=loess_fit,
            annotation_suffix=annotation_suffix,
        )
    else:
        apply_residual_plot_formatting(ax, cfg)

    if do_integer_xticks:
        _apply_integer_xticks(ax, x, cfg)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
        if created_fig and not show:
            plt.close(ax.figure)
    elif created_fig and not show:
        plt.close(ax.figure)

    return ax


def plot_residuals(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    xlabel: str | None = None,
    dpi: int = 300,
    fit_method: FitMethod = "ols",
    loess_frac: float = 0.3,
    loess_degree: int = 1,
    loess_iters: int = 3,
    config: ResidualPlotConfig | None = None,
) -> Axes:
    """Notebook-friendly wrapper around :func:`plot_residual_scatter`.

    Defaults to OLS (linear fit + 95% CI), matching the original notebook plots.
    Pass ``fit_method="loess"`` for a robust LOESS overlay, or ``fit_method="none"``
    for scatter only.
    """
    cfg = (config or DEFAULT_RESIDUAL_PLOT_CONFIG).update(
        dpi=dpi,
        ylabel="Residuals",
        legend_fontsize=11,
    )
    do_fit = fit_method != "none"
    return plot_residual_scatter(
        x_vals,
        y_vals,
        xlabel=xlabel,
        config=cfg,
        fit_regression=do_fit,
        fit_method=fit_method,
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        loess_iters=loess_iters,
        show=True,
    )


def residual_panel_specs(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Return (slug, x_values) pairs available for a residual frame."""
    specs: list[tuple[str, np.ndarray]] = [
        ("test_index", np.arange(len(frame))),
    ]
    if "reef_id_count" in frame.columns and frame["reef_id_count"].notna().any():
        specs.append(
            ("reef_training_count", frame["reef_id_count"].to_numpy(dtype=float))
        )
    if "time_diff_years" in frame.columns and frame["time_diff_years"].notna().any():
        specs.append(
            ("years_since_training", frame["time_diff_years"].to_numpy(dtype=float))
        )
    return specs


def write_residual_diagnostic_plots(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str = "",
    fold_col: str | None = None,
    config: ResidualPlotConfig | None = None,
) -> list[Path]:
    """Write standard residual diagnostic scatter plots."""
    cfg = config or DEFAULT_RESIDUAL_PLOT_CONFIG
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "residuals" not in frame.columns:
        raise ValueError("frame must contain a 'residuals' column")

    written: list[Path] = []
    fold_labels = (
        frame[fold_col].to_numpy() if fold_col and fold_col in frame.columns else None
    )
    use_meta = fold_labels is not None and len(np.unique(fold_labels)) > 1
    stem = f"{prefix}_" if prefix else ""
    y_vals = frame["residuals"].to_numpy(dtype=float)

    for slug, x_vals in residual_panel_specs(frame):
        panel = cfg.panel(slug)
        out = output_dir / f"{stem}residuals_{slug}.png"
        plot_residual_scatter(
            x_vals,
            y_vals,
            fold_labels=fold_labels,
            output_path=out,
            config=cfg,
            panel_slug=slug,
            use_meta_regression=use_meta
            if panel.use_meta_regression is None
            else panel.use_meta_regression,
        )
        written.append(out)
    return written


def write_fold_residual_diagnostics(
    predictions: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: Path,
    config: ResidualPlotConfig | None = None,
) -> list[Path]:
    frame = build_residual_frame(predictions, test_df=test_df, train_df=train_df)
    return write_residual_diagnostic_plots(frame, output_dir, config=config)


def write_cv_residual_diagnostics(
    pred_df: pd.DataFrame,
    full_df: pd.DataFrame,
    folds: list[Any],
    output_dir: Path,
    *,
    prefix: str = "",
    fold_col: str = "fold",
    config: ResidualPlotConfig | None = None,
) -> list[Path]:
    """Regime- or model-level residual plots with fold colour-coding."""
    if pred_df.empty:
        return []
    frame = add_fold_specific_features(pred_df, full_df, folds)
    return write_residual_diagnostic_plots(
        frame,
        output_dir,
        prefix=prefix,
        fold_col=fold_col if fold_col in frame.columns else "fold_tag",
        config=config,
    )
