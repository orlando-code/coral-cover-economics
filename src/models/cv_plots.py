"""Visualise cross-validation fold splits (spatial and summary views)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.models.coral_data import load_model_ready_data
from src.models.cv_methods import (
    ALL_CV_REGIMES,
    FoldSpec,
    build_all_folds,
    pick_first_existing,
    year_series,
)

FIG_DPI = 150
_REGIME_LABELS = {
    "random_kfold": "Random k-fold",
    "site_group_kfold": "Site-grouped k-fold",
    "ecoregion_group_kfold": "Ecoregion-grouped k-fold",
    "forward_time_blocks": "Forward time blocks",
    "forward_repeat_sites": "Forward repeat sites (temporal holdout)",
    "spatial_kfold": "Spatial k-fold",
    "in_sample": "In-sample (train = test)",
}
_FOLD_CMAP = plt.cm.tab10


def _lon_lat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lon_col = pick_first_existing(df, ["longitude.degrees", "lon", "Longitude.Degrees"])
    lat_col = pick_first_existing(df, ["latitude.degrees", "lat", "Latitude.Degrees"])
    if lon_col is None or lat_col is None:
        raise ValueError("Longitude/latitude columns required for CV spatial plots.")
    return df[lon_col].to_numpy(dtype=float), df[lat_col].to_numpy(dtype=float)


def _time_col(df: pd.DataFrame) -> str | None:
    return pick_first_existing(df, ["days_since_19811231", "year", "Year"])


def _cover_pct(df: pd.DataFrame) -> np.ndarray | None:
    cover_col = pick_first_existing(df, ["Average_coral_cover", "average_coral_cover"])
    if cover_col is None:
        return None
    cover = df[cover_col].astype(float).to_numpy()
    if np.nanmax(cover) <= 1.0:
        cover = cover * 100.0
    return cover


def plot_fold_sizes(folds: list[FoldSpec], *, output_path: Path) -> plt.Figure:
    rows = [
        {
            "regime": _REGIME_LABELS.get(f.name, f.name),
            "fold": f.fold,
            "n_train": len(f.train_idx),
            "n_test": len(f.test_idx),
        }
        for f in folds
    ]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(
        axes, ["n_train", "n_test"], ["Training rows per fold", "Test rows per fold"]
    ):
        for regime, sub in df.groupby("regime"):
            sub = sub.sort_values("fold")
            ax.plot(sub["fold"], sub[col], marker="o", label=regime)
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Rows")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_regime_spatial_folds(
    df: pd.DataFrame,
    regime_folds: list[FoldSpec],
    *,
    output_path: Path,
) -> plt.Figure:
    lon, lat = _lon_lat(df)
    n = len(regime_folds)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    regime = regime_folds[0].name
    in_sample = regime == "in_sample"
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False
    )

    for ax, fold in zip(axes.ravel(), regime_folds):
        if in_sample:
            ax.scatter(lon, lat, color="#1f77b4", s=8, alpha=0.55, linewidths=0)
            ax.set_title(f"All data (n={len(fold.test_idx):,})")
        else:
            train_mask = np.zeros(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
            train_mask[fold.train_idx] = True
            test_mask[fold.test_idx] = True
            ax.scatter(
                lon[train_mask],
                lat[train_mask],
                c="#bdbdbd",
                s=4,
                alpha=0.35,
                linewidths=0,
            )
            ax.scatter(
                lon[test_mask],
                lat[test_mask],
                color=_FOLD_CMAP(fold.fold % 10),
                s=10,
                alpha=0.8,
                linewidths=0,
            )
            ax.set_title(f"Fold {fold.fold} (test n={len(fold.test_idx):,})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle(_REGIME_LABELS.get(regime, regime), fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def _site_year_cover(
    df: pd.DataFrame,
    *,
    sites: np.ndarray,
    years: np.ndarray,
    cover: np.ndarray,
    repeat_sites: set[int],
) -> pd.DataFrame:
    """One mean cover value per site-year (handles duplicate visits in the same year)."""
    plot_df = pd.DataFrame({"site": sites, "year": years, "cover": cover})
    plot_df = plot_df.loc[plot_df["site"].isin(repeat_sites)]
    return (
        plot_df.groupby(["site", "year"], as_index=False)["cover"]
        .mean()
        .sort_values(["site", "year"])
    )


def _yearly_cover_summary(site_year: pd.DataFrame) -> pd.DataFrame:
    return (
        site_year.groupby("year", as_index=False)["cover"]
        .agg(
            median="median",
            p25=lambda s: float(s.quantile(0.25)),
            p75=lambda s: float(s.quantile(0.75)),
            n_sites="count",
        )
        .sort_values("year")
    )


def _add_cutoff_marks(
    ax: plt.Axes,
    *,
    cutoff_year: int | None,
    year_min: int,
    year_max: int,
) -> None:
    if cutoff_year is None:
        return
    ax.axvspan(year_min, cutoff_year, color="#bdbdbd", alpha=0.12, zorder=0)
    ax.axvspan(cutoff_year, year_max + 1, color="#d62728", alpha=0.08, zorder=0)
    ax.axvline(
        cutoff_year,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"Cutoff {cutoff_year}",
        zorder=4,
    )


def _plot_yearly_median_band(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    color: str,
    label: str,
    min_n_sites: int = 30,
) -> pd.DataFrame:
    """Plot median + IQR; omit sparse years from the line to avoid misleading cliffs."""
    reliable = summary.loc[summary["n_sites"] >= min_n_sites].copy()
    sparse = summary.loc[summary["n_sites"] < min_n_sites].copy()

    if not reliable.empty:
        ax.fill_between(
            reliable["year"],
            reliable["p25"],
            reliable["p75"],
            color=color,
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            reliable["year"],
            reliable["median"],
            color=color,
            linewidth=2.2,
            label=label,
            zorder=3,
        )

    if not sparse.empty:
        ax.scatter(
            sparse["year"],
            sparse["median"],
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            s=48,
            zorder=4,
            label=f"{label} (n<{min_n_sites} sites/yr)",
        )
        for _, row in sparse.iterrows():
            ax.annotate(
                f"n={int(row['n_sites'])}",
                (row["year"], row["median"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=color,
            )
    return reliable


def plot_repeat_site_cover_trajectories(
    df: pd.DataFrame,
    *,
    output_path: Path,
    cutoff_year: int | None = None,
    highlight_sites: set[int] | None = None,
    n_sample_sites: int = 24,
    seed: int = 42,
    min_n_sites_for_trend: int = 30,
) -> plt.Figure | None:
    """Coral cover vs year for repeat-visit sites (summary + sampled examples)."""
    if "site" not in df.columns:
        return None
    cover = _cover_pct(df)
    if cover is None:
        return None

    years = year_series(df).to_numpy()
    sites = df["site"].to_numpy()
    counts = df.groupby("site").size()
    repeat_sites = set(counts[counts >= 2].index.astype(int))
    site_year = _site_year_cover(
        df, sites=sites, years=years, cover=cover, repeat_sites=repeat_sites
    )
    year_min = int(site_year["year"].min())
    year_max = int(site_year["year"].max())

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        height_ratios=[1.15, 1],
        sharex=True,
    )

    # Top panel: population summary (median + IQR), not every site line.
    ax = axes[0]
    _add_cutoff_marks(ax, cutoff_year=cutoff_year, year_min=year_min, year_max=year_max)
    all_summary = _yearly_cover_summary(site_year)
    _plot_yearly_median_band(
        ax,
        all_summary,
        color="#4c72b0",
        label=f"All repeat sites (n={len(repeat_sites):,})",
        min_n_sites=min_n_sites_for_trend,
    )

    holdout_sites = highlight_sites or set()
    if holdout_sites:
        holdout_sy = site_year.loc[site_year["site"].isin(holdout_sites)]
        holdout_summary = _yearly_cover_summary(holdout_sy)
        _plot_yearly_median_band(
            ax,
            holdout_summary,
            color="#d62728",
            label=f"Holdout sites (n={len(holdout_sites):,})",
            min_n_sites=min_n_sites_for_trend,
        )

    ax_n = ax.twinx()
    ax_n.bar(
        all_summary["year"],
        all_summary["n_sites"],
        width=0.65,
        color="#bdbdbd",
        alpha=0.35,
        zorder=0,
        label="Sites with data (all repeat)",
    )
    ax_n.set_ylabel("Sites with data per year", fontsize=8)
    ax_n.tick_params(axis="y", labelsize=7)
    ax_n.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    ax.set_ylabel("% coral cover")
    ax.set_title(
        "Repeat-site cover over time — yearly median and IQR "
        f"(line stops when <{min_n_sites_for_trend} sites/year)"
    )
    ax.set_ylim(bottom=0)
    handles_l, labels_l = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_n.get_legend_handles_labels()
    ax.legend(handles_l + handles_r, labels_l + labels_r, loc="upper right", fontsize=7)

    # Bottom panel: a small random sample of individual trajectories.
    ax = axes[1]
    _add_cutoff_marks(ax, cutoff_year=cutoff_year, year_min=year_min, year_max=year_max)
    sample_pool = sorted(holdout_sites) if holdout_sites else sorted(repeat_sites)
    rng = np.random.default_rng(seed)
    n_draw = min(n_sample_sites, len(sample_pool))
    sample_sites = (
        rng.choice(sample_pool, size=n_draw, replace=False).tolist()
        if n_draw < len(sample_pool)
        else list(sample_pool)
    )
    reliable_years = set(
        all_summary.loc[all_summary["n_sites"] >= min_n_sites_for_trend, "year"].astype(
            int
        )
    )

    cmap = plt.cm.tab20
    for i, site in enumerate(sorted(sample_sites)):
        sub = site_year.loc[site_year["site"] == site].copy()
        reliable = sub["year"].isin(reliable_years)
        if reliable.any():
            ax.plot(
                sub.loc[reliable, "year"],
                sub.loc[reliable, "cover"],
                color=cmap(i % 20),
                linewidth=1.4,
                alpha=0.9,
                marker="o",
                markersize=3,
                zorder=3,
            )
        sparse_pts = sub.loc[~reliable]
        if not sparse_pts.empty:
            ax.scatter(
                sparse_pts["year"],
                sparse_pts["cover"],
                facecolors="none",
                edgecolors=cmap(i % 20),
                s=28,
                linewidths=1.2,
                zorder=3,
            )

    pool_label = "holdout" if holdout_sites else "repeat"
    ax.set_title(
        f"Example {pool_label} site trajectories "
        f"({n_draw} of {len(sample_pool):,} sites)"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("% coral cover")
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Repeat-site coral cover trajectories",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_forward_repeat_sites(
    df: pd.DataFrame,
    regime_folds: list[FoldSpec],
    *,
    output_path: Path,
) -> plt.Figure | None:
    fold = regime_folds[0]
    cutoff = fold.meta.get("cutoff_year")
    if cutoff is None:
        return None

    years = year_series(df).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(
        years[fold.train_idx],
        np.full(len(fold.train_idx), 0.25),
        c="#9e9e9e",
        s=14,
        alpha=0.55,
        label=f"Train (n={len(fold.train_idx):,})",
    )
    ax.scatter(
        years[fold.test_idx],
        np.full(len(fold.test_idx), 0.75),
        c="#d62728",
        s=18,
        alpha=0.85,
        label=f"Test repeat-site future (n={len(fold.test_idx):,})",
    )
    ax.axvline(cutoff, color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.set_yticks([0.25, 0.75], ["Train", "Test"])
    ax.set_xlabel("Year")
    ax.set_title("Temporal holdout split")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    frac = fold.meta.get("test_fraction_actual", np.nan)
    target = fold.meta.get("test_fraction_target", 0.2)
    bars = ax.bar(
        ["Train", "Test (eligible)"],
        [len(fold.train_idx), len(fold.test_idx)],
        color=["#9e9e9e", "#d62728"],
    )
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Rows")
    ax.set_title(
        f"~{100 * target:.0f}/{100 * (1 - target):.0f} target  |  "
        f"actual test fraction {100 * frac:.1f}%  |  "
        f"repeat sites {fold.meta.get('n_repeat_sites', '?')}"
    )

    fig.suptitle(_REGIME_LABELS.get(fold.name, fold.name), fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_forward_time_blocks(
    df: pd.DataFrame,
    regime_folds: list[FoldSpec],
    *,
    output_path: Path,
) -> plt.Figure | None:
    tcol = _time_col(df)
    if tcol is None:
        return None
    times = df[tcol].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(times, np.zeros(len(times)), c="#e0e0e0", s=8, alpha=0.4)
    for fold in regime_folds:
        test_t = times[fold.test_idx]
        ax.scatter(
            test_t,
            np.full(len(test_t), fold.fold),
            c=[_FOLD_CMAP(fold.fold % 10)],
            s=12,
            alpha=0.85,
            label=f"Block {fold.fold} (n={len(fold.test_idx):,})",
        )
    ax.set_xlabel(tcol)
    ax.set_ylabel("Test block")
    ax.set_title("Forward time blocks — test periods")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_group_coverage(
    df: pd.DataFrame,
    regime_folds: list[FoldSpec],
    group_col: str,
    *,
    output_path: Path,
) -> plt.Figure:
    regime = regime_folds[0].name
    rows: list[dict[str, object]] = []
    for fold in regime_folds:
        test_groups = df.iloc[fold.test_idx][group_col].nunique()
        train_groups = df.iloc[fold.train_idx][group_col].nunique()
        rows.append({"fold": fold.fold, "split": "train", "n_groups": train_groups})
        rows.append({"fold": fold.fold, "split": "test", "n_groups": test_groups})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for split, color in [("train", "#9e9e9e"), ("test", "#d62728")]:
        sub = plot_df.loc[plot_df["split"] == split]
        ax.bar(
            sub["fold"] + (-0.2 if split == "train" else 0.2),
            sub["n_groups"],
            width=0.35,
            label=split,
            color=color,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel(f"Unique {group_col}s")
    ax.set_title(f"{_REGIME_LABELS.get(regime, regime)} — group coverage")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_cv_regimes(
    df: pd.DataFrame,
    folds: list[FoldSpec],
    output_dir: Path,
) -> None:
    """Write spatial and summary plots for each CV regime."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fold_sizes(folds, output_path=output_dir / "fold_sizes.png")

    repeat_folds = [f for f in folds if f.name == "forward_repeat_sites"]
    cutoff_year = repeat_folds[0].meta.get("cutoff_year") if repeat_folds else None
    highlight_sites = None
    if repeat_folds:
        highlight_sites = {
            int(s) for s in df.iloc[repeat_folds[0].test_idx]["site"].unique()
        }
    plot_repeat_site_cover_trajectories(
        df,
        output_path=output_dir / "repeat_site_cover_trajectories.png",
        cutoff_year=cutoff_year,
        highlight_sites=highlight_sites,
    )

    regimes = sorted({f.name for f in folds})
    for regime in regimes:
        regime_folds = sorted(
            [f for f in folds if f.name == regime], key=lambda f: f.fold
        )
        plot_regime_spatial_folds(
            df,
            regime_folds,
            output_path=output_dir / f"spatial_{regime}.png",
        )
        if regime == "forward_time_blocks":
            plot_forward_time_blocks(
                df,
                regime_folds,
                output_path=output_dir / "time_forward_blocks.png",
            )
        if regime == "forward_repeat_sites":
            plot_forward_repeat_sites(
                df,
                regime_folds,
                output_path=output_dir / "time_forward_repeat_sites.png",
            )
        if regime == "site_group_kfold" and "site" in df.columns:
            plot_group_coverage(
                df, regime_folds, "site", output_path=output_dir / "groups_site.png"
            )
        if regime == "ecoregion_group_kfold" and "region" in df.columns:
            plot_group_coverage(
                df, regime_folds, "region", output_path=output_dir / "groups_region.png"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CV fold splits")
    parser.add_argument(
        "--regimes",
        type=str,
        default=",".join(ALL_CV_REGIMES),
        help="Comma-separated CV regimes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: sully_og/output/cross_validation/cv_regime_plots",
    )
    args = parser.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    out = args.output_dir or (
        config.sully_og_dir / "output" / "cross_validation" / "cv_regime_plots"
    )

    df = load_model_ready_data()
    folds, skipped = build_all_folds(
        df,
        validation_regimes=regimes,
        k_folds=args.k_folds,
        seed=args.seed,
        spatial_bins=args.spatial_bins,
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(out / "skipped_regimes.csv", index=False)
    plot_cv_regimes(df, folds, out)
    print(f"Wrote CV regime plots to {out}")


if __name__ == "__main__":
    main()
