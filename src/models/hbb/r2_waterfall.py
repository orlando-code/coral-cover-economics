#!/usr/bin/env python3
"""R² waterfall decomposition for hierarchical beta-GLMM CV folds.

Builds a combined figure: an isolated site-mean persistence reference bar plus
model-component increments (ecoregion RE without diversity, diversity moderation,
site RE, environmental covariates).

Run decomposition / CV automatically when outputs are missing::

    python -m src.models.hbb.r2_waterfall
    python -m src.models.hbb.r2_waterfall --run-models --regime forward_repeat_sites
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

from src import config
from src.models.hbb.variants import BETA_VARIANT_LABELS, VARIANTS
from src.models.cv_methods import is_in_sample_regime
from src.plots.plot_config import HBB_FIG_DPI

DEFAULT_PRIMARY_VARIANT = "reparam"
DEFAULT_ECO_VARIANT = "reparam_ecoregion_only"
DEFAULT_REGIME = "forward_repeat_sites"
WATERFALL_VARIANTS = (DEFAULT_PRIMARY_VARIANT, DEFAULT_ECO_VARIANT)

COLOR_CONTEXT = "#B8B8B8"
COLOR_ECO = "#E9C46A"
COLOR_DIV = "#8E6C8A"
COLOR_SITE = "#3B9AB2"
COLOR_ENV = "#2A9D8F"
COLOR_POS = "#3B9AB2"
COLOR_NEG = "#E76F51"

DECOMP_PRED_COLS = (
    "y_pred",
    "y_pred_site_mean",
    "y_pred_re_only",
    "y_pred_fe_only",
    "y_pred_re_eco_nodiv",
    "y_pred_re_eco_div",
)

BAR_WIDTH = 0.7


@dataclass(frozen=True)
class WaterfallStep:
    section: str  # context | model
    label: str
    increment: float
    cumulative: float
    r2_absolute: float
    kind: str  # context | increment | total


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(r2_score(y[mask], pred[mask]))


def decomposition_table_path(beta_glmm_root: Path) -> Path:
    return Path(beta_glmm_root) / "hierarchy_decomposition_all_variants.csv"


def load_decomposition_table(beta_glmm_root: Path) -> pd.DataFrame:
    path = decomposition_table_path(beta_glmm_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing decomposition table: {path}")
    return pd.read_csv(path)


def load_fold_predictions(variant_dir: Path, fold_tag: str) -> pd.DataFrame:
    fold_dir = Path(variant_dir) / "folds" / fold_tag
    path = fold_dir / "predictions_decomposition.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions decomposition: {path}")
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{path} must contain row_id.")
    return df


def _fold_predictions_ready(
    beta_glmm_root: Path,
    variant: str,
    fold_tag: str,
    *,
    eco_variant: str = DEFAULT_ECO_VARIANT,
) -> bool:
    path = (
        beta_glmm_root / variant / "folds" / fold_tag / "predictions_decomposition.csv"
    )
    if not path.exists():
        return False
    if variant != eco_variant:
        return True
    cols = pd.read_csv(path, nrows=0).columns
    return "y_pred_re_eco_nodiv" in cols and "y_pred_re_eco_div" in cols


def _fold_cv_ready(beta_glmm_root: Path, variant: str, fold_tag: str) -> bool:
    fold_dir = beta_glmm_root / variant / "folds" / fold_tag
    return (fold_dir / "predictions.csv").exists() and (fold_dir / "trace.nc").exists()


def _fold_regime(fold_tag: str) -> str:
    return fold_tag.rsplit("__", 1)[0]


def _resolve_fold_tag(
    beta_glmm_root: Path,
    *,
    fold_tag: str | None,
    primary_variant: str,
    regime: str,
) -> str:
    regime_prefix = f"{regime}__"
    if fold_tag:
        if _fold_regime(fold_tag) != regime:
            raise ValueError(
                f"fold_tag {fold_tag!r} belongs to regime {_fold_regime(fold_tag)!r}, "
                f"not {regime!r}"
            )
        return fold_tag
    table_path = decomposition_table_path(beta_glmm_root)
    if table_path.exists():
        table = pd.read_csv(table_path)
        matches = table.loc[
            (table["variant"] == primary_variant)
            & table["fold_tag"].astype(str).str.startswith(regime_prefix),
            "fold_tag",
        ].dropna()
        if not matches.empty:
            return str(matches.iloc[0])
    for variant_dir in (beta_glmm_root / primary_variant, beta_glmm_root):
        folds_root = variant_dir / "folds"
        if not folds_root.is_dir():
            continue
        for fold_dir in sorted(folds_root.iterdir()):
            if fold_dir.is_dir() and fold_dir.name.startswith(regime_prefix):
                return fold_dir.name
    return f"{regime}__1"


def ensure_waterfall_inputs(
    beta_glmm_root: Path,
    *,
    fold_tag: str,
    regime: str = DEFAULT_REGIME,
    primary_variant: str = DEFAULT_PRIMARY_VARIANT,
    eco_variant: str = DEFAULT_ECO_VARIANT,
    run_if_missing: bool = True,
    force_rerun: bool = False,
    cv_output_root: Path | None = None,
) -> None:
    """
    Ensure CV folds and hierarchy decomposition exist for waterfall plotting.

    When ``run_if_missing`` or ``force_rerun`` is set, runs beta-GLMM CV for the
    required variants and post-hoc decomposition.
    """
    beta_glmm_root = Path(beta_glmm_root)
    variants = [primary_variant, eco_variant]
    need_cv = force_rerun or any(
        not _fold_cv_ready(beta_glmm_root, v, fold_tag) for v in variants
    )
    need_decomp = force_rerun or any(
        not _fold_predictions_ready(
            beta_glmm_root, v, fold_tag, eco_variant=eco_variant
        )
        for v in variants
    )

    if not run_if_missing and not force_rerun:
        if need_cv or need_decomp:
            missing = []
            if need_cv:
                missing.append("CV fold outputs")
            if need_decomp:
                missing.append("hierarchy decomposition")
            raise FileNotFoundError(
                "Waterfall inputs missing (" + ", ".join(missing) + "). "
                "Re-run with --run-models."
            )
        return

    if not (need_cv or need_decomp):
        return

    cv_root = Path(cv_output_root or beta_glmm_root.parent)

    if need_cv:
        from src.models.run_cross_validation import run_cross_validation

        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown beta variant(s): {unknown}")

        print(f"Running beta-GLMM CV for {variants} ({regime})…")
        run_cross_validation(
            models=["beta_glmm"],
            regimes=[regime],
            output_dir=cv_root,
            beta_variants=list(variants),
        )
        missing_cv = [
            v for v in variants if not _fold_cv_ready(beta_glmm_root, v, fold_tag)
        ]
        if missing_cv:
            hints = []
            for variant in missing_cv:
                failures_path = beta_glmm_root / variant / "failures.csv"
                if failures_path.exists():
                    hints.append(f"{variant}: {failures_path}")
            hint = "; ".join(hints) if hints else "Re-run with --run-models."
            raise FileNotFoundError(
                f"CV did not produce fold outputs for {fold_tag} "
                f"(variants: {missing_cv}). {hint}"
            )
        need_decomp = any(
            not _fold_predictions_ready(
                beta_glmm_root, v, fold_tag, eco_variant=eco_variant
            )
            for v in variants
        )

    if need_decomp:
        from src.models.hbb.cv_decomposition import ensure_waterfall_decomposition

        print(f"Running hierarchy decomposition for {variants} ({fold_tag})…")
        ensure_waterfall_decomposition(
            beta_glmm_root,
            variants=variants,
            fold_tag=fold_tag,
            regime=regime,
            eco_variant=eco_variant,
        )
        still_missing = [
            v
            for v in variants
            if not _fold_predictions_ready(
                beta_glmm_root, v, fold_tag, eco_variant=eco_variant
            )
        ]
        if still_missing:
            raise FileNotFoundError(
                "Hierarchy decomposition outputs still missing for "
                f"{fold_tag} (variants: {still_missing})."
            )


def merge_variant_predictions(
    beta_glmm_root: Path,
    *,
    fold_tag: str,
    variants: dict[str, str],
) -> pd.DataFrame:
    """Merge prediction columns from multiple variant folds on ``row_id``."""
    beta_glmm_root = Path(beta_glmm_root)
    merged: pd.DataFrame | None = None
    for suffix, variant_name in variants.items():
        pred = load_fold_predictions(beta_glmm_root / variant_name, fold_tag)
        keep_cols = ["row_id", "y_obs"]
        rename: dict[str, str] = {}
        for col in DECOMP_PRED_COLS:
            if col in pred.columns:
                rename[col] = f"{col}_{suffix}"
                keep_cols.append(col)
        part = pred[keep_cols].rename(columns=rename)
        merged = part if merged is None else merged.merge(part, on=["row_id", "y_obs"])
    if merged is None:
        raise ValueError("No variant predictions merged.")
    return merged


def build_combined_waterfall_steps(
    merged: pd.DataFrame,
    *,
    primary_suffix: str = "full",
    eco_suffix: str = "eco",
) -> pd.DataFrame:
    """
    Combined steps: context persistence bar + model component increments.

    Model increments split ecoregion RE into diversity-off / diversity-on subsets,
    then site RE and environmental covariates, summing from persistence to full R².
    """
    y = merged["y_obs"].to_numpy(dtype=float)
    site_mean = merged[f"y_pred_site_mean_{primary_suffix}"].to_numpy(dtype=float)
    full_re = merged[f"y_pred_re_only_{primary_suffix}"].to_numpy(dtype=float)
    full = merged[f"y_pred_{primary_suffix}"].to_numpy(dtype=float)

    eco_nodiv_col = f"y_pred_re_eco_nodiv_{eco_suffix}"
    eco_div_col = f"y_pred_re_eco_div_{eco_suffix}"
    if eco_nodiv_col in merged.columns and eco_div_col in merged.columns:
        eco_nodiv = merged[eco_nodiv_col].to_numpy(dtype=float)
        eco_div = merged[eco_div_col].to_numpy(dtype=float)
    else:
        eco_re = merged[f"y_pred_re_only_{eco_suffix}"].to_numpy(dtype=float)
        eco_nodiv = eco_re
        eco_div = eco_re

    r2_site_mean = _r2(y, site_mean)
    r2_eco_nodiv = _r2(y, eco_nodiv)
    r2_eco_div = _r2(y, eco_div)
    r2_full_re = _r2(y, full_re)
    r2_full = _r2(y, full)

    inc_eco_base = r2_eco_nodiv - r2_site_mean
    inc_div = r2_eco_div - r2_eco_nodiv
    inc_site = r2_full_re - r2_eco_div
    inc_env = r2_full - r2_full_re

    steps = [
        WaterfallStep(
            "context",
            "Site-mean\npersistence",
            r2_site_mean,
            r2_site_mean,
            r2_site_mean,
            "context",
        ),
        WaterfallStep(
            "model",
            "Ecoregion RE\n(no diversity)",
            inc_eco_base,
            r2_eco_nodiv,
            r2_eco_nodiv,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Diversity\nmoderation",
            inc_div,
            r2_eco_div,
            r2_eco_div,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Site random\neffects",
            inc_site,
            r2_full_re,
            r2_full_re,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Environmental\ncovariates",
            inc_env,
            r2_full,
            r2_full,
            "increment",
        ),
        WaterfallStep("model", "Full model", 0.0, r2_full, r2_full, "total"),
    ]
    return _steps_to_frame(steps)


def build_combined_waterfall_from_table(
    table: pd.DataFrame,
    *,
    fold_tag: str,
    primary_variant: str = DEFAULT_PRIMARY_VARIANT,
    eco_variant: str = DEFAULT_ECO_VARIANT,
) -> pd.DataFrame:
    """Fallback combined steps from decomposition summary CSV."""
    base = table.loc[
        (table["fold_tag"] == fold_tag) & (table["variant"] == primary_variant)
    ]
    eco = table.loc[(table["fold_tag"] == fold_tag) & (table["variant"] == eco_variant)]
    if base.empty or eco.empty:
        raise ValueError(
            f"Need '{primary_variant}' and '{eco_variant}' rows for fold {fold_tag}."
        )
    b, e = base.iloc[0], eco.iloc[0]
    r2_site_mean = float(b["r2_site_mean"])
    r2_full_re = float(b["r2_re_only"])
    r2_full = float(b["r2_model"])

    if "r2_re_eco_nodiv" in e and "r2_re_eco_div" in e:
        r2_eco_nodiv = float(e["r2_re_eco_nodiv"])
        r2_eco_div = float(e["r2_re_eco_div"])
    else:
        r2_eco_div = float(e["r2_re_only"])
        r2_eco_nodiv = r2_eco_div

    steps = [
        WaterfallStep(
            "context",
            "Site-mean\npersistence",
            r2_site_mean,
            r2_site_mean,
            r2_site_mean,
            "context",
        ),
        WaterfallStep(
            "model",
            "Ecoregion RE\n(no diversity)",
            r2_eco_nodiv - r2_site_mean,
            r2_eco_nodiv,
            r2_eco_nodiv,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Diversity\nmoderation",
            r2_eco_div - r2_eco_nodiv,
            r2_eco_div,
            r2_eco_div,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Site random\neffects",
            r2_full_re - r2_eco_div,
            r2_full_re,
            r2_full_re,
            "increment",
        ),
        WaterfallStep(
            "model",
            "Environmental\ncovariates",
            r2_full - r2_full_re,
            r2_full,
            r2_full,
            "increment",
        ),
        WaterfallStep("model", "Full model", 0.0, r2_full, r2_full, "total"),
    ]
    return _steps_to_frame(steps)


def _steps_to_frame(steps: list[WaterfallStep]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "section": [s.section for s in steps],
            "label": [s.label for s in steps],
            "increment": [s.increment for s in steps],
            "cumulative": [s.cumulative for s in steps],
            "r2_absolute": [s.r2_absolute for s in steps],
            "kind": [s.kind for s in steps],
        }
    )


def _r2_ylabel(regime: str) -> str:
    if is_in_sample_regime(regime):
        return r"$R^2$ (in-sample)"
    return r"$R^2$ (CV test set)"


def plot_combined_r2_waterfall(
    steps: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
    regime: str = DEFAULT_REGIME,
    show: bool = False,
) -> plt.Figure:
    """
    Combined waterfall: isolated persistence reference + model ΔR² components.
    """
    output_path = Path(output_path)
    context = steps.loc[steps["section"] == "context"].iloc[0]
    model_steps = steps.loc[
        (steps["section"] == "model") & (steps["kind"] == "increment")
    ].copy()
    total_row = steps.loc[steps["kind"] == "total"]
    total_r2 = float(total_row["r2_absolute"].iloc[0])
    r2_persist = float(context["r2_absolute"])

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    # --- Context bar (standalone, from zero) ---
    context_x = 0.0
    ax.bar(
        context_x,
        r2_persist,
        bottom=0.0,
        width=BAR_WIDTH,
        color=COLOR_CONTEXT,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.text(
        context_x,
        r2_persist + 0.015,
        rf"$R^2$ = {r2_persist:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=2,
        ),
    )
    ax.text(
        context_x,
        -0.04,
        context["label"],
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.15,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=2,
        ),
    )

    # --- Visual separator between reference and model components ---
    sep_x = BAR_WIDTH / 2
    ax.axvline(sep_x, color="#999999", linestyle=":", linewidth=1.2, zorder=2)
    ax.text(
        0.1,
        0.92,
        "Observational baseline",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="medium",
        color="#555555",
    )
    ax.text(
        0.78,
        0.92,
        rf"Model components ($\Delta R^2$ from site-mean persistence). Full model $R^2$ = {total_r2:.3f}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="medium",
        color="#555555",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=2,
        ),
    )

    # --- Model component waterfall (starts at persistence level) ---
    n_model = len(model_steps)
    model_x0 = 1.35
    model_x = model_x0 + np.arange(n_model, dtype=float)
    increments = model_steps["increment"].to_numpy(dtype=float)
    labels = model_steps["label"].tolist()

    starts = np.empty(n_model, dtype=float)
    starts[0] = r2_persist
    for i in range(1, n_model):
        starts[i] = starts[i - 1] + increments[i - 1]

    for i, (x, inc, start, label) in enumerate(
        zip(model_x, increments, starts, labels)
    ):
        bar_color = COLOR_POS if inc >= 0 else COLOR_NEG
        ax.bar(
            x,
            inc,
            bottom=start,
            width=BAR_WIDTH,
            color=bar_color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        top = start + inc
        y_text = top + 0.012 if inc >= 0 else start - 0.012
        va = "bottom" if inc >= 0 else "top"
        ax.text(
            x,
            y_text,
            f"{inc:+.3f}",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="medium",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=2,
            ),
        )
        ax.text(
            x,
            -0.04,
            label,
            ha="center",
            va="top",
            fontsize=10,
            linespacing=1.15,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=2,
            ),
        )

    for i in range(n_model - 1):
        y = starts[i + 1]
        ax.plot(
            [model_x[i] + BAR_WIDTH / 2, model_x[i + 1] - BAR_WIDTH / 2],
            [y, y],
            color="#BBBBBB",
            linestyle=":",
            linewidth=0.8,
            zorder=2,
        )

    # Persistence reference line extending into model section
    ax.axhline(
        r2_persist,
        xmin=0.0,
        xmax=0.98,
        color="#888888",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )
    # ax.text(
    #     model_x[-1] + 0.55,
    #     r2_persist - 0.008,
    #     f"persistence\nR² = {r2_persist:.3f}",
    #     ha="left",
    #     va="top",
    #     fontsize=8.5,
    #     color="#666666",
    # )

    # Full model target
    ax.axhline(total_r2, color="#444444", linestyle="-", linewidth=1.0, zorder=2)
    # ax.text(
    #     model_x[-1] + 0.55,
    #     total_r2 + 0.008,
    #     rf"Full model $R^2$ = {total_r2:.3f}",
    #     ha="left",
    #     va="bottom",
    #     fontsize=10,
    #     color="#333333",
    #     fontweight="medium",
    # )

    # Bracket showing net model gain over persistence
    net_gain = total_r2 - r2_persist
    bracket_x = model_x[-1] + BAR_WIDTH / 1.9
    ax.annotate(
        "",
        xy=(bracket_x, total_r2),
        xytext=(bracket_x, r2_persist),
        arrowprops=dict(
            arrowstyle="->",
            color="#666666",
            lw=2.0,
        ),
    )
    ax.text(
        bracket_x + 0.15,
        (total_r2 + r2_persist) / 2,
        f"+{net_gain:.3f}",
        ha="left",
        va="center",
        fontsize=9,
        color="#444444",
    )

    ax.set_ylabel(_r2_ylabel(regime))
    ax.set_title(title, fontsize=13, pad=28)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=10)

    legend_handles = [
        Patch(
            facecolor=COLOR_CONTEXT,
            edgecolor="white",
            label="Observational baseline",
        ),
        Patch(
            facecolor=COLOR_POS,
            edgecolor="white",
            label=r"Improves $R^2$: $\Delta > 0$",
        ),
        Patch(
            facecolor=COLOR_NEG,
            edgecolor="white",
            label=r"$Depreciates $R^2$: $\Delta < 0$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.1),
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=9,
    )

    y_vals = np.concatenate([[0.0, r2_persist, total_r2], starts + increments])
    y_min = min(0.0, float(np.min(y_vals))) - 0.06
    y_max = max(total_r2, float(np.max(y_vals))) + 0.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.55, model_x[-1] + BAR_WIDTH / 1.5)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=HBB_FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_r2_waterfall(
    beta_glmm_root: Path,
    *,
    fold_tag: str | None = None,
    regime: str = DEFAULT_REGIME,
    primary_variant: str = DEFAULT_PRIMARY_VARIANT,
    eco_variant: str = DEFAULT_ECO_VARIANT,
    output_dir: Path | None = None,
    run_if_missing: bool = True,
    force_rerun: bool = False,
    cv_output_root: Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Ensure inputs, build combined steps, and write figure + CSV."""
    beta_glmm_root = Path(beta_glmm_root)
    fold_tag = _resolve_fold_tag(
        beta_glmm_root,
        fold_tag=fold_tag,
        primary_variant=primary_variant,
        regime=regime,
    )

    ensure_waterfall_inputs(
        beta_glmm_root,
        fold_tag=fold_tag,
        regime=regime,
        primary_variant=primary_variant,
        eco_variant=eco_variant,
        run_if_missing=run_if_missing,
        force_rerun=force_rerun,
        cv_output_root=cv_output_root,
    )

    table_path = decomposition_table_path(beta_glmm_root)
    table = pd.read_csv(table_path) if table_path.exists() else None
    split_source = "predictions"
    try:
        merged = merge_variant_predictions(
            beta_glmm_root,
            fold_tag=fold_tag,
            variants={"full": primary_variant, "eco": eco_variant},
        )
        steps = build_combined_waterfall_steps(merged)
    except FileNotFoundError:
        if run_if_missing or table is None:
            raise
        split_source = "table"
        steps = build_combined_waterfall_from_table(
            table,
            fold_tag=fold_tag,
            primary_variant=primary_variant,
            eco_variant=eco_variant,
        )

    out_dir = Path(output_dir or (beta_glmm_root / "r2_waterfall" / fold_tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_label = BETA_VARIANT_LABELS.get(primary_variant, primary_variant)
    subtitle = f"{regime} · {fold_tag} · {variant_label} · source={split_source}"
    csv_path = out_dir / f"r2_waterfall_{primary_variant}.csv"
    png_path = out_dir / f"r2_waterfall_{primary_variant}.png"

    steps.to_csv(csv_path, index=False)
    plot_combined_r2_waterfall(
        steps,
        png_path,
        title=r"$\Delta R^2$ decomposition: persistence baseline and model components",
        subtitle=subtitle,
        regime=regime,
        show=show,
    )

    meta = {
        "fold_tag": fold_tag,
        "regime": regime,
        "primary_variant": primary_variant,
        "eco_variant": eco_variant,
        "split_source": split_source,
        "outputs": {"csv": str(csv_path), "png": str(png_path)},
    }
    (out_dir / "r2_waterfall_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return {
        "fold_tag": fold_tag,
        "regime": regime,
        "split_source": split_source,
        "steps": steps,
        "outputs": meta["outputs"],
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=config.sully_og_dir / "output" / "cross_validation" / "beta_glmm",
        help="beta_glmm CV output root",
    )
    parser.add_argument(
        "--cv-output-root",
        type=Path,
        default=None,
        help="Cross-validation output root when running models (default: parent of --cv-root)",
    )
    parser.add_argument("--fold-tag", type=str, default=None)
    parser.add_argument("--regime", type=str, default=DEFAULT_REGIME)
    parser.add_argument("--variant", type=str, default=DEFAULT_PRIMARY_VARIANT)
    parser.add_argument(
        "--eco-variant",
        type=str,
        default=DEFAULT_ECO_VARIANT,
        help="Variant used for ecoregion-only RE predictions",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run beta-GLMM CV + decomposition if outputs are missing (default: on)",
    )
    parser.add_argument(
        "--no-run-models",
        action="store_true",
        help="Do not run CV/decomposition; fail if outputs are missing",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run CV and decomposition even when outputs already exist",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_if_missing = not args.no_run_models
    if args.run_models:
        run_if_missing = True

    results = run_r2_waterfall(
        args.cv_root,
        fold_tag=args.fold_tag,
        regime=args.regime,
        primary_variant=args.variant,
        eco_variant=args.eco_variant,
        output_dir=args.output_dir,
        run_if_missing=run_if_missing,
        force_rerun=args.force_rerun,
        cv_output_root=args.cv_output_root,
        show=args.show,
    )

    print(f"Fold: {results['fold_tag']}")
    print(f"Source: {results['split_source']}")
    print("\nCombined waterfall steps:")
    print(results["steps"].to_string(index=False))
    print("\nOutputs:")
    for key, path in results["outputs"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
