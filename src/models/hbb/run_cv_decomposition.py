#!/usr/bin/env python3
"""Post-hoc CV hierarchy decomposition and coefficient forest regeneration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.models.hbb.cv_decomposition import run_decomposition_for_beta_glmm_root
from src.models.hbb.variant_data import infer_design_col_names
from src.models.hbb.variants import VARIANTS
from src.plots.hb_beta_plots import (
    plot_coefficient_forest_df,
    plot_posterior_coefficient_forest,
)

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


def _col_names_for_cv_fold(fold_dir: Path, idata) -> list[str] | None:
    """Load or infer design-matrix column names for a CV fold trace."""
    stats_path = fold_dir / "fit_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        saved = stats.get("col_names")
        if saved:
            return list(saved)

    variant_name = "reparam"
    if stats_path.exists():
        variant_name = stats.get("metrics", {}).get("variant", variant_name)
    variant = VARIANTS.get(variant_name)
    if variant is None:
        return None

    n_beta = int(
        idata.posterior["beta"]
        .sizes[[d for d in idata.posterior["beta"].dims if d not in ("chain", "draw")][0]]
    )
    try:
        return infer_design_col_names(variant, n_beta)
    except ValueError:
        return None


def regenerate_investigation_coeff_forests(investigation_root: Path) -> int:
    """Rebuild diagnostics/coeff_forest.png from beta_est.csv using standard styling."""
    investigation_root = Path(investigation_root)
    count = 0
    for beta_path in sorted(investigation_root.rglob("beta_est.csv")):
        out = beta_path.parent / "diagnostics" / "coeff_forest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        beta_df = pd.read_csv(beta_path)
        title = f"Beta coefficients: {beta_path.parent.name}"
        plot_coefficient_forest_df(
            beta_df,
            out,
            title=title,
            label_col="variable",
        )
        count += 1
    return count


def regenerate_cv_coeff_forests(beta_glmm_root: Path) -> int:
    """Rebuild coefficient_diagnostics/coeff_forest.png from trace.nc."""
    if not HAS_ARVIZ:
        return 0
    beta_glmm_root = Path(beta_glmm_root)
    count = 0
    for trace_path in sorted(beta_glmm_root.rglob("trace.nc")):
        fold_dir = trace_path.parent
        diag_dir = fold_dir / "coefficient_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        idata = az.from_netcdf(trace_path)
        col_names = _col_names_for_cv_fold(fold_dir, idata)
        plot_posterior_coefficient_forest(
            idata,
            col_names,
            diag_dir / "coeff_forest.png",
        )
        plot_posterior_coefficient_forest(
            idata,
            col_names,
            diag_dir / "coefficient_posterior_forest.png",
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=config.sully_og_dir / "output" / "cross_validation" / "beta_glmm",
        help="Root beta_glmm CV output directory",
    )
    parser.add_argument(
        "--investigation-root",
        type=Path,
        default=config.sully_og_dir / "output_python" / "investigation",
        help="Investigation output root for coeff_forest regeneration",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Only regenerate forest plots",
    )
    parser.add_argument(
        "--skip-forests",
        action="store_true",
        help="Only run hierarchy decomposition",
    )
    args = parser.parse_args()

    if not args.skip_decomposition:
        combined = run_decomposition_for_beta_glmm_root(args.cv_root)
        if combined.empty:
            print(f"No decomposition results written under {args.cv_root}")
        else:
            out = args.cv_root / "hierarchy_decomposition_all_variants.csv"
            print(f"Wrote {len(combined)} decomposition row(s) → {out}")
            print(combined.to_string(index=False))

    if not args.skip_forests:
        n_inv = regenerate_investigation_coeff_forests(args.investigation_root)
        n_cv = regenerate_cv_coeff_forests(args.cv_root)
        print(f"Regenerated {n_inv} investigation coeff_forest.png file(s)")
        print(f"Regenerated {n_cv} CV coeff_forest.png file(s)")


if __name__ == "__main__":
    main()
