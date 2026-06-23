#!/usr/bin/env python3
"""Python investigation of paper vs reparameterized hierarchical beta models.

This mirrors ``src/native_r/run_beta_model_investigation.R`` using the existing
PyMC implementation.  It uses the same paper-style data preparation for all
variants, then isolates:

1. paper factor encoding of site -> ecoregion,
2. corrected region indexing,
3. fixed-intercept removal,
4. centered vs non-centered hierarchy.

Shared implementation lives in :mod:`src.models.hbb.variants`,
:mod:`src.models.hbb.investigation_data`, and
:mod:`src.models.hbb.investigation_fit`.
"""

from __future__ import annotations

# Re-export public API used by notebooks and scripts (backward compatibility).
from src.models.hbb.investigation_data import (  # noqa: F401
    CoverSimConfig,
    add_latitude_features,
    replace_diversity_from_ecoregions,
    save_cover_simulation_diagnostics,
    simulate_cover,
    standardization_vars,
)
from src.models.hbb.investigation_fit import (  # noqa: F401
    coefficient_summary,
    ecoregion_predictor_contributions,
    enrich_coefficients_delta_cover,
    fit_variant,
    load_trace_nc,
    logit_beta_to_delta_cover,
    plot_coefficients,
    plot_comparison,
    run_latitude_cap_sensitivity,
    save_trace_nc,
    save_variant_outputs,
)
from src.models.hbb.variant_data import build_variant_data, r_lexicographic_factor_codes  # noqa: F401
from src.models.hbb.variants import (  # noqa: F401
    COEF_LABELS,
    COEF_LABEL_BY_COLUMN,
    DIVERSITY_ECOREGION_ALIASES,
    FULL_INVESTIGATION_VARIANTS,
    KEY_OTHER_PARAMS,
    VARIANTS,
    Variant,
    apply_variant_options,
    coefficient_labels,
    parse_excluded_vars,
    parse_optional_bool,
    parse_variants,
    unique_variants,
)

__all__ = [
    "COEF_LABELS",
    "COEF_LABEL_BY_COLUMN",
    "CoverSimConfig",
    "DIVERSITY_ECOREGION_ALIASES",
    "FULL_INVESTIGATION_VARIANTS",
    "KEY_OTHER_PARAMS",
    "VARIANTS",
    "Variant",
    "add_latitude_features",
    "apply_variant_options",
    "build_variant_data",
    "coefficient_labels",
    "coefficient_summary",
    "ecoregion_predictor_contributions",
    "enrich_coefficients_delta_cover",
    "fit_variant",
    "load_trace_nc",
    "logit_beta_to_delta_cover",
    "parse_excluded_vars",
    "parse_optional_bool",
    "parse_variants",
    "plot_coefficients",
    "plot_comparison",
    "r_lexicographic_factor_codes",
    "replace_diversity_from_ecoregions",
    "run_latitude_cap_sensitivity",
    "save_cover_simulation_diagnostics",
    "save_trace_nc",
    "save_variant_outputs",
    "simulate_cover",
    "standardization_vars",
    "unique_variants",
]


def main() -> None:
    import argparse
    import json
    import os
    import time
    from dataclasses import asdict
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    from src import config
    from src.models.hbb.data import load_model_data_from_pipeline, standardize_variables
    from src.models.hbb.design import compute_correlation_matrix
    from src.models.hbb.mcmc_config import (
        INVESTIGATION_MCMC_DEFAULTS,
        apply_investigation_smoke,
    )
    from src.models.hbb.model import resolve_pymc_ncores
    from src.plots.hb_beta_plots import plot_correlation_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.sully_og_dir)
    parser.add_argument(
        "--output-root", type=Path, default=config.sully_og_dir / "output_python"
    )
    parser.add_argument(
        "--variants",
        default=os.getenv("PY_INV_VARIANTS", ""),
        help=(
            "Comma-separated variants to run. Use 'all' or 'full' for the full "
            "paper-vs-reparam investigation. If omitted, defaults to --base-model."
        ),
    )
    parser.add_argument(
        "--base-model",
        choices=list(VARIANTS),
        default=os.getenv("PY_INV_BASE_MODEL", "reparam"),
        help=(
            "Base variant for focused predictor investigations when --variants is "
            "omitted. Defaults to reparam."
        ),
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=int(os.getenv("PY_INV_DRAWS", str(INVESTIGATION_MCMC_DEFAULTS.n_samples))),
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=int(os.getenv("PY_INV_TUNE", str(INVESTIGATION_MCMC_DEFAULTS.n_tune))),
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=int(os.getenv("PY_INV_CHAINS", str(INVESTIGATION_MCMC_DEFAULTS.n_chains))),
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=int(os.getenv("PY_INV_NCORES", str(INVESTIGATION_MCMC_DEFAULTS.ncores or 6))),
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=INVESTIGATION_MCMC_DEFAULTS.target_accept,
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=INVESTIGATION_MCMC_DEFAULTS.max_treedepth,
    )
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--smoke", action="store_true", default=os.getenv("PY_INV_SMOKE") == "1"
    )
    parser.add_argument("--no-pymc-progress", action="store_true")
    parser.add_argument("--no-force-rebuild", action="store_true")
    parser.add_argument(
        "--exclude-vars",
        default=os.getenv("PY_INV_EXCLUDE_VARS", ""),
        help=(
            "Comma-separated base variables to exclude from every selected variant "
            "(e.g. 'lat' or 'lat,sst_mean'). Names may be raw or *_stzd."
        ),
    )
    parser.add_argument(
        "--latitude-transform",
        choices=["abs", "trig"],
        default=os.getenv("PY_INV_LATITUDE_TRANSFORM", ""),
        help=(
            "Override latitude encoding for selected variants. 'abs' uses absolute "
            "latitude; 'trig' replaces lat with standardized sin(latitude) and "
            "cos(latitude) based on signed latitude degrees."
        ),
    )
    parser.add_argument(
        "--site-hierarchy",
        choices=["true", "false"],
        default=os.getenv("PY_INV_SITE_HIERARCHY", ""),
        help="Override site random effects for selected variants.",
    )
    parser.add_argument(
        "--ecoregion-hierarchy",
        choices=["true", "false"],
        default=os.getenv("PY_INV_ECOREGION_HIERARCHY", ""),
        help="Override ecoregion random effects for selected variants.",
    )
    parser.add_argument(
        "--modified-only",
        action="store_true",
        help=(
            "For focused predictor investigations, run only the modified base "
            "model instead of base + modified comparison."
        ),
    )
    parser.add_argument(
        "--diversity-source",
        choices=["data_for_maps", "ecoregions_csv"],
        default=os.getenv("PY_INV_DIVERSITY_SOURCE", "data_for_maps"),
        help=(
            "Use existing data_for_maps.csv diversity.standardized values, or "
            "replace them with standardized total_species_number from ecoregions.csv."
        ),
    )
    parser.add_argument(
        "--ecoregions-diversity-path",
        type=Path,
        default=config.data_dir / "ecoregion_diversity" / "ecoregions.csv",
        help="Path to ecoregions.csv used when --diversity-source ecoregions_csv.",
    )
    parser.add_argument(
        "--simulate-cover",
        choices=["observed", "cosine_latitude"],
        default=os.getenv("PY_INV_SIMULATE_COVER", "observed"),
        help=(
            "Replace observed coral cover with a simulated response. "
            "'cosine_latitude' draws cover from a beta distribution whose logit-mean "
            "follows intercept + strength * standardized cos(signed latitude), plus "
            "modest site and observation noise (NUTS-friendly while preserving a strong "
            "lat gradient within the observed reef latitudes)."
        ),
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=None,
        help="RNG seed for cover simulation (defaults to --seed).",
    )
    parser.add_argument(
        "--sim-intercept",
        type=float,
        default=float(os.getenv("PY_INV_SIM_INTERCEPT", "0.0")),
        help="Intercept on the logit scale for cosine-latitude simulation.",
    )
    parser.add_argument(
        "--sim-lat-strength",
        type=float,
        default=float(os.getenv("PY_INV_SIM_LAT_STRENGTH", "2.0")),
        help="Logit-scale coefficient on standardized cos(latitude) for cover simulation.",
    )
    parser.add_argument(
        "--sim-precision",
        type=float,
        default=float(os.getenv("PY_INV_SIM_PRECISION", "50.0")),
        help="Beta precision for simulated cover draws (higher = less observation noise).",
    )
    parser.add_argument(
        "--sim-site-sd",
        type=float,
        default=float(os.getenv("PY_INV_SIM_SITE_SD", "0.15")),
        help="Site-level logit SD in the cover simulation DGP.",
    )
    parser.add_argument(
        "--sim-obs-sd",
        type=float,
        default=float(os.getenv("PY_INV_SIM_OBS_SD", "0.15")),
        help="Observation-level logit SD in the cover simulation DGP.",
    )
    args = parser.parse_args()

    mcmc = INVESTIGATION_MCMC_DEFAULTS
    if args.smoke:
        mcmc = apply_investigation_smoke(mcmc)
        args.draws = mcmc.n_samples
        args.tune = mcmc.n_tune
        args.chains = mcmc.n_chains
        args.ncores = mcmc.ncores or args.chains

    exclude_vars = parse_excluded_vars(args.exclude_vars)
    latitude_transform = args.latitude_transform or None
    has_focused_options = bool(
        exclude_vars
        or latitude_transform is not None
        or parse_optional_bool(args.site_hierarchy or None) is not None
        or parse_optional_bool(args.ecoregion_hierarchy or None) is not None
    )
    if args.variants:
        variants = apply_variant_options(
            parse_variants(args.variants, default=[VARIANTS[args.base_model]]),
            exclude_vars=exclude_vars,
            latitude_transform=latitude_transform,
            use_site_hierarchy=parse_optional_bool(args.site_hierarchy or None),
            use_ecoregion_hierarchy=parse_optional_bool(args.ecoregion_hierarchy or None),
        )
    elif has_focused_options:
        base_variant = VARIANTS[args.base_model]
        modified_variants = apply_variant_options(
            [base_variant],
            exclude_vars=exclude_vars,
            latitude_transform=latitude_transform,
            use_site_hierarchy=parse_optional_bool(args.site_hierarchy or None),
            use_ecoregion_hierarchy=parse_optional_bool(args.ecoregion_hierarchy or None),
        )
        variants = modified_variants if args.modified_only else [base_variant, *modified_variants]
    else:
        variants = [VARIANTS[args.base_model]]
    variants = unique_variants(variants)
    inv_dir = args.output_root / "investigation"
    inv_dir.mkdir(parents=True, exist_ok=True)
    console = Console()

    console.rule("[bold]Python beta-model investigation")
    console.print(f"Data dir: {args.data_dir}")
    console.print(f"Output:   {inv_dir}")
    console.print(f"Variants: {', '.join(v.name for v in variants)}")
    console.print(f"Diversity source: {args.diversity_source}")
    console.print(f"Cover response:   {args.simulate_cover}")
    if args.simulate_cover == "cosine_latitude":
        console.print(
            "Cover sim DGP:    logit_mean = "
            f"{args.sim_intercept:g} + {args.sim_lat_strength:g} * z(cos(lat)) "
            f"+ site N(0, {args.sim_site_sd:g}) + obs N(0, {args.sim_obs_sd:g}); "
            f"Beta precision={args.sim_precision:g}"
        )
    console.print(
        f"MCMC:     chains={args.chains}, tune={args.tune}, draws={args.draws}, "
        f"cores={args.ncores}"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    beta_frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []

    with progress:
        task = progress.add_task("Loading and standardizing paper-style data", total=1)
        df = load_model_data_from_pipeline(
            args.data_dir,
            force_rebuild=not args.no_force_rebuild,
            index_source="paper_factor",
        )
        corr_df = df.copy()
        if "lat" in corr_df.columns:
            corr_df["lat"] = corr_df["lat"].abs()
        corr_matrix = compute_correlation_matrix(corr_df)
        corr_matrix.to_csv(inv_dir / "corrplot.csv")
        plot_correlation_matrix(corr_matrix, inv_dir / "corrplot.png")
        df = add_latitude_features(df)
        if args.diversity_source == "ecoregions_csv":
            df, diversity_mapping = replace_diversity_from_ecoregions(
                df,
                path=args.ecoregions_diversity_path,
            )
            diversity_mapping.to_csv(
                inv_dir / "ecoregions_diversity_mapping.csv", index=False
            )
        sim_config = CoverSimConfig(
            intercept=args.sim_intercept,
            lat_strength=args.sim_lat_strength,
            precision=args.sim_precision,
            site_logit_sd=args.sim_site_sd,
            obs_logit_sd=args.sim_obs_sd,
            seed=args.sim_seed if args.sim_seed is not None else args.seed,
        )
        df, sim_metadata = simulate_cover(
            df, mode=args.simulate_cover, config=sim_config
        )
        if sim_metadata is not None:
            sim_metadata = save_cover_simulation_diagnostics(
                df,
                inv_dir,
                config=sim_config,
                metadata=sim_metadata,
            )
        df_std, std_stats = standardize_variables(df, standardization_vars())
        pd.DataFrame(
            [
                {"variable": name, "mean": vals[0], "sd": vals[1]}
                for name, vals in std_stats.items()
            ]
        ).to_csv(inv_dir / "standardization_stats.csv", index=False)
        progress.update(task, advance=1)

        main_task = progress.add_task("Fitting variants", total=len(variants))
        for i, variant in enumerate(variants, start=1):
            progress.update(
                main_task, description=f"Fitting {variant.name} ({i}/{len(variants)})"
            )
            out_dir = inv_dir / variant.subdir
            try:
                beta_df, conv, input_summary = fit_variant(
                    variant=variant,
                    df_std=df_std,
                    output_dir=out_dir,
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    ncores=resolve_pymc_ncores(
                        ncores=args.ncores, n_chains=args.chains
                    ),
                    target_accept=args.target_accept,
                    max_treedepth=args.max_treedepth,
                    seed=args.seed + i,
                    progressbar=not args.no_pymc_progress,
                )
                beta_frames.append(beta_df)
                input_rows.append(input_summary)
                rhat = conv["r_hat"] if "r_hat" in conv else pd.Series(dtype=float)
                neff = conv["n.eff"] if "n.eff" in conv else pd.Series(dtype=float)
                conv_rows.append(
                    {
                        "variant": variant.name,
                        "status": "ok",
                        "max_rhat": rhat.max(skipna=True) if not rhat.empty else np.nan,
                        "n_rhat_gt_1.05": int((rhat > 1.05).sum())
                        if not rhat.empty
                        else np.nan,
                        "n_rhat_gt_1.10": int((rhat > 1.10).sum())
                        if not rhat.empty
                        else np.nan,
                        "min_neff": neff.min(skipna=True) if not neff.empty else np.nan,
                        "median_neff": neff.median(skipna=True)
                        if not neff.empty
                        else np.nan,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]ERROR fitting {variant.name}: {exc}[/red]")
                conv_rows.append(
                    {
                        "variant": variant.name,
                        "status": "error",
                        "max_rhat": np.nan,
                        "n_rhat_gt_1.05": np.nan,
                        "n_rhat_gt_1.10": np.nan,
                        "min_neff": np.nan,
                        "median_neff": np.nan,
                        "error": str(exc),
                    }
                )
            progress.update(main_task, advance=1)

    if input_rows:
        pd.DataFrame(input_rows).to_csv(
            inv_dir / "model_input_summary.csv", index=False
        )
    if conv_rows:
        conv_df = pd.DataFrame(conv_rows)
        conv_df.to_csv(inv_dir / "convergence_comparison.csv", index=False)
    if beta_frames:
        combined = pd.concat(beta_frames, ignore_index=True)
        combined.to_csv(inv_dir / "beta_coeff_comparison.csv", index=False)
        if "paper_reproduction" in set(combined["variant"]):
            ref_name = "paper_reproduction"
        else:
            ref_name = combined["variant"].iloc[0]
        ref = combined.loc[
            combined["variant"] == ref_name, ["variable", "mean"]
        ].rename(columns={"mean": "paper_mean"})
        shifts = combined.merge(ref, on="variable", how="left")
        shifts["mean_shift_vs_paper"] = shifts["mean"] - shifts["paper_mean"]
        shifts["abs_shift_vs_paper"] = shifts["mean_shift_vs_paper"].abs()
        shifts.to_csv(inv_dir / "coefficient_shift_vs_paper.csv", index=False)
        plot_comparison(combined, inv_dir / "beta_coeff_comparison.png")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(args.data_dir),
        "output_dir": str(inv_dir),
        "variants": [asdict(v) for v in variants],
        "mcmc": {
            "chains": args.chains,
            "tune": args.tune,
            "draws": args.draws,
            "ncores": args.ncores,
            "target_accept": args.target_accept,
            "max_treedepth": args.max_treedepth,
            "smoke": args.smoke,
        },
        "predictor_options": {
            "base_model": args.base_model,
            "exclude_vars": exclude_vars,
            "latitude_transform_override": latitude_transform,
            "modified_only": args.modified_only,
        },
        "data_options": {
            "diversity_source": args.diversity_source,
            "ecoregions_diversity_path": str(args.ecoregions_diversity_path),
            "simulate_cover": args.simulate_cover,
            "cover_simulation": (
                None
                if args.simulate_cover == "observed"
                else {
                    "intercept": args.sim_intercept,
                    "lat_strength": args.sim_lat_strength,
                    "precision": args.sim_precision,
                    "site_logit_sd": args.sim_site_sd,
                    "obs_logit_sd": args.sim_obs_sd,
                    "seed": args.sim_seed if args.sim_seed is not None else args.seed,
                }
            ),
        },
        "application_outputs_per_variant": [
            "observed_vs_expected_coral_cover.png",
            "bright_dark_spots_map.png",
            "current_coral_cover_bright_and_dark_spots_a.png",
            "bright_spots_list.csv",
            "dark_spots_list.csv",
            "prediction_statistics.csv",
            "data_processed.csv",
            "coral_cover_by_ocean.csv",
            "residual_diagnostics/",
        ],
        "shared_application_outputs": ["corrplot.png", "corrplot.csv"],
    }
    (inv_dir / "investigation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    table = Table(title="Investigation complete")
    table.add_column("Variant")
    table.add_column("Status")
    table.add_column("max R-hat")
    table.add_column("median ESS")
    for row in conv_rows:
        table.add_row(
            row["variant"],
            row["status"],
            "" if pd.isna(row["max_rhat"]) else f"{row['max_rhat']:.3f}",
            "" if pd.isna(row["median_neff"]) else f"{row['median_neff']:.0f}",
        )
    console.print(table)
    console.print(f"[green]Outputs written to {inv_dir}[/green]")


if __name__ == "__main__":
    main()
