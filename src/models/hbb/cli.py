"""CLI entry point for hierarchical beta model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.hbb.analysis import run_full_analysis


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Run coral cover beta regression analysis"
    )
    parser.add_argument(
        "--data", "-d", type=str, default=None, help="Path to data.csv file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--save-diagnostics",
        "-sd",
        action="store_true",
        default=False,
        help="Save model diagnostics",
    )
    parser.add_argument(
        "--max-treedepth",
        "-mt",
        type=int,
        default=8,
        help="Maximum tree depth for NUTS sampler",
    )
    parser.add_argument(
        "--target-accept",
        "-ta",
        type=float,
        default=0.9,
        help="Target acceptance rate for NUTS sampler",
    )
    parser.add_argument(
        "--num-chains",
        "-nc",
        type=int,
        default=4,
        help="Number of chains for MCMC sampling",
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=2000, help="Number of posterior samples"
    )
    parser.add_argument(
        "--tune", "-t", type=int, default=1000, help="Number of tuning samples"
    )
    parser.add_argument(
        "--random-seed",
        "-rs",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--project-all-scenarios",
        action="store_true",
        help="After fitting, run future projections for default RCP scenarios/years "
        "(rcp45/rcp85 x 2050/2100).",
    )
    parser.add_argument(
        "--project-scenarios",
        nargs="+",
        default=None,
        help=(
            "Custom scenarios to project, as tokens like 'rcp45_2050 rcp85_2100'. "
            "Implies --project-all-scenarios."
        ),
    )
    parser.add_argument(
        "--projection-samples",
        type=int,
        default=1000,
        help="Number of posterior predictive samples per scenario/year for projections.",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=4,
        help="Number of cores to use for sampling",
    )
    parser.add_argument(
        "--legacy-r",
        action="store_true",
        help="Use centered JAGS spec from my_1_run_the_beta_model.Rmd (intercept in X).",
    )
    parser.add_argument(
        "--add-intercept",
        action="store_true",
        help="Add intercept to X (default for --legacy-r).",
    )

    args = parser.parse_args()
    data_path = Path(args.data) if args.data else None
    output_dir = Path(args.output) if args.output else None

    project_scenarios = None
    if args.project_all_scenarios or args.project_scenarios:
        if args.project_scenarios:
            parsed = []
            for token in args.project_scenarios:
                parts = token.lower().split("_")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid scenario token '{token}'. Expected format like 'rcp45_2050'."
                    )
                scen, year_str = parts
                parsed.append((scen, int(year_str)))
            project_scenarios = parsed
        else:
            project_scenarios = [
                ("rcp45", 2050),
                ("rcp45", 2100),
                ("rcp85", 2050),
                ("rcp85", 2100),
            ]

    results = run_full_analysis(
        data_path=data_path,
        output_dir=output_dir,
        save_diagnostics=args.save_diagnostics,
        n_samples=args.samples,
        n_tune=args.tune,
        n_chains=args.num_chains,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        random_seed=args.random_seed,
        project_scenarios=project_scenarios,
        n_prediction_samples=args.projection_samples,
        model_spec="legacy_r" if args.legacy_r else "reparam",
        add_intercept=args.add_intercept or args.legacy_r,
    )

    print("\nSummary:")
    print(f"  Observations: {results['n_observations']}")
    if "coefficient_summary" in results:
        print("\nCoefficient estimates:")
        print(results["coefficient_summary"][["mean", "hdi_5%", "hdi_95%"]])
    return results


if __name__ == "__main__":
    main()
