#!/usr/bin/env python3
"""
Coral Reef Economics Analysis Pipeline

This script runs the complete analysis pipeline:
1. Loads coral cover projections and economic value data
2. Aligns datasets spatially
3. Calculates projected economic losses under different scenarios
4. Generates summary statistics and figures

Usage:
    python run_analysis.py [--scenarios RCP45_2100 RCP85_2100] [--models linear compound]

Configuration:
    Edit the CONFIG dictionary below to customize the analysis.

Output:
    - figures/ : Static and interactive visualizations
    - results/ : CSV files with aggregated results
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from src.economics.cumulative_impact import CumulativeImpactResult

from src import config, utils
from src.economics import (
    get_model,
    list_models,
)
from src.economics.analysis import (
    AnalysisResults,
    calculate_depreciation,
    print_summary,
    validate_alignment,
    validate_depreciation_formula,
)
from src.economics.cumulative_impact import (
    calculate_cumulative_impacts_multi_scenario,
    print_cumulative_summary,
    summarize_cumulative_impacts,
)
from src.economics.data_loader import (
    align_coral_to_economic_data,
    compute_country_aggregations,
    load_coral_cover_data,
    load_gdp_data,
    load_tourism_data,
)
from src.economics.plotting import (
    generate_figure_set,
    generate_verification_plots,
    plot_annual_loss_trajectories,
    plot_annual_value_trajectories,
    plot_coral_cover_trajectories,
    plot_cumulative_loss_trajectories,
    plot_gdp_impact_scenario_comparison,
    plot_loss_as_gdp_pct_bar,
    plot_loss_as_gdp_pct_choropleth,
    plot_model_comparison,
    plot_model_comparison_interactive,
    plot_scenario_comparison,
    plot_trajectory_comparison_interactive,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Scenarios to analyze
    "scenarios": [
        "RCP45_yr_2050",
        "RCP45_yr_2100",
        "RCP85_yr_2050",
        "RCP85_yr_2100",
    ],
    # Depreciation models to compare
    "models": {
        "linear": {"rate_per_percent": 0.0381},  # Chen et al. default
        "compound": {"rate_per_percent": 0.0381},
        "tipping_point": {"threshold_cc": 0.10, "post_threshold_loss": 0.80},
    },
    # Value types to analyze
    "value_types": ["tourism"],  # Add "shoreline_protection" and fisheries when ready
    # Spatial alignment
    "max_distance_deg": 5.0,  # Maximum distance for coral-economic matching
    # Output
    "output_dir": config.figures_dir / "economics_analysis",
    "results_dir": config.repo_dir / "results",
    "save_formats": ["png", "html"],
    "top_n_countries": 20,
    "export_web_data": True,
}

# =============================================================================
# LOAD PREVIOUS RESULTS
# =============================================================================


def load_previous_results(run_name: str, verbose: bool = True):
    """
    Load previous results from a saved run.

    Parameters
    ----------
    run_name : str
        Name of the run directory (e.g., "run_20260119_105256")
    verbose : bool
        Print loading progress

    Returns
    -------
    dict
        Dictionary with 'results' and 'cumulative_results' keys
    """
    if verbose:
        print("\n" + "=" * 60)
        print("LOADING PREVIOUS RESULTS")
        print("=" * 60)

    results_dir = Path(CONFIG["results_dir"])
    run_dir = results_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} not found")

    if verbose:
        print(f"  Loading from: {run_dir}")

    results = AnalysisResults.load(run_dir / "results")
    cumulative_results = load_cumulative_results(run_dir, verbose=verbose)

    if verbose:
        print(f"\n  ✓ Loaded {len(results.results)} analysis results")
        print(f"  ✓ Loaded {len(cumulative_results)} cumulative impact results")

    return {
        "results": results,
        "cumulative_results": cumulative_results,
    }


def load_cumulative_results(
    run_dir: Path, verbose: bool = True
) -> Dict[str, "CumulativeImpactResult"]:
    """Load cumulative results from saved files."""
    from src.economics.cumulative_impact import CumulativeImpactResult

    cumulative_results_dir = run_dir / "cumulative_results"
    if not cumulative_results_dir.exists():
        if verbose:
            print(
                f"  ⚠️  Cumulative results directory not found: {cumulative_results_dir}"
            )
        return {}

    results = {}

    # Load all .pkl files in the directory
    for pkl_file in cumulative_results_dir.glob("*.pkl"):
        try:
            result = CumulativeImpactResult.load(pkl_file)
            # Use filename (without extension) as key
            key = pkl_file.stem
            results[key] = result
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to load {pkl_file.name}: {e}")

    if verbose:
        print(f"  ✓ Loaded {len(results)} cumulative impact results")

    return results


# =============================================================================
# PIPELINE STEPS
# =============================================================================


def step_load_data(verbose: bool = True):
    """Step 1: Load all datasets."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

    # Coral cover projections
    coral_data = load_coral_cover_data(verbose=verbose)  # checked

    # Economic value data
    tourism_data = load_tourism_data(
        apply_correction=True, validate=verbose
    )  # in progress, not matching with implementation

    # GDP data for computing tourism as % of GDP
    try:
        gdp_data = load_gdp_data()
    except FileNotFoundError:
        warnings.warn("GDP data not found, skipping GDP percentage calculations")
        gdp_data = None

    # Compute country aggregations for verification plots
    by_country = compute_country_aggregations(
        tourism_data.gdf,
        value_column=tourism_data.value_column,
        gdp_data=gdp_data,
        verbose=verbose,
    )

    # TODO: Add shoreline protection when value mapping is available
    # protection_data = load_shoreline_protection_data()

    return {
        "coral": coral_data,
        "tourism": tourism_data,
        "gdp": gdp_data,
        "by_country": by_country,
        # "protection": protection_data,
    }


def step_align_data(data: dict, verbose: bool = True):
    """Step 2: Spatially align coral cover to economic data."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: SPATIAL ALIGNMENT")
        print("=" * 60)

    aligned = {}

    # Align coral cover to tourism
    if "tourism" in data:
        aligned["tourism"] = align_coral_to_economic_data(
            coral_data=data["coral"],
            economic_data=data["tourism"],
            max_distance_deg=CONFIG["max_distance_deg"],
            verbose=verbose,
        )

        # Validate alignment
        validation = validate_alignment(aligned["tourism"], CONFIG["max_distance_deg"])
        if verbose:
            print(
                f"\n  Tourism alignment: {validation['pct_within_threshold']:.1f}% within threshold"
            )
            print(
                f"  Distance stats: mean={validation['distance_stats']['mean']:.2f}°, max={validation['distance_stats']['max']:.2f}°"
            )

    # TODO: Align coral cover to protection data
    # if "protection" in data:
    #     aligned["protection"] = align_coral_to_economic_data(...)

    return aligned


def step_validate_models(verbose: bool = True):
    """Step 3: Validate depreciation model formulas."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: MODEL VALIDATION")
        print("=" * 60)

    models = []
    for name, kwargs in CONFIG["models"].items():
        model = get_model(name, **kwargs)
        models.append(model)
        validate_depreciation_formula(model)

    return models


def step_run_analysis(aligned_data: dict, models: list, verbose: bool = True):
    """Step 4: Run depreciation analysis."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: DEPRECIATION ANALYSIS")
        print("=" * 60)

    all_results = AnalysisResults()

    for value_type in CONFIG["value_types"]:
        if value_type not in aligned_data:
            warnings.warn(f"Data for {value_type} not available, skipping")
            continue

        gdf = aligned_data[value_type]

        # Determine value column
        if value_type == "tourism":
            value_col = (
                "approx_price_corrected"
                if "approx_price_corrected" in gdf.columns
                else "approx_price"
            )
        else:
            value_col = "gdp_spared_value"

        for scenario in CONFIG["scenarios"]:
            # Column names are lowercase
            change_col = f"nearest_y_future_{scenario.lower()}_change"

            if change_col not in gdf.columns:
                warnings.warn(f"Column {change_col} not found, skipping {scenario}")
                continue

            for model in models:
                result = calculate_depreciation(
                    economic_gdf=gdf,
                    value_column=value_col,
                    change_column=change_col,
                    model=model,
                    value_type=value_type,
                )

                key = f"{value_type}_{scenario}_{model.name}"
                all_results.add(key, result)

                if verbose:
                    print(f"\n  {key}:")
                    print(
                        f"    Total loss: ${result.total_loss / 1e9:.2f}B ({result.loss_fraction * 100:.1f}%)"
                    )

    return all_results


def step_cumulative_impact(
    aligned_data: dict, models: list, data: dict, verbose: bool = True
):
    """Step 4b: Calculate cumulative impact over time."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4b: CUMULATIVE IMPACT ANALYSIS")
        print("=" * 60)

    all_cumulative = {}

    for value_type in CONFIG["value_types"]:
        if value_type not in aligned_data:
            continue

        gdf = aligned_data[value_type]

        # Get baseline values
        baseline_cover = gdf["nearest_y_new"].mean()
        value_col = (
            "approx_price_corrected"
            if "approx_price_corrected" in gdf.columns
            else "approx_price"
        )
        baseline_value = gdf[value_col].sum()

        if verbose:
            print(f"\n  {value_type.title()}:")
            print(f"    Baseline cover: {baseline_cover * 100:.1f}%")
            print(f"    Baseline value: ${baseline_value / 1e9:.2f}B")

        # Build scenario endpoints from the data
        # We need to extract the projected covers for each scenario/year
        scenario_endpoints = {}

        for scenario in CONFIG["scenarios"]:
            # Parse scenario
            parts = scenario.lower().replace("_yr_", "_").split("_")
            rcp = parts[0]  # e.g., "rcp45"
            year = int(parts[1])  # e.g., 2050

            if rcp not in scenario_endpoints:
                scenario_endpoints[rcp] = {}

            # Get projected cover for this scenario
            proj_col = f"nearest_y_future_{scenario.lower()}"
            if proj_col in gdf.columns:
                projected_cover = gdf[proj_col].mean()
                scenario_endpoints[rcp][year] = projected_cover

        if verbose:
            print("\n    Scenario endpoints:")
            for rcp, endpoints in scenario_endpoints.items():
                print(f"      {rcp.upper()}: {endpoints}")

        # Calculate cumulative impacts for each model and interpolation method
        for model in models:
            cumulative_results = calculate_cumulative_impacts_multi_scenario(
                baseline_year=2017,
                baseline_cover=baseline_cover,
                baseline_value=baseline_value,
                scenario_endpoints=scenario_endpoints,
                model=model,
                interpolation_methods=["linear", "exponential"],
                discount_rate=0.0,
                value_type=value_type,
            )

            for key, result in cumulative_results.items():
                full_key = f"{value_type}_{key}_{model.name}"
                all_cumulative[full_key] = result

                if verbose:
                    print(f"\n    {key} ({model.name}):")
                    print(
                        f"      Cumulative loss: ${result.total_cumulative_loss / 1e12:.2f}T"
                    )
                    print(
                        f"      Final annual loss: ${result.annual_loss_at_end / 1e9:.2f}B/year"
                    )

    return all_cumulative


def step_generate_outputs(
    results: AnalysisResults,
    cumulative_results: dict = None,
    data: dict = None,
    verbose: bool = True,
):
    """Step 5: Generate figures and save results in organized subdirectories."""
    import matplotlib.pyplot as plt

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING OUTPUTS")
        print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    results_dir = Path(CONFIG["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"

    # Create matching run directory in results_dir (organized like figures_dir)
    results_run_dir = results_dir / f"run_{timestamp}"
    results_run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    subdirs = {
        "models": run_dir / "01_model_comparison",
        "verification": run_dir / "02_verification",
        "scenarios": run_dir / "03_scenario_results",
        "gdp_impact": run_dir / "04_gdp_impact",
        "trajectories": run_dir / "05_trajectories",
        "summary": run_dir / "06_summary",
    }
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Model comparison plots
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  📊 Generating model comparison plots...")

    models = [get_model(name, **kwargs) for name, kwargs in CONFIG["models"].items()]
    plot_model_comparison(models, save_path=subdirs["models"] / "model_comparison.png")
    plot_model_comparison_interactive(
        models, save_path=subdirs["models"] / "model_comparison.html"
    )

    # -------------------------------------------------------------------------
    # 2. Verification plots (from original notebook)
    # -------------------------------------------------------------------------
    if data and "tourism" in data and "by_country" in data:
        if verbose:
            print("\n  ✅ Generating verification plots...")

        generate_verification_plots(
            tourism_gdf=data["tourism"].gdf,
            by_country_df=data["by_country"],
            output_dir=subdirs["verification"],
        )

    # -------------------------------------------------------------------------
    # 3. Per-scenario figures (organized by model, then scenario)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  📈 Generating per-scenario figures...")

    for key, result in results.results.items():
        # Organize by model first, then scenario
        model_name = utils.sanitize_filename(result.model.name)
        scenario_slug = utils.sanitize_filename(result.scenario)

        # Create model subdirectory, then scenario subdirectory
        model_dir = subdirs["scenarios"] / model_name
        scenario_dir = model_dir / scenario_slug
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"      {key} ({model_name})...")

        generate_figure_set(
            result=result,
            output_dir=scenario_dir,
            formats=CONFIG["save_formats"],
        )

    # -------------------------------------------------------------------------
    # 4. GDP impact plots (depreciation as % of GDP, organized by model)
    # -------------------------------------------------------------------------
    if data and "gdp" in data and data["gdp"] is not None:
        if verbose:
            print("\n  💵 Generating GDP impact plots...")

        gdp_data = data["gdp"]

        for key, result in results.results.items():
            # Organize by model first
            model_name = utils.sanitize_filename(result.model.name)
            scenario_slug = utils.sanitize_filename(result.scenario)

            # Create model subdirectory
            model_dir = subdirs["gdp_impact"] / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Bar chart
            fig = plot_loss_as_gdp_pct_bar(
                result,
                gdp_data,
                top_n=20,
                save_path=model_dir / f"{scenario_slug}_gdp_pct_bar.png",
            )
            if fig:
                plt.close(fig)

            # Choropleth
            fig = plot_loss_as_gdp_pct_choropleth(
                result,
                gdp_data,
                save_path=model_dir / f"{scenario_slug}_gdp_pct_choropleth.html",
            )

    # -------------------------------------------------------------------------
    # 5. Trajectory and cumulative impact plots
    # -------------------------------------------------------------------------
    if cumulative_results:
        if verbose:
            print("\n  📈 Generating trajectory and cumulative impact plots...")

        # Group results by model for separate sets of plots
        by_model = {}
        for key, result in cumulative_results.items():
            model_name = utils.sanitize_filename(result.model.name)
            if model_name not in by_model:
                by_model[model_name] = {}
            by_model[model_name][key] = result

        for model_name, model_results in by_model.items():
            model_subdir = subdirs["trajectories"] / model_name
            model_subdir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"      {model_name}...")

            # Static plots
            fig = plot_coral_cover_trajectories(
                model_results,
                save_path=model_subdir / "coral_cover_trajectories.png",
            )
            if fig:
                plt.close(fig)

            fig = plot_annual_value_trajectories(
                model_results,
                save_path=model_subdir / "annual_value_trajectories.png",
            )
            if fig:
                plt.close(fig)

            fig = plot_annual_loss_trajectories(
                model_results,
                save_path=model_subdir / "annual_loss_trajectories.png",
            )
            if fig:
                plt.close(fig)

            fig = plot_cumulative_loss_trajectories(
                model_results,
                save_path=model_subdir / "cumulative_loss_trajectories.png",
            )
            if fig:
                plt.close(fig)

            # Interactive 4-panel plot
            plot_trajectory_comparison_interactive(
                model_results,
                save_path=model_subdir / "trajectories_interactive.html",
            )

        # Save cumulative impact summary CSV
        cumulative_summary = summarize_cumulative_impacts(cumulative_results)
        cumulative_summary.to_csv(
            subdirs["trajectories"] / "cumulative_impact_summary.csv", index=False
        )

    # -------------------------------------------------------------------------
    # 6. Summary plots and comparisons
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  📋 Generating summary plots...")

    for value_type in CONFIG["value_types"]:
        fig = plot_scenario_comparison(
            results,
            value_type=value_type,
            top_n=CONFIG["top_n_countries"],
            save_path=subdirs["summary"] / f"{value_type}_scenario_comparison.png",
        )
        if fig:
            plt.close(fig)

    # GDP impact scenario comparison
    if data and "gdp" in data:
        if verbose:
            print("\n  💰 Generating GDP impact scenario comparison...")

        fig = plot_gdp_impact_scenario_comparison(
            results,
            gdp_data=data["gdp"],
            top_n=CONFIG["top_n_countries"],
            save_path=subdirs["summary"] / "gdp_impact_scenario_comparison.png",
        )
        if fig:
            plt.close(fig)

    # -------------------------------------------------------------------------
    # 7. Save CSV results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  💾 Saving CSV results...")

    # Create summary subdirectory in results_run_dir
    results_summary_dir = results_run_dir / "summary"
    results_summary_dir.mkdir(parents=True, exist_ok=True)

    # Summary table (save to both locations for convenience)
    summary = results.summary_table()
    summary.to_csv(subdirs["summary"] / f"summary_{timestamp}.csv", index=False)
    summary.to_csv(results_summary_dir / f"summary_{timestamp}.csv", index=False)

    # Per-country results (save to both locations)
    for key, result in results.results.items():
        by_country = result.by_country
        safe_key = utils.sanitize_filename(key)
        by_country.to_csv(
            subdirs["summary"] / f"{safe_key}_by_country.csv", index=False
        )
        by_country.to_csv(
            results_summary_dir / f"{safe_key}_by_country.csv", index=False
        )

    # -------------------------------------------------------------------------
    # 8. Save full results objects (for reloading)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  💾 Saving full results objects...")

    # Save AnalysisResults to results_run_dir (for loading)
    results_save_dir = results_run_dir / "results"
    results.save(results_save_dir)

    # Also save to run_dir for completeness
    run_results_dir = run_dir / "results"
    results.save(run_results_dir)

    # Save cumulative results to results_run_dir
    if cumulative_results:
        cumulative_save_dir = results_run_dir / "cumulative_results"
        cumulative_save_dir.mkdir(parents=True, exist_ok=True)

        for key, result in cumulative_results.items():
            safe_key = utils.sanitize_filename(key)
            result.save(cumulative_save_dir / f"{safe_key}.pkl")

        # Also save to run_dir for completeness
        run_cumulative_dir = run_dir / "cumulative_results"
        run_cumulative_dir.mkdir(parents=True, exist_ok=True)
        for key, result in cumulative_results.items():
            safe_key = utils.sanitize_filename(key)
            result.save(run_cumulative_dir / f"{safe_key}.pkl")

        if verbose:
            print(f"  ✓ Saved {len(cumulative_results)} cumulative impact results")

    if verbose:
        print(f"\n✓ Outputs saved to {run_dir}")
        print("  ├── 01_model_comparison/")
        print("  ├── 02_verification/")
        print("  ├── 03_scenario_results/")
        print("  │   ├── {model_name}/")
        print("  │   │   └── {scenario}/")
        print("  ├── 04_gdp_impact/")
        print("  │   └── {model_name}/")
        print("  ├── 05_trajectories/")
        print("  │   └── {model_name}/")
        print("  ├── 06_summary/")
        print("  ├── results/ (for reloading)")
        print("  └── cumulative_results/ (for reloading)")
        print(f"\n✓ Results saved to {results_run_dir}")
        print("  ├── summary/ (CSV files)")
        print("  ├── results/ (pickle files for reloading)")
        print("  └── cumulative_results/ (pickle files for reloading)")

    return run_dir


def step_print_summaries(results: AnalysisResults, verbose: bool = True):
    """Step 6: Print summary reports."""
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: SUMMARY REPORTS")
        print("=" * 60)

    for key, result in results.results.items():
        print_summary(result, top_n=10)

    # Overall comparison
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    summary = results.summary_table()
    print(summary.to_string(index=False))


def step_export_web_data(
    results: AnalysisResults,
    cumulative_results: dict,
    data: dict,
    output_dir: Path = None,
    sample_fraction: float = 0.1,
    verbose: bool = True,
):
    """Step 7: Export data for web visualization using pre-computed results."""
    from src.economics.export_web_data import (
        export_country_results,
        export_cumulative_country_results,
        export_cumulative_site_results,
        export_gdp_impact,
        export_model_comparison,
        export_site_results,
        export_summary_stats,
        export_trajectory_data,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7: EXPORTING WEB DATA")
        print("=" * 60)

    if output_dir is None:
        output_dir = Path("docs/exported_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export using pre-computed results (no re-running the pipeline!)
    if verbose:
        print("\n  Exporting data for web visualization...")

    export_country_results(results, output_dir)
    export_site_results(results, output_dir, sample_fraction=sample_fraction)

    if cumulative_results:
        export_trajectory_data(cumulative_results, output_dir)
        export_cumulative_country_results(results, cumulative_results, output_dir)
        export_cumulative_site_results(
            results, cumulative_results, output_dir, sample_fraction=sample_fraction
        )

    export_summary_stats(results, cumulative_results or {}, output_dir)
    export_model_comparison(output_dir)

    gdp_data = data.get("gdp") if data else None
    if gdp_data is not None:
        export_gdp_impact(results, gdp_data, output_dir)

    if verbose:
        print(f"\n  ✓ Web data exported to {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(verbose: bool = True):
    """
    Run the complete analysis pipeline.

    Returns
    -------
    dict
        Pipeline outputs including data, results, and output paths.
    """
    print("\n" + "#" * 60)
    print("# CORAL REEF ECONOMICS ANALYSIS PIPELINE")
    print("#" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load data
    data = step_load_data(verbose)

    # Step 2: Align data
    aligned = step_align_data(data, verbose)

    # Step 3: Validate models
    models = step_validate_models(verbose)

    # Step 4: Run analysis
    results = step_run_analysis(aligned, models, verbose)

    # Step 4b: Calculate cumulative impacts
    cumulative = step_cumulative_impact(aligned, models, data, verbose)

    # Step 5: Generate outputs
    output_dir = step_generate_outputs(
        results,
        cumulative_results=cumulative,
        data=data,
        verbose=verbose,
    )

    # Step 6: Print summaries
    step_print_summaries(results, verbose)

    # Print cumulative impact summary
    if cumulative:
        print_cumulative_summary(cumulative)

    if CONFIG.get("export_web_data", False):
        step_export_web_data(results, cumulative, data, verbose=verbose)

    print("\n" + "#" * 60)
    print("# PIPELINE COMPLETE")
    print("#" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        "data": data,
        "aligned": aligned,
        "models": models,
        "results": results,
        "cumulative": cumulative,
        "output_dir": output_dir,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run coral reef economics analysis pipeline"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to analyze (e.g., RCP45_yr_2100 RCP85_yr_2100)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Depreciation models to use (e.g., linear compound)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available depreciation models and exit",
    )
    parser.add_argument(
        "--export-web-data",
        action="store_true",
        help="Export data for web visualization",
    )
    parser.add_argument(
        "--load-run",
        type=str,
        default=None,
        help="Load previous results from a run directory (e.g., 'run_20260119_105256')",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable depreciation models:")
        for name, desc in list_models().items():
            print(f"\n  {name}:")
            print(f"    {desc}")
        return

    # Override config if CLI args provided
    if args.scenarios:
        CONFIG["scenarios"] = args.scenarios
    if args.models:
        CONFIG["models"] = {m: {} for m in args.models}
    if args.export_web_data:
        CONFIG["export_web_data"] = args.export_web_data

    # Load previous results if requested
    if args.load_run:
        print(f"\n{'=' * 60}")
        print(f"LOADING PREVIOUS RESULTS: {args.load_run}")
        print(f"{'=' * 60}\n")
        loaded = load_previous_results(args.load_run, verbose=not args.quiet)
        print("\n✓ Results loaded successfully!")
        print("\nTo use these results, access them via:")
        print("  - loaded['results']: AnalysisResults object")
        print(
            "  - loaded['cumulative_results']: Dict of CumulativeImpactResult objects"
        )
        return loaded

    # Run pipeline
    run_pipeline(verbose=not args.quiet)


if __name__ == "__main__":
    main()
