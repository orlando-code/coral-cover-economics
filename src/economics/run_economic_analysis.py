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

import numpy as np

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
    add_spatial_cumulative_losses,
    calculate_cumulative_impacts_multi_scenario_per_site,
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
        "tipping_point": {"threshold_cc": 0.10, "post_threshold_loss": 1.0},
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

                # Add spatial cumulative losses to the gdf
                try:
                    result = add_spatial_cumulative_losses(
                        result=result,
                        baseline_year=2013,
                        discount_rate=0.0,
                        interpolation_method="linear",
                    )
                    if verbose:
                        # Check if cumulative columns were added
                        cum_cols = [
                            c
                            for c in result.gdf.columns
                            if c.startswith("cumulative_loss_")
                        ]
                        if cum_cols:
                            print(
                                f"    Added spatial cumulative loss columns: {len(cum_cols)}"
                            )
                except Exception as e:
                    if verbose:
                        warnings.warn(
                            f"    Could not add spatial cumulative losses: {e}"
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
    """
    Step 4b: Calculate cumulative impact over time.

    **REFACTORED**: Now calculates cumulative impact per-site, then aggregates.
    This is critical for tipping point models where each site's original_cc
    determines when it crosses the threshold.

    Logic:
    1. For each site individually:
       - Use site's own baseline_cover as original_cc (for tipping point models)
       - Use site's own baseline_value
       - Use site's own projected_cover for each scenario
       - Generate site-specific trajectory
       - Calculate site's cumulative impact
    2. Aggregate all site results:
       - Sum annual_values across sites
       - Sum annual_losses across sites
       - Sum annual_value_lost across sites
       - Sum annual_opportunity_cost across sites
       - Sum cumulative_losses across sites
    3. Create aggregated CumulativeImpactResult with combined trajectory

    This preserves the true distribution of tipping point behaviors:
    - Sites with high original_cc (e.g., 70%) collapse later
    - Sites with low original_cc (e.g., 15%) collapse early
    - Total cumulative loss = sum of all these different trajectories
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4b: CUMULATIVE IMPACT ANALYSIS (Per-Site Calculation)")
        print("=" * 60)

    all_cumulative = {}

    for value_type in CONFIG["value_types"]:
        if value_type not in aligned_data:
            continue

        gdf = aligned_data[value_type]

        # Find baseline coral cover column
        baseline_cover_col = None
        for col in [
            "nearest_y_new",
            "nearest_average_coral_cover",
            "average_coral_cover",
        ]:
            if col in gdf.columns:
                baseline_cover_col = col
                break

        if baseline_cover_col is None:
            if verbose:
                print(
                    f"\n  ⚠️  {value_type.title()}: No baseline cover column found, skipping"
                )
            continue

        # Get value column
        value_col = (
            "approx_price_corrected"
            if "approx_price_corrected" in gdf.columns
            else "approx_price"
        )

        if value_col not in gdf.columns:
            if verbose:
                print(f"\n  ⚠️  {value_type.title()}: No value column found, skipping")
            continue

        # Extract per-site data (vectorized)
        baseline_covers = gdf[baseline_cover_col].fillna(0).values
        baseline_values = gdf[value_col].fillna(0).values

        # Create mask for valid sites (must have positive cover and value)
        valid_mask = (
            ~np.isnan(baseline_covers)
            & ~np.isnan(baseline_values)
            & (baseline_covers > 0)
            & (baseline_values > 0)
        )

        valid_indices = np.where(valid_mask)[0]
        valid_baseline_covers = baseline_covers[valid_indices]
        valid_baseline_values = baseline_values[valid_indices]

        if len(valid_indices) == 0:
            if verbose:
                print(f"\n  ⚠️  {value_type.title()}: No valid sites found, skipping")
            continue

        # Calculate aggregate statistics for reporting
        total_baseline_value = baseline_values.sum()
        mean_baseline_cover = baseline_covers[valid_indices].mean()

        if verbose:
            print(f"\n  {value_type.title()}:")
            print(f"\tValid sites: {len(valid_indices)} / {len(gdf)}")
            print(
                f"\t\tMissing (nan) baseline covers: {np.isnan(baseline_covers).sum()}"
            )
            print(
                f"\t\tMissing (nan) or zero baseline values: {(np.isnan(baseline_values) | (baseline_values <= 0)).sum()}"
            )
            print(f"\tMean baseline cover: {mean_baseline_cover * 100:.1f}%")
            print(f"\tTotal baseline value: ${total_baseline_value / 1e9:.2f}B")

        # Build scenario endpoints per site
        # Structure: scenario_endpoints[rcp][year] = array of projected covers (one per site)
        scenario_endpoints_per_site = {}

        for scenario in CONFIG["scenarios"]:
            # Parse scenario
            parts = scenario.lower().replace("_yr_", "_").split("_")
            rcp = parts[0]  # e.g., "rcp45"
            year = int(parts[1])  # e.g., 2050

            if rcp not in scenario_endpoints_per_site:
                scenario_endpoints_per_site[rcp] = {}

            # Get projected cover column for this scenario
            proj_col = f"nearest_y_future_{scenario.lower()}"
            if proj_col in gdf.columns:
                projected_covers = gdf[proj_col].fillna(0).values
                # Extract only valid sites
                valid_projected_covers = projected_covers[valid_indices]
                scenario_endpoints_per_site[rcp][year] = valid_projected_covers
            else:
                if verbose:
                    print(f"    ⚠️  Column not found: {proj_col}")

        if verbose:
            print("\n    Scenario endpoints (mean across sites):")
            for rcp, endpoints in scenario_endpoints_per_site.items():
                mean_endpoints = {
                    year: covers.mean() for year, covers in endpoints.items()
                }
                print(f"      {rcp.upper()}: {mean_endpoints}")

        # Calculate cumulative impacts for each model and interpolation method
        for model in models:
            print(f"\nCalculating cumulative impacts for {model.name}")
            # Calculate per-site cumulative impacts, then aggregate
            cumulative_results = calculate_cumulative_impacts_multi_scenario_per_site(
                baseline_year=2013,
                baseline_covers=valid_baseline_covers,  # Per-site baseline covers
                baseline_values=valid_baseline_values,  # Per-site baseline values
                scenario_endpoints_per_site=scenario_endpoints_per_site,  # Per-site projected covers
                model=model,
                interpolation_methods=["linear", "exponential"],
                discount_rate=0.0,
                value_type=value_type,
                verbose=verbose,
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

    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory in results_dir (figures will go here too)
    results_run_dir = results_dir / f"run_{timestamp}"
    results_run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories (all in results_run_dir, including figures)
    subdirs = {
        "models": results_run_dir / "figures" / "01_model_comparison",
        "verification": results_run_dir / "figures" / "02_verification",
        "scenarios": results_run_dir / "figures" / "03_scenario_results",
        "gdp_impact": results_run_dir / "figures" / "04_gdp_impact",
        "trajectories": results_run_dir / "figures" / "05_trajectories",
        "summary": results_run_dir / "summary",
    }
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Model comparison plots
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  📊 Generating model comparison plots...")

        # Ensure the subdirectory exists before saving plots
        subdirs["models"].mkdir(parents=True, exist_ok=True)

        models = [
            get_model(name, **kwargs) for name, kwargs in CONFIG["models"].items()
        ]
        plot_model_comparison(models, save_path=subdirs["models"])
    # plot_model_comparison_interactive(
    #     models, save_path=subdirs["models"] / "model_comparison.html"
    # ) # TODO: implement
    # TODO: fix this
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

    # Summary table (save to summary subdirectory)
    summary = results.summary_table()
    summary.to_csv(subdirs["summary"] / f"summary_{timestamp}.csv", index=False)

    # Per-country results (save to summary subdirectory)
    for key, result in results.results.items():
        by_country = result.by_country
        safe_key = utils.sanitize_filename(key)
        by_country.to_csv(
            subdirs["summary"] / f"{safe_key}_by_country.csv", index=False
        )

    # -------------------------------------------------------------------------
    # 8. Save full results objects (for reloading)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  💾 Saving full results objects...")

    # Save AnalysisResults to results_run_dir (for loading)
    results_save_dir = results_run_dir / "results"
    results.save(results_save_dir)

    # Save cumulative results to results_run_dir
    if cumulative_results:
        cumulative_save_dir = results_run_dir / "cumulative_results"
        cumulative_save_dir.mkdir(parents=True, exist_ok=True)

        for key, result in cumulative_results.items():
            safe_key = utils.sanitize_filename(key)
            result.save(cumulative_save_dir / f"{safe_key}.pkl")

        if verbose:
            print(f"  ✓ Saved {len(cumulative_results)} cumulative impact results")

    if verbose:
        print(f"\n✓ All outputs saved to {results_run_dir}")
        print("  ├── figures/")
        print("  │   ├── 01_model_comparison/")
        print("  │   ├── 02_verification/")
        print("  │   ├── 03_scenario_results/")
        print("  │   │   ├── {model_name}/")
        print("  │   │   │   └── {scenario}/")
        print("  │   ├── 04_gdp_impact/")
        print("  │   │   └── {model_name}/")
        print("  │   └── 05_trajectories/")
        print("  │       └── {model_name}/")
        print("  ├── summary/ (CSV files)")
        print("  ├── results/ (pickle files for reloading)")
        print("  └── cumulative_results/ (pickle files for reloading)")

    return results_run_dir


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
    sample_fraction: float = 1.0,
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
