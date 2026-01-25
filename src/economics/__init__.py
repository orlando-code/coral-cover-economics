"""
Economics analysis module for coral reef valuation.

This module provides:
- Pluggable depreciation models for mapping coral cover change to economic loss
- Data loading and alignment utilities
- Analysis pipelines for tourism and coastal protection values
- Static and interactive plotting utilities
"""

from .analysis import (
    AnalysisResults,
    DepreciationResult,
    calculate_depreciation,
    generate_summary_stats,
    print_summary,
    run_multi_scenario_analysis,
    validate_alignment,
    validate_depreciation_formula,
)
from .cumulative_impact import (
    CoverTrajectory,
    CumulativeImpactResult,
    TrajectoryPoint,
    calculate_cumulative_impact,
    calculate_cumulative_impacts_multi_scenario,
    interpolate_exponential,
    interpolate_linear,
    print_cumulative_summary,
    summarize_cumulative_impacts,
)
from .data_loader import (
    CoralCoverData,
    EconomicValueData,
    align_coral_to_economic_data,
    compute_country_aggregations,
    load_coral_cover_data,
    load_gdp_data,
    load_shoreline_protection_data,
    load_tourism_data,
)
from .depreciation_models import (
    CoastalProtectionModel,
    CompoundModel,
    DepreciationModel,
    LinearModel,
    TippingPointModel,
    compare_models,
    get_model,
    list_models,
)
from .plotting import (
    generate_figure_set,
    generate_verification_plots,
    plot_annual_loss_trajectories,
    plot_annual_value_trajectories,
    plot_choropleth_interactive,
    plot_coral_cover_trajectories,
    plot_country_losses,
    plot_cumulative_loss_trajectories,
    plot_cumulative_loss_scenario_comparison,
    plot_gdp_impact_scenario_comparison,
    plot_gdp_percentage_bar,
    plot_gdp_percentage_choropleth,
    plot_loss_as_gdp_pct_bar,
    plot_loss_as_gdp_pct_choropleth,
    plot_loss_distribution,
    plot_model_comparison,
    plot_model_comparison_interactive,
    plot_scenario_comparison,
    plot_total_revenue_choropleth,
    plot_tourism_value_bins_map,
    plot_trajectory_comparison_interactive,
)

__all__ = [
    # Models
    "DepreciationModel",
    "LinearModel",
    "CompoundModel",
    "TippingPointModel",
    "CoastalProtectionModel",
    "get_model",
    "list_models",
    "compare_models",
    # Data loading
    "CoralCoverData",
    "EconomicValueData",
    "load_coral_cover_data",
    "load_tourism_data",
    "load_shoreline_protection_data",
    "load_gdp_data",
    "align_coral_to_economic_data",
    "compute_country_aggregations",
    # Analysis
    "DepreciationResult",
    "AnalysisResults",
    "calculate_depreciation",
    "run_multi_scenario_analysis",
    "generate_summary_stats",
    "print_summary",
    "validate_alignment",
    "validate_depreciation_formula",
    # Plotting
    "plot_model_comparison",
    "plot_country_losses",
    "plot_scenario_comparison",
    "plot_loss_distribution",
    "plot_choropleth_interactive",
    "plot_model_comparison_interactive",
    "generate_figure_set",
    # Verification plots
    "plot_gdp_percentage_bar",
    "plot_gdp_percentage_choropleth",
    "plot_total_revenue_choropleth",
    "plot_tourism_value_bins_map",
    "generate_verification_plots",
    # GDP impact plots
    "plot_loss_as_gdp_pct_bar",
    "plot_loss_as_gdp_pct_choropleth",
    "plot_gdp_impact_scenario_comparison",
    # Trajectory plots
    "plot_coral_cover_trajectories",
    "plot_annual_value_trajectories",
    "plot_annual_loss_trajectories",
    "plot_cumulative_loss_trajectories",
    "plot_cumulative_loss_scenario_comparison",
    "plot_trajectory_comparison_interactive",
    # Cumulative impact
    "TrajectoryPoint",
    "CoverTrajectory",
    "CumulativeImpactResult",
    "interpolate_linear",
    "interpolate_exponential",
    "calculate_cumulative_impact",
    "calculate_cumulative_impacts_multi_scenario",
    "summarize_cumulative_impacts",
    "print_cumulative_summary",
]
