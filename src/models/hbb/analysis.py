"""Full analysis pipeline and spot/ocean summaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.hbb._config import (
    CV_PREDICTORS,
    HAS_PYMC,
    OUTPUT_DIR,
    VARS_TO_STANDARDIZE,
    ModelSpec,
)
from src.models.hbb.data import (
    load_model_data_from_pipeline,
    standardize_variables,
)
from src.models.hbb.design import (
    build_design_matrix,
    compute_correlation_matrix,
    inverse_transform_beta,
    transform_to_beta,
)
from src.models.hbb.indices import prepare_hierarchical_indices
from src.models.hbb.model import HierarchicalBetaModel
from src.models.hbb.projections import project_future_coral_cover
from src.plots.hb_beta_plots import (
    plot_bright_dark_spots_map,
    plot_correlation_matrix,
    plot_observed_vs_expected,
)


def create_output_dir_path(output_path: Path) -> Path:
    """
    Create output directory path.

    Parameters
    ----------
    output_path : Path
        Path to save diagnostics
    """
    # save with a timestamp in the output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = output_path / f"model_{timestamp}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path


def identify_bright_dark_spots(
    observed: np.ndarray, expected: np.ndarray, threshold_sd: float = 1.5
) -> pd.DataFrame:
    """
    Identify bright spots (observed >> expected) and dark spots (observed << expected).

    Parameters
    ----------
    observed : np.ndarray
        Observed coral cover values
    expected : np.ndarray
        Model-predicted expected coral cover
    threshold_sd : float
        Number of standard deviations for classification

    Returns
    -------
    pd.DataFrame
        Classification results
    """
    deviation = observed - expected
    sd = np.std(observed)

    classification = np.where(
        deviation > threshold_sd * sd,
        "bright_spot",
        np.where(deviation < -threshold_sd * sd, "dark_spot", "normal"),
    )

    return pd.DataFrame(
        {
            "observed": observed,
            "expected": expected,
            "deviation": deviation,
            "deviation_normalized": deviation / sd,
            "classification": classification,
        }
    )


# =============================================================================
# OCEAN/REGION STATISTICS
# =============================================================================


def calculate_coral_cover_by_ocean(
    df: pd.DataFrame,
    reef_id_col: str = "reef_id",
    ocean_col: str = "ocean",
    cover_col: str = "average_coral_cover",
    date_col: str = "days_since_19811231",
) -> pd.DataFrame:
    """
    Calculate mean coral cover per ocean, avoiding pseudo-replication.

    Uses the most recent survey for each reef.

    Parameters
    ----------
    df : pd.DataFrame
        Coral cover data
    reef_id_col : str
        Column name for reef ID
    ocean_col : str
        Column name for ocean
    cover_col : str
        Column name for coral cover
    date_col : str
        Column name for survey date

    Returns
    -------
    pd.DataFrame
        Summary statistics by ocean
    """
    # Get most recent survey for each reef
    idx = df.groupby(reef_id_col)[date_col].idxmax()
    reef_data = df.loc[idx].copy()

    # Calculate statistics by ocean
    stats = reef_data.groupby(ocean_col)[cover_col].agg(["mean", "std", "count"])
    stats.columns = ["mean_cover", "std_cover", "n_reefs"]

    # Add overall statistics
    overall = pd.DataFrame(
        {
            "mean_cover": [reef_data[cover_col].mean()],
            "std_cover": [reef_data[cover_col].std()],
            "n_reefs": [len(reef_data)],
        },
        index=["Overall"],
    )

    return pd.concat([stats, overall])


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_full_analysis(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fit_bayesian_model: bool = True,
    save_diagnostics: bool = True,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 6,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    random_seed: int = 42,
    project_scenarios: Optional[list[tuple[str, int]]] = None,
    n_prediction_samples: int = 1000,
    model_spec: ModelSpec = "reparam",
    add_intercept: bool = False,
) -> dict[str, Any]:
    """
    Run the complete coral cover analysis pipeline.

    Args:
        data_path (Path, optional): Path to data.csv
        output_dir (Path, optional): Directory for outputs
        fit_bayesian_model (bool): Whether to fit the full Bayesian model
        save_diagnostics (bool): Whether to save model diagnostics
        n_samples (int): Number of posterior samples
        n_tune (int): Number of tuning samples
        n_chains (int): Number of chains for MCMC sampling
        target_accept (float): Target acceptance rate for NUTS sampler
        max_treedepth (int): Maximum tree depth for NUTS sampler
        random_seed (int): Random seed for reproducibility
        project_scenarios (list of (str, int), optional):
        If provided, run future projections for each (scenario, year) tuple,
        e.g. [('rcp45', 2050), ('rcp45', 2100), ('rcp85', 2050), ('rcp85', 2100)].
        n_prediction_samples (int): Number of posterior predictive samples for projections.
        model_spec (str): ``reparam`` (default) or ``legacy_r`` (centered JAGS + intercept).
        add_intercept (bool): Add intercept column to ``X`` (forced True for ``legacy_r``).

    Returns:
        dict: Dictionary containing all results
    """

    # 0. Create output directory and return path
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = create_output_dir_path(output_dir)
    # if output_dir is None:
    #     output_dir = OUTPUT_DIR
    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Load and clean data (same pipeline as hbb.ipynb / CV)
    print("Loading data...")
    if data_path is not None:
        data_dir = Path(data_path).parent
        df = load_model_data_from_pipeline(data_dir)
    else:
        df = load_model_data_from_pipeline()
    results["n_observations"] = len(df)
    print(f"Loaded {len(df)} observations")

    # 2. Compute and plot correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df)
    results["correlation_matrix"] = corr_matrix
    print("Plotting correlation matrix...")
    plot_correlation_matrix(corr_matrix, output_dir / "corrplot.png")
    print("Saving correlation matrix to CSV...")
    corr_matrix.to_csv(output_dir / "corrplot.csv")

    # 3. Standardize variables
    print("Standardizing variables...")
    df["lat"] = np.abs(df["lat"])  # Use absolute latitude
    df, std_stats = standardize_variables(df, VARS_TO_STANDARDIZE)
    results["standardization_stats"] = std_stats

    # 4. Hierarchical indices (legacy join may expand rows — before design matrix)
    index_mode = "legacy_r" if model_spec == "legacy_r" else "reparam"
    print(f"Building hierarchical indices (mode={index_mode})...")
    hier = prepare_hierarchical_indices(df, mode=index_mode)
    if "df" in hier:
        df = hier["df"]
        results["n_observations"] = len(df)
        print(f"   Legacy join expanded to {len(df)} observations")
    site_idx = hier["site_idx"]
    region_idx = hier["region_idx"]
    site_region_map = hier["site_to_region"]
    diversity = hier["diversity"]
    reef_to_site_map = hier["reef_to_site_map"]
    print(
        f"   Sites: {hier['n_sites']}, regions: {hier['n_regions']}, "
        f"spec={model_spec}, intercept={add_intercept or model_spec == 'legacy_r'}"
    )
    print(f"Created reef_to_site_map with {len(reef_to_site_map)} unique reef_ids")
    results["reef_to_site_map"] = reef_to_site_map

    # 5. Design matrix: reparam has no intercept; legacy_r matches R model.matrix(~ ...)
    add_intercept = add_intercept or model_spec == "legacy_r"
    print("Building design matrix...")
    predictors = [p for p in CV_PREDICTORS if p in df.columns]
    X, col_names = build_design_matrix(df, predictors, add_intercept=add_intercept)
    results["col_names"] = col_names
    results["model_spec"] = model_spec

    # Response variable
    n = len(df)
    y = transform_to_beta(df["average_coral_cover"].values, n)
    y_new = df["average_coral_cover"].values.copy()

    # 6. Fit model
    if fit_bayesian_model and HAS_PYMC:
        print(
            f"\nFitting Bayesian model spec={model_spec} "
            f"(n_samples={n_samples}, n_tune={n_tune}, target_accept={target_accept}, "
            f"max_treedepth={max_treedepth}, random_seed={random_seed})"
        )
        model = HierarchicalBetaModel()
        model.fit(
            X,
            y,
            site_idx,
            region_idx,
            site_region_map,
            reef_to_site_map=reef_to_site_map,
            diversity=diversity,
            col_names=col_names,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            random_seed=random_seed,
            spec=model_spec,
        )
        # Store standardization stats in model for future use
        model.standardization_stats = std_stats
        results["model"] = model

        # Save model
        model_save_path = output_dir / "trained_model"
        print("Saving trained model...")
        model.save_model(model_save_path)

        # Save coefficient summary (including 5/95% HDIs and significance)
        coef_summary = model.get_coefficient_summary()
        print("Saving coefficient summary...")
        coef_summary.to_csv(output_dir / "beta_est.csv")
        results["coefficient_summary"] = coef_summary

        # Save model diagnostics
        if save_diagnostics:
            print("Saving model diagnostics...")
            model.save_diagnostics(output_dir / "diagnostics")

        print("Plotting coefficient traces and posteriors...")
        model.plot_coefficient_traces_and_posteriors(
            output_dir / "coefficient_diagnostics"
        )

        # Predictions
        print("Making predictions...")
        predictions = model.predict(X, site_idx)

        # Extract all statistics from predictions dict (still in beta space)
        y_new_beta_mean = predictions["mean"]
        y_new_beta_std = predictions["std"]
        y_new_beta_median = predictions["median"]
        y_new_beta_ci_lower_95 = predictions["ci_lower_95"]
        y_new_beta_ci_upper_95 = predictions["ci_upper_95"]
        y_new_beta_ci_lower_50 = predictions["ci_lower_50"]
        y_new_beta_ci_upper_50 = predictions["ci_upper_50"]
        y_new_beta_min = predictions["min"]
        y_new_beta_max = predictions["max"]
        y_new_significant = predictions["significant"]

        # Transform summary statistics back to original [0,1] scale
        y_new = inverse_transform_beta(y_new_beta_mean, n)
        y_new_std = inverse_transform_beta(y_new_beta_std, n)
        y_new_median = inverse_transform_beta(y_new_beta_median, n)
        y_new_ci_lower_95 = inverse_transform_beta(y_new_beta_ci_lower_95, n)
        y_new_ci_upper_95 = inverse_transform_beta(y_new_beta_ci_upper_95, n)
        y_new_ci_lower_50 = inverse_transform_beta(y_new_beta_ci_lower_50, n)
        y_new_ci_upper_50 = inverse_transform_beta(y_new_beta_ci_upper_50, n)
        y_new_min = inverse_transform_beta(y_new_beta_min, n)
        y_new_max = inverse_transform_beta(y_new_beta_max, n)

        # Store all statistics in results
        results["Y_New"] = y_new
        results["Y_New_stats"] = {
            "mean": y_new,
            "std": y_new_std,
            "median": y_new_median,
            "ci_lower_95": y_new_ci_lower_95,
            "ci_upper_95": y_new_ci_upper_95,
            "ci_lower_50": y_new_ci_lower_50,
            "ci_upper_50": y_new_ci_upper_50,
            "min": y_new_min,
            "max": y_new_max,
            "significant": y_new_significant,
        }

        # Save prediction statistics to CSV
        print("Saving prediction statistics...")
        pred_stats_df = pd.DataFrame(
            {
                "mean": y_new,
                "std": y_new_std,
                "median": y_new_median,
                "ci_lower_95": y_new_ci_lower_95,
                "ci_upper_95": y_new_ci_upper_95,
                "ci_lower_50": y_new_ci_lower_50,
                "ci_upper_50": y_new_ci_upper_50,
                "min": y_new_min,
                "max": y_new_max,
                "significant": y_new_significant,
            }
        )
        pred_stats_df.to_csv(output_dir / "prediction_statistics.csv", index=False)

        # Optionally run future projections for multiple scenarios/years
        if project_scenarios:
            print("\nRunning future projections for requested scenarios/years...")
            scenario_results: dict[tuple[str, int], pd.DataFrame] = {}
            metric_frames: list[pd.DataFrame] = []

            # Common per-site metadata
            base_meta_cols = [
                c for c in ["reef_id", "site", "lat", "lon"] if c in df.columns
            ]
            base_meta = (
                df[base_meta_cols].reset_index(drop=True) if base_meta_cols else None
            )

            for scenario, year in project_scenarios:
                print(f"  -> Projecting {scenario.upper()} {year}")

                changed_columns = {
                    "sst_mean_stzd": f"sst_mean_{scenario}_{year}",
                    "human_pop_stzd": f"human_pop_{year}_vals",
                }

                proj_df = project_future_coral_cover(
                    model=model,
                    df=df,
                    X_current=X,
                    site_idx=site_idx,
                    changed_columns=changed_columns,
                    standardization_stats=std_stats,
                    n_prediction_samples=n_prediction_samples,
                    verbose=False,
                    output_path=output_dir / f"projections_{scenario}_{year}.csv",
                )

                scenario_results[(scenario, year)] = proj_df

                # Build a MultiIndex column frame for metrics, sharing the same row index
                metrics = [
                    "Y_current",
                    "Y_current_std",
                    "Y_future",
                    "Y_future_std",
                    "Y_change",
                    "Y_relative_percent_change",
                ]
                metrics = [m for m in metrics if m in proj_df.columns]
                metrics_df = proj_df[metrics].copy()
                metrics_df.columns = pd.MultiIndex.from_product(
                    [(scenario, year), metrics],
                    names=["scenario", "year", "metric"],
                )
                metric_frames.append(metrics_df.reset_index(drop=True))

            if metric_frames:
                combined_metrics = pd.concat(metric_frames, axis=1)
                if base_meta is not None:
                    combined_df = pd.concat([base_meta, combined_metrics], axis=1)
                else:
                    combined_df = combined_metrics

                # Save multi-scenario projections as JSON (preserves MultiIndex)
                combined_df.to_json(
                    output_dir / "future_projections.json", orient="split"
                )

                # Also save a flattened CSV version for convenience
                flat_df = combined_df.copy()
                flat_cols = []
                for col in flat_df.columns:
                    if isinstance(col, tuple):
                        scen, yr, metric = col
                        flat_cols.append(f"{scen}_{yr}_{metric}")
                    else:
                        flat_cols.append(col)
                flat_df.columns = flat_cols
                flat_df.to_csv(output_dir / "future_projections.csv", index=False)

                results["future_projections"] = scenario_results

    # 6. Identify bright and dark spots
    print("Identifying bright and dark spots...")
    spots_df = identify_bright_dark_spots(df["average_coral_cover"].values, y_new)
    results["spots"] = spots_df

    # Plot observed vs expected
    plot_observed_vs_expected(
        df["average_coral_cover"].values,
        y_new,
        spots_df["classification"].values,
        output_dir / "observed_vs_expected_coral_cover.png",
    )

    # Plot map
    plot_bright_dark_spots_map(
        df, spots_df["classification"].values, output_dir / "bright_dark_spots_map.png"
    )

    # 7. Calculate ocean statistics
    print("Calculating ocean statistics...")
    ocean_stats = calculate_coral_cover_by_ocean(df)
    ocean_stats.to_csv(output_dir / "coral_cover_by_ocean.csv")
    results["ocean_stats"] = ocean_stats

    # 8. Save processed data
    print("Saving results...")
    df["y_new"] = y_new
    df["deviation_from_expected"] = spots_df["deviation_normalized"].values
    df["classification"] = spots_df["classification"].values

    # Add prediction statistics if available
    if fit_bayesian_model and HAS_PYMC and "Y_New_stats" in results:
        stats = results["Y_New_stats"]
        df["y_new_mean"] = stats["mean"]
        df["y_new_std"] = stats["std"]
        df["y_new_median"] = stats["median"]
        df["y_new_ci_lower_95"] = stats["ci_lower_95"]
        df["y_new_ci_upper_95"] = stats["ci_upper_95"]
        df["y_new_ci_lower_50"] = stats["ci_lower_50"]
        df["y_new_ci_upper_50"] = stats["ci_upper_50"]
        df["y_new_min"] = stats["min"]
        df["y_new_max"] = stats["max"]
        df["y_new_significant"] = stats["significant"]

    df.to_csv(output_dir / "data_processed.csv", index=False)

    print(f"Analysis complete. Results saved to {output_dir}")

    return results
