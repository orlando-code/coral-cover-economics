"""Future coral-cover projections from a fitted HBB model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.hbb.design import build_design_matrix, inverse_transform_beta
from src.models.hbb.model import HierarchicalBetaModel

def build_current_design_matrix(
    model: "HierarchicalBetaModel",
    df: pd.DataFrame,
    standardization_stats: Optional[dict[str, tuple[float, float]]] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the current design matrix from a dataframe using the model's predictors.

    Parameters
    ----------
    model : HierarchicalBetaModel
        Trained model (must have col_names and reef_to_site_map attributes)
    df : pd.DataFrame
        Dataframe with current/historical data
    standardization_stats : dict, optional
        Standardization statistics. If None, uses model.standardization_stats
    verbose : bool
        If True, print validation information

    Returns
    -------
    tuple
        (X_current, site_idx) - Design matrix and site indices
    """
    if verbose:
        print("=" * 60)
        print("BUILD CURRENT DESIGN MATRIX - VALIDATION")
        print("=" * 60)

    if model.col_names is None:
        raise ValueError("Model must have col_names attribute")

    if standardization_stats is None:
        if hasattr(model, "standardization_stats"):
            standardization_stats = model.standardization_stats
        else:
            raise ValueError(
                "standardization_stats must be provided or model must have standardization_stats"
            )

    # Get predictors (exclude 'Intercept' if present)
    predictors = [p for p in model.col_names if p != "Intercept"]

    if verbose:
        print(
            f"\n📊 Model predictors ({len(predictors)}): {predictors[:5]}..."
            if len(predictors) > 5
            else f"\n📊 Model predictors ({len(predictors)}): {predictors}"
        )

    # Standardize variables if needed
    df_work = df.copy()
    standardized_vars = []
    for pred in predictors:
        base_name = pred.replace("_stzd", "")
        if base_name not in standardization_stats:
            continue

        # Check if already standardized
        if pred not in df_work.columns:
            # Need to standardize
            if base_name in df_work.columns:
                mean_val, std_val = standardization_stats[base_name]
                df_work[pred] = (df_work[base_name] - mean_val) / std_val
                standardized_vars.append(pred)

                if verbose:
                    raw_vals = df_work[base_name]
                    std_vals = df_work[pred]
                    print(f"  📐 Standardizing '{base_name}' -> '{pred}':")
                    print(
                        f"      Raw:  min={raw_vals.min():.3f}, max={raw_vals.max():.3f}, mean={raw_vals.mean():.3f}"
                    )
                    print(
                        f"      Std:  min={std_vals.min():.3f}, max={std_vals.max():.3f}, mean={std_vals.mean():.3f}"
                    )
                    print(f"      Stats: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                raise ValueError(
                    f"Column '{base_name}' or '{pred}' not found in dataframe"
                )

    # Build design matrix
    X_current, _ = build_design_matrix(df_work, predictors, add_intercept=True)

    if verbose:
        print(f"\n📐 Design matrix shape: {X_current.shape}")
        print(
            f"    Column means: {X_current.mean(axis=0)[:5]}..."
            if X_current.shape[1] > 5
            else f"    Column means: {X_current.mean(axis=0)}"
        )

    # Get site indices - USE SAVED MAPPING IF AVAILABLE
    if model.reef_to_site_map is not None and "reef_id" in df_work.columns:
        # Use the saved mapping from training for consistent site effects
        reef_ids = df_work["reef_id"].values
        site_idx = np.array([model.reef_to_site_map.get(rid, -1) for rid in reef_ids])

        # Check for unmapped reef_ids
        unmapped_count = np.sum(site_idx == -1)
        n_sites_trained = len(model.reef_to_site_map)

        if verbose:
            print(
                f"\n🔗 Using saved reef_to_site_map ({n_sites_trained} sites in training)"
            )
            print(
                f"    Mapped {len(reef_ids) - unmapped_count}/{len(reef_ids)} reef_ids successfully"
            )

        if unmapped_count > 0:
            print(
                f"    ⚠️  WARNING: {unmapped_count} reef_ids not found in training data!"
            )
            print("       These will use mean site effect (index 0)")
            site_idx[site_idx == -1] = 0  # Default to site 0 for unknown sites

    elif "site" in df_work.columns:
        site_idx = df_work["site"].values
        if verbose:
            print("\n🔗 Using 'site' column directly")
            print(f"    Unique sites: {len(np.unique(site_idx))}")
    elif "reef_id" in df_work.columns:
        # FALLBACK: Create new codes - WARNING: may not match training!
        print("\n⚠️  WARNING: No reef_to_site_map available. Creating new site indices.")
        print("    Site effects may NOT match training data!")
        site_idx = pd.Categorical(df_work["reef_id"]).codes
        if verbose:
            print(f"    Created {len(np.unique(site_idx))} unique site indices")
    else:
        # Create dummy site indices if not available
        print("\n⚠️  WARNING: No site information available. Using dummy indices.")
        site_idx = np.zeros(len(df_work), dtype=int)

    if verbose:
        print("\n✅ Design matrix and site indices ready")
        print("=" * 60)

    return X_current, site_idx


def project_future_coral_cover(
    model: "HierarchicalBetaModel",
    df: pd.DataFrame,
    X_current: np.ndarray,
    site_idx: np.ndarray,
    changed_columns: dict[str, str],
    n_prediction_samples: int = 1000,
    standardization_stats: Optional[dict[str, tuple[float, float]]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Project future coral cover by updating specified columns in the design matrix.

    This is a simplified interface: just specify which standardized variable columns
    should be updated with which dataframe columns, and the function handles the rest.

    Parameters
    ----------
    model : HierarchicalBetaModel
        Fitted model (must have col_names attribute)
    df : pd.DataFrame
        Dataframe containing both current and future variable columns
    X_current : np.ndarray
        Current design matrix (n_obs x n_predictors)
    site_idx : np.ndarray
        Site indices for observations
    changed_columns : dict
        Dictionary mapping standardized variable names (as in model.col_names)
        to dataframe column names containing future values.
        Example: {'sst_mean_stzd': 'sst_mean_ssp585_2100', 'human_pop_stzd': 'human_pop_2100'}
    standardization_stats : dict, optional
        Dictionary mapping variable names (without _stzd suffix) to (mean, std) tuples.
        If None, will try to extract from model if available.
    output_path : Path, optional
        Path to save results CSV file
    verbose : bool
        If True, print validation information

    Returns
    -------
    pd.DataFrame
        DataFrame with current and projected coral cover, including:
        - Y_current: Current predictions (inverse-transformed to original scale)
        - Y_future: Future predictions (inverse-transformed to original scale)
        - Y_future_std: Standard deviation of future predictions
        - Y_change: Absolute change
        - Y_relative_change: Relative change (%)
    """
    if verbose:
        print("=" * 60)
        print("PROJECT FUTURE CORAL COVER - VALIDATION")
        print("=" * 60)

    if model.col_names is None:
        raise ValueError(
            "Model must have col_names attribute. Ensure model was fit with col_names parameter."
        )

    if standardization_stats is None:
        if hasattr(model, "standardization_stats"):
            standardization_stats = model.standardization_stats
        else:
            raise ValueError(
                "standardization_stats must be provided or model must have standardization_stats attribute"
            )

    # Get n_observations for inverse_transform_beta
    if hasattr(model, "n_observations") and model.n_observations is not None:
        n_obs = model.n_observations
        if verbose:
            print(f"\n📊 Using n_observations from model: {n_obs}")
    else:
        n_obs = len(df)
        if verbose:
            print(f"\n⚠️  n_observations not saved in model, using df length: {n_obs}")

    # Create future design matrix
    X_future = X_current.copy()

    if verbose:
        print(f"\n🔄 Updating {len(changed_columns)} variables for future projection:")

    # Update variables in design matrix
    for std_var_name, future_col_name in changed_columns.items():
        if std_var_name not in model.col_names:
            print(
                f"⚠️  Warning: Variable '{std_var_name}' not in model columns. Skipping."
            )
            continue

        if future_col_name not in df.columns:
            raise ValueError(
                f"Future column '{future_col_name}' not found in dataframe."
            )

        # Find column index in design matrix
        col_idx = model.col_names.index(std_var_name)

        # Get base variable name (without _stzd suffix)
        base_var_name = std_var_name.replace("_stzd", "")

        # Standardize future values using original statistics
        if base_var_name in standardization_stats:
            mean_val, std_val = standardization_stats[base_var_name]
            future_values = df[future_col_name].values

            # Check for NaN values
            nan_count = np.sum(np.isnan(future_values))

            # Current standardized values
            current_std_values = X_current[:, col_idx]

            # Standardize future values
            future_values_std = (future_values - mean_val) / std_val

            # Update design matrix
            X_future[:, col_idx] = future_values_std

            if verbose:
                print(f"\n  📐 {std_var_name} <- {future_col_name}:")
                print(f"      Standardization: mean={mean_val:.4f}, std={std_val:.4f}")
                print(
                    f"      Future raw: min={np.nanmin(future_values):.4f}, max={np.nanmax(future_values):.4f}, mean={np.nanmean(future_values):.4f}"
                )
                print(
                    f"      Current std: min={current_std_values.min():.4f}, max={current_std_values.max():.4f}, mean={current_std_values.mean():.4f}"
                )
                print(
                    f"      Future std:  min={np.nanmin(future_values_std):.4f}, max={np.nanmax(future_values_std):.4f}, mean={np.nanmean(future_values_std):.4f}"
                )
                print(
                    f"      Delta std:   mean={np.nanmean(future_values_std - current_std_values):.4f}"
                )
                if nan_count > 0:
                    print(f"      ⚠️  {nan_count} NaN values in future data")
        else:
            raise ValueError(f"Standardization stats not found for {base_var_name}.")

    # Get predictions (in beta-transformed space)
    if verbose:
        print("\n🔮 Generating predictions...")
    print("Generating current predictions...")
    current_stats = model.predict(
        X_current, site_idx, n_samples=n_prediction_samples, verbose=verbose
    )

    print("Generating future predictions...")
    future_stats = model.predict(
        X_future, site_idx, n_samples=n_prediction_samples, verbose=verbose
    )

    # Use means and standard deviations for summary outputs
    y_current_beta = current_stats["mean"]
    y_current_std = current_stats["std"]
    y_future_beta = future_stats["mean"]
    y_future_std = future_stats["std"]

    # Apply inverse_transform_beta to convert from beta space to original scale
    if verbose:
        print(f"\n🔄 Applying inverse_transform_beta (n={n_obs})...")
        print(
            f"    Before transform - Current: mean={y_current_beta.mean():.4f}, Future: mean={y_future_beta.mean():.4f}"
        )

    y_current = inverse_transform_beta(y_current_beta, n_obs)
    y_future = inverse_transform_beta(y_future_beta, n_obs)

    if verbose:
        print(
            f"    After transform  - Current: mean={y_current.mean():.4f}, Future: mean={y_future.mean():.4f}"
        )

    # Calculate changes
    y_change = y_future - y_current
    y_relative_percent_change = np.where(
        y_current > 0, (y_future - y_current) / y_current * 100, np.nan
    )

    if verbose:
        print("\n📊 PROJECTION SUMMARY:")
        print(
            f"    Current coral cover:  mean={y_current.mean() * 100:.2f}%, median={np.median(y_current) * 100:.2f}%"
        )
        print(
            f"    Future coral cover:   mean={y_future.mean() * 100:.2f}%, median={np.median(y_future) * 100:.2f}%"
        )
        print(
            f"    Absolute change:      mean={y_change.mean() * 100:.2f}%, median={np.median(y_change) * 100:.2f}%"
        )
        print(
            f"    Relative change:      mean={np.nanmean(y_relative_percent_change):.2f}%, median={np.nanmedian(y_relative_percent_change):.2f}%"
        )
        print("=" * 60)
    # Create results dataframe
    results = pd.DataFrame(
        {
            "Y_current": y_current,
            "Y_current_std": y_current_std,
            "Y_future": y_future,
            "Y_future_std": y_future_std,
            "Y_change": y_change,
            "Y_relative_percent_change": y_relative_percent_change,
        }
    )

    # Add metadata columns if available
    if "lat" in df.columns:
        results["latitude"] = df["lat"].values
    if "lon" in df.columns:
        results["longitude"] = df["lon"].values
    if "site" in df.columns:
        results["site"] = df["site"].values
    if "reef_id" in df.columns:
        results["reef_id"] = df["reef_id"].values

    # Save results if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return results


def load_model_and_project(
    model_path: Path,
    df: pd.DataFrame,
    X: np.ndarray,
    site_idx: np.ndarray,
    scenarios: list[tuple[str, int]],
    update_variables: Optional[dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> dict[tuple[str, int], pd.DataFrame]:
    """
    Convenience function to load a trained model and run projections for multiple scenarios.

    Parameters
    ----------
    model_path : Path
        Path to saved model directory
    df : pd.DataFrame
        Data with future projections
    X : np.ndarray
        Current design matrix
    site_idx : np.ndarray
        Site indices
    scenarios : list of tuples
        list of (scenario, year) tuples, e.g., [('ssp585', 2050), ('ssp585', 2100)]
    update_variables : dict, optional
        dictionary mapping standardized variable names to future column names
    output_dir : Path, optional
        Directory to save projection results

    Returns
    -------
    dict
        dictionary mapping (scenario, year) tuples to result DataFrames
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = HierarchicalBetaModel.load_model(model_path)

    # Run projections for each scenario
    results = {}
    for scenario, year in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Projecting for scenario: {scenario}, year: {year}")
        print(f"{'=' * 80}")

        # Build update_variables dict from scenario/year if not provided
        if update_variables is None:
            update_variables = {
                "sst_mean_stzd": f"sst_mean_{scenario}_{year}",
                "human_pop_stzd": f"human_pop_{year}_vals",
            }

        result = project_future_coral_cover(
            model=model,
            df=df,
            X_current=X,
            site_idx=site_idx,
            changed_columns=update_variables,
            standardization_stats=model.standardization_stats,
            output_path=output_dir / f"projections_{scenario}_{year}.csv"
            if output_dir
            else None,
        )

        results[(scenario, year)] = result

    return results