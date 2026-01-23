"""
Analysis pipeline for coral reef economics.

This module provides functions to:
1. Calculate projected economic losses under different scenarios
2. Aggregate results by country/region
3. Generate summary statistics
4. Compare different depreciation models
"""

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from .depreciation_models import DepreciationModel, get_model

# =============================================================================
# ANALYSIS RESULTS
# =============================================================================


@dataclass
class DepreciationResult:
    """Results from a depreciation analysis."""

    gdf: gpd.GeoDataFrame  # Site-level results
    scenario: str  # e.g., "RCP85_2100"
    model: DepreciationModel
    value_type: str  # "tourism" or "shoreline_protection"

    # Aggregations (computed on demand)
    _by_country: pd.DataFrame = field(default=None, repr=False)

    @property
    def total_original_value(self) -> float:
        return self.gdf["original_value"].sum()

    @property
    def total_remaining_value(self) -> float:
        return self.gdf["remaining_value"].sum()

    @property
    def total_loss(self) -> float:
        return self.total_original_value - self.total_remaining_value

    @property
    def loss_fraction(self) -> float:
        if self.total_original_value == 0:
            return 0
        return self.total_loss / self.total_original_value

    @property
    def by_country(self) -> pd.DataFrame:
        """Aggregate results by country."""
        if self._by_country is None:
            self._by_country = self._compute_by_country()
        return self._by_country

    def _compute_by_country(self) -> pd.DataFrame:
        """Compute country-level aggregations."""
        country_col = self._get_country_column()

        agg = (
            self.gdf.groupby(country_col)
            .agg(
                original_value=("original_value", "sum"),
                remaining_value=("remaining_value", "sum"),
                value_loss=("value_loss", "sum"),
                n_sites=("original_value", "count"),
                mean_coral_change=("coral_change", "mean"),
            )
            .reset_index()
        )

        agg["loss_fraction"] = agg["value_loss"] / agg["original_value"]
        agg["loss_fraction"] = agg["loss_fraction"].fillna(0)

        return agg.sort_values("value_loss", ascending=False)

    def _get_country_column(self) -> str:
        """Find the country column name."""
        for col in ["country", "nearest_country", "Country_Name"]:
            if col in self.gdf.columns:
                return col
        raise ValueError("No country column found in results")

    def save(self, path: Path) -> None:
        """Save result to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the full object (GeoDataFrame and model included)
        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Also save metadata as JSON for inspection
        metadata = {
            "scenario": self.scenario,
            "model_name": self.model.name,
            "model_type": type(self.model).__name__,
            "value_type": self.value_type,
            "total_original_value": float(self.total_original_value),
            "total_remaining_value": float(self.total_remaining_value),
            "total_loss": float(self.total_loss),
            "loss_fraction": float(self.loss_fraction),
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DepreciationResult":
        """Load result from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class AnalysisResults:
    """Container for multiple depreciation analyses."""

    results: Dict[str, DepreciationResult] = field(default_factory=dict)

    def add(self, key: str, result: DepreciationResult) -> None:
        """Add a result with a descriptive key."""
        self.results[key] = result

    def get(self, key: str) -> DepreciationResult:
        """Get a result by key."""
        return self.results[key]

    def summary_table(self) -> pd.DataFrame:
        """Generate summary table comparing all results."""
        rows = []
        for key, result in self.results.items():
            rows.append(
                {
                    "analysis": key,
                    "scenario": result.scenario,
                    "model": result.model.name,
                    "value_type": result.value_type,
                    "original_value": result.total_original_value,
                    "remaining_value": result.total_remaining_value,
                    "total_loss": result.total_loss,
                    "loss_fraction": result.loss_fraction,
                }
            )
        return pd.DataFrame(rows)

    def compare_models(self, value_type: str = None) -> pd.DataFrame:
        """Compare different models for the same scenario/value type."""
        df = self.summary_table()
        if value_type:
            df = df[df["value_type"] == value_type]
        return df.pivot_table(
            index="scenario",
            columns="model",
            values=["total_loss", "loss_fraction"],
            aggfunc="first",
        )

    def save(self, path: Path) -> None:
        """Save all results to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each individual result
        for key, result in self.results.items():
            safe_key = key.replace("/", "_").replace(" ", "_")
            result_path = path / f"{safe_key}.pkl"
            result.save(result_path)

        # Save summary metadata
        summary = self.summary_table()
        summary.to_csv(path / "summary.csv", index=False)

        metadata = {
            "n_results": len(self.results),
            "scenarios": list(set(r.scenario for r in self.results.values())),
            "models": list(set(r.model.name for r in self.results.values())),
            "value_types": list(set(r.value_type for r in self.results.values())),
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AnalysisResults":
        """Load all results from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results directory not found: {path}")

        results = cls()

        # Load all .pkl files
        for pkl_file in path.glob("*.pkl"):
            result = DepreciationResult.load(pkl_file)
            # Use filename as key (it should match the original key used in save)
            key = pkl_file.stem
            results.add(key, result)

        return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def calculate_depreciation(
    economic_gdf: gpd.GeoDataFrame,
    value_column: str,
    change_column: str,
    model: Union[str, DepreciationModel],
    scenario_name: str = None,
    value_type: str = "unknown",
    **model_kwargs,
) -> DepreciationResult:
    """
    Calculate economic depreciation for a given scenario.

    Parameters
    ----------
    economic_gdf : GeoDataFrame
        Economic data with aligned coral cover projections.
    value_column : str
        Column containing economic values (e.g., "approx_price_corrected").
    change_column : str
        Column containing coral cover change (e.g., "nearest_Y_future_RCP85_yr_2100_change").
    model : str or DepreciationModel
        Depreciation model name or instance.
    scenario_name : str, optional
        Human-readable scenario name. Inferred from change_column if not provided.
    value_type : str
        Type of economic value ("tourism", "shoreline_protection").
    **model_kwargs
        Additional arguments passed to model if instantiating from name.

    Returns
    -------
    DepreciationResult
        Analysis results.
    """
    # Get model instance
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)

    # Infer scenario name
    if scenario_name is None:
        # e.g., "nearest_Y_future_RCP85_yr_2100_change" -> "RCP85_2100"
        scenario_name = (
            change_column.replace("nearest_", "")
            .replace("Y_future_", "")
            .replace("_yr_", "_")
            .replace("_change", "")
        )

    # Create result GeoDataFrame
    result_gdf = economic_gdf.copy()

    # Get values
    original_value = result_gdf[value_column].fillna(0)
    coral_change = result_gdf[change_column].fillna(0)

    # Calculate remaining value using model
    # TippingPointModel requires additional parameters (original_cc, threshold)
    if isinstance(model, type) and model.__name__ == "TippingPointModel":
        # Handle if model is a class (shouldn't happen, but be safe)
        model = model()

    if model.model_type == "tipping_point":
        # Find baseline coral cover column
        baseline_cover_col = None
        for col in [
            "nearest_average_coral_cover",
            "average_coral_cover",
            "nearest_y_new",
        ]:
            if col in result_gdf.columns:
                baseline_cover_col = col
                break

        if baseline_cover_col is None:
            warnings.warn(
                "TippingPointModel requires baseline coral cover, but no suitable column found. "
                "Using default original_cc=0.5. Expected columns: "
                "'nearest_average_coral_cover', 'average_coral_cover', or 'nearest_y_new'"
            )
            original_cc = np.full_like(coral_change, 0.5)
        else:
            original_cc = result_gdf[baseline_cover_col].fillna(0.5).values

        # Use model's threshold_cc attribute, or default to 0.1
        threshold = getattr(model, "threshold_cc", 0.1)

        # Calculate with tipping point model parameters
        remaining_value = model.calculate(
            coral_change, original_value, original_cc=original_cc, threshold=threshold
        )
    else:
        # Standard models (Linear, Compound, CoastalProtection) use simple signature
        remaining_value = model.calculate(coral_change, original_value)

    # Store results
    result_gdf["original_value"] = original_value
    result_gdf["coral_change"] = coral_change
    result_gdf["remaining_value"] = remaining_value
    result_gdf["value_loss"] = original_value - remaining_value
    result_gdf["loss_fraction"] = np.where(
        original_value > 0, (original_value - remaining_value) / original_value, 0
    )

    return DepreciationResult(
        gdf=result_gdf,
        scenario=scenario_name,
        model=model,
        value_type=value_type,
    )


def run_multi_scenario_analysis(
    economic_gdf: gpd.GeoDataFrame,
    value_column: str,
    models: List[Union[str, DepreciationModel]] = None,
    scenarios: List[str] = None,
    value_type: str = "unknown",
) -> AnalysisResults:
    """
    Run depreciation analysis across multiple scenarios and models.

    Parameters
    ----------
    economic_gdf : GeoDataFrame
        Economic data with aligned coral cover projections.
    value_column : str
        Column containing economic values.
    models : list, optional
        List of model names or instances. Default: ["linear", "compound"].
    scenarios : list, optional
        List of scenario column suffixes (e.g., ["RCP45_yr_2050", "RCP85_yr_2100"]).
        Default: infer from columns.
    value_type : str
        Type of economic value.

    Returns
    -------
    AnalysisResults
        Container with all analysis results.
    """
    if models is None:
        models = ["linear", "compound"]

    # Infer scenarios from columns
    if scenarios is None:
        change_cols = [c for c in economic_gdf.columns if "_change" in c]
        scenarios = []
        for col in change_cols:
            # Extract scenario part
            scenario = (
                col.replace("nearest_", "")
                .replace("Y_future_", "")
                .replace("_change", "")
            )
            if scenario not in scenarios:
                scenarios.append(scenario)

    results = AnalysisResults()

    for scenario in scenarios:
        change_col = f"nearest_Y_future_{scenario}_change"
        if change_col not in economic_gdf.columns:
            warnings.warn(
                f"Column {change_col} not found, skipping scenario {scenario}"
            )
            continue

        for model in models:
            model_name = model if isinstance(model, str) else model.name
            result = calculate_depreciation(
                economic_gdf=economic_gdf,
                value_column=value_column,
                change_column=change_col,
                model=model,
                scenario_name=scenario.replace("_yr_", "_"),
                value_type=value_type,
            )

            key = f"{value_type}_{scenario}_{model_name}"
            results.add(key, result)

    return results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================


def generate_summary_stats(
    result: DepreciationResult, top_n: int = 20
) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for a depreciation result.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    gdf = result.gdf
    by_country = result.by_country

    # Global stats
    stats = {
        "scenario": result.scenario,
        "model": result.model.name,
        "value_type": result.value_type,
        "global": {
            "n_sites": len(gdf),
            "original_value": result.total_original_value,
            "remaining_value": result.total_remaining_value,
            "total_loss": result.total_loss,
            "loss_fraction": result.loss_fraction,
            "mean_coral_change_pp": gdf["coral_change"].mean() * 100,
            "median_coral_change_pp": gdf["coral_change"].median() * 100,
        },
        "top_countries_by_loss": by_country.head(top_n)[
            [
                "country" if "country" in by_country.columns else by_country.columns[0],
                "original_value",
                "value_loss",
                "loss_fraction",
                "mean_coral_change",
            ]
        ].to_dict("records"),
        "distribution": {
            "loss_percentiles": {
                f"p{p}": np.percentile(gdf["value_loss"], p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            },
            "coral_change_percentiles": {
                f"p{p}": np.percentile(gdf["coral_change"] * 100, p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            },
        },
    }

    return stats


def print_summary(result: DepreciationResult, top_n: int = 10) -> None:
    """Print formatted summary of depreciation result."""
    stats = generate_summary_stats(result, top_n=top_n)

    print(f"\n{'=' * 60}")
    print(f"DEPRECIATION ANALYSIS: {stats['scenario']}")
    print(f"Model: {stats['model']} | Value Type: {stats['value_type']}")
    print(f"{'=' * 60}")

    g = stats["global"]
    print("\n📊 GLOBAL SUMMARY")
    print(f"   Sites analyzed: {g['n_sites']:,}")
    print(f"   Original value: ${g['original_value'] / 1e9:.2f}B")
    print(f"   Remaining value: ${g['remaining_value'] / 1e9:.2f}B")
    print(
        f"   Total loss: ${g['total_loss'] / 1e9:.2f}B ({g['loss_fraction'] * 100:.1f}%)"
    )
    print(f"   Mean coral change: {g['mean_coral_change_pp']:.1f} pp")

    print(f"\n🏆 TOP {top_n} COUNTRIES BY VALUE LOSS")
    for i, c in enumerate(stats["top_countries_by_loss"][:top_n], 1):
        country_name = c.get("country", list(c.values())[0])
        print(
            f"   {i:2}. {country_name}: ${c['value_loss'] / 1e6:.1f}M ({c['loss_fraction'] * 100:.1f}%)"
        )

    print()


# =============================================================================
# VALIDATION / DIAGNOSTICS
# =============================================================================


def validate_alignment(
    economic_gdf: gpd.GeoDataFrame, max_distance: float = 5.0
) -> Dict[str, Any]:
    """
    Validate the spatial alignment between economic and coral data.

    Returns
    -------
    dict
        Validation statistics.
    """
    dist_col = "distance_to_coral_site"
    if dist_col not in economic_gdf.columns:
        raise ValueError(
            f"Column {dist_col} not found. Run align_coral_to_economic_data first."
        )

    distances = economic_gdf[dist_col]

    return {
        "n_total": len(economic_gdf),
        "n_within_threshold": (distances <= max_distance).sum(),
        "n_beyond_threshold": (distances > max_distance).sum(),
        "pct_within_threshold": 100 * (distances <= max_distance).mean(),
        "distance_stats": {
            "min": distances.min(),
            "max": distances.max(),
            "mean": distances.mean(),
            "median": distances.median(),
            "p95": distances.quantile(0.95),
        },
    }


def validate_depreciation_formula(model: DepreciationModel) -> None:
    """
    Validate depreciation formula with test cases.

    Prints test results for manual verification.
    """
    print(f"\n🧪 TESTING MODEL: {model.name}")
    print(f"   {model.description}")

    test_cases = [
        (-0.10, 1000, "10pp decrease on $1000"),
        (-0.50, 1000, "50pp decrease on $1000"),
        (0.0, 1000, "No change"),
        (0.10, 1000, "10pp increase (should not depreciate)"),
        (-0.10, 0, "10pp decrease on $0"),
    ]

    print("\n   Test cases:")
    for delta_cc, value, desc in test_cases:
        # Handle tipping point model which requires original_cc
        if model.model_type == "tipping_point":
            threshold = getattr(model, "threshold_cc", 0.1)
            # Use a reasonable default original_cc for testing
            original_cc = 0.5 if delta_cc >= 0 else 0.5 + abs(delta_cc)
            remaining = model.calculate(
                delta_cc, value, original_cc=original_cc, threshold=threshold
            )
        else:
            remaining = model.calculate(delta_cc, value)
        loss = value - remaining
        loss_pct = 100 * loss / value if value > 0 else 0
        print(
            f"   {desc}: ${value:.0f} → ${remaining:.0f} (loss: ${loss:.0f}, {loss_pct:.1f}%)"
        )
