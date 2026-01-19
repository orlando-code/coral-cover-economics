"""
Export analysis results to JSON format for interactive web visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.economics import run_economic_analysis
from src.economics.analysis import AnalysisResults
from src.economics.cumulative_impact import CumulativeImpactResult


def export_country_results(
    results: AnalysisResults,
    output_dir: Path,
) -> None:
    """Export country-level results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_countries = []

    for _, result in results.results.items():
        by_country = result.by_country.copy()

        # Get ISO codes
        if "iso_a3" not in by_country.columns and "iso_a3" in result.gdf.columns:
            country_col = result._get_country_column()
            iso_map = result.gdf.groupby(country_col)["iso_a3"].first()
            by_country["iso_a3"] = by_country[country_col].map(iso_map)

        for _, row in by_country.iterrows():
            all_countries.append(
                {
                    "scenario": result.scenario,
                    "model": result.model.name,
                    "country": row.get("country", row.iloc[0]),
                    "iso_a3": row.get("iso_a3", ""),
                    "original_value": float(row.get("original_value", 0)),
                    "remaining_value": float(row.get("remaining_value", 0)),
                    "value_loss": float(row.get("value_loss", 0)),
                    "loss_fraction": float(row.get("loss_fraction", 0)),
                }
            )

    with open(output_dir / "country_results.json", "w") as f:
        json.dump(all_countries, f, indent=2)

    print(f"Exported country results: {len(all_countries)} records")


def export_site_results(
    results: AnalysisResults,
    output_dir: Path,
    sample_fraction: float = 0.1,
) -> None:
    """Export site-level results to GeoJSON for point visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    features_by_scenario = {}

    for key, result in results.results.items():
        gdf = result.gdf.copy()

        # Sample for performance (full dataset is too large)
        if sample_fraction < 1.0:
            gdf = gdf.sample(frac=sample_fraction, random_state=42)

        # Convert to WGS84 and get centroids
        gdf = gdf.to_crs("EPSG:4326")
        if gdf.geometry.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]:
            centroids = gdf.geometry.centroid
        else:
            centroids = gdf.geometry

        features = []
        for idx, (_, row) in enumerate(gdf.iterrows()):
            geom = centroids.iloc[idx]
            if geom.is_empty:
                continue

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                    "properties": {
                        "country": str(row.get("country", "")),
                        "original_value": float(row.get("original_value", 0)),
                        "remaining_value": float(row.get("remaining_value", 0)),
                        "value_loss": float(row.get("value_loss", 0)),
                        "loss_fraction": float(row.get("loss_fraction", 0)),
                        "coral_change": float(row.get("coral_change", 0)),
                    },
                }
            )

        scenario_key = (
            f"{result.scenario}_{result.model.name}".replace(" ", "_")
            .replace("/", "_")
            .replace("%", "pct")
            .replace("(", "")
            .replace(")", "")
        )

        features_by_scenario[scenario_key] = {
            "type": "FeatureCollection",
            "scenario": result.scenario,
            "model": result.model.name,
            "features": features,
        }

    # Save each scenario separately
    for scenario_key, geojson in features_by_scenario.items():
        with open(output_dir / f"sites_{scenario_key}.json", "w") as f:
            json.dump(geojson, f)
        print(f"Exported sites for {scenario_key}: {len(geojson['features'])} points")

    # Save manifest
    manifest = {
        "scenarios": list(features_by_scenario.keys()),
        "generated": datetime.now().isoformat(),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def export_trajectory_data(
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
) -> None:
    """Export trajectory data for time series visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []

    for key, result in cumulative_results.items():
        traj = result.trajectory
        trajectories.append(
            {
                "key": key,
                "scenario": traj.scenario,
                "interpolation": traj.interpolation_method,
                "model": result.model.name,
                "years": traj.years.tolist(),
                "coral_cover": (traj.covers * 100).tolist(),  # as percentage
                "annual_value": (result.annual_values / 1e9).tolist(),  # billions
                "annual_loss": (result.annual_losses / 1e9).tolist(),  # billions
                "cumulative_loss": (
                    result.cumulative_losses / 1e12
                ).tolist(),  # trillions
                "baseline_value": result.baseline_value / 1e9,
                "total_cumulative_loss": result.total_cumulative_loss / 1e12,
            }
        )

    with open(output_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"Exported trajectory data: {len(trajectories)} scenarios")


def export_summary_stats(
    results: AnalysisResults,
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
) -> None:
    """Export summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated": datetime.now().isoformat(),
        "snapshot_results": [],
        "cumulative_results": [],
    }

    # Snapshot results
    for key, result in results.results.items():
        summary["snapshot_results"].append(
            {
                "scenario": result.scenario,
                "model": result.model.name,
                "original_value_billions": result.total_original_value / 1e9,
                "remaining_value_billions": result.total_remaining_value / 1e9,
                "total_loss_billions": result.total_loss / 1e9,
                "loss_fraction_pct": result.loss_fraction * 100,
            }
        )

    # Cumulative results
    for key, result in cumulative_results.items():
        traj = result.trajectory
        summary["cumulative_results"].append(
            {
                "key": key,
                "scenario": traj.scenario,
                "interpolation": traj.interpolation_method,
                "model": result.model.name,
                "period": f"{traj.start_year}-{traj.end_year}",
                "baseline_cover_pct": traj.covers[0] * 100,
                "final_cover_pct": traj.covers[-1] * 100,
                "cover_change_pp": traj.total_change * 100,
                "baseline_value_billions": result.baseline_value / 1e9,
                "final_value_billions": result.annual_values[-1] / 1e9,
                "annual_loss_at_end_billions": result.annual_loss_at_end / 1e9,
                "total_cumulative_loss_trillions": result.total_cumulative_loss / 1e12,
            }
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Exported summary statistics")


def export_model_comparison(output_dir: Path) -> None:
    """Export depreciation model curves for visualization."""
    from src.economics.depreciation_models import get_model

    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "linear": get_model("linear"),
        "compound": get_model("compound"),
    }

    # Generate curves
    delta_cc_range = np.linspace(-50, 10, 100)  # -50% to +10% change
    baseline = 100  # $100 baseline for easy percentage calculation

    curves = {}
    for name, model in models.items():
        remaining = [model.calculate(d / 100, baseline) for d in delta_cc_range]
        curves[name] = {
            "name": model.name,
            "delta_cc": delta_cc_range.tolist(),
            "remaining_value": remaining,
            "loss_pct": [(baseline - r) for r in remaining],
        }

    with open(output_dir / "model_curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    print("Exported model comparison curves")


def export_gdp_impact(
    results: AnalysisResults,
    gdp_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export GDP impact data for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gdp_map = gdp_data.set_index("iso_a3")["gdp"].to_dict()

    gdp_impacts = []

    for key, result in results.results.items():
        by_country = result.by_country.copy()

        # Get ISO codes
        if "iso_a3" not in by_country.columns and "iso_a3" in result.gdf.columns:
            country_col = result._get_country_column()
            iso_map = result.gdf.groupby(country_col)["iso_a3"].first()
            by_country["iso_a3"] = by_country[country_col].map(iso_map)

        for _, row in by_country.iterrows():
            iso = row.get("iso_a3", "")
            country = row.get("country", row.iloc[0])
            value_loss = float(row.get("value_loss", 0))
            national_gdp = gdp_map.get(iso, 0)

            if national_gdp > 0:
                loss_as_gdp_pct = 100 * value_loss / national_gdp
            else:
                loss_as_gdp_pct = 0

            gdp_impacts.append(
                {
                    "scenario": result.scenario,
                    "model": result.model.name,
                    "country": country,
                    "iso_a3": iso,
                    "value_loss": value_loss,
                    "national_gdp": national_gdp,
                    "loss_as_gdp_pct": loss_as_gdp_pct,
                }
            )

    with open(output_dir / "gdp_impacts.json", "w") as f:
        json.dump(gdp_impacts, f, indent=2)

    print(f"Exported GDP impact data: {len(gdp_impacts)} records")


def run_export(output_dir: Optional[Path] = None, sample_fraction: float = 0.1) -> Path:
    """Run the full export pipeline."""
    if output_dir is None:
        output_dir = Path("docs/exported_data")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPORTING DATA FOR WEB VISUALIZATION")
    print("=" * 60)

    # Run the analysis pipeline
    print("\nRunning analysis pipeline...")
    pipeline_results = run_economic_analysis.run_pipeline(verbose=False)

    results = pipeline_results["results"]
    cumulative = pipeline_results.get("cumulative", {})
    gdp_data = pipeline_results.get("data", {}).get("gdp")

    # Export all data
    print("\nExporting data...")
    export_country_results(results, output_dir)
    export_site_results(results, output_dir, sample_fraction=sample_fraction)
    export_trajectory_data(cumulative, output_dir)
    export_summary_stats(results, cumulative, output_dir)
    export_model_comparison(output_dir)

    if gdp_data is not None:
        export_gdp_impact(results, gdp_data, output_dir)

    print(f"\n✓ All data exported to {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_export()
