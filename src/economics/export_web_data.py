"""
Export analysis results to JSON format for interactive web visualization.
"""

import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except Exception:
    Console = None
    Panel = None
    Table = None
    _RICH_AVAILABLE = False

from src.economics import run_economic_analysis
from src.economics.analysis import AnalysisResults
from src.economics.cumulative_impact import CumulativeImpactResult
from src.utils import make_json_safe


POINT_TILE_ZOOM = 5
VECTOR_TILE_MIN_ZOOM = 0
VECTOR_TILE_MAX_ZOOM = 8
SITE_METRIC_FIELDS = (
    "value_loss",
    "loss_fraction",
    "coral_change",
    "annual_loss",
    "cumulative_loss",
    "cumulative_loss_fraction",
)
POLYGON_COORD_DECIMALS = 5
POINT_COORD_DECIMALS = 6
RETRYABLE_WRITE_ERRNOS = {5, 35, 54, 89}
_CONSOLE = Console() if _RICH_AVAILABLE else None


def _is_point_geometry(geom_dict: dict) -> bool:
    geom_type = (geom_dict or {}).get("type", "")
    return geom_type in {"Point", "MultiPoint"}


def _quantize_geometry(geom_dict: dict, decimals: int = POLYGON_COORD_DECIMALS) -> dict:
    """Round coordinates to shrink JSON while preserving map fidelity."""
    if not geom_dict or "coordinates" not in geom_dict:
        return geom_dict

    def _round_coords(coords):
        if isinstance(coords, (list, tuple)) and coords and isinstance(
            coords[0], (list, tuple)
        ):
            return [_round_coords(c) for c in coords]
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return [round(float(coords[0]), decimals), round(float(coords[1]), decimals)]
        return coords

    return {
        "type": geom_dict.get("type"),
        "coordinates": _round_coords(geom_dict.get("coordinates")),
    }


def _quantize_for_type(geom_dict: dict) -> dict:
    geom_type = (geom_dict or {}).get("type", "")
    decimals = POINT_COORD_DECIMALS if geom_type in {"Point", "MultiPoint"} else POLYGON_COORD_DECIMALS
    return _quantize_geometry(geom_dict, decimals=decimals)


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _point_tile_key_from_geom(geom_dict: dict, zoom: int) -> str:
    coords = (geom_dict or {}).get("coordinates")
    if not coords:
        return f"{zoom}/0/0"

    # For MultiPoint, use centroid-ish average to assign a stable tile.
    if (geom_dict or {}).get("type") == "MultiPoint" and isinstance(coords, list) and coords:
        lons = [float(c[0]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        lats = [float(c[1]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        lon = float(np.mean(lons)) if lons else 0.0
        lat = float(np.mean(lats)) if lats else 0.0
    else:
        lon = float(coords[0])
        lat = float(coords[1])

    x, y = _lonlat_to_tile(lon, lat, zoom)
    return f"{zoom}/{x}/{y}"


def _write_point_chunks_for_scenario(
    output_dir: Path,
    scenario_key: str,
    point_features: list[dict],
    tile_zoom: int = POINT_TILE_ZOOM,
) -> dict:
    """Write point-only chunk files and return tile->filename mapping."""
    tiles: Dict[str, list[dict]] = {}
    for feature in point_features:
        tile_key = _point_tile_key_from_geom(feature.get("geometry", {}), tile_zoom)
        tiles.setdefault(tile_key, []).append(feature)

    tile_files: Dict[str, str] = {}
    safe_key = scenario_key.replace("/", "_")
    for tile_key, features in tiles.items():
        z, x, y = tile_key.split("/")
        filename = f"sites_chunk_{safe_key}_z{z}_x{x}_y{y}.json"
        payload = {"type": "FeatureCollection", "features": features}
        _write_json_file(output_dir / filename, payload, separators=(",", ":"))
        tile_files[tile_key] = filename

    return tile_files


def _empty_site_metric_record() -> Dict[str, float]:
    return {k: 0.0 for k in SITE_METRIC_FIELDS}


def _write_json_file(
    path: Path,
    payload: Any,
    *,
    indent: Optional[int] = None,
    separators: Optional[tuple[str, str]] = None,
    retries: int = 5,
) -> None:
    """Write JSON atomically with retries for transient filesystem errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")

    for attempt in range(retries):
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=indent, separators=separators)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            is_retryable = exc.errno in RETRYABLE_WRITE_ERRNOS
            if (not is_retryable) or attempt == retries - 1:
                raise
            time.sleep(0.2 * (2**attempt))


def _log_section(title: str) -> None:
    if _RICH_AVAILABLE and _CONSOLE is not None:
        _CONSOLE.print(Panel.fit(title, style="bold cyan"))
    else:
        print(f"\n=== {title} ===")


def _log_site_summary(mode: str, stats_by_dataset: Dict[str, Dict[str, int]]) -> None:
    if _RICH_AVAILABLE and _CONSOLE is not None:
        table = Table(title=f"Site export summary ({mode})")
        table.add_column("Dataset", style="bold")
        table.add_column("Scenarios", justify="right")
        table.add_column("Polygons", justify="right")
        table.add_column("Points", justify="right")
        for dataset in sorted(stats_by_dataset.keys()):
            row = stats_by_dataset[dataset]
            table.add_row(
                dataset,
                f"{row['scenarios']:,}",
                f"{row['polygons']:,}",
                f"{row['points']:,}",
            )
        _CONSOLE.print(table)
        return

    print(f"Site export summary ({mode})")
    for dataset in sorted(stats_by_dataset.keys()):
        row = stats_by_dataset[dataset]
        print(
            f" - {dataset}: scenarios={row['scenarios']}, "
            f"polygons={row['polygons']}, points={row['points']}"
        )


def _ensure_site_metric_columns(
    metric_columns: Dict[str, list[float]],
    size: int,
) -> Dict[str, list[float]]:
    """Ensure all metric arrays exist and have exactly `size` entries."""
    for metric in SITE_METRIC_FIELDS:
        values = metric_columns.setdefault(metric, [])
        if len(values) < size:
            values.extend([0.0] * (size - len(values)))
    return metric_columns


def _append_site_defaults(
    scenario_metrics: Dict[str, Dict[str, list[float]]],
) -> None:
    """Append a default value for all metrics in all scenarios."""
    for metric_columns in scenario_metrics.values():
        metric_columns = _ensure_site_metric_columns(
            metric_columns,
            len(next(iter(metric_columns.values()))) if metric_columns else 0,
        )
        for metric in SITE_METRIC_FIELDS:
            metric_columns[metric].append(0.0)


def _build_tile_geometry_key(geom_dict: dict) -> str:
    geom_type = (geom_dict or {}).get("type", "")
    coordinates = (geom_dict or {}).get("coordinates", [])
    return f"{geom_type}:{json.dumps(coordinates, separators=(',', ':'))}"


def _scenario_key_from_result(
    result: Any,
    *,
    cumulative: bool = False,
    rcp: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    if cumulative:
        if rcp is None or year is None:
            raise ValueError("cumulative scenario key requires rcp and year")
        raw = f"{result.value_type}_cumulative_{rcp}_{year}_{result.model.name}"
    else:
        raw = f"{result.value_type}_{result.scenario}_{result.model.name}"
    return _sanitize_key(raw)


def _write_dataset_wide_point_tiles(
    *,
    output_dir: Path,
    mode: str,
    tile_payloads: Dict[str, Dict[str, dict]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Write point tiles once per dataset, with scenario/model metrics as columns.
    """
    dataset_manifest: Dict[str, Dict[str, Dict[str, str]]] = {}

    for dataset_key, tiles in tile_payloads.items():
        geometry_index: Dict[str, str] = {}
        attribute_index: Dict[str, str] = {}
        safe_dataset = _sanitize_key(dataset_key)

        for tile_key, tile_data in tiles.items():
            z, x, y = tile_key.split("/")
            geom_filename = (
                f"sites_dataset_{mode}_{safe_dataset}_geom_z{z}_x{x}_y{y}.json"
            )
            attr_filename = (
                f"sites_dataset_{mode}_{safe_dataset}_attrs_z{z}_x{x}_y{y}.json"
            )

            geometry_payload = {
                "type": "FeatureCollection",
                "value_type": dataset_key,
                "tile_key": tile_key,
                "features": tile_data["features"],
            }
            _write_json_file(
                output_dir / geom_filename,
                geometry_payload,
                separators=(",", ":"),
            )

            attribute_payload = {
                "tile_key": tile_key,
                "scenario_metrics": tile_data["scenario_metrics"],
            }
            _write_json_file(
                output_dir / attr_filename,
                attribute_payload,
                separators=(",", ":"),
            )

            geometry_index[tile_key] = geom_filename
            attribute_index[tile_key] = attr_filename

        dataset_manifest[dataset_key] = {
            "geometry": geometry_index,
            "attributes": attribute_index,
        }

    return dataset_manifest


def _upsert_point_tile_entry(
    *,
    tiles_for_dataset: Dict[str, dict],
    tile_key: str,
    geometry: dict,
    static_props: Dict[str, Any],
    scenario_key: str,
    metric_record: Dict[str, float],
) -> int:
    tile_data = tiles_for_dataset.setdefault(
        tile_key,
        {
            "features": [],
            "site_index": {},
            "scenario_metrics": {},
        },
    )

    geom_key = _build_tile_geometry_key(geometry)
    site_index = tile_data["site_index"]
    feature_list = tile_data["features"]

    if geom_key not in site_index:
        site_id = f"{tile_key}:{len(feature_list)}"
        site_index[geom_key] = len(feature_list)
        feature_list.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "site_id": site_id,
                    **static_props,
                },
            }
        )
        _append_site_defaults(tile_data["scenario_metrics"])

    idx = site_index[geom_key]
    scenario_metrics = tile_data["scenario_metrics"]
    metric_columns = scenario_metrics.setdefault(scenario_key, {})
    metric_columns = _ensure_site_metric_columns(metric_columns, len(feature_list))
    for metric in SITE_METRIC_FIELDS:
        metric_columns[metric][idx] = float(metric_record.get(metric, 0.0))
    scenario_metrics[scenario_key] = metric_columns
    return idx


def _ensure_site_cache_for_dataset(
    *,
    result: Any,
    gdf: Any,
    original_value: np.ndarray,
    dataset_tile_payloads: Dict[str, Dict[str, dict]],
) -> Dict[str, Any]:
    """Build static per-dataset geometry cache once, reused across scenarios."""
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    try:
        country_col = result._get_country_column()
        countries = gdf_wgs84[country_col].fillna("").astype(str).to_numpy()
    except Exception:
        countries = np.full(len(gdf_wgs84), "", dtype=object)

    tiles_for_dataset = dataset_tile_payloads.setdefault(result.value_type, {})
    point_indices: list[int] = []
    point_site_refs: list[tuple[str, int]] = []
    polygon_indices: list[int] = []
    polygon_static_rows: list[Dict[str, Any]] = []

    for idx, geom in enumerate(gdf_wgs84.geometry.values):
        if geom.is_empty:
            continue
        geom_dict = _quantize_for_type(geom.__geo_interface__)
        static_props = {
            "country": countries[idx],
            "value_type": result.value_type,
            "original_value": float(original_value[idx]),
        }
        if _is_point_geometry(geom_dict):
            tile_key = _point_tile_key_from_geom(geom_dict, POINT_TILE_ZOOM)
            site_idx = _upsert_point_tile_entry(
                tiles_for_dataset=tiles_for_dataset,
                tile_key=tile_key,
                geometry=geom_dict,
                static_props=static_props,
                scenario_key="__init__",
                metric_record=_empty_site_metric_record(),
            )
            point_indices.append(idx)
            point_site_refs.append((tile_key, site_idx))
        else:
            polygon_indices.append(idx)
            polygon_static_rows.append(
                {
                    "geometry": geom_dict,
                    "country": countries[idx],
                    "value_type": result.value_type,
                    "original_value": float(original_value[idx]),
                }
            )

    # Remove placeholder initializer scenario.
    for tile_data in tiles_for_dataset.values():
        tile_data.get("scenario_metrics", {}).pop("__init__", None)

    return {
        "point_indices": np.array(point_indices, dtype=np.int64),
        "point_site_refs": point_site_refs,
        "polygon_indices": np.array(polygon_indices, dtype=np.int64),
        "polygon_static_rows": polygon_static_rows,
    }


def _apply_point_metrics_for_scenario(
    *,
    tiles_for_dataset: Dict[str, dict],
    scenario_key: str,
    point_indices: np.ndarray,
    point_site_refs: list[tuple[str, int]],
    metric_arrays: Dict[str, np.ndarray],
) -> int:
    if len(point_site_refs) == 0:
        return 0

    prepared_columns: Dict[str, Dict[str, list[float]]] = {}
    for tile_key, _ in point_site_refs:
        if tile_key in prepared_columns:
            continue
        tile_data = tiles_for_dataset[tile_key]
        feature_count = len(tile_data["features"])
        scenario_metrics = tile_data["scenario_metrics"]
        metric_columns = scenario_metrics.setdefault(scenario_key, {})
        prepared_columns[tile_key] = _ensure_site_metric_columns(
            metric_columns, feature_count
        )

    for row_idx, (tile_key, site_idx) in zip(point_indices, point_site_refs):
        metric_columns = prepared_columns[tile_key]
        for metric in SITE_METRIC_FIELDS:
            values = metric_arrays.get(metric)
            metric_columns[metric][site_idx] = (
                float(values[row_idx]) if values is not None else 0.0
            )
    return len(point_site_refs)


def _sanitize_key(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
    )


def _ogr2ogr_available() -> bool:
    return shutil.which("ogr2ogr") is not None


def _export_geojson_to_vector_tiles(
    geojson_path: Path,
    tiles_dir: Path,
    layer_name: str = "sites",
    min_zoom: int = VECTOR_TILE_MIN_ZOOM,
    max_zoom: int = VECTOR_TILE_MAX_ZOOM,
) -> bool:
    """
    Convert a GeoJSON FeatureCollection into MVT XYZ tiles with ogr2ogr.

    Output structure:
        <tiles_dir>/{z}/{x}/{y}.pbf
    """
    if not _ogr2ogr_available():
        return False

    # GDAL's MVT driver fails if output directory already exists.
    # Recreate per-scenario tile directory to make reruns deterministic.
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir, ignore_errors=True)
    tiles_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ogr2ogr",
        "-f",
        "MVT",
        str(tiles_dir),
        str(geojson_path),
        "-dsco",
        f"MINZOOM={min_zoom}",
        "-dsco",
        f"MAXZOOM={max_zoom}",
        "-lco",
        f"NAME={layer_name}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"⚠️  Vector tile export failed for {geojson_path.name}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
        return False
    return True


def _build_vector_tile_manifest_entries(
    output_dir: Path,
    scenario_geojson_files: Dict[str, Path],
    is_cumulative: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Build vector tiles for scenario GeoJSON files and return manifest entries.
    """
    entries: Dict[str, Dict[str, str]] = {}
    if not scenario_geojson_files or not _ogr2ogr_available():
        return entries

    root = output_dir / "vector_tiles"
    root.mkdir(parents=True, exist_ok=True)

    def _build_one(item):
        scenario_key, geojson_path = item
        scenario_safe = _sanitize_key(scenario_key)
        tiles_dir = root / scenario_safe
        ok = _export_geojson_to_vector_tiles(
            geojson_path=geojson_path,
            tiles_dir=tiles_dir,
            layer_name="sites",
        )
        if not ok:
            return scenario_key, None
        return scenario_key, {
            "format": "mvt",
            "layer": "sites",
            "url_template": f"vector_tiles/{scenario_safe}" + "/{z}/{x}/{y}.pbf",
            "min_zoom": VECTOR_TILE_MIN_ZOOM,
            "max_zoom": VECTOR_TILE_MAX_ZOOM,
            "kind": "cumulative" if is_cumulative else "annual",
        }

    max_workers = min(4, len(scenario_geojson_files))
    with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
        futures = [
            executor.submit(_build_one, item)
            for item in scenario_geojson_files.items()
        ]
        for future in as_completed(futures):
            scenario_key, entry = future.result()
            if entry is not None:
                entries[scenario_key] = entry

    print(f"Exported vector tiles for {len(entries)} scenarios")
    return entries


def export_country_results(
    results: AnalysisResults,
    output_dir: Path,
) -> None:
    """Export country-level results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting country-level results to {output_dir}")
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
                    "value_type": result.value_type,
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

    _write_json_file(output_dir / "country_results.json", all_countries, indent=2)

    print(f"Exported country results: {len(all_countries)} records")


def export_cumulative_country_results(
    results: AnalysisResults,
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
) -> None:
    """Export cumulative country-level results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting cumulative country-level results to {output_dir}")

    # Create lookup for cumulative results by dataset, scenario, year, and model.
    cumulative_lookup = {}
    for _, cum_result in cumulative_results.items():
        interpolation = cum_result.trajectory.interpolation_method
        if interpolation != "linear":
            continue
        scenario = cum_result.trajectory.scenario.lower()
        model_name = cum_result.model.name
        years = cum_result.years

        for target_year in [2050, 2100]:
            if target_year in years:
                idx = np.where(years == target_year)[0]
                if len(idx) > 0:
                    lookup_key = (
                        cum_result.value_type,
                        scenario,
                        target_year,
                        model_name,
                    )
                    cumulative_lookup[lookup_key] = cum_result.cumulative_losses[idx[0]]

    all_countries = []

    for _, result in results.results.items():
        by_country = result.by_country.copy()

        # Get ISO codes
        if "iso_a3" not in by_country.columns and "iso_a3" in result.gdf.columns:
            country_col = result._get_country_column()
            iso_map = result.gdf.groupby(country_col)["iso_a3"].first()
            by_country["iso_a3"] = by_country[country_col].map(iso_map)

        # Parse scenario
        scenario = result.scenario.lower()
        rcp = "rcp45" if "45" in scenario else "rcp85"
        year = 2050 if "2050" in scenario else 2100
        model_name = result.model.name

        # Get cumulative loss for this scenario/model/year
        lookup_key = (result.value_type, rcp, year, model_name)
        total_cumulative = cumulative_lookup.get(lookup_key, None)

        for _, row in by_country.iterrows():
            # Calculate cumulative loss proportionally
            country_annual_loss = float(row.get("value_loss", 0))
            total_annual_loss = result.total_loss

            if total_cumulative is not None and total_annual_loss > 0:
                country_cumulative_loss = (
                    country_annual_loss / total_annual_loss
                ) * total_cumulative
                original_value = float(row.get("original_value", 1))
                if not np.isnan(original_value) and original_value > 0:
                    country_cumulative_fraction = (
                        country_cumulative_loss / original_value
                    )
                else:
                    country_cumulative_fraction = 0
            else:
                country_cumulative_loss = 0
                country_cumulative_fraction = 0

            all_countries.append(
                {
                    "value_type": result.value_type,
                    "scenario": f"cumulative_{rcp}_{year}",
                    "model": result.model.name,
                    "country": row.get("country", row.iloc[0]),
                    "iso_a3": row.get("iso_a3", ""),
                    "original_value": make_json_safe(row.get("original_value", 0)),
                    "cumulative_loss": make_json_safe(country_cumulative_loss),
                    "cumulative_loss_fraction": make_json_safe(
                        country_cumulative_fraction
                    ),
                    "annual_loss": make_json_safe(country_annual_loss),
                    "loss_fraction": make_json_safe(row.get("loss_fraction", 0)),
                }
            )

    _write_json_file(
        output_dir / "cumulative_country_results.json",
        all_countries,
        indent=2,
    )

    print(f"Exported cumulative country results: {len(all_countries)} records")


def export_cumulative_site_results(
    results: AnalysisResults,
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
    sample_fraction: float = 1,
) -> None:
    """Export site-level cumulative results for point visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_section(f"Exporting cumulative site-level results to {output_dir}")
    exported_scenarios: list[str] = []
    dataset_tile_payloads: Dict[str, Dict[str, dict]] = {}
    dataset_site_cache: Dict[str, Dict[str, Any]] = {}
    stats_by_dataset: Dict[str, Dict[str, int]] = {}
    warned_large_datasets: set[str] = set()

    # Create lookup for cumulative results by dataset, scenario, model, and year.
    cumulative_lookup = {}
    for _, cum_result in cumulative_results.items():
        interpolation = cum_result.trajectory.interpolation_method
        if interpolation != "linear":
            continue

        scenario = cum_result.trajectory.scenario.lower()
        model_name = cum_result.model.name
        years = cum_result.years
        for target_year in [2050, 2100]:
            if target_year in years:
                idx = np.where(years == target_year)[0]
                if len(idx) > 0:
                    lookup_key = (
                        cum_result.value_type,
                        scenario,
                        target_year,
                        model_name,
                    )
                    cumulative_lookup[lookup_key] = cum_result.cumulative_losses[idx[0]]

    items = list(results.results.items())
    for _, result in items:
        gdf = result.gdf.copy()
        if (
            gdf.shape[0] > 100000
            and sample_fraction == 1
            and result.value_type not in warned_large_datasets
        ):
            print(
                f"{result.value_type}: dataframe has {gdf.shape[0]:,} rows. "
                "Subsampling is recommended for faster export and smaller artifacts."
            )
            warned_large_datasets.add(result.value_type)
        if sample_fraction < 1.0:
            gdf = gdf.sample(frac=sample_fraction, random_state=42)

        scenario = result.scenario.lower()
        rcp = "rcp45" if "45" in scenario else "rcp85"
        year = 2050 if "2050" in scenario else 2100
        model_name = result.model.name
        scenario_key = (
            f"{result.value_type}_cumulative_{rcp}_{year}_{result.model.name}".replace(
                " ", "_"
            )
            .replace("/", "_")
            .replace("%", "pct")
            .replace("(", "")
            .replace(")", "")
        )

        lookup_key = (result.value_type, rcp, year, model_name)
        total_cumulative = cumulative_lookup.get(lookup_key, None)
        total_annual_loss = result.total_loss

        annual_loss = gdf.get("value_loss", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        original_value = gdf.get("original_value", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        loss_fraction = gdf.get("loss_fraction", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        coral_change = gdf.get("coral_change", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        if total_cumulative is not None and total_annual_loss > 0:
            cumulative_loss = (annual_loss / total_annual_loss) * total_cumulative
            with np.errstate(divide="ignore", invalid="ignore"):
                cumulative_fraction = np.where(original_value > 0, cumulative_loss / original_value, 0.0)
        else:
            cumulative_loss = np.zeros(len(gdf), dtype=float)
            cumulative_fraction = np.zeros(len(gdf), dtype=float)

        cache = dataset_site_cache.get(result.value_type)
        if cache is None:
            cache = _ensure_site_cache_for_dataset(
                result=result,
                gdf=gdf,
                original_value=original_value,
                dataset_tile_payloads=dataset_tile_payloads,
            )
            dataset_site_cache[result.value_type] = cache

        n_points = _apply_point_metrics_for_scenario(
            tiles_for_dataset=dataset_tile_payloads.setdefault(result.value_type, {}),
            scenario_key=scenario_key,
            point_indices=cache["point_indices"],
            point_site_refs=cache["point_site_refs"],
            metric_arrays={
                "value_loss": annual_loss,
                "loss_fraction": loss_fraction,
                "coral_change": coral_change,
                "annual_loss": annual_loss,
                "cumulative_loss": cumulative_loss,
                "cumulative_loss_fraction": cumulative_fraction,
            },
        )

        polygon_features = []
        for idx, static_row in zip(cache["polygon_indices"], cache["polygon_static_rows"]):
            polygon_features.append(
                {
                    "type": "Feature",
                    "geometry": static_row["geometry"],
                    "properties": {
                        "country": static_row["country"],
                        "value_type": static_row["value_type"],
                        "original_value": static_row["original_value"],
                        "cumulative_loss": float(cumulative_loss[idx]),
                        "cumulative_loss_fraction": float(cumulative_fraction[idx]),
                        "annual_loss": float(annual_loss[idx]),
                        "loss_fraction": float(loss_fraction[idx]),
                        "coral_change": float(coral_change[idx]),
                    },
                }
            )

        geojson = {
            "type": "FeatureCollection",
            "value_type": result.value_type,
            "scenario": f"cumulative_{rcp}_{year}",
            "model": result.model.name,
            "features": polygon_features,
        }
        geojson_path = output_dir / f"sites_{scenario_key}.json"
        _write_json_file(geojson_path, geojson, separators=(",", ":"))
        exported_scenarios.append(scenario_key)
        stat = stats_by_dataset.setdefault(
            result.value_type, {"scenarios": 0, "polygons": 0, "points": 0}
        )
        stat["scenarios"] += 1
        stat["polygons"] += len(polygon_features)
        stat["points"] += n_points

    # Update manifest
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"scenarios": [], "generated": datetime.now().isoformat()}

    manifest["cumulative_scenarios"] = sorted(exported_scenarios)
    dataset_tile_manifest = _write_dataset_wide_point_tiles(
        output_dir=output_dir,
        mode="cumulative",
        tile_payloads=dataset_tile_payloads,
    )
    if dataset_tile_manifest:
        manifest["site_dataset_tiles_cumulative"] = dataset_tile_manifest
        manifest["site_dataset_tile_zoom_cumulative"] = POINT_TILE_ZOOM
    _write_json_file(manifest_path, manifest, indent=2)
    _log_site_summary("cumulative", stats_by_dataset)


def export_site_results(
    results: AnalysisResults,
    output_dir: Path,
    sample_fraction: float = 0.1,
) -> None:
    """Export site-level results to GeoJSON for polygon visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_section(f"Exporting site-level results to {output_dir}")
    exported_scenarios: list[str] = []
    dataset_tile_payloads: Dict[str, Dict[str, dict]] = {}
    dataset_site_cache: Dict[str, Dict[str, Any]] = {}
    stats_by_dataset: Dict[str, Dict[str, int]] = {}

    items = list(results.results.items())
    for _, result in items:
        gdf = result.gdf.copy()
        if sample_fraction < 1.0:
            gdf = gdf.sample(frac=sample_fraction, random_state=42)

        scenario_key = _scenario_key_from_result(result)

        original_value = gdf.get("original_value", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        value_loss = gdf.get("value_loss", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        loss_fraction = gdf.get("loss_fraction", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        coral_change = gdf.get("coral_change", pd.Series(0, index=gdf.index)).fillna(0).astype(float).to_numpy()
        cache = dataset_site_cache.get(result.value_type)
        if cache is None:
            cache = _ensure_site_cache_for_dataset(
                result=result,
                gdf=gdf,
                original_value=original_value,
                dataset_tile_payloads=dataset_tile_payloads,
            )
            dataset_site_cache[result.value_type] = cache

        n_points = _apply_point_metrics_for_scenario(
            tiles_for_dataset=dataset_tile_payloads.setdefault(result.value_type, {}),
            scenario_key=scenario_key,
            point_indices=cache["point_indices"],
            point_site_refs=cache["point_site_refs"],
            metric_arrays={
                "value_loss": value_loss,
                "loss_fraction": loss_fraction,
                "coral_change": coral_change,
                "annual_loss": value_loss,
                "cumulative_loss": np.zeros(len(gdf), dtype=float),
                "cumulative_loss_fraction": np.zeros(len(gdf), dtype=float),
            },
        )

        polygon_features = []
        for idx, static_row in zip(cache["polygon_indices"], cache["polygon_static_rows"]):
            polygon_features.append(
                {
                    "type": "Feature",
                    "geometry": static_row["geometry"],
                    "properties": {
                        "country": static_row["country"],
                        "value_type": static_row["value_type"],
                        "original_value": static_row["original_value"],
                        "value_loss": float(value_loss[idx]),
                        "loss_fraction": float(loss_fraction[idx]),
                        "coral_change": float(coral_change[idx]),
                    },
                }
            )

        geojson = {
            "type": "FeatureCollection",
            "value_type": result.value_type,
            "scenario": result.scenario,
            "model": result.model.name,
            "features": polygon_features,
        }
        geojson_path = output_dir / f"sites_{scenario_key}.json"
        _write_json_file(geojson_path, geojson, separators=(",", ":"))
        exported_scenarios.append(scenario_key)
        stat = stats_by_dataset.setdefault(
            result.value_type, {"scenarios": 0, "polygons": 0, "points": 0}
        )
        stat["scenarios"] += 1
        stat["polygons"] += len(polygon_features)
        stat["points"] += n_points

    # Save manifest
    manifest = {
        "scenarios": sorted(exported_scenarios),
        "generated": datetime.now().isoformat(),
    }
    dataset_tile_manifest = _write_dataset_wide_point_tiles(
        output_dir=output_dir,
        mode="annual",
        tile_payloads=dataset_tile_payloads,
    )
    if dataset_tile_manifest:
        manifest["site_dataset_tiles_annual"] = dataset_tile_manifest
        manifest["site_dataset_tile_zoom_annual"] = POINT_TILE_ZOOM
    _write_json_file(output_dir / "manifest.json", manifest, indent=2)
    _log_site_summary("annual", stats_by_dataset)


def export_trajectory_data(
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
) -> None:
    """Export trajectory data for time series visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting trajectory data to {output_dir}")
    trajectories = []

    for key, result in cumulative_results.items():
        traj = result.trajectory
        trajectories.append(
            {
                "key": key,
                "value_type": result.value_type,
                "scenario": traj.scenario,
                "interpolation": traj.interpolation_method,
                "model": result.model.name,
                "years": traj.years.tolist(),
                "coral_cover": (traj.covers * 100).tolist(),  # as percentage
                "annual_value": (result.annual_values / 1e9).tolist(),  # billions
                "annual_loss": (result.annual_losses / 1e9).tolist(),  # billions
                "annual_value_lost": (
                    result.annual_value_lost / 1e9
                ).tolist(),  # billions - value lost each year (year-over-year decline)
                "annual_opportunity_cost": (
                    result.annual_opportunity_cost / 1e9
                ).tolist(),  # billions - opportunity cost (baseline revenue lost)
                "cumulative_loss": (
                    result.cumulative_losses / 1e12
                ).tolist(),  # trillions
                "baseline_value": result.baseline_value / 1e9,
                "total_cumulative_loss": result.total_cumulative_loss / 1e12,
            }
        )

    _write_json_file(output_dir / "trajectories.json", trajectories, indent=2)

    print(f"Exported trajectory data: {len(trajectories)} scenarios")


def export_summary_stats(
    results: AnalysisResults,
    cumulative_results: Dict[str, CumulativeImpactResult],
    output_dir: Path,
) -> None:
    """Export summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting summary statistics to {output_dir}")
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
                "value_type": result.value_type,
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
                "value_type": result.value_type,
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

    _write_json_file(output_dir / "summary.json", summary, indent=2)

    print("Exported summary statistics")


def export_model_comparison(output_dir: Path) -> None:
    """Export depreciation model curves for visualization."""
    from src.economics.depreciation_models import get_model

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting model comparison to {output_dir}")
    # Standard models (linear and compound)
    models = {
        "linear": get_model("linear"),
        "compound": get_model("compound"),
    }

    # Generate curves for standard models
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

    # Export tipping point model data with multiple original_cc values
    tipping_point_model = get_model("tipping_point")
    threshold = getattr(tipping_point_model, "threshold_cc", 0.1)
    original_cc_values = [0.1, 0.3, 0.5, 0.7]  # Different starting coral cover levels

    tipping_point_curves = {}
    for og_cc in original_cc_values:
        remaining = [
            tipping_point_model.calculate(
                d / 100, baseline, original_cc=og_cc, threshold=threshold
            )
            for d in delta_cc_range
        ]
        tipping_point_curves[f"tipping_point_{og_cc}"] = {
            "name": f"{int(og_cc * 100)}% initial cover",
            "delta_cc": delta_cc_range.tolist(),
            "remaining_value": remaining,
            "loss_pct": [(baseline - r) for r in remaining],
            "original_cc": og_cc,
        }

    # Combine all curves
    all_curves = {**curves, **tipping_point_curves}

    _write_json_file(output_dir / "model_curves.json", all_curves, indent=2)

    print("Exported model comparison curves")


def export_gdp_impact(
    results: AnalysisResults,
    gdp_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export GDP impact data for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting GDP impact to {output_dir}")
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
                    "value_type": result.value_type,
                    "model": result.model.name,
                    "country": country,
                    "iso_a3": iso,
                    "value_loss": value_loss,
                    "national_gdp": national_gdp,
                    "loss_as_gdp_pct": loss_as_gdp_pct,
                }
            )

    _write_json_file(output_dir / "gdp_impacts.json", gdp_impacts, indent=2)

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
    pipeline_results = run_economic_analysis.run_pipeline(
        verbose=False, sample_fraction=sample_fraction
    )

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
