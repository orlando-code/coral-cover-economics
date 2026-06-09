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
POINT_COORD_DECIMALS = 4  # 4 decimals ≈ 11 m precision, fine for web maps
RETRYABLE_WRITE_ERRNOS = {5, 35, 54, 89}

# ---------------------------------------------------------------------------
# Spatial grid aggregation
# ---------------------------------------------------------------------------
# Sites are snapped to a regular lat/lon grid before export.  This reduces
# 810K individual fishery points to ~10-15K occupied 1° cells — a ~50-100×
# reduction in data volume with no meaningful loss for global-scale maps.
#
# Grid cell size is set by the caller (CLI: --cell-resolution in
# run_economic_analysis.py) and written to manifest.json for the frontend.

# ---------------------------------------------------------------------------
# Compact site geometry format
# ---------------------------------------------------------------------------
# Instead of repeating full polygon geometries in every per-scenario GeoJSON
# file, we write geometry ONCE per value_type and store per-scenario metrics
# as compact columnar arrays referencing sites by integer index.
#
# On-disk layout (all in output_dir):
#   sites_geom_{value_type}.json   — [{site_id, lon, lat, country, original_value}, ...]
#   sites_metrics_{mode}_{value_type}.json — {scenario_key: {metric: [float, ...]}, ...}
#   sites_manifest.json            — {value_type: {geom_file, metrics_annual_file, metrics_cumulative_file}}
#
# The per-scenario `sites_{scenario_key}.json` polygon GeoJSONs are NOT written
# in this new format; they were the main source of multi-GB output.
_CONSOLE = Console() if _RICH_AVAILABLE else None


def _is_point_geometry(geom_dict: dict) -> bool:
    geom_type = (geom_dict or {}).get("type", "")
    return geom_type in {"Point", "MultiPoint"}


def _quantize_geometry(geom_dict: dict, decimals: int = POLYGON_COORD_DECIMALS) -> dict:
    """Round coordinates to shrink JSON while preserving map fidelity."""
    if not geom_dict or "coordinates" not in geom_dict:
        return geom_dict

    def _round_coords(coords):
        if (
            isinstance(coords, (list, tuple))
            and coords
            and isinstance(coords[0], (list, tuple))
        ):
            return [_round_coords(c) for c in coords]
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return [
                round(float(coords[0]), decimals),
                round(float(coords[1]), decimals),
            ]
        return coords

    return {
        "type": geom_dict.get("type"),
        "coordinates": _round_coords(geom_dict.get("coordinates")),
    }


def _quantize_for_type(geom_dict: dict) -> dict:
    geom_type = (geom_dict or {}).get("type", "")
    decimals = (
        POINT_COORD_DECIMALS
        if geom_type in {"Point", "MultiPoint"}
        else POLYGON_COORD_DECIMALS
    )
    return _quantize_geometry(geom_dict, decimals=decimals)


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _point_tile_key_from_geom(geom_dict: dict, zoom: int) -> str:
    coords = (geom_dict or {}).get("coordinates")
    if not coords:
        return f"{zoom}/0/0"

    # For MultiPoint, use centroid-ish average to assign a stable tile.
    if (
        (geom_dict or {}).get("type") == "MultiPoint"
        and isinstance(coords, list)
        and coords
    ):
        lons = [
            float(c[0]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2
        ]
        lats = [
            float(c[1]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2
        ]
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


# ---------------------------------------------------------------------------
# Spatial-grid helpers (vectorised)
# ---------------------------------------------------------------------------


def _build_polygon_geojson(
    ref_gdf: Any,
    grid_index: Dict[str, Any],
    cell_resolution: float,
) -> Dict[str, Any]:
    """
    Union the actual polygon geometries within each grid cell and return
    a GeoJSON FeatureCollection whose feature `id` matches the cell index used
    by the companion metrics file.

    Only called for datasets where the GDF contains real Polygon/MultiPolygon
    geometry (i.e. tourism reef polygons).  Cell indices are preserved so the
    JS can look up per-cell metrics by array position.
    """
    from collections import defaultdict

    import shapely.geometry as sg
    from shapely.ops import unary_union

    # Scale simplification with cell size (0.004 × resolution ≈ 0.002° at 0.5°).
    simplify_tolerance = cell_resolution * 0.004

    row_to_cell = grid_index["row_to_cell"]
    cells = grid_index["cells"]
    n_cells = grid_index["n_cells"]

    # Group row indices by cell – list of lists is faster than iloc in a loop
    geom_list = list(ref_gdf.geometry)
    cell_rows: Dict[int, list] = defaultdict(list)
    for ri, ci in enumerate(row_to_cell.tolist()):
        if ci >= 0:
            cell_rows[ci].append(ri)

    features = []
    for ci, cell in enumerate(cells):
        rows = cell_rows.get(ci, [])
        valid_geoms = [
            geom_list[r]
            for r in rows
            if geom_list[r] is not None and not geom_list[r].is_empty
        ]
        if not valid_geoms:
            continue
        merged = valid_geoms[0] if len(valid_geoms) == 1 else unary_union(valid_geoms)
        simplified = merged.simplify(simplify_tolerance, preserve_topology=True)
        if simplified is None or simplified.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "id": ci,
                "geometry": sg.mapping(simplified),
                "properties": {
                    "i": ci,
                    "n": cell["n"],
                    "ov": cell["ov"],
                    "co": cell["co"],
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "geom_type": "polygon",
        "grid_resolution_deg": cell_resolution,
        "n_cells": n_cells,
        "n_features": len(features),
        "features": features,
    }


def _build_grid_index(
    gdf_wgs84: Any,
    countries: np.ndarray,
    original_value: np.ndarray,
    resolution: float,
) -> Dict[str, Any]:
    """
    Snap site centroids to a regular lat/lon grid and return aggregation tables.

    Returns
    -------
    dict with keys:
        cells       : list of dicts {cell_id, lon, lat, n_sites, mean_original_value, country}
        row_to_cell : int32 array, length == len(gdf_wgs84), -1 for empty geometries
        n_cells     : int
    """
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", message=".*geographic CRS.*centroid.*", category=UserWarning
        )
        centroids = gdf_wgs84.geometry.centroid
    raw_lons = centroids.x.to_numpy()
    raw_lats = centroids.y.to_numpy()

    # Integer grid indices
    gx = np.round(raw_lons / resolution).astype(np.int32)
    gy = np.round(raw_lats / resolution).astype(np.int32)
    valid = np.isfinite(raw_lons) & np.isfinite(raw_lats)

    cell_key_list = list(zip(gx.tolist(), gy.tolist()))
    key_to_cell: Dict[tuple, int] = {}
    row_to_cell = np.full(len(gdf_wgs84), -1, dtype=np.int32)

    for row_idx, key in enumerate(cell_key_list):
        if not valid[row_idx]:
            continue
        if key not in key_to_cell:
            key_to_cell[key] = len(key_to_cell)
        row_to_cell[row_idx] = key_to_cell[key]

    n_cells = len(key_to_cell)

    # Aggregate static attributes per cell
    sum_val = np.zeros(n_cells, dtype=np.float64)
    cell_country = [""] * n_cells
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    valid_rows = np.where(row_to_cell >= 0)[0]
    for ri in valid_rows:
        ci = row_to_cell[ri]
        sum_val[ci] += float(original_value[ri])
        cell_counts[ci] += 1
        if not cell_country[ci]:
            cell_country[ci] = str(countries[ri])

    cells: list[dict] = [None] * n_cells  # type: ignore[list-item]
    for (gxi, gyi), ci in key_to_cell.items():
        cells[ci] = {
            "i": ci,
            "lon": round(gxi * resolution, 2),
            "lat": round(gyi * resolution, 2),
            "n": int(cell_counts[ci]),
            "ov": round(float(sum_val[ci] / max(cell_counts[ci], 1)), 2),
            "co": cell_country[ci],
        }

    return {"cells": cells, "row_to_cell": row_to_cell, "n_cells": n_cells}


def _aggregate_metrics_to_grid(
    row_to_cell: np.ndarray,
    n_cells: int,
    metric_arrays: Dict[str, np.ndarray],
) -> Dict[str, list]:
    """
    Vectorised aggregation of per-site metric arrays to grid cells.

    Uses mean aggregation for all metrics (sensible for both fractional and
    monetary values when displayed per-cell on a global map).
    """
    valid = row_to_cell >= 0
    valid_rows = np.where(valid)[0]
    valid_cells = row_to_cell[valid_rows]

    # Counts per cell
    counts = np.bincount(valid_cells, minlength=n_cells).astype(np.float64)

    result: Dict[str, list] = {}
    for metric in SITE_METRIC_FIELDS:
        arr = metric_arrays.get(metric)
        if arr is None:
            result[metric] = [0.0] * n_cells
            continue
        vals = np.array(arr, dtype=np.float64)
        sums = np.bincount(valid_cells, weights=vals[valid_rows], minlength=n_cells)
        means = np.where(counts > 0, sums / counts, 0.0)
        result[metric] = [round(float(v), 6) for v in means]

    return result


def export_gridded_site_results(
    results: "AnalysisResults",
    output_dir: Path,
    mode: str,
    cell_resolution: float,
    cumulative_results: Optional[Dict[str, "CumulativeImpactResult"]] = None,
) -> None:
    """
    Export site data as a spatial grid aggregation.

    For each value_type writes two files:
      sites_grid_{value_type}.json       — grid-cell geometry (once)
      sites_metrics_{mode}_{value_type}.json — per-scenario mean metrics per cell

    Replaces the per-scenario GeoJSON / tile-based approach entirely.
    Typical reduction: 810 K fishery points → ~12 K grid cells at 1° resolution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_resolution = float(cell_resolution)
    if cell_resolution <= 0:
        raise ValueError("cell_resolution must be > 0.")
    _log_section(
        f"Gridded site export [{mode}] → {output_dir} "
        f"(cell_resolution={cell_resolution}°)"
    )

    # Build cumulative lookup if needed
    cumulative_lookup: Dict[tuple, float] = {}
    if cumulative_results and mode == "cumulative":
        for _, cr in cumulative_results.items():
            if cr.trajectory.interpolation_method != "linear":
                continue
            sc = cr.trajectory.scenario.lower()
            mn = cr.model.name
            for ty in [2050, 2100]:
                if ty in cr.years:
                    idx = np.where(cr.years == ty)[0]
                    if len(idx):
                        cumulative_lookup[(cr.value_type, sc, ty, mn)] = (
                            cr.cumulative_losses[idx[0]]
                        )

    # Group by value_type
    by_vt: Dict[str, list] = {}
    for _, result in results.results.items():
        by_vt.setdefault(result.value_type, []).append(result)

    manifest_update: Dict[str, dict] = {}

    for vt, vt_results in by_vt.items():
        safe_vt = _sanitize_key(vt)
        grid_file = f"sites_grid_{safe_vt}.json"
        metrics_file = f"sites_metrics_{mode}_{safe_vt}.json"

        # Build grid index from the first (reference) result's GDF
        ref_result = vt_results[0]
        ref_gdf = ref_result.gdf.to_crs("EPSG:4326")
        n_sites = len(ref_gdf)

        try:
            country_col = ref_result._get_country_column()
            countries = ref_gdf[country_col].fillna("").astype(str).to_numpy()
        except Exception:
            countries = np.full(n_sites, "", dtype=object)

        orig_val = (
            ref_gdf.get("original_value", pd.Series(0, index=ref_gdf.index))
            .fillna(0)
            .astype(float)
            .to_numpy()
        )

        resolution = cell_resolution
        grid_index = _build_grid_index(ref_gdf, countries, orig_val, resolution)
        row_to_cell = grid_index["row_to_cell"]
        n_cells = grid_index["n_cells"]

        # Detect whether this dataset has real polygon geometries (e.g. tourism
        # reef polygons) or just point/centroid data (fisheries, coastal).
        has_polygons = ref_gdf.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        ).any()

        # Write geometry file (always regenerate so format changes take effect)
        geom_path = output_dir / grid_file
        if has_polygons:
            print(f"  {vt}: building polygon union per grid cell …", flush=True)
            geom_data = _build_polygon_geojson(ref_gdf, grid_index, cell_resolution)
            geom_data["value_type"] = vt
            geom_data["n_sites_raw"] = n_sites
        else:
            geom_data = {
                "value_type": vt,
                "grid_resolution_deg": resolution,
                "n_cells": n_cells,
                "n_sites_raw": n_sites,
                "cells": grid_index["cells"],
            }
        _write_json_file(geom_path, geom_data, separators=(",", ":"))

        # Aggregate metrics per scenario
        scenario_metrics: Dict[str, Dict[str, list]] = {}

        for result in vt_results:
            gdf = result.gdf

            annual_loss = (
                gdf.get("value_loss", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            orig_val_r = (
                gdf.get("original_value", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            loss_fraction = (
                gdf.get("loss_fraction", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            coral_change = (
                gdf.get("coral_change", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )

            if mode == "cumulative":
                sc = result.scenario.lower()
                rcp = "rcp45" if "45" in sc else "rcp85"
                yr = 2050 if "2050" in sc else 2100
                total_cum = cumulative_lookup.get((vt, rcp, yr, result.model.name))
                total_ann = result.total_loss
                if total_cum is not None and total_ann > 0:
                    cum_loss = (annual_loss / total_ann) * total_cum
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cum_frac = np.where(orig_val_r > 0, cum_loss / orig_val_r, 0.0)
                else:
                    cum_loss = np.zeros(len(gdf), dtype=float)
                    cum_frac = np.zeros(len(gdf), dtype=float)
                rcp_str = rcp
                skey = _sanitize_key(
                    f"{vt}_cumulative_{rcp_str}_{yr}_{result.model.name}"
                )
                metric_arrays = {
                    "value_loss": annual_loss,
                    "loss_fraction": loss_fraction,
                    "coral_change": coral_change,
                    "annual_loss": annual_loss,
                    "cumulative_loss": cum_loss,
                    "cumulative_loss_fraction": cum_frac,
                }
            else:
                skey = _sanitize_key(f"{vt}_{result.scenario}_{result.model.name}")
                metric_arrays = {
                    "value_loss": annual_loss,
                    "loss_fraction": loss_fraction,
                    "coral_change": coral_change,
                    "annual_loss": annual_loss,
                    "cumulative_loss": np.zeros(len(gdf), dtype=float),
                    "cumulative_loss_fraction": np.zeros(len(gdf), dtype=float),
                }

            # Re-project this GDF's centroids in case sampling differs — but since
            # all results share the same geometry structure, reuse row_to_cell.
            scenario_metrics[skey] = _aggregate_metrics_to_grid(
                row_to_cell, n_cells, metric_arrays
            )

        _write_json_file(
            output_dir / metrics_file,
            {
                "value_type": vt,
                "mode": mode,
                "grid_resolution_deg": resolution,
                "metric_fields": list(SITE_METRIC_FIELDS),
                "scenarios": scenario_metrics,
            },
            separators=(",", ":"),
        )

        print(
            f"  {vt}: {n_cells:,} grid cells (from {n_sites:,} sites), "
            f"{len(scenario_metrics)} scenarios [{mode}]"
        )
        manifest_update[vt] = {
            "grid_file": grid_file,
            "metrics_file": metrics_file,
            "geom_type": "polygon" if has_polygons else "grid",
            "grid_resolution_deg": resolution,
            "n_cells": n_cells,
            "n_sites_raw": n_sites,
            "n_scenarios": len(scenario_metrics),
        }

    # Update manifest
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[f"gridded_sites_{mode}"] = manifest_update
    manifest["cell_resolution_deg"] = cell_resolution
    _write_json_file(manifest_path, manifest, indent=2)


# ---------------------------------------------------------------------------
# Compact geometry+metrics helpers (kept for backward-compat; not used by
# the main export pipeline which now calls export_gridded_site_results)
# ---------------------------------------------------------------------------


def _centroid_lonlat(geom) -> tuple[float, float]:
    """Return (lon, lat) centroid for any geometry type."""
    try:
        c = geom.centroid
        return round(float(c.x), POINT_COORD_DECIMALS), round(
            float(c.y), POINT_COORD_DECIMALS
        )
    except Exception:
        return 0.0, 0.0


def _build_site_geometry_index(
    result: Any,
    gdf: Any,
    original_value: np.ndarray,
) -> Dict[str, Any]:
    """
    Build a stable ordered list of site records (centroid + static attrs) for
    a given value_type.  Returns a dict with:
        sites   : list of {site_id, lon, lat, country, original_value}
        geom_key_to_idx : dict mapping a geometry-string key → integer index
    """
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    try:
        country_col = result._get_country_column()
        countries = gdf_wgs84[country_col].fillna("").astype(str).to_numpy()
    except Exception:
        countries = np.full(len(gdf_wgs84), "", dtype=object)

    sites: list[dict] = []
    geom_key_to_idx: Dict[str, int] = {}

    for idx, geom in enumerate(gdf_wgs84.geometry.values):
        if geom is None or geom.is_empty:
            continue
        lon, lat = _centroid_lonlat(geom)
        geom_key = f"{lon:.4f}:{lat:.4f}"
        if geom_key not in geom_key_to_idx:
            geom_key_to_idx[geom_key] = len(sites)
            sites.append(
                {
                    "site_id": len(sites),
                    "lon": lon,
                    "lat": lat,
                    "country": str(countries[idx]),
                    "original_value": round(float(original_value[idx]), 2),
                }
            )

    return {"sites": sites, "geom_key_to_idx": geom_key_to_idx}


def _build_scenario_metric_arrays(
    gdf: Any,
    geom_key_to_idx: Dict[str, int],
    metric_arrays: Dict[str, np.ndarray],
) -> Dict[str, list[float]]:
    """
    Map per-row metric arrays onto the compact site index.

    Returns {metric: [float, ...]} aligned to the site geometry list.
    """
    n_sites = max(geom_key_to_idx.values()) + 1 if geom_key_to_idx else 0
    result_metrics: Dict[str, list[float]] = {
        m: [0.0] * n_sites for m in SITE_METRIC_FIELDS
    }

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    for row_idx, geom in enumerate(gdf_wgs84.geometry.values):
        if geom is None or geom.is_empty:
            continue
        lon, lat = _centroid_lonlat(geom)
        geom_key = f"{lon:.4f}:{lat:.4f}"
        site_idx = geom_key_to_idx.get(geom_key)
        if site_idx is None:
            continue
        for metric in SITE_METRIC_FIELDS:
            arr = metric_arrays.get(metric)
            if arr is not None and row_idx < len(arr):
                result_metrics[metric][site_idx] = round(float(arr[row_idx]), 6)

    return result_metrics


def export_compact_site_results(
    results: "AnalysisResults",
    output_dir: Path,
    mode: str,
    cumulative_results: Optional[Dict[str, "CumulativeImpactResult"]] = None,
) -> None:
    """
    Write compact site-level data:
      - sites_geom_{value_type}.json  (one per value_type, geometry as centroid)
      - sites_metrics_{mode}_{value_type}.json (per-scenario metric arrays)

    This replaces the per-scenario GeoJSON polygon approach that produced
    multi-gigabyte output for datasets with complex polygon geometries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_section(f"Compact site export [{mode}] → {output_dir}")

    # Build cumulative lookup if needed.
    cumulative_lookup: Dict[tuple, float] = {}
    if cumulative_results and mode == "cumulative":
        for _, cum_result in cumulative_results.items():
            if cum_result.trajectory.interpolation_method != "linear":
                continue
            scenario = cum_result.trajectory.scenario.lower()
            model_name = cum_result.model.name
            years = cum_result.years
            for target_year in [2050, 2100]:
                if target_year in years:
                    idx = np.where(years == target_year)[0]
                    if len(idx) > 0:
                        cumulative_lookup[
                            (cum_result.value_type, scenario, target_year, model_name)
                        ] = cum_result.cumulative_losses[idx[0]]

    # Group results by value_type so we write geometry once per dataset.
    by_value_type: Dict[str, list] = {}
    for _, result in results.results.items():
        by_value_type.setdefault(result.value_type, []).append(result)

    sites_manifest: Dict[str, dict] = {}

    for value_type, vt_results in by_value_type.items():
        safe_vt = _sanitize_key(value_type)
        geom_file = f"sites_geom_{safe_vt}.json"
        metrics_file = f"sites_metrics_{mode}_{safe_vt}.json"

        # Build geometry index from the first result (all share the same sites).
        first_result = vt_results[0]
        gdf0 = first_result.gdf
        original_value0 = (
            gdf0.get("original_value", pd.Series(0, index=gdf0.index))
            .fillna(0)
            .astype(float)
            .to_numpy()
        )
        geom_index = _build_site_geometry_index(first_result, gdf0, original_value0)
        geom_key_to_idx = geom_index["geom_key_to_idx"]

        # Write geometry file (only once per value_type).
        geom_payload = {
            "value_type": value_type,
            "mode": mode,
            "n_sites": len(geom_index["sites"]),
            "sites": geom_index["sites"],
        }
        _write_json_file(output_dir / geom_file, geom_payload, separators=(",", ":"))

        # Accumulate per-scenario metric arrays.
        scenario_metrics: Dict[str, Dict[str, list[float]]] = {}

        for result in vt_results:
            gdf = result.gdf

            annual_loss = (
                gdf.get("value_loss", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            original_value = (
                gdf.get("original_value", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            loss_fraction = (
                gdf.get("loss_fraction", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )
            coral_change = (
                gdf.get("coral_change", pd.Series(0, index=gdf.index))
                .fillna(0)
                .astype(float)
                .to_numpy()
            )

            if mode == "cumulative":
                scenario = result.scenario.lower()
                rcp = "rcp45" if "45" in scenario else "rcp85"
                year = 2050 if "2050" in scenario else 2100
                total_cumulative = cumulative_lookup.get(
                    (value_type, rcp, year, result.model.name)
                )
                total_annual_loss = result.total_loss
                if total_cumulative is not None and total_annual_loss > 0:
                    cumulative_loss = (
                        annual_loss / total_annual_loss
                    ) * total_cumulative
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cumulative_fraction = np.where(
                            original_value > 0, cumulative_loss / original_value, 0.0
                        )
                else:
                    cumulative_loss = np.zeros(len(gdf), dtype=float)
                    cumulative_fraction = np.zeros(len(gdf), dtype=float)

                rcp_str = "rcp45" if "45" in scenario else "rcp85"
                scenario_key = _sanitize_key(
                    f"{value_type}_cumulative_{rcp_str}_{year}_{result.model.name}"
                )
                metric_arrays = {
                    "value_loss": annual_loss,
                    "loss_fraction": loss_fraction,
                    "coral_change": coral_change,
                    "annual_loss": annual_loss,
                    "cumulative_loss": cumulative_loss,
                    "cumulative_loss_fraction": cumulative_fraction,
                }
            else:
                scenario_key = _sanitize_key(
                    f"{value_type}_{result.scenario}_{result.model.name}"
                )
                metric_arrays = {
                    "value_loss": annual_loss,
                    "loss_fraction": loss_fraction,
                    "coral_change": coral_change,
                    "annual_loss": annual_loss,
                    "cumulative_loss": np.zeros(len(gdf), dtype=float),
                    "cumulative_loss_fraction": np.zeros(len(gdf), dtype=float),
                }

            scenario_metrics[scenario_key] = _build_scenario_metric_arrays(
                gdf, geom_key_to_idx, metric_arrays
            )

        # Write metrics file.
        metrics_payload = {
            "value_type": value_type,
            "mode": mode,
            "metric_fields": list(SITE_METRIC_FIELDS),
            "scenarios": scenario_metrics,
        }
        _write_json_file(
            output_dir / metrics_file, metrics_payload, separators=(",", ":")
        )

        sites_manifest[value_type] = {
            "geom_file": geom_file,
            "metrics_file": metrics_file,
            "n_sites": len(geom_index["sites"]),
            "n_scenarios": len(scenario_metrics),
        }

        print(
            f"  {value_type}: {len(geom_index['sites']):,} sites, "
            f"{len(scenario_metrics)} scenarios [{mode}]"
        )

    # Merge into manifest.
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[f"compact_sites_{mode}"] = sites_manifest
    _write_json_file(manifest_path, manifest, indent=2)


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
            executor.submit(_build_one, item) for item in scenario_geojson_files.items()
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
    cell_resolution: float = 0.5,
) -> None:
    """Export cumulative site-level results as a spatial grid aggregation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_section(f"Exporting gridded cumulative site results → {output_dir}")
    export_gridded_site_results(
        results,
        output_dir,
        mode="cumulative",
        cell_resolution=cell_resolution,
        cumulative_results=cumulative_results,
    )
    exported_scenarios = [
        _sanitize_key(
            f"{r.value_type}_cumulative_"
            f"{'rcp45' if '45' in r.scenario.lower() else 'rcp85'}_"
            f"{'2050' if '2050' in r.scenario else '2100'}_{r.model.name}"
        )
        for _, r in results.results.items()
    ]
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["cumulative_scenarios"] = sorted(set(exported_scenarios))
    _write_json_file(manifest_path, manifest, indent=2)


def export_site_results(
    results: AnalysisResults,
    output_dir: Path,
    sample_fraction: float = 0.1,
    cell_resolution: float = 0.5,
) -> None:
    """Export site-level results as a spatial grid aggregation.

    Sites are snapped to a regular lat/lon grid at ``cell_resolution`` degrees
    and metrics averaged per cell.  This replaces the per-scenario GeoJSON
    approach and the tile system.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_section(f"Exporting gridded site results → {output_dir}")
    export_gridded_site_results(
        results, output_dir, mode="annual", cell_resolution=cell_resolution
    )
    # Collect scenario keys for manifest only
    exported_scenarios = [
        _sanitize_key(f"{r.value_type}_{r.scenario}_{r.model.name}")
        for _, r in results.results.items()
    ]
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["scenarios"] = sorted(set(exported_scenarios))
    manifest.setdefault("generated", datetime.now().isoformat())
    _write_json_file(manifest_path, manifest, indent=2)


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
    """Export depreciation model curves and metadata for the models page."""
    from src.economics.depreciation_models import apply_depreciation_model, get_model

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting model comparison to {output_dir}")

    delta_cc_range = np.linspace(-50, 10, 120)  # percentage points
    baseline = 100.0
    reference_cover = 0.35

    def _curve(model, delta_pp, *, value_type="tourism", initial_cover=reference_cover):
        remaining = [
            float(
                apply_depreciation_model(
                    model,
                    d / 100.0,
                    baseline,
                    value_type=value_type,
                    initial_cover=initial_cover,
                    original_cc=initial_cover,
                )
            )
            for d in delta_pp
        ]
        return {
            "delta_cc": delta_pp.tolist(),
            "remaining_value": remaining,
            "loss_pct": [baseline - r for r in remaining],
        }

    linear_model = get_model("linear")
    compound_model = get_model("compound")
    tipping_point_model = get_model("tipping_point")
    threshold = getattr(tipping_point_model, "threshold_cc", 0.1)

    curves: Dict[str, dict] = {}

    curves["linear_tourism"] = {
        "name": "Chen elasticity — tourism",
        "model_key": "linear",
        "value_type": "tourism",
        "initial_cover": reference_cover,
        **_curve(linear_model, delta_cc_range, value_type="tourism"),
    }
    curves["linear_fisheries"] = {
        "name": "Chen elasticity — fisheries",
        "model_key": "linear",
        "value_type": "fisheries",
        "initial_cover": reference_cover,
        **_curve(linear_model, delta_cc_range, value_type="fisheries"),
    }
    curves["linear_coastal"] = {
        "name": "Chen elasticity — coastal protection",
        "model_key": "linear",
        "value_type": "coastal_protection",
        "initial_cover": reference_cover,
        **_curve(linear_model, delta_cc_range, value_type="coastal_protection"),
    }
    curves["compound"] = {
        "name": compound_model.name,
        "model_key": "compound",
        **_curve(compound_model, delta_cc_range),
    }

    # 0.10 is degenerate (initial cover = threshold → any loss triggers collapse).
    # Use covers meaningfully above threshold so each line shows a real cliff.
    for og_cc in (0.15, 0.25, 0.40, 0.60):
        remaining = [
            float(
                tipping_point_model.calculate(
                    d / 100.0,
                    baseline,
                    original_cc=og_cc,
                    threshold=threshold,
                )
            )
            for d in delta_cc_range
        ]
        curves[f"tipping_point_{og_cc}"] = {
            "name": f"{int(og_cc * 100)}% initial cover",
            "model_key": "tipping_point",
            "original_cc": og_cc,
            "delta_cc": delta_cc_range.tolist(),
            "remaining_value": remaining,
            "loss_pct": [baseline - r for r in remaining],
        }

    metadata = {
        "reference_cover_pct": reference_cover * 100,
        "x_axis": "Change in coral cover (ΔC_pp)",
        "y_axis": "Remaining economic value (% of baseline)",
        "models": {
            "chen_elasticity": {
                "title": "Chen et al. elasticity (default “linear” model)",
                "short": "Sector-specific relative-loss functions from Chen et al. (2014/2015).",
                "equations": [
                    "Relative cover change: ΔC/C₀ = (C_final − C₀) / C₀",
                    "Tourism: V_rem = V₀ × max(0, 1 + 3.807 × ΔC/C₀)",
                    "Fisheries & coastal protection: V_rem = V₀ × (1 + ΔC/C₀)",
                ],
                "notes": [
                    "ΔC is the absolute change in live coral cover (proportion); C₀ is site baseline cover.",
                    "Tourism loss scales with relative decline (elasticity ≈ 3.81% value loss per 1% relative cover loss).",
                    "Fisheries and coastal protection follow a 1:1 proportional response to relative cover change.",
                    f"Curves shown at C₀ = {reference_cover * 100:.0f}% reference cover.",
                ],
            },
            "compound": {
                "title": compound_model.name,
                "short": "Sensitivity model: compound loss per absolute percentage-point decline.",
                "equations": [
                    "V_rem = V₀ × (1 − r)^|ΔC_pp|,  r = 0.0381",
                ],
                "notes": [
                    "ΔC_pp is coral cover change in percentage points (not relative).",
                    "Each percentage-point loss multiplies remaining value by (1 − r).",
                    "Used for scenario comparison; not the Chen et al. default.",
                ],
            },
            "tipping_point": {
                "title": tipping_point_model.name,
                "short": "Collapse scenario: gradual compound loss, then catastrophic threshold breach.",
                "equations": [
                    "Pre-threshold: V_rem = V₀ × (1 − r)^|ΔC_pp|",
                    f"Post-threshold (C₀ + ΔC < {threshold * 100:.0f}%): V_rem ← V_rem × (1 − λ),  λ = {tipping_point_model.post_threshold_loss:.0%}",
                ],
                "notes": [
                    "Threshold θ and catastrophic fraction λ are configurable.",
                    "Initial cover C₀ shifts when the reef crosses the tipping threshold.",
                ],
            },
        },
    }

    payload = {"metadata": metadata, "curves": curves}
    _write_json_file(output_dir / "model_curves.json", payload, indent=2)
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


def run_export(
    output_dir: Optional[Path] = None,
    sample_fraction: float = 0.1,
    cell_resolution: float = 0.5,
) -> Path:
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
        verbose=False,
        sample_fraction=sample_fraction,
        cell_resolution=cell_resolution,
    )

    results = pipeline_results["results"]
    cumulative = pipeline_results.get("cumulative", {})
    gdp_data = pipeline_results.get("data", {}).get("gdp")

    # Export all data
    print("\nExporting data...")
    export_country_results(results, output_dir)
    export_site_results(
        results,
        output_dir,
        sample_fraction=sample_fraction,
        cell_resolution=cell_resolution,
    )
    export_trajectory_data(cumulative, output_dir)
    export_summary_stats(results, cumulative, output_dir)
    export_model_comparison(output_dir)

    if gdp_data is not None:
        export_gdp_impact(results, gdp_data, output_dir)

    print(f"\n✓ All data exported to {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_export()
