from __future__ import annotations

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from tqdm import tqdm

_LINE_GEOMS = frozenset({"LineString", "LinearRing", "MultiLineString"})
PROJECTED_CRS = "EPSG:6933"
GEOGRAPHIC_CRS = "EPSG:4326"
DEFAULT_BATCH_SIZE = 10_000

# shape_assign_code values: W=within (unique), M=within (multiple), N=nearest, U=unassigned
SHAPE_ASSIGN_WITHIN = "W"
SHAPE_ASSIGN_WITHIN_AMBIGUOUS = "M"
SHAPE_ASSIGN_NEAREST = "N"
SHAPE_ASSIGN_UNASSIGNED = "U"

BoundarySource = Literal["eez", "land"] | Path


def _geometry_kind(gdf: gpd.GeoDataFrame) -> Literal["point", "line", "other"]:
    types = set(gdf.geometry.geom_type.unique())
    if types <= {"Point", "MultiPoint"}:
        return "point"
    if types <= _LINE_GEOMS:
        return "line"
    return "other"


def _read_geodataframe(source: gpd.GeoDataFrame | Path) -> gpd.GeoDataFrame:
    if isinstance(source, Path):
        if source.is_dir():
            matches = sorted(source.glob("*.shp"))
            if not matches:
                raise FileNotFoundError(f"No shapefile found in {source}")
            source = matches[0]
        return gpd.read_file(source)
    return source


def _ensure_crs(gdf: gpd.GeoDataFrame, default: str = GEOGRAPHIC_CRS) -> gpd.GeoDataFrame:
    return gdf.set_crs(default) if gdf.crs is None else gdf


def _prepare_polygons(
    polygons_gdf: gpd.GeoDataFrame,
    *,
    projected_crs: str = PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    out = _ensure_crs(polygons_gdf.copy())
    out["geometry"] = out.geometry.make_valid()
    return out.to_crs(projected_crs)


def _nearest_indices(
    tree: STRtree,
    geoms: np.ndarray,
    *,
    batch_size: int,
    desc: str,
    verbose: bool,
) -> np.ndarray:
    out = np.empty(len(geoms), dtype=np.intp)
    for start in tqdm(range(0, len(geoms), batch_size), desc=desc, disable=not verbose):
        end = min(start + batch_size, len(geoms))
        nearest = tree.query_nearest(geoms[start:end], all_matches=False)
        tree_ix = nearest[1] if isinstance(nearest, (tuple, np.ndarray)) and np.ndim(nearest) > 1 else nearest
        out[start:end] = np.asarray(tree_ix, dtype=np.intp).reshape(-1)
    return out


def _within_indices(
    tree: STRtree,
    point_geoms: np.ndarray,
    *,
    batch_size: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    assigned = np.full(len(point_geoms), -1, dtype=np.intp)
    ambiguous = np.zeros(len(point_geoms), dtype=bool)
    for start in tqdm(
        range(0, len(point_geoms), batch_size),
        desc="Assigning points (within)",
        disable=not verbose,
    ):
        end = min(start + batch_size, len(point_geoms))
        inp_ix, tree_ix = tree.query(point_geoms[start:end], predicate="within")
        for local_i, shape_i in zip(inp_ix, tree_ix, strict=False):
            global_i = start + int(local_i)
            if assigned[global_i] < 0:
                assigned[global_i] = int(shape_i)
            else:
                ambiguous[global_i] = True
    return assigned, ambiguous


def _assign_points_to_polygons(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    attr_cols: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fill_nearest: bool = True,
    verbose: bool = True,
    assignment_col: str | None = None,
    projected_crs: str = PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    src_crs = points_gdf.crs
    points = _ensure_crs(points_gdf).to_crs(projected_crs)
    polys = _prepare_polygons(polygons_gdf, projected_crs=projected_crs)
    tree = STRtree(polys.geometry.values)
    polygon_ix, ambiguous = _within_indices(
        tree, points.geometry.values, batch_size=batch_size, verbose=verbose
    )
    attr_vals = polys[attr_cols].to_numpy()

    out = points_gdf.copy()
    hit = polygon_ix >= 0
    for col_i, col in enumerate(attr_cols):
        values = np.full(len(out), np.nan, dtype=object)
        values[hit] = attr_vals[polygon_ix[hit], col_i]
        out[col] = values

    if assignment_col:
        codes = np.full(len(out), SHAPE_ASSIGN_UNASSIGNED, dtype=object)
        codes[hit & ~ambiguous] = SHAPE_ASSIGN_WITHIN
        codes[hit & ambiguous] = SHAPE_ASSIGN_WITHIN_AMBIGUOUS

    miss_mask = polygon_ix < 0
    if fill_nearest and miss_mask.any():
        nearest_ix = _nearest_indices(
            tree,
            points.geometry.values[miss_mask],
            batch_size=batch_size,
            desc="Filling unassigned (nearest)",
            verbose=verbose,
        )
        for col_i, col in enumerate(attr_cols):
            out.loc[miss_mask, col] = attr_vals[nearest_ix, col_i]
        if assignment_col:
            codes[miss_mask] = SHAPE_ASSIGN_NEAREST

    if assignment_col:
        out[assignment_col] = codes

    return out.to_crs(src_crs) if src_crs is not None else out


def _attribute_columns(cols: list[str] | None, default: list[str]) -> list[str]:
    """Normalize attribute column names (lowercase, no geometry, no duplicates)."""
    out: list[str] = []
    seen: set[str] = set()
    for col in cols or default:
        name = str(col).lower()
        if name == "geometry" or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def assign_points_to_shapes(
    gdf: gpd.GeoDataFrame,
    shapefile_gdf: gpd.GeoDataFrame | Path,
    cols_to_add: list[str] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
    assignment_col: str | None = "shape_assign_code",
    projected_crs: str = PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    """Assign MEOW (or other) polygon attributes to points.

    Adds ``assignment_col`` (default ``shape_assign_code``) with codes:
    W = within (unique polygon), M = within (multiple polygons; first kept),
    N = nearest fill, U = unassigned.
    Pass ``assignment_col=None`` to omit the column.
    """
    attr_cols = _attribute_columns(
        cols_to_add, ["lat_zone", "realm", "province", "ecoregion"]
    )
    shapes = _read_geodataframe(shapefile_gdf).rename(columns=str.lower)
    missing = [c for c in attr_cols if c not in shapes.columns]
    if missing:
        raise ValueError(f"Polygon layer missing columns: {missing}")
    shapes = shapes[attr_cols + ["geometry"]]
    return _assign_points_to_polygons(
        gdf,
        shapes,
        attr_cols,
        batch_size=batch_size,
        verbose=verbose,
        assignment_col=assignment_col,
        projected_crs=projected_crs,
    )


_BOUNDARY_SPECS: dict[str, dict] = {
    "eez": {
        "path": ("World_EEZ_v12_20231025_LR", "eez_v12_lowres.gpkg"),
        "columns": {"TERRITORY1": "country", "ISO_TER1": "iso_a3"},
    },
    "land": {
        "path": ("ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"),
        "columns": {"NAME": "country", "ISO_A3": "iso_a3"},
    },
}


def load_nation_boundaries(
    source: BoundarySource = "land",
    *,
    geographic_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Load nation polygons with country, iso_a3, and geometry columns."""
    if geographic_dir is None:
        from src import config

        geographic_dir = config.geographic_dir

    if isinstance(source, Path):
        gdf = _ensure_crs(gpd.read_file(source))
        return gdf.rename(columns=str.lower)[["country", "iso_a3", "geometry"]]

    spec = _BOUNDARY_SPECS[str(source)]
    path = geographic_dir.joinpath(*spec["path"])
    cols = list(spec["columns"].keys()) + ["geometry"]
    gdf = _ensure_crs(gpd.read_file(path)[cols].rename(columns=spec["columns"]))
    return gdf.rename(columns=str.lower)


def assign_points_to_region_by_within(
    points_gdf: gpd.GeoDataFrame,
    borders_gdf: gpd.GeoDataFrame,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """Assign points to polygons when the point lies within the polygon."""
    attr_cols = [c for c in borders_gdf.columns if c != "geometry"]
    return _assign_points_to_polygons(
        points_gdf, borders_gdf, attr_cols,
        batch_size=batch_size, fill_nearest=False, verbose=verbose,
    )


def assign_geometries_to_region_by_overlap(
    features_gdf: gpd.GeoDataFrame,
    borders_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Points: within join. Lines: intersects, keeping the longest overlap."""
    if features_gdf.empty:
        return features_gdf.copy()
    if _geometry_kind(features_gdf) == "point":
        return assign_points_to_region_by_within(features_gdf, borders_gdf)

    borders = _prepare_polygons(borders_gdf)
    features = features_gdf.to_crs(PROJECTED_CRS).copy()
    features["geometry"] = features.geometry.make_valid()
    joined = gpd.sjoin(features, borders, how="left", predicate="intersects")
    if joined["index_right"].isna().all() or not joined.index.duplicated(keep=False).any():
        return joined.drop(columns="index_right", errors="ignore").to_crs(features_gdf.crs)

    border_geoms, feat_geoms = borders.geometry, features.geometry
    overlap = np.zeros(len(joined))
    for k, (feat_idx, row) in enumerate(joined.iterrows()):
        border_idx = row["index_right"]
        if pd.notna(border_idx):
            overlap[k] = feat_geoms.loc[feat_idx].intersection(border_geoms.iloc[int(border_idx)]).length

    joined["_overlap_len"] = overlap
    best_idx = joined.groupby(level=0)["_overlap_len"].idxmax()
    return joined.loc[best_idx].drop(columns=["_overlap_len", "index_right"], errors="ignore").to_crs(features_gdf.crs)


def assign_country_by_nearest(
    points_gdf: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame,
    fill_nan_only: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Fill country assignments from the nearest polygon."""
    result = points_gdf.rename(columns=str.lower).copy()
    mask = result.isna().any(axis=1) if fill_nan_only else np.ones(len(result), dtype=bool)
    if not mask.any():
        return result

    geoms = np.asarray([
        g.representative_point() if g.geom_type in _LINE_GEOMS else g
        for g in result.loc[mask, "geometry"].values
    ], dtype=object)
    countries = _prepare_polygons(countries)
    indices = _nearest_indices(
        STRtree(countries.geometry.values), geoms,
        batch_size=batch_size, desc="Assigning nearest country", verbose=verbose,
    )
    for col in ("country", "iso_a3"):
        if col in countries.columns:
            result.loc[mask, col] = countries.iloc[indices][col].values
    return result


def _nation_boundaries(
    boundaries_gdf: gpd.GeoDataFrame | None,
    boundaries: BoundarySource,
    fill_boundaries: BoundarySource | gpd.GeoDataFrame | None,
    fill_unassigned: bool,
    geographic_dir: Path | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    b = boundaries_gdf or load_nation_boundaries(boundaries, geographic_dir=geographic_dir)
    if not fill_unassigned:
        return b, None
    if isinstance(fill_boundaries, gpd.GeoDataFrame):
        return b, fill_boundaries
    return b, load_nation_boundaries(fill_boundaries or "land", geographic_dir=geographic_dir)


def _assign_to_nations(
    features_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    fill_boundaries: gpd.GeoDataFrame | None,
    *,
    fill_unassigned: bool,
    verbose: bool,
    line_mode: bool,
) -> gpd.GeoDataFrame:
    result = (
        assign_geometries_to_region_by_overlap(features_gdf, boundaries_gdf)
        if line_mode
        else assign_points_to_region_by_within(features_gdf, boundaries_gdf)
    )
    check_cols = [c for c in ("country", "iso_a3") if c in result.columns]
    if fill_unassigned and check_cols and fill_boundaries is not None:
        missing = result[check_cols].isna().any(axis=1)
        if missing.any():
            result = assign_country_by_nearest(result, fill_boundaries, verbose=verbose)
    return result


def assign_points_to_nations(
    points_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame | None = None,
    *,
    boundaries: BoundarySource = "eez",
    fill_unassigned: bool = True,
    fill_boundaries: BoundarySource | gpd.GeoDataFrame | None = "land",
    geographic_dir: Path | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Assign points to nations via within join, then nearest fill."""
    b, fill = _nation_boundaries(
        boundaries_gdf, boundaries, fill_boundaries, fill_unassigned, geographic_dir
    )
    return _assign_to_nations(
        points_gdf, b, fill, fill_unassigned=fill_unassigned, verbose=verbose, line_mode=False
    )


def assign_geometries_to_nations(
    features_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame | None = None,
    *,
    boundaries: BoundarySource = "eez",
    fill_unassigned: bool = True,
    fill_boundaries: BoundarySource | gpd.GeoDataFrame | None = "land",
    geographic_dir: Path | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Assign points or linestrings to nations (overlap join + nearest fill)."""
    kind = _geometry_kind(features_gdf)
    if kind == "other":
        raise ValueError(
            "assign_geometries_to_nations supports Point and LineString/MultiLineString "
            f"geometries only; got {set(features_gdf.geometry.geom_type.unique())}"
        )
    b, fill = _nation_boundaries(
        boundaries_gdf, boundaries, fill_boundaries, fill_unassigned, geographic_dir
    )
    return _assign_to_nations(
        features_gdf, b, fill, fill_unassigned=fill_unassigned, verbose=verbose, line_mode=(kind == "line")
    )


def assign_gdfs_to_nations(
    gdfs: list[gpd.GeoDataFrame],
    *,
    boundaries_gdf: gpd.GeoDataFrame | None = None,
    boundaries: BoundarySource = "eez",
    fill_unassigned: bool = True,
    fill_boundaries: BoundarySource | gpd.GeoDataFrame | None = "land",
    geographic_dir: Path | None = None,
    verbose: bool = True,
) -> list[gpd.GeoDataFrame]:
    """Assign multiple GeoDataFrames to nations (shared boundaries, per-layer progress)."""
    b, fill = _nation_boundaries(
        boundaries_gdf, boundaries, fill_boundaries, fill_unassigned, geographic_dir
    )
    out = []
    for gdf in tqdm(gdfs, desc="Assigning nations", disable=not verbose):
        out.append(
            _assign_to_nations(
                gdf, b, fill,
                fill_unassigned=fill_unassigned, verbose=verbose,
                line_mode=_geometry_kind(gdf) == "line",
            )
        )
    return out
