from __future__ import annotations

from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from tqdm import tqdm

_LINE_GEOMS = frozenset({"LineString", "LinearRing", "MultiLineString"})

BoundarySource = Literal["eez", "land"] | Path


def assign_points_to_shapes(
    gdf: gpd.GeoDataFrame,
    shapefile_gdf: gpd.GeoDataFrame,
    cols_to_add: list[str] = ["lat_zone", "realm", "province", "ecoregion"],
) -> gpd.GeoDataFrame:
    """Assign each polygon to a shapefile using centroid spatial join."""
    shapes = shapefile_gdf.rename(columns=str.lower)

    shapes = shapes[cols_to_add + ["geometry"]].copy()
    shapes["geometry"] = shapes.geometry.make_valid()
    projected_crs = "EPSG:6933"
    points = gdf.to_crs(projected_crs).copy()
    shapes_proj = shapes.to_crs(projected_crs)

    # first attempt: sjoin within
    joined = gpd.sjoin(points, shapes_proj, how="left", predicate="within")

    # some points will be assigned to multiple polygons (on boundaries): keep first match
    attrs = joined[cols_to_add] if len(cols_to_add >= 1) else joined[cols_to_add[0]]
    if attrs.index.duplicated().any():
        n_dup = attrs.index.duplicated().sum()
        print(f"Warning: {n_dup:,} duplicate MEOW matches; keeping first per point")
        attrs = attrs.groupby(level=0).first()
    out = gdf.copy()
    for col in cols_to_add:
        out[col] = attrs[col].reindex(out.index)

    # fill problem points due to seam gaps via nearest polygon matching
    miss_ix = out.index[out["ecoregion"].isna()]
    if len(miss_ix):
        filled = gpd.sjoin_nearest(
            points.loc[miss_ix],
            shapes_proj,
            how="left",
            max_distance=1e6,  # metres in EPSG:6933. Set to ensure no nans.
            distance_col="dist_m",
        )
        filled = filled.groupby(level=0).first()
        for col in cols_to_add:
            out.loc[filled.index, col] = filled[col]
    return out


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


def _geometry_kind(gdf: gpd.GeoDataFrame) -> Literal["point", "line", "other"]:
    types = set(gdf.geometry.geom_type.unique())
    if types <= {"Point", "MultiPoint"}:
        return "point"
    if types <= _LINE_GEOMS:
        return "line"
    return "other"


def _geom_for_nearest_query(geom: BaseGeometry) -> BaseGeometry:
    """Use a point probe for line work when querying nearest polygons."""
    if geom.geom_type in _LINE_GEOMS:
        return geom.representative_point()
    return geom


def assign_geometries_to_region_by_overlap(
    features_gdf: gpd.GeoDataFrame, borders_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assign features to regions.

    Points use a ``within`` join. Lines and multilines use ``intersects``; when a
    feature crosses multiple regions, the region with the **largest shared length**
    is kept.
    """
    if features_gdf.empty:
        return features_gdf.copy()

    if _geometry_kind(features_gdf) == "point":
        return assign_points_to_region_by_within(features_gdf, borders_gdf)

    borders = borders_gdf.to_crs("EPSG:6933").copy()
    features = features_gdf.to_crs("EPSG:6933").copy()
    borders["geometry"] = borders.geometry.make_valid()
    features["geometry"] = features.geometry.make_valid()

    joined = gpd.sjoin(features, borders, how="left", predicate="intersects")
    if joined["index_right"].isna().all():
        return joined.drop(columns="index_right", errors="ignore").to_crs(
            features_gdf.crs
        )

    if not joined.index.duplicated(keep=False).any():
        return joined.drop(columns="index_right", errors="ignore").to_crs(
            features_gdf.crs
        )

    border_geoms = borders.geometry
    feat_geoms = features.geometry
    overlap = np.zeros(len(joined), dtype=float)
    for k, (feat_idx, row) in enumerate(joined.iterrows()):
        border_idx = row["index_right"]
        if pd.isna(border_idx):
            continue
        overlap[k] = (
            feat_geoms.loc[feat_idx]
            .intersection(border_geoms.iloc[int(border_idx)])
            .length
        )

    joined["_overlap_len"] = overlap
    best_idx = joined.groupby(level=0)["_overlap_len"].idxmax()
    out = joined.loc[best_idx].drop(
        columns=["_overlap_len", "index_right"], errors="ignore"
    )
    return out.to_crs(features_gdf.crs)


def assign_points_to_region_by_within(
    points_gdf: gpd.GeoDataFrame, borders_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assign points to countries based on whether they are within the country's borders.

    Args:
        points_gdf (gpd.GeoDataFrame): Points to assign to countries
        borders_gdf (gpd.GeoDataFrame): Dataframe containing a geometry column indicating the countries' borders (whether land, EEZ or otherwise)

    Returns:
        gpd.GeoDataFrame: Points with assigned information
    """
    borders = borders_gdf.to_crs("EPSG:6933").copy()
    points = points_gdf.to_crs("EPSG:6933").copy()

    borders["geometry"] = borders.geometry.make_valid()

    return (
        gpd.sjoin(points, borders, how="left", predicate="within").drop(
            columns="index_right"
        )
    ).to_crs(points_gdf.crs)


def assign_country_by_nearest(
    points_gdf: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame,
    fill_nan_only: bool = True,
    batch_size: int = 10000,
) -> gpd.GeoDataFrame:
    """
    Assign points to countries based on the nearest country.

    Args:
        points_gdf (gpd.GeoDataFrame): GeoDataFrame of points to assign to countries.
        countries (gpd.GeoDataFrame): GeoDataFrame of countries. Must have a NAME and ISO_A3 column
        fill_nan_only (bool): If True, only fill rows with missing country assignment.
                              If False, reassign all rows to nearest country.

    Returns:
        gpd.GeoDataFrame: Same size as input points_gdf, with missing (or all) country assignments filled in.
    """
    # Make a copy to avoid mutating input
    result = points_gdf.rename(columns=str.lower).copy()
    if fill_nan_only:
        mask = result.isna().any(axis=1)
    else:
        mask = np.full(len(result), True)

    if not mask.any():
        # Nothing to fill, return unchanged
        return result

    points_to_fill = result.loc[mask]
    geoms = [_geom_for_nearest_query(g) for g in points_to_fill["geometry"].values]
    tree = STRtree(countries.geometry.values)

    indices_out = np.empty(len(geoms), dtype=int)

    for start in tqdm(
        range(0, len(geoms), batch_size),
        desc="Assigning nearest country via STRtree-enabled nearest-neighbor search",
    ):
        end = min(start + batch_size, len(geoms))
        inds = tree.query_nearest(geoms[start:end], all_matches=False)[1, :]
        indices_out[start:end] = inds

    # Bulk assignment for matching country fields (add more fields as needed)
    for col in ["country", "iso_a3"]:
        result.loc[mask, col] = countries.iloc[indices_out][col].values

    return result


def load_nation_boundaries(
    source: BoundarySource = "land",
    *,
    geographic_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Load nation polygons with ``country``, ``iso_a3``, and ``geometry`` columns."""
    if geographic_dir is None:
        from src import config

        geographic_dir = config.geographic_dir

    if isinstance(source, Path):
        gdf = gpd.read_file(source)
        return gdf.rename(columns=str.lower)[["country", "iso_a3", "geometry"]]

    spec = _BOUNDARY_SPECS[str(source)]
    path = geographic_dir.joinpath(*spec["path"])
    cols = list(spec["columns"].keys()) + ["geometry"]
    gdf = gpd.read_file(path)[cols].rename(columns=spec["columns"])
    return gdf.rename(columns=str.lower)


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
    """Assign points to nations via boundary ``within``, then nearest land admin fill."""
    if boundaries_gdf is None:
        boundaries_gdf = load_nation_boundaries(
            boundaries, geographic_dir=geographic_dir
        )

    result = assign_points_to_region_by_within(points_gdf, boundaries_gdf)
    n_missing = int(result[["country", "iso_a3"]].isna().any(axis=1).sum())
    if verbose and n_missing:
        pct = 100 * n_missing / len(result)
        print(
            f"  {n_missing:,} points ({pct:.2f}%) unassigned after within-join; filling nearest…"
        )

    if fill_unassigned and n_missing:
        if fill_boundaries is None or isinstance(fill_boundaries, str):
            fill_gdf = load_nation_boundaries(
                fill_boundaries or "land", geographic_dir=geographic_dir
            )
        else:
            fill_gdf = fill_boundaries
        result = assign_country_by_nearest(result, fill_gdf, fill_nan_only=True)

    if verbose:
        n_left = int(result[["country", "iso_a3"]].isna().any(axis=1).sum())
        print(f"  {len(result) - n_left:,}/{len(result):,} points assigned to a nation")
    return result


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
    """
    Assign points or (multi)linestrings to nations.

    Points use a ``within`` join. Line geometries use ``intersects`` and keep the
    nation with the largest overlapping length, then optional nearest-neighbour fill
    for any remaining unassigned features (using each line's representative point).
    """
    kind = _geometry_kind(features_gdf)
    if kind == "other":
        raise ValueError(
            "assign_geometries_to_nations supports Point and LineString/MultiLineString "
            f"geometries only; got {set(features_gdf.geometry.geom_type.unique())}"
        )

    if boundaries_gdf is None:
        boundaries_gdf = load_nation_boundaries(
            boundaries, geographic_dir=geographic_dir
        )

    result = assign_geometries_to_region_by_overlap(features_gdf, boundaries_gdf)
    n_missing = int(result[["country", "iso_a3"]].isna().any(axis=1).sum())
    label = "lines" if kind == "line" else "points"
    if verbose and n_missing:
        pct = 100 * n_missing / len(result)
        print(
            f"  {n_missing:,} {label} ({pct:.2f}%) unassigned after spatial join; "
            "filling nearest…"
        )

    if fill_unassigned and n_missing:
        if fill_boundaries is None or isinstance(fill_boundaries, str):
            fill_gdf = load_nation_boundaries(
                fill_boundaries or "land", geographic_dir=geographic_dir
            )
        else:
            fill_gdf = fill_boundaries
        result = assign_country_by_nearest(result, fill_gdf, fill_nan_only=True)

    if verbose:
        n_left = int(result[["country", "iso_a3"]].isna().any(axis=1).sum())
        print(
            f"  {len(result) - n_left:,}/{len(result):,} {label} assigned to a nation"
        )
    return result


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
    if boundaries_gdf is None:
        boundaries_gdf = load_nation_boundaries(
            boundaries, geographic_dir=geographic_dir
        )
    if (
        fill_unassigned
        and fill_boundaries is not None
        and not isinstance(fill_boundaries, gpd.GeoDataFrame)
    ):
        fill_boundaries = load_nation_boundaries(
            fill_boundaries, geographic_dir=geographic_dir
        )

    out: list[gpd.GeoDataFrame] = []
    for i, gdf in enumerate(
        tqdm(gdfs, desc="Assigning nations", disable=not verbose), start=1
    ):
        if verbose:
            print(f"Layer {i}/{len(gdfs)} ({len(gdf):,} points)")
        out.append(
            assign_points_to_nations(
                gdf,
                boundaries_gdf,
                fill_unassigned=fill_unassigned,
                fill_boundaries=fill_boundaries,
                geographic_dir=geographic_dir,
                verbose=verbose,
            )
        )
    if verbose and out:
        n_left = sum(g["country"].isna().sum() for g in out)
        n_total = sum(len(g) for g in out)
        print(f"Total: {n_total - n_left:,}/{n_total:,} points assigned across layers")
    return out
