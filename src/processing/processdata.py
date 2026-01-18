import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
from tqdm import tqdm


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
    geoms = points_to_fill["geometry"].values
    tree = STRtree(countries.geometry.values)

    indices_out = np.empty(len(geoms), dtype=int)

    for start in tqdm(
        range(0, len(geoms), batch_size), desc="Assigning nearest country"
    ):
        end = min(start + batch_size, len(geoms))
        inds = tree.query_nearest(geoms[start:end], all_matches=False)[1, :]
        indices_out[start:end] = inds

    # Bulk assignment for matching country fields (add more fields as needed)
    for col in ["country", "iso_a3"]:
        result.loc[mask, col] = countries.iloc[indices_out][col].values

    return result
