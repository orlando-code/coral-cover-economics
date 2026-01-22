"""
Data loading and alignment utilities for coral reef economics analysis.

This module provides functions to:
1. Load coral cover projection data (Sully et al. HBB model)
2. Load tourism value data
3. Load shoreline protection value data
4. Align datasets spatially using nearest-neighbor matching
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src import config

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CoralCoverData:
    """Container for coral cover data with projections."""

    df: pd.DataFrame
    meta_columns: List[str]
    projection_columns: List[str]
    change_columns: List[str]

    @property
    def n_sites(self) -> int:
        return len(self.df)

    @property
    def scenarios(self) -> List[str]:
        """Extract unique RCP/year combinations from column names."""
        scenarios = set()
        for col in self.projection_columns:
            # e.g., "Y_future_RCP45_yr_2100" -> "cover_RCP45_2100"
            parts = col.replace("_yr_", "_").replace("Y_future_", "cover")
            scenarios.add(parts)
        return sorted(scenarios)


@dataclass
class EconomicValueData:
    """Container for economic value data with spatial info."""

    gdf: gpd.GeoDataFrame
    value_column: str
    value_type: str  # "tourism" or "shoreline_protection"

    @property
    def n_sites(self) -> int:
        return len(self.gdf)

    @property
    def total_value(self) -> float:
        return self.gdf[self.value_column].sum()


# =============================================================================
# CORAL COVER DATA
# =============================================================================


def load_coral_cover_data(
    filepath: Optional[Path] = None, validate: bool = True, verbose: bool = True
) -> CoralCoverData:
    """
    Load coral cover projection data from Sully et al. HBB model.

    Parameters
    ----------
    filepath : Path, optional
        Path to data_for_maps.csv. Default: config.sully_data_dir / "data_for_maps.csv"
    validate : bool
        If True, perform data validation checks.

    Returns
    -------
    CoralCoverData
        Container with DataFrame and column metadata.
    """
    if filepath is None:
        filepath = config.sully_data_dir / "data_for_maps.csv"

    df = pd.read_csv(filepath).rename(columns=str.lower)

    # Identify column types
    meta_columns = [
        "reef_id",
        "latitude.degrees",
        "longitude.degrees",
        "ocean",
        "realm",
        "ecoregion.x",
        "country_name",
        "average_coral_cover",
        "y_new",  # Y_New = HBB model's baseline prediction (close to observed: "Average_coral_cover")
    ]
    meta_columns = [c for c in meta_columns if c in df.columns]

    projection_columns = [
        c for c in df.columns if c.startswith("y_future_") and "_change" not in c
    ]
    change_columns = [c for c in df.columns if c.endswith("_change")]

    if validate:
        _validate_coral_cover_data(df, projection_columns, change_columns)

    if verbose:
        _report_coral_cover_data(filepath, df, projection_columns, change_columns)

    return CoralCoverData(
        df=df,
        meta_columns=meta_columns,
        projection_columns=projection_columns,
        change_columns=change_columns,
    )


def _validate_coral_cover_data(
    df: pd.DataFrame, projection_columns: List[str], change_columns: List[str]
) -> None:
    """Validate coral cover data integrity."""

    # Check for expected columns
    required = ["latitude.degrees", "longitude.degrees", "y_new"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate Y_New is the baseline for _change columns
    if "y_new" in df.columns and change_columns:
        for change_col in change_columns[:1]:  # Check first one
            projected_col = change_col.replace("_change", "")
            if projected_col in df.columns:
                computed = (
                    df[projected_col] - df["y_new"]
                )  # change column should be difference between projected and model-predicted baseline
                stored = df[change_col]
                max_diff = (computed - stored).abs().max()
                if max_diff > 1e-6:
                    warnings.warn(
                        f"_change columns may not be computed as expected. "
                        f"Max diff for {change_col}: {max_diff:.6f}"
                    )

    # Check value ranges
    if df["average_coral_cover"].min() < 0 or df["average_coral_cover"].max() > 1:
        warnings.warn("'average_coral_cover' values outside [0, 1] range")


def _report_coral_cover_data(
    filepath: Path,
    df: pd.DataFrame,
    projection_columns: List[str],
    change_columns: List[str],
) -> None:
    print(f"\n{'─' * 60}")
    print(f"📊 CORAL COVER DATA LOADED FROM {'/'.join(filepath.parts[-3:])}")
    print(f"{'─' * 60}")
    print(f"  Sites: {len(df):,}")
    print(
        f"  Countries: {df['country_name'].nunique() if 'country_name' in df.columns else 'N/A'}"
    )
    print(f"  Projection columns: {len(projection_columns)}")
    print(f"  Change columns: {len(change_columns)}")
    print("\n  📈 Coral Cover Statistics:")
    print(
        f"     average_coral_cover: min={df['average_coral_cover'].min():.3f}, max={df['average_coral_cover'].max():.3f}, mean={df['average_coral_cover'].mean():.3f}"
    )
    print(
        f"     y_new (model baseline): min={df['y_new'].min():.3f}, max={df['y_new'].max():.3f}, mean={df['y_new'].mean():.3f}"
    )

    # Show change statistics for each scenario
    print("\n  📉 Projected Changes (percentage points):")
    for col in change_columns:
        scenario = (
            col.replace("y_future_", "").replace("_change", "").replace("_yr_", " ")
        )
        changes = df[col] * 100
        print(
            f"     {scenario}: mean={changes.mean():+.1f}pp, range=[{changes.min():+.1f}, {changes.max():+.1f}]"
        )

    # N.B. This is how the statistics in the paper are reported, although I'm not currently using this in the analysis.
    print("\n  📉 Projected Changes (percentage points relative to baseline):")
    for col in change_columns:
        scenario = (
            col.replace("y_future_", "").replace("_change", "").replace("_yr_", " ")
        )
        changes = df[col] * 100
        changes_relative_to_baseline = changes / df["y_new"]
        print(
            f"     {scenario}: mean={changes_relative_to_baseline.mean():+.1f}pp, range=[{changes_relative_to_baseline.min():+.1f}, {changes_relative_to_baseline.max():+.1f}]"
        )
    print(f"{'─' * 60}\n")


# =============================================================================
# TOURISM VALUE DATA
# =============================================================================


def load_tourism_data(
    shapefile_path: Optional[Path] = None,
    appendix_path: Optional[Path] = None,
    countries_shapefile: Optional[Path] = None,
    apply_correction: bool = True,
    validate: bool = True,
) -> EconomicValueData:
    """
    Load tourism value data and optionally apply country-level correction.

    The raw shapefile contains binned values. We:
    1. Explode MultiPolygons to individual polygons
    2. Assign each polygon to a country
    3. Optionally correct using appendix summary data (actual values)

    Parameters
    ----------
    shapefile_path : Path, optional
        Path to Total_Dollar_Value shapefile.
    appendix_path : Path, optional
        Path to A1_summary_data.csv with actual per-country values.
    countries_shapefile : Path, optional
        Path to countries shapefile for assignment.
    apply_correction : bool
        If True, scale bin values to match appendix totals.
    validate : bool
        If True, perform validation checks.

    Returns
    -------
    EconomicValueData
        Container with GeoDataFrame and metadata.
    """
    if shapefile_path is None:
        shapefile_path = (
            config.tourism_dir
            / "total"
            / "Coral_Reef_Tourism_Global_Total_Dollar_Value.shp"
        )
    if appendix_path is None:
        appendix_path = config.tourism_dir / "A1_summary_data.csv"
    if countries_shapefile is None:
        countries_shapefile = (
            config.geographic_dir
            / "ne_10m_admin_0_countries"
            / "ne_10m_admin_0_countries.shp"
        )

    # Load and process shapefile
    gdf = (
        gpd.read_file(shapefile_path).to_crs("EPSG:4326").rename(columns=str.lower)
    )  # this is the aggregate data from Ocean Wealth

    # Map bin values to approximate prices (median value per data bin)
    approx_prices = {
        0: 0,
        1: 2000,
        2: 6000,
        3: 10000,
        4: 18000,
        5: 34000,
        6: 68000,
        7: 132000,
        8: 262000,
        9: 630000,
        10: 908000,
    }
    gdf["approx_price"] = gdf["bin_global"].map(approx_prices)

    # Explode MultiPolygons
    gdf_exploded = gdf.explode(index_parts=False).reset_index(drop=True)

    # Assign to countries
    gdf_with_countries = _assign_pts_to_countries(gdf_exploded, countries_shapefile)

    # Apply correction if requested
    value_column = "approx_price"
    if apply_correction and appendix_path.exists():
        gdf_with_countries, value_column = _apply_tourism_correction(
            gdf_with_countries, appendix_path, countries_shapefile
        )

    if validate:
        _summarise_tourism_data(gdf_with_countries, value_column)

    return EconomicValueData(
        gdf=gdf_with_countries, value_column=value_column, value_type="tourism"
    )


def _assign_pts_to_countries(
    gdf: gpd.GeoDataFrame, countries_shapefile: Path
) -> gpd.GeoDataFrame:
    """Assign each polygon to a country using centroid spatial join."""
    from shapely.strtree import STRtree

    countries = gpd.read_file(countries_shapefile).rename(columns=str.lower)[
        ["name", "iso_a3", "geometry"]
    ]

    # Project for accurate centroids
    projected_crs = "EPSG:6933"
    gdf_proj = gdf.to_crs(projected_crs)
    countries_proj = countries.to_crs(projected_crs)

    # Get centroids
    centroids = gdf_proj.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(
        gdf_proj.drop(columns=["geometry"]), geometry=centroids, crs=projected_crs
    )

    # Spatial join
    joined = gpd.sjoin(centroids_gdf, countries_proj, how="left", predicate="within")

    # Handle ocean polygons (no country match) - assign to nearest
    ocean_mask = joined["iso_a3"].isna()
    if ocean_mask.sum() > 0:
        tree = STRtree(countries_proj.geometry.values)
        ocean_centroids = joined.loc[ocean_mask, "geometry"].values
        nearest_idx = tree.query_nearest(
            ocean_centroids, return_distance=False, all_matches=False
        )[1, :]
        joined.loc[ocean_mask, "iso_a3"] = countries_proj.iloc[nearest_idx][
            "iso_a3"
        ].values
        joined.loc[ocean_mask, "name"] = countries_proj.iloc[nearest_idx]["name"].values

    # Restore original geometry
    joined["geometry"] = gdf.geometry.values
    joined = joined.set_crs(gdf.crs)

    # Clean up
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    joined = joined.rename(columns={"name": "country"})

    return joined


def _apply_tourism_correction(
    gdf: gpd.GeoDataFrame, appendix_path: Path, countries_shapefile: Path
) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Scale approximate prices to match country totals as stated in the Spalding et al. (2017) appendix (1-s2.0-S0308597X17300635-mmc1.pdf)
    # TODO: this is approximate, so should request original data

    The appendix contains the 'Sum of reef-associated visitor expenditure' for each country with total expenditure >$10 million USD annually.
    I compute a proportion of this metric relative to the sum of the value for all the reef points for that country, then multiply each of these points by this proportion.
    # TODO: check that the sum therefore matches the total value for the country as stated in the appendix.

    Returns
    -------
    tuple
        (corrected GeoDataFrame, name of corrected value column)
    """
    # Load appendix data
    appendix = pd.read_csv(appendix_path).rename(columns=str.lower)

    # Clean and extract reef-associated expenditure
    exp_col = "sum of reef-associated visitor expenditure (1000)"
    if exp_col in appendix.columns:
        appendix["reef_expenditure"] = (
            appendix[exp_col]
            .replace({",": ""}, regex=True)
            .pipe(pd.to_numeric, errors="coerce")
            * 1000
        )
    else:
        warnings.warn(f"Appendix column '{exp_col}' not found. Skipping correction.")
        return gdf, "approx_price"

    # Get ISO codes for appendix countries
    countries = gpd.read_file(countries_shapefile).rename(columns=str.lower)[
        ["name", "iso_a3"]
    ]

    iso_map = countries.set_index("name")["iso_a3"].to_dict()
    if "iso_country" in appendix.columns:
        appendix["iso_a3"] = appendix["iso_country"].map(iso_map)
        # Fill missing from country column
        mask = appendix["iso_a3"].isna()
        appendix.loc[mask, "iso_a3"] = appendix.loc[mask, "country"].map(iso_map)
    else:
        appendix["iso_a3"] = appendix["country"].map(iso_map)

    # Handle US special case (Florida + Hawaii combined)
    # TODO: would be better to keep these separate in future, but will need to circumvent iso_a3 mapping
    us_mask = appendix["iso_country"].fillna("").str.contains("United States")
    if us_mask.sum() > 1:
        us_total = appendix.loc[us_mask, "reef_expenditure"].sum()
        appendix = appendix[~us_mask]
        us_row = pd.DataFrame(
            [
                {
                    "iso_a3": "USA",
                    "country": "United States",
                    "reef_expenditure": us_total,
                }
            ]
        )
        appendix = pd.concat([appendix, us_row], ignore_index=True)

    appendix = appendix.dropna(subset=["iso_a3", "reef_expenditure"])

    # Calculate per-country sums from gdf
    gdf_totals = gdf.groupby("iso_a3")["approx_price"].sum()

    # Calculate correction factors
    correction_factors = {}
    for _, row in appendix.iterrows():
        iso = row["iso_a3"]
        if iso in gdf_totals.index and gdf_totals[iso] > 0:
            correction_factors[iso] = row["reef_expenditure"] / gdf_totals[iso]

    # Apply correction
    gdf = gdf.copy()
    gdf["correction_factor"] = gdf["iso_a3"].map(correction_factors).fillna(1.0)
    gdf["approx_price_corrected"] = gdf["approx_price"] * gdf["correction_factor"]

    return gdf, "approx_price_corrected"


def _summarise_tourism_data(gdf: gpd.GeoDataFrame, value_column: str) -> None:
    """Validate tourism data."""
    total = gdf[value_column].sum()
    by_country = gdf.groupby("country")[value_column].sum().sort_values(ascending=False)

    print(f"\n{'─' * 60}")
    print("💰 TOURISM VALUE DATA LOADED")
    print(f"{'─' * 60}")
    print(f"  Polygons: {len(gdf):,}")
    print(f"  Countries: {gdf['country'].nunique()}")
    print(f"  Value column: '{value_column}'")

    print("\n  📊 Value Statistics:")
    print(f"     Total: ${total / 1e9:.2f} billion")
    print(f"     Mean per polygon: ${gdf[value_column].mean():,.0f}")
    print(f"     Median per polygon: ${gdf[value_column].median():,.0f}")
    print(
        f"     Non-zero polygons: {(gdf[value_column] > 0).sum():,} ({100 * (gdf[value_column] > 0).mean():.1f}%)"
    )

    print("\n  🏆 Top 10 Countries by Value:")
    for i, (country, value) in enumerate(by_country.head(10).items(), 1):
        pct = 100 * value / total
        print(f"     {i:2}. {country}: ${value / 1e6:,.0f}M ({pct:.1f}%)")

    print(f"{'─' * 60}\n")


# =============================================================================
# SHORELINE PROTECTION DATA
# =============================================================================


def load_shoreline_protection_data(
    geoparquet_path: Optional[Path] = None,
    countries_shapefile: Optional[Path] = None,
    eez_path: Optional[Path] = None,
    validate: bool = True,
) -> EconomicValueData:
    """
    Load shoreline protection (GDP spared from flooding) data.

    Parameters
    ----------
    geoparquet_path : Path, optional
        Path to GDP_spared_PT.parquet file.
    countries_shapefile : Path, optional
        Path to countries shapefile for assignment.
    eez_path : Path, optional
        Path to EEZ boundaries for initial assignment.
    validate : bool
        If True, perform validation.

    Returns
    -------
    EconomicValueData
        Container with GeoDataFrame and metadata.

    Notes
    -----
    The GDP_spared values are classifications (1-10+), not dollar amounts.
    You'll need a separate mapping to convert to actual dollar values.
    """
    if geoparquet_path is None:
        geoparquet_path = (
            config.economics_data_dir
            / "CR_Fisheries_Shoreline_Protection"
            / "geoparquet"
            / "GDP_spared_PT.parquet"
        )
    if countries_shapefile is None:
        countries_shapefile = (
            config.geographic_dir
            / "ne_10m_admin_0_countries"
            / "ne_10m_admin_0_countries.shp"
        )

    # Load parquet
    gdf = (
        gpd.read_parquet(geoparquet_path).rename(columns=str.lower).to_crs("EPSG:4326")
    )

    # Rename grid_code to something meaningful
    if "grid_code" in gdf.columns:
        gdf = gdf.rename(columns={"grid_code": "gdp_spared_class"})

    # TODO: Map classification to actual dollar values if available
    # For now, use classification as a proxy value
    # This should be updated with actual value mapping
    gdf["gdp_spared_value"] = gdf["gdp_spared_class"]  # Placeholder

    # Assign to countries using nearest-neighbor (many points are offshore)
    gdf = _assign_protection_to_countries(gdf, countries_shapefile)

    if validate:
        _validate_protection_data(gdf)

    return EconomicValueData(
        gdf=gdf, value_column="gdp_spared_value", value_type="shoreline_protection"
    )


def _assign_protection_to_countries(
    gdf: gpd.GeoDataFrame, countries_shapefile: Path
) -> gpd.GeoDataFrame:
    """Assign protection points to countries."""
    from src.processing.processdata import assign_country_by_nearest

    countries = gpd.read_file(countries_shapefile)[["NAME", "ISO_A3", "geometry"]]
    countries = countries.rename(columns={"NAME": "country", "ISO_A3": "iso_a3"})

    gdf = assign_country_by_nearest(gdf, countries)

    return gdf


def _validate_protection_data(gdf: gpd.GeoDataFrame) -> None:
    """Validate shoreline protection data."""
    print(f"✓ Loaded shoreline protection data: {len(gdf):,} points")
    print(
        f"  Countries: {gdf['country'].nunique() if 'country' in gdf.columns else 'N/A'}"
    )
    print(f"  Value classes: {gdf['gdp_spared_class'].nunique()} unique")


# =============================================================================
# SPATIAL ALIGNMENT
# =============================================================================


def align_coral_to_economic_data(
    coral_data: CoralCoverData,
    economic_data: EconomicValueData,
    max_distance_deg: float = 5.0,
    columns_to_add: Optional[List[str]] = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Align coral cover projections to economic value data using nearest-neighbor.

    For each economic value polygon/point, find the nearest coral cover site
    and copy its projection data.

    Parameters
    ----------
    coral_data : CoralCoverData
        Coral cover data with projections.
    economic_data : EconomicValueData
        Economic value data to align to.
    max_distance_deg : float
        Maximum distance in degrees. Points beyond this get NaN values.
    columns_to_add : list, optional
        Columns to copy from coral data. Default: all projection and change columns.

    Returns
    -------
    GeoDataFrame
        Economic data with added coral cover columns.
    """
    coral_df = coral_data.df
    econ_gdf = economic_data.gdf.copy()

    # Determine columns to add
    if columns_to_add is None:
        columns_to_add = (
            ["average_coral_cover", "y_new"]
            + coral_data.projection_columns
            + coral_data.change_columns
        )
    columns_to_add = [c for c in columns_to_add if c in coral_df.columns]

    # Get coordinates
    # For economic data, use centroids if polygons
    if econ_gdf.geometry.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]:
        # Project for accurate centroids, then back to WGS84
        econ_proj = econ_gdf.to_crs("ESRI:54030")
        centroids = econ_proj.geometry.centroid
        centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=econ_proj.crs)
        centroids_ll = centroids_gdf.to_crs("EPSG:4326")
        econ_coords = np.array([[g.x, g.y] for g in centroids_ll.geometry])
    else:
        econ_coords = np.array([[g.x, g.y] for g in econ_gdf.geometry])

    # Coral cover coordinates (lowercase column names)
    coral_coords = coral_df[["longitude.degrees", "latitude.degrees"]].values

    # Build KD-tree and query
    tree = cKDTree(coral_coords)
    distances, nearest_idx = tree.query(econ_coords, k=1)

    # Add columns with "nearest_" prefix
    for col in columns_to_add:
        values = coral_df.iloc[nearest_idx][col].values
        # Set to NaN if beyond max distance
        values = np.where(distances > max_distance_deg, np.nan, values)
        econ_gdf[f"nearest_{col}"] = values

    # Add distance column
    econ_gdf["distance_to_coral_site"] = distances

    # Summary stats
    n_matched = (distances <= max_distance_deg).sum()
    n_total = len(econ_gdf)
    n_beyond = (distances > max_distance_deg).sum()

    if verbose:
        _report_alignment_results(
            coral_df,
            econ_gdf,
            max_distance_deg,
            n_matched,
            n_total,
            n_beyond,
            distances,
            columns_to_add,
        )

    return econ_gdf


def _report_alignment_results(
    coral_df: pd.DataFrame,
    econ_gdf: gpd.GeoDataFrame,
    max_distance_deg: float,
    n_matched: int,
    n_total: int,
    n_beyond: int,
    distances: np.ndarray,
    columns_to_add: List[str],
) -> None:
    """Report alignment results."""
    print(f"\n{'─' * 60}")
    print("🔗 SPATIAL ALIGNMENT COMPLETE")
    print(f"{'─' * 60}")
    print(f"  Economic sites: {n_total:,}")
    print(f"  Coral cover sites: {len(coral_df):,}")
    print(f"  Max distance threshold: {max_distance_deg}°")

    print("\n  📍 Alignment Results:")
    print(f"     Within threshold: {n_matched:,} ({100 * n_matched / n_total:.1f}%)")
    print(f"     Beyond threshold: {n_beyond:,} ({100 * n_beyond / n_total:.1f}%)")

    print("\n  📏 Distance Statistics:")
    print(f"     Min: {distances.min():.4f}°")
    print(f"     Max: {distances.max():.2f}°")
    print(f"     Mean: {distances.mean():.2f}°")
    print(f"     Median: {np.median(distances):.2f}°")
    print(f"     P95: {np.percentile(distances, 95):.2f}°")

    # Show columns added
    print(f"\n  📋 Columns added ({len(columns_to_add)}):")
    for col in columns_to_add[:5]:
        print(f"     - nearest_{col}")
    if len(columns_to_add) > 5:
        print(f"     ... and {len(columns_to_add) - 5} more")

    print(f"{'─' * 60}\n")

    return econ_gdf


# =============================================================================
# GDP DATA AND AGGREGATIONS
# =============================================================================


def load_gdp_data(
    gdp_csv_path: Optional[Path] = None,
    year: int = 2022,
) -> pd.DataFrame:
    """
    Load World Bank GDP data.

    Parameters
    ----------
    gdp_csv_path : Path, optional
        Path to World Bank GDP CSV. Default: from config.
    year : int
        Year to extract GDP for.

    Returns
    -------
    DataFrame
        DataFrame with 'iso_a3' and 'gdp' columns.
    """
    if gdp_csv_path is None:
        gdp_csv_path = (
            config.economics_data_dir
            / "gdp"
            / "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2"
            / "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2.csv"
        )

    # Read GDP data (skip first 4 rows of metadata)
    gdp_df = pd.read_csv(gdp_csv_path, skiprows=4)

    # Get GDP for specified year
    year_col = str(year)
    if year_col not in gdp_df.columns:
        # Find most recent year available
        year_cols = [c for c in gdp_df.columns if c.isdigit()]
        year_col = max(year_cols)
        warnings.warn(f"Year {year} not found, using {year_col}")

    result = pd.DataFrame(
        {
            "country_name": gdp_df["Country Name"],
            "iso_a3": gdp_df["Country Code"],
            "gdp": gdp_df[year_col],
        }
    )

    result = result.dropna(subset=["gdp"])

    print(f"✓ Loaded GDP data: {len(result)} countries for year {year_col}")

    return result


def compute_country_aggregations(
    tourism_gdf: gpd.GeoDataFrame,
    value_column: str = "approx_price_corrected",
    gdp_data: pd.DataFrame = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute country-level aggregations of tourism value.

    Optionally computes reef tourism as % of national GDP.

    Parameters
    ----------
    tourism_gdf : GeoDataFrame
        Tourism data with country assignments.
    value_column : str
        Column to aggregate.
    gdp_data : DataFrame, optional
        GDP data with 'iso_a3' and 'gdp' columns.

    Returns
    -------
    DataFrame
        Aggregated by country with optional GDP percentage.
    """
    if verbose:
        print(f"\n{'─' * 60}")
        print("📊 COMPUTING COUNTRY AGGREGATIONS")
        print(f"{'─' * 60}")

    # Aggregate by country
    by_country = (
        tourism_gdf.groupby(["country", "iso_a3"])
        .agg({value_column: "sum"})
        .reset_index()
    )
    by_country = by_country.set_index("iso_a3")

    # Add GDP percentage if GDP data provided
    if gdp_data is not None:
        gdp_map = gdp_data.set_index("iso_a3")["gdp"]
        by_country["national_gdp"] = by_country.index.map(gdp_map)
        by_country["reef_tourism_gdp_as_pct_of_national_gdp"] = (
            100 * by_country[value_column] / by_country["national_gdp"]
        )

    if verbose:
        _report_country_aggregations(by_country, value_column)

    return by_country


def _report_country_aggregations(by_country: pd.DataFrame, value_column: str) -> None:
    print(f"\n{'─' * 60}")
    print("📊 COUNTRY AGGREGATIONS")
    print(f"{'─' * 60}")
    print(f"  Countries: {len(by_country)}")
    print(f"  Total tourism value: ${by_country[value_column].sum() / 1e9:.2f}B")

    if "reef_tourism_gdp_as_pct_of_national_gdp" in by_country.columns:
        print("\n  🏆 Top 10 by GDP Contribution (%):")
        top = by_country.nlargest(10, "reef_tourism_gdp_as_pct_of_national_gdp")
        for i, (iso, row) in enumerate(top.iterrows(), 1):
            print(
                f"     {i:2}. {row['country']}: {row['reef_tourism_gdp_as_pct_of_national_gdp']:.2f}%"
            )

    print(f"{'─' * 60}\n")

    # plot barchart of top 10 countries by tourism value

    return by_country
