from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

from src import config
from src.processing.processdata import GEOGRAPHIC_CRS

REDLIST_DIR = config.diversity_dir / "redlist_species_data"
OBIS_CSV = config.diversity_dir / "obis_anthozoa.csv"
OBIS_ASSIGNED_PARQUET = config.diversity_dir / "obis_anthozoa_cotw_ecoregions.parquet"
COTW_SPECIES_CSV = config.diversity_dir / "ecoregion_species.csv"

# Shapefile ecoregion names -> COTW website names (see src.models.hbb.variants.DIVERSITY_ECOREGION_ALIASES).
ECOREGION_SHAPEFILE_TO_WEBSITE: dict[str, str] = {
    "Banda Sea and Molucca Islands": "Banda Sea and Moluccas",
    "Birds Head Peninsula, Papua": "Raja Ampat, Papua",
    "Central and northern Great Barrier Reef": "Great Barrier Reef north-central",
    "Colombia, Ecuador and Chile, Pacific coast": "Colombia and Ecuador, Pacific coast",
    "Cook Islands, south-west Pacific": "Cook Islands, central Pacific",
    "Eastern Hawaii": "Hawaii east",
    "Eastern coast South Africa": "South Africa east",
    "Gilbert Islands, west Kiribati": "Kiribati west, Gilbert Islands",
    "Gulf of Tomini, Indonesia": "Gulf of Tomini, Sulawesi",
    "Kenya and Tanzania coast": "Kenya and Tanzania",
    "Lakshadweep Islands": "Lakshadweep",
    "Makassar Strait, Indonesia": "Makassar Strait",
    "Maldive Islands": "Maldives",
    "Moreton Bay, eastern Australia": "Moreton Bay, east Australia",
    "North Madagascar": "Madagascar north",
    "North Mozambique coast": "Mozambique north",
    "North Myanmar and Bangladesh": "Myanmar north and Bangladesh",
    "North Philippines": "Philippines north",
    "North Ryukyu Islands, Japan": "Ryukyu Islands north",
    "North Sri Lanka and east India": "Sri Lanka north and India east",
    "North Vietnam": "Vietnam north",
    "North and central Red Sea": "Red Sea north-central",
    "Northern Seychelles": "Seychelles north",
    "South Java": "Java south",
    "South Madagascar": "Madagascar south",
    "South Mozambique coast": "Mozambique south",
    "South Red Sea": "Red Sea south",
    "South Ryukyu Islands, Japan": "Ryukyu Islands south",
    "South Vietnam": "Vietnam south",
    "South-east Philippines": "Philippines south-east",
    "Southern Great Barrier Reef": "Great Barrier Reef south",
    "Southern Seychelles": "Seychelles south",
    "Strait of Malacca": "Malacca Strait",
    "Sunda Shelf, south-east Asia": "Sunda Shelf",
    "West Sumatra": "Sumatra west",
    "Western Mexico and Revillagigedo Islands": "Mexico west and Revillagigedo Islands",
    "Western Tuamotu Archipelago, central Pacific": "Tuamotu Archipelago west, central Pacific",
}
COTW_ECOREGIONS_SHP = (
    config.data_dir
    / "sully_og"
    / "ecoregion_shapefiles"
    / "ecoregion_exportPolygon.shp"
)

DEFAULT_BOUNDS = (-180.0, -35.0, 180.0, 35.0)

_OBIS_USECOLS = ("decimallatitude", "decimallongitude", "scientificname", "speciesid")
_WORMS_BATCH_SIZE = 50


def _batch_iterable(items: list, batch_size: int = _WORMS_BATCH_SIZE):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _cotw_native_crs(shapefile: Path | None = None) -> str:
    """Pacific-centered Mercator CRS from the COTW export (CM 150°)."""
    path = shapefile or COTW_ECOREGIONS_SHP
    return str(gpd.read_file(path).crs)


def _to_geographic(
    gdf: gpd.GeoDataFrame, crs: str = GEOGRAPHIC_CRS
) -> gpd.GeoDataFrame:
    out = gdf.set_crs(crs) if gdf.crs is None else gdf
    return out.to_crs(crs)


def _lon_span(geom) -> float:
    minx, _, maxx, _ = geom.bounds
    return maxx - minx


def _unwrap_lon_lat_ring(
    lons: list[float], lats: list[float]
) -> tuple[list[float], list[float]]:
    """Make a ring continuous across the antimeridian for plotting and analysis."""
    if not lons:
        return lons, lats
    out_lon = [lons[0]]
    out_lat = [lats[0]]
    for lon, lat in zip(lons[1:], lats[1:], strict=False):
        while lon - out_lon[-1] > 180:
            lon -= 360
        while lon - out_lon[-1] < -180:
            lon += 360
        out_lon.append(lon)
        out_lat.append(lat)
    return out_lon, out_lat


def _unwrap_polygon(geom: Polygon) -> Polygon:
    x, y = geom.exterior.coords.xy
    ux, uy = _unwrap_lon_lat_ring(list(x), list(y))
    holes = []
    for interior in geom.interiors:
        ix, iy = interior.coords.xy
        uix, uiy = _unwrap_lon_lat_ring(list(ix), list(iy))
        holes.append(list(zip(uix, uiy, strict=False)))
    return Polygon(list(zip(ux, uy, strict=False)), holes)


def fix_dateline_geometry(
    geom,
    *,
    max_part_lon_span: float = 60.0,
) -> Polygon | MultiPolygon | None:
    """Repair COTW polygons that incorrectly wrap around the world."""
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == "GeometryCollection":
        parts = [
            fix_dateline_geometry(g, max_part_lon_span=max_part_lon_span)
            for g in geom.geoms
            if not g.is_empty
        ]
        parts = [p for p in parts if p is not None and not p.is_empty]
        return unary_union(parts) if parts else geom

    if geom.geom_type == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            if _lon_span(part) <= max_part_lon_span:
                parts.append(part)
            elif part.geom_type == "Polygon":
                unwrapped = _unwrap_polygon(part)
                if _lon_span(unwrapped) <= max_part_lon_span * 4:
                    parts.append(unwrapped)
        if not parts:
            return geom
        geom = unary_union(parts)

    if geom.geom_type == "Polygon" and _lon_span(geom) > max_part_lon_span:
        geom = _unwrap_polygon(geom)

    if not geom.is_valid:
        geom = make_valid(geom)

    return geom


def fix_cotw_dateline_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of COTW ecoregions with Pacific dateline artefacts removed."""
    out = _to_geographic(gdf).copy()
    out["geometry"] = [
        fix_dateline_geometry(geom) for geom in out.geometry
    ]
    return out


def normalize_species_name(name: str | None) -> str | None:
    """Lowercase and collapse whitespace for cross-dataset species matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    cleaned = re.sub(r"\s+", " ", str(name).strip().lower())
    return cleaned or None


def _pick_worms_match(name: str, matches: list[dict]) -> dict | None:
    """Pick the best WoRMS match from ``AphiaRecordsByMatchNames`` results."""
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    name_lower = str(name).strip().lower()
    exact = [
        match
        for match in matches
        if str(match.get("match_type", "")).lower() == "exact"
        or str(match.get("scientificname", "")).strip().lower() == name_lower
    ]
    if len(exact) == 1:
        return exact[0]
    if exact:
        accepted = [
            match
            for match in exact
            if str(match.get("status", "")).lower() == "accepted"
        ]
        return accepted[0] if accepted else exact[0]

    accepted = [
        match
        for match in matches
        if str(match.get("status", "")).lower() == "accepted"
    ]
    return accepted[0] if accepted else matches[0]


def _worms_lookup_row(name: str, match_list: list[dict]) -> dict:
    best = _pick_worms_match(name, match_list)
    if best is None:
        return {
            "original_name": name,
            "worms_match_count": 0,
            "worms_match_status": "no_match",
            "worms_scientificname": None,
            "worms_aphia_id": None,
            "worms_valid_name": None,
            "worms_valid_aphia_id": None,
            "worms_match_type": None,
            "worms_taxon_rank": None,
        }

    return {
        "original_name": name,
        "worms_match_count": len(match_list),
        "worms_match_status": (
            "ambiguous" if len(match_list) > 1 else best.get("status")
        ),
        "worms_scientificname": best.get("scientificname"),
        "worms_aphia_id": best.get("AphiaID"),
        "worms_valid_name": best.get("valid_name"),
        "worms_valid_aphia_id": best.get("valid_AphiaID"),
        "worms_match_type": best.get("match_type"),
        "worms_taxon_rank": best.get("rank"),
    }


def match_species_names_via_worms(
    names: pd.Series | list[str],
    *,
    marine_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Match scientific names to WoRMS accepted names via ``pyworms``.

    Returns one row per unique input name with WoRMS match metadata.
    """
    import pyworms as pw
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    unique_names = (
        pd.Series(names, dtype="object")
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )
    columns = [
        "original_name",
        "worms_match_count",
        "worms_match_status",
        "worms_scientificname",
        "worms_aphia_id",
        "worms_valid_name",
        "worms_valid_aphia_id",
        "worms_match_type",
        "worms_taxon_rank",
    ]
    if not unique_names:
        return pd.DataFrame(columns=columns)

    batches = list(_batch_iterable(unique_names))
    all_matches: list[list[dict]] = []

    if verbose:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"WoRMS name lookup ({len(unique_names):,} names)",
                total=len(batches),
            )
            for batch in batches:
                batch_matches = pw.aphiaRecordsByMatchNames(
                    batch, marine_only=marine_only
                )
                all_matches.extend(batch_matches or [[]] * len(batch))
                progress.advance(task)
    else:
        for batch in batches:
            batch_matches = pw.aphiaRecordsByMatchNames(batch, marine_only=marine_only)
            all_matches.extend(batch_matches or [[]] * len(batch))

    rows = [
        _worms_lookup_row(name, matches or [])
        for name, matches in zip(unique_names, all_matches, strict=False)
    ]
    return pd.DataFrame(rows, columns=columns)


def _attach_worms_lookup(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    name_col: str,
    species_id_col: str | None = "species_id",
) -> pd.DataFrame:
    name_key = name_col.lower()
    sid_key = species_id_col.lower() if species_id_col else None
    out = df.merge(lookup, left_on=name_key, right_on="original_name", how="left")
    out = out.drop(columns=["original_name"])
    out["species_name_standardized"] = out["worms_valid_name"].fillna(out[name_key])
    if sid_key and sid_key in out.columns:
        out["species_id_standardized"] = out["worms_valid_aphia_id"].fillna(
            pd.to_numeric(out[sid_key], errors="coerce")
        )
    else:
        out["species_id_standardized"] = out["worms_valid_aphia_id"]
    out["species_name_norm"] = out["species_name_standardized"].map(normalize_species_name)
    return out


def standardize_species_lists_for_comparison(
    *species_dfs: pd.DataFrame,
    name_col: str = "species_name",
    species_id_col: str = "species_id",
    marine_only: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, ...]:
    """Apply one shared WoRMS lookup to per-ecoregion species-list frames."""
    if not species_dfs:
        return ()

    name_key = name_col.lower()
    sid_key = species_id_col.lower()
    all_names = pd.concat(
        [df[name_key] for df in species_dfs],
        ignore_index=True,
    )
    lookup = match_species_names_via_worms(
        all_names,
        marine_only=marine_only,
        verbose=verbose,
    )

    if verbose:
        matched = lookup["worms_valid_name"].notna().sum()
        print(
            f"WoRMS matched {matched:,} / {len(lookup):,} unique names "
            f"across {len(species_dfs)} species lists"
        )
        print(f"No match: {(lookup['worms_match_status'] == 'no_match').sum():,}")
        print(f"Ambiguous matches: {(lookup['worms_match_count'] > 1).sum():,}")

    return tuple(
        _attach_worms_lookup(
            df,
            lookup,
            name_col=name_key,
            species_id_col=sid_key if sid_key in df.columns else None,
        )
        for df in species_dfs
    )


def apply_worms_name_standardization(
    df: pd.DataFrame,
    name_col: str = "scientificname",
    *,
    species_id_col: str = "speciesid",
    marine_only: bool = True,
    verbose: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add WoRMS-standardised scientific-name columns to an occurrence table."""
    out = df if inplace else df.copy()
    name_key = name_col.lower()
    sid_key = species_id_col.lower()
    out = out.rename(columns=str.lower)

    lookup = match_species_names_via_worms(
        out[name_key],
        marine_only=marine_only,
        verbose=verbose,
    )
    out = _attach_worms_lookup(
        out,
        lookup,
        name_col=name_key,
        species_id_col=sid_key if sid_key in out.columns else None,
    )
    return out.rename(
        columns={
            "species_name_standardized": "scientificname_standardized",
            "species_id_standardized": "speciesid_standardized",
        }
    )


def normalize_ecoregion_name(name: str | None) -> str | None:
    """Normalize ecoregion names for cross-source matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    return cleaned or None


def _website_ecoregion_candidates(shapefile_names: list[str]) -> dict[str, str]:
    """Map normalized COTW website names to shapefile ecoregion names."""
    lookup: dict[str, str] = {}
    for shp in shapefile_names:
        website = ECOREGION_SHAPEFILE_TO_WEBSITE.get(shp, shp)
        for label in (website, shp):
            norm = normalize_ecoregion_name(label)
            if norm:
                lookup[norm] = shp
    return lookup


def map_website_ecoregion_to_shapefile(
    website_name: str,
    shapefile_names: list[str],
    *,
    lookup: dict[str, str] | None = None,
) -> str | None:
    """Map a COTW website ecoregion label to the shapefile polygon name."""
    if website_name in shapefile_names:
        return website_name

    reverse_alias = {v: k for k, v in ECOREGION_SHAPEFILE_TO_WEBSITE.items()}
    if website_name in reverse_alias:
        return reverse_alias[website_name]

    norm_lookup = lookup or _website_ecoregion_candidates(shapefile_names)
    norm = normalize_ecoregion_name(website_name)
    if norm and norm in norm_lookup:
        return norm_lookup[norm]

    prefix_matches = list(
        {
            shp
            for web_norm, shp in norm_lookup.items()
            if norm and (web_norm.startswith(norm) or norm.startswith(web_norm))
        }
    )
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def load_redlist_species(
    redlist_dir: Path | None = None,
    *,
    pattern: str = "data_*.shp",
) -> gpd.GeoDataFrame:
    """Load and concatenate IUCN Red List species range shapefiles."""
    directory = redlist_dir or REDLIST_DIR
    shapefiles = sorted(directory.glob(pattern))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles matching {pattern!r} in {directory}")

    frames = [gpd.read_file(path) for path in shapefiles]
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return out.rename(columns=str.lower)


def load_cotw_ecoregions(
    shapefile: Path | None = None,
    *,
    fix_dateline: bool = True,
) -> gpd.GeoDataFrame:
    """Load COTW ecoregion polygons used in the original paper."""
    gdf = gpd.read_file(shapefile or COTW_ECOREGIONS_SHP).rename(columns=str.lower)
    if fix_dateline:
        return fix_cotw_dateline_geometries(gdf)
    return gdf


def load_cotw_ecoregions_for_assignment(
    shapefile: Path | None = None,
) -> gpd.GeoDataFrame:
    """Load COTW polygons in native Mercator CRS for spatial point assignment."""
    return gpd.read_file(shapefile or COTW_ECOREGIONS_SHP).rename(columns=str.lower)


def load_obis_occurrences(
    path: Path | None = None,
    *,
    assigned_parquet: Path | None = None,
) -> gpd.GeoDataFrame:
    """Load OBIS anthozoa occurrences as a point GeoDataFrame.

    If ``assigned_parquet`` exists and ``path`` is not given, load the cached
    ecoregion-assigned parquet instead of the raw CSV.
    """
    parquet_path = assigned_parquet or OBIS_ASSIGNED_PARQUET
    csv_path = path or OBIS_CSV

    if path is None and parquet_path.exists():
        return gpd.read_parquet(parquet_path)

    df = pd.read_csv(csv_path, usecols=_OBIS_USECOLS, low_memory=False).rename(
        columns=str.lower
    )
    df = df.dropna(subset=["decimallatitude", "decimallongitude", "scientificname"])
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["decimallongitude"], df["decimallatitude"]),
        crs=GEOGRAPHIC_CRS,
    )


def prepare_redlist_species(
    species_gdf: gpd.GeoDataFrame,
    *,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
    species_id_col: str = "id_no",
) -> gpd.GeoDataFrame:
    """Keep one valid geometry per species within the analysis bounds."""
    species = _to_geographic(species_gdf.rename(columns=str.lower))
    sid = species_id_col.lower()
    if sid not in species.columns:
        raise ValueError(f"Species layer missing column: {sid!r}")

    species = species.loc[species.geometry.notna() & ~species.geometry.is_empty].copy()
    clip = box(*bounds)
    species = species.loc[species.intersects(clip)].copy()
    return species.drop_duplicates(subset=sid, keep="first").reset_index(drop=True)


def list_redlist_species_by_region(
    species_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame,
    *,
    species_id_col: str = "id_no",
    species_name_col: str = "sci_name",
    region_col: str = "ecoregion",
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
    fix_dateline: bool = True,
) -> pd.DataFrame:
    """Return unique Red List species present in each COTW ecoregion."""
    species = prepare_redlist_species(
        species_gdf, bounds=bounds, species_id_col=species_id_col
    )
    regions = (
        fix_cotw_dateline_geometries(regions_gdf.rename(columns=str.lower))
        if fix_dateline
        else _to_geographic(regions_gdf.rename(columns=str.lower))
    )

    sid = species_id_col.lower()
    sname = species_name_col.lower()
    rid = region_col.lower()
    if sname not in species.columns:
        raise ValueError(f"Species layer missing column: {sname!r}")
    if rid not in regions.columns:
        raise ValueError(f"Region layer missing column: {rid!r}")

    joined = gpd.sjoin(
        species[[sid, sname, "geometry"]],
        regions[[rid, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    out = (
        joined[[rid, sid, sname]]
        .drop_duplicates()
        .rename(
            columns={
                rid: "ecoregion",
                sid: "species_id",
                sname: "species_name",
            }
        )
        .sort_values(["ecoregion", "species_name"])
        .reset_index(drop=True)
    )
    out["species_name_norm"] = out["species_name"].map(normalize_species_name)
    out["source"] = "redlist"
    return out


def assign_obis_to_regions(
    obis_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame | None = None,
    *,
    region_col: str = "ecoregion",
    shapefile: Path | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Assign OBIS occurrence points to COTW ecoregion polygons.

    Uses the shapefile's native Pacific-centered Mercator CRS for assignment.
    Dateline-fixed WGS84 geometries are for plotting only; reprojecting them to
    EPSG:6933 can make Pacific ecoregions spuriously overlap (false ``M`` codes).
    """
    from src.processing import processdata

    if region_col in obis_gdf.columns and obis_gdf[region_col].notna().any():
        return obis_gdf

    regions = regions_gdf if regions_gdf is not None else load_cotw_ecoregions_for_assignment(
        shapefile
    )
    regions = regions.rename(columns=str.lower)
    if regions.crs is None:
        raise ValueError("COTW ecoregion layer must have a CRS for point assignment")

    return processdata.assign_points_to_shapes(
        obis_gdf,
        regions,
        cols_to_add=[region_col],
        assignment_col=None,
        projected_crs=str(regions.crs),
        verbose=verbose,
    )


def list_obis_species_by_region(
    regions_gdf: gpd.GeoDataFrame,
    obis_gdf: gpd.GeoDataFrame | None = None,
    *,
    region_col: str = "ecoregion",
    species_id_col: str = "speciesid",
    species_name_col: str = "scientificname",
    assign_if_needed: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return unique OBIS species recorded in each COTW ecoregion."""
    occurrences = obis_gdf if obis_gdf is not None else load_obis_occurrences()
    if assign_if_needed:
        occurrences = assign_obis_to_regions(
            occurrences, regions_gdf, region_col=region_col, verbose=verbose
        )

    rid = region_col.lower()
    sid = species_id_col.lower()
    sname = species_name_col.lower()
    cols = {c.lower() for c in occurrences.columns}
    if rid not in cols:
        raise ValueError(f"OBIS data missing assigned region column: {rid!r}")
    if sname not in cols:
        raise ValueError(f"OBIS data missing column: {sname!r}")

    df = pd.DataFrame(occurrences.drop(columns="geometry", errors="ignore")).rename(
        columns=str.lower
    )
    df = df.dropna(subset=[rid, sname])
    df["species_key"] = df[sid].where(df[sid].notna(), df[sname].map(normalize_species_name))
    df = df.dropna(subset=["species_key"]).drop_duplicates(subset=[rid, "species_key"])

    out = df[[rid, sid, sname]].rename(
        columns={rid: "ecoregion", sid: "species_id", sname: "species_name"}
    )
    out = out.sort_values(["ecoregion", "species_name"]).reset_index(drop=True)
    out["species_name_norm"] = out["species_name"].map(normalize_species_name)
    out["source"] = "obis"
    return out


def load_cotw_species(path: Path | None = None) -> pd.DataFrame:
    """Load scraped COTW species lists (one row per ecoregion/species)."""
    csv_path = path or COTW_SPECIES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"COTW species CSV not found: {csv_path}")
    return pd.read_csv(csv_path).rename(columns=str.lower)


def list_cotw_species_by_region(
    regions_gdf: gpd.GeoDataFrame | None = None,
    cotw_species_df: pd.DataFrame | None = None,
    *,
    region_col: str = "ecoregion",
    website_region_col: str = "ecoregion_name",
    species_name_col: str = "species_name",
    species_id_col: str = "species_slug",
    shapefile: Path | None = None,
) -> pd.DataFrame:
    """Return unique COTW website species listed for each shapefile ecoregion."""
    species_df = cotw_species_df if cotw_species_df is not None else load_cotw_species()
    if regions_gdf is None:
        regions_gdf = load_cotw_ecoregions_for_assignment(shapefile)

    rid = region_col.lower()
    web_rid = website_region_col.lower()
    sname = species_name_col.lower()
    sid = species_id_col.lower()
    regions = regions_gdf.rename(columns=str.lower)
    if rid not in regions.columns:
        raise ValueError(f"Region layer missing column: {rid!r}")
    if web_rid not in species_df.columns:
        raise ValueError(f"COTW species layer missing column: {web_rid!r}")
    if sname not in species_df.columns:
        raise ValueError(f"COTW species layer missing column: {sname!r}")

    shapefile_names = regions[rid].dropna().astype(str).unique().tolist()
    lookup = _website_ecoregion_candidates(shapefile_names)
    df = species_df.rename(columns=str.lower).copy()
    df[rid] = df[web_rid].map(
        lambda name: map_website_ecoregion_to_shapefile(
            str(name), shapefile_names, lookup=lookup
        )
    )
    df = df.dropna(subset=[rid, sname])
    if sid not in df.columns:
        df[sid] = df[sname].map(normalize_species_name)

    out = (
        df[[rid, sid, sname, web_rid]]
        .drop_duplicates(subset=[rid, sid])
        .rename(
            columns={
                rid: "ecoregion",
                sid: "species_id",
                sname: "species_name",
                web_rid: "cotw_ecoregion_name",
            }
        )
        .sort_values(["ecoregion", "species_name"])
        .reset_index(drop=True)
    )
    out["species_name_norm"] = out["species_name"].map(normalize_species_name)
    out["source"] = "cotw"
    return out


def compare_ecoregion_species_lists(
    redlist_species_df: pd.DataFrame,
    obis_species_df: pd.DataFrame,
    cotw_species_df: pd.DataFrame | None = None,
    *,
    region_col: str = "ecoregion",
    name_col: str = "species_name_norm",
) -> pd.DataFrame:
    """Build a presence table for cross-dataset comparison per ecoregion."""
    rid = region_col
    source_frames: list[tuple[str, pd.DataFrame]] = [
        ("redlist", redlist_species_df),
        ("obis", obis_species_df),
    ]
    if cotw_species_df is not None:
        source_frames.append(("cotw", cotw_species_df))

    long_parts: list[pd.DataFrame] = []
    for source, df in source_frames:
        tmp = df[[rid, name_col, "species_name"]].drop_duplicates(subset=[rid, name_col])
        tmp = tmp.assign(source=source)
        long_parts.append(tmp)

    long = pd.concat(long_parts, ignore_index=True)
    presence = long.groupby([rid, name_col], as_index=False).agg(
        in_redlist=("source", lambda s: (s == "redlist").any()),
        in_obis=("source", lambda s: (s == "obis").any()),
        in_cotw=("source", lambda s: (s == "cotw").any()),
    )

    for source in ("redlist", "obis", "cotw"):
        sub = (
            long.loc[long["source"].eq(source), [rid, name_col, "species_name"]]
            .drop_duplicates(subset=[rid, name_col])
            .rename(columns={"species_name": f"{source}_species_name"})
        )
        presence = presence.merge(sub, on=[rid, name_col], how="left")

    if cotw_species_df is not None and "cotw_ecoregion_name" in cotw_species_df.columns:
        cotw_regions = cotw_species_df[[rid, name_col, "cotw_ecoregion_name"]].drop_duplicates(
            subset=[rid, name_col]
        )
        presence = presence.merge(cotw_regions, on=[rid, name_col], how="left")

    presence["in_both"] = presence["in_redlist"] & presence["in_obis"]
    presence["in_all_three"] = (
        presence["in_redlist"] & presence["in_obis"] & presence["in_cotw"]
    )
    presence["n_sources"] = presence[["in_redlist", "in_obis", "in_cotw"]].sum(axis=1)
    return presence.sort_values([rid, name_col]).reset_index(drop=True)


def summarize_ecoregion_species_agreement(
    presence_df: pd.DataFrame,
    *,
    region_col: str = "ecoregion",
) -> pd.DataFrame:
    """Summarise agreement counts per ecoregion from a presence table."""
    rid = region_col
    summary = presence_df.groupby(rid, as_index=False).agg(
        n_redlist=("in_redlist", "sum"),
        n_obis=("in_obis", "sum"),
        n_cotw=("in_cotw", "sum"),
        n_redlist_obis=("in_both", "sum"),
        n_all_three=("in_all_three", "sum"),
        n_union=("in_redlist", "size"),
    )
    summary["n_redlist_only"] = summary["n_redlist"] - summary["n_redlist_obis"]
    summary["n_obis_only"] = summary["n_obis"] - summary["n_redlist_obis"]
    summary["jaccard_redlist_obis"] = summary["n_redlist_obis"] / summary["n_union"].where(
        summary["n_union"] > 0
    )
    if presence_df["in_cotw"].any():
        summary["jaccard_all_three"] = summary["n_all_three"] / summary["n_union"].where(
            summary["n_union"] > 0
        )
    return summary.sort_values("n_union", ascending=False).reset_index(drop=True)


def count_species_by_region(
    species_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame,
    *,
    species_id_col: str = "id_no",
    region_col: str = "ecoregion",
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
) -> pd.Series:
    """Count unique species whose Red List range intersects each region polygon."""
    species_list = list_redlist_species_by_region(
        species_gdf,
        regions_gdf,
        species_id_col=species_id_col,
        region_col=region_col,
        bounds=bounds,
    )
    counts = species_list.groupby("ecoregion")["species_id"].nunique().sort_index()
    counts.name = "species_count"
    return counts


def count_unique_species_by_region(
    species_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame,
    **kwargs,
) -> pd.Series:
    """Alias for :func:`count_species_by_region`."""
    return count_species_by_region(species_gdf, regions_gdf, **kwargs)
