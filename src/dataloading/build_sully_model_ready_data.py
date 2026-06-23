#!/usr/bin/env python3
"""Build and cache the shared model-ready coral-cover dataset.

Pipeline (aligned with ``beta_model_reparam_utils.R``):
1. Load ``data.csv``
2. Optional ecoregion shapefile spatial join
3. Apply trusted reef metadata from ``data_for_maps.csv`` (diversity.standardized, site, region)
4. Filter to model-ready rows and write ``model_ready_data.csv``

All model families (baselines, beta-GLMM, CV) should load via :func:`load_model_ready_data`.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src import config

# Title-case covariates shared with src/models/coral_data.py and native R.
FEATURE_VARS = [
    "lat",
    "Depth",
    "Human_pop",
    "Cyclone",
    "SST_mean",
    "SSTA_Mean",
    "SSTA_min",
    "SSTA_freqstdev",
    "SSTA_dhwmax",
    "TSA_max",
    "TSA_freqstdev",
    "Turbidity_mean",
    "Historical_SST_max",
]

CACHE_FILENAME = "model_ready_data.csv"
META_FILENAME = "model_ready_data.meta.json"
DEFAULT_LOOKUP_FILENAME = "data_for_maps.csv"
SPATIAL_DISAGREEMENT_FILENAME = "spatial_trusted_ecoregion_disagreements.csv"

_PRE_SPATIAL_REQUIRED = [
    "Average_coral_cover",
    "SST_mean",
    "SSTA_stdev",
    "SSTA_freqmax",
    "SSTA_freqmean",
    "Turbidity_mean",
    "Cyclone",
    "Depth",
    "Historical_SST_max",
    "sst_mean_rcp85_2100",
]


def model_ready_cache_path(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or config.sully_og_dir) / CACHE_FILENAME


def diversity_lookup_path(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or config.sully_og_dir) / DEFAULT_LOOKUP_FILENAME


def spatial_trusted_disagreement_path(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or config.sully_og_dir) / SPATIAL_DISAGREEMENT_FILENAME


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Need one of {names}")


def _norm_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def _trusted_ecoregion_by_reef(maps: pd.DataFrame) -> pd.DataFrame:
    reef_col = _first_col(maps, ("Reef_ID", "reef_id", "reef"))
    eco_col = _first_col(
        maps, ("Ecoregion.x", "ecoregion.x", "Ecoregion", "ecoregion", "ERName")
    )
    trusted = maps[[reef_col, eco_col]].rename(
        columns={reef_col: "Reef_ID", eco_col: "reassigned_ecoregion"}
    )
    trusted["_reef_key"] = _norm_key(trusted["Reef_ID"])
    return trusted.drop_duplicates(subset="_reef_key", keep="first")


def write_spatial_trusted_ecoregion_disagreements(
    df: pd.DataFrame,
    trusted_lookup: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Write reef sites whose spatial ecoregion differs from ``data_for_maps``."""
    reef_col = _first_col(df, ("Reef_ID", "reef_id", "reef"))
    eco_col = _first_col(df, ("Ecoregion", "ecoregion"))

    work = df.copy()
    spatial = (
        work["spatial_ecoregion"].astype(str)
        if "spatial_ecoregion" in work.columns
        else work[eco_col].astype(str)
    )
    work["spatial_ecoregion"] = spatial
    valid_spatial = ~spatial.str.strip().str.lower().isin({"", "nan", "none", "<na>"})

    trusted = _trusted_ecoregion_by_reef(trusted_lookup)
    work["_reef_key"] = _norm_key(work[reef_col])
    work = work.merge(
        trusted[["_reef_key", "reassigned_ecoregion"]],
        on="_reef_key",
        how="left",
    )

    mapped = valid_spatial & work["reassigned_ecoregion"].notna()
    disagree = mapped & (
        _norm_key(work["spatial_ecoregion"]) != _norm_key(work["reassigned_ecoregion"])
    )

    out_cols = [reef_col, "spatial_ecoregion", "reassigned_ecoregion"]
    if "lon" in work.columns:
        out_cols.insert(1, "lon")
    if "Latitude.Degrees" in work.columns:
        out_cols.insert(2 if "lon" in work.columns else 1, "Latitude.Degrees")

    out = (
        work.loc[disagree, out_cols]
        .drop_duplicates(subset=reef_col)
        .sort_values(reef_col)
        .reset_index(drop=True)
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _to_proportion(cover: pd.Series) -> pd.Series:
    y = cover.astype(float)
    if y.max(skipna=True) > 1.5:
        y = y / 100.0
    return y


def filter_model_ready_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Final model-ready filter (R ``filter_model_ready_rows`` parity)."""
    out = df.copy()
    if "lat" not in out.columns and "Latitude.Degrees" in out.columns:
        out["lat"] = np.abs(out["Latitude.Degrees"].astype(float))
    out["Average_coral_cover"] = _to_proportion(out["Average_coral_cover"])

    mask = (
        out["Average_coral_cover"].notna()
        & (out["Average_coral_cover"] > 0)
        & out[FEATURE_VARS].notna().all(axis=1)
        & out["site"].notna()
        & out["region"].notna()
        & out["diversity.standardized"].notna()
    )
    out = out.loc[mask].reset_index(drop=True)
    out["site"] = out["site"].astype(int)
    out["region"] = out["region"].astype(int)
    return out


def assign_paper_site_region_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate ``my_1_run_the_beta_model.Rmd`` site/region codes after filtering.

    R source (chunk ``create a dataframe containing information for each site...``)::

        data$Reef_ID <- as.factor(as.character(as.factor(data$Reef_ID)))

        sites_and_region_df <- data %>% distinct(Reef_ID, Ecoregion) %>% ungroup()
        sites_and_region_df$site <- as.numeric(as.factor(sites_and_region_df$Reef_ID))
        sites_and_region_df$region <- as.numeric(as.factor(sites_and_region_df$Ecoregion))

        # region_diversity_df <- ...  (handled separately when building diversity vector)

        data <- left_join(data, sites_and_region_df[, c("Reef_ID", "site", "region")],
                          by = "Reef_ID")
    """
    # R: data  (post-filter observation table; one row per reef-year / survey)
    out = df.copy()

    # R: data$Reef_ID and data$Ecoregion  (accept lowercase aliases in cached CSVs)
    reef_col = _first_col(out, ("Reef_ID", "reef_id", "reef"))
    eco_col = _first_col(out, ("Ecoregion", "ecoregion"))

    # R: levels(as.factor(sites_and_region_df$Reef_ID))
    #     as.factor() on the distinct(Reef_ID, Ecoregion) table; we use sorted
    #     unique strings so codes are stable/reproducible (R uses first-appearance order).
    reef_levels = sorted(out[reef_col].astype(str).unique())

    # R: levels(as.factor(sites_and_region_df$Ecoregion))
    eco_levels = sorted(out[eco_col].astype(str).unique())

    # R: sites_and_region_df$site <- as.numeric(as.factor(sites_and_region_df$Reef_ID))
    #     then left_join(..., by = "Reef_ID")  — 1-based integer site index per reef
    out["site"] = (
        pd.Categorical(out[reef_col].astype(str), categories=reef_levels).codes + 1
    )

    # R: sites_and_region_df$region <- as.numeric(as.factor(sites_and_region_df$Ecoregion))
    #     then left_join(..., by = "Reef_ID")  — 1-based integer region index per ecoregion
    out["region"] = (
        pd.Categorical(out[eco_col].astype(str), categories=eco_levels).codes + 1
    )
    return out


def _apply_trusted_mapping(
    df: pd.DataFrame,
    maps: pd.DataFrame,
    *,
    disagreement_path: Optional[Path] = None,
) -> pd.DataFrame:
    reef_col = _first_col(df, ("Reef_ID", "reef_id", "reef"))
    map_reef_col = _first_col(maps, ("Reef_ID", "reef_id", "reef"))
    map_eco_col = _first_col(
        maps, ("Ecoregion.x", "ecoregion.x", "Ecoregion", "ecoregion", "ERName")
    )

    required = ["diversity.standardized", "site", "region"]
    missing = [c for c in required if c not in maps.columns]
    if missing:
        raise ValueError(f"Trusted lookup missing columns: {missing}")

    cols = [map_reef_col, map_eco_col, "diversity.standardized", "site", "region"]
    if "ERG" in maps.columns:
        cols.append("ERG")
    if "ERName" in maps.columns:
        cols.append("ERName")

    trusted = maps[cols].copy()
    trusted["_reef_key"] = _norm_key(trusted[map_reef_col])
    conflict_counts = trusted.groupby("_reef_key")[map_eco_col].nunique(dropna=False)
    if (conflict_counts > 1).any():
        examples = ", ".join(
            conflict_counts[conflict_counts > 1].head().index.astype(str)
        )
        raise ValueError(
            f"Conflicting trusted ecoregion mappings for reefs: {examples}"
        )

    trusted = trusted.drop_duplicates(subset="_reef_key", keep="first")
    rename = {
        map_eco_col: "_trusted_ecoregion",
        "diversity.standardized": "_trusted_diversity",
        "site": "_trusted_site",
        "region": "_trusted_region",
    }
    if "ERG" in trusted.columns:
        rename["ERG"] = "_trusted_erg"
    if "ERName" in trusted.columns:
        rename["ERName"] = "_trusted_ername"
    trusted = trusted.rename(columns=rename)

    out = df.copy()
    out["_reef_key"] = _norm_key(out[reef_col])
    out = out.merge(
        trusted[
            [
                "_reef_key",
                "_trusted_ecoregion",
                "_trusted_diversity",
                "_trusted_site",
                "_trusted_region",
            ]
            + [c for c in ("_trusted_erg", "_trusted_ername") if c in trusted.columns]
        ],
        on="_reef_key",
        how="left",
    )

    missing_trusted = out["_trusted_ecoregion"].isna()
    if missing_trusted.any():
        examples = ", ".join(out.loc[missing_trusted, reef_col].astype(str).head())
        warnings.warn(
            f"No trusted lookup mapping for {int(missing_trusted.sum())} "
            f"row(s); they will be dropped. Example reef_id(s): {examples}",
            stacklevel=2,
        )

    eco_col = _first_col(out, ("Ecoregion", "ecoregion"))
    eco_candidates = [c for c in ("Ecoregion", "ecoregion") if c in out.columns]
    disagreement = pd.Series(False, index=out.index)
    for col in eco_candidates:
        mapped = out[col].notna() & out["_trusted_ecoregion"].notna()
        disagreement |= mapped & (
            _norm_key(out[col]) != _norm_key(out["_trusted_ecoregion"])
        )
    if disagreement.any():
        warnings.warn(
            "Trusted lookup overrides "
            f"{int(disagreement.sum())} row(s) from the spatial join.",
            stacklevel=2,
        )
        if disagreement_path is not None:
            try:
                written = write_spatial_trusted_ecoregion_disagreements(
                    out.assign(
                        spatial_ecoregion=out.get("spatial_ecoregion", out[eco_col])
                    ),
                    maps,
                    disagreement_path,
                )
                if len(written) == 0:
                    warnings.warn(
                        f"No spatial/trusted disagreements to write at {disagreement_path}.",
                        stacklevel=2,
                    )
            except OSError as exc:
                warnings.warn(
                    f"Could not write mapping disagreements to {disagreement_path}: {exc}",
                    stacklevel=2,
                )

    out[eco_col] = out["_trusted_ecoregion"]
    out["ecoregion"] = out["_trusted_ecoregion"]
    if "_trusted_ername" in out.columns:
        out["ERName"] = out["_trusted_ername"]
    if "_trusted_erg" in out.columns:
        out["ERG"] = out["_trusted_erg"]
    out["diversity.standardized"] = pd.to_numeric(
        out["_trusted_diversity"], errors="coerce"
    )
    out["site"] = pd.to_numeric(out["_trusted_site"], errors="coerce")
    out["region"] = pd.to_numeric(out["_trusted_region"], errors="coerce")

    return out.drop(
        columns=[
            "_reef_key",
            "_trusted_ecoregion",
            "_trusted_diversity",
            "_trusted_site",
            "_trusted_region",
            "_trusted_erg",
            "_trusted_ername",
        ],
        errors="ignore",
    )


def _spatial_join_ecoregions(
    df: pd.DataFrame,
    shapefile: Path,
    *,
    trusted_lookup: Optional[pd.DataFrame] = None,
    disagreement_path: Optional[Path] = None,
) -> pd.DataFrame:
    import geopandas as gpd

    gdf = gpd.read_file(shapefile)
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    if pts.crs != gdf.crs:
        pts = pts.to_crs(gdf.crs)
    joined = gpd.sjoin(pts, gdf, how="left", predicate="intersects")
    out = joined.copy()
    if "index_right" in out.columns:
        out = out.drop(columns=["index_right"])
    # Match the R/terra pipeline: shapefile attributes overwrite raw columns,
    # but rows with missing spatial ERG are kept until the trusted lookup is applied.
    if "Ecoregion_right" in out.columns:
        out["Ecoregion"] = out["Ecoregion_right"]
        out = out.drop(columns=["Ecoregion_right"])
    if "Ecoregion_left" in out.columns:
        out = out.drop(columns=["Ecoregion_left"])
    if "ERG_right" in out.columns:
        out["ERG"] = out["ERG_right"]
        out = out.drop(columns=["ERG_right"])
    if "ERG_left" in out.columns:
        out = out.drop(columns=["ERG_left"])

    eco_col = _first_col(out, ("Ecoregion", "ecoregion"))
    out["spatial_ecoregion"] = out[eco_col]

    if trusted_lookup is not None and disagreement_path is not None:
        written = write_spatial_trusted_ecoregion_disagreements(
            out, trusted_lookup, disagreement_path
        )
        print(
            f"Wrote {len(written):,} spatial/trusted ecoregion disagreements "
            f"→ {disagreement_path}"
        )

    return pd.DataFrame(out)


def _filter_pre_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in _PRE_SPATIAL_REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"data.csv missing required columns: {missing}")

    mask = out["Average_coral_cover"].notna() & (out["Average_coral_cover"] > 0)
    for col in _PRE_SPATIAL_REQUIRED[1:]:
        mask &= out[col].notna()
    mask &= out["Turbidity_mean"] < 0.35
    return out.loc[mask].reset_index(drop=True)


def build_sully_model_ready_data(
    *,
    data_dir: Optional[Path] = None,
    raw_data_path: Optional[Path] = None,
    lookup_path: Optional[Path] = None,
    shapefile_path: Optional[Path] = None,
    skip_shapefile: bool = False,
    index_source: Literal["paper_factor", "data_for_maps"] = "paper_factor",
) -> pd.DataFrame:
    """Build model-ready data from ``data.csv`` and the diversity/site lookup."""
    data_dir = Path(data_dir or config.sully_og_dir)
    raw_path = Path(raw_data_path or data_dir / "data.csv")
    lookup = Path(lookup_path or diversity_lookup_path(data_dir))
    shp = Path(
        shapefile_path or data_dir / "shapefiles" / "ecoregion_exportPolygon.shp"
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
    if not lookup.exists():
        raise FileNotFoundError(f"Missing diversity lookup: {lookup}")

    df = pd.read_csv(raw_path)
    n0 = len(df)
    df["row_id"] = np.arange(n0)
    if "Reef_ID" in df.columns:
        df["reef"] = df["Reef_ID"]
    df["lat"] = np.abs(df["Latitude.Degrees"].astype(float))
    df["lon"] = df["Longitude.Degrees"].astype(float)
    df["Longitude"] = df["lon"]

    maps = pd.read_csv(lookup)
    disagreement_path = spatial_trusted_disagreement_path(data_dir)

    if not skip_shapefile and shp.exists():
        try:
            df = _spatial_join_ecoregions(df, shp)
            print(f"After shapefile join: {len(df):,} rows (from {n0:,})")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Shapefile join failed ({exc}); continuing without spatial join.",
                stacklevel=2,
            )
    elif not skip_shapefile:
        warnings.warn(
            f"Shapefile not found at {shp}; skipping spatial join.", stacklevel=2
        )

    df = _filter_pre_mapping(df)
    print(f"After pre-mapping filters: {len(df):,}/{n0:,} rows")

    if "spatial_ecoregion" in df.columns:
        written = write_spatial_trusted_ecoregion_disagreements(
            df, maps, disagreement_path
        )
        print(
            f"Wrote {len(written):,} spatial/trusted ecoregion disagreements "
            f"→ {disagreement_path}"
        )

    df = _apply_trusted_mapping(
        df,
        maps,
        disagreement_path=None,
    )
    if index_source == "paper_factor":
        df = assign_paper_site_region_indices(df)
    out = filter_model_ready_rows(df)
    print(
        f"Model-ready dataset: {len(out):,} rows | "
        f"{out['site'].nunique():,} sites | {out['region'].nunique():,} regions | "
        f"index_source={index_source}"
    )
    return out


def _source_paths(
    data_dir: Path,
    raw_data_path: Optional[Path],
    lookup_path: Optional[Path],
) -> dict[str, Path]:
    return {
        "data.csv": Path(raw_data_path or data_dir / "data.csv"),
        DEFAULT_LOOKUP_FILENAME: Path(lookup_path or diversity_lookup_path(data_dir)),
    }


def _cache_is_stale(
    cache_path: Path,
    sources: dict[str, Path],
) -> bool:
    if not cache_path.exists():
        return True
    cache_mtime = cache_path.stat().st_mtime
    return any(path.stat().st_mtime > cache_mtime for path in sources.values())


def write_model_ready_cache(
    df: pd.DataFrame,
    cache_path: Path,
    *,
    sources: dict[str, Path],
    index_source: str = "paper_factor",
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(len(df)),
        "n_sites": int(df["site"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "index_source": index_source,
        "cache_path": str(cache_path),
        "sources": {name: str(path) for name, path in sources.items()},
        "source_mtimes": {name: path.stat().st_mtime for name, path in sources.items()},
    }
    cache_path.with_name(META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")


def load_model_ready_data(
    *,
    data_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    raw_data_path: Optional[Path] = None,
    lookup_path: Optional[Path] = None,
    force_rebuild: bool = False,
    skip_shapefile: bool = False,
    index_source: Literal["paper_factor", "data_for_maps"] = "paper_factor",
) -> pd.DataFrame:
    """Load cached model-ready data, rebuilding when sources are newer."""
    data_dir = Path(data_dir or config.sully_og_dir)
    cache = Path(cache_path or model_ready_cache_path(data_dir))
    sources = _source_paths(data_dir, raw_data_path, lookup_path)

    meta_path = cache.with_name(META_FILENAME)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("index_source") != index_source:
                force_rebuild = True
        except Exception:  # noqa: BLE001
            force_rebuild = True

    if force_rebuild or _cache_is_stale(cache, sources):
        reason = "forced rebuild" if force_rebuild else "cache missing or stale"
        print(f"Building model-ready data ({reason})…")
        df = build_sully_model_ready_data(
            data_dir=data_dir,
            raw_data_path=raw_data_path,
            lookup_path=lookup_path,
            skip_shapefile=skip_shapefile,
            index_source=index_source,
        )
        try:
            write_model_ready_cache(
                df, cache, sources=sources, index_source=index_source
            )
            print(f"Cached → {cache}")
        except OSError as exc:
            warnings.warn(
                f"Could not write model-ready cache to {cache}: {exc}. "
                "Continuing with in-memory data.",
                stacklevel=2,
            )
        return df

    df = pd.read_csv(cache)
    print(f"Loaded cached model-ready data from {cache} (n={len(df):,})")
    return filter_model_ready_rows(df)


def to_hbb_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names for the hierarchical beta-GLMM Python pipeline."""
    out = df.copy()
    rename = {c: c.lower() for c in out.columns if c != "diversity.standardized"}
    out = out.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    if "reef_id" not in out.columns and "reef" in out.columns:
        out["reef_id"] = out["reef"]
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/cache model-ready coral data")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--lookup", type=Path, default=None, help="Diversity lookup CSV"
    )
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if cache is fresh"
    )
    parser.add_argument(
        "--skip-shapefile",
        action="store_true",
        help="Skip ecoregion shapefile spatial join",
    )
    parser.add_argument(
        "--index-source",
        choices=["paper_factor", "data_for_maps"],
        default="paper_factor",
        help="How to assign site/region IDs after trusted ecoregion mapping.",
    )
    args = parser.parse_args()

    load_model_ready_data(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        lookup_path=args.lookup,
        force_rebuild=args.force,
        skip_shapefile=args.skip_shapefile,
        index_source=args.index_source,
    )


if __name__ == "__main__":
    main()
