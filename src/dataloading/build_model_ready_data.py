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
from typing import Any, Optional

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


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Need one of {names}")


def _norm_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


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
        examples = ", ".join(conflict_counts[conflict_counts > 1].head().index.astype(str))
        raise ValueError(f"Conflicting trusted ecoregion mappings for reefs: {examples}")

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
        disagreement |= _norm_key(out[col]) != _norm_key(out["_trusted_ecoregion"])
    if disagreement.any():
        examples = out.loc[
            disagreement, [reef_col] + eco_candidates + ["_trusted_ecoregion"]
        ].drop_duplicates()
        warnings.warn(
            "Trusted lookup overrides "
            f"{int(disagreement.sum())} row(s) from the spatial join.",
            stacklevel=2,
        )
        if disagreement_path is not None:
            examples.to_csv(disagreement_path, index=False)

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


def _spatial_join_ecoregions(df: pd.DataFrame, shapefile: Path) -> pd.DataFrame:
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
    out = joined[joined["ERG"].notna()].copy()
    if "index_right" in out.columns:
        out = out.drop(columns=["index_right"])
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


def build_model_ready_data(
    *,
    data_dir: Optional[Path] = None,
    raw_data_path: Optional[Path] = None,
    lookup_path: Optional[Path] = None,
    shapefile_path: Optional[Path] = None,
    skip_shapefile: bool = False,
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
        warnings.warn(f"Shapefile not found at {shp}; skipping spatial join.", stacklevel=2)

    df = _filter_pre_mapping(df)
    print(f"After pre-mapping filters: {len(df):,}/{n0:,} rows")

    maps = pd.read_csv(lookup)
    df = _apply_trusted_mapping(
        df,
        maps,
        disagreement_path=data_dir / "model_ready_mapping_disagreements.csv",
    )
    out = filter_model_ready_rows(df)
    print(
        f"Model-ready dataset: {len(out):,} rows | "
        f"{out['site'].nunique():,} sites | {out['region'].nunique():,} regions"
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
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(len(df)),
        "n_sites": int(df["site"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "cache_path": str(cache_path),
        "sources": {name: str(path) for name, path in sources.items()},
        "source_mtimes": {
            name: path.stat().st_mtime for name, path in sources.items()
        },
    }
    cache_path.with_name(META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n"
    )


def load_model_ready_data(
    *,
    data_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    raw_data_path: Optional[Path] = None,
    lookup_path: Optional[Path] = None,
    force_rebuild: bool = False,
    skip_shapefile: bool = False,
) -> pd.DataFrame:
    """Load cached model-ready data, rebuilding when sources are newer."""
    data_dir = Path(data_dir or config.sully_og_dir)
    cache = Path(cache_path or model_ready_cache_path(data_dir))
    sources = _source_paths(data_dir, raw_data_path, lookup_path)

    if force_rebuild or _cache_is_stale(cache, sources):
        reason = "forced rebuild" if force_rebuild else "cache missing or stale"
        print(f"Building model-ready data ({reason})…")
        df = build_model_ready_data(
            data_dir=data_dir,
            raw_data_path=raw_data_path,
            lookup_path=lookup_path,
            skip_shapefile=skip_shapefile,
        )
        write_model_ready_cache(df, cache, sources=sources)
        print(f"Cached → {cache}")
        return df

    df = pd.read_csv(cache)
    print(f"Loaded cached model-ready data from {cache} (n={len(df):,})")
    return filter_model_ready_rows(df)


def to_hbb_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names for the hierarchical beta-GLMM Python pipeline."""
    out = df.copy()
    rename = {c: c.lower() for c in out.columns if c != "diversity.standardized"}
    out = out.rename(columns=rename)
    if "reef_id" not in out.columns and "reef" in out.columns:
        out["reef_id"] = out["reef"]
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/cache model-ready coral data")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--lookup", type=Path, default=None, help="Diversity lookup CSV")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache is fresh")
    parser.add_argument(
        "--skip-shapefile",
        action="store_true",
        help="Skip ecoregion shapefile spatial join",
    )
    args = parser.parse_args()

    load_model_ready_data(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        lookup_path=args.lookup,
        force_rebuild=args.force,
        skip_shapefile=args.skip_shapefile,
    )


if __name__ == "__main__":
    main()
