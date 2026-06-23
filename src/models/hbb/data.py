from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import geopandas as gpd
import pandas as pd

from src import config
from src.models.hbb._config import (
    CLEAN_REQUIRED,
    FEATURE_VARS,
    LOAD_NA_COLS,
    SULLY_DATA_DIR,
)

__all__ = [
    "clean_data",
    "filter_model_ready_rows",
    "load_data",
    "load_model_data_for_cv",
    "load_model_data_from_maps",
    "load_model_data_from_pipeline",
    "standardize_train_test",
    "standardize_variables",
]


def _drop_na(df: pd.DataFrame, cols: list[str], report: bool = False) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        n0 = len(df)
        df = df[~df[col].isna()]
        if report and n0 != len(df):
            print(f"{col} removed {n0 - len(df)} row(s)")
    return df


def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else SULLY_DATA_DIR / "data.csv"
    df = pd.read_csv(path).rename(columns=str.lower)
    df["lat"] = df["latitude.degrees"]
    df["lon"] = df["longitude.degrees"]
    if len(df) > 0:
        print(f"df len: {len(df)}")
    df = _drop_na(df, LOAD_NA_COLS, report=True)
    n0 = len(df)
    df = df[df["average_coral_cover"] > 0]
    df = df[df["turbidity_mean"] < 0.35]
    print(f"Row count after filters: {len(df)} (dropped {n0 - len(df)})")

    gdf = gpd.read_file(
        config.sully_og_dir / "shapefiles" / "ecoregion_exportPolygon.shp"
    )
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    if pts.crs != gdf.crs:
        pts = pts.to_crs(gdf.crs)
    joined = gpd.sjoin(pts, gdf, how="left", predicate="intersects")
    out = joined[joined.ERG.notna()].copy()
    print(f"After shapefile join: {len(out)} rows")
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().rename(columns=str.lower)
    df = df[df["average_coral_cover"].notna() & (df["average_coral_cover"] > 0)]
    df = _drop_na(df, CLEAN_REQUIRED)
    if "turbidity_mean" in df.columns:
        df = df[df["turbidity_mean"] < 0.35]
    if "erg" in df.columns:
        df = df[df["erg"].notna()]
    return df.reset_index(drop=True)


def standardize_variables(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    df = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        m, s = df[col].dropna().mean(), df[col].dropna().std()
        stats[col] = (m, s)
        df[f"{col}_stzd"] = (df[col] - m) / s
    return df, stats


def standardize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    columns = columns or FEATURE_VARS
    train_std, stats = standardize_variables(train_df, columns)
    test_std = test_df.copy()
    for col, (m, s) in stats.items():
        test_std[f"{col}_stzd"] = (test_df[col] - m) / s
    return {"train": train_std, "test": test_std, "stats": stats}


def filter_model_ready_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out["average_coral_cover"].max() > 1.5:
        out["average_coral_cover"] /= 100.0
    div = (
        "diversity.standardized"
        if "diversity.standardized" in out.columns
        else "diversity"
    )
    need = FEATURE_VARS + ["average_coral_cover", "site", "region", div]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    mask = out[need].notna().all(axis=1) & (out["average_coral_cover"] > 0)
    return out.loc[mask].reset_index(drop=True)


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Need one of {names}")


def _norm_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def _apply_trusted_mapping(df: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    """Use data_for_maps.csv as the source of truth for reef-region metadata."""
    reef_col = _first_col(df, ("reef_id", "reef"))
    map_reef_col = _first_col(maps, ("Reef_ID", "reef_id", "reef"))
    map_eco_col = _first_col(
        maps, ("Ecoregion.x", "ecoregion.x", "Ecoregion", "ecoregion", "ERName")
    )

    required = ["diversity.standardized", "site", "region"]
    missing = [c for c in required if c not in maps.columns]
    if missing:
        raise ValueError(f"Trusted mapping missing columns: {missing}")

    cols = [map_reef_col, map_eco_col, "diversity.standardized", "site", "region"]
    if "ERG" in maps.columns:
        cols.append("ERG")
    if "ERName" in maps.columns:
        cols.append("ERName")

    trusted = maps[cols].copy()
    trusted["_reef_key"] = _norm_key(trusted[map_reef_col])
    conflict_counts = trusted.groupby("_reef_key")[map_eco_col].nunique(dropna=False)
    if (conflict_counts > 1).any():
        examples = ", ".join(conflict_counts[conflict_counts > 1].head().index)
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
        raise ValueError(
            f"Missing trusted data_for_maps mapping for {int(missing_trusted.sum())} "
            f"row(s). Example reef_id(s): {examples}"
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
        print(
            "WARNING: data_for_maps.csv ecoregion mapping overrides "
            f"{int(disagreement.sum())} row(s) from the raw/spatial join."
        )
        print(examples.head().to_string(index=False))
        # save examples to a file
        examples.to_csv("disagreement_examples.csv", index=False)

    out[eco_col] = out["_trusted_ecoregion"]
    out["ecoregion"] = out["_trusted_ecoregion"]
    if "_trusted_ername" in out.columns:
        out["ERName"] = out["_trusted_ername"]
    if "_trusted_erg" in out.columns:
        out["ERG"] = out["_trusted_erg"]
    out["diversity.standardized"] = pd.to_numeric(
        out["_trusted_diversity"], errors="coerce"
    )
    out["site"] = pd.to_numeric(out["_trusted_site"], errors="raise").astype(int)
    out["region"] = pd.to_numeric(out["_trusted_region"], errors="raise").astype(int)

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


def load_model_data_from_maps(
    data_dir: Optional[Path] = None,
    maps_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Deprecated alias — use :func:`load_model_data_for_cv`."""
    return load_model_data_for_cv(data_dir=data_dir, lookup_path=maps_path)


def load_model_data_from_pipeline(
    data_dir: Optional[Path] = None,
    diversity_lookup_path: Optional[Path] = None,
    force_rebuild: bool = False,
    index_source: Literal["paper_factor", "data_for_maps"] = "paper_factor",
) -> pd.DataFrame:
    """Model-ready data from the shared cached build (``data.csv`` + lookup)."""
    from src.dataloading.build_sully_model_ready_data import (
        load_model_ready_data,
        to_hbb_frame,
    )

    df = load_model_ready_data(
        data_dir=data_dir,
        lookup_path=diversity_lookup_path,
        force_rebuild=force_rebuild,
        index_source=index_source,
    )
    return to_hbb_frame(df)


def load_model_data_for_cv(
    data_dir: Optional[Path] = None,
    *,
    lookup_path: Optional[Path] = None,
    force_rebuild: bool = False,
    index_source: Literal["paper_factor", "data_for_maps"] = "paper_factor",
) -> pd.DataFrame:
    """Load shared cached model-ready data in HBB (lowercase) column format."""
    from src.dataloading.build_sully_model_ready_data import (
        load_model_ready_data,
        to_hbb_frame,
    )

    df = load_model_ready_data(
        data_dir=data_dir,
        lookup_path=lookup_path,
        force_rebuild=force_rebuild,
        index_source=index_source,
    )
    return to_hbb_frame(df)
