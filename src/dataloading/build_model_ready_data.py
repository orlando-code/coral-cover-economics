#!/usr/bin/env python3
"""Build Reef Check survey table with per-point environmental covariates.

Pipeline (from ``notebooks/cover.ipynb`` + ``notebooks/env_data.ipynb``):
1. Load / QC Reef Check substrate → survey-level coral cover
2. Attach cyclone frequency, per-site QDM SST means (1850–1950 and 1985–2014)
3. Attach MEOW / COTW ecoregions, MPA protection, WorldPop coastal population density

Outputs ``reef_check/processed/reef_check_model_ready.csv`` by default.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

from src import config
from src.dataloading import cmip_sst

EPOCH_19811231 = pd.Timestamp("1981-12-31")
DATE_COLS = ["date"]
NUMERIC_COLS = ["year", "depth (m)", "total"]
ARCHIVED_REEF_ID_COL = "static_descriptors_reef_id (archived field)"

CACHE_FILENAME = "reef_check_model_ready.csv"
META_FILENAME = "reef_check_model_ready.meta.json"
CYCLONE_GRID_CACHE = "cyclone_freq_annual_mean.nc"
SST_MEANS_CACHE = "site_sst_means.parquet"
LOC_SST_HIST_CACHE = "loc_sst_mean_1850_1950_{model}.parquet"
DEFAULT_QDM_MODEL = "BCC-CSM2-MR"
DEFAULT_QDM_SCENARIO = "ssp370"
DEFAULT_POP_SSP = "SSP2"
HIST_SST_START = "1850-01-01"
HIST_SST_END = "1950-12-31"
HIST_SST_T_START = pd.Timestamp(HIST_SST_START)
HIST_SST_T_END = pd.Timestamp(HIST_SST_END)
# Mapped parquets are daily from 1850-01-01; historic_qmapped rows precede forecast_qdm.
HIST_SST_N_ROWS = 36_865


def _is_retryable_io_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "errno 60" in msg


def model_ready_cache_path(output_dir: Optional[Path] = None) -> Path:
    return Path(output_dir or config.reef_check_dir / "processed") / CACHE_FILENAME


def default_substrate_path() -> Path:
    return config.reef_check_dir / "021126" / "Substrate.csv"


def default_processed_snapshot_path(substrate_path: Path | None = None) -> Path:
    substrate_path = Path(substrate_path or default_substrate_path())
    return (
        config.reef_check_dir
        / "processed"
        / f"coral_covers_processed_{substrate_path.parent.name}.csv"
    )


def load_reef_check_substrate(path: Path) -> pd.DataFrame:
    """Load Reef Check substrate CSV (``cover.ipynb``)."""
    df = pd.read_csv(
        path,
        parse_dates=DATE_COLS,
        date_format=lambda x: pd.to_datetime(x, format="%Y-%m-%d", errors="coerce"),
        low_memory=False,
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["depth (m)"] = pd.to_numeric(df["depth (m)"], errors="coerce").astype("Float64")
    df["total"] = pd.to_numeric(df["total"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], format="%d-%B-%y", errors="coerce")

    latlon_cols = df["coordinates_in_decimal_degree_format"].str.split(",", expand=True)
    latlon_cols = latlon_cols.apply(pd.to_numeric, errors="coerce")
    latlon_cols.columns = ["latitude", "longitude"]
    df[["latitude", "longitude"]] = latlon_cols
    return df


def remove_substrate_errors(
    df: pd.DataFrame,
    completed_errors_csv: Path,
) -> pd.DataFrame:
    """Drop substrate rows flagged in the manual error CSV."""
    df = df[df["total"].notna()].copy()
    error_info = pd.read_csv(completed_errors_csv)
    seg_cols = [c for c in ["S1", "S2", "S3", "S4"] if c in error_info.columns]
    drop_rules = (
        error_info.assign(
            what_errors=lambda d: d["what_errors"].fillna("").str.strip().str.lower(),
            **{
                c: pd.to_numeric(error_info[c], errors="coerce").fillna(0).astype(int)
                for c in seg_cols
            },
        )
        .melt(
            id_vars="what_errors",
            value_vars=seg_cols,
            var_name="segment_code",
            value_name="keep",
        )
        .query("keep == 0")[["what_errors", "segment_code"]]
        .drop_duplicates()
    )

    df["what_errors"] = df["what_errors"].fillna("").str.strip().str.lower()
    df["segment_code"] = (
        df["segment_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^SEGMENT\s*", "", regex=True)
        .str.replace(r"^S\s*", "S", regex=True)
        .str.replace(r"^(\d)$", r"S\1", regex=True)
    )
    n0 = len(df)
    df = df.merge(
        drop_rules.assign(_drop=True), on=["what_errors", "segment_code"], how="left"
    )
    df = df[df["_drop"].isna()].drop(columns="_drop")
    print(f"Error filter: kept {len(df):,}/{n0:,} substrate rows")
    return df


def filter_substrate_total_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop (survey, segment) groups whose quadrat totals are not 40."""
    df = df.copy()
    gb = df.groupby(["survey_id", "segment_code"])["total"].sum()
    bad_keys = set(gb.index[gb != 40])
    mask = ~df.set_index(["survey_id", "segment_code"]).index.isin(bad_keys)
    print(
        f"Segment-total filter: dropped {len(df) - mask.sum():,} rows "
        f"({100 * (len(df) - mask.sum()) / len(df):.2f}%)"
    )
    return df.loc[mask].reset_index(drop=True)


def calculate_substrate_cover(
    df: pd.DataFrame, *, substrate_code: str = "HC"
) -> pd.Series:
    """Proportional hard-coral cover per survey (mean over segments)."""
    segment_cover = (
        df.query(f"substrate_code == '{substrate_code}'")
        .groupby(["survey_id", "segment_code"])["total"]
        .sum()
        .div(40)
    )
    return segment_cover.groupby("survey_id").mean().rename("coral_cover")


def build_reef_check_surveys(
    *,
    substrate_path: Path | None = None,
    errors_csv: Path | None = None,
    processed_path: Path | None = None,
    use_processed_cache: bool = True,
    force_reprocess: bool = False,
) -> pd.DataFrame:
    """Survey-level Reef Check table with coral cover and coordinates."""
    substrate_path = Path(substrate_path or default_substrate_path())
    processed_path = Path(processed_path or default_processed_snapshot_path(substrate_path))
    errors_csv = Path(
        errors_csv or config.sully_og_dir / "reefcheck_error_info_completed.csv"
    )

    if (
        use_processed_cache
        and not force_reprocess
        and processed_path.exists()
        and processed_path.stat().st_mtime >= substrate_path.stat().st_mtime
    ):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        print(f"Loaded processed Reef Check surveys from {processed_path} (n={len(df):,})")
        return _standardize_survey_frame(df)

    if not substrate_path.exists():
        raise FileNotFoundError(f"Missing Reef Check substrate file: {substrate_path}")
    if not errors_csv.exists():
        raise FileNotFoundError(f"Missing error QC file: {errors_csv}")

    raw = load_reef_check_substrate(substrate_path)
    cleaned = filter_substrate_total_errors(
        remove_substrate_errors(raw, errors_csv)
    )
    survey_cover = calculate_substrate_cover(cleaned)
    drop_cols = [
        "substrate_code",
        "segment_code",
        "total",
        "substrate_recorded_by",
        "errors",
        "what_errors",
    ]
    surveys = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])
    surveys = surveys.drop_duplicates(subset=["survey_id"]).copy()
    surveys["coral_cover"] = surveys["survey_id"].map(survey_cover)
    surveys["days_since_19811231"] = (
        pd.to_datetime(surveys["date"]) - EPOCH_19811231
    ).dt.days.astype(int)
    surveys["Depth"] = pd.to_numeric(surveys["depth (m)"], errors="coerce").astype(float)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    surveys.to_csv(processed_path, index=False)
    print(f"Wrote processed Reef Check surveys → {processed_path} (n={len(surveys):,})")
    return _standardize_survey_frame(surveys)


def _standardize_survey_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)
    out["Average_coral_cover"] = out["coral_cover"].astype(float)
    out["lat"] = np.abs(out["latitude"].astype(float))
    out["lon"] = out["longitude"].astype(float)
    out["Latitude.Degrees"] = out["latitude"].astype(float)
    out["Longitude.Degrees"] = out["longitude"].astype(float)
    if "year" not in out.columns and "date" in out.columns:
        out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
    return out


def _points_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )


def _unique_coords(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["latitude", "longitude", "lat", "lon"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _map_coord_values(
    df: pd.DataFrame,
    coord_values: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Map per-coordinate values back onto all survey rows."""
    out = df.merge(
        coord_values,
        on=["latitude", "longitude"],
        how="left",
        suffixes=("", "_coord"),
    )
    return out


def join_meow_ecoregions(df: pd.DataFrame, *, meow_dir: Path | None = None) -> pd.DataFrame:
    """Assign MEOW lat_zone / realm / province / ecoregion."""
    meow_dir = Path(meow_dir or config.meow_dir)
    shp_paths = sorted(meow_dir.glob("*.shp"))
    if not shp_paths:
        warnings.warn(f"No MEOW shapefile under {meow_dir}; skipping.", stacklevel=2)
        return df

    polys = gpd.read_file(shp_paths[0])
    polys.columns = [str(c).lower() for c in polys.columns]
    attr_cols = [c for c in ("lat_zone", "realm", "province", "ecoregion") if c in polys.columns]
    if not attr_cols:
        warnings.warn("MEOW layer missing expected columns; skipping.", stacklevel=2)
        return df

    pts = _points_gdf(df)
    if pts.crs != polys.crs:
        pts = pts.to_crs(polys.crs)
    joined = gpd.sjoin(pts, polys[attr_cols + ["geometry"]], how="left", predicate="within")
    for col in attr_cols:
        values = joined.groupby(joined.index)[col].first()
        df[col] = values.reindex(df.index).to_numpy()
    return df


def join_cotw_ecoregions(
    df: pd.DataFrame,
    *,
    shapefile: Path | None = None,
) -> pd.DataFrame:
    """Spatial join to COTW ecoregion polygons (Sully shapefile)."""
    shp = Path(
        shapefile
        or config.sully_og_dir / "ecoregion_shapefiles" / "ecoregion_exportPolygon.shp"
    )
    if not shp.exists():
        warnings.warn(f"COTW shapefile missing at {shp}; skipping.", stacklevel=2)
        return df
    polys = gpd.read_file(shp)
    pts = _points_gdf(df)
    if pts.crs != polys.crs:
        pts = pts.to_crs(polys.crs)
    joined = gpd.sjoin(pts, polys, how="left", predicate="intersects")
    eco_col = "Ecoregion" if "Ecoregion" in joined.columns else "ecoregion"
    erg_col = "ERG" if "ERG" in joined.columns else None
    df["cotw_ecoregion"] = joined[eco_col].to_numpy()
    if erg_col:
        df["cotw_erg"] = joined[erg_col].to_numpy()
    return df


def _build_cyclone_frequency_grid(
    cyclones_nc: Path,
    *,
    grid_step: float = 0.1,
) -> xr.DataArray:
    """Mean annual cyclone count grid (``env_data.ipynb``)."""
    ds = xr.open_dataset(cyclones_nc)
    try:
        valid = ds.lon.notnull() & ds.lat.notnull()
        storm_year = ds.time.dt.year.where(valid)
        track_df = (
            xr.Dataset(
                {
                    "year": storm_year,
                    "lon": ds.lon,
                    "lat": ds.lat,
                    "storm": ds.storm,
                    "time": ds.time,
                }
            )
            .to_dataframe()
            .reset_index()
            .dropna(subset=["year", "lon", "lat", "storm"])
        )
        track_df["year"] = track_df["year"].astype(int)

        def _interp_track(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < 2:
                return group[["year", "lon", "lat", "storm"]]
            group = group.sort_values("time")
            lons = group["lon"].to_numpy(dtype=float)
            lats = group["lat"].to_numpy(dtype=float)
            lons_unwrapped = np.unwrap(lons, period=360)
            interp_lons: list[float] = []
            interp_lats: list[float] = []
            for i in range(len(group) - 1):
                lon1, lon2 = lons_unwrapped[i], lons_unwrapped[i + 1]
                lat1, lat2 = lats[i], lats[i + 1]
                dist = np.hypot(lon2 - lon1, lat2 - lat1)
                n_pts = max(2, int(np.ceil(dist / (grid_step * 0.5))))
                interp_lons.extend(np.linspace(lon1, lon2, n_pts)[:-1].tolist())
                interp_lats.extend(np.linspace(lat1, lat2, n_pts)[:-1].tolist())
            interp_lons.append(float(lons_unwrapped[-1]))
            interp_lats.append(float(lats[-1]))
            interp_lons_arr = (np.array(interp_lons) + 180) % 360 - 180
            return pd.DataFrame(
                {
                    "year": group["year"].iloc[0],
                    "lon": interp_lons_arr,
                    "lat": interp_lats,
                    "storm": group["storm"].iloc[0],
                }
            )

        interp_df = pd.concat(
            [_interp_track(g) for _, g in track_df.groupby("storm")],
            ignore_index=True,
        )
        lon_edges = np.arange(-180, 180 + grid_step, grid_step)
        lat_edges = np.arange(-90, 90 + grid_step, grid_step)
        lon_centers = lon_edges[:-1] + grid_step / 2
        lat_centers = lat_edges[:-1] + grid_step / 2
        interp_df["lon_i"] = np.clip(
            np.digitize(interp_df["lon"], lon_edges) - 1, 0, len(lon_edges) - 2
        )
        interp_df["lat_i"] = np.clip(
            np.digitize(interp_df["lat"], lat_edges) - 1, 0, len(lat_edges) - 2
        )
        cell_year_storms = interp_df.drop_duplicates(
            subset=["year", "lat_i", "lon_i", "storm"]
        )
        storms_per_cell_year = (
            cell_year_storms.groupby(["year", "lat_i", "lon_i"])
            .size()
            .rename("n_storms")
            .reset_index()
        )
        mean_annual = storms_per_cell_year.groupby(["lat_i", "lon_i"])["n_storms"].mean()
        grid = np.zeros((len(lat_centers), len(lon_centers)), dtype=float)
        for (lat_i, lon_i), value in mean_annual.items():
            grid[int(lat_i), int(lon_i)] = value
        grid = gaussian_filter(grid, sigma=1.0)
        grid[grid < 1e-4] = np.nan
        return xr.DataArray(
            grid,
            coords={"lat": lat_centers, "lon": lon_centers},
            dims=["lat", "lon"],
            name="cyclone_freq_annual_mean",
        )
    finally:
        ds.close()


def join_cyclone_frequency(
    df: pd.DataFrame,
    *,
    cyclones_nc: Path | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Attach mean annual cyclone frequency at each survey coordinate."""
    cyclones_nc = Path(cyclones_nc or config.data_dir / "cyclones" / "cyclones.nc")
    cache_path = Path(
        cache_path
        or config.reef_check_dir / "processed" / CYCLONE_GRID_CACHE
    )
    if not cyclones_nc.exists():
        warnings.warn(f"Cyclone dataset missing at {cyclones_nc}; skipping.", stacklevel=2)
        return df

    if cache_path.exists() and cache_path.stat().st_mtime >= cyclones_nc.stat().st_mtime:
        grid = xr.open_dataarray(cache_path).load()
    else:
        grid = _build_cyclone_frequency_grid(cyclones_nc)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_netcdf(cache_path)

    coords = _unique_coords(df)
    sampled = grid.sel(
        lat=xr.DataArray(coords["latitude"].to_numpy(), dims="points"),
        lon=xr.DataArray(coords["longitude"].to_numpy(), dims="points"),
        method="nearest",
    ).values
    coord_values = coords.copy()
    coord_values["cyclone_freq_annual_mean"] = sampled
    out = _map_coord_values(df, coord_values, ["cyclone_freq_annual_mean"])
    if "Cyclone" not in out.columns:
        out["Cyclone"] = out["cyclone_freq_annual_mean"]
    return out


def _coord_key(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray) -> pd.Series:
    lat = pd.to_numeric(lat, errors="coerce").round(6)
    lon = pd.to_numeric(lon, errors="coerce").round(6)
    return lat.astype(str) + "_" + lon.astype(str)


def _finite_unique_coords(df: pd.DataFrame) -> pd.DataFrame:
    coords = _unique_coords(df)
    finite = coords["latitude"].notna() & coords["longitude"].notna()
    return coords.loc[finite].reset_index(drop=True)


def _attach_cmip_loc_ids(coords: pd.DataFrame) -> pd.DataFrame:
    """Map survey coordinates to pre-extracted CMIP reef location ids."""
    locations = cmip_sst.load_locations()[["latitude", "longitude", "loc_id"]].copy()
    locations["_coord_key"] = _coord_key(locations["latitude"], locations["longitude"])
    out = coords.copy()
    out["_coord_key"] = _coord_key(out["latitude"], out["longitude"])
    out = out.merge(
        locations[["_coord_key", "loc_id"]],
        on="_coord_key",
        how="left",
    ).drop(columns=["_coord_key"])
    return out


def _mean_sst_from_mapped_parquet(path: Path) -> float:
    """Mean SST (°C) over 1850–1950 from a pre-mapped continuous parquet.

    Reads only ``sst_c`` and the first ``HIST_SST_N_ROWS`` daily values (historic
    segment starting 1850-01-01). Avoids loading ``time`` / forecast rows (~90k).
    """
    sst = pd.read_parquet(path, columns=["sst_c"])["sst_c"]
    if len(sst) < HIST_SST_N_ROWS:
        raise ValueError(f"expected ≥{HIST_SST_N_ROWS} rows, got {len(sst)}")
    return float(sst.iloc[:HIST_SST_N_ROWS].mean())


def _load_loc_sst_mean_1850_1950(
    model: str,
    *,
    scenario: str = DEFAULT_QDM_SCENARIO,
    mapped_dir: Path | None = None,
    retries: int = 8,
    retry_passes: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load or build QDM-mapped historic SST mean (1850–1950) per CMIP loc_id.

    Reads pre-built continuous series from ``cmip_mapped_point_timeseries``
    (``historic_qmapped`` segment), matching ``env_data.ipynb`` / ``load_mapped_timeseries``.

    Slowness is usually OneDrive/cloud latency opening ~5k Parquet files (each
    ~90k rows on disk, but we only scan the first ~37k ``sst_c`` values).
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    console = Console()
    cache = (
        config.reef_check_dir
        / "processed"
        / LOC_SST_HIST_CACHE.format(model=model)
    )
    locations = cmip_sst.load_locations()
    n_locations = len(locations)

    if cache.exists():
        cached = pd.read_parquet(cache)
        if cached["sst_mean_1850_1950"].notna().sum() >= n_locations:
            if verbose:
                console.print(
                    f"[green]SST mean 1850–1950:[/] loaded cache "
                    f"({n_locations:,} locations) → {cache}"
                )
            return cached.drop_duplicates("loc_id", keep="last")

    mapped_root = Path(mapped_dir or cmip_sst.CMIP_MAPPED_POINT_DIR)
    mapped_model_dir = mapped_root / scenario / model
    if not mapped_model_dir.is_dir():
        raise FileNotFoundError(mapped_model_dir)

    base = (
        pd.read_parquet(cache)
        if cache.exists()
        else pd.DataFrame(
            {"loc_id": pd.Series(dtype=str), "sst_mean_1850_1950": pd.Series(dtype=float)}
        )
    )
    base = base.drop_duplicates("loc_id", keep="last")

    def _pending_ids(frame: pd.DataFrame) -> list[str]:
        ok = set(frame.loc[frame["sst_mean_1850_1950"].notna(), "loc_id"].astype(str))
        return [str(x) for x in locations["loc_id"].astype(str) if str(x) not in ok]

    pending_ids = _pending_ids(base)
    if not pending_ids:
        return base

    def _read_mean(loc_id: str) -> tuple[str, float, bool]:
        """Return (loc_id, mean, timed_out)."""
        path = mapped_model_dir / f"{loc_id}.parquet"
        last_exc: BaseException | None = None
        for attempt in range(retries):
            try:
                return loc_id, _mean_sst_from_mapped_parquet(path), False
            except Exception as exc:  # noqa: BLE001
                if _is_retryable_io_error(exc):
                    last_exc = exc
                    time.sleep(min(2**attempt, 60))
                    continue
                warnings.warn(f"SST mean failed for {loc_id}: {exc}", stacklevel=2)
                return loc_id, np.nan, False
        warnings.warn(f"SST mean failed for {loc_id}: {last_exc}", stacklevel=2)
        return loc_id, np.nan, True

    def _flush(current: pd.DataFrame, rows: list[tuple[str, float]]) -> pd.DataFrame:
        if not rows:
            return current
        out = pd.concat(
            [current, pd.DataFrame(rows, columns=["loc_id", "sst_mean_1850_1950"])],
            ignore_index=True,
        ).drop_duplicates("loc_id", keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
        return out

    flush_every = 100
    total_timeouts = 0

    progress_ctx = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        if verbose
        else nullcontext()
    )

    with progress_ctx as progress:
        for pass_idx in range(retry_passes):
            base = pd.read_parquet(cache).drop_duplicates("loc_id", keep="last") if cache.exists() else base
            pending_ids = _pending_ids(base)
            if not pending_ids:
                break

            task = None
            if verbose and progress is not None:
                label = (
                    f"SST 1850–1950 {model}/{scenario}"
                    if pass_idx == 0
                    else f"Retry {pass_idx + 1}/{retry_passes}"
                )
                task = progress.add_task(
                    label,
                    total=len(pending_ids),
                    stats="ok=0 fail=0",
                )

            rows: list[tuple[str, float]] = []
            n_ok = n_fail = 0
            for i, loc_id in enumerate(pending_ids, start=1):
                loc_id, mean, timed_out = _read_mean(loc_id)
                rows.append((loc_id, mean))
                if timed_out:
                    total_timeouts += 1
                if np.isfinite(mean):
                    n_ok += 1
                else:
                    n_fail += 1

                if verbose and progress is not None and task is not None:
                    progress.update(
                        task,
                        advance=1,
                        stats=f"ok={n_ok:,} fail={n_fail:,} to={total_timeouts:,}",
                    )

                if i % flush_every == 0:
                    base = _flush(base, rows)
                    rows = []

            base = _flush(base, rows)
            if verbose and progress is not None and task is not None:
                progress.remove_task(task)

    done = pd.read_parquet(cache).drop_duplicates("loc_id", keep="last")
    n_ok = int(done["sst_mean_1850_1950"].notna().sum())
    if verbose:
        style = "green" if n_ok >= n_locations else "yellow"
        console.print(
            f"[{style}]SST mean 1850–1950:[/] {n_ok:,}/{n_locations:,} locations → {cache}"
        )
        if total_timeouts:
            console.print(
                "[dim]Note: OneDrive/cloud reads dominate runtime; "
                f"{total_timeouts:,} location(s) hit timeouts (re-run to retry).[/]"
            )
    if n_ok < n_locations:
        warnings.warn(
            f"{n_locations - n_ok:,} locations still missing SST mean 1850–1950; "
            "re-run to retry (OneDrive timeouts).",
            stacklevel=2,
        )
    return done


def _sample_sst_mean_1985_2014(
    coords: pd.DataFrame,
    *,
    model: str,
    daily_sst_dir: Path | None,
    batch_size: int,
    verbose: bool = True,
) -> np.ndarray:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    files = cmip_sst.qdm_historic_files(model, daily_sst_dir=daily_sst_dir)
    ref_files = [p for p in files if "1985_2014" in p.name] or files
    if not ref_files:
        raise FileNotFoundError(f"No QDM historic NetCDF files found for {model!r}")

    means = np.full(len(coords), np.nan, dtype=float)
    batch_starts = list(range(0, len(coords), batch_size))
    console = Console()

    progress_ctx = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        if verbose
        else nullcontext()
    )

    with progress_ctx as progress:
        task = None
        if verbose and progress is not None:
            task = progress.add_task(
                f"SST 1985–2014 QDM {model}",
                total=len(batch_starts),
                stats="ok=0 fail=0",
            )

        n_ok = n_fail = 0
        for bi, start in enumerate(batch_starts):
            sub = coords.iloc[start : start + batch_size]
            lats = sub["latitude"].to_numpy(dtype=float)
            lons = sub["longitude"].to_numpy(dtype=float)
            try:
                batch = cmip_sst.sample_qdm_timeseries_all_points(
                    lats,
                    lons,
                    ref_files,
                    dataset_label=f"qdm_{model}",
                    margin_deg=90.0,
                )
                ref_mask = (batch.times >= pd.Timestamp(cmip_sst.REF_START)) & (
                    batch.times <= pd.Timestamp(cmip_sst.REF_END)
                )
                if not ref_mask.any():
                    ref_mask = np.ones(len(batch.times), dtype=bool)
                batch_means = np.nanmean(batch.values[ref_mask, :], axis=0)
                means[start : start + len(sub)] = batch_means
                n_ok += int(np.isfinite(batch_means).sum())
                n_fail += int((~np.isfinite(batch_means)).sum())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"SST batch {start}-{start + len(sub)} failed ({exc}); "
                    "falling back to point extraction.",
                    stacklevel=2,
                )
                for j, (lat, lon) in enumerate(zip(lats, lons, strict=False)):
                    try:
                        series = cmip_sst.extract_qdm_at_point(
                            float(lat),
                            float(lon),
                            ref_files,
                            dataset_label=f"qdm_{model}",
                        )
                        ref = series.loc[cmip_sst.REF_START : cmip_sst.REF_END]
                        val = float(ref.mean())
                        means[start + j] = val
                        if np.isfinite(val):
                            n_ok += 1
                        else:
                            n_fail += 1
                    except Exception:  # noqa: BLE001
                        n_fail += 1
                        continue

            if verbose and progress is not None and task is not None:
                progress.update(
                    task,
                    advance=1,
                    stats=f"ok={n_ok:,} fail={n_fail:,}",
                )

    return means


def join_site_sst_means(
    df: pd.DataFrame,
    *,
    model: str = DEFAULT_QDM_MODEL,
    scenario: str = DEFAULT_QDM_SCENARIO,
    daily_sst_dir: Path | None = None,
    cache_path: Path | None = None,
    batch_size: int = 400,
) -> pd.DataFrame:
    """Attach QDM SST means at each survey coordinate.

    - ``sst_mean_1985_2014``: mean of QDM-corrected daily SST (NetCDF grid)
    - ``sst_mean_1850_1950``: mean of QDM-quantile-mapped CMIP historic SST
    """
    cache = Path(
        cache_path
        or config.reef_check_dir / "processed" / SST_MEANS_CACHE
    )
    coords = _finite_unique_coords(df)
    sst_cols = ["sst_mean_1850_1950", "sst_mean_1985_2014"]

    legacy_cache = cache.with_name("site_sst_mean_1985_2014.parquet")
    if not cache.exists() and legacy_cache.exists():
        legacy = pd.read_parquet(legacy_cache)
        if "sst_mean_1985_2014" in legacy.columns:
            legacy.to_parquet(cache, index=False)
            print(f"Migrated legacy SST cache → {cache}")

    if cache.exists():
        cached = pd.read_parquet(cache)
        if all(c in cached.columns for c in sst_cols):
            cached_keys = set(_coord_key(cached["latitude"], cached["longitude"]))
            coord_keys = set(_coord_key(coords["latitude"], coords["longitude"]))
            if coord_keys.issubset(cached_keys):
                coord_values = coords.merge(
                    cached[["latitude", "longitude", *sst_cols]],
                    on=["latitude", "longitude"],
                    how="left",
                )
                out = _map_coord_values(df, coord_values, sst_cols)
                for col in sst_cols:
                    print(
                        f"{col}: loaded cache "
                        f"({out[col].notna().sum():,}/{len(out):,} surveys)"
                    )
                return out
        if "sst_mean_1985_2014" in cached.columns:
            coord_values = coords.merge(
                cached[["latitude", "longitude", "sst_mean_1985_2014"]],
                on=["latitude", "longitude"],
                how="left",
            )
        else:
            coord_values = coords.copy()
    else:
        coord_values = coords.copy()

    if "sst_mean_1985_2014" not in coord_values.columns:
        try:
            coord_values["sst_mean_1985_2014"] = _sample_sst_mean_1985_2014(
                coord_values,
                model=model,
                daily_sst_dir=daily_sst_dir,
                batch_size=batch_size,
            )
        except FileNotFoundError as exc:
            warnings.warn(f"{exc}; skipping 1985–2014 SST mean.", stacklevel=2)

    if "sst_mean_1850_1950" not in coord_values.columns:
        try:
            loc_means = _load_loc_sst_mean_1850_1950(model, scenario=scenario)
            coord_values = _attach_cmip_loc_ids(coord_values)
            lookup = loc_means.set_index("loc_id")["sst_mean_1850_1950"]
            coord_values["sst_mean_1850_1950"] = coord_values["loc_id"].map(lookup)
            coord_values = coord_values.drop(columns=["loc_id"], errors="ignore")
        except FileNotFoundError as exc:
            warnings.warn(f"{exc}; skipping 1850–1950 SST mean.", stacklevel=2)

    write_cols = ["latitude", "longitude"] + [
        c for c in sst_cols if c in coord_values.columns
    ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    coord_values[write_cols].to_parquet(cache, index=False)

    from rich.console import Console

    console = Console()
    console.print(f"[green]Cached site SST means[/] → {cache} (n={len(coord_values):,})")

    out = _map_coord_values(df, coord_values, [c for c in sst_cols if c in coord_values.columns])
    for col in sst_cols:
        if col in out.columns:
            console.print(
                f"  [dim]{col}:[/] {out[col].notna().sum():,}/{len(out):,} surveys"
            )
    return out


def join_mpa_protected(
    df: pd.DataFrame,
    *,
    mpas_dir: Path | None = None,
) -> pd.DataFrame:
    """Binary marine/coastal MPA protection flag per survey point."""
    mpas_dir = Path(mpas_dir or config.mpas_dir)
    shp_paths = sorted(mpas_dir.glob("WDPA_*shp*/WDPA_*polygons.shp"))
    if not shp_paths:
        warnings.warn(f"No WDPA polygon shapefiles under {mpas_dir}; skipping.", stacklevel=2)
        return df

    mpas = pd.concat([gpd.read_file(p) for p in shp_paths], ignore_index=True)
    mpas.columns = [str(c).lower() for c in mpas.columns]
    if "realm" in mpas.columns:
        mpas = mpas[mpas["realm"].isin(["Marine", "Coastal"])]
    mpas = gpd.GeoDataFrame(mpas, geometry="geometry", crs=mpas.crs)

    pts = _points_gdf(df)
    if pts.crs != mpas.crs:
        pts = pts.to_crs(mpas.crs)
    joined = gpd.sjoin(pts, mpas[["geometry"]], how="left", predicate="within")
    protected = joined.groupby(joined.index)["index_right"].apply(lambda s: s.notna().any())
    df["mpa_protected"] = protected.reindex(df.index, fill_value=False).to_numpy()
    return df


def _worldpop_raster_for_year(
    year: int,
    *,
    ssp: str = DEFAULT_POP_SSP,
    population_dir: Path | None = None,
) -> Path | None:
    pop_dir = Path(population_dir or config.data_dir / "population")
    folder = pop_dir / f"FuturePop_{ssp}_1km_v0_2"
    if not folder.is_dir():
        return None
    candidates = sorted(folder.glob("FuturePop_*.tif"))
    if not candidates:
        return None
    years = []
    for path in candidates:
        parts = path.stem.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            years.append((int(parts[2]), path))
    if not years:
        return None
    best_year, best_path = min(years, key=lambda item: abs(item[0] - year))
    return best_path


def join_worldpop_density(
    df: pd.DataFrame,
    *,
    ssp: str = DEFAULT_POP_SSP,
    population_dir: Path | None = None,
) -> pd.DataFrame:
    """Sample nearest-year WorldPop density (people / km²) at each survey point."""
    import rasterio

    out = df.copy()
    out["worldpop_density"] = np.nan
    year_groups = out.groupby(out["year"].astype(int), sort=False)
    for year, idx in year_groups.groups.items():
        raster_path = _worldpop_raster_for_year(
            int(year), ssp=ssp, population_dir=population_dir
        )
        if raster_path is None:
            continue
        sub = out.loc[idx]
        with rasterio.open(raster_path) as src:
            samples = [
                val[0] if val else np.nan
                for val in src.sample(list(zip(sub["lon"], sub["lat"])))
            ]
        out.loc[idx, "worldpop_density"] = samples
    return out


def build_model_ready_data(
    *,
    substrate_path: Path | None = None,
    errors_csv: Path | None = None,
    processed_path: Path | None = None,
    use_processed_cache: bool = True,
    force_reprocess: bool = False,
    qdm_model: str = DEFAULT_QDM_MODEL,
    qdm_scenario: str = DEFAULT_QDM_SCENARIO,
    skip_meow: bool = False,
    skip_cotw: bool = False,
    skip_cyclones: bool = False,
    skip_sst: bool = False,
    skip_mpa: bool = False,
    skip_population: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the full Reef Check + environment dataframe."""
    from rich.console import Console

    console = Console() if verbose else None

    def _step(label: str) -> None:
        if console is not None:
            console.print(f"[bold blue]→[/] {label}")

    df = build_reef_check_surveys(
        substrate_path=substrate_path,
        errors_csv=errors_csv,
        processed_path=processed_path,
        use_processed_cache=use_processed_cache,
        force_reprocess=force_reprocess,
    )
    n0 = len(df)

    if not skip_meow:
        _step("MEOW ecoregions")
        df = join_meow_ecoregions(df)
    if not skip_cotw:
        _step("COTW ecoregions")
        df = join_cotw_ecoregions(df)
    if not skip_cyclones:
        _step("Cyclone frequency")
        df = join_cyclone_frequency(df)
    if not skip_sst:
        _step(
            f"SST means (1985–2014 QDM grid; 1850–1950 mapped parquets, "
            f"{qdm_model}/{qdm_scenario})"
        )
        df = join_site_sst_means(df, model=qdm_model, scenario=qdm_scenario)
    if not skip_mpa:
        _step("MPA protection")
        df = join_mpa_protected(df)
    if not skip_population:
        _step("WorldPop coastal density")
        df = join_worldpop_density(df)

    msg = (
        f"Reef Check model-ready table: {len(df):,} surveys "
        f"(base {n0:,}; env columns attached where data exist)"
    )
    if console is not None:
        console.print(f"[green]{msg}[/]")
    else:
        print(msg)
    return df


def _source_paths(
    *,
    substrate_path: Path,
    errors_csv: Path,
    cyclones_nc: Path,
) -> dict[str, Path]:
    return {
        "substrate": substrate_path,
        "errors_csv": errors_csv,
        "cyclones.nc": cyclones_nc,
    }


def _cache_is_stale(cache_path: Path, sources: dict[str, Path]) -> bool:
    if not cache_path.exists():
        return True
    cache_mtime = cache_path.stat().st_mtime
    for path in sources.values():
        if path.exists() and path.stat().st_mtime > cache_mtime:
            return True
    return False


def write_model_ready_cache(
    df: pd.DataFrame,
    cache_path: Path,
    *,
    sources: dict[str, Path],
    build_config: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "cache_path": str(cache_path),
        "build_config": build_config,
        "sources": {name: str(path) for name, path in sources.items()},
        "source_mtimes": {
            name: path.stat().st_mtime for name, path in sources.items() if path.exists()
        },
    }
    cache_path.with_name(META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")


def load_model_ready_data(
    *,
    cache_path: Path | None = None,
    substrate_path: Path | None = None,
    errors_csv: Path | None = None,
    force_rebuild: bool = False,
    **build_kwargs: Any,
) -> pd.DataFrame:
    """Load cached Reef Check model-ready data, rebuilding when sources are newer."""
    cache = Path(cache_path or model_ready_cache_path())
    substrate_path = Path(substrate_path or default_substrate_path())
    errors_csv = Path(errors_csv or config.sully_og_dir / "reefcheck_error_info_completed.csv")
    cyclones_nc = Path(config.data_dir / "cyclones" / "cyclones.nc")
    sources = _source_paths(
        substrate_path=substrate_path,
        errors_csv=errors_csv,
        cyclones_nc=cyclones_nc,
    )

    if force_rebuild or _cache_is_stale(cache, sources):
        reason = "forced rebuild" if force_rebuild else "cache missing or stale"
        print(f"Building Reef Check model-ready data ({reason})…")
        df = build_model_ready_data(
            substrate_path=substrate_path,
            errors_csv=errors_csv,
            **build_kwargs,
        )
        write_model_ready_cache(
            df,
            cache,
            sources=sources,
            build_config=build_kwargs,
        )
        print(f"Cached → {cache}")
        return df

    df = pd.read_csv(cache, parse_dates=["date"])
    print(f"Loaded cached Reef Check model-ready data from {cache} (n={len(df):,})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/cache Reef Check survey table with environmental covariates"
    )
    parser.add_argument("--substrate", type=Path, default=None)
    parser.add_argument("--errors-csv", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild model-ready CSV")
    parser.add_argument("--force-reprocess-substrate", action="store_true")
    parser.add_argument("--qdm-model", default=DEFAULT_QDM_MODEL)
    parser.add_argument("--qdm-scenario", default=DEFAULT_QDM_SCENARIO)
    parser.add_argument("--skip-meow", action="store_true")
    parser.add_argument("--skip-cotw", action="store_true")
    parser.add_argument("--skip-cyclones", action="store_true")
    parser.add_argument("--skip-sst", action="store_true")
    parser.add_argument("--skip-mpa", action="store_true")
    parser.add_argument("--skip-population", action="store_true")
    parser.add_argument(
        "--build-sst-cache-only",
        action="store_true",
        help="Only build loc/site SST mean caches (no full model-ready CSV)",
    )
    args = parser.parse_args()

    if args.build_sst_cache_only:
        surveys = build_reef_check_surveys(
            substrate_path=args.substrate,
            errors_csv=args.errors_csv,
            force_reprocess=args.force_reprocess_substrate,
        )
        join_site_sst_means(
            surveys,
            model=args.qdm_model,
            scenario=args.qdm_scenario,
        )
        return

    load_model_ready_data(
        cache_path=args.cache_path,
        substrate_path=args.substrate,
        errors_csv=args.errors_csv,
        force_rebuild=args.force,
        force_reprocess=args.force_reprocess_substrate,
        qdm_model=args.qdm_model,
        qdm_scenario=args.qdm_scenario,
        skip_meow=args.skip_meow,
        skip_cotw=args.skip_cotw,
        skip_cyclones=args.skip_cyclones,
        skip_sst=args.skip_sst,
        skip_mpa=args.skip_mpa,
        skip_population=args.skip_population,
    )


if __name__ == "__main__":
    main()
