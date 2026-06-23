"""CMIP6 SST point timeseries: load, QDM sampling, and quantile mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from src import config

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

CMIP_HISTORIC_POINT_DIR = config.env_dir / "cmip_historic_point_timeseries"
CMIP_MAPPED_POINT_DIR = config.env_dir / "cmip_mapped_point_timeseries"
QDM_OFFSET_DIRNAME = "qdm_extraction_offsets"
DAILY_SST_DIR = config.env_dir / "DailySST"
DEFAULT_MARGIN_DEG = 3.0

SCENARIO_DIRS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

REF_START = "1985-01-01"
REF_END = "2014-12-31"
FORECAST_START = "2015-01-01"
KELVIN_TO_C = 273.15


@dataclass(frozen=True)
class QdmSampleBatch:
    """Batch QDM point samples and extraction diagnostics."""

    values: np.ndarray
    times: pd.DatetimeIndex
    metadata: pd.DataFrame


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km."""
    lat1_r = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lon1_r = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lat2_r = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon2_r = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _latlon_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64))
    cos_lat = np.cos(lat_r)
    return np.stack(
        (cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)),
        axis=-1,
    )


def _spatial_subset(
    da: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    margin_deg: float = DEFAULT_MARGIN_DEG,
) -> xr.DataArray:
    """Restrict to a lat/lon bounding box around survey points."""
    lat_min = max(-90.0, float(lats.min()) - margin_deg)
    lat_max = min(90.0, float(lats.max()) + margin_deg)
    lon_min = float(lons.min()) - margin_deg
    lon_max = float(lons.max()) + margin_deg

    if lon_max - lon_min >= 350.0:
        return da.sel(lat=slice(lat_min, lat_max))

    if lon_min < -180.0 or lon_max > 180.0:
        lon_min_n = ((lon_min + 180.0) % 360.0) - 180.0
        lon_max_n = ((lon_max + 180.0) % 360.0) - 180.0
        if lon_min_n <= lon_max_n:
            return da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_n, lon_max_n))
        left = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_n, 180.0))
        right = da.sel(lat=slice(lat_min, lat_max), lon=slice(-180.0, lon_max_n))
        return xr.concat([left, right], dim="lon")

    return da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def _resolve_sample_indices(
    lats: np.ndarray,
    lons: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick nearest valid ocean grid cell for each point."""
    n_points = int(lats.size)
    lat_idx = np.empty(n_points, dtype=np.int32)
    lon_idx = np.empty(n_points, dtype=np.int32)
    used_fallback = np.zeros(n_points, dtype=bool)
    nearest_valid = np.zeros(n_points, dtype=bool)

    lat_idx_geo = (
        np.abs(lat_vals[:, None] - lats[None, :]).argmin(axis=0).astype(np.int32)
    )
    lon_idx_geo = (
        np.abs(lon_vals[:, None] - lons[None, :]).argmin(axis=0).astype(np.int32)
    )
    nearest_valid[:] = valid_mask[lat_idx_geo, lon_idx_geo]
    lat_idx[:] = lat_idx_geo
    lon_idx[:] = lon_idx_geo

    if not nearest_valid.all():
        valid_i, valid_j = np.nonzero(valid_mask)
        if valid_i.size == 0:
            raise ValueError("No valid ocean cells in QDM grid for this region")
        valid_lats = lat_vals[valid_i]
        valid_lons = lon_vals[valid_j]
        tree = cKDTree(_latlon_to_unit_xyz(valid_lats, valid_lons))
        fallback_mask = ~nearest_valid
        query_xyz = _latlon_to_unit_xyz(lats[fallback_mask], lons[fallback_mask])
        _, nearest = tree.query(query_xyz, k=1, workers=-1)
        lat_idx[fallback_mask] = valid_i[nearest]
        lon_idx[fallback_mask] = valid_j[nearest]
        used_fallback[fallback_mask] = True

    grid_lat = lat_vals[lat_idx]
    grid_lon = lon_vals[lon_idx]
    offset_km = haversine_km(lats, lons, grid_lat, grid_lon)
    return lat_idx, lon_idx, grid_lat, grid_lon, offset_km, used_fallback


def _build_extraction_metadata(
    loc_ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    offset_km: np.ndarray,
    used_fallback: np.ndarray,
    *,
    dataset_label: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loc_id": loc_ids.astype(str),
            "input_latitude": lats.astype(float),
            "input_longitude": lons.astype(float),
            f"{dataset_label}_grid_latitude": grid_lat.astype(float),
            f"{dataset_label}_grid_longitude": grid_lon.astype(float),
            f"{dataset_label}_offset_km": offset_km.astype(float),
            f"{dataset_label}_used_nearest_valid_fallback": used_fallback.astype(bool),
        }
    )


def write_qdm_offset_tables(
    model: str,
    scenario: str,
    historic_meta: pd.DataFrame,
    forecast_meta: pd.DataFrame,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write historic and forecast QDM extraction offset tables."""
    offset_dir = output_dir / QDM_OFFSET_DIRNAME
    offset_dir.mkdir(parents=True, exist_ok=True)

    hist_path = offset_dir / f"{model}_historic_offsets.csv"
    hist_out = historic_meta.copy()
    hist_out.insert(0, "model", model)
    hist_out.to_csv(hist_path, index=False)

    fc_path = offset_dir / f"{model}_{scenario}_forecast_offsets.csv"
    fc_out = forecast_meta.copy()
    fc_out.insert(0, "model", model)
    fc_out.insert(1, "scenario", scenario)
    fc_out.to_csv(fc_path, index=False)

    combined_path = offset_dir / f"{model}_{scenario}_combined_offsets.csv"
    combined = historic_meta.merge(
        forecast_meta,
        on=["loc_id", "input_latitude", "input_longitude"],
        how="outer",
    )
    combined.insert(0, "model", model)
    combined.insert(1, "scenario", scenario)
    combined.to_csv(combined_path, index=False)
    return hist_path, fc_path


def load_locations(historic_dir: Path | None = None) -> pd.DataFrame:
    """Load unique reef survey locations from historic extraction indexes."""
    root = historic_dir or CMIP_HISTORIC_POINT_DIR
    return pd.read_parquet(root / "indexes" / "locations.parquet")


def load_raw_historic_timeseries(
    model: str,
    loc_id: str,
    *,
    historic_dir: Path | None = None,
) -> pd.DataFrame:
    """Load raw CMIP6 historical ``tos`` (Kelvin) for one model/location."""
    root = historic_dir or CMIP_HISTORIC_POINT_DIR
    path = root / model / f"{loc_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["sst_c"] = df["tos"].astype(float) - KELVIN_TO_C
    return df.sort_values("time").reset_index(drop=True)


def qdm_historic_files(model: str, daily_sst_dir: Path | None = None) -> list[Path]:
    root = daily_sst_dir or DAILY_SST_DIR
    return sorted((root / "Historical").glob(f"{model}_historical_qdmCorrected_*.nc"))


def qdm_forecast_files(
    model: str,
    scenario: str = "ssp370",
    daily_sst_dir: Path | None = None,
) -> list[Path]:
    root = daily_sst_dir or DAILY_SST_DIR
    if scenario not in SCENARIO_DIRS:
        raise ValueError(
            f"Unknown scenario {scenario!r}; choose from {list(SCENARIO_DIRS)}"
        )
    scenario_dir = root / SCENARIO_DIRS[scenario]
    return sorted(scenario_dir.glob(f"{model}_{scenario}_qdmCorrected_*.nc"))


def classify_points_on_qdm_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    da: xr.DataArray,
    *,
    time_idx: int = 0,
) -> pd.DataFrame:
    """Classify reef points by geographic-nearest QDM grid cell validity.

    Matches the pre-fallback check in ``_resolve_sample_indices``: a point is
    invalid when its nearest lat/lon grid node is NaN (land/missing ocean).
    """
    if "latitude" in da.coords:
        da = da.rename({"latitude": "lat", "longitude": "lon"})

    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lat_vals = da["lat"].values.astype(np.float64)
    lon_vals = da["lon"].values.astype(np.float64)
    field = da.isel(time=time_idx)
    valid_mask = field.notnull().values.astype(bool)

    lat_idx_geo = (
        np.abs(lat_vals[:, None] - lats[None, :]).argmin(axis=0).astype(np.int32)
    )
    lon_idx_geo = (
        np.abs(lon_vals[:, None] - lons[None, :]).argmin(axis=0).astype(np.int32)
    )
    nearest_cell_valid = valid_mask[lat_idx_geo, lon_idx_geo]

    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "grid_latitude": lat_vals[lat_idx_geo],
            "grid_longitude": lon_vals[lon_idx_geo],
            "nearest_cell_valid": nearest_cell_valid,
            "nearest_cell_sst_c": field.values[lat_idx_geo, lon_idx_geo],
        }
    )


def sample_qdm_timeseries_all_points(
    lats: np.ndarray,
    lons: np.ndarray,
    nc_paths: list[Path] | tuple[Path, ...],
    *,
    loc_ids: np.ndarray | None = None,
    dataset_label: str = "qdm",
    margin_deg: float = DEFAULT_MARGIN_DEG,
) -> QdmSampleBatch:
    """Sample QDM SST (degC) at reef points using nearest valid ocean grid cells.

    Coastal / complex topography points often fall on land in the QDM NetCDF
    (NaN at the geographic-nearest cell). Those points use the nearest valid
    ocean cell instead; offsets are returned in ``metadata``.
    """
    if not nc_paths:
        raise FileNotFoundError("No QDM NetCDF paths supplied")
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if loc_ids is None:
        loc_ids = np.array([f"loc_{i:06d}" for i in range(len(lats))], dtype=object)

    ds = xr.open_mfdataset([str(p) for p in sorted(nc_paths)], chunks="auto")
    try:
        da = ds["tos"]
        if "latitude" in da.coords:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = _spatial_subset(da, lats, lons, margin_deg=margin_deg)
        lat_vals = da["lat"].values.astype(np.float64)
        lon_vals = da["lon"].values.astype(np.float64)
        valid_mask = da.isel(time=0).notnull().compute().values.astype(bool)

        lat_idx, lon_idx, grid_lat, grid_lon, offset_km, used_fallback = (
            _resolve_sample_indices(lats, lons, lat_vals, lon_vals, valid_mask)
        )
        sampled = da.isel(
            lat=xr.DataArray(lat_idx, dims="points"),
            lon=xr.DataArray(lon_idx, dims="points"),
        )
        values = np.asarray(sampled.load().values, dtype=np.float32)
        times = pd.to_datetime(sampled["time"].values)
        metadata = _build_extraction_metadata(
            loc_ids,
            lats,
            lons,
            grid_lat,
            grid_lon,
            offset_km,
            used_fallback,
            dataset_label=dataset_label,
        )
        return QdmSampleBatch(values=values, times=times, metadata=metadata)
    finally:
        ds.close()


def extract_qdm_at_point(
    lat: float,
    lon: float,
    nc_paths: list[Path] | tuple[Path, ...],
    *,
    dataset_label: str = "qdm",
) -> pd.Series:
    """Sample QDM-corrected SST (degC) at the nearest valid ocean grid cell."""
    batch = sample_qdm_timeseries_all_points(
        np.array([lat]),
        np.array([lon]),
        nc_paths,
        dataset_label=dataset_label,
    )
    return pd.Series(batch.values[:, 0], index=batch.times, name="tos").sort_index()


def extract_qdm_historic_at_point(
    lat: float,
    lon: float,
    model: str,
    daily_sst_dir: Path | None = None,
) -> pd.Series:
    return extract_qdm_at_point(
        lat, lon, qdm_historic_files(model, daily_sst_dir), dataset_label="historic"
    )


def extract_qdm_forecast_at_point(
    lat: float,
    lon: float,
    model: str,
    scenario: str = "ssp370",
    daily_sst_dir: Path | None = None,
) -> pd.Series:
    return extract_qdm_at_point(
        lat,
        lon,
        qdm_forecast_files(model, scenario, daily_sst_dir),
        dataset_label="forecast",
    )


def quantile_map_empirical(
    values: np.ndarray,
    source_ref: np.ndarray,
    target_ref: np.ndarray,
) -> np.ndarray:
    """Map ``values`` through empirical quantiles of ref periods (pooled QDM)."""
    values = np.asarray(values, dtype=float)
    src_ref = np.asarray(source_ref, dtype=float)
    tgt_ref = np.asarray(target_ref, dtype=float)
    src_ref = src_ref[np.isfinite(src_ref)]
    tgt_ref = tgt_ref[np.isfinite(tgt_ref)]
    if src_ref.size == 0 or tgt_ref.size == 0:
        raise ValueError("Reference arrays must contain finite values")

    src_sorted = np.sort(src_ref)
    tgt_sorted = np.sort(tgt_ref)
    n = src_sorted.size
    ranks = np.searchsorted(src_sorted, values, side="right")
    quantiles = np.clip(ranks / n, 0.0, 1.0)
    # Mid-rank quantile positions; np.interp avoids the lo==hi boundary bug.
    ref_quantiles = (np.arange(n) + 0.5) / n
    return np.interp(quantiles, ref_quantiles, tgt_sorted)


def quantile_map_historic_timeseries(
    raw_df: pd.DataFrame,
    qdm_ref: pd.Series,
    *,
    ref_start: str = REF_START,
    ref_end: str = REF_END,
    value_col: str = "sst_c",
) -> pd.DataFrame:
    """Quantile-map raw CMIP historic SST onto the QDM historic reference."""
    out = raw_df.copy()
    qdm_ref = qdm_ref.copy()
    qdm_ref.index = pd.to_datetime(qdm_ref.index)
    qdm_ref = qdm_ref.loc[ref_start:ref_end].rename("qdm_ref")

    ref = (
        out.loc[
            (out["time"] >= ref_start) & (out["time"] <= ref_end), ["time", value_col]
        ]
        .set_index("time")
        .join(qdm_ref, how="inner")
        .dropna()
    )
    if ref.empty:
        raise ValueError(
            f"No overlapping finite reference data between {ref_start} and {ref_end}"
        )

    out["sst_qmapped"] = quantile_map_empirical(
        out[value_col].to_numpy(),
        ref[value_col].to_numpy(),
        ref["qdm_ref"].to_numpy(),
    )
    return out


def build_continuous_timeseries_from_series(
    raw_df: pd.DataFrame,
    qdm_hist: pd.Series,
    qdm_fc: pd.Series,
) -> pd.DataFrame:
    """Build continuous SST from pre-extracted raw and QDM series."""
    mapped = quantile_map_historic_timeseries(raw_df, qdm_hist)
    hist_out = mapped[["time", "sst_qmapped"]].rename(columns={"sst_qmapped": "sst_c"})
    hist_out["source"] = "historic_qmapped"

    fc_out = qdm_fc.loc[FORECAST_START:].rename("sst_c").reset_index()
    fc_out.columns = ["time", "sst_c"]
    fc_out["source"] = "forecast_qdm"

    return (
        pd.concat([hist_out, fc_out], ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )


def load_mapped_timeseries(
    model: str,
    loc_id: str,
    scenario: str = "ssp370",
    *,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load pre-built continuous mapped SST (1850–2100) from batch output."""
    path = (output_dir or CMIP_MAPPED_POINT_DIR) / scenario / model / f"{loc_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build_continuous_timeseries(
    model: str,
    loc_id: str,
    lat: float,
    lon: float,
    *,
    scenario: str = "ssp370",
    historic_dir: Path | None = None,
    daily_sst_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a continuous SST series: QM-mapped CMIP historic (1850-2014) + QDM forecast.

    Returns a dataframe with columns ``time``, ``sst_c``, and ``source``
    (``historic_qmapped`` or ``forecast_qdm``).
    """
    raw = load_raw_historic_timeseries(model, loc_id, historic_dir=historic_dir)
    qdm_hist = extract_qdm_historic_at_point(lat, lon, model, daily_sst_dir)
    qdm_fc = extract_qdm_forecast_at_point(lat, lon, model, scenario, daily_sst_dir)
    return build_continuous_timeseries_from_series(raw, qdm_hist, qdm_fc)


def discover_models(
    *,
    historic_dir: Path | None = None,
    daily_sst_dir: Path | None = None,
) -> list[str]:
    """Return models with both raw historic point extracts and QDM NetCDF files."""
    hist_root = historic_dir or CMIP_HISTORIC_POINT_DIR
    sst_root = daily_sst_dir or DAILY_SST_DIR
    hist_models = {
        p.name for p in hist_root.iterdir() if p.is_dir() and p.name != "indexes"
    }
    qdm_models = {
        p.name.split("_historical_qdmCorrected_")[0]
        for p in (sst_root / "Historical").glob("*_historical_qdmCorrected_*.nc")
    }
    return sorted(hist_models & qdm_models)


_WORKER_CTX: dict[str, object] = {}


def _init_map_worker(ctx: dict[str, object]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _map_one_location(
    idx: int,
    loc_id: str,
    raw_path: str,
    ctx: dict[str, object] | None,
) -> dict[str, object]:
    data = _WORKER_CTX if ctx is None else ctx
    out_root = Path(str(data["out_root"]))
    model = str(data["model"])
    scenario = str(data["scenario"])
    out_path = out_root / scenario / model / f"{loc_id}.parquet"
    try:
        raw = pd.read_parquet(raw_path)
        raw["time"] = pd.to_datetime(raw["time"])
        raw["sst_c"] = raw["tos"].astype(float) - KELVIN_TO_C

        qdm_hist = pd.Series(
            np.asarray(data["hist_values"][:, idx], dtype=float),
            index=pd.to_datetime(data["hist_times"]),
        )
        qdm_fc = pd.Series(
            np.asarray(data["fc_values"][:, idx], dtype=float),
            index=pd.to_datetime(data["fc_times"]),
        )
        frame = build_continuous_timeseries_from_series(raw, qdm_hist, qdm_fc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path, index=False)
        return {
            "model": model,
            "scenario": scenario,
            "loc_id": loc_id,
            "parquet_path": str(out_path),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model": model,
            "scenario": scenario,
            "loc_id": loc_id,
            "parquet_path": None,
            "status": f"error: {exc}",
        }


def map_model_scenario(
    model: str,
    scenario: str,
    locations: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    historic_dir: Path | None = None,
    daily_sst_dir: Path | None = None,
    overwrite: bool = False,
    location_workers: int = 1,
    progress: Progress | None = None,
    scenario_task: TaskID | None = None,
    location_task: TaskID | None = None,
) -> pd.DataFrame:
    """Quantile-map and write continuous SST for all locations (batch QDM + parallel I/O)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _tick_location() -> None:
        if progress is not None and location_task is not None:
            progress.advance(location_task)

    def _set_status(text: str) -> None:
        if progress is not None and scenario_task is not None:
            progress.update(
                scenario_task, description=f"[cyan]{model}[/] — {scenario} · {text}"
            )

    out_root = output_dir or CMIP_MAPPED_POINT_DIR
    hist_root = historic_dir or CMIP_HISTORIC_POINT_DIR
    lats = locations["latitude"].to_numpy(dtype=np.float64)
    lons = locations["longitude"].to_numpy(dtype=np.float64)
    loc_ids = locations["loc_id"].astype(str).to_numpy()

    hist_files = qdm_historic_files(model, daily_sst_dir)
    fc_files = qdm_forecast_files(model, scenario, daily_sst_dir)
    if not hist_files:
        _set_status("skipped — no QDM historic")
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "scenario": scenario,
                    "loc_id": None,
                    "parquet_path": None,
                    "status": "skipped_missing_qdm_historic",
                }
            ]
        )
    if not fc_files:
        _set_status("skipped — no QDM forecast")
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "scenario": scenario,
                    "loc_id": None,
                    "parquet_path": None,
                    "status": "skipped_missing_qdm_forecast",
                }
            ]
        )

    _set_status("loading QDM historic")
    hist_batch = sample_qdm_timeseries_all_points(
        lats,
        lons,
        hist_files,
        loc_ids=loc_ids,
        dataset_label="historic",
    )
    _set_status("loading QDM forecast")
    fc_batch = sample_qdm_timeseries_all_points(
        lats,
        lons,
        fc_files,
        loc_ids=loc_ids,
        dataset_label="forecast",
    )
    write_qdm_offset_tables(
        model,
        scenario,
        hist_batch.metadata,
        fc_batch.metadata,
        output_dir=out_root,
    )

    hist_values = hist_batch.values
    hist_times = hist_batch.times
    fc_values = fc_batch.values
    fc_times = fc_batch.times

    ref_start = np.datetime64(REF_START)
    ref_end = np.datetime64(REF_END)
    ref_time_mask = (hist_times.values >= ref_start) & (hist_times.values <= ref_end)

    rows: list[dict[str, object]] = []
    tasks: list[tuple[int, str, str]] = []
    for idx, loc_id in enumerate(loc_ids):
        out_path = out_root / scenario / model / f"{loc_id}.parquet"
        if out_path.exists() and not overwrite:
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "loc_id": loc_id,
                    "parquet_path": str(out_path),
                    "status": "skipped_existing",
                    **_offset_manifest_fields(
                        idx, hist_batch.metadata, fc_batch.metadata
                    ),
                }
            )
            continue
        raw_path = hist_root / model / f"{loc_id}.parquet"
        if not raw_path.exists():
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "loc_id": loc_id,
                    "parquet_path": None,
                    "status": "error: missing raw historic parquet",
                }
            )
            continue
        ref_vals = hist_values[ref_time_mask, idx]
        if np.isfinite(ref_vals).sum() < 30:
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "loc_id": loc_id,
                    "parquet_path": None,
                    "status": "error: insufficient finite QDM reference after fallback",
                    **_offset_manifest_fields(
                        idx, hist_batch.metadata, fc_batch.metadata
                    ),
                }
            )
            continue
        tasks.append((idx, loc_id, str(raw_path)))

    if progress is not None and location_task is not None:
        progress.reset(location_task, total=max(len(tasks), 1))
        progress.update(
            location_task, description=f"[cyan]{model}[/] — {scenario} · mapping"
        )

    if not tasks:
        return pd.DataFrame(rows)

    ctx: dict[str, object] = {
        "hist_values": hist_values,
        "hist_times": hist_times.to_numpy(dtype="datetime64[ns]"),
        "fc_values": fc_values,
        "fc_times": fc_times.to_numpy(dtype="datetime64[ns]"),
        "out_root": str(out_root),
        "model": model,
        "scenario": scenario,
        "hist_meta": hist_batch.metadata,
        "fc_meta": fc_batch.metadata,
    }

    workers = max(1, min(location_workers, len(tasks)))
    if workers == 1:
        for idx, loc_id, raw_path in tasks:
            row = _map_one_location(idx, loc_id, raw_path, ctx)
            rows.append(_enrich_manifest_row(row, ctx, idx))
            _tick_location()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_map_worker,
            initargs=(ctx,),
        ) as executor:
            futures = {
                executor.submit(_map_one_location, idx, loc_id, raw_path, None): idx
                for idx, loc_id, raw_path in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                row = _enrich_manifest_row(future.result(), ctx, idx)
                rows.append(row)
                _tick_location()

    return pd.DataFrame(rows)


def _offset_manifest_fields(
    idx: int,
    hist_meta: pd.DataFrame,
    fc_meta: pd.DataFrame,
) -> dict[str, object]:
    return {
        "historic_offset_km": float(hist_meta.iloc[idx]["historic_offset_km"]),
        "forecast_offset_km": float(fc_meta.iloc[idx]["forecast_offset_km"]),
        "used_historic_fallback": bool(
            hist_meta.iloc[idx]["historic_used_nearest_valid_fallback"]
        ),
        "used_forecast_fallback": bool(
            fc_meta.iloc[idx]["forecast_used_nearest_valid_fallback"]
        ),
    }


def _enrich_manifest_row(
    row: dict[str, object],
    ctx: dict[str, object],
    idx: int,
) -> dict[str, object]:
    return {
        **dict(row),
        **_offset_manifest_fields(idx, ctx["hist_meta"], ctx["fc_meta"]),
    }


def boundary_jump(
    left: pd.Series | pd.DataFrame,
    right: pd.Series | pd.DataFrame,
    *,
    left_year: int = 2014,
    right_year: int = 2015,
    value_col: str | None = None,
) -> float:
    """Mean jump between consecutive periods (e.g. 2014 historic vs 2015 forecast)."""
    if isinstance(left, pd.DataFrame):
        left_vals = left.loc[
            pd.to_datetime(left["time"]).dt.year == left_year, value_col
        ]
    else:
        left_vals = left.loc[left.index.year == left_year]
    if isinstance(right, pd.DataFrame):
        right_vals = right.loc[
            pd.to_datetime(right["time"]).dt.year == right_year, value_col
        ]
    else:
        right_vals = right.loc[right.index.year == right_year]
    return float(right_vals.mean() - left_vals.mean())


