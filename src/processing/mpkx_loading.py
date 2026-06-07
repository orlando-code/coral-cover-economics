"""Utilities for extracting and processing ArcGIS `.mpkx` packages.

Public entry points:
- `process_mpkx`: one-shot extract + vectors + raster discovery/export + manifest
- `load_mpkx`: backward-compatible vector-only loader
- `list_mpkx_rasters` / `open_mpkx_raster`: inspect/open raster sources
- `export_mpkx_rasters`: export discovered rasters to GeoTIFF
"""

# general
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

# spatial
import geopandas as gpd
import py7zr
import pyogrio
import rasterio


def _looks_extracted(extract_dir: Path) -> bool:
    """Return True if a directory looks like an extracted MPKX tree."""
    if not extract_dir.is_dir():
        return False
    return (
        any(extract_dir.rglob("*.gdb"))
        or (extract_dir / "esriinfo").is_dir()
        or (extract_dir / "commondata").is_dir()
    )


def _resolve_paths(
    source: str | Path, extract_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Resolve `(mpkx_path, extract_dir)` from either a package path or extract directory."""
    source_path = Path(source)
    if source_path.suffix.lower() == ".mpkx":
        mpkx_path = source_path
        out_dir = (
            Path(extract_dir)
            if extract_dir is not None
            else mpkx_path.parent / mpkx_path.stem
        )
    else:
        out_dir = Path(extract_dir) if extract_dir is not None else source_path
        mpkx_path = out_dir.parent / f"{out_dir.name}.mpkx"
    return mpkx_path, out_dir


def _ensure_extracted(
    mpkx_path: Path,
    extract_dir: Path,
    *,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """Extract `mpkx_path` to `extract_dir` if needed."""
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if _looks_extracted(extract_dir):
        if verbose:
            print(f"Using existing extract at {extract_dir}")
        return

    if not mpkx_path.is_file():
        raise FileNotFoundError(
            f"Could not find MPKX at {mpkx_path}. Pass source as an existing .mpkx path."
        )

    if extract_dir.exists() and any(extract_dir.iterdir()):
        warnings.warn(
            f"Incomplete extract at {extract_dir}; re-extracting from {mpkx_path.name}.",
            stacklevel=2,
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Extracting {mpkx_path.name} -> {extract_dir}")
    with py7zr.SevenZipFile(mpkx_path, mode="r") as archive:
        archive.extractall(path=extract_dir)


def _slug(name: str) -> str:
    """Normalize names for robust matching (e.g. `Predicted Catch` -> `predicted_catch`)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _discover_raster_sources(extract_dir: Path) -> dict[str, str]:
    """Discover raster sources in an extracted MPKX.

    Returns a mapping:
      normalized raster name -> source string
    where source is either:
      - direct file path to a GeoTIFF, or
      - rasterio URI like `OpenFileGDB:/path/to.gdb:Dataset`.
    """
    rasters: dict[str, str] = {}

    # 1) Direct GeoTIFFs
    for tif in extract_dir.rglob("*.tif"):
        if ".ovr" in tif.name.lower():
            continue
        key = _slug(tif.stem)
        # Prefer /<name>/<name>.tif over duplicate output folders.
        if key not in rasters or tif.parent.name == tif.stem:
            rasters[key] = str(tif)

    # 2) Rasters referenced from .mapx definitions
    for mapx in extract_dir.rglob("*.mapx"):
        try:
            obj = json.loads(mapx.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        for layer in obj.get("layerDefinitions", []):
            if layer.get("type") != "CIMRasterLayer":
                continue
            conn = layer.get("dataConnection", {})
            if conn.get("datasetType") != "esriDTRasterDataset":
                continue

            dataset = conn.get("dataset")
            ws = conn.get("workspaceConnectionString", "")
            if not dataset or not ws.startswith("DATABASE="):
                continue

            gdb_rel = ws.split("DATABASE=", 1)[1].replace("\\", "/")
            gdb_path = (mapx.parent / gdb_rel).resolve()
            if not gdb_path.exists():
                continue

            uri = f"OpenFileGDB:{gdb_path}:{dataset}"
            layer_name = layer.get("name", dataset)
            for key in {_slug(layer_name), _slug(dataset)}:
                rasters.setdefault(key, uri)

    return rasters


def _discover_vector_layers(
    extract_dir: Path,
    *,
    cache_parquet: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, str]]:
    """Load vector layers from extracted GDBs.

    Returns:
      - layer name -> GeoDataFrame
      - layer name -> source path (parquet cache or gdb:layer)
    """
    gdb_paths = list(extract_dir.rglob("*.gdb"))
    if not gdb_paths:
        warnings.warn(f"No *.gdb under {extract_dir}", stacklevel=2)
        return {}, {}

    parquet_dir = extract_dir / "geoparquet"
    if cache_parquet:
        parquet_dir.mkdir(parents=True, exist_ok=True)

    layers: dict[str, gpd.GeoDataFrame] = {}
    sources: dict[str, str] = {}

    for gdb in gdb_paths:
        try:
            layer_list = list(pyogrio.list_layers(gdb))
        except Exception as exc:
            # Common and expected for raster GDBs with vector-only drivers.
            msg = str(exc).lower()
            if "not recognized" in msg or "supported file format" in msg:
                if verbose:
                    print(f"  skipped {gdb.name} (raster geodatabase not supported)")
                continue
            warnings.warn(f"Could not list layers in {gdb}: {exc}", stacklevel=2)
            continue

        for name, _ in layer_list:
            pq = parquet_dir / f"{name}.parquet"
            if cache_parquet and pq.exists():
                layers[name] = gpd.read_parquet(pq)
                sources[name] = str(pq)
                continue

            try:
                gdf = gpd.read_file(gdb, layer=name)
            except Exception as exc:
                warnings.warn(f"{name} ({gdb.name}): {exc}", stacklevel=2)
                continue

            if cache_parquet:
                gdf.to_parquet(pq)
                sources[name] = str(pq)
            else:
                sources[name] = f"{gdb}:{name}"

            layers[name] = gdf
            if verbose:
                print(f"  {name}: {len(gdf)} from {gdb.name}")

    return layers, sources


def list_mpkx_rasters(
    source: str | Path,
    extract_dir: str | Path | None = None,
    *,
    force_reextract: bool = False,
    verbose: bool = False,
) -> dict[str, str]:
    """List raster sources available in an MPKX.

    Args:
        source: `.mpkx` file path or extracted directory.
        extract_dir: Optional extraction target when `source` is `.mpkx`.
        force_reextract: Re-extract package before listing.
        verbose: Print extraction status.

    Returns:
        Dict mapping normalized raster name to openable source string.
    """
    mpkx_path, out_dir = _resolve_paths(source, extract_dir=extract_dir)
    _ensure_extracted(mpkx_path, out_dir, force=force_reextract, verbose=verbose)
    return _discover_raster_sources(out_dir)


def open_mpkx_raster(
    source: str | Path,
    name: str = "predicted_catch",
    *,
    extract_dir: str | Path | None = None,
    force_reextract: bool = False,
):
    """Open a named MPKX raster with rasterio.

    The `name` is normalized internally (case/punctuation-insensitive).
    """
    rasters = list_mpkx_rasters(
        source,
        extract_dir=extract_dir,
        force_reextract=force_reextract,
        verbose=False,
    )
    key = _slug(name)
    if key not in rasters:
        available = ", ".join(sorted(rasters)) or "(none)"
        raise KeyError(f"Raster '{name}' not found in MPKX. Available: {available}")
    return rasterio.open(rasters[key])


def export_mpkx_rasters(
    source: str | Path,
    extract_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    force_reextract: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Export discovered MPKX rasters to GeoTIFF files.

    Duplicate aliases that resolve to the same source are de-duplicated.
    """
    mpkx_path, out_dir = _resolve_paths(source, extract_dir=extract_dir)
    _ensure_extracted(mpkx_path, out_dir, force=force_reextract, verbose=verbose)
    raster_sources = _discover_raster_sources(out_dir)

    output = (
        Path(output_dir) if output_dir is not None else out_dir / "rasters_exported"
    )
    output.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    source_to_dst: dict[str, Path] = {}

    for name, src in sorted(raster_sources.items()):
        if src in source_to_dst:
            exported[name] = source_to_dst[src]
            continue

        dst = output / f"{name}.tif"
        if dst.exists() and not overwrite:
            source_to_dst[src] = dst
            exported[name] = dst
            continue

        with rasterio.open(src) as ds:
            profile = ds.profile.copy()
            profile.update(driver="GTiff")
            with rasterio.open(dst, "w", **profile) as out:
                for band in range(1, ds.count + 1):
                    out.write(ds.read(band), band)

        source_to_dst[src] = dst
        exported[name] = dst
        if verbose:
            print(f"  exported raster {name} -> {dst}")

    return exported


def process_mpkx(
    source: str | Path,
    extract_dir: str | Path | None = None,
    *,
    force_reextract: bool = False,
    cache_parquet: bool = True,
    export_rasters: bool = True,
    raster_output_dir: str | Path | None = None,
    overwrite_rasters: bool = False,
    save_manifest: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run full MPKX processing.

    Steps:
      1) Extract package
      2) Load vector layers (optional parquet cache)
      3) Discover raster sources
      4) Optionally export rasters to GeoTIFF
      5) Optionally write `mpkx_manifest.json`
    """
    mpkx_path, out_dir = _resolve_paths(source, extract_dir=extract_dir)
    _ensure_extracted(mpkx_path, out_dir, force=force_reextract, verbose=verbose)

    if verbose:
        print("Loading vector layers...")
    vectors, vector_sources = _discover_vector_layers(
        out_dir, cache_parquet=cache_parquet, verbose=verbose
    )

    if verbose:
        print("Discovering rasters...")
    raster_sources = _discover_raster_sources(out_dir)

    exported_rasters: dict[str, Path] = {}
    if export_rasters and raster_sources:
        if verbose:
            print("Exporting rasters to GeoTIFF...")
        exported_rasters = export_mpkx_rasters(
            mpkx_path,
            extract_dir=out_dir,
            output_dir=raster_output_dir,
            force_reextract=False,
            overwrite=overwrite_rasters,
            verbose=verbose,
        )

    manifest = {
        "mpkx_path": str(mpkx_path),
        "extract_dir": str(out_dir),
        "vector_layers": {
            name: {"source": vector_sources.get(name, ""), "n_features": len(gdf)}
            for name, gdf in vectors.items()
        },
        "raster_sources": raster_sources,
        "exported_rasters": {k: str(v) for k, v in exported_rasters.items()},
    }

    manifest_path = None
    if save_manifest:
        manifest_path = out_dir / "mpkx_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if verbose:
            print(f"Saved manifest -> {manifest_path}")

    return {
        "vectors": vectors,
        "raster_sources": raster_sources,
        "exported_rasters": exported_rasters,
        "manifest_path": manifest_path,
        "extract_dir": out_dir,
    }


def load_mpkx(
    path: str | Path,
    extract_dir: str | Path | None = None,
    *,
    force_reextract: bool = False,
    cache_parquet: bool = True,
    verbose: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """Backward-compatible wrapper returning only vector layers."""
    result = process_mpkx(
        path,
        extract_dir=extract_dir,
        force_reextract=force_reextract,
        cache_parquet=cache_parquet,
        export_rasters=False,
        save_manifest=False,
        verbose=verbose,
    )
    return result["vectors"]


def load_parquet_layers(path: str | Path) -> dict[str, gpd.GeoDataFrame]:
    """Load cached geoparquet layers from an extracted MPKX directory."""
    path_obj = Path(path)
    parquet_dir = (
        path_obj / "geoparquet"
        if path_obj.is_dir()
        else path_obj.parent / path_obj.stem / "geoparquet"
    )
    files = list(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    return {f.stem: gpd.read_parquet(f) for f in files}
