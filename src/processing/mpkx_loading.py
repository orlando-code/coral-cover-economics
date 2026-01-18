"""Load ArcGIS Map Package (.mpkx) files into GeoDataFrames."""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import py7zr
import pyogrio


def load_mpkx(
    path: str | Path,
    extract_dir: Optional[Path] = None,
    export_parquet: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """Extract an .mpkx file and load all vector layers as GeoDataFrames.
    
    Args:
        path: Path to the .mpkx file
        extract_dir: Directory to extract to. Defaults to sibling folder with same name.
        export_parquet: If True, exports layers to geoparquet for fast future loading.
        
    Returns:
        Dictionary mapping layer names to GeoDataFrames.
    """
    path = Path(path)
    extract_dir = extract_dir or path.parent / path.stem
    
    # Extract if not already done
    if not extract_dir.exists():
        with py7zr.SevenZipFile(path, mode='r') as archive:
            archive.extractall(path=extract_dir)
            
    print(f"Extracted mpkx file to {extract_dir}")
    # Load all vector layers from geodatabases
    layers = {}
    parquet_dir = extract_dir / "geoparquet"
    
    for gdb_path in extract_dir.rglob("*.gdb"):
        try:
            for layer_name, _ in pyogrio.list_layers(gdb_path):
                # check if the layer already exists as a parquet file
                if (parquet_dir / f"{layer_name}.parquet").exists():
                    print(f"Layer {layer_name} already exists as a parquet file. Skipping extraction...")
                    layers[layer_name] = gpd.read_parquet(parquet_dir / f"{layer_name}.parquet")
                    continue
                else:
                    gdf = gpd.read_file(gdb_path, layer=layer_name)
                    layers[layer_name] = gdf                    
        except Exception:
            continue  # Skip raster geodatabases
    
    return layers


def load_parquet_layers(path: str | Path) -> dict[str, gpd.GeoDataFrame]:
    """Load previously exported geoparquet layers from an extracted mpkx directory."""
    path = Path(path)
    parquet_dir = path / "geoparquet" if path.is_dir() else path.parent / path.stem / "geoparquet"
    return {f.stem: gpd.read_parquet(f) for f in parquet_dir.glob("*.parquet")}
