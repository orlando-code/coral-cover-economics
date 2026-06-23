#!/usr/bin/env python3
"""Compile per-ecoregion species lists and cross-dataset comparison tables.

Builds Red List, OBIS, and COTW species-by-ecoregion CSVs, standardises names
against WoRMS, and writes a three-way presence table plus agreement summary.
Outputs are cached under ``data/ecoregion_diversity/`` and rebuilt only when
source files change or ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src import config
from src.dataloading import diversity

META_FILENAME = "diversity_compile.meta.json"
OUTPUT_FILES = {
    "redlist": "redlist_species_by_ecoregion.csv",
    "obis": "obis_species_by_ecoregion.csv",
    "cotw": "cotw_species_by_ecoregion.csv",
    "presence": "ecoregion_species_presence_redlist_obis_cotw.csv",
    "summary": "ecoregion_species_agreement_summary.csv",
}


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / filename for key, filename in OUTPUT_FILES.items()}


def _source_paths(
    *,
    redlist_dir: Path,
    obis_parquet: Path,
    cotw_species_csv: Path,
) -> dict[str, Path]:
    sources = {
        "cotw_shapefile": diversity.COTW_ECOREGIONS_SHP,
        "obis_parquet": obis_parquet,
        "cotw_species_csv": cotw_species_csv,
    }
    for path in sorted(redlist_dir.glob("data_*.shp")):
        sources[f"redlist/{path.name}"] = path
    return sources


def _cache_is_stale(meta_path: Path, sources: dict[str, Path]) -> bool:
    if not meta_path.exists():
        return True
    cache_mtime = meta_path.stat().st_mtime
    for path in sources.values():
        if not path.exists() or path.stat().st_mtime > cache_mtime:
            return True
    return False


def _outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def _write_cache_meta(
    meta_path: Path,
    *,
    sources: dict[str, Path],
    marine_only: bool,
    paths: dict[str, Path],
    row_counts: dict[str, int],
) -> None:
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "marine_only": marine_only,
        "row_counts": row_counts,
        "outputs": {key: str(path) for key, path in paths.items()},
        "sources": {name: str(path) for name, path in sources.items()},
        "source_mtimes": {
            name: path.stat().st_mtime for name, path in sources.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _load_cached(
    paths: dict[str, Path],
    *,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        print(f"Loading cached diversity compile from {paths['redlist'].parent}")
    return (
        pd.read_csv(paths["redlist"]),
        pd.read_csv(paths["obis"]),
        pd.read_csv(paths["cotw"]),
        pd.read_csv(paths["presence"]),
        pd.read_csv(paths["summary"]),
    )


def compile_diversity(
    *,
    output_dir: Path | None = None,
    redlist_dir: Path | None = None,
    obis_parquet: Path | None = None,
    cotw_species_csv: Path | None = None,
    marine_only: bool = True,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build or load cached species lists and cross-dataset comparison outputs."""
    out_dir = Path(output_dir or config.diversity_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(out_dir)
    meta_path = out_dir / META_FILENAME

    resolved_redlist_dir = Path(redlist_dir or diversity.REDLIST_DIR)
    resolved_obis_parquet = Path(obis_parquet or diversity.OBIS_ASSIGNED_PARQUET)
    resolved_cotw_species_csv = Path(cotw_species_csv or diversity.COTW_SPECIES_CSV)
    sources = _source_paths(
        redlist_dir=resolved_redlist_dir,
        obis_parquet=resolved_obis_parquet,
        cotw_species_csv=resolved_cotw_species_csv,
    )

    if not force_rebuild and _outputs_complete(paths) and not _cache_is_stale(
        meta_path, sources
    ):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("marine_only", True) == marine_only:
                return _load_cached(paths, verbose=verbose)
        except (json.JSONDecodeError, OSError):
            pass

    if verbose:
        reason = "forced rebuild" if force_rebuild else "cache missing or stale"
        print(f"Building diversity compile ({reason})...")

    if verbose:
        print("Loading COTW ecoregion polygons...")
    cotw_eco_gdf = diversity.load_cotw_ecoregions_for_assignment()

    if verbose:
        print("Loading IUCN Red List species ranges...")
    redlist_gdf = diversity.load_redlist_species(resolved_redlist_dir)

    if verbose:
        print("Loading OBIS occurrences with ecoregion assignments...")
    out_cotw_gdf = diversity.load_obis_occurrences(
        assigned_parquet=resolved_obis_parquet
    )

    if verbose:
        print("Building per-ecoregion species lists...")
    redlist_species_by_ecoregion = diversity.list_redlist_species_by_region(
        redlist_gdf, cotw_eco_gdf
    )
    obis_species_by_ecoregion = diversity.list_obis_species_by_region(
        cotw_eco_gdf, out_cotw_gdf, assign_if_needed=False, verbose=verbose
    )
    cotw_species_by_ecoregion = diversity.list_cotw_species_by_region(
        cotw_eco_gdf, diversity.load_cotw_species(resolved_cotw_species_csv)
    )

    if verbose:
        print("Standardising names via WoRMS...")
    (
        redlist_species_by_ecoregion,
        obis_species_by_ecoregion,
        cotw_species_by_ecoregion,
    ) = diversity.standardize_species_lists_for_comparison(
        redlist_species_by_ecoregion,
        obis_species_by_ecoregion,
        cotw_species_by_ecoregion,
        marine_only=marine_only,
        verbose=verbose,
    )

    if verbose:
        print("Building three-way species presence table...")
    species_presence = diversity.compare_ecoregion_species_lists(
        redlist_species_by_ecoregion,
        obis_species_by_ecoregion,
        cotw_species_by_ecoregion,
    )
    species_agreement_summary = diversity.summarize_ecoregion_species_agreement(
        species_presence
    )

    redlist_species_by_ecoregion.to_csv(paths["redlist"], index=False)
    obis_species_by_ecoregion.to_csv(paths["obis"], index=False)
    cotw_species_by_ecoregion.to_csv(paths["cotw"], index=False)
    species_presence.to_csv(paths["presence"], index=False)
    species_agreement_summary.to_csv(paths["summary"], index=False)

    _write_cache_meta(
        meta_path,
        sources=sources,
        marine_only=marine_only,
        paths=paths,
        row_counts={
            "redlist": len(redlist_species_by_ecoregion),
            "obis": len(obis_species_by_ecoregion),
            "cotw": len(cotw_species_by_ecoregion),
            "presence": len(species_presence),
            "summary": len(species_agreement_summary),
        },
    )

    if verbose:
        print(f"Wrote {paths['redlist']}")
        print(f"Wrote {paths['obis']}")
        print(f"Wrote {paths['cotw']}")
        print(f"Wrote {paths['presence']}")
        print(f"Wrote {paths['summary']}")
        print(f"Wrote {meta_path}")

    return (
        redlist_species_by_ecoregion,
        obis_species_by_ecoregion,
        cotw_species_by_ecoregion,
        species_presence,
        species_agreement_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile Red List, OBIS, and COTW species lists by ecoregion, "
            "standardise names via WoRMS, and write cross-dataset comparison tables."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {config.diversity_dir})",
    )
    parser.add_argument(
        "--redlist-dir",
        type=Path,
        default=None,
        help="Directory containing IUCN Red List shapefiles (data_*.shp)",
    )
    parser.add_argument(
        "--obis-parquet",
        type=Path,
        default=None,
        help="OBIS occurrences parquet with ecoregion assignments",
    )
    parser.add_argument(
        "--cotw-species-csv",
        type=Path,
        default=None,
        help="Scraped COTW ecoregion species CSV",
    )
    parser.add_argument(
        "--no-marine-only",
        action="store_true",
        help="Include non-marine taxa in WoRMS matching",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cached outputs are fresh",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    compile_diversity(
        output_dir=args.output_dir,
        redlist_dir=args.redlist_dir,
        obis_parquet=args.obis_parquet,
        cotw_species_csv=args.cotw_species_csv,
        marine_only=not args.no_marine_only,
        force_rebuild=args.force,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
