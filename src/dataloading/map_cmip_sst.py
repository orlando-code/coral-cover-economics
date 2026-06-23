#!/usr/bin/env python3
"""Batch quantile-map CMIP historic point SST onto QDM-corrected continuous series."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src import config
from src.dataloading.cmip_sst import (
    CMIP_MAPPED_POINT_DIR,
    QDM_OFFSET_DIRNAME,
    SCENARIO_DIRS,
    discover_models,
    load_locations,
    map_model_scenario,
)

console = Console(highlight=False)


def _status_table(manifest: pd.DataFrame) -> Table:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("status")
    table.add_column("count", justify="right")
    if manifest.empty:
        return table
    for status, count in manifest["status"].value_counts().items():
        table.add_row(str(status), f"{count:,}")
    if "used_historic_fallback" in manifest.columns:
        n_hist = int(manifest["used_historic_fallback"].eq(True).sum())
        n_fc = int(manifest["used_forecast_fallback"].eq(True).sum())
        table.add_row("used_historic_fallback", f"{n_hist:,}")
        table.add_row("used_forecast_fallback", f"{n_fc:,}")
    return table


def run_mapping(
    *,
    models: list[str],
    scenarios: list[str],
    output_dir: Path,
    historic_dir: Path,
    daily_sst_dir: Path,
    location_workers: int,
    overwrite: bool,
    limit_locations: int | None,
) -> pd.DataFrame:
    locations = load_locations(historic_dir)
    if limit_locations is not None:
        locations = locations.head(limit_locations).copy()

    manifests: list[pd.DataFrame] = []
    total_jobs = len(models) * len(scenarios)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description:<36}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        overall_task = progress.add_task("Overall", total=total_jobs, detail="")
        location_task = progress.add_task(
            "Locations",
            total=1,
            detail="waiting",
        )
        job_i = 0

        for model in models:
            scenario_task = progress.add_task(
                f"[cyan]{model}[/]",
                total=len(scenarios),
                detail="queued",
            )
            for scenario in scenarios:
                job_i += 1
                progress.update(
                    overall_task,
                    description=f"Overall ({job_i}/{total_jobs})",
                    detail=f"{model} / {scenario}",
                )
                progress.update(
                    scenario_task,
                    description=f"[cyan]{model}[/]",
                    completed=scenarios.index(scenario),
                    detail=scenario,
                )

                t0 = time.perf_counter()
                manifest = map_model_scenario(
                    model,
                    scenario,
                    locations,
                    output_dir=output_dir,
                    historic_dir=historic_dir,
                    daily_sst_dir=daily_sst_dir,
                    overwrite=overwrite,
                    location_workers=location_workers,
                    progress=progress,
                    scenario_task=scenario_task,
                    location_task=location_task,
                )
                elapsed = time.perf_counter() - t0
                if manifest["status"].str.startswith("skipped_missing").any():
                    console.print(
                        f"[yellow]Skipping {model}/{scenario}: "
                        f"{manifest['status'].iloc[0]}[/]"
                    )
                manifests.append(manifest)

                progress.advance(scenario_task)
                progress.advance(overall_task)
                progress.update(
                    scenario_task,
                    detail=f"{scenario} done in {elapsed:.0f}s",
                )

            progress.remove_task(scenario_task)

    return pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantile-map CMIP historic point SST and stitch to QDM forecast "
            "for all reef locations, models, and scenarios."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="CMIP6 models (default: all with historic + QDM data).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_DIRS),
        choices=list(SCENARIO_DIRS),
        help="SSP scenarios to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CMIP_MAPPED_POINT_DIR,
        help="Root directory for mapped Parquet output.",
    )
    parser.add_argument(
        "--historic-dir",
        type=Path,
        default=config.env_dir / "cmip_historic_point_timeseries",
        help="Raw CMIP historic point Parquet directory.",
    )
    parser.add_argument(
        "--daily-sst-dir",
        type=Path,
        default=config.env_dir / "DailySST",
        help="Sherwood QDM-corrected DailySST NetCDF directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for location quantile-mapping and Parquet writes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mapped Parquet files.",
    )
    parser.add_argument(
        "--limit-locations",
        type=int,
        default=None,
        help="Process only the first N locations (for testing).",
    )
    args = parser.parse_args()

    models = args.models or discover_models(
        historic_dir=args.historic_dir,
        daily_sst_dir=args.daily_sst_dir,
    )
    if not models:
        raise SystemExit(
            "No models found with both historic point data and QDM NetCDF files."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Models[/bold]: {', '.join(models)}",
                    f"[bold]Scenarios[/bold]: {', '.join(args.scenarios)}",
                    f"[bold]Workers[/bold]: {args.workers}",
                    f"[bold]Output[/bold]: {args.output_dir}",
                    (
                        "[dim]Coastal points on land in the QDM grid use the nearest "
                        "valid ocean cell; offsets are written to "
                        f"{QDM_OFFSET_DIRNAME}/[/dim]"
                    ),
                ]
            ),
            title="CMIP SST quantile mapping",
            border_style="blue",
        )
    )

    t0 = time.perf_counter()
    manifest = run_mapping(
        models=models,
        scenarios=args.scenarios,
        output_dir=args.output_dir,
        historic_dir=args.historic_dir,
        daily_sst_dir=args.daily_sst_dir,
        location_workers=max(1, args.workers),
        overwrite=args.overwrite,
        limit_locations=args.limit_locations,
    )
    manifest_path = args.output_dir / "mapping_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    elapsed = time.perf_counter() - t0
    console.print()
    console.print(
        Panel.fit(
            f"Finished in [bold]{elapsed / 60:.1f} min[/bold]\n"
            f"Manifest: [link={manifest_path}]{manifest_path}[/link]\n"
            f"Offsets: [bold]{args.output_dir / QDM_OFFSET_DIRNAME}/[/bold]",
            title="Done",
            border_style="green",
        )
    )
    if not manifest.empty:
        console.print(_status_table(manifest))


if __name__ == "__main__":
    main()
