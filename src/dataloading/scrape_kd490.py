### Download Kd490 data from the CEDA archive via xarray [accessed 12.02.26]

import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src import config

BASE_URL = "https://data.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/geographic/netcdf/kd/monthly/v6.0"
YEARS = list(range(1997, 2023))

PART_SUFFIX = ".part"


def part_path_for(dest_path: Path) -> Path:
    return dest_path.with_name(dest_path.name + PART_SUFFIX)


def remove_stale_part_files(root: Path) -> list[Path]:
    """Remove incomplete downloads left from interrupted runs."""
    removed = []
    for path in sorted(root.rglob(f"*{PART_SUFFIX}")):
        path.unlink()
        removed.append(path)
    return removed


def find_netcdf_links(year_url):
    """Get a list of .nc relative URLs for a given year directory, including full links with parameters

    Some of the <a href="..."> links are full URLs and contain '?download=1'.
    We want to capture links that end with '.nc' optionally followed by a query string like '?download=1'.
    """
    response = requests.get(year_url)
    if response.status_code != 200:
        print(f"Could not fetch {year_url} (status {response.status_code})")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")

    netcdf_links = []
    for link in links:
        href = link.get("href", "")
        # accept links that end with '.nc' or '.nc?download=1' etc.
        if ".nc" in href:
            # only accept those where '.nc' is at end or immediately followed by a query string
            idx = href.find(".nc")
            if idx != -1 and (
                idx + 3 == len(href) or href[idx + 3] == "?" or href[idx + 3] == "#"
            ):
                netcdf_links.append(href)
    # some urls are duplicated: use set to drop duplicates
    return list(set(netcdf_links))


def _reset_task_speed(progress: Progress, task_id: TaskID) -> None:
    """Clear speed samples so TimeRemaining reflects only post-prime advances."""
    with progress._lock:
        progress._tasks[task_id]._reset()


def download_file(
    url,
    dest_path,
    overwrite=False,
    chunk_size=5_000_000,
    progress: Progress | None = None,
) -> bool:
    """Stream download to a .part file, then rename when complete."""
    if dest_path.exists() and not overwrite:
        return True

    part_path = part_path_for(dest_path)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        part_path.unlink()

    try:
        with requests.get(url, stream=True, timeout=(30, 600)) as r:
            r.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                    if progress is not None:
                        progress.refresh()
        part_path.replace(dest_path)
        return True
    except Exception as e:
        if part_path.exists():
            part_path.unlink()
        print(f"Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Kd490 data from the CEDA archive"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=str(config.data_dir / "kd490"),
        help="Output directory",
    )
    args = parser.parse_args()
    overwrite = args.overwrite
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    stale_parts = remove_stale_part_files(output_directory)
    if stale_parts:
        print(
            f"Removed {len(stale_parts)} incomplete {PART_SUFFIX} file"
            f"{'s' if len(stale_parts) != 1 else ''} from previous runs."
        )

    # Gather all needed files by year
    print("\nDiscovering .nc files for all years...")
    year_to_links = {}
    total_files = 0
    for year in YEARS:
        year_url = f"{BASE_URL}/{year}/"
        links = find_netcdf_links(year_url)
        total_files += len(links) if links else 0
        year_to_links[year] = links

    # Check existence and prepare download plan
    missing_files_by_year = {year: [] for year in YEARS}
    missing_files = []
    for year, netcdf_links in year_to_links.items():
        year_dir = output_directory / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for href in netcdf_links:
            if href.startswith("http"):
                name_part = urlparse(href).path.split("/")[-1]
            else:
                name_part = href.split("?")[0]
            file_path = year_dir / name_part
            if not file_path.exists() or overwrite:
                missing_files.append((year, href, name_part))
                missing_files_by_year[year].append((href, name_part))

    existing_files = total_files - len(missing_files)
    print(
        f"{existing_files} file{'s' if existing_files != 1 else ''} already present | "
        f"{len(missing_files)} file{'s' if len(missing_files) != 1 else ''} will be downloaded."
    )
    if not missing_files:
        print("All files are present and up to date. Nothing to download.")
        return

    years_with_downloads = [year for year in YEARS if missing_files_by_year[year]]

    total_files_count = sum(len(year_to_links[year]) for year in YEARS)
    already_downloaded_count = total_files_count - len(missing_files)
    missing_count = len(missing_files)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=10,
        transient=False,
    ) as progress:
        overall_task = progress.add_task(
            f"All files [{already_downloaded_count}/{total_files_count}]",
            total=total_files_count,
            completed=already_downloaded_count,
        )
        _reset_task_speed(progress, overall_task)

        # Run-only task: bar 0→N and ETA from this session's downloads only
        run_task = progress.add_task(
            f"This run [0/{missing_count}]",
            total=missing_count,
            completed=0,
        )

        downloads_this_run = 0

        for year in years_with_downloads:
            year_missing_files = missing_files_by_year[year]
            if not year_missing_files:
                continue

            year_total = len(year_to_links[year])
            year_already = year_total - len(year_missing_files)
            year_task = progress.add_task(
                f"Year: {year} ({year_already}/{year_total})",
                total=year_total,
                completed=year_already,
            )
            _reset_task_speed(progress, year_task)

            year_url = f"{BASE_URL}/{year}/"
            year_dir = output_directory / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            year_downloaded_this_run = 0

            for href, name_part in year_missing_files:
                if href.startswith("http"):
                    nc_url = href
                else:
                    nc_url = urljoin(year_url, href)
                out_path = year_dir / name_part

                if download_file(
                    nc_url, out_path, overwrite=overwrite, progress=progress
                ):
                    downloads_this_run += 1
                    year_downloaded_this_run += 1
                    done_in_year = year_already + year_downloaded_this_run
                    done_total = already_downloaded_count + downloads_this_run
                    progress.update(
                        year_task,
                        description=f"Year {year} [{done_in_year}/{year_total}]",
                    )
                    progress.update(
                        overall_task,
                        description=f"All files [{done_total}/{total_files_count}]",
                    )
                    progress.update(
                        run_task,
                        description=f"This run [{downloads_this_run}/{missing_count}]",
                    )
                    progress.advance(year_task)
                    progress.advance(overall_task)
                    progress.advance(run_task)

            progress.remove_task(year_task)

    print(
        f"\nDownload completed ({downloads_this_run}/{missing_count} "
        f"file{'s' if downloads_this_run != 1 else ''} fetched this run)."
    )


if __name__ == "__main__":
    main()
