"""
Download the International Best Track Archive for Climate Stewardship (IBTrACS) cyclone data.

Link to page: https://www.ncei.noaa.gov/products/international-best-track-archive
Dataset DOI: 10.25921/82ty-9e16
NCEI data set identification (DSI): 9637

"""

import argparse
from pathlib import Path

import requests

# custom
from src import config

CYCLONES_FP = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc"


def main():
    parser = argparse.ArgumentParser(description="Download IBTrACS cyclones data")
    parser.add_argument(
        "--output_directory",
        type=str,
        default=str(config.data_dir / "cyclones"),
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )
    args = parser.parse_args()
    overwrite = args.overwrite
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / "cyclones.nc"
    if out_path.exists() and not overwrite:
        print(f"Data already exists in {out_path}. Use --overwrite flag to overwrite.")
        return

    print("Downloading IBTrACS cyclones data...")
    response = requests.get(CYCLONES_FP)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"Data downloaded to {out_path}")


if __name__ == "__main__":
    main()
