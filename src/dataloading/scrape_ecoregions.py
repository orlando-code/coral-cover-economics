#!/usr/bin/env python3
"""
Scrape ecoregion fact sheets from coralsoftheworld.org.

Produces:
  - ecoregions.csv: ecoregion name, total species number, QA flag, and page URL
  - ecoregion_species.csv: one row per ecoregion/species pair from species pages
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src import config

BASE_URL = "https://www.coralsoftheworld.org/ecoregion_factsheets/ecoregion_factsheet_general_info/"
SPECIES_URL = (
    "https://www.coralsoftheworld.org/ecoregion_factsheets/ecoregion_factsheet_species/"
)
PARENT_URL = "https://www.coralsoftheworld.org/ecoregion_factsheets/"
SITE_URL = "https://www.coralsoftheworld.org"

REQUEST_DELAY = 0.1
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def get_index_soup() -> BeautifulSoup:
    """Fetch the ecoregion list page (parent). The general_info index returns 500."""
    resp = requests.get(PARENT_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _path_to_slug(path: str, base_path: str, parent_path: str) -> str | None:
    """Extract ecoregion slug from URL path; return None if not a single-segment ecoregion."""
    path = path.rstrip("/") + "/"
    if path.startswith(base_path) and path != base_path:
        slug = path[len(base_path) :].rstrip("/")
    elif path.startswith(parent_path) and path != parent_path:
        rest = path[len(parent_path) :].rstrip("/")
        if not rest or "/" in rest or rest == "ecoregion_factsheet_general_info":
            return None
        slug = rest
    else:
        return None
    return slug if slug and "/" not in slug else None


def _slug_from_name(name: str) -> str:
    """Convert display name to URL slug: lowercase, spaces/commas to hyphens, drop other punctuation."""
    s = name.strip().lower()
    s = re.sub(r"[\s,]+", "-", s)
    s = re.sub(r"[^\w\-]", "", s)
    return re.sub(r"-+", "-", s).strip("-")


def _ecoregion_names_from_page_text(soup: BeautifulSoup) -> list[str]:
    """
    Heuristic: ecoregion names appear as concatenated text (e.g. '...centralRed Sea south...').
    Split at boundaries where a lowercase letter is followed by an uppercase (start of next name).
    """
    text = soup.get_text(separator=" ", strip=True)
    parts = re.split(r"(?<=[a-z0-9\)])(?=[A-Z])", text)
    single_word_ecoregions = {
        "maldives",
        "palau",
        "marianas",
        "vanuatu",
        "fiji",
        "bermuda",
        "brazil",
    }
    names = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < 3:
            continue
        if " " in p or "," in p or p.lower() in single_word_ecoregions:
            if (
                "ecoregion" not in p.lower()
                and "select" not in p.lower()
                and "apply" not in p.lower()
            ):
                names.append(p)
    return names


def _is_name_based_slug(slug: str) -> bool:
    """True if slug looks like a name slug (letters and hyphens), not a numeric ID."""
    if not slug or slug.isdigit():
        return False
    return bool(re.match(r"^[a-z0-9\-]+$", slug)) and any(c.isalpha() for c in slug)


def discover_ecoregion_urls(soup: BeautifulSoup) -> list[str]:
    """
    Build ecoregion fact sheet URLs from names. The site uses name-based slugs
    (e.g. red-sea-north-central), not numeric IDs.
    """
    base_path = urlparse(BASE_URL).path.rstrip("/") + "/"
    parent_path = urlparse(PARENT_URL).path.rstrip("/") + "/"
    seen_slugs: set[str] = set()
    urls: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href == "#" or href.startswith("javascript:"):
            continue
        full_url = urljoin(PARENT_URL, href)
        parsed = urlparse(full_url)
        slug = _path_to_slug(parsed.path, base_path, parent_path)
        if slug and _is_name_based_slug(slug) and slug not in seen_slugs:
            seen_slugs.add(slug)
            urls.append(urljoin(BASE_URL, slug + "/"))

    for opt in soup.find_all("option", value=True):
        name = opt.get_text(strip=True)
        if not name or name in ("", "Select saved ecoregion list"):
            continue
        slug = _slug_from_name(name)
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            urls.append(urljoin(BASE_URL, slug + "/"))

    for opt in soup.find_all("option", value=True):
        val = opt["value"].strip().strip("/")
        if not val:
            continue
        bare = val.split("/")[-1] if "/" in val else val
        if _is_name_based_slug(bare) and bare not in seen_slugs:
            seen_slugs.add(bare)
            urls.append(urljoin(BASE_URL, bare + "/"))

    if not urls:
        names = _ecoregion_names_from_page_text(soup)
        for name in names:
            slug = _slug_from_name(name)
            if slug and slug not in seen_slugs:
                seen_slugs.add(slug)
                urls.append(urljoin(BASE_URL, slug + "/"))
        if urls:
            print("(Used ecoregion names from page text to build URLs.)")

    return sorted(set(urls))


def slug_from_ecoregion_url(url: str) -> str:
    """Return the ecoregion slug from a general-info or species page URL."""
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1]


def species_url_from_ecoregion_url(url: str) -> str:
    """Convert a general-info ecoregion URL to the species-list page URL."""
    slug = slug_from_ecoregion_url(url)
    return urljoin(SPECIES_URL, slug + "/")


def parse_species_number(text: str) -> int | None:
    """Extract integer from text like 'Total species number: 310'."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    match = re.search(r"Total\s+species\s+number\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r":\s*(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def parse_species_list(html: str) -> list[dict[str, str]]:
    """Extract species entries embedded as ``speciesList`` JSON in page scripts."""
    match = re.search(r"var\s+speciesList\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not match:
        return []
    try:
        species = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(species, list):
        return []
    return [entry for entry in species if isinstance(entry, dict) and entry.get("name")]


def scrape_ecoregion_page(url: str) -> tuple[str | None, int | None, int]:
    """Scrape one general-info ecoregion page. Returns (name, species_number, qa)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if not resp.ok:
            return None, None, 0
    except Exception:
        return None, None, 0

    soup = BeautifulSoup(resp.text, "html.parser")

    name = None
    left = soup.find("span", class_="header-left-container")
    if left:
        h3 = left.find("h3")
        if h3 and h3.get_text(strip=True):
            name = h3.get_text(strip=True)

    species_number = None
    right = soup.find("span", class_="header-right-container")
    if right:
        info = right.find("span", class_="info-container")
        if info:
            h5 = info.find("h5")
            if h5:
                species_number = parse_species_number(h5.get_text())

    qa = 1 if (name is not None and species_number is not None) else 0
    return name, species_number, qa


def scrape_ecoregion_species_page(
    url: str,
    *,
    ecoregion_name: str | None = None,
    ecoregion_slug: str | None = None,
) -> tuple[list[dict[str, str | int | None]], int]:
    """
    Scrape one species-list page. Returns (rows, qa).

    Each row contains ecoregion metadata and one species entry.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if not resp.ok:
            return [], 0
    except Exception:
        return [], 0

    soup = BeautifulSoup(resp.text, "html.parser")
    slug = ecoregion_slug or slug_from_ecoregion_url(url)

    name = ecoregion_name
    if not name:
        left = soup.find("span", class_="header-left-container")
        if left:
            h3 = left.find("h3")
            if h3 and h3.get_text(strip=True):
                name = h3.get_text(strip=True)

    listed_total = None
    right = soup.find("span", class_="header-right-container")
    if right:
        info = right.find("span", class_="info-container")
        if info:
            h5 = info.find("h5")
            if h5:
                listed_total = parse_species_number(h5.get_text())

    species_entries = parse_species_list(resp.text)
    rows: list[dict[str, str | int | None]] = []
    for entry in species_entries:
        species_name = entry["name"].strip()
        species_path = entry.get("species_factsheet_url", "")
        rows.append(
            {
                "ecoregion_name": name,
                "ecoregion_slug": slug,
                "species_name": species_name,
                "species_slug": slug_from_ecoregion_url(urljoin(SITE_URL, species_path))
                if species_path
                else _slug_from_name(species_name),
                "species_factsheet_url": urljoin(SITE_URL, species_path)
                if species_path
                else None,
                "ecoregion_species_url": url,
                "listed_total_species_number": listed_total,
            }
        )

    count_matches = listed_total is None or len(rows) == listed_total
    qa = 1 if (name and rows and count_matches) else 0
    return rows, qa


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Corals of the World ecoregion pages for biodiversity data"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing ecoregions.csv"
    )
    parser.add_argument(
        "--overwrite-species",
        action="store_true",
        help="Overwrite existing ecoregion_species.csv",
    )
    parser.add_argument(
        "--skip-species",
        action="store_true",
        help="Only scrape ecoregion summary pages, not species lists",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=str(config.data_dir / "ecoregion_diversity"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / "ecoregions.csv"
    species_out_path = output_directory / "ecoregion_species.csv"

    if out_path.exists() and not args.overwrite and not args.overwrite_species:
        print(f"Data already exists in {out_path}. Use --overwrite to overwrite.")
        return

    scrape_summaries = args.overwrite or not out_path.exists()
    scrape_species = not args.skip_species and (
        args.overwrite or args.overwrite_species or not species_out_path.exists()
    )

    if not scrape_summaries and not scrape_species:
        print(
            "Nothing to do: output files exist. Use --overwrite or --overwrite-species."
        )
        return

    if scrape_summaries:
        print("Fetching ecoregion list...")
        soup = get_index_soup()
        urls = discover_ecoregion_urls(soup)
        print(f"Found {len(urls)} ecoregion pages.")
    else:
        existing = pd.read_csv(out_path)
        urls = existing["url"].dropna().tolist()
        print(f"Using {len(urls)} ecoregion URLs from {out_path}.")

    ecoregion_rows = []
    species_rows = []
    species_qa_failures = 0
    existing_names: dict[str, str | None] = {}
    if not scrape_summaries and out_path.exists():
        existing = pd.read_csv(out_path)
        if "ecoregion_slug" not in existing.columns:
            existing["ecoregion_slug"] = existing["url"].map(slug_from_ecoregion_url)
        existing_names = existing.set_index("ecoregion_slug")[
            "ecoregion_name"
        ].to_dict()

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description} [bold]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Scraping ecoregion pages", total=len(urls))
        for i, url in enumerate(urls, 1):
            slug = slug_from_ecoregion_url(url)
            progress.update(
                task,
                completed=i - 1,
                description=f"[{i}/{len(urls)}] {slug}",
            )

            if scrape_summaries:
                name, species_number, qa = scrape_ecoregion_page(url)
                ecoregion_rows.append(
                    {
                        "ecoregion_name": name,
                        "total_species_number": species_number,
                        "QA": qa,
                        "url": url,
                        "ecoregion_slug": slug,
                        "species_url": species_url_from_ecoregion_url(url),
                    }
                )
            else:
                name = existing_names.get(slug)

            if scrape_species:
                species_url = species_url_from_ecoregion_url(url)
                page_rows, species_qa = scrape_ecoregion_species_page(
                    species_url,
                    ecoregion_name=name,
                    ecoregion_slug=slug,
                )
                if species_qa == 0:
                    species_qa_failures += 1
                species_rows.extend(page_rows)

            progress.advance(task)
            time.sleep(REQUEST_DELAY)

    if scrape_summaries:
        ecoregions_df = pd.DataFrame(ecoregion_rows)
        ecoregions_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(ecoregions_df)} ecoregions to {out_path}")

        if (ecoregions_df["QA"] == 0).any():
            n_bad = (ecoregions_df["QA"] == 0).sum()
            print(f"Ecoregion QA flag: {n_bad} row(s) with possible formatting issues.")

    if scrape_species:
        species_df = pd.DataFrame(species_rows)
        species_df.to_csv(species_out_path, index=False)
        print(
            f"Saved {len(species_df)} species records "
            f"({species_df['ecoregion_slug'].nunique()} ecoregions) to {species_out_path}"
        )
        if species_qa_failures:
            print(
                f"Species QA flag: {species_qa_failures} ecoregion page(s) "
                "with missing or mismatched species lists."
            )


if __name__ == "__main__":
    main()
