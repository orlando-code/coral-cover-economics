#!/usr/bin/env python3
"""
Scrape ecoregion fact sheets from coralsoftheworld.org.
Produces a DataFrame with: ecoregion name, total species number, QA flag, and page URL.
"""

# general
import argparse
import re
import time
from pathlib import Path

# web
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

# custom
from src import config

# Individual fact sheet pages use this base + ecoregion slug (e.g. <base_url>/red-sea-north-central/).
# The index at BASE_URL (no slug) returns 500; only direct slug URLs work.
BASE_URL = "https://www.coralsoftheworld.org/ecoregion_factsheets/ecoregion_factsheet_general_info/"
PARENT_URL = "https://www.coralsoftheworld.org/ecoregion_factsheets/"
REQUEST_DELAY = 1.0  # wait time (s) between requests to be polite
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}  # add user agent header to avoid being blocked by the server


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
    # split before capital when previous character is lowercase (no space between names on site)
    parts = re.split(r"(?<=[a-z0-9\)])(?=[A-Z])", text)
    # filter to plausible ecoregion names: contain space or comma, or known short names
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
    (e.g. red-sea-north-central), not numeric IDs. Option values on the parent
    page are numeric IDs that do not correspond to direct URLs.
    Strategy: collect ecoregion names from <option> text and/or page text, slugify, build BASE_URL + slug.
    """
    base_path = urlparse(BASE_URL).path.rstrip("/") + "/"
    parent_path = urlparse(PARENT_URL).path.rstrip("/") + "/"
    seen_slugs: set[str] = set()
    urls: list[str] = []

    # 1. links from <a href> that point to name-based slugs under base or parent
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

    # 2. ecoregion names from <option> text (dropdown labels), then slugify
    for opt in soup.find_all("option", value=True):
        name = opt.get_text(strip=True)
        if not name or name in ("", "Select saved ecoregion list"):
            continue
        slug = _slug_from_name(name)
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            urls.append(urljoin(BASE_URL, slug + "/"))

    # 3. option value: only use if it's a name-based slug (not numeric)
    for opt in soup.find_all("option", value=True):
        val = opt["value"].strip().strip("/")
        if not val:
            continue
        bare = val.split("/")[-1] if "/" in val else val
        if _is_name_based_slug(bare) and bare not in seen_slugs:
            seen_slugs.add(bare)
            urls.append(urljoin(BASE_URL, bare + "/"))

    # 4. fallback: parse ecoregion names from page text and slugify
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


def parse_species_number(text: str) -> int | None:
    """
    Extract integer from text like 'Total species number: 310'.
    Returns None if format is unexpected.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # Match "Total species number: 310" or similar
    match = re.search(r"Total\s+species\s+number\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: any number at end after colon
    match = re.search(r":\s*(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def scrape_ecoregion_page(url: str) -> tuple[str | None, int | None, int]:
    """
    Scrape one ecoregion page. Returns (name, species_number, qa).
    qa: 1 = expected format, 0 = problem/irregular.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if not resp.ok:
            return None, None, 0
    except Exception:
        return None, None, 0

    soup = BeautifulSoup(resp.text, "html.parser")

    # Ecoregion name: span.header-left-container > h3
    name = None
    left = soup.find("span", class_="header-left-container")
    if left:
        h3 = left.find("h3")
        if h3 and h3.get_text(strip=True):
            name = h3.get_text(strip=True)

    # Total species: span.header-right-container > span.info-container > h5
    species_number = None
    right = soup.find("span", class_="header-right-container")
    if right:
        info = right.find("span", class_="info-container")
        if info:
            h5 = info.find("h5")
            if h5:
                species_number = parse_species_number(h5.get_text())

    # QA: 1 if both name and species number found in expected format
    qa = 1 if (name is not None and species_number is not None) else 0

    return name, species_number, qa


def main():

    parser = argparse.ArgumentParser(
        description="Scrape Corals of the World ecoregion pages for biodiversity data"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=str(config.data_dir / "ecoregion_diversity"),
        help="Output directory",
    )

    args = parser.parse_args()
    overwrite = args.overwrite
    output_directory = Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / "ecoregions.csv"
    if out_path.exists() and not overwrite:
        print(f"Data already exists in {out_path}. Use --overwrite flag to overwrite.")
        return

    print("Fetching ecoregion list...")
    soup = get_index_soup()
    urls = discover_ecoregion_urls(soup)
    print(
        f"Found {len(urls)} ecoregion page{'s' if len(urls) > 1 else ''}. Saving to {out_path}."
    )

    rows = []
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description} [bold]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Scraping ecoregion pages ({len(urls)} page{'s' if len(urls) > 1 else ''})",
            total=len(urls),
        )
        for i, url in enumerate(urls, 1):
            progress.update(
                task,
                completed=i,
                description=f"[{i}/{len(urls)}] {Path(url).name}",
            )
            name, species_number, qa = scrape_ecoregion_page(url)
            rows.append(
                {
                    "ecoregion_name": name,
                    "total_species_number": species_number,
                    "QA": qa,
                    "url": url,
                }
            )
        time.sleep(REQUEST_DELAY)  # necessary to prevent rate limiting error

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} page{'s' if len(urls) > 1 else ''} to {out_path}")
    if (df["QA"] == 0).any():
        n_bad = (df["QA"] == 0).sum()
        print(f"QA flag: {n_bad} row(s) with possible formatting issues.")


if __name__ == "__main__":
    main()
