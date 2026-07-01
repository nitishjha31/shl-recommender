"""
Re-scrapes the SHL Individual Test Solutions catalog and overwrites data/catalog.json.

Why this file exists
---------------------
The bundled data/catalog.json is a curated *seed* set (~40 real assessments) built
from search snippets, not a full crawl of every page in the live catalog. Run this
script from a machine with normal internet access to pull the current, complete
catalog before you submit -- it is the difference between "works on the 10 public
traces" and "works on the holdout set too".

Usage
-----
    pip install requests beautifulsoup4
    python data/scrape_catalog.py

The SHL catalog page is paginated and split into two tabs: "Individual Test
Solutions" (type=1) and "Pre-packaged Job Solutions" (type=2). We only want
type=1, per the assignment scope.
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LIST_URL = BASE + "/solutions/products/product-catalog/"
OUT_PATH = Path(__file__).parent / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.shl.com/",
    "Connection": "keep-alive",
}

_session = requests.Session()
_session.headers.update(HEADERS)

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch(url: str, params: dict | None = None) -> BeautifulSoup:
    resp = _session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def warm_up_session():
    """Visit the homepage first so the session picks up any cookies the site sets
    before it'll serve the catalog pages -- helps get past basic bot-detection."""
    try:
        _session.get(BASE + "/", timeout=30)
        time.sleep(1)
    except Exception as exc:
        print(f"  (warm-up request failed, continuing anyway: {exc})")


def list_individual_test_solutions() -> list[str]:
    """Walk the paginated 'Individual Test Solutions' table and collect detail-page URLs."""
    urls: list[str] = []
    start = 0
    page_size = 12  # SHL's listing page paginates ~12 rows at a time; adjust if it changes
    while True:
        soup = fetch(LIST_URL, params={"start": start, "type": 1})
        rows = soup.select("table tr a[href*='/product-catalog/view/']")
        if not rows:
            break
        new = 0
        for a in rows:
            href = a.get("href")
            full = href if href.startswith("http") else BASE + href
            if full not in urls:
                urls.append(full)
                new += 1
        if new == 0:
            break
        start += page_size
        time.sleep(0.5)  # be polite
    return urls


def parse_detail_page(url: str) -> dict:
    soup = fetch(url)
    name = soup.select_one("h1")
    name = name.get_text(strip=True) if name else url.rstrip("/").split("/")[-1]

    description = ""
    desc_el = soup.select_one("div.product-description, div.description, main p")
    if desc_el:
        description = desc_el.get_text(" ", strip=True)

    # Job level chips / labels
    job_levels = [el.get_text(strip=True) for el in soup.select(".job-level, .product-level li")]

    # Test type letter badges (A/B/C/D/E/K/P/S)
    type_letters = [el.get_text(strip=True) for el in soup.select(".test-type, .product-type-badge")]
    test_type = "".join(sorted(set(t for t in type_letters if t in TEST_TYPE_MAP))) or "K"

    duration = None
    m = re.search(r"(\d+)\s*min", soup.get_text())
    if m:
        duration = int(m.group(1))

    remote = "remote testing" in soup.get_text(" ", strip=True).lower()

    return {
        "name": name,
        "url": url,
        "test_type": test_type,
        "job_levels": job_levels,
        "description": description,
        "keywords": [],  # fill in manually or derive via a keyword-extraction pass
        "duration_minutes": duration,
        "remote_testing": remote,
    }


def main():
    print("Warming up session ...")
    warm_up_session()
    print("Listing Individual Test Solutions ...")
    urls = list_individual_test_solutions()
    print(f"Found {len(urls)} candidate URLs")

    catalog = []
    for i, url in enumerate(urls, 1):
        try:
            item = parse_detail_page(url)
            catalog.append(item)
            print(f"[{i}/{len(urls)}] {item['name']}")
        except Exception as exc:  # keep going on individual page failures
            print(f"  !! failed to parse {url}: {exc}")
        time.sleep(0.3)

    OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"Wrote {len(catalog)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()