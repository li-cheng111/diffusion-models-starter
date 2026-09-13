"""Download a reproducible, public-domain Van Gogh style dataset.

The script queries Wikimedia Commons for files in the Van Gogh paintings
category, keeps only public-domain raster images, downloads 20 images, and
writes a manifest containing the page URL, license and SHA256.  The images
belong in ``.local/datasets`` (ignored by Git); only the manifest is suitable
for committing as an experiment record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    from .experiment_utils import sha256_file, write_json
except ImportError:  # direct script execution from the project directory
    from experiment_utils import sha256_file, write_json


API_URL = "https://commons.wikimedia.org/w/api.php"
# The title-indexed category is the stable Commons source referenced by the
# assignment and contains the public-domain files needed for the 20-image set.
CATEGORY = "Category:Paintings_by_Vincent_van_Gogh_by_title"


def _api(params: dict[str, str]) -> dict:
    url = API_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "diffusion-project3-coursework/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            # Commons may rate-limit a burst of category requests. Honour
            # Retry-After when present, with a bounded exponential fallback.
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 2.0 ** attempt
            except (TypeError, ValueError):
                delay = 2.0 ** attempt
            time.sleep(delay)


def _members(category: str, member_type: str) -> list[dict]:
    """Return one category's members, following the API continuation token."""
    results: list[dict] = []
    cont: dict[str, str] = {}
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": category, "cmtype": member_type, "cmlimit": "500",
        }
        params.update(cont)
        payload = _api(params)
        results.extend(payload.get("query", {}).get("categorymembers", []))
        cont = payload.get("continue", {})
        if not cont:
            return results


def discover(limit: int) -> list[dict]:
    results: list[dict] = []
    seen_files: set[str] = set()
    # Commons organises the title-indexed category into subcategories. Walk
    # those categories breadth-first, while retaining a hard cap for safety.
    queue: list[tuple[str, int]] = [(CATEGORY, 0)]
    seen_categories: set[str] = set()
    while queue and len(results) < limit and len(seen_categories) < 50:
        category, depth = queue.pop(0)
        if not category or category in seen_categories:
            continue
        seen_categories.add(category)
        if depth < 3:
            for member in _members(category, "subcat"):
                queue.append((member.get("title", ""), depth + 1))
        cont: dict[str, str] = {}
        while len(results) < limit and len(results) < 200:
            params = {
                "action": "query",
                "format": "json",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": "50",
                "prop": "imageinfo|info",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "1024",
            }
            params.update(cont)
            payload = _api(params)
            for page in payload.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                metadata = info.get("extmetadata") or {}
                license_name = str((metadata.get("LicenseShortName") or {}).get("value", ""))
                mime = str(info.get("mime", ""))
                if not mime.startswith("image/") or "public domain" not in license_name.lower():
                    continue
                title = page.get("title", "")
                if title in seen_files:
                    continue
                seen_files.add(title)
                results.append(
                    {
                        "title": title,
                        "page_url": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                        "license": license_name,
                        "source_url": info.get("thumburl") or info.get("url"),
                        "mime": mime,
                    }
                )
                if len(results) >= limit:
                    break
            cont = payload.get("continue", {})
            if not cont:
                break
            time.sleep(0.2)
    if len(results) < limit:
        raise RuntimeError(f"only found {len(results)} public-domain images in {CATEGORY}")
    return results[:limit]


def download(entries: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for index, entry in enumerate(entries):
        suffix = ".jpg" if "jpeg" in entry["mime"] or "jpg" in entry["mime"] else ".png"
        path = output_dir / f"vangogh_{index:02d}{suffix}"
        request = Request(entry["source_url"], headers={"User-Agent": "diffusion-project3-coursework/1.0"})
        with urlopen(request, timeout=60) as response, path.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        item = dict(entry)
        item["local_file"] = str(path)
        item["sha256"] = sha256_file(path)
        item["bytes"] = path.stat().st_size
        manifest.append(item)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".local/datasets/project3_vangogh")
    parser.add_argument("--manifest_output", default="projects/project3_stable_diffusion/outputs/lora/vangogh_manifest.json")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    entries = discover(args.count)
    manifest = download(entries, Path(args.output_dir))
    write_json(args.manifest_output, {"category": CATEGORY, "count": len(manifest), "images": manifest})
    print(f"[done] downloaded {len(manifest)} images to {args.output_dir}")
    print(f"[saved] manifest -> {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
