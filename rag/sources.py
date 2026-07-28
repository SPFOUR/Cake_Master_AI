"""
Source manifest: loads citation info for each file from data/sources.csv.

sources.csv columns: filename, url, citation
"""

import csv
import os

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sources.csv")


def _load_source_map(path: str = _CSV_PATH) -> dict:
    source_map = {}
    if not os.path.exists(path):
        return source_map

    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(
            "sources.csv could not be read as UTF-8 or cp1252. "
            "Try re-saving it as 'CSV UTF-8' from Excel."
        )

    for row in rows:
        filename = row["filename"].strip()
        source_map[filename] = {
            "url": row["url"].strip() or None,
            "citation": row["citation"].strip() or "Unknown source",
        }
    return source_map


SOURCE_MAP = _load_source_map()