# -*- coding: utf-8 -*-
"""
Fetch and save Danish seed datasets from HuggingFace for use with
da_synthetic_data_generation.py.

Supported sources:
  - wikipedia          oliverkinch/danish_wikipedia
  - danmarks_statistik oliverkinch/danmarks-statistik
  - dynaword           danish-foundation-models/danish-dynaword  (7 subsets)
  - tidsskrift         oliverkinch/tidsskrift-dk

Each source is saved to Data/<source>.json as a list of normalised article dicts.

Usage:
    python fetch_seed_data.py                          # all sources
    python fetch_seed_data.py --sources wikipedia,dynaword
    python fetch_seed_data.py --max-wikipedia 50000
"""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm


# ── Column mapping ────────────────────────────────────────────────────────────
# Maps output field name  →  HuggingFace column name (or None = no such field).
# Fields absent from the dataset are simply omitted from the saved record.

DATASET_CONFIGS = {
    "wikipedia": {
        "hf_name": "oliverkinch/danish_wikipedia",
        "hf_config": None,          # no config / subset name
        "split": "train",
        "subsets": None,            # not a multi-subset dataset
        "id_field": "url",          # used as article ID in the generation script
        "columns": {                # output_name → hf_column_name
            "url":   "url",
            "title": "title",
            "text":  "text",
        },
        "output_file": "Data/danish_wikipedia.json",
    },
    "danmarks_statistik": {
        "hf_name": "oliverkinch/danmarks-statistik",
        "hf_config": None,
        "split": "train",
        "subsets": None,
        "id_field": "url",
        "columns": {
            "url":          "url",
            "title":        "title",
            "text":         "text",
            "date":         "date",
            "content_type": "content_type",
            "series":       "series",
        },
        "output_file": "Data/danmarks_statistik.json",
    },
    "dynaword": {
        "hf_name": "danish-foundation-models/danish-dynaword",
        "hf_config": None,          # config is specified per-subset
        "split": "train",
        "subsets": None,            # resolved dynamically from HF configs
        "dynamic_subsets": True,
        "exclude_subsets": ["relig"],
        "id_field": "id",
        "columns": {
            "id":     "id",
            "text":   "text",
            "source": "source",
            "date":   "added",      # 'added' → 'date'
            "created": "created",
        },
        "output_file": "Data/dynaword.json",
    },
    "tidsskrift": {
        "hf_name": "oliverkinch/tidsskrift-dk",
        "hf_config": None,
        "split": "train",
        "subsets": None,
        "id_field": "doi",          # fall back to 'url' in generation script
        "columns": {
            "doi":     "doi",
            "url":     "url",
            "title":   "title",
            "text":    "text",
            "date":    "date",
            "journal": "journal",
            "authors": "authors",
        },
        "output_file": "Data/tidsskrift.json",
    },
}

MIN_TEXT_LENGTH = 300   # skip articles with fewer characters (press-release stubs, etc.)

# Substrings that signal a record is navigation/boilerplate rather than content.
# Checked case-insensitively against the article text.
BOILERPLATE_SIGNALS = [
    "accept cookies",
    "cookie policy",
    "vi bruger cookies",
    "javascript is required",
    "javascript er påkrævet",
    "enable javascript",
    "you need to enable javascript",
    "this site requires javascript",
    "403 forbidden",
    "404 not found",
    "access denied",
]


def is_boilerplate(text: str) -> bool:
    """Return True if the text looks like a cookie-consent or nav page."""
    lower = text.lower()
    return any(signal in lower for signal in BOILERPLATE_SIGNALS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_record(row, columns: dict) -> dict:
    """Extract and rename fields according to the column mapping."""
    record = {}
    for out_name, hf_name in columns.items():
        val = row.get(hf_name)
        if val is not None:
            # Coerce to plain Python types (HF datasets can return custom objects)
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (date, datetime)):
                val = val.isoformat()
            record[out_name] = val
    return record


def _parse_date_part(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def is_older_than_years(record: dict, years: int = 30) -> bool:
    """
    Return True if the newest known creation date in the record is older than
    `years` years from today.

    Records with missing/invalid `created` are kept (return False).
    """
    created = record.get("created")
    if not created or not isinstance(created, str):
        return False

    parts = [p for p in (part.strip() for part in created.split(",")) if p]
    parsed = [_parse_date_part(p) for p in parts]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return False

    newest_created = max(parsed)
    today = date.today()
    try:
        cutoff = today.replace(year=today.year - years)
    except ValueError:
        cutoff = today.replace(month=2, day=28, year=today.year - years)
    return newest_created < cutoff


def resolve_subsets(name: str, cfg: dict) -> list[str]:
    """Resolve subset list, using dynamic Hugging Face config names when needed."""
    if cfg.get("subsets"):
        return cfg["subsets"]

    all_configs = get_dataset_config_names(cfg["hf_name"])
    excluded = set(cfg.get("exclude_subsets", []))
    return [c for c in all_configs if c != "default" and c not in excluded]


def fetch_single_dataset(name: str, cfg: dict, max_samples: Optional[int], base_path: str) -> int:
    """
    Download one dataset (no subsets) and save to disk.
    Returns the number of records saved.
    """
    print(f"[{name}] Loading {cfg['hf_name']} …")
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])

    output_path = Path(base_path) / cfg["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    limit = max_samples or len(ds)

    for row in tqdm(ds, desc=f"  Processing {name}", total=min(limit, len(ds))):
        if len(records) >= limit:
            break
        record = normalise_record(row, cfg["columns"])
        text = record.get("text", "")
        if not text or len(text) < MIN_TEXT_LENGTH:
            continue
        if is_boilerplate(text):
            continue
        records.append(record)

    print(f"[{name}] Saving {len(records):,} articles to {output_path} …")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records)


def fetch_multi_subset_dataset(
    name: str, cfg: dict, max_samples: Optional[int], base_path: str,
    per_subset: Optional[int] = None,
) -> int:
    """
    Download a dataset that has multiple named configs/subsets, merge them
    with round-robin interleaving, and save to disk.

    Interleaving ensures that a small global limit (max_samples) gets
    representation from every subset rather than exhausting subset 1 first.

    per_subset: if set, take at most this many records from each subset.
    """
    output_path = Path(base_path) / cfg["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subsets = resolve_subsets(name, cfg)
    print(f"[{name}] Using {len(subsets)} subset(s): {', '.join(subsets)}")

    # Determine per-subset cap
    num_subsets = len(subsets)
    if per_subset is not None:
        subset_cap = per_subset
    elif max_samples is not None:
        # Divide evenly; each subset gets at most ceil(max_samples / num_subsets)
        subset_cap = -(-max_samples // num_subsets)  # ceiling division
    else:
        subset_cap = None  # no limit

    # Collect records per subset
    subset_buckets: list[list] = []

    for subset in subsets:
        print(f"[{name}] Loading subset '{subset}' from {cfg['hf_name']} …")
        try:
            ds = load_dataset(cfg["hf_name"], subset, split=cfg["split"])
        except Exception as e:
            print(f"[{name}] WARNING: could not load subset '{subset}': {e}")
            subset_buckets.append([])
            continue

        bucket: list = []
        for row in tqdm(ds, desc=f"  Processing {subset}"):
            if subset_cap is not None and len(bucket) >= subset_cap:
                break
            record = normalise_record(row, cfg["columns"])
            text = record.get("text", "")
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue
            if is_boilerplate(text):
                continue
            if name == "dynaword" and is_older_than_years(record, years=30):
                continue
            record.setdefault("source", subset)
            if "id" in record:
                record["id"] = str(record["id"])
            bucket.append(record)

        print(f"[{name}]   → {len(bucket):,} records from '{subset}'")
        subset_buckets.append(bucket)

    # Round-robin interleave across subsets
    records: list = []
    max_len = max((len(b) for b in subset_buckets), default=0)
    for i in range(max_len):
        for bucket in subset_buckets:
            if i < len(bucket):
                records.append(bucket[i])
        if max_samples and len(records) >= max_samples:
            records = records[:max_samples]
            break

    print(f"[{name}] Saving {len(records):,} total records to {output_path} …")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records)


def fetch_source(
    name: str,
    max_samples: Optional[int] = None,
    base_path: str = "./",
    per_subset: Optional[int] = None,
) -> int:
    """Fetch a single named source and return the number of saved records."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown source: '{name}'. Choose from: {list(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[name]

    if cfg.get("dynamic_subsets") or cfg.get("subsets"):
        return fetch_multi_subset_dataset(name, cfg, max_samples, base_path, per_subset=per_subset)
    else:
        return fetch_single_dataset(name, cfg, max_samples, base_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv):
    args = argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    # --sources wikipedia,dynaword,...
    sources = list(DATASET_CONFIGS.keys())
    if "--sources" in args:
        idx = args.index("--sources")
        sources = [s.strip() for s in args[idx + 1].split(",")]

    def _int_arg(flag):
        if flag in args:
            idx = args.index(flag)
            try:
                return int(args[idx + 1])
            except (IndexError, ValueError):
                print(f"Error: {flag} must be followed by an integer")
                sys.exit(1)
        return None

    limits = {
        "wikipedia":          _int_arg("--max-wikipedia"),
        "danmarks_statistik": _int_arg("--max-danmarks-statistik"),
        "dynaword":           _int_arg("--max-dynaword"),
        "tidsskrift":         _int_arg("--max-tidsskrift"),
    }

    per_subset = _int_arg("--per-subset")

    base_path = "./"
    if "--base-path" in args:
        idx = args.index("--base-path")
        base_path = args[idx + 1]

    return sources, limits, base_path, per_subset


def main():
    sources, limits, base_path, per_subset = parse_args(sys.argv)

    print("=" * 60)
    print("Danish Seed Data Fetcher")
    print(f"Sources: {sources}")
    print("=" * 60)

    totals = {}
    for name in sources:
        max_s = limits.get(name)
        print(f"\n{'─' * 40}")
        count = fetch_source(name, max_samples=max_s, base_path=base_path, per_subset=per_subset)
        totals[name] = count
        print(f"[{name}] Done — {count:,} records saved.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, count in totals.items():
        cfg = DATASET_CONFIGS[name]
        path = Path(base_path) / cfg["output_file"]
        print(f"  {name:30s} {count:>8,} records  →  {path}")
    print("\nAll done! Run da_synthetic_data_generation.py to generate pairs.")


if __name__ == "__main__":
    main()
