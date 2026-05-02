#!/usr/bin/env python3
"""HP5 sampler: balance N pilots across categories x configs from a combined inventory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_COLUMNS: Tuple[str, ...] = (
    "pilot_id",
    "source_id",
    "instance_id",
    "model_name",
    "category",
    "subcategory",
    "trajectory_length",
    "raw_path_or_dataset_index",
    "selection_reason",
)
DEFAULT_CATEGORIES = ("Terminal & Coding", "Repository Tasks", "File Operations")
DEFAULT_CONFIGS = ("kimi", "glm-5.1")
MIN_CONV_LEN = 6


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HP5 balanced sampler.")
    p.add_argument("--inventory-csv", required=True, type=Path, action="append",
                   help="Inventory CSV (may be repeated to combine).")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--n-pilots", type=int, default=30)
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--min-len", type=int, default=MIN_CONV_LEN)
    p.add_argument("--pilot-prefix", default="hermes_pilot_h5_")
    return p.parse_args(argv)


def _parse_int(value: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _load(paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8", newline="") as fh:
            out.extend(csv.DictReader(fh))
    return out


def _filter(rows: List[Dict[str, Any]], cats: List[str], cfgs: List[str], min_len: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    funnel: Dict[str, int] = {"total": len(rows)}
    s1 = [r for r in rows if r.get("category") in cats]
    funnel["after_category"] = len(s1)
    s2 = [r for r in s1 if r.get("model_name") in cfgs]
    funnel["after_config"] = len(s2)
    s3 = [r for r in s2 if (_parse_int(r.get("trajectory_length", "")) or 0) >= min_len]
    funnel["after_min_len"] = len(s3)
    s4 = [r for r in s3 if r.get("trajectory_available") == "True"]
    funnel["after_traj_available"] = len(s4)
    seen: set = set()
    s5: List[Dict[str, Any]] = []
    for r in s4:
        key = (r.get("model_name", ""), r.get("instance_id", ""))
        if key[1] and key not in seen:
            seen.add(key)
            s5.append(r)
    funnel["after_dedupe"] = len(s5)
    return s5, funnel


def _balance(rows: List[Dict[str, Any]], n: int, cats: List[str], cfgs: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {(c, m): [] for c in cats for m in cfgs}
    for r in rows:
        key = (r.get("category", ""), r.get("model_name", ""))
        if key in buckets:
            buckets[key].append(r)
    for k in buckets:
        buckets[k].sort(key=lambda r: r.get("instance_id", ""))
    keys = sorted(buckets.keys())
    picks: List[Dict[str, Any]] = []
    cursors = {k: 0 for k in keys}
    while len(picks) < n:
        progressed = False
        for k in keys:
            if len(picks) >= n:
                break
            i = cursors[k]
            if i < len(buckets[k]):
                picks.append(buckets[k][i])
                cursors[k] = i + 1
                progressed = True
        if not progressed:
            break
    return picks


def _project(row: Dict[str, Any], pilot_id: str) -> Dict[str, Any]:
    return {
        "pilot_id": pilot_id,
        "source_id": row.get("source_id", ""),
        "instance_id": row.get("instance_id", ""),
        "model_name": row.get("model_name", ""),
        "category": row.get("category", ""),
        "subcategory": row.get("subcategory", ""),
        "trajectory_length": row.get("trajectory_length", ""),
        "raw_path_or_dataset_index": row.get("raw_path_or_dataset_index", ""),
        "selection_reason": f"hp5_balanced::{row.get('category','')}::{row.get('model_name','')}",
    }


def _write(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(list(CSV_COLUMNS))
        for r in rows:
            w.writerow([str(r[c]) if r[c] is not None else "" for c in CSV_COLUMNS])


def select_pilots_v2(rows: List[Dict[str, Any]], n: int, cats: List[str], cfgs: List[str], min_len: int, prefix: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    filtered, funnel = _filter(rows, cats, cfgs, min_len)
    picks = _balance(filtered, n, cats, cfgs)
    out = [_project(r, f"{prefix}{i+1:03d}") for i, r in enumerate(picks)]
    return out, funnel


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rows = _load(args.inventory_csv)
    out_rows, funnel = select_pilots_v2(rows, args.n_pilots, cats, cfgs, args.min_len, args.pilot_prefix)
    if not out_rows:
        print("[sample_hermes_pilot_v2] FATAL: no rows survived filters", file=sys.stderr)
        return 2
    _write(out_rows, args.out_csv)
    for k, v in funnel.items():
        print(f"[sample_hermes_pilot_v2] {k}: {v}", file=sys.stderr)
    print(f"[sample_hermes_pilot_v2] selected {len(out_rows)} pilots -> {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
