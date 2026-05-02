#!/usr/bin/env python3
"""Hermes pilot sampler (HP2). Implements I1-I5 from PILOT_SAMPLING_POLICY.md."""

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
TARGET_CATEGORY = "Terminal & Coding"
TARGET_CONFIG = "kimi"
MIN_CONV_LEN = 6


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample Hermes pilot from inventory.")
    p.add_argument("--inventory-csv", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--n-pilots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _parse_bool(value: str) -> Optional[bool]:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_inventory(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _apply_filters(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    funnel: Dict[str, int] = {"total": len(rows)}
    s1 = [r for r in rows if r.get("category") == TARGET_CATEGORY]
    funnel["after_I1_category"] = len(s1)
    s2 = [r for r in s1 if r.get("model_name") == TARGET_CONFIG]
    funnel["after_I2_config"] = len(s2)
    s3: List[Dict[str, Any]] = []
    for r in s2:
        n = _parse_int(r.get("trajectory_length", ""))
        if n is not None and n >= MIN_CONV_LEN:
            s3.append(r)
    funnel["after_I3_min_len"] = len(s3)
    s4 = [r for r in s3 if _parse_bool(r.get("trajectory_available", "")) is True]
    funnel["after_I4_traj_available"] = len(s4)
    seen: set = set()
    s5: List[Dict[str, Any]] = []
    for r in s4:
        iid = r.get("instance_id", "")
        if iid and iid not in seen:
            seen.add(iid)
            s5.append(r)
    funnel["after_I5_dedupe"] = len(s5)
    return s5, funnel


def _select(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: r.get("instance_id", ""))
    return sorted_rows[:n]


def _format_pilot_id(idx_1based: int) -> str:
    return f"hermes_pilot_{idx_1based:02d}"


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value)


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
        "selection_reason": f"primary_terminal_coding_{TARGET_CONFIG}",
    }


def _write(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(list(CSV_COLUMNS))
        for r in rows:
            w.writerow([_format_cell(r[c]) for c in CSV_COLUMNS])


def select_pilots(rows: List[Dict[str, Any]], n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    filtered, funnel = _apply_filters(rows)
    picks = _select(filtered, n, seed)
    out = [_project(r, _format_pilot_id(i + 1)) for i, r in enumerate(picks)]
    return out, funnel


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rows = _load_inventory(args.inventory_csv)
    out_rows, funnel = select_pilots(rows, args.n_pilots, args.seed)
    if not out_rows:
        print("[sample_hermes_pilot] FATAL: no rows survived filters", file=sys.stderr)
        return 2
    _write(out_rows, args.out_csv)
    for k, v in funnel.items():
        print(f"[sample_hermes_pilot] {k}: {v}", file=sys.stderr)
    print(f"[sample_hermes_pilot] selected {len(out_rows)} pilots -> {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
