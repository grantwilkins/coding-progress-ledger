#!/usr/bin/env python3
"""Hermes raw inventory builder (HP2). Streams via HF datasets-server REST."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


CSV_COLUMNS: Tuple[str, ...] = (
    "source_id",
    "instance_id",
    "model_name",
    "trajectory_available",
    "trajectory_length",
    "final_success_available",
    "final_success",
    "patch_available",
    "eval_log_available",
    "category",
    "subcategory",
    "raw_path_or_dataset_index",
    "parse_status",
    "parse_error",
)

DATASET = "lambda/hermes-agent-reasoning-traces"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
CONFIGS = ("kimi", "glm-5.1")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a deterministic raw manifest of Hermes traces.")
    p.add_argument("--config", required=True, choices=("kimi", "glm-5.1", "both"))
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="If set, dump each fetched row JSON to <cache-dir>/<config>/<index>.json.")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def _sanitize(msg: str) -> str:
    return msg.replace("\n", " ").replace("\r", " ").replace(",", " ").strip()


def _fetch_page(config: str, offset: int, length: int) -> Dict[str, Any]:
    qs = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": config,
        "split": "train",
        "offset": offset,
        "length": length,
    })
    url = f"{ROWS_API}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-inventory/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_rows(config: str, max_rows: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    fetched = 0
    offset = 0
    while fetched < max_rows:
        length = min(PAGE_SIZE, max_rows - fetched)
        payload = _fetch_page(config, offset, length)
        rows = payload.get("rows") or []
        if not rows:
            return
        for entry in rows:
            row_index = entry.get("row_idx", offset)
            row = entry.get("row") or {}
            yield row_index, row
            fetched += 1
            if fetched >= max_rows:
                return
        offset += len(rows)


def _row_to_record(row: Dict[str, Any], config: str, row_index: int) -> Dict[str, Any]:
    instance_id = row.get("id") if isinstance(row.get("id"), str) else ""
    convs = row.get("conversations")
    if isinstance(convs, list):
        traj_len = len(convs)
        traj_avail = traj_len > 0
    else:
        traj_len = 0
        traj_avail = False
    category = row.get("category") if isinstance(row.get("category"), str) else ""
    subcategory = row.get("subcategory") if isinstance(row.get("subcategory"), str) else ""
    source_id = f"hermes:{config}:{instance_id}"
    raw_path = f"{DATASET}:{config}:train:{row_index}"
    return {
        "source_id": source_id,
        "instance_id": instance_id,
        "model_name": config,
        "trajectory_available": traj_avail,
        "trajectory_length": traj_len,
        "final_success_available": False,
        "final_success": "",
        "patch_available": False,
        "eval_log_available": False,
        "category": category,
        "subcategory": subcategory,
        "raw_path_or_dataset_index": raw_path,
        "parse_status": "ok",
        "parse_error": "",
    }


def _format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value)


def _write_csv(records: List[Dict[str, Any]], out: Path) -> None:
    sorted_recs = sorted(records, key=lambda r: (r["model_name"], r["instance_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(list(CSV_COLUMNS))
        for r in sorted_recs:
            w.writerow([_format_cell(r[c]) for c in CSV_COLUMNS])


def _collect(config: str, max_rows: int, cache_dir: Optional[Path], progress: bool) -> List[Dict[str, Any]]:
    if cache_dir is not None:
        (cache_dir / config).mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    started = time.monotonic()
    for n, (row_index, row) in enumerate(stream_rows(config, max_rows), start=1):
        rec = _row_to_record(row, config, row_index)
        records.append(rec)
        if cache_dir is not None and rec["instance_id"]:
            (cache_dir / config / f"{row_index}.json").write_text(
                json.dumps(row, ensure_ascii=False), encoding="utf-8"
            )
        if progress and n % 50 == 0:
            elapsed = time.monotonic() - started
            print(f"[hermes_inventory] {config}: streamed {n} rows in {elapsed:.1f}s",
                  file=sys.stderr, flush=True)
    return records


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    configs = CONFIGS if args.config == "both" else (args.config,)
    all_records: List[Dict[str, Any]] = []
    for cfg in configs:
        recs = _collect(cfg, args.max_rows, args.cache_dir, args.progress)
        all_records.extend(recs)
        print(f"[hermes_inventory] {cfg}: collected {len(recs)} records", file=sys.stderr)
    _write_csv(all_records, args.out_csv)
    print(f"[hermes_inventory] wrote manifest -> {args.out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
