#!/usr/bin/env python3
"""Populate the SWE-agent pilot raw-row cache (one-shot).

Reads ``external_data/swe_agent/manifests/swe_agent_pilot_sample.csv``
(produced by B2), streams ``nebius/SWE-agent-trajectories`` (split=train)
once, and writes the matching raw rows to
``--cache-dir/<pilot_id>.json``. Each cached file is the canonical
byte source for `import_swe_agent_trace.py` (C3) — that importer is
deterministic and offline; THIS script is the only place where the
network is touched.

Each row is validated against the pilot CSV's ``instance_id`` and
``model_name`` so a silent shift in HF streaming order would fail
loudly rather than corrupt the pilot.

CLI
---
    python scripts/populate_swe_agent_pilot_cache.py \
      --sample-csv external_data/swe_agent/manifests/swe_agent_pilot_sample.csv \
      --cache-dir  external_data/swe_agent/pilot_cache/ \
      [--overwrite] [--max-extra-rows 1000] [--progress]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class PilotTarget:
    pilot_id: str
    instance_id: str
    model_name: str
    dataset_index: int


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate the SWE-agent pilot raw-row cache.")
    p.add_argument("--sample-csv", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch and overwrite cache files that already exist.",
    )
    p.add_argument(
        "--max-extra-rows",
        type=int,
        default=200,
        help="Stop streaming this many rows past the largest wanted "
        "dataset index (defensive cap; default 200).",
    )
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--source",
        default="nebius",
        choices=("nebius",),
        help="Upstream source. Only nebius is wired up.",
    )
    return p.parse_args(argv)


def _parse_dataset_index(value: str) -> Optional[int]:
    if not isinstance(value, str) or not value:
        return None
    tail = value.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None


def load_targets(sample_csv: Path) -> List[PilotTarget]:
    with sample_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out: List[PilotTarget] = []
        for r in reader:
            idx = _parse_dataset_index(r.get("raw_path_or_dataset_index", ""))
            if idx is None:
                raise ValueError(
                    f"pilot row {r.get('pilot_id')!r} has unparseable "
                    f"raw_path_or_dataset_index="
                    f"{r.get('raw_path_or_dataset_index')!r}"
                )
            out.append(
                PilotTarget(
                    pilot_id=r["pilot_id"],
                    instance_id=r["instance_id"],
                    model_name=r["model_name"],
                    dataset_index=idx,
                )
            )
    return out


def collect_matches(
    stream: Iterator[Tuple[int, Dict[str, Any]]],
    targets: List[PilotTarget],
    *,
    max_extra_rows: int = 200,
    progress_cb: Optional[callable] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Walk the stream, yielding (pilot_id -> raw_row) for matched indices.

    Returns (matches, errors). Each element of ``errors`` is a short
    human-readable string describing a mismatch or a not-found pilot.

    Validates ``instance_id`` and ``model_name`` against the pilot CSV
    when a match is hit; an upstream re-order would surface here.
    """
    by_index: Dict[int, PilotTarget] = {t.dataset_index: t for t in targets}
    if len(by_index) != len(targets):
        raise ValueError("duplicate dataset_index in pilot CSV")
    if not by_index:
        return {}, []
    max_wanted = max(by_index)
    cutoff = max_wanted + max_extra_rows

    matches: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    for hf_index, row in stream:
        if progress_cb is not None:
            progress_cb(hf_index)
        if hf_index in by_index:
            t = by_index[hf_index]
            row_iid = row.get("instance_id") if isinstance(row, dict) else None
            row_mn = row.get("model_name") if isinstance(row, dict) else None
            if row_iid != t.instance_id or row_mn != t.model_name:
                errors.append(
                    f"{t.pilot_id} at index {hf_index}: expected "
                    f"({t.instance_id!r}, {t.model_name!r}) but stream "
                    f"yielded ({row_iid!r}, {row_mn!r}); HF order may "
                    "have shifted"
                )
            else:
                matches[t.pilot_id] = row
        if hf_index >= cutoff:
            break

    found = set(matches.keys())
    for t in targets:
        if t.pilot_id not in found:
            already_errored = any(t.pilot_id in e for e in errors)
            if not already_errored:
                errors.append(
                    f"{t.pilot_id}: dataset_index {t.dataset_index} not "
                    "matched (stream ended early or row missing)"
                )

    return matches, errors


def _stream_nebius() -> Iterator[Tuple[int, Dict[str, Any]]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=True,
    )
    for hf_index, row in enumerate(ds):
        yield hf_index, row


def write_matches(
    matches: Dict[str, Dict[str, Any]],
    cache_dir: Path,
    *,
    overwrite: bool,
    skipped_existing: List[str],
) -> List[str]:
    """Write raw rows as <pilot_id>.json. Returns list of paths written."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for pilot_id, row in matches.items():
        path = cache_dir / f"{pilot_id}.json"
        if path.is_file() and not overwrite:
            skipped_existing.append(pilot_id)
            continue
        path.write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(pilot_id)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    targets_all = load_targets(args.sample_csv)

    if args.overwrite:
        wanted = targets_all
        already_have: List[str] = []
    else:
        already_have = [
            t.pilot_id
            for t in targets_all
            if (args.cache_dir / f"{t.pilot_id}.json").is_file()
        ]
        wanted = [
            t for t in targets_all if t.pilot_id not in set(already_have)
        ]
        if already_have:
            print(
                f"[populate_swe_agent_pilot_cache] cache hit for "
                f"{len(already_have)}/{len(targets_all)} pilots; will fetch "
                f"{len(wanted)}",
                file=sys.stderr,
            )

    if not wanted:
        print(
            "[populate_swe_agent_pilot_cache] cache complete; nothing to do.",
            file=sys.stderr,
        )
        return 0

    started = time.monotonic()

    def _progress(hf_index: int) -> None:
        if args.progress and (hf_index + 1) % 5000 == 0:
            elapsed = time.monotonic() - started
            rate = (hf_index + 1) / elapsed if elapsed > 0 else float("inf")
            print(
                f"[populate_swe_agent_pilot_cache] streamed {hf_index + 1} rows "
                f"({rate:.1f} rows/s)",
                file=sys.stderr,
                flush=True,
            )

    matches, errors = collect_matches(
        _stream_nebius(),
        wanted,
        max_extra_rows=args.max_extra_rows,
        progress_cb=_progress,
    )

    skipped: List[str] = []
    written = write_matches(
        matches, args.cache_dir, overwrite=args.overwrite, skipped_existing=skipped
    )

    for e in errors:
        print(f"[populate_swe_agent_pilot_cache] ERROR: {e}", file=sys.stderr)
    print(
        f"[populate_swe_agent_pilot_cache] wrote {len(written)} rows to "
        f"{args.cache_dir}; skipped {len(skipped)} pre-existing; "
        f"{len(errors)} errors",
        file=sys.stderr,
    )

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
