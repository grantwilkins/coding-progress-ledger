"""Vagrant CLI entry points: vagrant-trace, vagrant-manifest, vagrant-bench."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ledger_progress import from_jsonl

from .manifest import build_manifest
from .manifest_io import write_json


def manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vagrant-manifest", description="Build a Serving Group Manifest from a vagrant trace.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="replay a trace and emit a manifest JSON")
    build.add_argument("--trace", required=True, type=Path, help="path to trace JSONL")
    build.add_argument("--out", required=True, type=Path, help="path to manifest JSON output")
    args = parser.parse_args(argv)
    ledger = from_jsonl(str(args.trace))
    manifest = build_manifest(ledger)
    write_json(manifest, args.out)
    print(
        f"wrote {args.out}: {len(manifest.nodes)} nodes, "
        f"{len(manifest.state_objects)} state objects, {len(manifest.edges)} edges"
    )
    return 0


def trace_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vagrant-trace", description="Inspect a vagrant trace.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    summarize = sub.add_parser("summarize", help="print event-type counts for a trace")
    summarize.add_argument("trace", type=Path)
    args = parser.parse_args(argv)

    ledger = from_jsonl(str(args.trace))
    counts: dict[str, int] = {}
    for event in ledger.events:
        key = event.event_type.value if hasattr(event.event_type, "value") else event.event_type
        counts[key] = counts.get(key, 0) + 1
    print(f"trace: {args.trace}")
    print(f"root_task: {ledger.root_task}")
    print(f"subtasks: {len(ledger.subtasks)}")
    print(f"events: {len(ledger.events)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    return 0


def bench_main(argv: list[str] | None = None) -> int:
    print("vagrant-bench: not implemented yet (Workstream E).", file=sys.stderr)
    return 2
