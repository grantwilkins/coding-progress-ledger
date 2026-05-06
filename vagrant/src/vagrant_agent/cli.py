"""Vagrant CLI entry points: vagrant-trace, vagrant-manifest, vagrant-bench."""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(prog="vagrant-bench", description="Run policies on a trace and emit results + plot.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--policies",
        default="request_level_no_reuse,request_level_with_site_cache,shared_state_aware",
        help="comma-separated list of policy names",
    )
    parser.add_argument("--tau", type=int, default=1, help="shared_state_aware threshold in tokens")
    parser.add_argument("--model", default="compact_kv")
    parser.add_argument("--model-config", type=Path,
                        default=Path("configs/model_profiles.yaml"))
    parser.add_argument("--sites-config", type=Path,
                        default=Path("configs/sites_2site.yaml"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    from .bench import run_bench
    from .plots import plot_duplication_factor
    from .policies import POLICIES

    requested = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in requested if p not in POLICIES]
    if unknown:
        parser.error(f"unknown policy/policies: {unknown}; known: {sorted(POLICIES)}")

    summary = run_bench(
        trace_path=args.trace,
        out_dir=args.out,
        policies=requested,
        model_path=args.model_config,
        sites_path=args.sites_config,
        model_name=args.model,
        tau=args.tau,
    )
    if not args.no_plot:
        plot_duplication_factor(summary, args.out / "plots" / "duplication_factor.png")
    print(f"wrote {args.out}/results.csv, summary.json, state_materialization_breakdown.csv")
    for name, info in summary["policies"].items():
        print(f"  {name}: total_cost_s={info['total_cost_s']:.4f}, "
              f"dup_factor={info['cost_weighted_duplication_factor']:.4f}")
    return 0
