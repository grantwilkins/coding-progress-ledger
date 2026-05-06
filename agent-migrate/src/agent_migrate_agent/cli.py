"""Vagrant CLI entry points: agent_migrate_trace, agent_migrate_manifest, agent_migrate_bench."""
from __future__ import annotations

import argparse
from pathlib import Path

from ledger_progress import from_jsonl

from .manifest import build_manifest
from .manifest_io import write_json


def manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_migrate_manifest", description="Build a Serving Group Manifest from a agent_migrate trace.")
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
    parser = argparse.ArgumentParser(prog="agent_migrate_trace", description="Inspect a agent_migrate trace.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    summarize = sub.add_parser("summarize", help="print event_type counts for a trace")
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


def sensitivity_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_migrate_sensitivity",
        description="Sweep over (kv_bytes_per_token, link_bps) and report whether "
                    "the policy gap survives. Use to defend the headline against "
                    "load_bearing_constants critique.",
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="compact_kv")
    parser.add_argument("--model_config", type=Path,
                        default=Path("configs/model_profiles.yaml"))
    parser.add_argument("--sites_config", type=Path,
                        default=Path("configs/sites_2site.yaml"))
    parser.add_argument(
        "--kv_bytes",
        default="10000,70656,327680",
        help="comma_separated kv_bytes_per_token grid; defaults bracket "
             "frontier_v4_fp8 / DeepSeek_V3 MLA / Llama_3-70B FP16",
    )
    parser.add_argument(
        "--link_bps",
        default="5e9,25e9,100e9,400e9",
        help="comma_separated link bandwidth (bps) grid; defaults bracket "
             "single_flow inter_region / aggregate cross_region / 100 GbE / "
             "RDMA_class",
    )
    parser.add_argument(
        "--policies",
        default="request_level_no_reuse,request_level_with_site_cache,shared_state_aware",
        help="comma_separated list of policies to evaluate at each grid point",
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--reference_policy", default="request_level_with_site_cache")
    parser.add_argument("--challenger_policy", default="shared_state_aware")
    args = parser.parse_args(argv)

    from .policies import POLICIES
    from .sensitivity import gap_survival_rate, run_sweep

    kv_bytes_grid = [int(float(x)) for x in args.kv_bytes.split(",") if x.strip()]
    link_bps_grid = [float(x) for x in args.link_bps.split(",") if x.strip()]
    requested = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in requested if p not in POLICIES]
    if unknown:
        parser.error(f"unknown policy/policies: {unknown}; known: {sorted(POLICIES)}")

    rows = run_sweep(
        trace_path=args.trace,
        out_dir=args.out,
        model_path=args.model_config,
        sites_path=args.sites_config,
        model_name=args.model,
        kv_bytes_grid=kv_bytes_grid,
        link_bps_grid=link_bps_grid,
        policies=requested,
        tau=args.tau,
        reference_policy=args.reference_policy,
        challenger_policy=args.challenger_policy,
    )
    survival = gap_survival_rate(rows)
    print(f"wrote {args.out}/sensitivity.csv ({len(rows)} rows)")
    print(f"gap survival rate: {survival:.0%} of {len(kv_bytes_grid) * len(link_bps_grid)} grid points "
          f"({args.challenger_policy} < {args.reference_policy})")
    return 0


def bench_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_migrate_bench", description="Run policies on a trace and emit results + plot.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--policies",
        default="request_level_no_reuse,request_level_with_site_cache,shared_state_aware",
        help="comma_separated list of policy names",
    )
    parser.add_argument("--tau", type=int, default=1, help="shared_state_aware threshold in tokens")
    parser.add_argument("--model", default="compact_kv")
    parser.add_argument("--model_config", type=Path,
                        default=Path("configs/model_profiles.yaml"))
    parser.add_argument("--sites_config", type=Path,
                        default=Path("configs/sites_2site.yaml"))
    parser.add_argument("--no_plot", action="store_true")
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
