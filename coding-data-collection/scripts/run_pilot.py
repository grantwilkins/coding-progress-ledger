from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.pilot_plan import (
    DEFAULT_TASK_ROOTS,
    build_pilot_plan,
    execute_plan,
    parse_arm,
    write_plan,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight and optionally run a Terminal-Bench pilot plan.")
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("manifests/pilots/terminal_bench_candidate_scores.csv"),
    )
    parser.add_argument("--task-root", action="append", type=Path, default=list(DEFAULT_TASK_ROOTS))
    parser.add_argument("--run-root", type=Path, default=Path("runs/terminal_bench_pilot"))
    parser.add_argument("--out", type=Path, default=Path("manifests/pilots/terminal_bench_pilot_plan.json"))
    parser.add_argument("--arm", action="append", default=[], help="Pilot arm as name=agent-command; repeat per arm.")
    parser.add_argument(
        "--network-exception",
        action="append",
        default=[],
        help="Verifier network exception as task-id=reason; repeat per task.",
    )
    parser.add_argument("--expected-tasks", type=int, default=12)
    parser.add_argument("--expected-arms", type=int, default=2)
    parser.add_argument("--allow-harbor", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    try:
        arms = [parse_arm(text) for text in args.arm]
        network_exceptions = _parse_mapping(args.network_exception, "--network-exception")
    except ValueError as exc:
        parser.error(str(exc))

    plan = build_pilot_plan(
        candidate_scores_path=args.candidate_scores,
        task_roots=args.task_root,
        run_root=args.run_root,
        arms=arms,
        expected_tasks=args.expected_tasks,
        expected_arms=args.expected_arms,
        allow_harbor=args.allow_harbor,
        network_exceptions=network_exceptions,
    )
    write_plan(args.out, plan)
    print(f"wrote {args.out}")

    if not plan["passed"]:
        sys.stderr.write("Pilot preflight failed; refusing to launch collection.\n")
        for blocker in plan["blockers"]:
            sys.stderr.write(f"- {blocker}\n")
        return 2
    if args.execute:
        return execute_plan(plan)
    print("Pilot preflight passed. Re-run with --execute to launch collection.")
    return 0


def _parse_mapping(values: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        key, sep, text = value.partition("=")
        if not sep or not key.strip() or not text.strip():
            raise ValueError(f"{flag} must have the form key=value")
        out[key.strip()] = text.strip()
    return out


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
