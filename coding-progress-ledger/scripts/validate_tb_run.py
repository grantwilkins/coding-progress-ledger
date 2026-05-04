#!/usr/bin/env python3
"""Post-run validator for a TB-live subagent run.

Steps:
  1. Run the sidecar over <run_dir>/events.jsonl → ledger.jsonl + companions.
  2. Run the task verifier.sh against <run_dir>/repo, capture stdout/err.
  3. git diff <run_dir>/repo → <run_dir>/final_diff.patch (best-effort).
  4. ledger-run check-run <run_dir>.
  5. Write <run_dir>/live_instrumentation.json.

Usage:
    validate_tb_run.py <task_id> [--runs-dir runs/tb_live] [--tasks-dir tasks/tb_live]
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task_id")
    p.add_argument("--runs-dir", default="runs/tb_live")
    p.add_argument("--tasks-dir", default="tasks/tb_live")
    args = p.parse_args()

    run_dir = (REPO_ROOT / args.runs_dir / args.task_id).resolve()
    repo_dir = run_dir / "repo"
    task_dir = (REPO_ROOT / args.tasks_dir / args.task_id).resolve()
    events = run_dir / "events.jsonl"
    if not events.exists():
        print(f"missing {events}", file=sys.stderr)
        return 2

    subprocess.run(
        [sys.executable, "-m", "ledger_progress.sidecar",
         "--run-dir", str(run_dir), "--input-file", str(events),
         "--root-task", args.task_id],
        check=True, cwd=REPO_ROOT,
    )

    test_output = run_dir / "test_output.txt"
    v = subprocess.run(
        ["bash", str(task_dir / "verifier.sh"), str(repo_dir)],
        capture_output=True, text=True,
    )
    test_output.write_text(v.stdout + "\n--- stderr ---\n" + v.stderr)
    verifier_pass = v.returncode == 0

    diff = subprocess.run(
        ["git", "diff", "--no-index", "--", "/dev/null", str(repo_dir)],
        capture_output=True, text=True,
    )
    (run_dir / "final_diff.patch").write_text(diff.stdout)

    subprocess.run(
        [sys.executable, "-m", "ledger_progress.run_manager", "check-run", str(run_dir)],
        check=True, cwd=REPO_ROOT,
    )

    ledger_lines = (run_dir / "ledger.jsonl").read_text().splitlines()
    timestamps = [json.loads(line).get("timestamp") for line in ledger_lines if line.strip()]
    timestamps = [t for t in timestamps if t]
    span = 0.0
    if len(timestamps) >= 2:
        t0 = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        tn = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
        span = (tn - t0).total_seconds()

    (run_dir / "live_instrumentation.json").write_text(json.dumps({
        "task_id": args.task_id,
        "timestamp_source": "wallclock",
        "verifier_exit_code": v.returncode,
        "verifier_pass": verifier_pass,
        "ledger_event_count": len(ledger_lines),
        "ledger_event_timestamp_count": len(timestamps),
        "timestamp_span_seconds": span,
        "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }, indent=2, sort_keys=True) + "\n")

    print(f"verifier_pass={verifier_pass} events={len(ledger_lines)} span={span:.1f}s")
    return 0 if verifier_pass else 1


if __name__ == "__main__":
    sys.exit(main())
