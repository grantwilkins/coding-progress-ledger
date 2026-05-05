"""tb_live_v2 batch CLI.

Two subcommands. The Agent-tool subagent invocation happens BETWEEN
them (not inside this script). Typical orchestration:

    # 1. set up the workspace and print the prompt the orchestrator
    #    should hand to the Agent tool. Outputs JSON: {run_id, run_dir,
    #    workspace, prompt} to stdout.
    uv run python scripts/run_tb_live_v2_batch.py prepare \\
        --task-dir tasks/tb_live_v2/low_progress_success_01_oneline_fix \\
        --arm A

    # 2. orchestrator spawns Agent tool with the prompt, waits for the
    #    subagent to finish, then:
    uv run python scripts/run_tb_live_v2_batch.py finalize \\
        --task-dir tasks/tb_live_v2/low_progress_success_01_oneline_fix \\
        --run-dir runs/tb_live_v2/<run_id>

A `run` subcommand is provided for non-Claude-Code orchestrators that
already drive the subagent themselves; it expects `transcript.jsonl`
to already exist in the run dir before being called and is
equivalent to finalize.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from coding_estimator.runner.driver import (
    ARM_BUDGETS,
    ARM_MODELS,
    finalize,
    prepare,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO / "runs" / "tb_live_v2"


def _cmd_prepare(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    prep = prepare(Path(args.task_dir), args.arm, runs_root=runs_root)
    payload = {
        "run_id": prep.run_id,
        "task_id": prep.task_id,
        "arm": prep.arm,
        "workspace": str(prep.workspace),
        "run_dir": str(prep.run_dir),
        "started_at": prep.started_at,
        "model_name": ARM_MODELS[prep.arm],
        "budget_lines": ARM_BUDGETS[prep.arm],
        "prompt": prep.prompt,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    task_dir = Path(args.task_dir)
    prompt_path = run_dir / "prompt.txt"
    workspace_path = run_dir / "workspace_path.txt"
    if not prompt_path.is_file() or not workspace_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} missing prompt.txt / workspace_path.txt — "
            f"prepare must run first"
        )
    from coding_estimator.runner.driver import RunPrep
    prep = RunPrep(
        run_id=run_dir.name,
        task_id=task_dir.name,
        arm=_arm_from_run_id(run_dir.name),
        workspace=Path(workspace_path.read_text().strip()),
        run_dir=run_dir,
        prompt=prompt_path.read_text(),
        started_at=_started_at_from_run_dir(run_dir),
    )
    result = finalize(prep, task_dir, skip_sidecar=args.skip_sidecar)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.final_success is not None else 1


def _arm_from_run_id(run_id: str) -> str:
    parts = run_id.split("__")
    for p in parts:
        if p.startswith("arm") and p[3:] in ARM_BUDGETS:
            return p[3:]
    raise ValueError(f"cannot infer arm from run_id: {run_id}")


def _started_at_from_run_dir(run_dir: Path) -> str:
    manifest = run_dir / "run_manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text()).get("start_time") or ""
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tb_live_v2 batch CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="set up workspace + render prompt")
    p_prep.add_argument("--task-dir", required=True)
    p_prep.add_argument("--arm", required=True, choices=sorted(ARM_BUDGETS))
    p_prep.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p_prep.set_defaults(func=_cmd_prepare)

    p_fin = sub.add_parser("finalize", help="convert transcript -> ledger -> verifier -> manifest")
    p_fin.add_argument("--task-dir", required=True)
    p_fin.add_argument("--run-dir", required=True)
    p_fin.add_argument("--skip-sidecar", action="store_true",
                       help="skip ledger_progress.sidecar replay (for tests)")
    p_fin.set_defaults(func=_cmd_finalize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
