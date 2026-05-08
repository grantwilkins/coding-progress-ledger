from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.artifacts import read_json, write_run_manifest
from coding_data_collection.ledger import replay_sidecar, transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events, read_jsonl, write_jsonl
from coding_data_collection.protocol import RunStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize a protocol-shaped run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger-root", default="../coding-progress-ledger")
    parser.add_argument("--verifier-exit-code", type=int)
    parser.add_argument("--skip-sidecar", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    wire_events = transcript_to_wire_events(
        transcript,
        run_id=args.run_id,
        verifier_exit_code=args.verifier_exit_code,
    )
    write_wire_events(run_dir / "events.jsonl", wire_events)
    observation_events = build_observation_events(
        transcript,
        run_id=args.run_id,
        verifier_exit_code=args.verifier_exit_code,
        expected_paths=set(read_json(run_dir / "task_metadata.json").get("expected_paths", [])),
    )
    write_jsonl(run_dir / "observation_events.jsonl", observation_events)

    if not args.skip_sidecar:
        proc = replay_sidecar(run_dir=run_dir, ledger_root=Path(args.ledger_root))
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            write_run_manifest(
                run_dir,
                run_id=args.run_id,
                run_status=RunStatus.INFRASTRUCTURE_FAILURE,
                final_success=None,
                termination_reason="sidecar_replay_failed",
            )
            return proc.returncode

    final_success = None if args.verifier_exit_code is None else args.verifier_exit_code == 0
    if final_success is None:
        status = RunStatus.INFRASTRUCTURE_FAILURE
        termination_reason = "verifier_not_run"
    else:
        status = RunStatus.COMPLETED_SUCCESS if final_success else RunStatus.COMPLETED_FAILURE
        termination_reason = "verifier_pass" if final_success else "verifier_fail"
    write_run_manifest(
        run_dir,
        run_id=args.run_id,
        run_status=status,
        final_success=final_success,
        termination_reason=termination_reason,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
