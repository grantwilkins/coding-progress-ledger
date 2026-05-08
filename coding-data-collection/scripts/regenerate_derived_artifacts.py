from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.artifacts import read_json
from coding_data_collection.ledger import replay_sidecar, transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events, read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate events, observation events, and sidecar outputs for existing run directories."
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Run directory or corpus root containing run directories.")
    parser.add_argument("--ledger-root", type=Path, default=Path("../coding-progress-ledger"))
    parser.add_argument("--skip-sidecar", action="store_true")
    args = parser.parse_args(argv)

    issues: list[str] = []
    run_dirs = _discover_run_dirs(args.roots)
    for run_dir in run_dirs:
        try:
            _regenerate_run(run_dir, ledger_root=args.ledger_root, skip_sidecar=args.skip_sidecar)
        except Exception as exc:  # pragma: no cover - command-line error aggregation
            issues.append(f"{run_dir}: {exc}")

    for issue in issues:
        print(issue, file=sys.stderr)
    print(f"regenerated={len(run_dirs) - len(issues)} failed={len(issues)}")
    return 1 if issues else 0


def _discover_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for root in roots:
        if (root / "run_manifest.json").is_file():
            run_dirs.append(root)
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "run_manifest.json").is_file():
                run_dirs.append(child)
    return run_dirs


def _regenerate_run(run_dir: Path, *, ledger_root: Path, skip_sidecar: bool) -> None:
    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.is_file():
        return
    manifest = read_json(run_dir / "run_manifest.json")
    verifier_exit_code = _verifier_exit_code(manifest.get("final_success"))
    transcript = read_jsonl(transcript_path)
    metadata = read_json(run_dir / "task_metadata.json") if (run_dir / "task_metadata.json").is_file() else {}
    expected_paths = set(metadata.get("expected_paths", []))
    write_wire_events(
        run_dir / "events.jsonl",
        transcript_to_wire_events(
            transcript,
            run_id=run_dir.name,
            verifier_exit_code=verifier_exit_code,
        ),
    )
    write_jsonl(
        run_dir / "observation_events.jsonl",
        build_observation_events(
            transcript,
            run_id=run_dir.name,
            verifier_exit_code=verifier_exit_code,
            expected_paths=expected_paths,
        ),
    )
    if not skip_sidecar:
        sidecar = replay_sidecar(run_dir=run_dir, ledger_root=ledger_root)
        if sidecar.returncode != 0:
            raise RuntimeError(sidecar.stderr.strip() or "sidecar replay failed")


def _verifier_exit_code(final_success: object) -> int | None:
    if final_success is True:
        return 0
    if final_success is False:
        return 1
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
