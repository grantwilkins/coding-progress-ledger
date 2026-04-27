from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .serialization import from_jsonl
from .session import LedgerSession


REQUIRED_ARTIFACTS = (
    "task.md",
    "ledger.jsonl",
    "progress.csv",
    "progress_by_category.csv",
    "summary_by_category.json",
    "final_diff.patch",
    "test_output.txt",
    "run_notes.md",
)
GENERATED_PLACEHOLDERS = (
    "ledger.jsonl",
    "progress.csv",
    "progress_by_category.csv",
    "summary_by_category.json",
    "final_diff.patch",
    "test_output.txt",
    "test_result.json",
)
GENERATOR = "ledger-run export-run"
ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger-run", description="Manage coding progress ledger run artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_run = subparsers.add_parser("init-run", help="Initialize a run artifact directory.")
    init_run.add_argument("run_dir")
    init_run.set_defaults(func=_cmd_init_run)

    export_run = subparsers.add_parser("export-run", help="Regenerate derived progress artifacts from ledger.jsonl.")
    export_run.add_argument("run_dir")
    export_run.set_defaults(func=_cmd_export_run)

    capture_tests = subparsers.add_parser("capture-tests", help="Run a test command and capture its output.")
    capture_tests.add_argument("run_dir")
    capture_tests.add_argument("test_command", nargs=argparse.REMAINDER)
    capture_tests.set_defaults(func=_cmd_capture_tests)

    capture_diff = subparsers.add_parser("capture-diff", help="Capture the current git diff.")
    capture_diff.add_argument("run_dir")
    capture_diff.set_defaults(func=_cmd_capture_diff)

    check_run = subparsers.add_parser("check-run", help="Check for required run artifacts.")
    check_run.add_argument("run_dir")
    check_run.set_defaults(func=_cmd_check_run)

    summarize_run = subparsers.add_parser("summarize-run", help="Print a compact run summary.")
    summarize_run.add_argument("run_dir")
    summarize_run.set_defaults(func=_cmd_summarize_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_init_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_if_missing(run_dir / "task.md", _task_template())
    _write_if_missing(run_dir / "run_notes.md", _run_notes_template())
    for artifact in GENERATED_PLACEHOLDERS:
        _write_if_missing(run_dir / f"{artifact}.README", _placeholder_text(artifact))
    manifest = {
        "run_id": run_dir.name,
        "created_by": "ledger-run init-run",
        "status": "initialized",
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "generated_artifacts": [],
    }
    _write_if_missing(run_dir / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"initialized {run_dir}")
    return 0


def _cmd_export_run(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        raise FileNotFoundError(f"{ledger_path} is required")

    before = _sha256_file(ledger_path)
    ledger = from_jsonl(str(ledger_path))
    LedgerSession(ledger).export_curve_csv(str(run_dir / "progress.csv"))

    rescore = _load_rescore_module()
    summary = rescore.rescore_run(run_dir)
    after = _sha256_file(ledger_path)
    if before != after:
        raise RuntimeError("export-run changed ledger.jsonl")

    summary.update(
        {
            "source_ledger_sha256": before,
            "generated_at": _utc_now(),
            "generator": GENERATOR,
        }
    )
    (run_dir / "summary_by_category.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _update_manifest_generated(run_dir, ["progress.csv", "progress_by_category.csv", "summary_by_category.json"])
    print(f"exported {run_dir}")
    return 0


def _cmd_capture_tests(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    command = list(args.test_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("capture-tests requires a command after --")

    cwd = str(Path.cwd())
    started_at = _utc_now()
    start = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    finished_at = _utc_now()
    duration = time.monotonic() - start

    result = {
        "command": command,
        "exit_code": completed.returncode,
        "success": completed.returncode == 0,
        "cwd": cwd,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
    }
    (run_dir / "test_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (run_dir / "test_output.txt").write_text(_test_output_text(result, completed.stdout, completed.stderr))
    _update_manifest_generated(run_dir, ["test_output.txt", "test_result.json"])
    return completed.returncode


def _cmd_capture_diff(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    completed = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return completed.returncode
    (run_dir / "final_diff.patch").write_text(completed.stdout)
    _update_manifest_generated(run_dir, ["final_diff.patch"])
    print(f"captured diff to {run_dir / 'final_diff.patch'}")
    return 0


def _cmd_check_run(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    missing = missing_artifacts(run_dir)
    if missing:
        for artifact in missing:
            print(f"missing: {artifact}")
        return 1
    print("all required artifacts present")
    return 0


def _cmd_summarize_run(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    summary = _load_json_if_present(run_dir / "summary_by_category.json")
    missing = missing_artifacts(run_dir)
    final_success, final_success_source = resolve_final_success(run_dir, summary)

    if summary:
        _print_summary_value("final_coding_progress", summary.get("final_coding_progress"))
        _print_summary_value("final_overall_progress", summary.get("final_overall_progress"))
        _print_summary_value("coding_nonmonotonic", summary.get("coding_nonmonotonic"))
        _print_summary_value("overall_nonmonotonic", summary.get("overall_nonmonotonic"))
        _print_summary_value("weak_completion_evidence_count", summary.get("weak_completion_evidence_count"))
    else:
        print("summary_by_category.json: missing")
    _print_summary_value("final_success", final_success)
    _print_summary_value("final_success_source", final_success_source)

    warning = stale_summary_warning(run_dir, summary)
    if warning:
        print(f"warning: {warning}")
    if missing:
        print("missing artifacts: " + ", ".join(missing))
    else:
        print("missing artifacts: none")
    return 0


def missing_artifacts(run_dir: Path) -> list[str]:
    return [artifact for artifact in REQUIRED_ARTIFACTS if not (run_dir / artifact).exists()]


def resolve_final_success(run_dir: Path, summary: dict[str, Any] | None = None) -> tuple[bool | None, str]:
    manifest = _load_json_if_present(run_dir / "run_manifest.json")
    if isinstance(manifest.get("final_success"), bool):
        return manifest["final_success"], manifest.get("final_success_source", "run_manifest.final_success")

    summary = summary if summary is not None else _load_json_if_present(run_dir / "summary_by_category.json")
    if isinstance(summary.get("final_success"), bool) and summary.get("final_success_source") in {
        "summary.final_success",
        "hidden_check_tests",
        "manual",
    }:
        return summary["final_success"], summary.get("final_success_source", "summary_by_category.final_success")

    test_result = _load_json_if_present(run_dir / "test_result.json")
    if isinstance(test_result.get("success"), bool):
        return test_result["success"], "test_result.success"

    if isinstance(summary.get("final_success"), bool):
        return summary["final_success"], summary.get("final_success_source", "summary_by_category.final_success")
    return None, "unknown"


def stale_summary_warning(run_dir: Path, summary: dict[str, Any] | None = None) -> str | None:
    summary = summary if summary is not None else _load_json_if_present(run_dir / "summary_by_category.json")
    recorded = summary.get("source_ledger_sha256")
    ledger_path = run_dir / "ledger.jsonl"
    if not recorded or not ledger_path.exists():
        return None
    current = _sha256_file(ledger_path)
    if current != recorded:
        return "summary was generated from a different ledger.jsonl"
    return None


def _existing_run_dir(value: str) -> Path:
    run_dir = Path(value)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} does not exist")
    return run_dir


def _load_rescore_module():
    path = ROOT / "scripts" / "rescore_suite_by_category.py"
    spec = importlib.util.spec_from_file_location("rescore_suite_by_category", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _update_manifest_generated(run_dir: Path, artifacts: list[str]) -> None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return
    manifest = _load_json_if_present(path)
    generated = set(manifest.get("generated_artifacts", []))
    generated.update(artifacts)
    manifest["generated_artifacts"] = sorted(generated)
    manifest["updated_at"] = _utc_now()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _test_output_text(result: dict[str, Any], stdout: str, stderr: str) -> str:
    command = " ".join(result["command"])
    rows = [
        f"command: {command}",
        f"cwd: {result['cwd']}",
        f"started_at: {result['started_at']}",
        f"finished_at: {result['finished_at']}",
        f"duration_seconds: {result['duration_seconds']:.6f}",
        f"exit_code: {result['exit_code']}",
        "",
        "stdout:",
        stdout,
        "",
        "stderr:",
        stderr,
    ]
    return "\n".join(rows)


def _print_summary_value(key: str, value: Any) -> None:
    if value is None:
        value = "unknown"
    print(f"{key}: {value}")


def _task_template() -> str:
    return """# Task

Describe the user-facing coding task here.

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Expected Validation Command

```bash
uv run pytest tests
```

## Notes

Add constraints or non-goals here.
"""


def _run_notes_template() -> str:
    return """# Run Notes

## Summary

What changed?

## Ledger Observations

- Where did progress increase?
- Where did progress decrease?
- Was any new work discovered?
- Were any subtasks reopened, split, or invalidated?

## Evidence Notes

What evidence supports completed product and validation leaves?

## Awkwardness / Protocol Issues

What felt awkward about maintaining the ledger?

## Final Result

- Final success:
- Success source:
- Test command:
"""


def _placeholder_text(artifact: str) -> str:
    if artifact == "ledger.jsonl":
        return """This run has been initialized, but no ledger has been exported yet.

During the coding task, use LedgerSession to record discovered work.
At the end of the run, export the event log to ledger.jsonl.

Do not create an empty ledger.jsonl as a placeholder.
check-run treats ledger.jsonl as missing until a real event log exists.
"""
    if artifact in {"progress.csv", "progress_by_category.csv", "summary_by_category.json"}:
        return f"""{artifact} is generated from ledger.jsonl.

Run:

  ledger-run export-run <run_dir>

after ledger.jsonl exists.
"""
    if artifact == "final_diff.patch":
        return """final_diff.patch should contain the final git diff for this run.

Run:

  ledger-run capture-diff <run_dir>
"""
    if artifact in {"test_output.txt", "test_result.json"}:
        return f"""{artifact} is generated by the validation command capture.

Run:

  ledger-run capture-tests <run_dir> -- <test command>
"""
    return f"{artifact} is generated during the run.\n"


if __name__ == "__main__":
    raise SystemExit(main())
