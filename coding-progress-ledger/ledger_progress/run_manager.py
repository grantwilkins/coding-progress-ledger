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

from . import queries
from .core import EventType, Status, SubtaskCategory
from .serialization import from_jsonl, load_events_jsonl
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

    watch = subparsers.add_parser("watch", help="Tail ledger.jsonl and print per-event progress updates.")
    watch.add_argument("run_dir")
    watch.add_argument("--poll-interval", type=float, default=0.5)
    watch.add_argument("--exit-after-events", type=int, default=0, help="Exit after observing this many total events (0 = run until interrupted).")
    watch.set_defaults(func=_cmd_watch)

    query = subparsers.add_parser("query", help="Run live queries against ledger.jsonl. Output is JSON.")
    query.add_argument("run_dir")
    query.add_argument("--status", choices=["blocked"], help="List active leaves with the given status.")
    query.add_argument("--stalled-for", type=int, metavar="N", help="Print stalled_for(BLOCKED) in steps and whether it is >= N.")
    query.add_argument("--reopens-since", type=int, metavar="STEP")
    query.add_argument("--newly-discovered-since", type=int, metavar="STEP")
    query.add_argument("--last-validation-event", action="store_true")
    query.set_defaults(func=_cmd_query)

    serve = subparsers.add_parser("serve", help="Run an HTTP progress probe for a single run dir.")
    serve.add_argument("run_dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--exit-after-events", type=int, default=0)
    serve.set_defaults(func=_cmd_serve)

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


def _cmd_watch(args: argparse.Namespace) -> int:
    from .core import replay
    from .scoring import score

    run_dir = _existing_run_dir(args.run_dir)
    ledger_path = run_dir / "ledger.jsonl"
    seen = 0
    target = args.exit_after_events
    while True:
        if not ledger_path.exists():
            time.sleep(args.poll_interval)
            continue
        events = load_events_jsonl(str(ledger_path))
        if len(events) > seen:
            for end in range(seen + 1, len(events) + 1):
                ledger = replay(events[:end])
                event = events[end - 1]
                obs_coding = score(ledger, categories=("product", "validation", "investigation"))
                update = {
                    "event_index": end - 1,
                    "step": event.step,
                    "event_type": event.event_type.value,
                    "subtask_id": event.subtask_id,
                    "timestamp": event.timestamp,
                    "coding_progress": obs_coding.progress,
                    "active_blocked_leaves": [s.id for s in queries.active_blocked_leaves(ledger)],
                    "stalled_for_blocked": queries.stalled_for(ledger),
                }
                print(json.dumps(update, sort_keys=True))
            seen = len(events)
            sys.stdout.flush()
        if target and seen >= target:
            return 0
        time.sleep(args.poll_interval)


def _cmd_query(args: argparse.Namespace) -> int:
    run_dir = _existing_run_dir(args.run_dir)
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        raise FileNotFoundError(f"{ledger_path} is required")
    ledger = from_jsonl(str(ledger_path))

    out: dict[str, Any] = {"run_dir": str(run_dir), "current_step": queries.current_step(ledger)}
    if args.status == "blocked":
        out["active_blocked_leaves"] = [_subtask_to_dict(s) for s in queries.active_blocked_leaves(ledger)]
    if args.stalled_for is not None:
        stalled = queries.stalled_for(ledger)
        out["stalled_for_blocked"] = stalled
        out["stalled_for_threshold"] = args.stalled_for
        out["meets_threshold"] = stalled >= args.stalled_for
    if args.reopens_since is not None:
        out["reopens_since"] = [_event_to_dict(e) for e in queries.reopens_since(ledger, args.reopens_since)]
    if args.newly_discovered_since is not None:
        out["newly_discovered_since"] = [_subtask_to_dict(s) for s in queries.newly_discovered_since(ledger, args.newly_discovered_since)]
    if args.last_validation_event:
        event = queries.last_validation_event(ledger)
        out["last_validation_event"] = _event_to_dict(event) if event else None

    print(json.dumps(out, sort_keys=True))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    from .core import replay
    from .scoring import score
    from .serialization import event_from_dict, load_events_jsonl

    run_dir = _existing_run_dir(args.run_dir)
    ledger_path = run_dir / "ledger.jsonl"
    events = load_events_jsonl(str(ledger_path)) if ledger_path.exists() and ledger_path.stat().st_size > 0 else []
    state = {"posts": 0}
    target = args.exit_after_events

    def current_ledger():
        return replay(events) if events else None

    def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def do_POST(self):
            if urlparse(self.path).path != "/events":
                write_json(self, 404, {"error": "not found"}); return
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode())
            events.append(event_from_dict(data))
            with ledger_path.open("a") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
            state["posts"] += 1
            write_json(self, 200, {"ok": True, "events": state["posts"]})
            if target and state["posts"] >= target:
                threading.Thread(target=server.shutdown, daemon=True).start()

        def do_GET(self):
            parsed = urlparse(self.path)
            ledger = current_ledger()
            if parsed.path == "/progress":
                if ledger is None:
                    write_json(self, 200, {"coding_progress": 0.0, "validation_progress": 0.0, "current_step": 0}); return
                coding = score(ledger, categories=("product", "validation", "investigation"))
                validation = score(ledger, categories=("validation",))
                write_json(self, 200, {
                    "coding_progress": coding.progress,
                    "validation_progress": validation.progress,
                    "current_step": queries.current_step(ledger),
                })
            elif parsed.path == "/blocked":
                blocked = [_subtask_to_dict(s) for s in queries.active_blocked_leaves(ledger)] if ledger else []
                write_json(self, 200, {"active_blocked_leaves": blocked})
            elif parsed.path == "/stalled":
                params = parse_qs(parsed.query)
                threshold = int(params.get("threshold", ["0"])[0])
                stalled = queries.stalled_for(ledger) if ledger else 0
                write_json(self, 200, {"stalled_for_blocked": stalled, "meets_threshold": stalled >= threshold})
            else:
                write_json(self, 404, {"error": "not found"})

    server = HTTPServer((args.host, args.port), Handler)
    addr = {"host": server.server_address[0], "port": server.server_address[1]}
    (run_dir / "serve_address.json").write_text(json.dumps(addr))
    print(json.dumps(addr), flush=True)
    server.serve_forever()
    server.server_close()
    return 0


def _subtask_to_dict(subtask: Any) -> dict[str, Any]:
    return {
        "id": subtask.id,
        "description": subtask.description,
        "status": subtask.status.value,
        "category": subtask.category.value,
        "weight": subtask.weight,
        "parent_id": subtask.parent_id,
        "created_at_step": subtask.created_at_step,
        "updated_at_step": subtask.updated_at_step,
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "step": event.step,
        "event_type": event.event_type.value,
        "subtask_id": event.subtask_id,
        "timestamp": event.timestamp,
        "reason": event.reason,
    }


def missing_artifacts(run_dir: Path) -> list[str]:
    return [artifact for artifact in REQUIRED_ARTIFACTS if not (run_dir / artifact).exists()]


def resolve_final_success(run_dir: Path, summary: dict[str, Any] | None = None) -> tuple[bool | None, str]:
    manifest = _load_json_if_present(run_dir / "run_manifest.json")
    if isinstance(manifest.get("final_success"), bool):
        return manifest["final_success"], manifest.get("final_success_source", "run_manifest.final_success")

    # Authoritative upstream label from source_metadata.json (e.g. C3's
    # SWE-agent importer pins final_success from the upstream `target`).
    # Take this before any heuristic that scans test_output.txt, because
    # eval logs from SWE-bench-style harnesses interleave "passed",
    # "error", and "failed" tokens that fool the keyword scan.
    md_fs, md_source = _final_success_from_source_metadata(run_dir)
    if md_fs is not None:
        return md_fs, md_source

    summary = summary if summary is not None else _load_json_if_present(run_dir / "summary_by_category.json")
    if isinstance(summary.get("final_success"), bool) and summary.get("final_success_source") in {
        "summary.final_success",
        "hidden_check_tests",
        "manual",
        "source_metadata.target",
    }:
        return summary["final_success"], summary.get("final_success_source", "summary_by_category.final_success")

    test_result = _load_json_if_present(run_dir / "test_result.json")
    if isinstance(test_result.get("success"), bool):
        return test_result["success"], "test_result.success"

    if isinstance(summary.get("final_success"), bool):
        return summary["final_success"], summary.get("final_success_source", "summary_by_category.final_success")
    return None, "unknown"


def _final_success_from_source_metadata(run_dir: Path) -> tuple[bool | None, str]:
    """Return upstream-declared final_success when source_metadata.json
    has it pinned by construction (final_success_source == "source_label").
    """
    md_path = run_dir / "source_metadata.json"
    if not md_path.is_file():
        return None, ""
    try:
        md = json.loads(md_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, ""
    if md.get("final_success_source") != "source_label":
        return None, ""
    fs = md.get("final_success")
    if isinstance(fs, bool):
        return fs, "source_metadata.target"
    return None, ""


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
