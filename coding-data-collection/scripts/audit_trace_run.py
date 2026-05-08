from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from coding_data_collection.artifacts import read_json
from coding_data_collection.audits import redaction_audit_report, scan_agent_workspace_for_leakage
from coding_data_collection.observation import read_jsonl


MODEL_METADATA_FIELDS = {
    "collection_kind",
    "model_provider",
    "model_name",
    "temperature",
    "max_model_calls",
    "max_tool_calls",
    "max_wall_time_s",
    "max_tokens_out",
    "total_model_calls",
    "total_tokens_in",
    "total_tokens_out",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one model-agent trace run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-transcript-steps", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = audit_trace_run(args.run_dir, min_transcript_steps=args.min_transcript_steps)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report["passed"] else 1


def audit_trace_run(run_dir: Path, *, min_transcript_steps: int = 5) -> dict[str, Any]:
    issues: list[str] = []
    manifest = read_json(run_dir / "run_manifest.json")
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    ledger = read_jsonl(run_dir / "ledger.jsonl")

    _require(metrics.get("agent_backend") == "model_tool_loop", issues, "run_manifest.metrics.agent_backend != model_tool_loop")
    _require(metrics.get("collection_kind") == "model_agent", issues, "run_manifest.metrics.collection_kind != model_agent")
    for field in ("started_at", "ended_at", "wallclock_seconds"):
        _require(field in manifest, issues, f"run_manifest.{field} missing")
    _require(len(transcript) >= min_transcript_steps, issues, f"transcript has fewer than {min_transcript_steps} rows")
    _require(bool(observations), issues, "observation_events.jsonl is missing or empty")
    _require(bool(events), issues, "events.jsonl is missing or empty")
    _require(bool(ledger), issues, "ledger.jsonl is missing or empty")
    _require((run_dir / "verifier_output.txt").is_file(), issues, "verifier_output.txt missing")

    final_agent_step = max(
        (int(row.get("step", 0)) for row in transcript if row.get("visible_to_agent") is not False),
        default=0,
    )
    terminal_events = [
        event
        for event in observations
        if event.get("event_type") in {"verifier_pass", "verifier_fail", "verifier_disagreement"}
    ]
    _require(bool(terminal_events), issues, "no terminal verifier observation event")
    for event in terminal_events:
        if int(event.get("step", 0)) <= final_agent_step:
            issues.append("terminal verifier observation is not after final agent-visible step")

    workspace = run_dir / "agent_workspace"
    if workspace.exists():
        leakage = scan_agent_workspace_for_leakage(workspace)
        _require(leakage["passed"], issues, f"agent workspace leakage hits: {leakage['leakage_hits']}")
    redaction = redaction_audit_report(run_dir)
    _require(redaction["passed"], issues, f"redaction leakage hits: {redaction['redaction_hits']}")

    missing_model_fields = sorted(field for field in MODEL_METADATA_FIELDS if field not in metrics)
    _require(not missing_model_fields, issues, f"missing model metadata fields: {missing_model_fields}")

    return {
        "run_dir": str(run_dir),
        "passed": not issues,
        "issues": issues,
        "metrics": {
            "transcript_rows": len(transcript),
            "observation_event_rows": len(observations),
            "event_rows": len(events),
            "ledger_rows": len(ledger),
            "final_agent_visible_step": final_agent_step,
            "terminal_event_steps": [int(event.get("step", 0)) for event in terminal_events],
        },
    }


def _require(condition: bool, issues: list[str], message: str) -> None:
    if not condition:
        issues.append(message)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
