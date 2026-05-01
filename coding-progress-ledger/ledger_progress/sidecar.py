from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import EventType, Ledger, LedgerEvent, SubtaskCategory, apply_event
from .run_manager import _load_rescore_module, _sha256_file, _update_manifest_generated
from .serialization import to_jsonl
from .session import LedgerSession


SCHEMA_MAJOR = "1"
GENERATED = ["ledger.jsonl", "progress.csv", "progress_by_category.csv", "summary_by_category.json"]


class LedgerSidecar:
    def __init__(self, run_dir: str | Path, adapter_name: str = "generic", root_task: str | None = None):
        self.run_dir = Path(run_dir)
        self.adapter = _load_adapter(adapter_name)
        self.root_task = root_task
        self.run_id: str | None = None
        self.ledger: Ledger | None = None
        self.last_step = -1

    @property
    def session(self) -> LedgerSession:
        if self.ledger is None:
            raise ValueError("sidecar has not received any events")
        return LedgerSession(self.ledger, clock=lambda: None)

    def process_lines(self, lines: Iterable[str]) -> None:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            self.process_event(json.loads(line), line_number)

    def process_event(self, data: dict[str, Any], line_number: int = 1) -> None:
        event = _validate_event(data, line_number)
        self._ensure_ledger(event["run_id"], event["timestamp"])
        step = event["step"]
        if step < self.last_step:
            raise ValueError(f"line {line_number}: step must be monotonic per run_id")
        self.last_step = step

        ops = event.get("ledger_ops") or []
        if ops:
            inferred = ops
        else:
            agent_step = event.get("agent_step")
            if not isinstance(agent_step, dict):
                raise ValueError(f"line {line_number}: agent_step is required when ledger_ops is empty")
            inferred = self.adapter.infer_events({**agent_step, "step": step})
        for op in inferred:
            self._apply_op(op, step, event["timestamp"])
        self._write_artifacts()

    def _ensure_ledger(self, run_id: str, timestamp: str) -> None:
        if self.run_id is not None and run_id != self.run_id:
            raise ValueError(f"sidecar run_dir accepts one run_id; got {run_id!r} after {self.run_id!r}")
        if self.ledger is None:
            self.run_id = run_id
            root_task = self.root_task or run_id
            self.ledger = Ledger(root_task, events=[LedgerEvent(0, EventType.INIT, None, {"root_task": root_task}, timestamp=timestamp)])

    def _apply_op(self, op: dict[str, Any], step: int, timestamp: str) -> None:
        if self.ledger is None:
            raise ValueError("ledger is not initialized")
        kind = _required(op, "op")
        subtask_id = op.get("id")
        if kind == "add":
            payload = {
                "description": _required(op, "description"),
                "parent_id": op.get("parent_id"),
                "weight": op.get("weight", 1.0),
                "category": _category(op.get("category", "product")),
            }
            self._apply(EventType.ADD_SUBTASK, subtask_id, payload, op.get("reason"), step, timestamp)
        elif kind == "start":
            self._apply(EventType.UPDATE_STATUS, subtask_id, _status_payload("in_progress", op), op.get("reason"), step, timestamp)
        elif kind == "complete":
            self._apply(EventType.UPDATE_STATUS, subtask_id, _status_payload("complete", op), op.get("reason"), step, timestamp)
        elif kind == "block":
            reason = _required(op, "reason")
            self._apply(EventType.UPDATE_STATUS, subtask_id, _status_payload("blocked", op), reason, step, timestamp)
        elif kind == "reopen":
            reason = _required(op, "reason")
            self._apply(EventType.REOPEN_SUBTASK, subtask_id, {"reason": reason}, reason, step, timestamp)
        elif kind == "invalidate":
            reason = _required(op, "reason")
            self._apply(EventType.INVALIDATE_SUBTASK, subtask_id, {"reason": reason}, reason, step, timestamp)
        elif kind == "split":
            self._apply(EventType.SPLIT_SUBTASK, subtask_id, {"children": _children(_required(op, "children"))}, _required(op, "reason"), step, timestamp)
        elif kind == "add_evidence":
            self._apply(EventType.ADD_EVIDENCE, subtask_id, {"evidence": _evidence(op)}, op.get("reason"), step, timestamp)
        else:
            raise ValueError(f"unknown ledger op: {kind}")

    def _apply(
        self,
        event_type: EventType,
        subtask_id: str | None,
        payload: dict[str, Any],
        reason: str | None,
        step: int,
        timestamp: str,
    ) -> None:
        if not subtask_id:
            raise ValueError(f"{event_type.value} requires op.id")
        apply_event(self.ledger, LedgerEvent(step, event_type, subtask_id, payload, reason, timestamp))

    def _write_artifacts(self) -> None:
        if self.ledger is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_scaffold(self.run_dir, self.run_id or self.run_dir.name, self.ledger.root_task)
        ledger_path = self.run_dir / "ledger.jsonl"
        to_jsonl(self.ledger, str(ledger_path))
        before = _sha256_file(ledger_path)
        LedgerSession(self.ledger, clock=lambda: None).export_curve_csv(str(self.run_dir / "progress.csv"))
        summary = _load_rescore_module().rescore_run(self.run_dir)
        after = _sha256_file(ledger_path)
        if after != before:
            raise RuntimeError("sidecar export changed ledger.jsonl")
        summary.update({"source_ledger_sha256": before, "generator": "ledger_progress.sidecar"})
        (self.run_dir / "summary_by_category.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _update_manifest_generated(self.run_dir, GENERATED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ledger_progress.sidecar")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adapter", default="generic")
    parser.add_argument("--input-file")
    parser.add_argument("--root-task")
    args = parser.parse_args(argv)

    try:
        sidecar = LedgerSidecar(args.run_dir, args.adapter, args.root_task)
        if args.input_file:
            with Path(args.input_file).open() as file:
                sidecar.process_lines(file)
        else:
            sidecar.process_lines(sys.stdin)
    except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _validate_event(data: dict[str, Any], line_number: int) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"line {line_number}: event must be an object")
    version = _required(data, "schema_version")
    if not isinstance(version, str) or version.split(".", 1)[0] != SCHEMA_MAJOR:
        raise ValueError(f"line {line_number}: unsupported schema_version {version!r}")
    run_id = _required(data, "run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"line {line_number}: run_id must be a non-empty string")
    step = _required(data, "step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"line {line_number}: step must be a non-negative integer")
    timestamp = _required(data, "timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(f"line {line_number}: timestamp must be a non-empty ISO-8601 string")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    ops = data.get("ledger_ops")
    if ops is not None and not isinstance(ops, list):
        raise ValueError(f"line {line_number}: ledger_ops must be a list when present")
    return data


def _load_adapter(name: str):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid adapter name: {name}")
    try:
        return importlib.import_module(f"ledger_progress.adapters.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"unknown adapter: {name}") from exc


def _status_payload(status: str, op: dict[str, Any]) -> dict[str, Any]:
    payload = {"status": status}
    if "evidence" in op:
        payload["evidence"] = _evidence(op)
    return payload


def _evidence(op: dict[str, Any]) -> list[str]:
    evidence = _required(op, "evidence")
    if isinstance(evidence, str):
        return [evidence]
    if not isinstance(evidence, list) or any(not isinstance(item, str) or not item for item in evidence):
        raise ValueError("op.evidence must be a non-empty string or list of non-empty strings")
    return evidence


def _children(children: Any) -> list[dict[str, Any]]:
    if not isinstance(children, list) or not children:
        raise ValueError("op.children must be a non-empty list")
    out = []
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("op.children entries must be objects")
        item = {"id": _required(child, "id"), "description": _required(child, "description")}
        if "category" in child:
            item["category"] = _category(child["category"])
        if "weight" in child:
            item["weight"] = child["weight"]
        if "status" in child:
            item["status"] = child["status"]
        out.append(item)
    return out


def _category(value: Any) -> str:
    return SubtaskCategory(value).value


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"missing {key}")
    return data[key]


def _write_run_scaffold(run_dir: Path, run_id: str, root_task: str) -> None:
    _write_if_missing(run_dir / "task.md", f"# Task\n\n{root_task}\n")
    _write_if_missing(run_dir / "run_notes.md", "# Run Notes\n\nLive sidecar-generated run.\n")
    _write_if_missing(run_dir / "final_diff.patch", "")
    _write_if_missing(run_dir / "test_output.txt", "")
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"run_id": run_id, "created_by": "ledger_progress.sidecar", "generated_artifacts": []}, indent=2, sort_keys=True) + "\n")


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text)


if __name__ == "__main__":
    raise SystemExit(main())
