#!/usr/bin/env python3
"""Append one wire-format ledger event to <run_dir>/events.jsonl.

Usage:
    tb_emit.py <run_dir> <step> <ledger_ops_json>

Timestamp is wall-clock now (UTC, ISO-8601). Run id is the run dir basename.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
event = {
    "schema_version": "1.0",
    "run_id": run_dir.name,
    "step": int(sys.argv[2]),
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ledger_ops": json.loads(sys.argv[3]),
}
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "events.jsonl").open("a") as f:
    f.write(json.dumps(event) + "\n")
