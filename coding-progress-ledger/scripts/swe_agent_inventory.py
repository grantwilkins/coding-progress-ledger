#!/usr/bin/env python3
"""SWE-agent raw manifest builder (Workstream A, task A3).

Scans the chosen upstream SWE-agent trajectory source (currently
``nebius/SWE-agent-trajectories`` on Hugging Face, accessed via
``datasets.load_dataset(..., streaming=True)``) and emits a deterministic
manifest CSV at the path passed via ``--output``.

Determinism contract
--------------------
The acceptance criterion in ``TASKS.md`` (A3) is "same raw -> same CSV
byte-for-byte". HF streaming order is fixed in practice but not
contractually stable, so this script does NOT trust streaming order.
Instead it:

  1. Streams every row, keeping only small per-row metadata in memory
     (no trajectory / patch / eval log content is retained).
  2. Sorts the accumulated rows by ``(instance_id, model_name)``
     before writing.
  3. Writes with ``csv.QUOTE_MINIMAL``, ``newline=""``,
     ``lineterminator="\n"``, UTF-8, header row first. Booleans are
     written as the literals ``True``/``False`` and missing booleans
     as the empty string.

Source mutation contract
------------------------
The script never writes or modifies anything under
``external_data/swe_agent/raw/``. The Hugging Face cache (outside the
repo) is the only place row content is materialized.

CLI
---
    python scripts/swe_agent_inventory.py \
      --source nebius \
      --output external_data/swe_agent/manifests/swe_agent_inventory.csv \
      [--max-rows N] [--progress]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_COLUMNS: Tuple[str, ...] = (
    "source_id",
    "instance_id",
    "model_name",
    "trajectory_available",
    "trajectory_length",
    "final_success_available",
    "final_success",
    "patch_available",
    "eval_log_available",
    "repo_name",
    "issue_id",
    "raw_path_or_dataset_index",
    "parse_status",
    "parse_error",
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic raw manifest of SWE-agent trajectories.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("nebius",),
        help="Upstream source to scan. Only 'nebius' is wired up today; "
        "the flag exists so adding the SWE-smith fallback is a small change.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV path (e.g. "
        "external_data/swe_agent/manifests/swe_agent_inventory.csv).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of streamed rows; default is no cap "
        "(full ~80,036 for nebius).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Log progress to stderr every K rows while streaming.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="If --progress is set, log every this many streamed rows "
        "(default 1000).",
    )
    return parser.parse_args(argv)


def _sanitize_error(msg: str) -> str:
    """Collapse a one-line, comma-free error string for CSV safety."""
    return msg.replace("\n", " ").replace("\r", " ").replace(",", " ").strip()


def _parse_repo_and_issue(instance_id: str) -> Tuple[str, str, Optional[str]]:
    """Return (repo_name, issue_id, warn_reason).

    SWE-bench convention is ``<owner>__<repo>-<issue>``. Split on the
    LAST ``-`` to get the trailing issue id, then split the remaining on
    the FIRST ``__`` to get owner/repo. ``warn_reason`` is ``None`` on
    success, or a short string when parsing fails.
    """
    if not isinstance(instance_id, str) or not instance_id:
        return "", "", "missing instance_id"
    if "-" not in instance_id:
        return "", "", "no dash in instance_id"
    head, _, issue = instance_id.rpartition("-")
    if not issue or not head:
        return "", "", "empty head or issue after rpartition"
    if "__" not in head:
        return "", issue, "no double-underscore in head"
    owner, _, repo = head.partition("__")
    if not owner or not repo:
        return "", issue, "empty owner or repo after partition"
    return f"{owner}/{repo}", issue, None


def _row_to_record(row: Dict[str, Any], hf_index: int) -> Dict[str, Any]:
    """Convert one streamed row to a manifest record dict.

    Captures only metadata; never retains trajectory/patch/eval log
    content. Per-row exceptions are caught at the caller; this function
    raises only on truly unexpected types.
    """
    instance_id_raw = row.get("instance_id")
    instance_id = instance_id_raw if isinstance(instance_id_raw, str) else ""

    model_name_raw = row.get("model_name")
    model_name = model_name_raw if isinstance(model_name_raw, str) else ""

    trajectory = row.get("trajectory")
    if isinstance(trajectory, list):
        trajectory_length = len(trajectory)
        trajectory_available = trajectory_length > 0
    else:
        trajectory_length = 0
        trajectory_available = False

    target = row.get("target")
    if isinstance(target, bool):
        final_success_available = True
        final_success: Any = target
    else:
        final_success_available = False
        final_success = ""

    generated_patch = row.get("generated_patch")
    patch_available = isinstance(generated_patch, str) and len(generated_patch) > 0

    eval_logs = row.get("eval_logs")
    eval_log_available = isinstance(eval_logs, str) and len(eval_logs) > 0

    repo_name, issue_id, warn_reason = _parse_repo_and_issue(instance_id)

    if warn_reason is not None:
        parse_status = "warn_repo_parse"
        parse_error = _sanitize_error(warn_reason)
    else:
        parse_status = "ok"
        parse_error = ""

    source_id = f"nebius:{instance_id}:{model_name}"
    raw_path_or_dataset_index = (
        f"nebius/SWE-agent-trajectories:train:{hf_index}"
    )

    return {
        "source_id": source_id,
        "instance_id": instance_id,
        "model_name": model_name,
        "trajectory_available": trajectory_available,
        "trajectory_length": trajectory_length,
        "final_success_available": final_success_available,
        "final_success": final_success,
        "patch_available": patch_available,
        "eval_log_available": eval_log_available,
        "repo_name": repo_name,
        "issue_id": issue_id,
        "raw_path_or_dataset_index": raw_path_or_dataset_index,
        "parse_status": parse_status,
        "parse_error": parse_error,
    }


def _format_cell(value: Any) -> str:
    """Render a manifest cell to a CSV-safe string.

    Booleans become ``"True"``/``"False"``; ``""`` (the empty string,
    used for missing booleans) stays empty so QUOTE_MINIMAL won't quote
    it. Everything else is ``str(...)``.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value)


def _stream_records(
    source: str,
    max_rows: Optional[int],
    progress: bool,
    progress_every: int,
) -> List[Dict[str, Any]]:
    """Stream rows, build records, accumulate. Catches per-row errors."""
    if source != "nebius":
        # argparse already restricts choices, but be defensive.
        raise ValueError(f"unsupported --source {source!r}")

    # Local import: keeps the module importable even if `datasets` is not
    # installed, e.g. for static analysis.
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=True,
    )

    records: List[Dict[str, Any]] = []
    error_count = 0
    started = time.monotonic()

    for hf_index, row in enumerate(ds):
        if max_rows is not None and hf_index >= max_rows:
            break
        try:
            record = _row_to_record(row, hf_index)
        except Exception as exc:  # noqa: BLE001 - we want to log and continue
            error_count += 1
            # Try to extract whatever we can; use blanks otherwise.
            instance_id = ""
            model_name = ""
            try:
                if isinstance(row, dict):
                    if isinstance(row.get("instance_id"), str):
                        instance_id = row["instance_id"]
                    if isinstance(row.get("model_name"), str):
                        model_name = row["model_name"]
            except Exception:  # noqa: BLE001
                pass
            record = {
                "source_id": f"nebius:{instance_id}:{model_name}",
                "instance_id": instance_id,
                "model_name": model_name,
                "trajectory_available": False,
                "trajectory_length": 0,
                "final_success_available": False,
                "final_success": "",
                "patch_available": False,
                "eval_log_available": False,
                "repo_name": "",
                "issue_id": "",
                "raw_path_or_dataset_index": (
                    f"nebius/SWE-agent-trajectories:train:{hf_index}"
                ),
                "parse_status": "error",
                "parse_error": _sanitize_error(
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        records.append(record)

        if progress and ((hf_index + 1) % progress_every == 0):
            elapsed = time.monotonic() - started
            rate = (hf_index + 1) / elapsed if elapsed > 0 else float("inf")
            print(
                f"[swe_agent_inventory] streamed {hf_index + 1} rows "
                f"({rate:.1f} rows/s, errors so far={error_count})",
                file=sys.stderr,
                flush=True,
            )

    if records and error_count == len(records):
        # Every row failed to parse: this is fatal per A3 acceptance.
        raise RuntimeError(
            f"all {error_count} streamed rows failed to parse; "
            "refusing to write a manifest."
        )

    return records


def _write_csv(records: List[Dict[str, Any]], output: Path) -> None:
    """Sort records deterministically and write the manifest CSV."""
    # Determinism: sort by (instance_id, model_name). Both are strings
    # (possibly empty) by construction in _row_to_record.
    records_sorted = sorted(
        records, key=lambda r: (r["instance_id"], r["model_name"])
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(
            fh,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(list(CSV_COLUMNS))
        for record in records_sorted:
            writer.writerow([_format_cell(record[col]) for col in CSV_COLUMNS])


def _log_summary(records: List[Dict[str, Any]]) -> None:
    """Log to stderr: total rows, status counts, success-label distribution."""
    total = len(records)
    status_counts: Counter[str] = Counter(r["parse_status"] for r in records)

    success_true = 0
    success_false = 0
    success_missing = 0
    for r in records:
        if not r["final_success_available"]:
            success_missing += 1
        elif r["final_success"] is True:
            success_true += 1
        elif r["final_success"] is False:
            success_false += 1
        else:
            # Defensive: shouldn't happen given _row_to_record.
            success_missing += 1

    print(
        f"[swe_agent_inventory] total rows scanned: {total}",
        file=sys.stderr,
    )
    print(
        f"[swe_agent_inventory] parse_status counts: "
        f"{dict(sorted(status_counts.items()))}",
        file=sys.stderr,
    )
    print(
        f"[swe_agent_inventory] final_success: True={success_true} "
        f"False={success_false} missing={success_missing}",
        file=sys.stderr,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    records = _stream_records(
        source=args.source,
        max_rows=args.max_rows,
        progress=args.progress,
        progress_every=args.progress_every,
    )

    if not records:
        print(
            "[swe_agent_inventory] WARNING: no rows streamed; "
            "writing header-only CSV.",
            file=sys.stderr,
        )

    _write_csv(records, args.output)
    _log_summary(records)

    print(
        f"[swe_agent_inventory] wrote manifest -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
