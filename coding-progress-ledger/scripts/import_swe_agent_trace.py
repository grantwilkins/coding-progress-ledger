#!/usr/bin/env python3
"""Bulk SWE-agent pilot importer (Workstream C, tasks C3 + C4).

Reads ``external_data/swe_agent/manifests/swe_agent_pilot_sample.csv``
(produced by B2) and materializes one run directory per pilot row
under ``--runs-dir``. Critically: NO ``ledger.jsonl`` is generated.
Annotation is its own workstream (D).

Source of truth for raw rows
----------------------------
Each pilot row's raw upstream content must already exist on disk at
``<raw-cache-dir>/<pilot_id>.json``. This script does NOT fetch from
Hugging Face; populating the cache (e.g. by re-streaming the dataset
and matching ``raw_path_or_dataset_index``) is a separate one-shot
operation deliberately kept out of the importer so the importer is
deterministic, offline, and unit-testable.

Per-run output layout (framework-standard names):

    <runs-dir>/<pilot_id>/
        task.md
        source_trace.json          (byte-equivalent copy from cache)
        normalized_trace.json
        trajectory_summary.md
        final_diff.patch           (sourced from upstream `generated_patch`)
        test_output.txt            (sourced from upstream `eval_logs`)
        run_notes.md
        source_metadata.json

Naming choice: `test_output.txt` is the framework's standard artifact
name (`ledger-run check-run` requires it; toy / control runs use it
too). The upstream nebius rows ship the same content under
`eval_logs`; we map upstream-field-name -> framework-artifact-name
here so every retrospective import produces framework-shaped run
dirs regardless of source. The mapping is the importer's
responsibility, not a per-run convention.

Verification
------------
After materializing each run dir, the importer self-verifies all
PRE_ANNOTATION_ARTIFACTS exist and (for ones the pilot CSV claims
should be non-empty) are non-empty. ``--verify-only`` re-runs that
check across an existing runs-dir without re-importing.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalize_swe_agent_trace import normalize_row, render_summary


PRE_ANNOTATION_ARTIFACTS: Tuple[str, ...] = (
    "task.md",
    "source_trace.json",
    "normalized_trace.json",
    "trajectory_summary.md",
    "final_diff.patch",
    "test_output.txt",
    "run_notes.md",
    "source_metadata.json",
)

# Files that must exist AND be non-empty when the corresponding pilot
# CSV flag is True. Mapping artifact -> pilot CSV column.
NONEMPTY_IF_FLAG: Dict[str, str] = {
    "final_diff.patch": "patch_available",
    "test_output.txt": "eval_log_available",
}

RUN_NOTES_TEMPLATE = """\
# Run notes — {pilot_id}

Source: SWE-agent retrospective pilot
instance_id: `{instance_id}`
model_name: `{model_name}`
final_success (upstream label, NOT a feature): `{final_success}`

## TODO — annotation

- [ ] Read `trajectory_summary.md` end-to-end before opening the source trace.
- [ ] Decide step-level subtasks per D1 protocol; do NOT infer from outcome.
- [ ] Capture evidence quotes (step indices) in the ledger payload.
- [ ] Mark validation evidence as weak/strong per K1.

## TODO — annotation log

- (record any deviations, ambiguous cases, or annotator uncertainty here)
"""


@dataclass(frozen=True)
class PilotRow:
    pilot_id: str
    instance_id: str
    model_name: str
    final_success: Optional[bool]
    patch_available: bool
    eval_log_available: bool

    @classmethod
    def from_csv(cls, row: Dict[str, str]) -> "PilotRow":
        def _b(v: str) -> Optional[bool]:
            if v == "True":
                return True
            if v == "False":
                return False
            return None

        return cls(
            pilot_id=row["pilot_id"],
            instance_id=row["instance_id"],
            model_name=row["model_name"],
            final_success=_b(row.get("final_success", "")),
            patch_available=_b(row.get("patch_available", "")) is True,
            eval_log_available=_b(row.get("eval_log_available", "")) is True,
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import the B2 pilot sample into per-run directories (no ledger).",
    )
    p.add_argument("--sample-csv", required=True, type=Path)
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=None,
        help="Directory containing one <pilot_id>.json per pilot row. "
        "Required unless --verify-only is set.",
    )
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args(argv)


def _load_pilot_rows(path: Path) -> List[PilotRow]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [PilotRow.from_csv(r) for r in reader]


def _write_task_md(run_dir: Path, normalized: Dict[str, object], pilot: PilotRow) -> None:
    issue = normalized.get("issue_text") or ""
    if not isinstance(issue, str):
        issue = ""
    if issue:
        body = (
            f"# Task — {pilot.pilot_id} ({pilot.instance_id})\n\n"
            "Source: extracted from the leading environment turn of the upstream trajectory.\n\n"
            f"{issue}\n"
        )
    else:
        body = (
            f"# Task — {pilot.pilot_id} ({pilot.instance_id})\n\n"
            "(no task description in source trace — the leading non-system "
            "turn carried no issue text)\n"
        )
    (run_dir / "task.md").write_text(body, encoding="utf-8")


def _write_run_notes(run_dir: Path, pilot: PilotRow) -> None:
    body = RUN_NOTES_TEMPLATE.format(
        pilot_id=pilot.pilot_id,
        instance_id=pilot.instance_id,
        model_name=pilot.model_name,
        final_success=pilot.final_success,
    )
    (run_dir / "run_notes.md").write_text(body, encoding="utf-8")


def _write_source_metadata(run_dir: Path, pilot: PilotRow, normalized: Dict[str, object]) -> None:
    md = {
        "source": "swe_agent",
        "pilot_id": pilot.pilot_id,
        "instance_id": pilot.instance_id,
        "model_name": pilot.model_name,
        "final_success": pilot.final_success,
        "final_success_source": "source_label",
        "patch_available": pilot.patch_available,
        "eval_log_available": pilot.eval_log_available,
        "trajectory_length": normalized["trajectory_length"],
        "annotation_mode": "not_annotated",
    }
    (run_dir / "source_metadata.json").write_text(
        json.dumps(md, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_artifact_strings(run_dir: Path, raw: Dict[str, object]) -> None:
    patch = raw.get("generated_patch")
    eval_logs = raw.get("eval_logs")
    (run_dir / "final_diff.patch").write_text(patch if isinstance(patch, str) else "", encoding="utf-8")
    (run_dir / "test_output.txt").write_text(eval_logs if isinstance(eval_logs, str) else "", encoding="utf-8")


def import_one(pilot: PilotRow, raw_cache_dir: Path, runs_dir: Path) -> Path:
    cache_path = raw_cache_dir / f"{pilot.pilot_id}.json"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"raw cache missing for {pilot.pilot_id}: expected {cache_path}"
        )

    run_dir = runs_dir / pilot.pilot_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Byte-equivalent copy preserves the upstream JSON byte-for-byte.
    shutil.copyfile(cache_path, run_dir / "source_trace.json")

    raw_row = json.loads(cache_path.read_text(encoding="utf-8"))
    normalized = normalize_row(raw_row, source="swe_agent_nebius")

    (run_dir / "normalized_trace.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "trajectory_summary.md").write_text(
        render_summary(normalized), encoding="utf-8"
    )
    _write_task_md(run_dir, normalized, pilot)
    _write_run_notes(run_dir, pilot)
    _write_artifact_strings(run_dir, raw_row)
    _write_source_metadata(run_dir, pilot, normalized)
    return run_dir


def verify_run(run_dir: Path, pilot: PilotRow) -> List[str]:
    """Return a list of human-readable error strings; empty means OK."""
    errors: List[str] = []
    for name in PRE_ANNOTATION_ARTIFACTS:
        p = run_dir / name
        if not p.is_file():
            errors.append(f"{run_dir.name}: missing {name}")
            continue
        flag_col = NONEMPTY_IF_FLAG.get(name)
        if flag_col is not None:
            should_be_nonempty = getattr(pilot, flag_col)
            if should_be_nonempty and p.stat().st_size == 0:
                errors.append(f"{run_dir.name}: {name} is empty but {flag_col}=True")
        else:
            if p.stat().st_size == 0:
                errors.append(f"{run_dir.name}: {name} is empty")
    # Pre-annotation runs MUST NOT carry a ledger.jsonl.
    if (run_dir / "ledger.jsonl").is_file():
        errors.append(
            f"{run_dir.name}: unexpected ledger.jsonl — annotation has not been authorized yet"
        )
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    pilots = _load_pilot_rows(args.sample_csv)

    if args.verify_only:
        all_errors: List[str] = []
        for pilot in pilots:
            run_dir = args.runs_dir / pilot.pilot_id
            if not run_dir.is_dir():
                all_errors.append(f"{pilot.pilot_id}: run dir missing at {run_dir}")
                continue
            all_errors.extend(verify_run(run_dir, pilot))
        for e in all_errors:
            print(f"[import_swe_agent_trace] VERIFY: {e}", file=sys.stderr)
        if all_errors:
            return 1
        print(
            f"[import_swe_agent_trace] verify ok: {len(pilots)} run dirs",
            file=sys.stderr,
        )
        return 0

    if args.raw_cache_dir is None:
        print(
            "[import_swe_agent_trace] FATAL: --raw-cache-dir is required unless "
            "--verify-only is set.",
            file=sys.stderr,
        )
        return 2

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    all_errors = []
    for pilot in pilots:
        run_dir = import_one(pilot, args.raw_cache_dir, args.runs_dir)
        errors = verify_run(run_dir, pilot)
        for e in errors:
            print(f"[import_swe_agent_trace] VERIFY: {e}", file=sys.stderr)
        all_errors.extend(errors)
        print(
            f"[import_swe_agent_trace] imported {pilot.pilot_id} -> {run_dir}",
            file=sys.stderr,
        )

    if all_errors:
        return 1
    print(
        f"[import_swe_agent_trace] imported {len(pilots)} run dirs into {args.runs_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
