"""
Claim:
import_swe_agent_trace.py materializes one run dir per pilot row from
the B2 sample CSV and a raw-row cache. source_trace.json is BYTE-
equivalent to the cached upstream JSON. No ledger.jsonl is created.
Each run dir self-verifies, and --verify-only re-runs the check
across an existing runs-dir without needing the cache.

Plausible wrong implementations:
- source_trace.json written via json.dump(json.load(...)), changing
  whitespace / key order so bytes differ from the cache.
- Importer auto-creates ledger.jsonl (violates §0 of TASKS.md).
- Verify treats a missing artifact as a warning and still returns 0.
- Verify accepts an empty final_diff.patch even when patch_available
  is True for that pilot row.
- task.md always says "(no task description)" because the issue text
  extraction takes the wrong source.
- source_metadata.json marks annotation_mode as "annotated" or omits
  it; coerces final_success=None to False.
- Importer silently skips rows whose cache file is missing instead of
  failing.
- Re-running the importer changes artifact bytes (non-idempotent).
- --verify-only refuses to run without --raw-cache-dir, even though it
  doesn't need one.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_swe_agent_trace import PRE_ANNOTATION_ARTIFACTS, main


# ---------- fixtures ----------


def _write_pilot_csv(path: Path, rows):
    cols = [
        "pilot_id",
        "source_id",
        "instance_id",
        "model_name",
        "final_success",
        "trajectory_length",
        "patch_available",
        "eval_log_available",
        "raw_path_or_dataset_index",
        "selection_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _pilot_row(pilot_id, *, instance_id="r__r-1", final_success="True", patch="True", eval_log="True"):
    return {
        "pilot_id": pilot_id,
        "source_id": f"nebius:{instance_id}:swe-agent-llama-70b",
        "instance_id": instance_id,
        "model_name": "swe-agent-llama-70b",
        "final_success": final_success,
        "trajectory_length": "10",
        "patch_available": patch,
        "eval_log_available": eval_log,
        "raw_path_or_dataset_index": "nebius/SWE-agent-trajectories:train:0",
        "selection_reason": "primary_balanced_1_1",
    }


def _raw_row(*, instance_id="r__r-1", target=True, issue="ISSUE: bug", patch="diff --git a/x b/x\n", eval_logs="ok\n"):
    return {
        "instance_id": instance_id,
        "model_name": "swe-agent-llama-70b",
        "target": target,
        "trajectory": [
            {"role": "system", "text": None, "system_prompt": "SETTING: programmer", "mask": False, "cutoff_date": "01.01.2023"},
            {"role": "user", "text": issue, "system_prompt": None, "mask": False, "cutoff_date": None},
            {"role": "ai", "text": "step 1\n```\nls\n```", "system_prompt": None, "mask": True, "cutoff_date": None},
            {"role": "user", "text": "file_a\nfile_b\n", "system_prompt": None, "mask": False, "cutoff_date": None},
        ],
        "exit_status": "submitted (exit_context)",
        "generated_patch": patch,
        "eval_logs": eval_logs,
    }


def _write_cache(cache_dir: Path, pilot_id: str, raw: dict, *, indent=2) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(raw, indent=indent, ensure_ascii=False).encode("utf-8")
    (cache_dir / f"{pilot_id}.json").write_bytes(body)
    return body


def _setup_one(tmp_path: Path, *, pilot_id="swe_agent_pilot_s_01", raw_overrides=None, pilot_overrides=None):
    sample = tmp_path / "manifests" / "swe_agent_pilot_sample.csv"
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    pilot_kw = pilot_overrides or {}
    raw_kw = raw_overrides or {}
    pilot_row = _pilot_row(pilot_id, **pilot_kw)
    raw = _raw_row(**raw_kw)
    _write_pilot_csv(sample, [pilot_row])
    cache_bytes = _write_cache(cache, pilot_id, raw)
    return sample, cache, runs, pilot_row, raw, cache_bytes


# ---------- artifact materialization ----------


def test_import_creates_all_pre_annotation_artifacts(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    rc = main([
        "--sample-csv", str(sample),
        "--runs-dir", str(runs),
        "--raw-cache-dir", str(cache),
    ])
    assert rc == 0
    run_dir = runs / "swe_agent_pilot_s_01"
    for name in PRE_ANNOTATION_ARTIFACTS:
        assert (run_dir / name).is_file(), f"missing {name}"
    assert not (run_dir / "ledger.jsonl").exists()


def test_source_trace_is_byte_equivalent_to_cache(tmp_path):
    sample, cache, runs, _pilot_row, _raw, cache_bytes = _setup_one(tmp_path)
    main([
        "--sample-csv", str(sample),
        "--runs-dir", str(runs),
        "--raw-cache-dir", str(cache),
    ])
    written = (runs / "swe_agent_pilot_s_01" / "source_trace.json").read_bytes()
    assert written == cache_bytes


def test_source_trace_byte_identity_holds_for_unusual_json_formatting(tmp_path):
    # If a future cache decides to use no indent / sorted keys / a
    # trailing newline, the importer must still copy bytes verbatim.
    sample, cache, runs, _pilot_row, _raw, _ = _setup_one(
        tmp_path, raw_overrides={"issue": "ISSUE: weird formatting"}
    )
    quirky = b'{"instance_id":"r__r-1","model_name":"m","target":true,"trajectory":[],"exit_status":"x","generated_patch":"d","eval_logs":"e"}\n'
    (cache / "swe_agent_pilot_s_01.json").write_bytes(quirky)
    main([
        "--sample-csv", str(sample),
        "--runs-dir", str(runs),
        "--raw-cache-dir", str(cache),
    ])
    assert (runs / "swe_agent_pilot_s_01" / "source_trace.json").read_bytes() == quirky


def test_task_md_contains_issue_text_when_present(tmp_path):
    sample, cache, runs, *_ = _setup_one(
        tmp_path, raw_overrides={"issue": "ISSUE: TypeError on memset table render"}
    )
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    body = (runs / "swe_agent_pilot_s_01" / "task.md").read_text(encoding="utf-8")
    assert "TypeError on memset table render" in body
    assert "(no task description" not in body


def test_task_md_explicitly_notes_absence_when_no_issue_text(tmp_path):
    sample = tmp_path / "p.csv"
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    _write_pilot_csv(sample, [_pilot_row("swe_agent_pilot_s_01")])
    # Trajectory with only a system entry → no issue text.
    raw = {
        "instance_id": "r__r-1",
        "model_name": "m",
        "target": True,
        "trajectory": [{"role": "system", "text": None, "system_prompt": "sp"}],
        "exit_status": "x",
        "generated_patch": "p",
        "eval_logs": "e",
    }
    _write_cache(cache, "swe_agent_pilot_s_01", raw)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    body = (runs / "swe_agent_pilot_s_01" / "task.md").read_text(encoding="utf-8")
    assert "no task description in source trace" in body


def test_run_notes_is_initialized_with_todo_sections(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    body = (runs / "swe_agent_pilot_s_01" / "run_notes.md").read_text(encoding="utf-8")
    assert "TODO" in body
    assert "annotation" in body.lower()


def test_source_metadata_marks_annotation_mode_not_annotated(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    md = json.loads((runs / "swe_agent_pilot_s_01" / "source_metadata.json").read_text(encoding="utf-8"))
    assert md["annotation_mode"] == "not_annotated"
    assert md["source"] == "swe_agent"
    assert md["final_success_source"] == "source_label"


def test_source_metadata_preserves_final_success_distinction_between_false_and_missing(tmp_path):
    # final_success="False" in CSV → metadata records explicit False.
    sample, cache, runs, *_ = _setup_one(
        tmp_path,
        pilot_overrides={"final_success": "False"},
        raw_overrides={"target": False},
    )
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    md = json.loads((runs / "swe_agent_pilot_s_01" / "source_metadata.json").read_text(encoding="utf-8"))
    assert md["final_success"] is False  # not None, not "False" string


def test_idempotency_running_twice_produces_byte_identical_artifacts(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    run_dir = runs / "swe_agent_pilot_s_01"
    snapshot = {name: (run_dir / name).read_bytes() for name in PRE_ANNOTATION_ARTIFACTS}
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    for name, expected in snapshot.items():
        assert (run_dir / name).read_bytes() == expected, name


# ---------- failure paths ----------


def test_missing_raw_cache_file_raises_with_pilot_id_in_message(tmp_path):
    sample = tmp_path / "p.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    runs = tmp_path / "runs"
    _write_pilot_csv(sample, [_pilot_row("swe_agent_pilot_s_01")])
    # Cache exists but has no <pilot_id>.json.
    with pytest.raises(FileNotFoundError) as exc_info:
        main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    assert "swe_agent_pilot_s_01" in str(exc_info.value)


def test_main_without_verify_only_requires_raw_cache_dir(tmp_path, capsys):
    sample = tmp_path / "p.csv"
    runs = tmp_path / "runs"
    _write_pilot_csv(sample, [_pilot_row("swe_agent_pilot_s_01")])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "raw-cache-dir" in err


# ---------- verify-only ----------


def test_verify_only_passes_on_complete_run_dir(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 0


def test_verify_only_does_not_require_raw_cache_dir(tmp_path):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    # No --raw-cache-dir on the verify call.
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 0


def test_verify_only_flags_missing_artifact_naming_run_dir_and_artifact(tmp_path, capsys):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    (runs / "swe_agent_pilot_s_01" / "trajectory_summary.md").unlink()
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "swe_agent_pilot_s_01" in err
    assert "trajectory_summary.md" in err


def test_verify_only_rejects_empty_final_diff_patch_when_flag_is_true(tmp_path, capsys):
    sample, cache, runs, *_ = _setup_one(
        tmp_path,
        raw_overrides={"patch": ""},  # empty patch despite patch_available=True
    )
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "final_diff.patch" in err
    assert "patch_available" in err


def test_verify_only_rejects_unexpected_ledger_jsonl(tmp_path, capsys):
    sample, cache, runs, *_ = _setup_one(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    (runs / "swe_agent_pilot_s_01" / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unexpected ledger.jsonl" in err


def test_verify_only_flags_missing_run_dir_for_pilot_row(tmp_path, capsys):
    sample = tmp_path / "p.csv"
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_pilot_csv(sample, [_pilot_row("swe_agent_pilot_s_01"), _pilot_row("swe_agent_pilot_f_01", instance_id="r__r-2")])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "swe_agent_pilot_s_01" in err and "swe_agent_pilot_f_01" in err
