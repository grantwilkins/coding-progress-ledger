"""Hermes pilot importer tests (HP2)."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_hermes_trace import PRE_ANNOTATION_ARTIFACTS, main


def _write_pilot_csv(path: Path, rows):
    cols = ["pilot_id", "source_id", "instance_id", "model_name", "category",
            "subcategory", "trajectory_length", "raw_path_or_dataset_index", "selection_reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _pilot_row(pid, *, instance_id="abc", model_name="kimi", idx=0):
    return {
        "pilot_id": pid,
        "source_id": f"hermes:{model_name}:{instance_id}",
        "instance_id": instance_id,
        "model_name": model_name,
        "category": "Terminal & Coding",
        "subcategory": "x",
        "trajectory_length": "4",
        "raw_path_or_dataset_index": f"lambda/hermes-agent-reasoning-traces:{model_name}:train:{idx}",
        "selection_reason": "primary",
    }


def _raw_row(*, id="abc"):
    return {
        "id": id,
        "task": "do x",
        "tools": "[]",
        "category": "Terminal & Coding",
        "subcategory": "x",
        "conversations": [
            {"from": "system", "value": "sp"},
            {"from": "human", "value": "issue body"},
            {"from": "gpt", "value": "<think>r</think>\n<tool_call>\n{\"name\": \"ls\", \"arguments\": {}}\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n{\"tool_call_id\": \"a\", \"name\": \"ls\", \"content\": \"ok\"}\n</tool_response>"},
        ],
    }


def _setup(tmp_path: Path, pid="hermes_pilot_01"):
    sample = tmp_path / "sample.csv"
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    pilot = _pilot_row(pid)
    raw = _raw_row(id="abc")
    _write_pilot_csv(sample, [pilot])
    cache.mkdir()
    (cache / f"{pid}.json").write_bytes(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    return sample, cache, runs


def test_import_creates_all_eight_artifacts(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    assert rc == 0
    rd = runs / "hermes_pilot_01"
    for name in PRE_ANNOTATION_ARTIFACTS:
        assert (rd / name).is_file(), f"missing {name}"
    assert len(PRE_ANNOTATION_ARTIFACTS) == 8
    assert not (rd / "ledger.jsonl").exists()


def test_source_metadata_final_success_is_null(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    md = json.loads((runs / "hermes_pilot_01" / "source_metadata.json").read_text(encoding="utf-8"))
    assert md["final_success"] is None
    assert md["source"] == "hermes_agent_reasoning"
    assert md["model_name"] == "kimi"
    assert md["category"] == "Terminal & Coding"


def test_placeholder_files_have_explanatory_content(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    rd = runs / "hermes_pilot_01"
    patch_body = (rd / "final_diff.patch").read_text(encoding="utf-8")
    test_body = (rd / "test_output.txt").read_text(encoding="utf-8")
    assert "no final_diff" in patch_body.lower()
    assert "no test_output" in test_body.lower()


def test_source_trace_byte_equivalent_to_cache(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    cache_bytes = (cache / "hermes_pilot_01.json").read_bytes()
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    written = (runs / "hermes_pilot_01" / "source_trace.json").read_bytes()
    assert written == cache_bytes


def test_task_md_contains_issue_text(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    body = (runs / "hermes_pilot_01" / "task.md").read_text(encoding="utf-8")
    assert "issue body" in body


def test_verify_only_passes_on_complete_run(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 0


def test_verify_only_flags_missing_artifact(tmp_path, capsys):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    (runs / "hermes_pilot_01" / "trajectory_summary.md").unlink()
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    assert "trajectory_summary.md" in capsys.readouterr().err


def test_verify_only_rejects_unexpected_ledger(tmp_path, capsys):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    (runs / "hermes_pilot_01" / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 1
    assert "unexpected ledger" in capsys.readouterr().err


def test_verify_only_does_not_require_raw_cache_dir(tmp_path):
    sample, cache, runs = _setup(tmp_path)
    main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--verify-only"])
    assert rc == 0


def test_main_without_verify_requires_cache(tmp_path):
    sample = tmp_path / "s.csv"
    runs = tmp_path / "runs"
    _write_pilot_csv(sample, [_pilot_row("hermes_pilot_01")])
    rc = main(["--sample-csv", str(sample), "--runs-dir", str(runs)])
    assert rc == 2


def test_missing_cache_raises(tmp_path):
    sample = tmp_path / "s.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    runs = tmp_path / "runs"
    _write_pilot_csv(sample, [_pilot_row("hermes_pilot_01")])
    with pytest.raises(FileNotFoundError) as exc_info:
        main(["--sample-csv", str(sample), "--runs-dir", str(runs), "--raw-cache-dir", str(cache)])
    assert "hermes_pilot_01" in str(exc_info.value)
