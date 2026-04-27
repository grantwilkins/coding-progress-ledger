import csv
import hashlib
import json
import subprocess
import sys

from ledger_progress import LedgerSession, SubtaskCategory, score
from ledger_progress.run_manager import main as run_manager_main
from ledger_progress.serialization import from_jsonl


def test_check_run_flags_missing_artifacts_after_init(tmp_path, capsys):
    run_dir = tmp_path / "run_a"

    assert run_manager_main(["init-run", str(run_dir)]) == 0
    assert (run_dir / "task.md").exists()
    assert (run_dir / "run_notes.md").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert not (run_dir / "ledger.jsonl").exists()
    assert not (run_dir / "progress.csv").exists()
    assert (run_dir / "ledger.jsonl.README").exists()
    assert (run_dir / "progress.csv.README").exists()
    assert (run_dir / "test_result.json.README").exists()

    assert run_manager_main(["check-run", str(run_dir)]) == 1

    output = capsys.readouterr().out
    assert "missing: ledger.jsonl" in output
    assert "missing: progress.csv" in output
    assert "missing: final_diff.patch" in output
    assert "missing: test_output.txt" in output
    assert "missing: ledger.jsonl.README" not in output


def test_export_run_requires_real_ledger_jsonl(tmp_path):
    run_dir = tmp_path / "run_without_ledger"
    run_dir.mkdir()

    assert run_manager_main(["export-run", str(run_dir)]) == 1


def test_export_run_regenerates_outputs_and_preserves_ledger_hash(tmp_path):
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    session = LedgerSession("Export run")
    product = session.add("Patch behavior", step=1, category=SubtaskCategory.PRODUCT)
    validation = session.add("Run validation", step=2, category=SubtaskCategory.VALIDATION)
    session.complete(product, "final_diff.patch shows patch", step=3)
    session.complete(validation, "test_output.txt shows pytest passed", step=4)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    before = _sha256(run_dir / "ledger.jsonl")
    before_score = score(from_jsonl(str(run_dir / "ledger.jsonl")))

    assert run_manager_main(["export-run", str(run_dir)]) == 0

    after = _sha256(run_dir / "ledger.jsonl")
    after_score = score(from_jsonl(str(run_dir / "ledger.jsonl")))
    summary = json.loads((run_dir / "summary_by_category.json").read_text())
    progress_rows = list(csv.DictReader((run_dir / "progress.csv").open()))

    assert after == before
    assert after_score == before_score
    assert (run_dir / "progress_by_category.csv").exists()
    assert progress_rows[-1]["progress"] == "1.0"
    assert summary["source_ledger_sha256"] == before
    assert summary["generator"] == "ledger-run export-run"
    assert isinstance(summary["generated_at"], str)
    assert summary["final_coding_progress"] == 1.0


def test_summarize_run_warns_when_summary_was_generated_from_different_ledger(tmp_path, capsys):
    run_dir = tmp_path / "run_c"
    run_dir.mkdir()
    session = LedgerSession("Stale summary")
    product = session.add("Patch behavior", step=1)
    session.complete(product, "final_diff.patch shows patch", step=2)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    assert run_manager_main(["export-run", str(run_dir)]) == 0

    validation = session.add("Run validation", step=3, category=SubtaskCategory.VALIDATION)
    session.complete(validation, "test_output.txt shows pytest passed", step=4)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))

    assert run_manager_main(["summarize-run", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "warning: summary was generated from a different ledger.jsonl" in output


def test_summarize_run_prints_summary_values_and_does_not_mutate_ledger(tmp_path, capsys):
    run_dir = tmp_path / "run_summary"
    run_dir.mkdir()
    session = LedgerSession("Summarize run")
    product = session.add("Patch behavior", step=1)
    session.complete(product, "final_diff.patch shows patch", step=2)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    assert run_manager_main(["export-run", str(run_dir)]) == 0
    before = _sha256(run_dir / "ledger.jsonl")

    assert run_manager_main(["summarize-run", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert _sha256(run_dir / "ledger.jsonl") == before
    assert "final_coding_progress: 1.0" in output
    assert "final_overall_progress: 1.0" in output
    assert "weak_completion_evidence_count:" in output
    assert "missing artifacts:" in output


def test_check_run_does_not_mutate_ledger_jsonl(tmp_path):
    run_dir = tmp_path / "run_check"
    run_dir.mkdir()
    session = LedgerSession("Check run")
    product = session.add("Patch behavior", step=1)
    session.complete(product, "final_diff.patch shows patch", step=2)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    before = _sha256(run_dir / "ledger.jsonl")

    assert run_manager_main(["check-run", str(run_dir)]) == 1

    assert _sha256(run_dir / "ledger.jsonl") == before


def test_capture_diff_writes_patch_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run_dir = repo / "run"
    repo.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init"], check=True, capture_output=True)
    path = repo / "tracked.txt"
    path.write_text("before\n")
    subprocess.run(["git", "add", "tracked.txt"], check=True)
    path.write_text("after\n")

    assert run_manager_main(["capture-diff", str(run_dir)]) == 0

    patch = (run_dir / "final_diff.patch").read_text()
    assert "diff --git a/tracked.txt b/tracked.txt" in patch
    assert "+after" in patch


def test_capture_diff_does_not_require_or_mutate_ledger_jsonl(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run_dir = repo / "run"
    repo.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init"], check=True, capture_output=True)
    path = repo / "tracked.txt"
    path.write_text("before\n")
    subprocess.run(["git", "add", "tracked.txt"], check=True)
    path.write_text("after\n")

    assert run_manager_main(["capture-diff", str(run_dir)]) == 0
    assert not (run_dir / "ledger.jsonl").exists()

    session = LedgerSession("Capture diff")
    product = session.add("Patch behavior", step=1)
    session.complete(product, "final_diff.patch shows patch", step=2)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    before = _sha256(run_dir / "ledger.jsonl")

    assert run_manager_main(["capture-diff", str(run_dir)]) == 0

    assert _sha256(run_dir / "ledger.jsonl") == before


def test_capture_tests_records_output_and_result_json(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_d"
    run_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr); raise SystemExit(7)",
    ]

    assert run_manager_main(["capture-tests", str(run_dir), "--", *command]) == 7

    output = (run_dir / "test_output.txt").read_text()
    result = json.loads((run_dir / "test_result.json").read_text())
    assert "hello stdout" in output
    assert "hello stderr" in output
    assert result["command"] == command
    assert result["exit_code"] == 7
    assert result["success"] is False
    assert result["cwd"] == str(tmp_path)
    assert isinstance(result["duration_seconds"], float)


def test_capture_tests_does_not_mutate_ledger_or_overwrite_explicit_success(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_capture"
    run_dir.mkdir()
    session = LedgerSession("Capture tests")
    product = session.add("Patch behavior", step=1)
    session.complete(product, "final_diff.patch shows patch", step=2)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "final_success": False,
        "final_success_source": "hidden_check_tests",
        "generated_artifacts": [],
    }))
    before = _sha256(run_dir / "ledger.jsonl")
    monkeypatch.chdir(tmp_path)

    assert run_manager_main(["capture-tests", str(run_dir), "--", sys.executable, "-c", "print('ok')"]) == 0

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert _sha256(run_dir / "ledger.jsonl") == before
    assert manifest["final_success"] is False
    assert manifest["final_success_source"] == "hidden_check_tests"


def test_summarize_run_final_success_priority_preserves_explicit_failure(tmp_path, capsys):
    run_dir = tmp_path / "run_e"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "final_success": False,
        "final_success_source": "hidden_check_tests",
        "generated_artifacts": [],
    }))
    (run_dir / "summary_by_category.json").write_text(json.dumps({
        "final_coding_progress": 1.0,
        "final_overall_progress": 1.0,
        "coding_nonmonotonic": False,
        "overall_nonmonotonic": False,
        "weak_completion_evidence_count": 0,
        "final_success": True,
        "final_success_source": "summary.final_success",
    }))
    (run_dir / "test_result.json").write_text(json.dumps({"success": True}))

    assert run_manager_main(["summarize-run", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "final_success: False" in output
    assert "final_success_source: hidden_check_tests" in output


def test_summarize_run_uses_test_result_when_no_explicit_success_metadata(tmp_path, capsys):
    run_dir = tmp_path / "run_f"
    run_dir.mkdir()
    (run_dir / "summary_by_category.json").write_text(json.dumps({
        "final_coding_progress": 1.0,
        "final_overall_progress": 1.0,
        "coding_nonmonotonic": False,
        "overall_nonmonotonic": False,
        "weak_completion_evidence_count": 0,
        "final_success": False,
        "final_success_source": "inferred_from_test_output",
    }))
    (run_dir / "test_result.json").write_text(json.dumps({"success": True}))

    assert run_manager_main(["summarize-run", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "final_success: True" in output
    assert "final_success_source: test_result.success" in output


def test_summarize_run_reports_unknown_success_without_metadata(tmp_path, capsys):
    run_dir = tmp_path / "run_unknown_success"
    run_dir.mkdir()
    (run_dir / "summary_by_category.json").write_text(json.dumps({
        "final_coding_progress": 0.5,
        "final_overall_progress": 0.5,
        "coding_nonmonotonic": False,
        "overall_nonmonotonic": False,
        "weak_completion_evidence_count": 0,
    }))

    assert run_manager_main(["summarize-run", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "final_success: unknown" in output
    assert "final_success_source: unknown" in output


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
