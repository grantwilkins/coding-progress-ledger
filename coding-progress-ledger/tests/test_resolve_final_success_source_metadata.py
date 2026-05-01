"""
Claim:
resolve_final_success and final_success_from_metadata must short-
circuit to source_metadata.json:final_success when an importer has
declared it authoritatively (final_success_source == "source_label").
The keyword scan over test_output.txt was tuned for toy/live pytest
output and misclassifies SWE-bench eval logs that interleave
"passed", "error", and "failed" tokens.

Plausible wrong implementations:
- Heuristic test_output.txt scan runs before source_metadata.json,
  producing inferred_from_test_output even when an authoritative
  upstream label is on disk.
- Heuristic scan flags any text containing "error" as failure even
  when the same text contains an unambiguous "passed" marker.
- source_metadata.json with final_success_source != "source_label"
  is silently treated as authoritative anyway.
- A run dir with source_metadata.json:final_success = false but
  test_output.txt missing returns None instead of False.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress.run_manager import resolve_final_success
from scripts.rescore_suite_by_category import final_success_from_metadata


def _write_md(run_dir: Path, **kw):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source_metadata.json").write_text(json.dumps(kw), encoding="utf-8")


def _write_test_output(run_dir: Path, text: str):
    (run_dir / "test_output.txt").write_text(text, encoding="utf-8")


# ---------- the bug case (SWE-bench eval log fooling the keyword scan) ----------


def test_swe_bench_style_eval_log_with_error_token_does_not_override_upstream_true(tmp_path):
    """C3 wrote final_success=True from upstream `target`; eval log mentions
    'error' (e.g. inside a stderr block). Heuristic alone returns False;
    correct behavior returns True via source_metadata."""
    _write_md(tmp_path, source="swe_agent", final_success=True, final_success_source="source_label")
    _write_test_output(tmp_path, "============= test session starts =============\n"
                                  "ran 12 tests, 12 passed\n"
                                  "stderr: numpy.exceptions.VisibleDeprecationWarning: ...\n"
                                  "1 deprecation error suppressed\n")
    fs, source = resolve_final_success(tmp_path)
    assert fs is True
    assert source == "source_metadata.target"

    fs2, source2 = final_success_from_metadata(tmp_path, summary={})
    assert fs2 is True
    assert source2 == "source_metadata.target"


def test_authoritative_upstream_false_overrides_misleading_passed_text(tmp_path):
    _write_md(tmp_path, source="swe_agent", final_success=False, final_success_source="source_label")
    _write_test_output(tmp_path, "build passed; test session starts\n"
                                  "ok: 5 cases ok\n"
                                  "(but eval reports patch did not resolve issue)\n")
    fs, source = resolve_final_success(tmp_path)
    assert fs is False
    assert source == "source_metadata.target"


# ---------- guard rails: only "source_label" is treated as authoritative ----------


def test_source_metadata_with_other_source_string_is_ignored(tmp_path):
    """A future source might want to declare final_success without claiming
    authoritative-by-construction. The check must require the exact source
    string, otherwise any heuristic-tagged metadata gets promoted."""
    _write_md(tmp_path, source="swe_agent", final_success=True, final_success_source="some_other_heuristic")
    _write_test_output(tmp_path, "test session starts\nfailed\n")
    # Tested at the heuristic entry point (final_success_from_metadata)
    # because resolve_final_success only invokes the heuristic via
    # summary_by_category.json which doesn't exist here.
    fs, source = final_success_from_metadata(tmp_path, summary={})
    # source_metadata is ignored; falls through to test_output.txt heuristic.
    assert fs is False
    assert source == "inferred_from_test_output"


def test_source_metadata_without_final_success_field_is_ignored(tmp_path):
    """A C3-style importer may declare final_success_source: "source_label"
    but with a missing/None value (when the upstream label is unavailable).
    The check must require the bool be present."""
    _write_md(tmp_path, source="swe_agent", final_success=None, final_success_source="source_label")
    _write_test_output(tmp_path, "test session starts\nall tests passed\n")
    fs, source = final_success_from_metadata(tmp_path, summary={})
    # source_metadata's None is not authoritative; fall through to heuristic.
    assert fs is True
    assert source == "inferred_from_test_output"


# ---------- ordering: source_metadata wins over heuristic, but loses to manifest ----------


def test_run_manifest_final_success_still_wins_over_source_metadata(tmp_path):
    """run_manifest.json is the human-curated label; if it's set, it must
    win even if source_metadata also claims a value. Otherwise an importer
    silently overrides a deliberate manual override."""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"final_success": False, "final_success_source": "run_manifest.final_success"}),
        encoding="utf-8",
    )
    _write_md(tmp_path, source="swe_agent", final_success=True, final_success_source="source_label")
    fs, source = resolve_final_success(tmp_path)
    assert fs is False
    assert source == "run_manifest.final_success"


# ---------- absent source_metadata: heuristic still behaves as before ----------


def test_absent_source_metadata_falls_back_to_heuristic_unchanged(tmp_path):
    """The fix must not change behavior for runs without source_metadata
    (toy/live runs). The heuristic at final_success_from_metadata
    should still fire on test_output.txt."""
    _write_test_output(tmp_path, "test session starts\nall passed\nok: 3\n")
    fs, source = final_success_from_metadata(tmp_path, summary={})
    assert fs is True
    assert source == "inferred_from_test_output"
