"""
Claim:
The progress label audit measures when opened-unit progress first reaches the
final opened-unit count, preserves dense row-index accounting separately from
source step numbers, and emits enough selected trace evidence to classify long
tails manually.

Plausible wrong implementations:
- Use non-dense step numbers as row counts, hiding skipped or mixed-source step
  conventions.
- Count the opened-100% row itself as remaining tail work.
- Miss category transitions after opened progress reaches 100%.
- Let artifact-final traces leak into the non-artifact ranking.
- Plot closed-unit progress without the terminal close performed by finalize().
- Emit snippets without the command/tool or observation evidence needed for the
  classifier audit.
"""

import csv
from pathlib import Path

from observation_channel.cli import main
from observation_channel.io import write_jsonl
from observation_channel.progress_label_audit import evaluate_progress_label_audit


def _write_audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    traces = tmp_path / "traces.csv"
    turns = tmp_path / "turns.csv"
    raw_dir = tmp_path / "raw"
    traces.write_text(
        "trace_key,source,raw_row_index,final_total,final_done,total_turns,exit_status,parse_error\n"
        "swe-agent:000000:case,swe-agent,0,2,2,10,submitted,\n"
        "swe-agent:000001:artifact,swe-agent,1,1,1,9,submitted,\n",
        encoding="utf-8",
    )
    turns.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,had_stuck_episode,"
        "recent_error_bucket,recent_error_rate,touched_source,investigation_ratio_bucket,investigation_ratio,kind,tool\n"
        "swe-agent:000000:case,swe-agent,case,1,1,0,PRODUCT,1,False,clean,0.0,False,low,0.0,action,bash\n"
        "swe-agent:000000:case,swe-agent,case,4,2,1,VALIDATION,1,False,clean,0.0,False,low,0.0,action,python\n"
        "swe-agent:000000:case,swe-agent,case,7,2,1,VALIDATION,4,False,clean,0.0,False,low,0.0,observation,\n"
        "swe-agent:000000:case,swe-agent,case,10,2,1,INVESTIGATION,1,False,clean,0.0,False,low,0.0,action,rg\n"
        "swe-agent:000001:artifact,swe-agent,artifact,2,1,0,ARTIFACT,1,False,clean,0.0,False,low,0.0,action,submit\n"
        "swe-agent:000001:artifact,swe-agent,artifact,9,1,0,ARTIFACT,8,False,clean,0.0,False,low,0.0,observation,\n",
        encoding="utf-8",
    )
    write_jsonl(
        raw_dir / "swe-agent" / "train.jsonl",
        [
            {
                "instance_id": "case",
                "trajectory": [
                    {"role": "assistant", "content": "cat <<'EOF' > src/app.py\nx=1\nEOF"},
                    {"role": "user", "observation": "wrote file"},
                    {"role": "assistant", "content": "python reproduce.py"},
                    {"role": "user", "observation": "returncode=1 failing reproduction output"},
                    {"role": "assistant", "content": "rg target"},
                    {"role": "user", "observation": "target found"},
                ],
            },
            {
                "instance_id": "artifact",
                "trajectory": [
                    {"role": "assistant", "content": "submit"},
                    {"role": "user", "observation": "done"},
                ],
            },
        ],
    )
    return turns, traces, raw_dir


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_tail_progress_audit_uses_step_and_row_index_accounting(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    result = evaluate_progress_label_audit(turns, traces, raw_dir, report, top_n=2, category_n=1, plot_limit=1)
    rows = {row["trace_key"]: row for row in _rows(report / "tail_progress_audit.csv")}
    case = rows["swe-agent:000000:case"]

    assert result["traces"] == 2
    assert case["first_step_where_opened_units_reach_final_count"] == "4"
    assert case["first_row_index_where_opened_units_reach_final_count"] == "2"
    assert case["steps_remaining_after_that_point"] == "6"
    assert case["rows_remaining_after_that_point"] == "2"
    assert float(case["fraction_of_trace_remaining_after_that_point"]) == 0.6
    assert case["age_of_final_unit_at_end"] == "1"


def test_tail_category_metrics_distinguish_single_long_unit_from_phase_changes(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    evaluate_progress_label_audit(turns, traces, raw_dir, report, top_n=2, category_n=1, plot_limit=1)
    case = {row["trace_key"]: row for row in _rows(report / "tail_progress_audit.csv")}["swe-agent:000000:case"]
    artifact = {row["trace_key"]: row for row in _rows(report / "tail_progress_audit.csv")}["swe-agent:000001:artifact"]

    assert case["tail_steps_after_opened_100pct_category_change_count"] == "1"
    assert float(case["tail_steps_after_opened_100pct_same_category_fraction"]) == 0.5
    assert artifact["tail_steps_after_opened_100pct_category_change_count"] == "0"
    assert float(artifact["tail_steps_after_opened_100pct_same_category_fraction"]) == 1.0


def test_non_artifact_ranking_and_classifier_samples_are_separate(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    evaluate_progress_label_audit(turns, traces, raw_dir, report, top_n=2, category_n=1, plot_limit=1)
    non_artifact_keys = {row["trace_key"] for row in _rows(report / "tail_progress_audit_non_artifact.csv")}
    sample_categories = {row["category_of_final_unit"] for row in _rows(report / "classifier_final_unit_samples.csv")}

    assert "swe-agent:000001:artifact" not in non_artifact_keys
    assert {"ARTIFACT", "VALIDATION"} <= sample_categories


def test_progress_curves_add_closed_unit_terminal_point(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    evaluate_progress_label_audit(turns, traces, raw_dir, report, top_n=2, category_n=0, plot_limit=1)
    curve_rows = [row for row in _rows(report / "progress_curve_traces.csv") if row["trace_key"] == "swe-agent:000000:case"]
    terminal = [row for row in curve_rows if row["point_type"] == "closed_terminal"]

    assert terminal
    assert terminal[0]["row_index"] == "4"
    assert terminal[0]["closed_unit_progress"] == "1.0"
    assert any(row["point_type"] == "observed" and row["closed_unit_progress"] == "0.5" for row in curve_rows)


def test_inspection_snippets_include_command_tool_and_observation_evidence(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    evaluate_progress_label_audit(turns, traces, raw_dir, report, top_n=2, category_n=0, plot_limit=1)
    snippets = _rows(report / "inspection_snippets.csv")

    assert any(row["tool"] == "python" and row["command"] == "python reproduce.py" for row in snippets)
    assert any("failing reproduction" in row["observation_snippet"] for row in snippets)


def test_progress_label_audit_cli_writes_report(tmp_path: Path) -> None:
    turns, traces, raw_dir = _write_audit_fixture(tmp_path)
    report = tmp_path / "report"

    rc = main(
        [
            "progress-label-audit",
            "--turns-csv",
            str(turns),
            "--traces-csv",
            str(traces),
            "--raw-dir",
            str(raw_dir),
            "--report-dir",
            str(report),
            "--top-n",
            "1",
            "--category-n",
            "0",
            "--plot-limit",
            "1",
        ]
    )

    assert rc == 0
    assert "Does opened-unit progress reach 100%" in (report / "README.md").read_text(encoding="utf-8")
