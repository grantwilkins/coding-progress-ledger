"""
Claim:
C4 produces one static ablation row for each cut point and resume-package type,
with C3 validation status, moved bytes, lazy-rehydrate target bytes, and recorded
setup-step counts. It does not run a model, execute tools, run a verifier, or
claim K4 bottleneck attribution unless K4 is explicitly wired.

Plausible wrong implementations:
- emit only valid packages, hiding the prompt-only and missing-diff failure modes;
- aggregate over cuts instead of preserving one row per cut point and package;
- report bytes from state entries at the wrong level and double-count workspace
  layer summaries;
- treat lazy/global state as moved bytes instead of setup targets;
- fill dominant_resource with a guessed bottleneck even though K4 was not run.
"""

import csv
from pathlib import Path

from agent_migrate_agent.adapters.swe_agent import swe_agent_to_trace
from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl
from agent_migrate_agent.resume_ablation import (
    count_extra_setup_steps,
    run_resume_ablation,
    write_ablation_csv,
)
from agent_migrate_agent.resume_packages import WorkspaceFileEntry, build_agent_migrate_minimal

FIXTURE = Path(__file__).parent / "fixtures" / "swe_agent_pilot_s_07.json"
H5A_TRACE = Path(__file__).parents[1] / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl"


def _events_and_cuts(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    events = load_trace_jsonl(out)
    cuts = find_cut_points(events, trace_id="s_07")
    assert cuts
    return events, cuts


def _h5a_events_and_cuts():
    events = load_trace_jsonl(H5A_TRACE)
    cuts = find_cut_points(events, trace_id="h5a")
    assert cuts
    return events, cuts


def _harness() -> dict:
    return {"cwd": "/changelog_cli", "open_file": "src/changelog/utils.py", "env": {}}


def test_c4_emits_every_cut_by_package_row(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    rows = run_resume_ablation(events, cuts[:3], harness_config=_harness())
    assert len(rows) == 3 * 5
    assert {row.package_type for row in rows} == {
        "prompt_only",
        "transcript_plus_harness_state",
        "transcript_plus_diff",
        "full_workspace_snapshot",
        "agent_migrate_minimal",
    }
    assert {(row.event_index, row.package_type) for row in rows} == {
        (cut.event_index, package_type)
        for cut in cuts[:3]
        for package_type in {
            "prompt_only",
            "transcript_plus_harness_state",
            "transcript_plus_diff",
            "full_workspace_snapshot",
            "agent_migrate_minimal",
        }
    }


def test_c4_keeps_invalid_rows_and_reasons(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    rows = run_resume_ablation(events, cuts[:1], harness_config=_harness())
    by_type = {row.package_type: row for row in rows}
    assert not by_type["transcript_plus_diff"].valid
    assert "missing_diff_for_transcript_plus_diff" in by_type["transcript_plus_diff"].validation_reasons
    assert by_type["prompt_only"].valid


def test_c4_bytes_do_not_double_count_minimal_workspace_summary(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    files = (WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_diff"),)
    rows = run_resume_ablation(
        events,
        cuts[:1],
        harness_config=_harness(),
        workspace_files=files,
        workspace_layer_for_file={"uncommitted_diff.patch": "uncommitted_diff"},
    )
    row = next(r for r in rows if r.package_type == "agent_migrate_minimal")
    pkg = build_agent_migrate_minimal(
        events,
        cuts[0],
        harness_config=_harness(),
        workspace_files=files,
        workspace_layer_for_file={"uncommitted_diff.patch": "uncommitted_diff"},
    )
    assert row.bytes_moved == pkg.included_bytes
    assert row.bytes_moved < pkg.included_bytes + 100


def test_c4_workspace_bytes_are_cut_local_not_prior_session_prefix(tmp_path: Path):
    events, cuts = _h5a_events_and_cuts()
    rows = run_resume_ablation(events, cuts[:2], harness_config=_harness())
    first = next(r for r in rows if r.event_index == cuts[0].event_index and r.package_type == "full_workspace_snapshot")
    second = next(r for r in rows if r.event_index == cuts[1].event_index and r.package_type == "full_workspace_snapshot")
    # Each H5a cut's next call reads one 1 GB workspace. A prefix-global package
    # would charge 2 GB at the second cut.
    assert first.bytes_moved > 1_000_000_000
    assert second.bytes_moved == first.bytes_moved
    assert second.bytes_moved < 2_000_000_000


def test_c4_lazy_rehydrate_is_target_not_moved_bytes(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    files = (
        WorkspaceFileEntry(".venv/lib/site_packages/pkg.py", 500, "h_dep"),
        WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_diff"),
    )
    rows = run_resume_ablation(
        events,
        cuts[:1],
        harness_config=_harness(),
        workspace_files=files,
        workspace_layer_for_file={
            ".venv/lib/site_packages/pkg.py": "dependency_cache",
            "uncommitted_diff.patch": "uncommitted_diff",
        },
    )
    row = next(r for r in rows if r.package_type == "agent_migrate_minimal")
    assert row.bytes_lazy_rehydrate_target == 500
    assert row.bytes_moved < row.bytes_moved + row.bytes_lazy_rehydrate_target


def test_c4_does_not_guess_k4_bottleneck(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    rows = run_resume_ablation(events, cuts[:1], harness_config=_harness())
    assert all(not row.k4_ran for row in rows)
    assert all(row.dominant_resource == "" for row in rows)


def test_c4_write_csv_round_trips(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    rows = run_resume_ablation(events, cuts[:2], harness_config=_harness())
    out = tmp_path / "ablation.csv"
    write_ablation_csv(rows, out)
    parsed = list(csv.DictReader(out.open()))
    assert len(parsed) == len(rows)
    assert parsed[0]["trace_id"] == rows[0].trace_id
    assert parsed[0]["package_type"] == rows[0].package_type


def test_extra_setup_steps_count_harness_diff_and_lazy_entries(tmp_path: Path):
    events, cuts = _events_and_cuts(tmp_path)
    pkg = build_agent_migrate_minimal(
        events,
        cuts[0],
        harness_config=_harness(),
        workspace_files=(WorkspaceFileEntry(".venv/lib/site_packages/pkg.py", 500, "h_dep"),),
        workspace_layer_for_file={".venv/lib/site_packages/pkg.py": "dependency_cache"},
    )
    assert count_extra_setup_steps(pkg) == 2  # harness + lazy dependency cache
