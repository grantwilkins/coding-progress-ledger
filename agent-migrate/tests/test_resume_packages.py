from pathlib import Path

import pytest

from agent_migrate_agent.adapters.swe_agent import swe_agent_to_trace
from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl
from agent_migrate_agent.resume_packages import (
    PACKAGE_TYPES,
    ResumePackage,
    StateEntry,
    WorkspaceFileEntry,
    build_full_workspace_snapshot,
    build_prompt_only,
    build_transcript_plus_diff,
    build_transcript_plus_harness_state,
    build_agent_migrate_minimal,
    transcript_prefix_hash,
)

FIXTURE = Path(__file__).parent / "fixtures" / "swe_agent_pilot_s_07.json"


def _events_and_cut(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    events = load_trace_jsonl(out)
    cps = find_cut_points(events, trace_id="s_07")
    assert cps
    return events, cps


def _harness() -> dict:
    return {"cwd": "/changelog_cli", "open_file": "src/changelog/utils.py", "env": {}}


def test_package_types_match_taxonomy():
    assert set(PACKAGE_TYPES) == {
        "prompt_only",
        "transcript_plus_harness_state",
        "transcript_plus_diff",
        "full_workspace_snapshot",
        "agent_migrate_minimal",
    }


def test_prompt_only_includes_only_prompt_context_states(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    assert pkg.package_type == "prompt_only"
    assert pkg.harness_config is None
    assert pkg.diff_blob is None
    assert pkg.base_commit is None
    assert pkg.workspace_files == ()
    sids = {e.state_id for e in pkg.state_entries}
    assert "system_prompt" in sids
    assert "issue_text" in sids
    assert all(e.layer == "prompt_context" for e in pkg.state_entries)
    assert all(e.materialization == "included" for e in pkg.state_entries)


def test_state_entries_are_sorted_for_determinism(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg1 = build_prompt_only(events, cps[2])
    pkg2 = build_prompt_only(events, cps[2])
    sids1 = [e.state_id for e in pkg1.state_entries]
    sids2 = [e.state_id for e in pkg2.state_entries]
    assert sids1 == sids2 == sorted(sids1)


def test_transcript_prefix_hash_changes_with_cut_index(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    h1 = transcript_prefix_hash(events, cps[0].event_index)
    h2 = transcript_prefix_hash(events, cps[1].event_index)
    assert h1 != h2
    assert len(h1) == 64  # sha256 hex


def test_transcript_prefix_hash_out_of_range_raises():
    with pytest.raises(ValueError):
        transcript_prefix_hash([], -1)
    with pytest.raises(ValueError):
        transcript_prefix_hash([{"event_type": "init"}], 5)


def test_transcript_plus_harness_carries_harness_config(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    assert pkg.package_type == "transcript_plus_harness_state"
    assert pkg.harness_config == _harness()
    assert pkg.diff_blob is None
    assert pkg.workspace_files == ()


def test_transcript_plus_diff_carries_diff(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    diff = "--- a/x\n+++ b/x\n@@\n_foo\n+bar\n"
    pkg = build_transcript_plus_diff(events, cps[0],
                                     harness_config=_harness(),
                                     base_commit="abc123",
                                     diff_blob=diff)
    assert pkg.package_type == "transcript_plus_diff"
    assert pkg.base_commit == "abc123"
    assert pkg.diff_blob == diff
    assert pkg.included_bytes >= len(diff.encode("utf_8"))


def test_full_workspace_snapshot_sorts_files(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    files = [
        WorkspaceFileEntry("z.py", 10, "h_z"),
        WorkspaceFileEntry("a.py", 20, "h_a"),
    ]
    pkg = build_full_workspace_snapshot(events, cps[0],
                                        harness_config=_harness(),
                                        workspace_files=files)
    assert [f.rel_path for f in pkg.workspace_files] == ["a.py", "z.py"]
    assert pkg.included_bytes >= 30


def test_agent_migrate_minimal_includes_must_move_excludes_globally_available(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    files = [
        WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_diff"),
        WorkspaceFileEntry(".git/HEAD", 41, "h_git"),
        WorkspaceFileEntry(".venv/lib/python3.11/site_packages/foo/__init__.py", 5000, "h_dep"),
    ]
    layer_for = {
        "uncommitted_diff.patch": "uncommitted_diff",
        ".git/HEAD": "base_repo_checkout",
        ".venv/lib/python3.11/site_packages/foo/__init__.py": "dependency_cache",
    }
    pkg = build_agent_migrate_minimal(events, cps[0],
                                harness_config=_harness(),
                                workspace_files=files,
                                workspace_layer_for_file=layer_for)
    assert pkg.package_type == "agent_migrate_minimal"
    included_paths = {f.rel_path for f in pkg.workspace_files}
    assert "uncommitted_diff.patch" in included_paths
    assert ".git/HEAD" not in included_paths
    assert ".venv/lib/python3.11/site_packages/foo/__init__.py" not in included_paths

    layer_entries = {e.state_id: e for e in pkg.state_entries
                     if e.state_id.startswith("workspace_layer:")}
    assert layer_entries["workspace_layer:uncommitted_diff"].materialization == "included"
    assert layer_entries["workspace_layer:uncommitted_diff"].role_at_cut == "correctness_critical"
    assert layer_entries["workspace_layer:base_repo_checkout"].materialization == "globally_available"
    assert layer_entries["workspace_layer:base_repo_checkout"].role_at_cut == "reconstructable"
    assert layer_entries["workspace_layer:dependency_cache"].materialization == "lazy_rehydrate"
    assert layer_entries["workspace_layer:dependency_cache"].role_at_cut == "performance_critical"
    assert pkg.lazy_rehydrate_bytes == 5000


def test_agent_migrate_minimal_included_bytes_do_not_double_count_workspace_layer_summary(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    files = [WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_diff")]
    pkg = build_agent_migrate_minimal(events, cps[0],
                                harness_config=_harness(),
                                workspace_files=files,
                                workspace_layer_for_file={"uncommitted_diff.patch": "uncommitted_diff"})
    prompt_bytes = sum(e.bytes for e in pkg.state_entries if e.layer == "prompt_context")
    harness_bytes = len('{"cwd":"/changelog_cli","env":{},"open_file":"src/changelog/utils.py"}'.encode("utf_8"))
    assert pkg.included_bytes == prompt_bytes + 100 + harness_bytes


def test_agent_migrate_minimal_unknown_layer_hard_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    files = [WorkspaceFileEntry("weird.bin", 100, "h_weird")]
    with pytest.raises(ValueError, match="unknown S1 layer"):
        build_agent_migrate_minimal(events, cps[0],
                              harness_config=_harness(),
                              workspace_files=files,
                              workspace_layer_for_file={"weird.bin": "not_a_real_layer"})


def test_agent_migrate_minimal_missing_layer_classification_hard_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    files = [WorkspaceFileEntry("a.py", 100, "h_a")]
    with pytest.raises(ValueError, match="missing entry"):
        build_agent_migrate_minimal(events, cps[0],
                              harness_config=_harness(),
                              workspace_files=files,
                              workspace_layer_for_file={})


def test_unknown_package_type_hard_fails(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    with pytest.raises(ValueError, match="unknown package_type"):
        ResumePackage(
            package_type="not_a_thing",
            cut_point=cps[0],
            state_entries=pkg.state_entries,
            transcript_prefix_hash=pkg.transcript_prefix_hash,
        )


def test_unknown_materialization_hard_fails():
    with pytest.raises(ValueError, match="unknown materialization"):
        StateEntry(state_id="x", layer="prompt_context", bytes=0,
                   content_hash="h", materialization="garbage", validator="digest")


def test_to_dict_includes_byte_summaries(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_prompt_only(events, cps[0])
    d = pkg.to_dict()
    assert d["package_type"] == "prompt_only"
    assert d["included_bytes"] == pkg.included_bytes
    assert d["lazy_rehydrate_bytes"] == 0


def test_metrics_returns_flat_csv_row(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    pkg = build_transcript_plus_harness_state(events, cps[0], harness_config=_harness())
    m = pkg.metrics()
    expected_keys = {
        "package_type", "trace_id", "session_id", "event_index", "phase",
        "transcript_prefix_hash", "n_state_entries", "n_workspace_files",
        "included_bytes", "lazy_rehydrate_bytes", "has_diff", "has_harness",
    }
    assert set(m.keys()) == expected_keys
    assert m["package_type"] == "transcript_plus_harness_state"
    assert m["has_harness"] is True
    assert m["has_diff"] is False


def test_unknown_validator_on_state_entry_hard_fails():
    with pytest.raises(ValueError, match="unknown validator"):
        StateEntry(state_id="x", layer="prompt_context", bytes=0,
                   content_hash="h", materialization="included", validator="garbage")


def test_harness_config_normalized_for_determinism(tmp_path: Path):
    """Two builds with differently_ordered dict keys produce identical packages."""
    import json as _json
    events, cps = _events_and_cut(tmp_path)
    h1 = {"cwd": "/x", "open_file": "a.py", "env": {"PATH": "/usr/bin", "HOME": "/h"}}
    h2 = {"env": {"HOME": "/h", "PATH": "/usr/bin"}, "open_file": "a.py", "cwd": "/x"}
    p1 = build_transcript_plus_harness_state(events, cps[0], harness_config=h1)
    p2 = build_transcript_plus_harness_state(events, cps[0], harness_config=h2)
    assert _json.dumps(p1.to_dict(), sort_keys=True) == _json.dumps(p2.to_dict(), sort_keys=True)


def test_harness_config_deepcopy_isolates_from_caller(tmp_path: Path):
    events, cps = _events_and_cut(tmp_path)
    env = {"HOME": "/h"}
    h = {"cwd": "/x", "open_file": "a.py", "env": env}
    pkg = build_transcript_plus_harness_state(events, cps[0], harness_config=h)
    env["HOME"] = "/MUTATED"
    h["cwd"] = "/MUTATED"
    assert pkg.harness_config["cwd"] == "/x"
    assert pkg.harness_config["env"]["HOME"] == "/h"


@pytest.mark.parametrize("builder_name", [
    "prompt_only", "transcript_plus_harness_state", "transcript_plus_diff",
    "full_workspace_snapshot", "agent_migrate_minimal",
])
def test_determinism_byte_identical_across_two_builds(tmp_path: Path, builder_name: str):
    import json as _json
    events, cps = _events_and_cut(tmp_path)
    cp = cps[0]

    def _build():
        if builder_name == "prompt_only":
            return build_prompt_only(events, cp)
        if builder_name == "transcript_plus_harness_state":
            return build_transcript_plus_harness_state(events, cp, harness_config=_harness())
        if builder_name == "transcript_plus_diff":
            return build_transcript_plus_diff(events, cp, harness_config=_harness(),
                                              base_commit="abc123",
                                              diff_blob="--- a/x\n+++ b/x\n@@\n_a\n+b\n")
        if builder_name == "full_workspace_snapshot":
            files = [WorkspaceFileEntry("z.py", 10, "h_z"), WorkspaceFileEntry("a.py", 20, "h_a")]
            return build_full_workspace_snapshot(events, cp, harness_config=_harness(),
                                                 workspace_files=files)
        if builder_name == "agent_migrate_minimal":
            files = [WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_d")]
            return build_agent_migrate_minimal(events, cp, harness_config=_harness(),
                                         workspace_files=files,
                                         workspace_layer_for_file={"uncommitted_diff.patch": "uncommitted_diff"})
        raise AssertionError(builder_name)

    a = _build()
    b = _build()
    assert _json.dumps(a.to_dict(), sort_keys=True) == _json.dumps(b.to_dict(), sort_keys=True)


def test_agent_migrate_minimal_skips_can_be_discarded_layers(tmp_path: Path):
    """test_logs (`can_be_discarded`) should not appear in the manifest at all."""
    events, cps = _events_and_cut(tmp_path)
    files = [
        WorkspaceFileEntry("uncommitted_diff.patch", 100, "h_d"),
        WorkspaceFileEntry("test_logs/run.log", 5000, "h_log"),
    ]
    pkg = build_agent_migrate_minimal(events, cps[0],
                                harness_config=_harness(),
                                workspace_files=files,
                                workspace_layer_for_file={
                                    "uncommitted_diff.patch": "uncommitted_diff",
                                    "test_logs/run.log": "test_logs",
                                })
    sids = {e.state_id for e in pkg.state_entries}
    assert "workspace_layer:test_logs" not in sids
    assert "workspace_layer:uncommitted_diff" in sids
    # `test_logs` files should not be in workspace_files either.
    assert "test_logs/run.log" not in {f.rel_path for f in pkg.workspace_files}
