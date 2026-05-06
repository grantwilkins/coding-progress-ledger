from __future__ import annotations

"""
Claim:
S1 defines an 11-layer state-layer taxonomy with explicit mobility classes,
and S2's `audit_workflow_directory` walks a captured workflow dir, classifies
each regular file into exactly one layer, and emits per-(layer, mobility_class)
byte sums that conserve total bytes. Hardlinks are counted once; symlinks are
not followed.

Plausible wrong implementations:
- a layer is silently absent from the registry (e.g., dropping
  `dependency_cache` so a 600 MB .venv shows up as `files_read`);
- mobility class drift — `dependency_cache` marked `must_move` so callers
  ship hundreds of MB across sites;
- the auditor follows symlinks and counts external bytes, inflating
  `must_move_bytes` for caller-decided budgets;
- hardlinks double-counted (a layer's bytes exceed the disk usage);
- bytes accounting drops files that don't match any explicit pattern
  (the `else` branch silently zeroes them out);
- classification is leaf-name-only and misses files in deep `.git/` /
  `node_modules/` subtrees.
"""

import json
import os
import platform
from pathlib import Path

import pytest

from vagrant_agent.state_layers import (
    MOBILITY_CLASSES,
    S1_LAYERS,
    audit_workflow_directory,
    classify_file,
    write_audit_artifacts,
)


# ---------------------------------------------------------------------------
# S1 — taxonomy invariants
# ---------------------------------------------------------------------------


def test_s1_registry_has_eleven_named_layers():
    """Claim (paper-to-code): TASKS.md § Workstream S1 lists exactly 11
    layers; the registry must enumerate all 11. A future contributor
    dropping a layer (or aliasing two) is the failure mode."""
    expected = {
        "base_repo_checkout",
        "uncommitted_diff",
        "files_read",
        "files_touched",
        "tool_outputs",
        "test_logs",
        "build_artifacts",
        "dependency_cache",
        "retrieved_documents",
        "subagent_transcripts",
        "summaries_compaction",
    }
    actual = {layer.name for layer in S1_LAYERS}
    assert actual == expected
    assert len(S1_LAYERS) == 11  # no duplicates


def test_s1_mobility_classes_match_domain_claims():
    """Claim (paper-to-code): mobility classes are domain claims, not
    cosmetic labels. A buggy classifier that flips dependency_cache to
    `must_move` would have callers ship hundreds of MB across sites
    unnecessarily; flipping uncommitted_diff to `cheaply_rehydratable`
    would lose agent progress on migration."""
    by_name = {layer.name: layer for layer in S1_LAYERS}
    assert by_name["base_repo_checkout"].mobility_class == "globally_available"
    assert by_name["dependency_cache"].mobility_class == "cheaply_rehydratable"
    assert by_name["files_read"].mobility_class == "cheaply_rehydratable"
    assert by_name["uncommitted_diff"].mobility_class == "must_move"
    assert by_name["files_touched"].mobility_class == "must_move"
    assert by_name["tool_outputs"].mobility_class == "must_move"
    assert by_name["retrieved_documents"].mobility_class == "must_move"
    assert by_name["subagent_transcripts"].mobility_class == "must_move"
    assert by_name["summaries_compaction"].mobility_class == "must_move"
    assert by_name["build_artifacts"].mobility_class == "can_be_recomputed"
    assert by_name["test_logs"].mobility_class == "can_be_discarded"


# ---------------------------------------------------------------------------
# Classifier — boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path,expected_layer", [
    # Base repo: anything under .git/ goes to base_repo_checkout regardless of depth.
    (".git/HEAD", "base_repo_checkout"),
    (".git/objects/pack/pack-1234.pack", "base_repo_checkout"),
    # Dependency cache patterns must catch deep subtrees.
    (".venv/lib/python3.13/site-packages/numpy/__init__.py", "dependency_cache"),
    ("node_modules/react/index.js", "dependency_cache"),
    # Build artifacts by extension AND by directory.
    ("src/__pycache__/foo.cpython-313.pyc", "build_artifacts"),
    ("build/lib/extension.so", "build_artifacts"),
    ("foo.pyc", "build_artifacts"),
    # Test logs by extension OR by directory.
    ("test_logs/run-001.log", "test_logs"),
    ("results/pytest.log", "test_logs"),
    # Explicit must_move sentinels.
    ("uncommitted_diff.patch", "uncommitted_diff"),
    ("files_touched.txt", "files_touched"),
    # Tool / subagent / retrieval / summary directories.
    ("tool_outputs/curl_response.json", "tool_outputs"),
    ("subagents/subagent_0/transcript.jsonl", "subagent_transcripts"),
    ("retrieved/wiki_results.jsonl", "retrieved_documents"),
    ("summaries/compaction_001.md", "summaries_compaction"),
    # Default: an unknown file under src/ is classified as files_read.
    ("src/unknown_module.py", "files_read"),
])
def test_classify_file_handles_known_cases(rel_path, expected_layer):
    """Claim (boundary tests): classifier dispatches by directory component
    and extension. The deep `.git/objects/pack/...` case catches a
    leaf-only-name classifier; the `__pycache__` case catches an
    extension-only classifier."""
    assert classify_file(rel_path) == expected_layer


def test_classify_file_dependency_cache_beats_extension():
    """Claim (precedence): a `.py` file inside `.venv/site-packages` is
    `dependency_cache`, NOT `files_read`. Wrong precedence ordering would
    classify everything under .venv/ as the catch-all."""
    layer = classify_file(".venv/lib/python3.13/site-packages/foo.py")
    assert layer == "dependency_cache"


def test_classify_file_rejects_empty_path():
    """Claim (hard-fail boundary): an empty rel_path is a programmer
    error, not a silent miss-classification."""
    with pytest.raises(ValueError):
        classify_file("")


# ---------------------------------------------------------------------------
# Audit — accounting invariants
# ---------------------------------------------------------------------------


def _write_file(path: Path, n_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * n_bytes)


def _build_synthetic_workflow_dir(root: Path) -> dict[str, int]:
    """Build a small synthetic workflow tree with hand-known per-layer bytes.

    Returns the expected bytes_per_layer dict so tests can compare directly.
    """
    expected: dict[str, int] = {layer.name: 0 for layer in S1_LAYERS}

    _write_file(root / ".git/HEAD", 100)
    expected["base_repo_checkout"] += 100
    _write_file(root / ".git/objects/pack/pack-1.pack", 1_000)
    expected["base_repo_checkout"] += 1_000

    _write_file(root / "uncommitted_diff.patch", 200)
    expected["uncommitted_diff"] += 200
    _write_file(root / "files_touched.txt", 50)
    expected["files_touched"] += 50

    _write_file(root / ".venv/lib/site-packages/foo.py", 5_000)
    expected["dependency_cache"] += 5_000

    _write_file(root / "src/main.py", 800)
    expected["files_read"] += 800

    _write_file(root / "src/__pycache__/main.cpython-313.pyc", 600)
    expected["build_artifacts"] += 600

    _write_file(root / "test_logs/run.log", 400)
    expected["test_logs"] += 400

    _write_file(root / "tool_outputs/curl.json", 150)
    expected["tool_outputs"] += 150

    _write_file(root / "subagents/s0/transcript.jsonl", 2_000)
    expected["subagent_transcripts"] += 2_000

    _write_file(root / "retrieved/docs.jsonl", 3_000)
    expected["retrieved_documents"] += 3_000

    _write_file(root / "summaries/compact_1.md", 250)
    expected["summaries_compaction"] += 250

    return expected


def test_audit_per_layer_bytes_match_hand_known_values(tmp_path):
    """Claim (hand-checkable case): on a synthetic workflow tree where
    each layer has hand-placed file sizes, the auditor reports each
    layer's bytes exactly. A classifier that drops 'unknown' files into
    a wrong layer would mis-distribute the totals."""
    expected = _build_synthetic_workflow_dir(tmp_path)
    report = audit_workflow_directory(tmp_path)
    assert report.bytes_per_layer == expected


def test_audit_total_bytes_equals_disk_usage(tmp_path):
    """Claim (conservation invariant): `total_bytes` equals the sum of
    `st_size` for every regular file under the root. Bytes do not vanish
    in classification."""
    _build_synthetic_workflow_dir(tmp_path)
    on_disk_total = 0
    for root, _, files in os.walk(tmp_path):
        for name in files:
            full = Path(root) / name
            if full.is_symlink():
                continue
            on_disk_total += full.stat().st_size
    report = audit_workflow_directory(tmp_path)
    assert report.total_bytes == on_disk_total


def test_audit_mobility_class_sums_match_layer_sums(tmp_path):
    """Claim (decomposition / aggregation level invariant): summing
    bytes_per_layer over a mobility class equals bytes_per_mobility_class.
    A wrong-aggregation-level bug (summing layers under the wrong class)
    would break this."""
    _build_synthetic_workflow_dir(tmp_path)
    report = audit_workflow_directory(tmp_path)
    by_name = {layer.name: layer for layer in S1_LAYERS}
    for cls in MOBILITY_CLASSES:
        expected = sum(
            report.bytes_per_layer[name]
            for name in report.bytes_per_layer
            if by_name[name].mobility_class == cls
        )
        assert report.bytes_per_mobility_class[cls] == expected


def test_audit_does_not_follow_symlinks(tmp_path):
    """Claim (safety invariant): a symlink pointing outside the workflow
    dir contributes zero bytes and bumps `skipped_symlinks`. Following
    symlinks would inflate `must_move_bytes` with caller-external state.

    Skipped on Windows (symlink permissions differ)."""
    if platform.system() == "Windows":
        pytest.skip("symlink semantics differ on Windows")
    external = tmp_path / "external_state.bin"
    _write_file(external, 100_000)
    workflow = tmp_path / "wf"
    workflow.mkdir()
    _write_file(workflow / "src/main.py", 500)
    # Symlink inside the workflow dir pointing to the external file.
    (workflow / "src/external_link.bin").symlink_to(external)
    report = audit_workflow_directory(workflow)
    assert report.skipped_symlinks >= 1
    assert report.total_bytes == 500


def test_audit_does_not_double_count_hardlinks(tmp_path):
    """Claim (uniqueness invariant): hardlinks share an inode, so the
    auditor must count their bytes once. A naive `os.walk` that adds
    `st_size` for every dirent would inflate totals."""
    if platform.system() == "Windows":
        pytest.skip("hardlink permissions differ on Windows")
    workflow = tmp_path / "wf"
    workflow.mkdir()
    src = workflow / "src/main.py"
    _write_file(src, 1_000)
    target = workflow / "src/main_hardlink.py"
    try:
        os.link(src, target)
    except OSError:
        pytest.skip("hardlinks not supported on this filesystem")
    report = audit_workflow_directory(workflow)
    # Two dirents, one inode → 1000 bytes total. (Both dirents land in
    # `files_read`, so the layer total is also 1000, not 2000.)
    assert report.total_bytes == 1_000
    assert report.bytes_per_layer["files_read"] == 1_000


def test_audit_must_move_excludes_dependency_cache(tmp_path):
    """Claim (direction / domain): a workflow with a 600 MB-equivalent
    dep cache and a 200-byte uncommitted diff has `must_move_bytes` ≈
    200 — the dep cache is `cheaply_rehydratable`. A wrong mobility
    class on `dependency_cache` would inflate `must_move_bytes` by
    orders of magnitude.

    Uses small absolute sizes for fast IO; the ratio is what matters."""
    workflow = tmp_path / "wf"
    workflow.mkdir()
    _write_file(workflow / "uncommitted_diff.patch", 200)
    _write_file(workflow / ".venv/lib/site-packages/big.py", 600_000)
    report = audit_workflow_directory(workflow)
    assert report.must_move_bytes == 200
    assert report.bytes_per_mobility_class["cheaply_rehydratable"] == 600_000


def test_audit_is_deterministic(tmp_path):
    """Claim (determinism invariant): two audits of the same tree must
    return byte-identical reports. A non-determinism bug (e.g., dict
    iteration order leaking into accumulation) would break this."""
    _build_synthetic_workflow_dir(tmp_path)
    a = audit_workflow_directory(tmp_path)
    b = audit_workflow_directory(tmp_path)
    assert a.bytes_per_layer == b.bytes_per_layer
    assert a.bytes_per_mobility_class == b.bytes_per_mobility_class
    assert a.file_count_per_layer == b.file_count_per_layer


def test_write_audit_artifacts_round_trips_json(tmp_path):
    """Claim (artifact integrity): the JSON artifact carries enough info
    to reconstruct the per-layer / per-class accounting. A drop of any
    field would silently lose audit content for a caller reading from
    disk."""
    workflow = tmp_path / "wf"
    workflow.mkdir()
    _build_synthetic_workflow_dir(workflow)
    report = audit_workflow_directory(workflow)
    out_dir = tmp_path / "out"
    write_audit_artifacts(report, out_dir)
    parsed = json.loads((out_dir / "audit.json").read_text())
    assert parsed["total_bytes"] == report.total_bytes
    assert parsed["must_move_bytes"] == report.must_move_bytes
    assert parsed["bytes_per_layer"] == report.bytes_per_layer
    assert parsed["bytes_per_mobility_class"] == report.bytes_per_mobility_class
    layers_csv = (out_dir / "audit_layers.csv").read_text().splitlines()
    # Header + 11 layer rows.
    assert len(layers_csv) == 12
