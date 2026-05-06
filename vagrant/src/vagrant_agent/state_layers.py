"""S1 + S2 — state-layer taxonomy and mobile-state auditor.

S1 extends the A1 audit (`docs/A1_workspace_payload_audit.md`) into the
production-facing 11-layer split. Each layer carries a `mobility_class`:

    globally_available    — replicated to every site in advance
    cheaply_rehydratable  — rebuildable from a manifest at any site
    must_move             — agent's progress is lost without it
    can_be_recomputed     — fast-rebuildable from already-moved layers
    can_be_discarded      — agent rarely re-reads; discardable on migration

S2 audits a captured workflow directory: walks the tree, classifies each
regular file into one of the 11 layers using path/extension heuristics,
aggregates bytes per (layer, mobility_class), and writes JSON/CSV. The
classifier is intentionally simple — the goal is a load-bearing byte
budget, not a perfect taxonomy.

The auditor:
  * follows no symlinks (avoids counting bytes outside the workflow dir),
  * counts each `(device, inode)` once (no hardlink double-counting),
  * skips zero-byte files only when explicitly requested.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MOBILITY_CLASSES: tuple[str, ...] = (
    "globally_available",
    "cheaply_rehydratable",
    "must_move",
    "can_be_recomputed",
    "can_be_discarded",
)


@dataclass(frozen=True)
class StateLayerSpec:
    name: str
    mobility_class: str
    description: str

    def __post_init__(self) -> None:
        if self.mobility_class not in MOBILITY_CLASSES:
            raise ValueError(
                f"layer {self.name!r}: unknown mobility_class {self.mobility_class!r}"
            )


# S1 canonical layer registry. Mobility classes are domain claims:
#   * `base_repo_checkout` is `globally_available` because every site can
#     `git clone` from origin; it's not the agent's data.
#   * `dependency_cache` is `cheaply_rehydratable` (rebuilt from
#     pyproject.toml at the destination) — NOT `must_move`. A buggy
#     classifier that ships .venv or node_modules would silently move
#     hundreds of MB.
#   * `build_artifacts` is `can_be_recomputed` because tests reproduce them.
#   * `uncommitted_diff` and `tool_outputs` are `must_move` — the agent's
#     in-progress work and prompt context evaporate without them.
#   * `test_logs` are `can_be_discarded`: the agent rarely re-reads.
S1_LAYERS: tuple[StateLayerSpec, ...] = (
    StateLayerSpec(
        name="base_repo_checkout",
        mobility_class="globally_available",
        description="upstream HEAD; rehydratable from origin at any site",
    ),
    StateLayerSpec(
        name="uncommitted_diff",
        mobility_class="must_move",
        description="agent's in-progress edits (working-tree diff vs HEAD)",
    ),
    StateLayerSpec(
        name="files_read",
        mobility_class="cheaply_rehydratable",
        description="files the agent read but didn't modify; rehydratable from base repo",
    ),
    StateLayerSpec(
        name="files_touched",
        mobility_class="must_move",
        description="files the agent modified mid-iteration; lost if not transported",
    ),
    StateLayerSpec(
        name="tool_outputs",
        mobility_class="must_move",
        description="captured tool stdout/stderr in the agent's prompt context",
    ),
    StateLayerSpec(
        name="test_logs",
        mobility_class="can_be_discarded",
        description="pytest -v output; rarely re-read on migration",
    ),
    StateLayerSpec(
        name="build_artifacts",
        mobility_class="can_be_recomputed",
        description="__pycache__, *.pyc, compiled extensions, build/ outputs",
    ),
    StateLayerSpec(
        name="dependency_cache",
        mobility_class="cheaply_rehydratable",
        description="installed dependency tree (.venv, node_modules); rebuilt from manifest",
    ),
    StateLayerSpec(
        name="retrieved_documents",
        mobility_class="must_move",
        description="result of retrieval (RAG fetches); not cached at destination",
    ),
    StateLayerSpec(
        name="subagent_transcripts",
        mobility_class="must_move",
        description="per-subagent prompt + tool output trace",
    ),
    StateLayerSpec(
        name="summaries_compaction",
        mobility_class="must_move",
        description="compaction outputs / summary buffers / merge contexts",
    ),
)


_LAYER_BY_NAME: dict[str, StateLayerSpec] = {layer.name: layer for layer in S1_LAYERS}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


# Path-segment patterns are checked against the file's relative path *parts*
# split on os.sep; extension patterns are checked against the suffix.
_DEPENDENCY_CACHE_DIRS = (
    ".venv", "venv", "node_modules", ".cache", "site-packages", ".tox",
    ".gradle", "vendor",
)
_BUILD_ARTIFACT_DIRS = ("__pycache__", "build", "dist", ".pytest_cache", ".mypy_cache")
_BUILD_ARTIFACT_EXTS = (".pyc", ".pyo", ".so", ".o", ".class")
_BASE_REPO_DIRS = (".git",)
_TEST_LOG_DIRS = ("test_logs", "test-logs", "test-results")
_TEST_LOG_EXTS = (".log",)
_RETRIEVED_DOCS_DIRS = ("retrieved", "rag_cache", "retrievals")
_SUBAGENT_TRANSCRIPT_DIRS = ("subagents", "subagent_transcripts", "agent_logs")
_SUMMARIES_DIRS = ("summaries", "compaction", "merge_buffers")
_TOOL_OUTPUT_DIRS = ("tool_outputs", "tool_logs")


_UNCOMMITTED_DIFF_FILE = "uncommitted_diff.patch"
_FILES_READ_LIST = "files_read.txt"
_FILES_TOUCHED_LIST = "files_touched.txt"


def classify_file(rel_path: str) -> str:
    """Return the layer name for a file at relative path `rel_path` (POSIX-style).

    The classifier is dispatch-by-prefix-matching over directory components
    plus extension. An unrecognized path falls through to `files_read` —
    it's a captured but unmodified file by default.
    """
    parts = [p for p in rel_path.replace(os.sep, "/").split("/") if p]
    if not parts:
        raise ValueError(f"empty rel_path: {rel_path!r}")
    ext = Path(parts[-1]).suffix
    leaf = parts[-1]

    # Order matters: more specific patterns first.
    if leaf == _UNCOMMITTED_DIFF_FILE:
        return "uncommitted_diff"
    if leaf == _FILES_TOUCHED_LIST:
        return "files_touched"
    if leaf == _FILES_READ_LIST:
        return "files_read"
    if any(part in _BASE_REPO_DIRS for part in parts):
        return "base_repo_checkout"
    if any(part in _DEPENDENCY_CACHE_DIRS for part in parts):
        return "dependency_cache"
    if any(part in _BUILD_ARTIFACT_DIRS for part in parts) or ext in _BUILD_ARTIFACT_EXTS:
        return "build_artifacts"
    if any(part in _SUBAGENT_TRANSCRIPT_DIRS for part in parts):
        return "subagent_transcripts"
    if any(part in _RETRIEVED_DOCS_DIRS for part in parts):
        return "retrieved_documents"
    if any(part in _SUMMARIES_DIRS for part in parts):
        return "summaries_compaction"
    if any(part in _TOOL_OUTPUT_DIRS for part in parts):
        return "tool_outputs"
    if any(part in _TEST_LOG_DIRS for part in parts) or ext in _TEST_LOG_EXTS:
        return "test_logs"
    return "files_read"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditReport:
    workflow_dir: str
    bytes_per_layer: dict[str, int]
    bytes_per_mobility_class: dict[str, int]
    file_count_per_layer: dict[str, int]
    skipped_symlinks: int
    skipped_unreadable: int

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_per_layer.values())

    @property
    def must_move_bytes(self) -> int:
        return self.bytes_per_mobility_class.get("must_move", 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_dir": self.workflow_dir,
            "total_bytes": self.total_bytes,
            "must_move_bytes": self.must_move_bytes,
            "bytes_per_layer": dict(sorted(self.bytes_per_layer.items())),
            "bytes_per_mobility_class": dict(sorted(self.bytes_per_mobility_class.items())),
            "file_count_per_layer": dict(sorted(self.file_count_per_layer.items())),
            "skipped_symlinks": self.skipped_symlinks,
            "skipped_unreadable": self.skipped_unreadable,
        }


def audit_workflow_directory(path: str | Path) -> AuditReport:
    """Walk a captured workflow directory and emit per-layer byte sums.

    * Does not follow symlinks: a symlinked directory or file is counted
      as `skipped_symlinks` and contributes zero bytes (the audit is for
      payload that would actually move, and a symlink to outside the
      workflow dir is by definition not the agent's state).
    * Counts each `(device, inode)` once (no hardlink double-counting).
    * Reads `st_size` from `os.stat` (no symlink traversal). Files that
      `stat` cannot read are counted as `skipped_unreadable`.
    """
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workflow_dir is not a directory: {root}")

    bytes_per_layer: dict[str, int] = {layer.name: 0 for layer in S1_LAYERS}
    file_count_per_layer: dict[str, int] = {layer.name: 0 for layer in S1_LAYERS}
    seen_inodes: set[tuple[int, int]] = set()
    skipped_symlinks = 0
    skipped_unreadable = 0

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Skip symlinked directories.
        kept_dirs: list[str] = []
        for d in dirnames:
            full = Path(dirpath) / d
            if full.is_symlink():
                skipped_symlinks += 1
                continue
            kept_dirs.append(d)
        dirnames[:] = kept_dirs
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                skipped_symlinks += 1
                continue
            try:
                st = os.stat(full, follow_symlinks=False)
            except OSError:
                skipped_unreadable += 1
                continue
            if not (st.st_mode & 0o170000) == 0o100000:
                # not a regular file (sockets, fifos, char/block devices)
                continue
            inode_key = (st.st_dev, st.st_ino)
            if inode_key in seen_inodes:
                continue
            seen_inodes.add(inode_key)
            rel = str(full.relative_to(root))
            layer = classify_file(rel)
            bytes_per_layer[layer] += st.st_size
            file_count_per_layer[layer] += 1

    bytes_per_mobility_class: dict[str, int] = {cls: 0 for cls in MOBILITY_CLASSES}
    for layer_name, n_bytes in bytes_per_layer.items():
        bytes_per_mobility_class[_LAYER_BY_NAME[layer_name].mobility_class] += n_bytes

    return AuditReport(
        workflow_dir=str(root),
        bytes_per_layer=bytes_per_layer,
        bytes_per_mobility_class=bytes_per_mobility_class,
        file_count_per_layer=file_count_per_layer,
        skipped_symlinks=skipped_symlinks,
        skipped_unreadable=skipped_unreadable,
    )


def write_audit_artifacts(report: AuditReport, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    with (out / "audit_layers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "mobility_class", "bytes", "file_count",
        ])
        writer.writeheader()
        for layer in S1_LAYERS:
            writer.writerow({
                "layer": layer.name,
                "mobility_class": layer.mobility_class,
                "bytes": report.bytes_per_layer.get(layer.name, 0),
                "file_count": report.file_count_per_layer.get(layer.name, 0),
            })
    with (out / "audit_mobility_classes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mobility_class", "bytes"])
        writer.writeheader()
        for cls in MOBILITY_CLASSES:
            writer.writerow({
                "mobility_class": cls,
                "bytes": report.bytes_per_mobility_class.get(cls, 0),
            })
