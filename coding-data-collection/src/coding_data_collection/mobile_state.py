from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .audits import scan_agent_workspace_for_leakage


LAYER_FIELDS = (
    "clean_repo_bytes",
    "initial_workspace_bytes",
    "final_workspace_bytes",
    "unchanged_initial_bytes",
    "modified_file_bytes",
    "new_file_bytes",
    "deleted_file_bytes",
    "final_diff_bytes",
    "touched_file_bytes",
    "read_file_bytes",
    "tool_output_bytes",
    "test_log_bytes",
    "build_artifact_bytes",
    "dependency_cache_bytes",
    "retrieved_document_bytes",
    "workspace_total_bytes",
    "hidden_or_protected_bytes",
)

HIDDEN_COMPONENTS = {
    "protected",
    "hidden",
    ".git",
    "tests",
    "test",
    "oracle",
    "gold",
    "verifier",
    "solution",
    "solution_reference",
}
DEPENDENCY_COMPONENTS = {".venv", "venv", "node_modules", "site-packages", ".cache", "cache"}
BUILD_COMPONENTS = {"__pycache__", "build", "dist", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
RETRIEVED_SUFFIXES = {".parquet", ".csv", ".jsonl", ".json", ".db", ".sqlite", ".sqlite3"}


@dataclass(frozen=True)
class MobileStateSnapshot:
    run_id: str
    run_dir: str
    run_status: str
    final_success: bool | None
    eligible_for_l_gate: bool | None
    agent_workspace_retained: bool
    run_validation_passed: bool | None
    leakage_passed: bool | None
    leakage_hit_count: int
    row_usable_for_claims: bool
    clean_repo_bytes: int
    clean_repo_bytes_provenance: str
    initial_workspace_bytes: int
    initial_workspace_bytes_provenance: str
    final_workspace_bytes: int
    final_workspace_bytes_provenance: str
    unchanged_initial_bytes: int
    unchanged_initial_bytes_provenance: str
    modified_file_bytes: int
    modified_file_bytes_provenance: str
    new_file_bytes: int
    new_file_bytes_provenance: str
    deleted_file_bytes: int
    deleted_file_bytes_provenance: str
    final_diff_bytes: int
    final_diff_bytes_provenance: str
    touched_file_bytes: int
    touched_file_bytes_provenance: str
    read_file_bytes: int
    read_file_bytes_provenance: str
    tool_output_bytes: int
    tool_output_bytes_provenance: str
    test_log_bytes: int
    test_log_bytes_provenance: str
    build_artifact_bytes: int
    build_artifact_bytes_provenance: str
    dependency_cache_bytes: int
    dependency_cache_bytes_provenance: str
    retrieved_document_bytes: int
    retrieved_document_bytes_provenance: str
    workspace_total_bytes: int
    workspace_total_bytes_provenance: str
    hidden_or_protected_bytes: int
    hidden_or_protected_bytes_provenance: str
    skipped_symlink_count: int
    setup_command_count: int
    lockfile_count: int
    final_diff_semantics: str


def snapshot_run(run_dir: str | Path) -> MobileStateSnapshot:
    run = Path(run_dir)
    manifest = _json(run / "run_manifest.json") if (run / "run_manifest.json").exists() else {}
    workspace = run / "agent_workspace"
    files, skipped_symlinks = _regular_files(workspace) if workspace.exists() else ([], 0)
    rel_to_size = {rel: size for rel, size in files}
    touched = _transcript_paths(run / "transcript.jsonl", {"write_file", "apply_patch"})
    read = _transcript_paths(run / "transcript.jsonl", {"read_file", "chunked_file_read"})
    layer_bytes = _workspace_layer_bytes(files)
    final_diff = _file_size(run / "final_diff.patch")
    tool_output = _tool_output_bytes(run / "transcript.jsonl")
    test_logs = _file_size(run / "test_output.txt") + _file_size(run / "verifier_output.txt")
    leakage = scan_agent_workspace_for_leakage(workspace) if workspace.exists() else None
    leakage_passed = None if leakage is None else bool(leakage["passed"])
    leakage_hit_count = 0 if leakage is None else len(leakage["leakage_hits"])
    final_workspace = layer_bytes["workspace_total_bytes"]
    touched_bytes = _path_set_bytes(workspace, touched)
    read_bytes = _path_set_bytes(workspace, read)
    row_usable = bool(workspace.exists() and leakage_passed)
    return MobileStateSnapshot(
        run_id=str(manifest.get("run_id") or run.name),
        run_dir=str(run),
        run_status=str(manifest.get("run_status") or "missing"),
        final_success=manifest.get("final_success"),
        eligible_for_l_gate=(manifest.get("metrics") or {}).get("eligible_for_L_gate"),
        agent_workspace_retained=workspace.exists(),
        run_validation_passed=None,
        leakage_passed=leakage_passed,
        leakage_hit_count=leakage_hit_count,
        row_usable_for_claims=row_usable,
        clean_repo_bytes=0,
        clean_repo_bytes_provenance="missing",
        initial_workspace_bytes=0,
        initial_workspace_bytes_provenance="missing",
        final_workspace_bytes=final_workspace,
        final_workspace_bytes_provenance="measured" if workspace.exists() else "missing",
        unchanged_initial_bytes=0,
        unchanged_initial_bytes_provenance="missing_without_initial_workspace_manifest",
        modified_file_bytes=final_diff,
        modified_file_bytes_provenance="patch_file_bytes" if (run / "final_diff.patch").exists() else "missing",
        new_file_bytes=touched_bytes,
        new_file_bytes_provenance="trace_derived_touched_file_upper_bound" if touched else "missing",
        deleted_file_bytes=0,
        deleted_file_bytes_provenance="missing_without_initial_workspace_manifest",
        final_diff_bytes=final_diff,
        final_diff_bytes_provenance="measured" if (run / "final_diff.patch").exists() else "missing",
        touched_file_bytes=touched_bytes,
        touched_file_bytes_provenance="trace_derived" if touched else "missing",
        read_file_bytes=read_bytes,
        read_file_bytes_provenance="trace_derived" if read else "missing",
        tool_output_bytes=tool_output,
        tool_output_bytes_provenance="lower_bound_transcript_snippet_bytes" if tool_output else "missing",
        test_log_bytes=test_logs,
        test_log_bytes_provenance="measured" if test_logs else "missing",
        build_artifact_bytes=layer_bytes["build_artifact_bytes"],
        build_artifact_bytes_provenance="measured" if workspace.exists() else "missing",
        dependency_cache_bytes=layer_bytes["dependency_cache_bytes"],
        dependency_cache_bytes_provenance="measured" if workspace.exists() else "missing",
        retrieved_document_bytes=layer_bytes["retrieved_document_bytes"],
        retrieved_document_bytes_provenance="measured" if workspace.exists() else "missing",
        workspace_total_bytes=final_workspace,
        workspace_total_bytes_provenance="measured" if workspace.exists() else "missing",
        hidden_or_protected_bytes=layer_bytes["hidden_or_protected_bytes"],
        hidden_or_protected_bytes_provenance="measured" if workspace.exists() else "missing",
        skipped_symlink_count=skipped_symlinks,
        setup_command_count=_setup_command_count(run / "transcript.jsonl"),
        lockfile_count=sum(1 for rel in rel_to_size if _is_lockfile(rel)),
        final_diff_semantics="patch_file_bytes_not_touched_file_payload",
    )


def snapshot_run_roots(run_roots: Iterable[str | Path]) -> list[MobileStateSnapshot]:
    runs: list[MobileStateSnapshot] = []
    for root in run_roots:
        for manifest in sorted(Path(root).glob("**/run_manifest.json")):
            runs.append(snapshot_run(manifest.parent))
    return runs


def write_snapshots(snapshots: Iterable[MobileStateSnapshot], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(snapshots)
    for row in rows:
        suffix = hashlib.sha256(row.run_dir.encode("utf-8")).hexdigest()[:10]
        safe = row.run_id.replace("/", "__")
        safe = f"{safe}.{suffix}"
        (out / f"{safe}.mobile_state.json").write_text(json.dumps(asdict(row), indent=2) + "\n")
    with (out / "raw_snapshot_index.csv").open("w", newline="") as f:
        fields = list(MobileStateSnapshot.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _regular_files(root: Path) -> tuple[list[tuple[str, int]], int]:
    if not root.exists():
        return [], 0
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, int]] = []
    skipped_symlinks = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept_dirs: list[str] = []
        for dirname in dirnames:
            path = Path(dirpath) / dirname
            if path.is_symlink():
                skipped_symlinks += 1
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink():
                skipped_symlinks += 1
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            out.append((path.relative_to(root).as_posix(), stat.st_size))
    return out, skipped_symlinks


def _workspace_layer_bytes(files: list[tuple[str, int]]) -> dict[str, int]:
    out = {
        "build_artifact_bytes": 0,
        "dependency_cache_bytes": 0,
        "retrieved_document_bytes": 0,
        "workspace_total_bytes": 0,
        "hidden_or_protected_bytes": 0,
    }
    for rel, size in files:
        parts = set(Path(rel).parts)
        out["workspace_total_bytes"] += size
        if parts & HIDDEN_COMPONENTS:
            out["hidden_or_protected_bytes"] += size
        elif parts & DEPENDENCY_COMPONENTS:
            out["dependency_cache_bytes"] += size
        elif parts & BUILD_COMPONENTS:
            out["build_artifact_bytes"] += size
        elif Path(rel).suffix.lower() in RETRIEVED_SUFFIXES:
            out["retrieved_document_bytes"] += size
    return out


def _transcript_paths(path: Path, kinds: set[str]) -> set[str]:
    paths: set[str] = set()
    for event in _jsonl(path):
        if event.get("kind") not in kinds:
            continue
        p = event.get("path")
        if isinstance(p, str):
            paths.add(p.removeprefix("./"))
    return paths


def _path_set_bytes(root: Path, rel_paths: set[str]) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for rel in sorted(rel_paths):
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        candidate = root / rel
        if candidate.is_symlink() or not candidate.is_file():
            continue
        stat = candidate.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += stat.st_size
    return total


def _tool_output_bytes(path: Path) -> int:
    total = 0
    for event in _jsonl(path):
        if event.get("kind") not in {"shell", "read_file", "grep", "find_files", "apply_patch"}:
            continue
        for key in ("stdout_snippet", "stderr_snippet"):
            value = event.get(key)
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
    return total


def _setup_command_count(path: Path) -> int:
    markers = ("pip install", "uv pip", "npm install", "apt-get", "conda install", "Rscript")
    count = 0
    for event in _jsonl(path):
        command = event.get("command")
        if isinstance(command, str) and any(marker in command for marker in markers):
            count += 1
    return count


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() and path.is_file() else 0


def _is_lockfile(rel: str) -> bool:
    return Path(rel).name in {"poetry.lock", "uv.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "requirements.txt"}


__all__ = ["MobileStateSnapshot", "snapshot_run", "snapshot_run_roots", "write_snapshots"]
