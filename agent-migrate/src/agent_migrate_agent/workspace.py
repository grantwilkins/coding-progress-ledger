"""Real workspace_bytes computation (Workstream H4).

Closes the H2 framing gap: H2's per_session workspace bytes were synthetic
(integers passed by the fixture builder). H4 lets a caller point at a real
filesystem snapshot — e.g., the repository state at the end of an SWE_agent
rollout — and compute the workspace state's `bytes` field from disk.

The function is intentionally thin: it walks a directory tree and sums file
sizes. It does NOT capture content_hash, mtime, or any temporal/identity
signal — agent_migrate's cost model only cares about bytes for `artifact_copy`,
and treating the same workspace path as a stable identity is the caller's
responsibility (use the same `state_id` across calls).

Default exclusions:

    .git/ — git internals can be 5_50x the working tree on active repos.
            Almost always wrong to ship across a serving link.
            Pass exclude_patterns=() to include everything.

Symlinks: NOT followed for either files or directories. A symlinked file
contributes the symlink's own size (a few bytes), not the target's,
mirroring what a `tar -h=no` snapshot of the workspace would actually
transfer. This is symmetric with `os.walk(followlinks=False)` for
directories — both ends of the walk treat symlinks as opaque references
rather than expansions. Broken symlinks contribute their (small) size.

This module does not depend on or invoke any harness; it is a pure helper.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_EXCLUDES: tuple[str, ...] = (".git",)


def compute_repo_bytes(
    path: str | Path,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> int:
    """Recursive sum of file sizes under `path`. Hard_fails on missing path.

    `exclude_patterns` matches against directory *names* (not paths), pruning
    the walk. To include everything pass `()`.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"workspace path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"workspace path is not a directory: {root}")

    total = 0
    excludes = set(exclude_patterns)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                st = os.lstat(fpath)
            except (FileNotFoundError, OSError):
                continue
            total += st.st_size
    return total
