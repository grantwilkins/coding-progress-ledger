from __future__ import annotations

from pathlib import Path


class PathGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, requested: str) -> Path:
        raw = requested or "."
        rel = Path(raw)
        if rel.is_absolute():
            raise ValueError(f"absolute paths are not allowed: {requested}")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes workspace: {requested}")
        if _contains_symlink(self.root, target):
            raise ValueError(f"path crosses symlink inside workspace: {requested}")
        return target


def _contains_symlink(root: Path, target: Path) -> bool:
    current = root
    try:
        rel = target.relative_to(root)
    except ValueError:
        return True
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False
