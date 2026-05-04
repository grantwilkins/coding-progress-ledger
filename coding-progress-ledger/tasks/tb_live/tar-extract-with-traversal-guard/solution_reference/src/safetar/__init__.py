import os
import tarfile
from pathlib import Path


class UnsafeTarError(Exception):
    pass


def safe_extract(tar_path: str, dest_dir: str) -> None:
    dest = Path(dest_dir).resolve()

    def _safe(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved == dest or str(resolved).startswith(str(dest) + os.sep)

    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            # Absolute path check
            if os.path.isabs(member.name):
                raise UnsafeTarError(f"Absolute path: {member.name}")

            # Path traversal check for the member name itself
            member_dest = (dest / member.name)
            if not _safe(member_dest):
                raise UnsafeTarError(f"Path traversal: {member.name}")

            # Symlink target check
            if member.issym():
                link_dir = (dest / member.name).parent
                target = link_dir / member.linkname
                if not _safe(target):
                    raise UnsafeTarError(f"Symlink escape: {member.name} -> {member.linkname}")

            # Hardlink target check
            if member.islnk():
                if os.path.isabs(member.linkname):
                    raise UnsafeTarError(f"Hardlink absolute target: {member.linkname}")
                target = dest / member.linkname
                if not _safe(target):
                    raise UnsafeTarError(f"Hardlink escape: {member.name} -> {member.linkname}")

        tf.extractall(dest)
