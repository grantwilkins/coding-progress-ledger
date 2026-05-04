import os
from pathlib import Path


def rotate(path: str, max_bytes: int) -> list[str]:
    p = Path(path)
    size = p.stat().st_size  # raises FileNotFoundError if missing
    if size <= max_bytes:
        return []
    data = p.read_bytes()
    parts = []
    i = 0
    chunk_num = 1
    while i < len(data):
        chunk = data[i : i + max_bytes]
        part_path = f"{path}.{chunk_num:04d}"
        Path(part_path).write_bytes(chunk)
        parts.append(part_path)
        i += max_bytes
        chunk_num += 1
    p.unlink()
    return parts
