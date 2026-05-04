import sqlite3
import struct
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
INTACT_DB = FIXTURE_DIR / "intact.sqlite"
CORRUPT_DB = FIXTURE_DIR / "corrupt.sqlite"

NUM_ROWS = 500
MIN_SURVIVING_ROWS = 200

# Wide name field so rows spread across many pages (enables meaningful truncation).
ORIGINAL_ROWS: list[tuple[int, str, int]] = [
    (i, "x" * 200 + f"_{i}", i * 10) for i in range(1, NUM_ROWS + 1)
]


def _build_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE records (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)"
    )
    con.executemany("INSERT INTO records VALUES (?, ?, ?)", ORIGINAL_ROWS)
    con.commit()
    con.close()


def _truncate_and_patch(src: Path, dst: Path, keep_fraction: float = 0.60) -> None:
    """Keep keep_fraction of pages and patch the header page count."""
    data = src.read_bytes()
    page_size = struct.unpack_from(">H", data, 16)[0]
    total_pages = struct.unpack_from(">I", data, 28)[0]
    keep_pages = max(1, int(total_pages * keep_fraction))
    truncated = bytearray(data[: keep_pages * page_size])
    # Patch stored page count so SQLite doesn't immediately error on open.
    struct.pack_into(">I", truncated, 28, keep_pages)
    # Bump change counter so SQLite doesn't use a stale cache.
    cc = struct.unpack_from(">I", data, 24)[0]
    struct.pack_into(">I", truncated, 24, cc + 1)
    struct.pack_into(">I", truncated, 92, cc + 1)
    dst.write_bytes(bytes(truncated))


# Build fixtures at import/collection time.
_build_db(INTACT_DB)
_truncate_and_patch(INTACT_DB, CORRUPT_DB)
