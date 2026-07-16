from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def pytest_collect_file(file_path, parent):
    if Path(file_path).parent == HERE and str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
