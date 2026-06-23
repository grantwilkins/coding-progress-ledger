from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAC = ROOT.parent / "evacuation"


def pytest_collect_file(file_path, parent):
    if Path(file_path).parent == HERE:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        for name in ("instance", "test_instance"):
            mod = sys.modules.get(name)
            if mod and str(getattr(mod, "__file__", "")).startswith(str(EVAC)):
                sys.modules.pop(name)
