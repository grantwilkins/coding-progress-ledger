"""Every TB-live task's verifier must pass against its solution_reference."""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks" / "tb_live"

TASK_IDS = sorted(
    p.name for p in TASKS_DIR.iterdir()
    if p.is_dir() and not p.name.startswith("_") and (p / "verifier.sh").exists()
)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_verifier_passes_reference(task_id):
    task_dir = TASKS_DIR / task_id
    reference = task_dir / "solution_reference"
    assert reference.is_dir(), f"missing solution_reference for {task_id}"
    result = subprocess.run(
        ["bash", str(task_dir / "verifier.sh"), str(reference)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + "\n--- stderr ---\n" + result.stderr
