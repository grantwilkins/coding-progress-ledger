from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
REPO = RUN_ROOT / "repo"
TEST_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TEST_CMD = [str(TEST_PYTHON), "-m", "pytest"]
TEST_CMD_DISPLAY = "../../../.venv/bin/python -m pytest"

sys.path.insert(0, str(PROJECT_ROOT))
from ledger_progress import LedgerSession  # noqa: E402


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def must(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd} failed in {cwd}:\n{result.stdout}")
    return result


def config_loader_initial() -> str:
    return '''class ConfigError(Exception):
    """Raised when a user supplied configuration cannot be loaded."""


REQUIRED_KEYS = ("name", "timeout")


def load_config(raw):
    """Validate and normalize a minimal application config."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")

    for key in REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(f"missing required config key: {key}")

    name = raw["name"]
    timeout = raw["timeout"]

    if not isinstance(name, str):
        raise ConfigError("config key 'name' must be str")
    if not isinstance(timeout, int):
        raise ValueError("config key 'timeout' must be int")
    if timeout <= 0:
        raise ConfigError("config key 'timeout' must be positive")

    return {"name": name.strip(), "timeout": timeout}
'''


def config_loader_missing_fixed() -> str:
    return config_loader_initial().replace(
        'raise ValueError(f"missing required config key: {key}")',
        'raise ConfigError(f"missing required config key: {key}")',
    )


def config_loader_final() -> str:
    return config_loader_missing_fixed().replace(
        "raise ValueError(\"config key 'timeout' must be int\")",
        "raise ConfigError(\"config key 'timeout' must be int\")",
    )


def tests_content() -> str:
    return '''"""
Claim:
load_config exposes a single domain exception, ConfigError, for invalid user
configuration while preserving the message text that identifies the bad field.

Plausible wrong implementations:
- Define ConfigError but accidentally raise built-in ValueError at one branch.
- Fix the missing-key branch but leave a type-validation branch inconsistent.
- Wrap errors in ConfigError while replacing the specific diagnostic message.
"""

import pytest

from config_loader import ConfigError, load_config


def test_missing_required_key_raises_config_error():
    with pytest.raises(ConfigError, match="missing required config key: timeout"):
        load_config({"name": "worker"})


def test_wrong_value_type_raises_config_error():
    with pytest.raises(ConfigError, match="config key 'timeout' must be int"):
        load_config({"name": "worker", "timeout": "30"})


def test_error_message_is_preserved_for_callers():
    with pytest.raises(ConfigError) as excinfo:
        load_config({"name": "worker"})

    assert str(excinfo.value) == "missing required config key: timeout"
'''


def read_progress(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)

    session = LedgerSession("TASK 3: Wrong exception type")
    transcript: list[str] = []
    step = 1

    s_repo = session.add("Create toy Python repo with intentionally inconsistent config exceptions", step=step, weight=1.0)
    s_tests = session.add("Add deterministic pytest contract tests for ConfigError behavior", step=step, weight=1.0)
    s_fix = session.add("Patch config loader to raise ConfigError consistently", step=step, weight=1.0)
    s_verify = session.add("Run tests and capture reproducible evidence", step=step, weight=1.0)
    s_artifacts = session.add("Export ledger and required run artifact bundle", step=step, weight=1.0)
    transcript.append("Step 1: Discovered five concrete subtasks for the empirical run.")

    step += 1
    session.start(s_repo, step=step, evidence="Creating package, tests, and git baseline")
    write(REPO / "config_loader.py", config_loader_initial())
    write(REPO / "tests" / "test_config_loader.py", tests_content())
    write(
        REPO / "pyproject.toml",
        """[project]
name = "task-3-config-error-type"
version = "0.1.0"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
""",
    )
    write(REPO / "README.md", "# Config Loader Toy Repo\n\nRun tests with `../../../.venv/bin/python -m pytest`.\n")
    must(["git", "init"], REPO)
    must(["git", "add", "."], REPO)
    must(["git", "-c", "user.name=Ledger Worker", "-c", "user.email=ledger@example.invalid", "commit", "-m", "Initial buggy config loader"], REPO)
    session.complete(s_repo, "Initial repo committed with ConfigError plus two ValueError raise sites", step=step)
    transcript.append("Step 2: Created and committed the intentionally buggy toy repo.")

    step += 1
    session.start(s_tests, step=step, evidence="Running initial tests against buggy baseline")
    initial = run(TEST_CMD, REPO)
    session.complete(s_tests, f"Initial pytest failed as expected with return code {initial.returncode}", step=step)
    transcript.append("Step 3: Tests fail on the baseline, proving they detect the wrong exception type.")

    step += 1
    session.start(s_fix, step=step, evidence="Applying first narrow ConfigError replacement")
    write(REPO / "config_loader.py", config_loader_missing_fixed())
    after_first = run(TEST_CMD, REPO)
    session.complete(s_fix, "Patched missing-key raise site to ConfigError", step=step)
    transcript.append("Step 4: Fixed the missing-key branch; pytest still reports the timeout type branch as ValueError.")

    step += 1
    session.reopen(s_fix, step=step, reason="Verification showed a second ValueError raise site remained")
    s_fix_missing, s_fix_type = session.split(
        s_fix,
        ["Keep missing-key raise site on ConfigError", "Convert timeout type raise site to ConfigError"],
        step=step,
        reason="The broad patch task hid two independently checkable branches",
    )
    session.complete(s_fix_missing, "Missing required key test now raises ConfigError with the original message", step=step)
    transcript.append("Step 5: Reopened and split the fix task after discovering the second wrong exception site.")

    step += 1
    session.start(s_fix_type, step=step, evidence="Replacing timeout type ValueError with ConfigError")
    write(REPO / "config_loader.py", config_loader_final())
    session.complete(s_fix_type, "Timeout type branch now raises ConfigError with preserved message", step=step)
    transcript.append("Step 6: Fixed the second ValueError raise site.")

    step += 1
    session.start(s_verify, step=step, evidence="Running final pytest command")
    final = run(TEST_CMD, REPO)
    test_output = (
        f"$ {TEST_CMD_DISPLAY}\n\n"
        "Initial baseline output:\n"
        f"{initial.stdout}\n"
        "After first raise-site fix output:\n"
        f"{after_first.stdout}\n"
        "Final output:\n"
        f"{final.stdout}"
    )
    write(RUN_ROOT / "test_output.txt", test_output)
    if final.returncode == 0:
        session.complete(s_verify, f"Final {TEST_CMD_DISPLAY} passed; output captured in test_output.txt", step=step)
    else:
        session.block(s_verify, step=step, reason=f"Final pytest failed with return code {final.returncode}", evidence=final.stdout)
    transcript.append(f"Step 7: Final pytest exited with status {final.returncode}.")

    step += 1
    session.start(s_artifacts, step=step, evidence="Writing run metadata and patch artifacts")
    patch = must(["git", "diff", "HEAD"], REPO)
    write(RUN_ROOT / "final_diff.patch", patch.stdout)
    write(
        RUN_ROOT / "task.md",
        """# TASK 3: Wrong exception type

Create a tiny Python config loader that defines ConfigError but initially raises
ValueError for invalid config in multiple places, then solve it so all invalid
config paths consistently raise ConfigError.
""",
    )
    write(
        RUN_ROOT / "README.md",
        """# Task 3 Config Error Type

Toy repo: `repo/`

From the repo directory, run:

```bash
../../../.venv/bin/python -m pytest
```

The final implementation should pass all tests.
""",
    )
    write(RUN_ROOT / "agent_transcript.md", "# Agent Transcript\n\n" + "\n".join(f"- {line}" for line in transcript) + "\n")
    session.complete(s_artifacts, "Required artifact files written; ledger export pending as final side effect", step=step)

    session.export_jsonl(str(RUN_ROOT / "ledger.jsonl"))
    session.export_curve_csv(str(RUN_ROOT / "progress.csv"))

    rows = read_progress(RUN_ROOT / "progress.csv")
    progress_values = [float(row["progress"]) for row in rows]
    drops = [progress_values[i] - progress_values[i + 1] for i in range(len(progress_values) - 1)]
    largest_drop = max([drop for drop in drops if drop > 0], default=0.0)
    events = session.ledger.events
    summary = {
        "task_id": "task_3_config_error_type",
        "final_progress": session.score().progress,
        "subtasks_created": len(session.ledger.subtasks),
        "completed_subtasks": sum(1 for subtask in session.ledger.subtasks.values() if subtask.status.value == "complete"),
        "splits": sum(1 for event in events if event.event_type.value == "split_subtask"),
        "reopens": sum(1 for event in events if event.event_type.value == "reopen_subtask"),
        "invalidations": sum(1 for event in events if event.event_type.value == "invalidate_subtask"),
        "largest_progress_drop": largest_drop,
        "non_monotonic": any(drop > 0 for drop in drops),
        "test_command": TEST_CMD_DISPLAY,
        "test_status": "passed" if final.returncode == 0 else "failed",
        "artifact_paths": [
            "task.md",
            "README.md",
            "agent_transcript.md",
            "ledger.jsonl",
            "progress.csv",
            "final_diff.patch",
            "test_output.txt",
            "run_notes.md",
            "summary.json",
            "repo/",
        ],
    }
    write(RUN_ROOT / "summary.json", json.dumps(summary, indent=2) + "\n")
    write(
        RUN_ROOT / "run_notes.md",
        f"""# Run Notes

## Progress Changes

The run began with five concrete active subtasks. Progress rose as the toy repo
and tests were completed, then dropped when the broad fix subtask was reopened
and split after evidence showed only one of two ValueError raise sites had been
converted. The largest observed progress drop was {largest_drop:.6f}.

## New Work Discovery

The second ValueError site was discovered after the first narrow fix and a
pytest rerun. That changed the work model from one broad implementation task to
two branch-specific checks: missing required keys and wrong timeout type.

## Evidence-Backed Completions

The initial repo completion is backed by a git commit containing two wrong
ValueError raise sites. The test completion is backed by failing pytest output
on the baseline. The final verification completion is backed by the passing
`../../../.venv/bin/python -m pytest` output in `test_output.txt`.

## Ledger Notes

The ledger was useful for making the non-monotonic discovery explicit instead
of treating the first partial fix as steady progress. The awkward part is that
artifact export itself is real work but the ledger files are only available
after the final ledger event, so the export completion evidence has to describe
the pending final side effect.
""",
    )

    if final.returncode != 0:
        raise SystemExit(final.returncode)


if __name__ == "__main__":
    main()
