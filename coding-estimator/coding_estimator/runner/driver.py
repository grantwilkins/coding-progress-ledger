"""tb_live_v2 task runner — prepare / finalize.

Two phases per run, separated by the subagent invocation:

  1. prepare(task_dir, arm) -> RunPrep
       creates a fresh tempdir workspace, copies task seed files
       (excluding tests/), creates a fresh venv if requirements.txt is
       present, renders the subagent prompt, returns the prompt and
       paths the orchestrator passes to the Agent tool.

  2. finalize(workspace, task_dir, run_dir, manifest_seed)
       parses workspace/transcript.jsonl, converts to events.jsonl,
       runs the upstream sidecar to build ledger.jsonl, copies the
       held-back tests/ into the workspace and runs pytest, captures
       final_success, writes run_manifest.json + run_notes.md.

Hard-fail on any deviation (missing transcript, malformed events,
verifier crash). No silent fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from coding_estimator.runner.prompts import PROMPT_VERSION, render_prompt
from coding_estimator.runner.observation_events import (
    build_observation_events,
    write_observation_events,
)
from coding_estimator.runner.transcript_to_events import (
    read_transcript,
    transcript_to_events,
    write_events,
)

ARM_BUDGETS = {"A": 30, "B": 20, "C": 20}
ARM_MODELS = {"A": "claude-opus-4-7", "B": "claude-sonnet-4-6", "C": "claude-haiku-4-5"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RunPrep:
    run_id: str
    task_id: str
    arm: str
    workspace: Path
    run_dir: Path
    prompt: str
    started_at: str


def _new_run_id(task_id: str, arm: str) -> str:
    return f"{task_id}__arm{arm}__{uuid.uuid4().hex[:8]}"


def _copy_task_seed(task_dir: Path, workspace: Path) -> None:
    if not task_dir.is_dir():
        raise NotADirectoryError(task_dir)
    for child in task_dir.iterdir():
        if child.name == "tests":
            continue
        dest = workspace / child.name
        if child.is_dir():
            shutil.copytree(child, dest, symlinks=True)
        else:
            shutil.copy2(child, dest)
    task_md = workspace / "task.md"
    if not task_md.is_file():
        task_md.write_text(_render_task_md(task_dir))


def _render_task_md(task_dir: Path) -> str:
    import yaml  # local import; pyproject already lists pyyaml indirectly via pytest plugins
    spec = yaml.safe_load((task_dir / "task.yaml").read_text())
    descs = spec.get("descriptions") or []
    base = next((d for d in descs if d.get("key") == "base"), None)
    if base is None:
        raise ValueError(f"{task_dir}/task.yaml: missing descriptions[base]")
    return f"# Task — {task_dir.name}\n\n{base['description'].strip()}\n"


def _run_seed_script(workspace: Path) -> None:
    """Run `seed.sh` in the workspace if present.

    Tasks ship `seed.sh` to materialize their seed files in `cwd`
    (workspace) for host execution. The Dockerfile is preserved as the
    Docker-arm equivalent but is not invoked here.
    """
    script = workspace / "seed.sh"
    if not script.is_file():
        return
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seed.sh failed (exit {proc.returncode}):\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )


def _maybe_create_venv(workspace: Path) -> None:
    if not (workspace / "requirements.txt").is_file():
        return
    subprocess.run(
        [sys.executable, "-m", "venv", str(workspace / ".venv")],
        check=True, capture_output=True,
    )


def prepare(task_dir: Path, arm: str, runs_root: Path) -> RunPrep:
    if arm not in ARM_BUDGETS:
        raise ValueError(f"unknown arm: {arm!r}; expected one of {list(ARM_BUDGETS)}")
    if not task_dir.is_dir():
        raise NotADirectoryError(task_dir)
    if not (task_dir / "task.yaml").is_file():
        raise FileNotFoundError(task_dir / "task.yaml")
    if not (task_dir / "tests" / "test_outputs.py").is_file():
        raise FileNotFoundError(task_dir / "tests/test_outputs.py")

    run_id = _new_run_id(task_dir.name, arm)
    workspace = Path(tempfile.mkdtemp(prefix=f"tb_live_v2_{run_id}_"))
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _copy_task_seed(task_dir, workspace)
    _run_seed_script(workspace)
    _maybe_create_venv(workspace)

    description = (workspace / "task.md").read_text()
    prompt = render_prompt(
        workspace=workspace,
        run_dir=run_dir,
        task_description=description,
        budget_lines=ARM_BUDGETS[arm],
    )
    started_at = _utc_now()

    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "workspace_path.txt").write_text(str(workspace))

    return RunPrep(
        run_id=run_id,
        task_id=task_dir.name,
        arm=arm,
        workspace=workspace,
        run_dir=run_dir,
        prompt=prompt,
        started_at=started_at,
    )


@dataclass
class RunResult:
    run_id: str
    final_success: bool | None
    termination_reason: str
    num_events: int
    verifier_exit: int | None


def _run_verifier(workspace: Path, task_dir: Path, run_dir: Path) -> tuple[int | None, str]:
    tests_src = task_dir / "tests"
    tests_dst = workspace / "tests"
    if tests_dst.exists():
        shutil.rmtree(tests_dst)
    shutil.copytree(tests_src, tests_dst)
    # The verifier always runs with the host's Python, NOT the agent's
    # venv. The venv is for the agent's runtime; the verifier checks
    # filesystem state and may need pytest, which the fresh venv lacks.
    env = {**os.environ, "TB_LIVE_V2_WORKSPACE": str(workspace)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short",
         "--rootdir", str(workspace), "-p", "no:cacheprovider", str(tests_dst)],
        cwd=str(workspace),
        env=env,
        capture_output=True, text=True, timeout=120,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    (run_dir / "verifier_output.txt").write_text(out)
    return proc.returncode, out


def finalize(
    prep: RunPrep,
    task_dir: Path,
    skip_sidecar: bool = False,
) -> RunResult:
    transcript_path = prep.run_dir / "transcript.jsonl"
    if not transcript_path.is_file():
        # Subagent never logged. Mark infrastructure_failure and stop.
        result = RunResult(
            run_id=prep.run_id,
            final_success=None,
            termination_reason="infrastructure_failure",
            num_events=0,
            verifier_exit=None,
        )
        _write_manifest(prep, task_dir, result, ended_at=_utc_now())
        return result

    transcript = read_transcript(transcript_path)
    has_done = any(line.get("kind") == "done" for line in transcript)
    events = transcript_to_events(transcript, run_id=prep.run_id)
    if not events:
        result = RunResult(
            run_id=prep.run_id,
            final_success=None,
            termination_reason="infrastructure_failure",
            num_events=0,
            verifier_exit=None,
        )
        _write_manifest(prep, task_dir, result, ended_at=_utc_now())
        return result

    write_events(events, prep.run_dir / "events.jsonl")

    if not skip_sidecar:
        _replay_sidecar(prep.run_dir)

    verifier_exit, _verifier_output = _run_verifier(prep.workspace, task_dir, prep.run_dir)
    if verifier_exit == 0:
        final_success = True
        termination = "verifier_pass"
    else:
        final_success = False
        termination = "verifier_fail" if has_done else "no_done_record"

    result = RunResult(
        run_id=prep.run_id,
        final_success=final_success,
        termination_reason=termination,
        num_events=len(events),
        verifier_exit=verifier_exit,
    )
    _write_manifest(prep, task_dir, result, ended_at=_utc_now())
    observation_events = build_observation_events(
        run_dir=prep.run_dir,
        run_id=prep.run_id,
        task_id=prep.task_id,
    )
    write_observation_events(observation_events, prep.run_dir / "observation_events.jsonl")
    return result


def _replay_sidecar(run_dir: Path) -> None:
    """Invoke ledger_progress.sidecar to convert events.jsonl → ledger.jsonl."""
    abs_run_dir = run_dir.resolve()
    proc = subprocess.run(
        [sys.executable, "-m", "ledger_progress.sidecar",
         "--run-dir", str(abs_run_dir),
         "--input-file", str(abs_run_dir / "events.jsonl")],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2].parent / "coding-progress-ledger"),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sidecar replay failed (exit {proc.returncode}):\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )


def _write_manifest(prep: RunPrep, task_dir: Path, result: RunResult, ended_at: str) -> None:
    spec = _load_task_yaml(task_dir)
    descs = spec.get("descriptions") or []
    base = next((d for d in descs if d.get("key") == "base"), {})
    shape = _maybe_yaml(task_dir / "shape.yaml")
    manifest = {
        "task_id": prep.task_id,
        "task_family": "internal",
        "difficulty": spec.get("difficulty"),
        "category": (spec.get("tags") or [None])[0],
        "subagent_type": "general-purpose",
        "model_name": ARM_MODELS[prep.arm],
        "arm": prep.arm,
        "budget_lines": ARM_BUDGETS[prep.arm],
        "prompt_version": PROMPT_VERSION,
        "start_time": prep.started_at,
        "end_time": ended_at,
        "final_success": result.final_success,
        "final_success_source": "internal_verifier",
        "termination_reason": result.termination_reason,
        "num_ledger_events": result.num_events,
        "has_real_wallclock": True,
        "verifier_exit": result.verifier_exit,
        "target_shape": shape.get("target_shape"),
    }
    (prep.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (prep.run_dir / "run_notes.md").write_text(_render_run_notes(prep, result, manifest))


def _load_task_yaml(task_dir: Path) -> dict:
    import yaml
    return yaml.safe_load((task_dir / "task.yaml").read_text())


def _maybe_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _render_run_notes(prep: RunPrep, result: RunResult, manifest: dict) -> str:
    return (
        f"# Run notes — {prep.run_id}\n\n"
        f"- task_id: `{prep.task_id}`\n"
        f"- arm: `{prep.arm}` (model: `{manifest['model_name']}`, budget_lines: {manifest['budget_lines']})\n"
        f"- final_success: `{result.final_success}`\n"
        f"- termination_reason: `{result.termination_reason}`\n"
        f"- num_ledger_events: {result.num_events}\n"
        f"- verifier_exit: {result.verifier_exit}\n"
        f"- workspace: `{prep.workspace}`\n"
    )
