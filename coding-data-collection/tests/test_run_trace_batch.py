from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_trace_batch_executes_explicit_plan_without_default_experiment_controls(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_trace_batch.py")
    plan_path = tmp_path / "plan.json"
    out_path = tmp_path / "execution.json"
    plan_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "client": "provider",
                    "model_provider": "openrouter",
                    "model_name": "fixed/model",
                    "cpus": 2,
                },
                "runs": [
                    {
                        "run_id": "trace-1",
                        "task_dir": "tasks/one",
                        "run_dir": "runs/one",
                        "image_tag": "trace-one:latest",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, text: bool) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main([str(plan_path), "--out", str(out_path)]) == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    assert payload["nonzero_count"] == 0
    assert payload["results"][0]["run_id"] == "trace-1"
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == [module.sys.executable, "scripts/run_model_agent_trace.py"]
    assert "--task-dir" in command
    assert "--run-dir" in command
    assert "--image-tag" in command
    assert "--require-validation-before-done" not in command
    assert "--min-steps-before-done" not in command


def test_trace_batch_requires_one_explicit_run() -> None:
    module = _load_script_module("run_trace_batch.py")

    try:
        module._plan_runs({"runs": []})
    except ValueError as exc:
        assert "non-empty 'runs' list" in str(exc)
    else:
        raise AssertionError("empty plan should fail")


def _load_script_module(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
