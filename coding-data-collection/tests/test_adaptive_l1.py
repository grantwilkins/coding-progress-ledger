from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from coding_data_collection.artifacts import write_json


def test_adaptive_l1_classifies_eligible_and_replaceable_runs(tmp_path: Path) -> None:
    module = _load_script_module("run_adaptive_l1.py")
    eligible = tmp_path / "eligible"
    eligible.mkdir()
    write_json(
        eligible / "run_manifest.json",
        {
            "run_status": "completed_failure",
            "termination_reason": "verifier_fail",
            "metrics": {"eligible_for_L_gate": True},
        },
    )
    failed = tmp_path / "failed"
    failed.mkdir()
    write_json(
        failed / "run_manifest.json",
        {
            "run_status": "environment_setup_failure",
            "termination_reason": "provider_route_preflight_failed",
            "metrics": {"eligible_for_L_gate": False},
        },
    )

    assert module.classify_run(eligible)["accepted"] is True
    rejected = module.classify_run(failed)
    assert rejected["accepted"] is False
    assert rejected["kind"] == "replaceable_preflight_or_setup_failure"


def test_adaptive_l1_stops_before_runs_when_provider_arm_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_adaptive_l1.py")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "passed": True,
                "planned_runs": [
                    {
                        "run_id": "run1",
                        "task_id": "task",
                        "arm": "bad",
                        "command": [sys.executable, "runner.py", "--run-dir", str(tmp_path / "run1")],
                    }
                ],
                "arms": [{"name": "bad", "backend": "model_tool_loop", "client": "provider"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_preflight_provider_arms",
        lambda plan: {"passed": False, "arms": [{"name": "bad", "model": "bad", "issues": ["no json"]}]},
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("run command should not execute")

    monkeypatch.setattr(subprocess, "run", fail_run)

    rc = module.main(
        [
            str(plan),
            "--out",
            str(tmp_path / "execution.json"),
            "--accepted-out",
            str(tmp_path / "accepted.json"),
            "--rejected-out",
            str(tmp_path / "rejected.json"),
            "--gate-out",
            str(tmp_path / "gate.json"),
            "--failure-out",
            str(tmp_path / "failure.md"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--corpus-id",
            "adaptive",
        ]
    )

    assert rc == 2
    rejected = json.loads((tmp_path / "rejected.json").read_text(encoding="utf-8"))
    assert rejected["runs"][0]["kind"] == "provider_arm_preflight_failed"


def test_adaptive_l1_hard_safety_excludes_validation_precision() -> None:
    module = _load_script_module("run_adaptive_l1.py")
    hardening = {
        "artifact_completeness": {"passed": True},
        "redaction": {"passed": True},
        "validation_attempt_precision": {"passed": False},
        "passed": False,
    }

    assert module._hard_safety_passed(hardening) is True


def test_adaptive_l1_reuses_existing_eligible_runs(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_adaptive_l1.py")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_status": "completed_success",
            "termination_reason": "verifier_pass",
            "metrics": {"eligible_for_L_gate": True},
        },
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "passed": True,
                "arms": [],
                "planned_runs": [
                    {
                        "run_id": "run1",
                        "task_id": "task",
                        "arm": "arm",
                        "command": [sys.executable, "runner.py", "--run-dir", str(run_dir)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_build_and_gate", lambda args, run_dirs: {"passed": True, "gate_inputs": {}})
    monkeypatch.setattr(
        module,
        "corpus_hardening_report",
        lambda run_dirs: {"artifact_completeness": {"passed": True}, "redaction": {"passed": True}},
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("existing eligible run should be reused")

    monkeypatch.setattr(subprocess, "run", fail_run)

    rc = module.main(
        [
            str(plan),
            "--out",
            str(tmp_path / "execution.json"),
            "--accepted-out",
            str(tmp_path / "accepted.json"),
            "--rejected-out",
            str(tmp_path / "rejected.json"),
            "--gate-out",
            str(tmp_path / "gate.json"),
            "--failure-out",
            str(tmp_path / "failure.md"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--corpus-id",
            "adaptive",
            "--target-eligible-runs",
            "1",
        ]
    )

    assert rc == 0
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["accepted_count"] == 1
    assert execution["target_eligible_runs_met"] is True


def _load_script_module(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
