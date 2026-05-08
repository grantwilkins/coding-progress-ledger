from __future__ import annotations

import json
import os
import subprocess
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from coding_data_collection.agents.base import AgentBudget
from coding_data_collection.agents.model_client import ModelClientConfig, ProviderModelClient, ScriptedModelClient
from coding_data_collection.agents.model_tool_loop import ModelToolLoopBackend
from coding_data_collection import agent_preflight
from coding_data_collection import docker_substrate
from coding_data_collection.agent_preflight import infer_agent_readiness_checks
from coding_data_collection.artifacts import write_json
from coding_data_collection.observation import read_jsonl, write_jsonl
from coding_data_collection.recording import RunRecorder
from coding_data_collection.sandbox.docker_executor import ToolResult
from coding_data_collection.sandbox.path_guard import PathGuard


class FakeSandbox:
    def shell(self, command: str, *, timeout_s: int) -> ToolResult:
        del timeout_s
        if "pytest" in command:
            return ToolResult(exit_code=1, stdout="FAILED visible check\n", stderr="AssertionError\n", duration_ms=7)
        return ToolResult(exit_code=0, stdout="ok\n", duration_ms=3)

    def list_dir(self, path: str, *, timeout_s: int) -> ToolResult:
        del path, timeout_s
        return ToolResult(exit_code=0, stdout="app.py\n")

    def read_file(
        self,
        path: str,
        *,
        timeout_s: int,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        del path, timeout_s
        if start_line or end_line:
            return ToolResult(exit_code=0, stdout=f"Lines: {start_line}-{end_line} of 20\n")
        return ToolResult(exit_code=0, stdout="print('bug')\n")

    def write_file(self, path: str, content: str, *, timeout_s: int) -> ToolResult:
        del path, content, timeout_s
        return ToolResult(exit_code=0, stdout="written\n")

    def edit_file(self, path: str, instruction: str, *, timeout_s: int) -> ToolResult:
        del path, instruction, timeout_s
        return ToolResult(exit_code=2, stderr="unsupported\n")

    def find_files(self, pattern: str, *, path: str = ".", timeout_s: int) -> ToolResult:
        del pattern, path, timeout_s
        return ToolResult(exit_code=0, stdout="app.py\n")

    def grep(self, pattern: str, *, path: str = ".", file_glob: str | None = None, timeout_s: int) -> ToolResult:
        del pattern, path, file_glob, timeout_s
        return ToolResult(exit_code=0, stdout="app.py:1:print('bug')\n")

    def apply_patch(self, unified_diff: str, *, timeout_s: int) -> ToolResult:
        del unified_diff, timeout_s
        return ToolResult(exit_code=0, stdout="patching file app.py\n")


def test_model_tool_loop_records_real_multi_step_trajectory(tmp_path: Path) -> None:
    actions = [
        {"thought": "Inspect files.", "action": {"type": "list_dir", "path": "."}},
        {"thought": "Open the implementation.", "action": {"type": "read_file", "path": "app.py"}},
        {
            "thought": "Patch the implementation.",
            "action": {"type": "write_file", "path": "app.py", "content": "print('fixed')\n"},
        },
        {"thought": "Run visible tests.", "action": {"type": "shell", "command": "pytest -q"}},
        {"thought": "Stop after observing the failing check.", "action": {"type": "done", "summary": "attempt complete"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    assert backend.eligible_for_L_gate is False
    assert backend.pilot_type == "scripted_model_smoke"

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(max_steps=10, max_tool_time_s=5),
        recorder=recorder,
    )
    recorder.write_derived_artifacts(verifier_exit_code=1)

    transcript = read_jsonl(tmp_path / "transcript.jsonl")
    observations = read_jsonl(tmp_path / "observation_events.jsonl")

    assert result.completed is True
    assert len(transcript) == 10
    assert any(row["kind"] == "write_file" for row in transcript)
    assert any(event["event_type"] == "product_file_written" for event in observations)
    assert any(event["event_type"] == "validation_attempt" for event in observations)
    assert any(event["event_type"] == "validation_fail_observed" for event in observations)
    assert max(event["step"] for event in observations if event["event_type"] == "verifier_fail") == 11


def test_agent_preflight_fails_when_hidden_image_paths_are_readable(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="agent can read hidden image path(s): /protected\n",
        )

    monkeypatch.setattr(agent_preflight.subprocess, "run", fake_run)
    report = agent_preflight.run_agent_readiness_preflight(
        image_tag="image",
        workspace_dir=tmp_path,
        task_md="Solve the task.",
    )

    assert report["passed"] is False
    assert report["failed_checks"] == ["hidden_image_artifacts_unreadable"]
    assert "--entrypoint" in commands[0]
    assert " /protected /oracle /verifier /gold /solution /tests /test" in commands[0][-1]


def test_agent_preflight_timeout_handles_bytes_output(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        del command
        raise subprocess.TimeoutExpired(cmd="docker", timeout=kwargs["timeout"], output=b"partial", stderr=b"slow")

    monkeypatch.setattr(agent_preflight.subprocess, "run", fake_run)
    report = agent_preflight.run_agent_readiness_preflight(
        image_tag="image",
        workspace_dir=tmp_path,
        task_md="Solve the task.",
    )

    assert report["passed"] is False
    assert report["results"][0]["exit_code"] == 124
    assert "partial" in report["results"][0]["stdout_snippet"]
    assert "preflight timed out" in report["results"][0]["stderr_snippet"]


def test_hydrate_workspace_from_image_app_merges_visible_generated_files(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "agent_workspace"
    workspace.mkdir()
    (workspace / "existing.txt").write_text("prepared\n", encoding="utf-8")

    def fake_check_output(command, **kwargs):
        del command, kwargs
        return "container-id\n"

    def fake_run(command, **kwargs):
        del kwargs
        if command[:2] == ["docker", "cp"]:
            snapshot = Path(command[3])
            (snapshot / "archive.tar").write_text("archive\n", encoding="utf-8")
            (snapshot / "existing.txt").write_text("image copy\n", encoding="utf-8")
            (snapshot / "Dockerfile").write_text("hidden\n", encoding="utf-8")
            (snapshot / "protected").mkdir()
            (snapshot / "protected" / "secret.py").write_text("secret\n", encoding="utf-8")
            (snapshot / "canary.txt").write_text("terminal-bench-canary secret\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker_substrate.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(docker_substrate.subprocess, "run", fake_run)

    report = docker_substrate.hydrate_workspace_from_image_app(image_tag="image", workspace=workspace)

    assert (workspace / "archive.tar").read_text(encoding="utf-8") == "archive\n"
    assert (workspace / "existing.txt").read_text(encoding="utf-8") == "prepared\n"
    assert not (workspace / "Dockerfile").exists()
    assert not (workspace / "protected").exists()
    assert not (workspace / "canary.txt").exists()
    assert report["copied_from_image_app"] == ["archive.tar"]


def test_model_tool_loop_retries_invalid_model_output(tmp_path: Path) -> None:
    actions = [
        "not json",
        {"thought_summary": "Bad action.", "action": {"type": "nope"}},
        {"thought_summary": "Recover.", "action": {"type": "done", "summary": "stopped"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(max_steps=10, max_tool_time_s=5),
        recorder=recorder,
    )
    transcript = read_jsonl(tmp_path / "transcript.jsonl")

    assert result.completed is True
    assert [row["kind"] for row in transcript[:2]] == ["model_parse_error", "invalid_action"]
    assert transcript[-1]["kind"] == "done"


def test_model_tool_loop_supports_search_chunked_read_and_patch(tmp_path: Path) -> None:
    actions = [
        {"thought": "Find Python files.", "action": {"type": "find_files", "pattern": "*.py"}},
        {"thought": "Search for the bug.", "action": {"type": "grep", "pattern": "bug", "path": ".", "file_glob": "*.py"}},
        {
            "thought": "Read the focused chunk.",
            "action": {"type": "read_file", "path": "app.py", "start_line": 1, "end_line": 5},
        },
        {
            "thought": "Patch the bug.",
            "action": {
                "type": "apply_patch",
                "unified_diff": "--- app.py\n+++ app.py\n@@\n-print('bug')\n+print('fixed')\n",
            },
        },
        {"thought": "Run visible tests.", "action": {"type": "shell", "command": "pytest -q"}},
        {"thought": "Stop after validation.", "action": {"type": "done", "summary": "attempt complete"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(max_steps=10, max_tool_time_s=5),
        recorder=recorder,
    )
    recorder.write_derived_artifacts(verifier_exit_code=1)
    transcript = read_jsonl(tmp_path / "transcript.jsonl")
    observations = read_jsonl(tmp_path / "observation_events.jsonl")

    assert result.completed is True
    assert {"find_files", "grep", "apply_patch"}.issubset({row["kind"] for row in transcript})
    assert any(row["kind"] == "read_file" and row["start_line"] == 1 and row["end_line"] == 5 for row in transcript)
    assert any(event["event_type"] == "chunked_file_read" for event in observations)


def test_model_tool_loop_classifies_network_and_dependency_blocks(tmp_path: Path) -> None:
    class BlockingSandbox(FakeSandbox):
        def shell(self, command: str, *, timeout_s: int) -> ToolResult:
            del command, timeout_s
            return ToolResult(
                exit_code=1,
                stderr="Temporary failure resolving deb.debian.org\nModuleNotFoundError: No module named 'x'\n",
            )

    actions = [
        {"thought": "Try setup.", "action": {"type": "shell", "command": "apt-get update && python -c 'import x'"}},
        {"thought": "Blocked.", "action": {"type": "done", "summary": "cannot proceed because network is disabled"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=BlockingSandbox(),
        budget=AgentBudget(max_steps=10, max_tool_time_s=5),
        recorder=recorder,
    )
    recorder.write_derived_artifacts(verifier_exit_code=1)
    transcript = read_jsonl(tmp_path / "transcript.jsonl")
    observations = read_jsonl(tmp_path / "observation_events.jsonl")

    assert result.completed is True
    assert {"network_blocked", "dependency_missing"}.issubset({row["kind"] for row in transcript})
    assert {"network_blocked", "dependency_missing"}.issubset({event["event_type"] for event in observations})


def test_model_tool_loop_rejects_early_done_until_depth_and_validation(tmp_path: Path) -> None:
    actions = [
        {"thought_summary": "Stop immediately.", "action": {"type": "done", "summary": "done"}},
        {"thought_summary": "Run a visible check.", "action": {"type": "shell", "command": "pytest -q"}},
        {"thought_summary": "Stop after validation.", "action": {"type": "done", "summary": "attempt complete"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(
            max_steps=10,
            max_tool_time_s=5,
            min_steps_before_done=4,
            require_validation_before_done=True,
        ),
        recorder=recorder,
    )
    recorder.write_derived_artifacts(verifier_exit_code=1)
    transcript = read_jsonl(tmp_path / "transcript.jsonl")
    observations = read_jsonl(tmp_path / "observation_events.jsonl")

    assert result.completed is True
    assert [row["kind"] for row in transcript[:2]] == ["thought", "early_done_denied"]
    assert transcript[-1]["kind"] == "done"
    assert any(event["event_type"] == "early_done_denied" for event in observations)
    assert any(event["event_type"] == "validation_attempt" for event in observations)


def test_model_tool_loop_treats_pip_dry_run_as_validation(tmp_path: Path) -> None:
    actions = [
        {
            "thought_summary": "Validate pip.",
            "action": {
                "type": "shell",
                "command": "pip --version && pip install --dry-run --no-index wheel.whl",
            },
        },
        {"thought_summary": "Stop after validation.", "action": {"type": "done", "summary": "pip repaired"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix pip.",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(
            max_steps=10,
            max_tool_time_s=5,
            min_steps_before_done=2,
            require_validation_before_done=True,
        ),
        recorder=recorder,
    )
    recorder.write_derived_artifacts(verifier_exit_code=1)
    transcript = read_jsonl(tmp_path / "transcript.jsonl")
    observations = read_jsonl(tmp_path / "observation_events.jsonl")

    assert result.completed is True
    assert transcript[-1]["kind"] == "done"
    assert any(event["event_type"] == "validation_attempt" for event in observations)


def test_model_tool_loop_treats_custom_validation_output_as_validation(tmp_path: Path) -> None:
    actions = [
        {
            "thought_summary": "Run custom check.",
            "action": {"type": "shell", "command": "python - <<'PY'\nprint('final_validation True')\nPY"},
        },
        {"thought_summary": "Stop after validation.", "action": {"type": "done", "summary": "custom validation passed"}},
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Implement solver.",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(
            max_steps=10,
            max_tool_time_s=5,
            min_steps_before_done=2,
            require_validation_before_done=True,
        ),
        recorder=recorder,
    )

    assert result.completed is True
    assert read_jsonl(tmp_path / "transcript.jsonl")[-1]["kind"] == "done"


def test_model_tool_loop_allows_blocked_done_before_depth(tmp_path: Path) -> None:
    actions = [
        {
            "thought_summary": "The task appears blocked.",
            "action": {
                "type": "done",
                "summary": "Cannot proceed because required task files are missing.",
            },
        }
    ]
    recorder = RunRecorder(run_dir=tmp_path, run_id="run1")
    backend = ModelToolLoopBackend(ScriptedModelClient(actions))

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(
            max_steps=10,
            max_tool_time_s=5,
            min_steps_before_done=15,
            require_validation_before_done=True,
            allow_blocked_done=True,
        ),
        recorder=recorder,
    )
    transcript = read_jsonl(tmp_path / "transcript.jsonl")

    assert result.completed is True
    assert [row["kind"] for row in transcript] == ["thought", "done"]


def test_provider_model_client_is_pilot_eligible_and_records_usage(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "\n".join(
            [
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({",
                "  'action': {'thought_summary': 'Provider action.', 'action': {'type': 'done', 'summary': 'done'}},",
                "  'usage': {'tokens_in': 11, 'tokens_out': 7, 'estimated_cost_usd': 0.02}",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="command", model_name="real"),
    )
    backend = ModelToolLoopBackend(client, model_name="real")
    recorder = RunRecorder(run_dir=tmp_path / "run", run_id="run1")

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(max_steps=10, max_tool_time_s=5),
        recorder=recorder,
    )

    assert backend.eligible_for_L_gate is True
    assert backend.pilot_type == "real_agent_pilot"
    assert result.completed is True
    assert result.tokens_in == 11
    assert result.tokens_out == 7
    assert result.estimated_cost_usd == 0.02


def test_provider_model_client_can_be_forced_l_ineligible_for_smoke() -> None:
    client = ScriptedModelClient([], model_name="fake-provider")
    client.provider_backed = True
    backend = ModelToolLoopBackend(
        client,
        model_name="fake-provider",
        eligible_for_l_gate=False,
        pilot_type="openrouter_free_smoke",
    )

    assert backend.eligible_for_L_gate is False
    assert backend.pilot_type == "openrouter_free_smoke"


def test_provider_model_client_records_provider_routing_metadata(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "\n".join(
            [
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({",
                "  'action': {'thought_summary': 'Provider action.', 'action': {'type': 'done', 'summary': 'done'}},",
                "  'usage': {'tokens_in': 11, 'tokens_out': 7, 'estimated_cost_usd': 0.02, 'cost_credits': 0.03},",
                "  'provider': {",
                "    'adapter': 'openai_compatible_chat_completions',",
                "    'requested_model': 'baidu/cobuddy:free',",
                "    'resolved_model': 'baidu/cobuddy:free',",
                "    'fallback_used': False,",
                "    'provider_routing_policy': {'allow_fallbacks': False}",
                "  }",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="openrouter", model_name="baidu/cobuddy:free"),
    )

    raw = client.next_action(
        system_prompt="system",
        task_prompt="task",
        transcript_prefix=[],
        tool_specs=[],
        budget_state={},
    )

    assert isinstance(raw, dict)
    metrics = client.metrics()
    assert metrics["resolved_models"] == ["baidu/cobuddy:free"]
    assert metrics["fallback_call_count"] == 0
    assert metrics["total_cost_credits"] == 0.03
    assert metrics["last_provider_call"]["provider_routing_policy"] == {"allow_fallbacks": False}


def test_provider_model_client_adapter_failure_is_not_done_action(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text("import sys\nsys.stderr.write('bad response')\nsys.exit(1)\n", encoding="utf-8")
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="command", model_name="real"),
    )

    raw = client.next_action(
        system_prompt="system",
        task_prompt="task",
        transcript_prefix=[],
        tool_specs=[],
        budget_state={},
    )

    assert isinstance(raw, dict)
    assert raw["provider_adapter_error"]["message"] == "provider adapter exited 1"
    assert raw["provider_adapter_error"]["stderr_snippet"] == "bad response"


def test_provider_model_client_timeout_is_not_uncaught(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], stderr="slow provider")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ProviderModelClient(
        command="provider-command",
        config=ModelClientConfig(provider="command", model_name="real", timeout_s=7),
    )

    raw = client.next_action(
        system_prompt="system",
        task_prompt="task",
        transcript_prefix=[],
        tool_specs=[],
        budget_state={},
    )

    assert isinstance(raw, dict)
    assert raw["provider_adapter_error"]["message"] == "provider adapter timed out after 7s"
    assert client.metrics()["total_model_calls"] == 1


def test_model_tool_loop_records_provider_adapter_error_as_typed_event(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text("import sys\nsys.stderr.write('bad provider route')\nsys.exit(1)\n", encoding="utf-8")
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="command", model_name="real"),
    )
    recorder = RunRecorder(run_dir=tmp_path / "run", run_id="run1")
    backend = ModelToolLoopBackend(client, model_name="real")

    result = backend.run(
        run_id="run1",
        task_md="Fix app.py",
        workspace_dir=tmp_path,
        sandbox=FakeSandbox(),
        budget=AgentBudget(max_steps=1, max_tool_time_s=5),
        recorder=recorder,
    )
    transcript = read_jsonl(tmp_path / "run" / "transcript.jsonl")

    assert result.completed is False
    assert transcript[0]["kind"] == "provider_adapter_error"
    assert transcript[0]["stderr_snippet"] == "bad provider route"


def test_run_model_agent_pilot_provider_route_l_eligibility_requires_metadata(tmp_path: Path) -> None:
    module = _load_script_module("run_model_agent_pilot.py")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "\n".join(
            [
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "print(json.dumps({",
                "  'action': {'thought_summary': 'Provider action.', 'action': {'type': 'done', 'summary': 'done'}},",
                "  'usage': {'tokens_in': 11, 'tokens_out': 7},",
                "  'provider': {'resolved_model': 'fixed', 'fallback_used': False}",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="command", model_name="real"),
    )

    preflight = module._run_provider_route_preflight(client, task_md="Fix app.py")

    assert preflight["passed"] is True
    assert module._provider_route_l_eligible(client.metrics()) is True
    assert module._provider_route_l_eligible({"provider_calls": [], "total_tokens_in": 0}) is False


def test_run_model_agent_pilot_provider_route_preflight_rejects_adapter_error(tmp_path: Path) -> None:
    module = _load_script_module("run_model_agent_pilot.py")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("import sys\nsys.stderr.write('HTTP 400 unsupported response_format')\nsys.exit(1)\n", encoding="utf-8")
    client = ProviderModelClient(
        command=f"{sys.executable} {adapter}",
        config=ModelClientConfig(provider="openrouter", model_name="bad-route"),
    )

    preflight = module._run_provider_route_preflight(client, task_md="Fix app.py")

    assert preflight["passed"] is False
    assert any("provider adapter exited 1" in issue for issue in preflight["issues"])
    assert any("unsupported response_format" in issue for issue in preflight["issues"])


def test_execute_pilot_plan_preflights_provider_arms_before_runs(tmp_path: Path) -> None:
    module = _load_script_module("execute_pilot_plan_continue.py")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("import sys\nsys.stderr.write('route unavailable')\nsys.exit(1)\n", encoding="utf-8")
    plan = {
        "arms": [
            {
                "name": "bad",
                "backend": "model_tool_loop",
                "client": "provider",
                "model_provider": "openrouter",
                "model": "bad-route",
                "model_command": f"{sys.executable} {adapter}",
            }
        ]
    }

    report = module._preflight_provider_arms(plan)

    assert report["passed"] is False
    assert report["arms"][0]["name"] == "bad"
    assert any("provider adapter exited 1" in issue for issue in report["arms"][0]["issues"])


def test_openai_adapter_builds_structured_response_payload() -> None:
    module = _load_script_module("openai_model_client.py")

    payload = module.build_openai_payload(
        {
            "system_prompt": "system",
            "task_prompt": "task",
            "transcript_prefix": [{"step": 1}],
            "tool_specs": [{"type": "shell"}],
            "budget_state": {"steps_used": 1},
            "model": {"name": "gpt-5.1", "max_tokens_out": 123},
        }
    )

    assert payload["model"] == "gpt-5.1"
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["properties"]["action"]["properties"]["type"]["enum"] == [
        "list_dir",
        "find_files",
        "grep",
        "read_file",
        "write_file",
        "edit_file",
        "apply_patch",
        "shell",
        "done",
    ]


def test_openai_adapter_extracts_action_and_usage_shape() -> None:
    module = _load_script_module("openai_model_client.py")

    action = module.extract_action(
        {
            "output": [
                {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "thought_summary": "Inspect.",
                                    "action": {
                                        "type": "list_dir",
                                        "path": ".",
                                        "command": None,
                                        "content": None,
                                        "instruction": None,
                                        "summary": None,
                                    },
                                }
                            )
                        }
                    ]
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 6},
        }
    )

    assert action["action"]["type"] == "list_dir"
    assert action["action"]["path"] == "."


def test_openai_adapter_main_emits_provider_metadata(monkeypatch, capsys) -> None:
    module = _load_script_module("openai_model_client.py")
    response = {
        "id": "resp-1",
        "model": "gpt-5.4-20260401",
        "output_text": json.dumps(
            {
                "thought_summary": "Inspect.",
                "action": {
                    "type": "read_file",
                    "path": "task.md",
                    "command": None,
                    "content": None,
                    "instruction": None,
                    "summary": None,
                    "start_line": None,
                    "end_line": None,
                    "pattern": None,
                    "file_glob": None,
                    "unified_diff": None,
                },
            }
        ),
        "usage": {"input_tokens": 5, "output_tokens": 6},
    }

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(module, "call_openai", lambda *args, **kwargs: response)
    monkeypatch.setattr(
        sys,
        "stdin",
        SimpleNamespace(
            read=lambda: json.dumps(
                {
                    "system_prompt": "system",
                    "task_prompt": "task",
                    "transcript_prefix": [],
                    "tool_specs": [],
                    "budget_state": {},
                    "model": {"name": "gpt-5.4", "max_tokens_out": 100},
                }
            )
        ),
    )

    assert module.main(["--model", "gpt-5.4"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["usage"]["tokens_in"] == 5
    assert payload["usage"]["tokens_out"] == 6
    assert payload["provider"]["adapter"] == "openai_responses"
    assert payload["provider"]["requested_model"] == "gpt-5.4"
    assert payload["provider"]["resolved_model"] == "gpt-5.4-20260401"
    assert payload["provider"]["fallback_used"] is False


def test_openai_adapter_extracts_first_valid_action_from_multiple_output_chunks() -> None:
    module = _load_script_module("openai_model_client.py")

    action = module.extract_action(
        {
            "output": [
                {"type": "reasoning"},
                {
                    "type": "message",
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "thought_summary": "Inspect first.",
                                    "action": {
                                        "type": "read_file",
                                        "path": "task.md",
                                        "command": None,
                                        "content": None,
                                        "instruction": None,
                                        "summary": None,
                                    },
                                }
                            )
                        }
                    ],
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "thought_summary": "A later chunk should not be concatenated.",
                                    "action": {
                                        "type": "shell",
                                        "path": None,
                                        "command": "pytest -q",
                                        "content": None,
                                        "instruction": None,
                                        "summary": None,
                                    },
                                }
                            )
                        }
                    ],
                },
            ]
        }
    )

    assert action["thought_summary"] == "Inspect first."
    assert action["action"]["type"] == "read_file"


def test_openai_compatible_adapter_builds_openrouter_payload() -> None:
    module = _load_script_module("openai_compatible_model_client.py")
    args = SimpleNamespace(
        model=None,
        fallback_model=["qwen/qwen3-coder"],
        allow_fallbacks=False,
        require_parameters=True,
        data_collection="deny",
        provider_order=[],
        provider_only=[],
        provider_ignore=[],
        max_price_input="0.1",
        max_price_output="0.2",
        response_format="json_schema",
    )

    payload = module.build_chat_payload(
        {
            "system_prompt": "system",
            "task_prompt": "task",
            "transcript_prefix": [{"step": 1}],
            "tool_specs": [{"type": "shell"}],
            "budget_state": {"steps_used": 1},
            "model": {"name": "baidu/cobuddy:free", "max_tokens_out": 123, "temperature": 0.0},
        },
        args=args,
    )

    assert payload["model"] == "baidu/cobuddy:free"
    assert payload["models"] == ["baidu/cobuddy:free", "qwen/qwen3-coder"]
    assert payload["provider"] == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "max_price": {"input": 0.1, "output": 0.2},
    }
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert "Do not echo a JSON schema" in payload["messages"][0]["content"]
    assert '"additionalProperties":' not in payload["messages"][0]["content"]


def test_openai_compatible_adapter_extracts_action_usage_and_provider_payload() -> None:
    module = _load_script_module("openai_compatible_model_client.py")
    response = {
        "id": "gen-1",
        "model": "baidu/cobuddy:free",
        "choices": [
            {
                "message": {
                    "content": "```json\n"
                    + json.dumps(
                        {
                            "thought_summary": "Inspect.",
                            "action": {
                                "type": "read_file",
                                "path": "task.md",
                                "command": None,
                                "content": None,
                                "instruction": None,
                                "summary": None,
                                "start_line": None,
                                "end_line": None,
                                "pattern": None,
                                "file_glob": None,
                                "unified_diff": None,
                            },
                        }
                    )
                    + "\n```"
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11, "cost": 0},
    }

    action = module.extract_action(response)
    usage = module._usage_payload(response["usage"])
    provider = module._provider_payload(
        response,
        requested_model="openrouter/auto",
        fallback_models=[],
        provider_policy={"allow_fallbacks": False},
        base_url="https://openrouter.ai/api/v1",
    )

    assert action["action"]["type"] == "read_file"
    assert usage["tokens_in"] == 5
    assert usage["tokens_out"] == 6
    assert usage["estimated_cost_usd"] is None
    assert usage["cost_credits"] == 0
    assert provider["requested_model"] == "openrouter/auto"
    assert provider["resolved_model"] == "baidu/cobuddy:free"
    assert provider["fallback_used"] is False
    assert provider["model_alias_or_route_changed"] is True


def test_openai_compatible_adapter_extracts_first_action_from_repeated_json_objects() -> None:
    module = _load_script_module("openai_compatible_model_client.py")
    first = {
        "thought_summary": "Inspect the task.",
        "action": {
            "type": "read_file",
            "path": "task.md",
            "command": None,
            "content": None,
            "instruction": None,
            "summary": None,
            "start_line": None,
            "end_line": None,
            "pattern": None,
            "file_glob": None,
            "unified_diff": None,
        },
    }
    second = {
        "thought_summary": "Run tests.",
        "action": {
            "type": "shell",
            "path": None,
            "command": "pytest -q",
            "content": None,
            "instruction": None,
            "summary": None,
            "start_line": None,
            "end_line": None,
            "pattern": None,
            "file_glob": None,
            "unified_diff": None,
        },
    }

    action = module.extract_action(
        {
            "id": "gen-1",
            "model": "openai/gpt-5.4-20260305",
            "choices": [{"message": {"content": json.dumps(first) + "\n\n" + json.dumps(second)}}],
        }
    )

    assert action == first


def test_openai_compatible_adapter_does_not_count_alias_resolution_as_fallback() -> None:
    module = _load_script_module("openai_compatible_model_client.py")
    provider = module._provider_payload(
        {"id": "gen-1", "model": "baidu/cobuddy-20260430:free"},
        requested_model="baidu/cobuddy:free",
        fallback_models=[],
        provider_policy={"allow_fallbacks": False},
        base_url="https://openrouter.ai/api/v1",
    )

    assert provider["model_alias_or_route_changed"] is True
    assert provider["fallback_used"] is False


def test_openai_compatible_adapter_response_excerpt_for_bad_model_output() -> None:
    module = _load_script_module("openai_compatible_model_client.py")
    excerpt = module._response_excerpt(
        {
            "id": "gen-1",
            "model": "baidu/cobuddy:free",
            "choices": [{"message": {"content": "not an action"}}],
        }
    )

    payload = json.loads(excerpt)
    assert payload["model"] == "baidu/cobuddy:free"
    assert payload["content_excerpt"] == ["not an action"]


def test_task_yaml_instruction_is_materialized_without_canary(tmp_path: Path) -> None:
    module = _load_script_module("run_model_agent_pilot.py")
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "\n".join(
            [
                "# Terminal-Bench Canary String, DO NOT REMOVE:",
                "# terminal-bench-canary GUID secret",
                "instruction: |-",
                "  Fix the visible bug.",
                "  Write answer.txt.",
                "other: hidden",
            ]
        ),
        encoding="utf-8",
    )

    task_md = module._task_md_from_yaml(task_yaml, task_dir_name="task")

    assert "Fix the visible bug." in task_md
    assert "Write answer.txt." in task_md
    assert "terminal-bench-canary" not in task_md
    assert "hidden" not in task_md


def test_audit_real_model_run_passes_mocked_complete_provider_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("print('fixed')\n", encoding="utf-8")
    write_jsonl(
        run_dir / "transcript.jsonl",
        [
            {"step": 1, "kind": "thought", "summary": "Inspect.", "visible_to_agent": True},
            {"step": 2, "kind": "read_file", "path": "task.md", "stdout_snippet": "task", "visible_to_agent": True},
            {"step": 3, "kind": "write_file", "path": "app.py", "stdout_snippet": "written", "visible_to_agent": True},
            {
                "step": 4,
                "kind": "shell",
                "command": "pytest -q",
                "exit_code": 1,
                "stdout_snippet": "FAILED",
                "stderr_snippet": "AssertionError",
                "visible_to_agent": True,
            },
            {"step": 5, "kind": "done", "summary": "attempt complete", "visible_to_agent": True},
        ],
    )
    write_jsonl(run_dir / "events.jsonl", [{"step": 1}])
    write_jsonl(run_dir / "ledger.jsonl", [{"step": 1}])
    write_jsonl(
        run_dir / "observation_events.jsonl",
        [
            {"step": 3, "event_type": "product_file_written", "payload": {"visible_to_agent": True}},
            {"step": 4, "event_type": "validation_attempt", "payload": {"visible_to_agent": True}},
            {"step": 6, "event_type": "verifier_fail", "payload": {"visible_to_agent": False}},
        ],
    )
    (run_dir / "progress.csv").write_text("step,coding_progress\n1,0.1\n", encoding="utf-8")
    (run_dir / "progress_by_category.csv").write_text("step,coding_progress\n1,0.1\n", encoding="utf-8")
    (run_dir / "verifier_output.txt").write_text("failed\n", encoding="utf-8")
    (run_dir / "task.md").write_text("Visible task\n", encoding="utf-8")
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_status": "completed_failure",
            "final_success": False,
            "metrics": {
                "agent_backend": "model_tool_loop",
                "pilot_type": "real_agent_pilot",
                "eligible_for_L_gate": True,
                "model_provider": "command",
                "model_name": "real",
                "temperature": 0.0,
                "max_model_calls": 40,
                "max_tool_calls": 40,
                "max_wall_time_s": 1800,
                "max_tokens_out": 2048,
                "total_model_calls": 5,
                "total_tokens_in": 100,
                "total_tokens_out": 50,
            },
        },
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_real_model_run.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(run_dir), "--min-transcript-steps", "5"],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["passed"] is True


def _load_script_module(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_path_guard_rejects_escape_and_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("ok\n", encoding="utf-8")
    (workspace / "link").symlink_to(tmp_path)
    guard = PathGuard(workspace)

    assert guard.resolve("safe.txt") == (workspace / "safe.txt").resolve()

    for bad in ("../outside.txt", "/etc/passwd", "link/escaped.txt"):
        try:
            guard.resolve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"path was not rejected: {bad}")


def test_agent_readiness_infers_runtime_checks_from_task_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "code.py").write_text("import numpy as np\nimport torch\n", encoding="utf-8")

    checks = infer_agent_readiness_checks(
        task_md="Install Nginx and write your code in R.",
        workspace_dir=workspace,
    )

    by_id = {check.check_id: check for check in checks}
    assert {"nginx_available", "r_runtime_available", "python_imports_available", "no_solve_time_network_install"}.issubset(by_id)
    assert "import numpy; import torch" in by_id["python_imports_available"].command
    assert by_id["no_solve_time_network_install"].expected == "manual_review"
