"""
Claim:
The Docker substrate separates agent-visible work from verifier-only benchmark
materials. The agent phase gets a writable product workspace and read-only task
spec, network is disabled unless explicitly allowed, environment manifests
record reproducibility limits and hashes, and the verifier phase starts from a
clean copy of the agent workspace with hidden tests mounted only after the
agent phase has ended.

Plausible wrong implementations:
- Mount the original task directory into the agent container, exposing tests or
  oracle files while still using Docker.
- Default network access to enabled, making benchmark behavior depend on live
  package indexes unless a task explicitly permits it.
- Report image metadata but omit task/Dockerfile hashes or resource limits.
- Run the verifier against the live agent workspace instead of a clean copied
  phase workspace.
- Mount hidden tests without setting the verifier's test directory, causing a
  verifier to pass for the wrong reason or read from the wrong level.
"""

from __future__ import annotations

from pathlib import Path

from coding_data_collection.docker_substrate import (
    DockerImageInfo,
    DockerResourceLimits,
    agent_phase_command,
    environment_manifest_payload,
    oracle_phase_command,
    prepare_verifier_workspace,
    task_directory_hash,
    task_docker_build_context,
    task_dockerfile_path,
    verifier_phase_command,
)


def test_agent_phase_mounts_only_workspace_and_task_spec_not_task_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    task_dir = tmp_path / "task"
    (run_dir / "agent_workspace").mkdir(parents=True)
    (run_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test_outputs.py").write_text("secret\n", encoding="utf-8")

    command = agent_phase_command(
        image_tag="image",
        run_dir=run_dir,
        command="true",
        limits=DockerResourceLimits(),
    )
    command_text = " ".join(command)

    assert f"{(run_dir / 'agent_workspace').resolve()}:/app:rw" in command
    assert f"{(run_dir / 'task.md').resolve()}:/task/task.md:ro" in command
    assert str(task_dir.resolve()) not in command_text
    assert "/task/tests" not in command_text


def test_network_is_disabled_by_default_and_manifest_records_limits(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    run_dir = tmp_path / "run"
    task_dir.mkdir()
    (task_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    limits = DockerResourceLimits(cpus=2, memory_mb=3072, storage_mb=4096, wall_clock_limit_sec=123)

    command = agent_phase_command(
        image_tag="image",
        run_dir=run_dir,
        command="true",
        limits=limits,
    )
    manifest = environment_manifest_payload(
        run_dir=run_dir,
        task_dir=task_dir,
        image=DockerImageInfo(image_tag="image", image_id="sha256:abc", image_digest="sha256:abc"),
        limits=limits,
    )

    assert "--network" in command
    assert "none" in command
    assert manifest["network_policy"] == "disabled"
    assert manifest["cpu_limit"] == 2
    assert manifest["memory_limit_mb"] == 3072
    assert manifest["disk_limit_mb"] == 4096
    assert manifest["disk_limit_enforced"] is False
    assert manifest["wall_clock_limit_sec"] == 123
    assert manifest["dockerfile_sha256"]
    assert manifest["task_archive_sha256"]


def test_task_directory_hash_changes_when_hidden_verifier_file_changes(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test_outputs.py").write_text("assert answer == 1\n", encoding="utf-8")
    before = task_directory_hash(task_dir)

    (task_dir / "tests" / "test_outputs.py").write_text("assert answer == 2\n", encoding="utf-8")

    assert task_directory_hash(task_dir) != before


def test_compose_style_client_dockerfile_is_supported(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    client_dir = task_dir / "client"
    client_dir.mkdir(parents=True)
    dockerfile = client_dir / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    manifest = environment_manifest_payload(
        run_dir=tmp_path / "run",
        task_dir=task_dir,
        image=DockerImageInfo(image_tag="image", image_id="sha256:abc", image_digest="sha256:abc"),
        limits=DockerResourceLimits(),
    )

    assert task_dockerfile_path(task_dir) == dockerfile
    assert task_docker_build_context(task_dir) == client_dir
    assert manifest["dockerfile_sha256"]


def test_verifier_phase_uses_clean_workspace_copy_and_mounts_tests_only_there(tmp_path: Path) -> None:
    agent_workspace = tmp_path / "run" / "agent_workspace"
    verifier_workspace = tmp_path / "run" / "verifier_workspace"
    task_dir = tmp_path / "task"
    agent_workspace.mkdir(parents=True)
    task_dir.mkdir()
    (agent_workspace / "result.txt").write_text("79\n", encoding="utf-8")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test_outputs.py").write_text("hidden\n", encoding="utf-8")

    prepare_verifier_workspace(agent_workspace, verifier_workspace)
    (agent_workspace / "result.txt").write_text("mutated after copy\n", encoding="utf-8")
    command = verifier_phase_command(
        image_tag="image",
        task_dir=task_dir,
        verifier_workspace=verifier_workspace,
        command="bash /task/run-tests.sh",
        limits=DockerResourceLimits(),
    )

    assert (verifier_workspace / "result.txt").read_text(encoding="utf-8") == "79\n"
    assert f"{verifier_workspace.resolve()}:/app:rw" in command
    assert f"{task_dir.resolve()}:/task:ro" in command
    assert "TEST_DIR=/task/tests" in command
    assert f"{agent_workspace.resolve()}:/app:rw" not in command


def test_verifier_workspace_rejects_agent_created_symlinks(tmp_path: Path) -> None:
    agent_workspace = tmp_path / "agent_workspace"
    verifier_workspace = tmp_path / "verifier_workspace"
    agent_workspace.mkdir()
    (agent_workspace / "link").symlink_to("/etc/passwd")

    try:
        prepare_verifier_workspace(agent_workspace, verifier_workspace)
    except ValueError as exc:
        assert "agent workspace contains symlinks" in str(exc)
        assert "link" in str(exc)
    else:
        raise AssertionError("symlinked agent workspace should be rejected before verifier copy")

    assert not verifier_workspace.exists()


def test_oracle_phase_is_privileged_but_separate_from_agent_phase(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    task_dir = tmp_path / "task"
    verifier_workspace = run_dir / "verifier_workspace"
    (run_dir / "agent_workspace").mkdir(parents=True)
    (run_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    verifier_workspace.mkdir(parents=True)
    (task_dir / "solution.sh").parent.mkdir(parents=True)
    (task_dir / "solution.sh").write_text("echo oracle\n", encoding="utf-8")

    agent_command = agent_phase_command(
        image_tag="image",
        run_dir=run_dir,
        command="true",
        limits=DockerResourceLimits(),
    )
    oracle_command = oracle_phase_command(
        image_tag="image",
        task_dir=task_dir,
        verifier_workspace=verifier_workspace,
        command="bash /task/solution.sh",
        limits=DockerResourceLimits(),
    )

    assert f"{task_dir.resolve()}:/task:ro" not in agent_command
    assert f"{task_dir.resolve()}:/task:ro" in oracle_command
    assert "bash /task/solution.sh" in oracle_command
