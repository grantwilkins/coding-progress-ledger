from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentReadinessCheck:
    check_id: str
    command: str
    reason: str
    expected: str = "exit_code_0"


def infer_agent_readiness_checks(*, task_md: str, workspace_dir: Path) -> list[AgentReadinessCheck]:
    checks: list[AgentReadinessCheck] = []
    task_lower = task_md.lower()

    if _mentions_r_runtime(task_lower) or any(path.suffix == ".R" for path in workspace_dir.rglob("*") if path.is_file()):
        checks.append(
            AgentReadinessCheck(
                check_id="r_runtime_available",
                command="command -v Rscript >/dev/null && Rscript --version >/dev/null",
                reason="task requires R execution in the agent/verifier environment",
            )
        )

    if "nginx" in task_lower:
        checks.append(
            AgentReadinessCheck(
                check_id="nginx_available",
                command="command -v nginx >/dev/null && nginx -v >/dev/null 2>&1",
                reason="task asks the agent to install/configure/start nginx while agent network is disabled",
            )
        )

    python_imports = _python_imports_required(workspace_dir)
    if python_imports:
        import_expr = "; ".join(f"import {name}" for name in python_imports)
        checks.append(
            AgentReadinessCheck(
                check_id="python_imports_available",
                command=f"python3 - <<'PY'\n{import_expr}\nPY",
                reason="visible Python files import third-party modules needed for local validation",
            )
        )

    if _mentions_package_install(task_lower):
        checks.append(
            AgentReadinessCheck(
                check_id="no_solve_time_network_install",
                command="test ! -f /etc/apt/sources.list || true",
                reason="task text asks for package installation; no-network agent runs must bake dependencies or exclude the task",
                expected="manual_review",
            )
        )

    return _dedupe_checks(checks)


def run_agent_readiness_preflight(
    *,
    image_tag: str,
    workspace_dir: Path,
    task_md: str,
    timeout_s: int = 30,
) -> dict[str, Any]:
    checks = [
        AgentReadinessCheck(
            check_id="hidden_image_artifacts_unreadable",
            command=(
                "bad=''; "
                "for p in /protected /oracle /verifier /gold /solution /tests /test; do "
                "if [ -e \"$p\" ]; then "
                "if [ -d \"$p\" ]; then ls \"$p\" >/dev/null 2>&1 && bad=\"$bad $p\"; "
                "else head -c 1 \"$p\" >/dev/null 2>&1 && bad=\"$bad $p\"; fi; "
                "fi; "
                "done; "
                "if [ -n \"$bad\" ]; then echo \"agent can read hidden image path(s):$bad\" >&2; exit 1; fi"
            ),
            reason="agent image must not expose hidden tests, protected files, oracle files, or verifier internals",
        ),
        *infer_agent_readiness_checks(task_md=task_md, workspace_dir=workspace_dir),
    ]
    results = [_run_check(image_tag=image_tag, workspace_dir=workspace_dir, check=check, timeout_s=timeout_s) for check in checks]
    failed = [result for result in results if result["status"] == "failed"]
    manual_review = [result for result in results if result["status"] == "manual_review"]
    return {
        "schema_version": "0.1.0",
        "image_tag": image_tag,
        "network_policy": "disabled",
        "check_count": len(results),
        "passed": not failed and not manual_review,
        "failed_checks": [result["check_id"] for result in failed],
        "manual_review_checks": [result["check_id"] for result in manual_review],
        "results": results,
    }


def write_agent_readiness_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_check(*, image_tag: str, workspace_dir: Path, check: AgentReadinessCheck, timeout_s: int) -> dict[str, Any]:
    if check.expected == "manual_review":
        return {
            "check_id": check.check_id,
            "command": check.command,
            "reason": check.reason,
            "expected": check.expected,
            "status": "manual_review",
            "exit_code": None,
            "stdout_snippet": "",
            "stderr_snippet": "task requires solve-time package installation under a no-network agent policy",
        }
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{workspace_dir.resolve()}:/app:ro",
        "-w",
        "/app",
        "--entrypoint",
        "bash",
        image_tag,
        "-lc",
        check.command,
    ]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        stderr = _coerce_text(exc.stderr or "") + f"\npreflight timed out after {timeout_s}s"
        return {
            "check_id": check.check_id,
            "command": check.command,
            "reason": check.reason,
            "expected": check.expected,
            "status": "failed",
            "exit_code": 124,
            "stdout_snippet": _snippet(exc.stdout or ""),
            "stderr_snippet": _snippet(stderr),
        }
    return {
        "check_id": check.check_id,
        "command": check.command,
        "reason": check.reason,
        "expected": check.expected,
        "status": "passed" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "stdout_snippet": _snippet(proc.stdout),
        "stderr_snippet": _snippet(proc.stderr),
    }


def _mentions_r_runtime(task_lower: str) -> bool:
    return bool(re.search(r"\b(rscript|write your code in r|code in r| in r\b| r )", task_lower))


def _mentions_package_install(task_lower: str) -> bool:
    return bool(re.search(r"\b(apt-get|pip install|install nginx|install r|install .* package|install .* server)\b", task_lower))


def _python_imports_required(workspace_dir: Path) -> list[str]:
    stdlib_or_local = _local_python_modules(workspace_dir) | {
        "argparse",
        "collections",
        "csv",
        "dataclasses",
        "functools",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "random",
        "re",
        "statistics",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "time",
        "typing",
    }
    imports: set[str] = set()
    for path in workspace_dir.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(workspace_dir).parts):
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(r"\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                name = match.group(1)
                if name not in stdlib_or_local:
                    imports.add(name)
    return sorted(imports)


def _local_python_modules(workspace_dir: Path) -> set[str]:
    modules = {path.stem for path in workspace_dir.rglob("*.py") if path.name != "__init__.py"}
    packages = {path.parent.name for path in workspace_dir.rglob("__init__.py")}
    return modules | packages


def _dedupe_checks(checks: list[AgentReadinessCheck]) -> list[AgentReadinessCheck]:
    seen: set[str] = set()
    out: list[AgentReadinessCheck] = []
    for check in checks:
        if check.check_id in seen:
            continue
        seen.add(check.check_id)
        out.append(check)
    return out


def _snippet(text: str | bytes, limit: int = 1000) -> str:
    text = _coerce_text(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[cdc:preflight_output_truncated]\n" + text[-half:]


def _coerce_text(text: str | bytes) -> str:
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text
