from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from coding_data_collection.observation import read_jsonl


NETWORK_RE = re.compile(
    r"\b(apt-get|pip install|npm install|curl|wget|network|temporary failure resolving|could not resolve|"
    r"failed to connect|no module named|modulenotfounderror|command not found|unable to locate package)\b",
    re.IGNORECASE,
)
TRUNCATION_RE = re.compile(r"\[cdc:output_truncated\]|omitted \d+ (?:lines|chars)|Use read_file\(path=", re.IGNORECASE)
VALIDATION_RE = re.compile(r"\b(pytest|unittest|test|verify|check|lint|mypy|ruff|cargo test|go test|curl)\b", re.IGNORECASE)
MISSING_RE = re.compile(r"\b(no such file|not found|missing|expected file)\b", re.IGNORECASE)


TRIAGE_COLUMNS = [
    "run_id",
    "task_id",
    "model",
    "failure_stage",
    "verifier_failure_summary",
    "last_agent_claim",
    "validation_attempts",
    "visible_validation_failures",
    "network_blocked_events",
    "repeated_tool_patterns",
    "long_file_truncation_events",
    "expected_file_missing_events",
    "product_files_written",
    "did_agent_write_expected_target",
    "did_agent_run_relevant_tests",
    "primary_failure_class",
    "recommended_fix",
]


def triage_corpus(run_root: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(path for path in run_root.iterdir() if (path / "run_manifest.json").is_file())
    return [triage_run(path) for path in run_dirs]


def triage_run(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    verifier_output = _read_text(run_dir / "verifier_output.txt")
    final_diff = _read_text(run_dir / "final_diff.patch")

    run_id = str(manifest.get("run_id") or run_dir.name)
    task_id = run_id.split("__", 1)[0]
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    model = str(metrics.get("model_name") or run_id.rsplit("__", 1)[-1])
    failure_stage = str(manifest.get("termination_reason") or manifest.get("run_status") or "unknown")
    last_agent_claim = _last_agent_claim(transcript)

    validation_rows = [row for row in transcript if _validation_row(row)]
    visible_validation_failures = sum(1 for row in validation_rows if row.get("visible_to_agent", True) and row.get("exit_code") not in (None, 0))
    event_counts = Counter(str(event.get("event_type") or "") for event in observations)
    network_rows = [row for row in transcript if _network_row(row)]
    repeated_patterns = _repeated_tool_patterns(transcript)
    truncation_events = event_counts["tool_output_truncated"] + sum(1 for row in transcript if _truncation_row(row))
    product_files = _product_files(observations, final_diff, transcript)
    did_write_expected = bool(product_files) and not event_counts["expected_file_missing"]
    did_run_relevant_tests = bool(validation_rows)
    task_md = _read_text(run_dir / "task.md")
    primary_class = _primary_failure_class(
        network_events=event_counts["network_blocked"] + len(network_rows),
        truncation_events=truncation_events,
        repeated_patterns=repeated_patterns,
        validation_rows=validation_rows,
        product_files=product_files,
        verifier_output=verifier_output,
        expected_missing=event_counts["expected_file_missing"],
        transcript=transcript,
        task_md=task_md,
    )

    return {
        "run_id": run_id,
        "task_id": task_id,
        "model": model,
        "failure_stage": failure_stage,
        "verifier_failure_summary": _verifier_summary(verifier_output),
        "last_agent_claim": last_agent_claim,
        "validation_attempts": len(validation_rows) or event_counts["validation_attempt"],
        "visible_validation_failures": visible_validation_failures or event_counts["validation_fail_observed"],
        "network_blocked_events": event_counts["network_blocked"] + len(network_rows),
        "repeated_tool_patterns": "; ".join(repeated_patterns),
        "long_file_truncation_events": truncation_events,
        "expected_file_missing_events": event_counts["expected_file_missing"],
        "product_files_written": "; ".join(product_files),
        "did_agent_write_expected_target": str(did_write_expected).lower(),
        "did_agent_run_relevant_tests": str(did_run_relevant_tests).lower(),
        "primary_failure_class": primary_class,
        "recommended_fix": _recommended_fix(primary_class),
    }


def write_triage_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAGE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TRIAGE_COLUMNS})
    return path


def write_triage_markdown(rows: list[dict[str, Any]], path: Path) -> Path:
    counts = Counter(str(row["primary_failure_class"]) for row in rows)
    lines = [
        "# Real Model Mini3 Failure Triage",
        "",
        "Scope: GPT-5.4 / GPT-5.4-mini provider-backed mini-pilot failures under `runs/real_model_mini3_gpt54_vs_mini_v3/`.",
        "",
        "## Classification Counts",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run_id | failure_class | validation | network | truncation | product_files | recommended_fix |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run_id} | `{primary_failure_class}` | {validation_attempts} | {network_blocked_events} | "
            "{long_file_truncation_events} | {product_files_written} | {recommended_fix} |".format(**_md_row(row))
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Treat this corpus as a harness-realism pass and scale no-go. The failures are not clean model-quality evidence until tool-affordance and environment-mismatch classes are removed or explicitly balanced.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_tool_gap_markdown(rows: list[dict[str, Any]], path: Path) -> Path:
    counts = Counter(str(row["primary_failure_class"]) for row in rows)
    lines = [
        "# Tool Affordance Gaps",
        "",
        "Evidence comes from the six GPT-5.4 / GPT-5.4-mini completed-failure runs.",
        "",
        "## Supported Fixes",
        "",
        "- Add head+tail default file reads and line-range chunked reads.",
        "- Add first-class `find_files` and `grep` tools with bounded output.",
        "- Add first-class `apply_patch` so agents can make targeted edits without whole-file rewrites.",
        "- Classify network and dependency dead ends with visible controller messages.",
        "- Record truncation, repeated file inspection, and chunked-read observation events.",
        "",
        "## Failure Mix",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Remaining Work",
            "",
            "- Run oracle-hidden success checks for the same three tasks before treating verifier failures as task difficulty.",
            "- Add task compatibility tags for solve-time network, package install, service bootstrap, long-file context, and visible-test availability.",
            "- Run a 6-task / 12-run calibration mini-pilot before the 12-task / 24-run pilot.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _primary_failure_class(
    *,
    network_events: int,
    truncation_events: int,
    repeated_patterns: list[str],
    validation_rows: list[dict[str, Any]],
    product_files: list[str],
    verifier_output: str,
    expected_missing: int,
    transcript: list[dict[str, Any]],
    task_md: str,
) -> str:
    text = "\n".join([task_md, verifier_output, *(_row_text(row) for row in transcript)]).lower()
    if _agent_image_missing_runtime(text):
        return "agent_image_missing_runtime"
    if _solve_time_network_mismatch(text):
        return "no_network_install_mismatch"
    if network_events:
        return "no_network_install_mismatch"
    if truncation_events or any("read_file:" in item for item in repeated_patterns):
        return "tool_affordance"
    if expected_missing or (not product_files and MISSING_RE.search(verifier_output)):
        return "prompt_contract"
    if not validation_rows:
        return "prompt_contract"
    if "FAILED" in verifier_output or "AssertionError" in verifier_output:
        return "task_difficulty"
    return "unclear"


def _recommended_fix(primary_class: str) -> str:
    return {
        "agent_image_missing_runtime": "Bake task runtimes/imports into the agent image and preflight them before model calls.",
        "no_network_install_mismatch": "Exclude or retag solve-time package-install tasks unless dependencies are baked into the image.",
        "tool_affordance": "Use head+tail/chunked reads, grep/find, and patch tools before rerunning calibration.",
        "prompt_contract": "Tighten prompt/tool contract around target files, validation, and blocked-state behavior.",
        "task_difficulty": "Keep as calibrated hard task only after oracle/assisted diagnostics pass.",
        "verifier_harness": "Run oracle-hidden success and verifier determinism checks.",
        "unclear": "Inspect transcript and final diff manually before scale selection.",
    }[primary_class]


def _agent_image_missing_runtime(text: str) -> bool:
    runtime_markers = (
        "modulenotfounderror: no module named",
        "no module named 'numpy'",
        "no module named 'torch'",
        "no module named 'scipy'",
        "r could not be installed",
        "unable to run r",
        "rscript: command not found",
        "r: command not found",
    )
    return any(marker in text for marker in runtime_markers)


def _solve_time_network_mismatch(text: str) -> bool:
    if "install nginx" in text or "nginx is not installed" in text:
        return True
    return any(
        marker in text
        for marker in (
            "temporary failure resolving",
            "could not resolve",
            "failed to connect",
            "network-restricted sandbox",
            "sandbox has no network",
        )
    )


def _product_files(observations: list[dict[str, Any]], final_diff: str, transcript: list[dict[str, Any]]) -> list[str]:
    files: set[str] = set()
    for event in observations:
        if event.get("event_type") not in {"product_file_written", "product_file_edited"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        path = str(payload.get("path") or "").strip()
        if path:
            files.add(path)
    for line in final_diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split("\t", 1)[0].strip()
            if path and path != "/dev/null":
                files.add(path.removeprefix("a/").removeprefix("b/"))
    for row in transcript:
        if row.get("kind") != "shell":
            continue
        command = str(row.get("command") or "")
        for match in re.finditer(r">\s*([A-Za-z0-9_./-]+)", command):
            path = match.group(1).strip()
            if path and not path.startswith("/"):
                files.add(path.removeprefix("./"))
        if command.startswith(("touch ", "mkdir ")):
            parts = command.split()
            for part in parts[1:]:
                if part.startswith("-") or part.startswith("/"):
                    continue
                files.add(part.removeprefix("./"))
    return sorted(files)


def _repeated_tool_patterns(transcript: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in transcript:
        kind = str(row.get("kind") or "")
        if kind == "read_file":
            counts[f"read_file:{row.get('path') or ''}"] += 1
        elif kind == "shell":
            command = str(row.get("command") or "").strip()
            if command:
                counts[f"shell:{command[:120]}"] += 1
    return [f"{key} x{count}" for key, count in sorted(counts.items()) if count > 1]


def _validation_row(row: dict[str, Any]) -> bool:
    return str(row.get("kind") or "") == "shell" and bool(VALIDATION_RE.search(_row_text(row)))


def _network_row(row: dict[str, Any]) -> bool:
    return str(row.get("kind") or "") in {"shell", "network_blocked"} and bool(NETWORK_RE.search(_row_text(row)))


def _truncation_row(row: dict[str, Any]) -> bool:
    text = _row_text(row)
    if TRUNCATION_RE.search(text):
        return True
    if "truncated" in text.lower():
        return True
    return any(len(str(row.get(key) or "")) >= 1900 for key in ("stdout_snippet", "stderr_snippet"))


def _last_agent_claim(transcript: list[dict[str, Any]]) -> str:
    for row in reversed(transcript):
        if row.get("kind") == "done":
            return str(row.get("summary") or "")
    return ""


def _verifier_summary(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = [line for line in lines if re.search(r"\b(FAILED|ERROR|AssertionError|failed|error|Expected|expected)\b", line)]
    chosen = interesting[:4] if interesting else lines[-4:]
    return " / ".join(chosen)[:600]


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("summary", "command", "path", "stdout_snippet", "stderr_snippet"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _md_row(row: dict[str, Any]) -> dict[str, str]:
    return {key: _escape_md(str(value)) for key, value in row.items()}


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
