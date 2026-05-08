#!/usr/bin/env python
"""Audit observation loss on the frozen tb_live_v2 corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from coding_estimator.runner.observation_events import build_observation_events

VALIDATION_CMD_RE = re.compile(r"\b(pytest|test|verify|smoke|curl|check|python3?)\b", re.IGNORECASE)


def _read_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            out.append(json.loads(raw))
    return out


def _task_id(run_dir: Path) -> str | None:
    manifest = run_dir / "run_manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("task_id")


def _final_progress(run_dir: Path) -> float | None:
    progress = run_dir / "progress.csv"
    if not progress.is_file():
        return None
    df = pd.read_csv(progress)
    if df.empty or "coding_progress" not in df.columns:
        return None
    return float(df["coding_progress"].iloc[-1])


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["| none |", "|---|"]
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return lines


def build_report(*, runs_root: Path, repo_root: Path) -> str:
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    validation_rows: list[dict] = []
    nonzero_rows: list[dict] = []
    oracle_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    done_fail_rows: list[dict] = []
    final_progress_fail_rows: list[dict] = []
    failure_types: Counter[str] = Counter()

    for run_dir in run_dirs:
        transcript = _read_lines(run_dir / "transcript.jsonl")
        events = build_observation_events(
            run_dir=run_dir,
            run_id=run_dir.name,
            task_id=_task_id(run_dir),
            repo_root=repo_root,
        )
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

        validation_cmds = 0
        nonzero_shell = 0
        oracle_reads = 0
        for line in transcript:
            text = " ".join(str(line.get(key) or "") for key in ("summary", "command", "obs_snippet"))
            if line.get("kind") == "shell" and VALIDATION_CMD_RE.search(text):
                validation_cmds += 1
            if line.get("kind") == "shell" and line.get("exit_code") not in (None, 0):
                nonzero_shell += 1
            if line.get("kind") == "read_file" and Path(str(line.get("path") or "")).name == "solution.sh":
                oracle_reads += 1

        validation_rows.append({"run_id": run_dir.name, "validation_like_shell_commands": validation_cmds})
        nonzero_rows.append({"run_id": run_dir.name, "nonzero_shell_commands": nonzero_shell})
        oracle_rows.append({"run_id": run_dir.name, "solution_sh_reads": oracle_reads})

        unexpected = [
            event for event in events
            if event["event_type"] == "product_file_written"
            and not bool((event.get("payload") or {}).get("matches_expected_path"))
        ]
        if unexpected:
            mismatch_rows.append(
                {
                    "run_id": run_dir.name,
                    "unexpected_write_count": len(unexpected),
                    "unexpected_paths": ", ".join(
                        sorted({str((event.get("payload") or {}).get("basename")) for event in unexpected})
                    ),
                }
            )

        if manifest.get("final_success") is False:
            verifier_fail = next((event for event in events if event["event_type"] == "verifier_fail"), None)
            if verifier_fail:
                failure_types[str((verifier_fail.get("payload") or {}).get("failure_type") or "other")] += 1
            if any(event["event_type"] == "agent_claims_done" for event in events):
                done_fail_rows.append(
                    {
                        "run_id": run_dir.name,
                        "termination_reason": manifest.get("termination_reason"),
                    }
                )
            final_progress = _final_progress(run_dir)
            if final_progress == 1.0:
                final_progress_fail_rows.append(
                    {
                        "run_id": run_dir.name,
                        "final_progress": final_progress,
                        "termination_reason": manifest.get("termination_reason"),
                    }
                )

    validation_df = pd.DataFrame(validation_rows).sort_values("validation_like_shell_commands", ascending=False).head(15)
    nonzero_df = (
        pd.DataFrame(nonzero_rows)
        .query("nonzero_shell_commands > 0")
        .sort_values("nonzero_shell_commands", ascending=False)
        .head(15)
    )
    oracle_df = pd.DataFrame(oracle_rows).sort_values("solution_sh_reads", ascending=False).head(15)
    mismatch_df = pd.DataFrame(mismatch_rows).sort_values("unexpected_write_count", ascending=False).head(15)
    done_fail_df = pd.DataFrame(done_fail_rows).head(15)
    final_progress_fail_df = pd.DataFrame(final_progress_fail_rows).head(15)
    failure_type_df = pd.DataFrame(
        [{"failure_type": key, "n_runs": value} for key, value in sorted(failure_types.items())]
    )

    lines = [
        "# TB Live V2 Observation Loss Audit",
        "",
        "This report quantifies signal present in `transcript.jsonl` / `verifier_output.txt` that the current ledger path does not preserve explicitly.",
        "",
        f"- runs audited: {len(run_dirs)}",
        f"- runs with verifier failure after a done claim: {len(done_fail_rows)}",
        f"- runs with final ledger progress = 1.0 but verifier failed: {len(final_progress_fail_rows)}",
        "",
        "## Validation-Like Shell Commands",
        "",
        *_markdown_table(validation_df),
        "",
        "## Nonzero Shell Commands",
        "",
        *_markdown_table(nonzero_df),
        "",
        "## solution.sh Reads",
        "",
        *_markdown_table(oracle_df),
        "",
        "## Unexpected Product Writes",
        "",
        *_markdown_table(mismatch_df),
        "",
        "## Done Claims Before Verifier Failure",
        "",
        *_markdown_table(done_fail_df),
        "",
        "## Verifier Failure Types",
        "",
        *_markdown_table(failure_type_df),
        "",
        "## Final Progress = 1.0 But Verifier Failed",
        "",
        *_markdown_table(final_progress_fail_df),
        "",
        "## Takeaways",
        "",
        "- The current ledger preserves coarse work-frontier movement but throws away explicit validation, shell failure, oracle-read, and wrong-path signals.",
        "- Those discarded signals are plausible drivers for high-progress failure detection and terminal success calibration.",
        "- `observation_events.jsonl` can be backfilled from the frozen corpus, but some future improvements still require first-class live emission rather than post-hoc heuristics.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(runs_root=args.runs_root, repo_root=args.repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
