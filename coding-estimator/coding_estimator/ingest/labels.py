"""Final-label loader. Source-aware wrapper over upstream
`resolve_final_success` plus a `tb_live` fallback to
`live_instrumentation.verifier_pass`.

A run with an unresolvable label hard fails. The caller is responsible
for catching the exception and skipping the run if that is policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ledger_progress.run_manager import resolve_final_success

from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.ingest.sources import SOURCES

FinalLabelSource = Literal[
    "verifier_exit",
    "swe_agent_target",
    "hermes_resolved",
    "manual",
    "missing",
]


@dataclass(frozen=True)
class FinalLabel:
    final_success: bool
    final_success_source: FinalLabelSource
    finish_step: int | None
    finish_seconds: float | None
    timeout: bool
    termination_reason: str | None


class UnresolvableLabelError(RuntimeError):
    """Raised when no source-of-truth pins a final_success for the run."""


def _classify_upstream_source(label_source: str) -> FinalLabelSource:
    # Upstream emits a small set of strings. Map them onto our enum.
    if label_source.startswith("source_metadata"):
        return "swe_agent_target"
    if label_source.startswith("test_result"):
        return "verifier_exit"
    if label_source.startswith("run_manifest"):
        return "manual"
    if label_source.startswith("summary"):
        return "hermes_resolved"
    if label_source in {"manual", "hidden_check_tests"}:
        return "manual"
    return "manual"


def _tb_live_fallback(run: RunRecord) -> FinalLabel | None:
    instr_path = run.ledger_path.parent / "live_instrumentation.json"
    if not instr_path.is_file():
        return None
    instr = json.loads(instr_path.read_text(encoding="utf-8"))
    verifier_pass = instr.get("verifier_pass")
    if not isinstance(verifier_pass, bool):
        return None
    finish_step = run.events[-1].step if run.events else None
    finish_seconds = float(instr.get("timestamp_span_seconds")) if instr.get(
        "timestamp_span_seconds"
    ) is not None else None
    return FinalLabel(
        final_success=verifier_pass,
        final_success_source="verifier_exit",
        finish_step=finish_step,
        finish_seconds=finish_seconds,
        timeout=False,
        termination_reason=None,
    )


def load_final_label(run: RunRecord) -> FinalLabel:
    src = SOURCES[run.source]
    run_dir = run.ledger_path.parent

    # Source-specific path lock-in: tb_live's authoritative signal is
    # live_instrumentation.verifier_pass. Use it BEFORE the upstream
    # resolver, which would otherwise return ('unknown',) on this source.
    if src.label_field_path == "live_instrumentation.verifier_pass":
        fallback = _tb_live_fallback(run)
        if fallback is not None:
            return fallback
        raise UnresolvableLabelError(
            f"tb_live run {run.run_id} has no live_instrumentation.verifier_pass"
        )

    # Retrospective sources: defer to upstream.
    fs, source_str = resolve_final_success(run_dir)
    if not isinstance(fs, bool):
        raise UnresolvableLabelError(
            f"run {run.source}/{run.run_id}: upstream returned {(fs, source_str)!r}; "
            "caller must skip per § 0.9"
        )

    finish_step = run.events[-1].step if run.events else None
    finish_seconds: float | None = None
    if run.has_real_wallclock and run.start_wall_time and run.end_wall_time:
        finish_seconds = (run.end_wall_time - run.start_wall_time).total_seconds()

    return FinalLabel(
        final_success=fs,
        final_success_source=_classify_upstream_source(source_str),
        finish_step=finish_step,
        finish_seconds=finish_seconds,
        timeout=False,
        termination_reason=None,
    )


def load_final_label_or_none(run: RunRecord) -> FinalLabel | None:
    """Convenience wrapper for callers that want the skip-on-unresolvable
    behavior of § 0.9 without the try/except dance."""
    try:
        return load_final_label(run)
    except UnresolvableLabelError:
        return None
