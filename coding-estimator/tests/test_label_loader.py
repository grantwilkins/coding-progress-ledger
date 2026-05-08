"""Final-label loader covers each canonical source.

Hard fails on runs with unresolvable labels; the convenience wrapper
returns None instead. Final labels are NEVER joined into the feature
frame (the loader returns a separate dataclass).

Claim:
    _classify_upstream_source maps upstream label-source strings onto a
    5-value enum. _tb_live_fallback only accepts a Python bool for
    verifier_pass and emits None (not 0.0) when timestamp_span_seconds
    is absent.

Plausible wrong implementations:
    - prefix-match-in-wrong-order so "manual" gets matched as a substring
    - truthy check `if value:` instead of isinstance(value, bool)
    - float(instr.get("timestamp_span_seconds")) -> TypeError on None
    - default 0.0 when the live span is missing -> silent fabrication
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.ingest.labels import (
    FinalLabel,
    UnresolvableLabelError,
    load_final_label,
    load_final_label_or_none,
)
from coding_estimator.ingest.run_record import load_run


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_tb_live_uses_verifier_pass(real_ledger: None) -> None:
    run = load_run("tb_live", "markdown-to-html-cli")
    label = load_final_label(run)
    assert label.final_success is True
    assert label.final_success_source == "verifier_exit"
    # finish_seconds comes from live_instrumentation.timestamp_span_seconds.
    assert label.finish_seconds is not None and label.finish_seconds > 0
    # finish_step is the last event's step.
    assert label.finish_step == run.events[-1].step


def test_tb_live_v2_uses_run_manifest_verifier_label(real_ledger: None) -> None:
    run = load_run(
        "tb_live_v2",
        "validation_new_work_05_quoted_field_in_tsv__armB__87f7ab5e",
    )
    label = load_final_label(run)
    assert label.final_success is False
    assert label.final_success_source == "verifier_exit"
    assert label.finish_seconds is not None and label.finish_seconds > 0
    assert label.termination_reason == "verifier_fail"


def test_swe_agent_uses_source_metadata_target(real_ledger: None) -> None:
    run = load_run("swe_agent_pilot", "swe_agent_pilot_s_06")
    label = load_final_label(run)
    assert label.final_success is True
    assert label.final_success_source == "swe_agent_target"
    # No real wallclock => no finish_seconds.
    assert label.finish_seconds is None


def test_hermes_null_label_hard_fails(real_ledger: None) -> None:
    # hermes_pilot_h5_001 has source_metadata.final_success = null.
    run = load_run("hermes_pilot_h5_v2", "hermes_pilot_h5_001")
    with pytest.raises(UnresolvableLabelError):
        load_final_label(run)


def test_or_none_skips_unresolvable(real_ledger: None) -> None:
    run = load_run("hermes_pilot_h5_v2", "hermes_pilot_h5_001")
    assert load_final_label_or_none(run) is None
    # Resolvable runs come back as labels, not None.
    tb = load_run("tb_live", "markdown-to-html-cli")
    out = load_final_label_or_none(tb)
    assert isinstance(out, FinalLabel)


def test_classify_upstream_source_for_each_emitted_string() -> None:
    # Direct table check against the upstream strings actually emitted by
    # ledger_progress.run_manager.resolve_final_success.
    from coding_estimator.ingest.labels import _classify_upstream_source

    cases = {
        "source_metadata.target": "swe_agent_target",
        "test_result.success": "verifier_exit",
        "internal_verifier": "verifier_exit",
        "run_manifest.final_success": "manual",
        "summary.final_success": "hermes_resolved",
        "summary_by_category.final_success": "hermes_resolved",
        "hidden_check_tests": "manual",
        "manual": "manual",
    }
    for upstream, expected in cases.items():
        assert _classify_upstream_source(upstream) == expected, upstream


def test_tb_live_fallback_rejects_non_bool_verifier_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `1` is truthy but not a bool. A truthy-check would accept it; the
    # contract requires isinstance(value, bool), so the run must hard-fail.
    import json as _json

    from coding_estimator.ingest.labels import (
        UnresolvableLabelError as _UnresolvableLabelError,
    )
    from coding_estimator.ingest.labels import (
        load_final_label as _load_final_label,
    )
    from coding_estimator.ingest.run_record import load_run as _load_run

    fake_root = tmp_path / "ledger"
    (fake_root / "ledger_progress").mkdir(parents=True)
    rd = fake_root / "runs" / "tb_live" / "bad_run"
    rd.mkdir(parents=True)
    (rd / "ledger.jsonl").write_text(
        '{"step":0,"event_type":"init","subtask_id":null,'
        '"payload":{},"reason":null,"timestamp":"2026-05-04T00:00:00Z"}\n'
    )
    (rd / "live_instrumentation.json").write_text(
        _json.dumps({"timestamp_source": "wallclock", "verifier_pass": 1})
    )
    monkeypatch.setenv("LEDGER_ROOT", str(fake_root))
    run = _load_run("tb_live", "bad_run")
    with pytest.raises(_UnresolvableLabelError):
        _load_final_label(run)


def test_tb_live_fallback_emits_none_when_span_seconds_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The live file has verifier_pass=true but no timestamp_span_seconds.
    # The loader must emit finish_seconds=None, not 0.0 (silent fabrication).
    import json as _json

    from coding_estimator.ingest.labels import load_final_label as _load_final_label
    from coding_estimator.ingest.run_record import load_run as _load_run

    fake_root = tmp_path / "ledger"
    (fake_root / "ledger_progress").mkdir(parents=True)
    rd = fake_root / "runs" / "tb_live" / "no_span"
    rd.mkdir(parents=True)
    (rd / "ledger.jsonl").write_text(
        '{"step":0,"event_type":"init","subtask_id":null,'
        '"payload":{},"reason":null,"timestamp":"2026-05-04T00:00:00Z"}\n'
    )
    (rd / "live_instrumentation.json").write_text(
        _json.dumps({"timestamp_source": "wallclock", "verifier_pass": True})
    )
    monkeypatch.setenv("LEDGER_ROOT", str(fake_root))
    run = _load_run("tb_live", "no_span")
    label = _load_final_label(run)
    assert label.final_success is True
    assert label.finish_seconds is None  # NOT 0.0


def test_or_none_does_not_swallow_unrelated_errors(real_ledger: None) -> None:
    """load_final_label_or_none must catch ONLY UnresolvableLabelError;
    a genuine bug that produces a different exception (e.g. AttributeError
    on a malformed RunRecord) must propagate so it isn't silently masked."""

    class FakeRun:
        source = "tb_live"
        ledger_path = None  # tb_live fallback uses ledger_path.parent
        events = ()

    with pytest.raises(AttributeError):
        load_final_label_or_none(FakeRun())  # type: ignore[arg-type]


def test_or_none_handles_multiple_hermes_runs(real_ledger: None) -> None:
    """The hermes hard-fail behavior must hold for runs other than _001;
    the loader must not have a special case for that specific run_id."""
    for rid in ("hermes_pilot_h5_002", "hermes_pilot_h5_003", "hermes_pilot_h5_010"):
        run = load_run("hermes_pilot_h5_v2", rid)
        assert load_final_label_or_none(run) is None


def test_finish_step_is_max_step(real_ledger: None) -> None:
    run = load_run("tb_live", "markdown-to-html-cli")
    label = load_final_label(run)
    max_step = max(e.step for e in run.events)
    assert label.finish_step == max_step
