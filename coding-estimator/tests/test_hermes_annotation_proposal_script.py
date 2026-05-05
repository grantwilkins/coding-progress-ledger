"""
Claim:
scripts/build_hermes_annotation_proposal.main hard-fails when the
upstream Hermes corpus drifts from the pinned 30-run expectation.

Plausible wrong implementations:
- silently writes a partial proposal package on != 30 runs.
- raises a generic error that doesn't reference the pinning expectation.
"""

from __future__ import annotations

import importlib
import json

import pytest


def _make_run(parent, run_id, events):
    d = parent / run_id
    d.mkdir()
    (d / "normalized_trace.json").write_text(
        json.dumps({"events": events, "issue_text": "x"})
    )


def test_main_hard_fails_with_descriptive_message_on_corpus_drift(tmp_path,
                                                                  monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in range(2):
        _make_run(runs, f"r_{i:02d}", [{"role": "assistant", "action": "terminal"}])
    out = tmp_path / "out"

    mod = importlib.import_module("scripts.build_hermes_annotation_proposal")
    monkeypatch.setattr(mod, "SOURCE_RUNS", runs)
    monkeypatch.setattr(mod, "OUT_DIR", out)

    with pytest.raises(RuntimeError) as exc:
        mod.main()
    msg = str(exc.value)
    assert "got 2" in msg
    assert "pinning" in msg.lower() or "upstream corpus" in msg.lower()
