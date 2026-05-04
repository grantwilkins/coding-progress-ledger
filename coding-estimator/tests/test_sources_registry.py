"""Source registry resolves under ledger_root, has exactly one canonical
swe_agent and exactly one canonical hermes source, and SOURCES.md
contains the leakage acknowledgment verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.ingest import paths
from coding_estimator.ingest.sources import SOURCES, canonical_sources, source

LEAKAGE_NOTE_FRAGMENT = (
    "Retrospective sources (`swe_agent_pilot`, `hermes_pilot*`) were\n"
    "> annotated post-hoc with knowledge of the run's outcome."
)


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_source_count_locked_at_eight() -> None:
    # Adding or removing a source is a non-trivial decision; surface it.
    assert len(SOURCES) == 8


def test_every_source_resolves_under_ledger_root(real_ledger: None) -> None:
    root = paths.ledger_root()
    for s in SOURCES.values():
        full = root / s.runs_dir
        assert full.is_dir(), f"runs_dir does not exist: {full}"
        run_ids = [p.name for p in full.iterdir() if p.is_dir()]
        assert len(run_ids) >= 1, f"no runs found under {full}"


def test_canonical_for_v0_has_one_per_family() -> None:
    canonical = canonical_sources()
    swe_canonical = [s for s in canonical if s.source_id.startswith("swe_agent")]
    hermes_canonical = [s for s in canonical if s.source_id.startswith("hermes")]
    live_canonical = [s for s in canonical if s.source_id == "tb_live"]
    assert len(swe_canonical) == 1, [s.source_id for s in swe_canonical]
    assert len(hermes_canonical) == 1, [s.source_id for s in hermes_canonical]
    assert len(live_canonical) == 1
    assert {s.source_id for s in canonical} == {
        "swe_agent_pilot",
        "hermes_pilot_h5_v2",
        "tb_live",
    }


def test_swe_agent_live_wallclock_not_canonical() -> None:
    assert SOURCES["swe_agent_live_wallclock"].canonical_for_v0 is False
    assert any(
        "WORKSTREAM_N_TB_PLAN" in c
        for c in SOURCES["swe_agent_live_wallclock"].known_caveats
    )


def test_tb_live_label_field_is_verifier_pass() -> None:
    s = SOURCES["tb_live"]
    assert s.label_field_path == "live_instrumentation.verifier_pass"
    assert s.timestamp_quality == "real"


def test_swe_agent_pilot_label_field_is_source_metadata() -> None:
    s = SOURCES["swe_agent_pilot"]
    assert s.label_field_path == "source_metadata.final_success"
    assert s.timestamp_quality == "none"


def test_unknown_source_raises() -> None:
    with pytest.raises(KeyError):
        source("does_not_exist")


def test_sources_md_contains_leakage_note() -> None:
    md = (Path(__file__).resolve().parents[1] / "docs" / "SOURCES.md").read_text()
    assert LEAKAGE_NOTE_FRAGMENT in md
    assert "tb_live" in md and "verifier_pass" in md
