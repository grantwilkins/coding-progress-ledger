"""Per-source adapters: each canonical source returns a manifest covering
every run found on disk; malformed runs are reported, not dropped.

Claim:
    _row_for_run distinguishes 'malformed_run:' (load_run raised) from
    'unresolvable_label:' (load_run succeeded, load_final_label raised).
    Both produce a manifest row; neither silently drops the run.

Plausible wrong implementations:
    - catch UnresolvableLabelError around load_run() -> a malformed run
      with a missing ledger would be classified 'unresolvable_label'
    - catch Exception broadly -> hides bugs in load_final_label
    - return None (drop the row) on any error -> silent loss of accounting
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.ingest import paths
from coding_estimator.ingest.adapters import (
    ingest_canonical_sources,
    ingest_source,
    write_source_manifest,
)


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_swe_agent_pilot_covers_every_dir(real_ledger: None) -> None:
    rows = ingest_source("swe_agent_pilot")
    on_disk_dirs = paths.list_run_ids("swe_agent_pilot")
    assert {r.run_id for r in rows} == set(on_disk_dirs)


def test_swe_agent_pilot_reports_malformed_runs(real_ledger: None) -> None:
    rows = ingest_source("swe_agent_pilot")
    malformed = [r for r in rows if r.notes.startswith("malformed_run")]
    # `plots` is a non-run sibling dir under swe_agent_pilot; it must be
    # reported, not silently dropped.
    assert any(r.run_id == "plots" for r in malformed)


def test_swe_agent_pilot_resolves_real_runs(real_ledger: None) -> None:
    rows = ingest_source("swe_agent_pilot")
    real_runs = [r for r in rows if r.run_id.startswith("swe_agent_pilot_")]
    # 10 successes + 10 failures.
    assert len(real_runs) == 20
    resolved = [r for r in real_runs if r.final_success is not None]
    assert len(resolved) == 20
    assert all(r.final_success_source == "swe_agent_target" for r in resolved)


def test_hermes_pilot_h5_v2_records_unresolvable(real_ledger: None) -> None:
    rows = ingest_source("hermes_pilot_h5_v2")
    # All v2 runs we sampled have null source_metadata.final_success.
    unresolved = [r for r in rows if r.notes.startswith("unresolvable_label")]
    assert len(unresolved) >= 1
    # final_success is None on every unresolvable row.
    for r in unresolved:
        assert r.final_success is None
        assert r.final_success_source == "missing"


def test_tb_live_resolves_via_verifier_pass(real_ledger: None) -> None:
    rows = ingest_source("tb_live")
    assert len(rows) == 12
    for r in rows:
        assert r.final_success is not None, r
        assert r.final_success_source == "verifier_exit"
        assert r.has_real_wallclock is True
        assert r.finish_seconds is not None and r.finish_seconds > 0


def test_write_source_manifest_byte_stable(real_ledger: None, tmp_path: Path) -> None:
    a, _ = write_source_manifest("tb_live", tmp_path / "a")
    b, _ = write_source_manifest("tb_live", tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_malformed_and_unresolvable_use_different_note_prefixes(real_ledger: None) -> None:
    # `plots` under swe_agent_pilot has no ledger.jsonl -> 'malformed_run:'.
    # All hermes_pilot_h5_v2 runs have null source_metadata.final_success
    # -> 'unresolvable_label:'. The two prefixes must be DISTINCT, otherwise
    # a downstream filter `notes.startswith("unresolvable_label")` would
    # over-match and drop genuinely malformed runs from the unresolvable
    # accounting.
    swe = ingest_source("swe_agent_pilot")
    her = ingest_source("hermes_pilot_h5_v2")
    plots_row = next(r for r in swe if r.run_id == "plots")
    assert plots_row.notes.startswith("malformed_run:")
    assert not plots_row.notes.startswith("unresolvable_label")
    her_row = her[0]
    assert her_row.notes.startswith("unresolvable_label:")
    assert not her_row.notes.startswith("malformed_run")


def test_canonical_sources_writes_three_manifests(real_ledger: None, tmp_path: Path) -> None:
    out = ingest_canonical_sources(tmp_path)
    assert set(out) == {"swe_agent_pilot", "hermes_pilot_h5_v2", "tb_live"}
    for src in out:
        assert (tmp_path / f"{src}.csv").is_file()
    df = pd.read_csv(tmp_path / "tb_live.csv")
    assert "final_success" in df.columns
