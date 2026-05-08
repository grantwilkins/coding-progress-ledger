"""Combined all_runs.csv covers every canonical source, is byte-stable,
and contains only valid final_success_source enum values.

Claim:
    write_combined_manifest validates final_success_source against the
    declared enum and raises ValueError when ANY row is out of enum.

Plausible wrong implementations:
    - check only df['final_success_source'].iloc[0] -> misses violations
      in later rows
    - substring check (e.g. `if "verifier" in val`) -> falsely accepts
      out-of-enum strings that share a substring with a valid value
    - silent passthrough -> downstream consumers see invalid values
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.ingest.adapters import (
    FINAL_SUCCESS_SOURCE_ENUM,
    write_combined_manifest,
)


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_combined_covers_every_canonical_source(real_ledger: None, tmp_path: Path) -> None:
    csv_path, df = write_combined_manifest(tmp_path)
    assert csv_path.is_file()
    assert csv_path.name == "all_runs.csv"
    assert set(df["source"].unique()) == {
        "swe_agent_pilot",
        "hermes_pilot_h5_v2",
        "tb_live",
    }
    # Per-source counts: 21 dirs under swe_agent_pilot (one is `plots`),
    # 30 under hermes_pilot_h5_v2, 12 under tb_live. The combined
    # manifest records ALL of them, not just the resolvable ones.
    counts = df.groupby("source").size().to_dict()
    assert counts["tb_live"] == 12
    assert counts["swe_agent_pilot"] == 21
    assert counts["hermes_pilot_h5_v2"] == 30


def test_final_success_source_enum_respected(real_ledger: None, tmp_path: Path) -> None:
    _, df = write_combined_manifest(tmp_path)
    bad = set(df["final_success_source"].unique()) - FINAL_SUCCESS_SOURCE_ENUM
    assert not bad


def test_combined_manifest_byte_stable(real_ledger: None, tmp_path: Path) -> None:
    a, _ = write_combined_manifest(tmp_path / "a")
    b, _ = write_combined_manifest(tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_resolvable_runs_have_final_success_set(real_ledger: None, tmp_path: Path) -> None:
    _, df = write_combined_manifest(tmp_path)
    # Every resolved row must have a non-null final_success and a
    # non-"missing" final_success_source. Conversely, every "missing"
    # source row must have null final_success.
    resolved = df[df["final_success_source"] != "missing"]
    assert resolved["final_success"].notna().all()
    missing = df[df["final_success_source"] == "missing"]
    assert missing["final_success"].isna().all()


def test_enum_check_raises_on_out_of_enum_value(
    real_ledger: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject a row whose final_success_source is OUT OF the enum but
    # shares a substring with a valid value ('verifier_exit_code' vs
    # 'verifier_exit'). A correct set-difference check must catch it; a
    # wrong implementation that uses substring matching, or only checks
    # the first row, would let it through.
    from dataclasses import replace

    from coding_estimator.ingest import adapters

    real_rows = adapters.ingest_source("tb_live")
    poisoned = list(real_rows)
    poisoned[-1] = replace(poisoned[-1], final_success_source="verifier_exit_code")

    def _fake(_out_dir: Path) -> dict[str, list[adapters.RunManifestRow]]:
        return {"tb_live": poisoned}

    monkeypatch.setattr(adapters, "ingest_canonical_sources", _fake)
    with pytest.raises(ValueError, match="not in enum"):
        adapters.write_combined_manifest(tmp_path)


def test_enum_check_catches_violation_when_only_last_row_is_bad(
    real_ledger: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Defeats a 'check the first row only' regression: only the LAST row
    # is poisoned. The validator must still raise.
    from dataclasses import replace

    from coding_estimator.ingest import adapters

    real_rows = adapters.ingest_source("tb_live")
    assert len(real_rows) > 2
    poisoned = list(real_rows)
    poisoned[-1] = replace(poisoned[-1], final_success_source="totally_invalid")

    def _fake(_out_dir: Path) -> dict[str, list[adapters.RunManifestRow]]:
        return {"tb_live": poisoned}

    monkeypatch.setattr(adapters, "ingest_canonical_sources", _fake)
    with pytest.raises(ValueError, match="totally_invalid"):
        adapters.write_combined_manifest(tmp_path)


def test_csv_columns_match_spec(real_ledger: None, tmp_path: Path) -> None:
    _, df = write_combined_manifest(tmp_path)
    required = {
        "run_id",
        "source",
        "ledger_path",
        "ledger_event_count",
        "has_real_wallclock",
        "start_wall_time",
        "end_wall_time",
        "task_id",
        "task_family",
        "arm",
        "difficulty",
        "agent_scaffold",
        "model_name",
        "final_success",
        "final_success_source",
        "timeout",
        "finish_step",
        "finish_seconds",
        "termination_reason",
        "notes",
    }
    assert required.issubset(set(df.columns))


def test_combined_manifest_can_target_tb_live_v2_only(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    _, df = write_combined_manifest(tmp_path, source_ids=["tb_live_v2"])
    assert set(df["source"].unique()) == {"tb_live_v2"}
    assert len(df) == 102
    assert set(df["final_success_source"].unique()) == {"verifier_exit"}


def test_tb_live_v2_manifest_preserves_arm_and_difficulty_metadata(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    _, df = write_combined_manifest(tmp_path, source_ids=["tb_live_v2"])
    row = df.loc[
        df["run_id"] == "validation_new_work_05_quoted_field_in_tsv__armB__87f7ab5e"
    ].iloc[0]
    assert row["task_id"] == "validation_new_work_05_quoted_field_in_tsv"
    assert row["task_family"] == "validation_new_work"
    assert row["arm"] == "B"
    assert row["difficulty"] == "medium"
    assert row["model_name"] == "claude-sonnet-4-6"
    assert row["termination_reason"] == "verifier_fail"
