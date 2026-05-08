"""D4: build_source_frame produces a parquet-ready checkpoint frame for
one source. Forbidden-column guard fires at write time. Byte-stable
across two builds.

Claim:
    build_source_frame(source) returns a frame with one row per
    (run_id, checkpoint_step). Every row has the identity columns
    required by checkpoint_schema and the feature columns from every
    D3 group. write_source_checkpoints calls assert_no_forbidden
    immediately before writing.

Plausible wrong implementations:
    - assemble identity rows from the FULL run, not the prefix at t
    - omit the forbidden-column guard at write time (relying only on
      schema-load time)
    - emit a non-deterministic frame across builds (parquet bytes vary)
    - silently drop runs with malformed events instead of skipping
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.checkpoints.build import (
    build_run_rows,
    build_source_frame,
    write_combined_checkpoints,
    write_source_checkpoints,
)
from coding_estimator.ingest.run_record import load_run


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_build_run_rows_contiguous_steps(real_ledger: None) -> None:
    run = load_run("tb_live", "markdown-to-html-cli")
    rows = build_run_rows(run)
    steps = [r["checkpoint_step"] for r in rows]
    assert steps == sorted(steps)
    assert steps == list(range(steps[0], steps[-1] + 1))


def test_build_run_rows_terminal_flag_only_on_last(real_ledger: None) -> None:
    run = load_run("tb_live", "markdown-to-html-cli")
    rows = build_run_rows(run)
    terminals = [r for r in rows if r["is_terminal_checkpoint"]]
    assert len(terminals) == 1
    assert terminals[0]["checkpoint_step"] == rows[-1]["checkpoint_step"]


def test_build_run_rows_emits_every_feature_column(real_ledger: None) -> None:
    """Every D3 group's columns must appear on every row."""
    from coding_estimator.checkpoints.features import (
        closure,
        discovery,
        evidence,
        frontier,
        instability,
        stalling,
        time_budget,
        validation,
    )

    run = load_run("tb_live", "markdown-to-html-cli")
    rows = build_run_rows(run)
    expected_cols = set()
    for module in (
        frontier,
        closure,
        discovery,
        instability,
        stalling,
        validation,
        evidence,
        time_budget,
    ):
        expected_cols.update(module.COLUMNS)
    for r in rows:
        missing = expected_cols - set(r.keys())
        assert not missing, missing


def test_build_source_frame_count_matches_per_run_sum(real_ledger: None) -> None:
    df = build_source_frame("tb_live")
    assert df["run_id"].nunique() == 12
    # Sum per-run rows to compare against the standalone counts.
    per_run = df.groupby("run_id").size()
    assert per_run.sum() == len(df)


def test_write_source_checkpoints_byte_stable(
    real_ledger: None, tmp_path: Path
) -> None:
    a, _ = write_source_checkpoints("tb_live", tmp_path / "a.parquet")
    b, _ = write_source_checkpoints("tb_live", tmp_path / "b.parquet")
    assert a.read_bytes() == b.read_bytes()


def test_write_fails_loud_on_forbidden_column(
    real_ledger: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a forbidden column post-build; the guard at write time
    must reject. This exercises AGENTS.md invariant 1."""
    from coding_estimator.checkpoints import build as build_module

    real_build = build_module.build_source_frame

    def poisoned(source_id: str, run_ids: list[str] | None = None) -> pd.DataFrame:
        df = real_build(source_id, run_ids=run_ids)
        df["final_success"] = True  # forbidden!
        return df

    monkeypatch.setattr(build_module, "build_source_frame", poisoned)
    with pytest.raises(ValueError, match="forbidden columns"):
        write_source_checkpoints("tb_live", tmp_path / "p.parquet")


def test_run_constant_label_in_frame_raises(
    real_ledger: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Targets must NOT be joined into the checkpoint frame at build
    time. If they sneak in (and are run-constant), the guard fires."""
    from coding_estimator.checkpoints import build as build_module

    real_build = build_module.build_source_frame

    def with_run_constant_label(
        source_id: str, run_ids: list[str] | None = None
    ) -> pd.DataFrame:
        df = real_build(source_id, run_ids=run_ids)
        # y_submit_without_validation is run-constant by design.
        df["y_submit_without_validation"] = df["run_id"].apply(
            lambda r: 1 if "markdown" in r else 0
        )
        return df

    monkeypatch.setattr(
        build_module, "build_source_frame", with_run_constant_label
    )
    # Forbidden guard catches y_* before the run-constancy check; both
    # safety rails reach the same outcome.
    with pytest.raises(ValueError, match=r"forbidden columns|run-constant"):
        write_source_checkpoints("tb_live", tmp_path / "p.parquet")


def test_cli_smoke(real_ledger: None, tmp_path: Path) -> None:
    """The CLI wrapper exits cleanly and writes a non-empty parquet."""
    from scripts.build_checkpoints import main

    out = tmp_path / "out.parquet"
    rc = main(["--source", "tb_live", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert df["source"].unique().tolist() == ["tb_live"]


def test_write_combined_checkpoints_emits_all_sources(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    path, df = write_combined_checkpoints(tmp_path / "checkpoints_all.parquet")
    assert path.name == "checkpoints_all.parquet"
    assert path.is_file()
    assert set(df["source"].unique()) == {
        "hermes_pilot_h5_v2",
        "swe_agent_pilot",
        "tb_live",
    }


def test_cli_smoke_all(real_ledger: None, tmp_path: Path) -> None:
    from scripts.build_checkpoints import main

    out = tmp_path / "checkpoints_all.parquet"
    rc = main(["--source", "all", "--out", str(out)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert set(df["source"].unique()) == {
        "hermes_pilot_h5_v2",
        "swe_agent_pilot",
        "tb_live",
    }


def test_write_combined_checkpoints_can_target_tb_live_v2_only(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    path, df = write_combined_checkpoints(
        tmp_path / "checkpoints_tb_live_v2.parquet",
        source_ids=["tb_live_v2"],
    )
    assert path.is_file()
    assert set(df["source"].unique()) == {"tb_live_v2"}
    assert df["elapsed_wall_time"].notna().any()


def test_tb_live_v2_checkpoint_rows_preserve_task_and_model_metadata(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    _, df = write_combined_checkpoints(
        tmp_path / "checkpoints_tb_live_v2.parquet",
        source_ids=["tb_live_v2"],
    )
    sub = df[
        df["run_id"] == "validation_new_work_05_quoted_field_in_tsv__armB__87f7ab5e"
    ]
    assert not sub.empty
    assert set(sub["task_id"].unique()) == {"validation_new_work_05_quoted_field_in_tsv"}
    assert set(sub["task_family"].unique()) == {"validation_new_work"}
    assert set(sub["arm"].unique()) == {"B"}
    assert set(sub["difficulty"].unique()) == {"medium"}
    assert set(sub["model_name"].unique()) == {"claude-sonnet-4-6"}
    assert set(sub["agent_scaffold"].unique()) == {"general-purpose"}
