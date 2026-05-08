"""H1 — split builder writes valid JSON, honors B5 disjointness, and
correctly slices per source.

Claim:
    For a frame with K sources and runs-per-source n_s, the builder
    emits one file per (scheme, source) for {loro, holdout, temporal*,
    ltfo*} (* = subject to feasibility) plus exactly one `loso_all.json`
    when K >= 2. Every emitted file matches `schemas/split_schema.json`,
    `assert_disjoint` holds on every fold, and per-source loro folds
    contain only run_ids drawn from that source.

Plausible wrong implementations:
    - LOSO emitted per source instead of once across all sources
    - LTFO emitted for sources with no task_family (would crash)
    - run_ids from source A leak into a per-source split for source B
    - JSON missing schema_version or scheme field
    - On-disk JSON differs from in-memory Split.to_dict
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pandas as pd
import pytest

from coding_estimator.splits import builder as B
from coding_estimator.splits.protocol import Split

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "split_schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())


def _frame(*, with_families: bool = True) -> pd.DataFrame:
    rows = []
    for i, (rid, src, fam) in enumerate(
        [
            ("a1", "src_a", "fx" if with_families else None),
            ("a2", "src_a", "fy" if with_families else None),
            ("a3", "src_a", "fx" if with_families else None),
            ("b1", "src_b", "fz" if with_families else None),
            ("b2", "src_b", "fz" if with_families else None),
        ]
    ):
        for c in range(2):
            rows.append(
                {
                    "run_id": rid,
                    "source": src,
                    "checkpoint_id": f"{rid}::{c}",
                    "checkpoint_step": c,
                    "checkpoint_wall_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                    "timestamp_quality": "real",
                    "task_family": fam,
                }
            )
    return pd.DataFrame(rows)


def _families_from_frame(df: pd.DataFrame, src: str) -> dict[str, str | None]:
    sub = df[df["source"] == src][["run_id", "task_family"]].drop_duplicates()
    return dict(zip(sub["run_id"], sub["task_family"], strict=True))


def test_build_split_loro_disjoint() -> None:
    df = _frame()
    src_df = df[df["source"] == "src_a"]
    s = B.build_split("loro", src_df)
    assert s is not None
    jsonschema.validate(s.to_dict(), SCHEMA)
    runs = set(src_df["run_id"].unique())
    for f in s.folds:
        assert set(f.train_run_ids).isdisjoint(set(f.test_run_ids))
        assert set(f.train_run_ids) | set(f.test_run_ids) == runs


def test_build_split_ltfo_skips_without_families() -> None:
    df = _frame(with_families=False)
    src_df = df[df["source"] == "src_a"]
    fams = _families_from_frame(df, "src_a")
    s = B.build_split("ltfo", src_df, task_families=fams)
    assert s is None


def test_build_split_loso_skips_single_source() -> None:
    df = _frame()
    src_df = df[df["source"] == "src_a"]
    assert B.build_split("loso", src_df) is None


def test_build_split_loso_two_folds_when_two_sources() -> None:
    df = _frame()
    s = B.build_split("loso", df)
    assert s is not None
    assert {f.fold_id for f in s.folds} == {"loso::src_a", "loso::src_b"}


def test_build_split_unknown_scheme_raises() -> None:
    with pytest.raises(KeyError):
        B.build_split("nonsense", _frame())


def test_per_source_split_contains_only_that_source_runs(tmp_path: Path, monkeypatch) -> None:
    df = _frame()
    monkeypatch.setattr(B, "task_family_map", lambda src: _families_from_frame(df, src))
    out = B.build_all(df, tmp_path)
    a_runs = set(df[df["source"] == "src_a"]["run_id"].unique())
    loro_a = json.loads((tmp_path / "loro_src_a.json").read_text())
    for fold in loro_a["folds"]:
        assert set(fold["train_run_ids"]) | set(fold["test_run_ids"]) <= a_runs
    paths = {p.name for p in out}
    assert "loso_all.json" in paths
    assert {"loro_src_a.json", "loro_src_b.json"} <= paths


def test_emitted_json_matches_in_memory_split(tmp_path: Path, monkeypatch) -> None:
    df = _frame()
    monkeypatch.setattr(B, "task_family_map", lambda src: _families_from_frame(df, src))
    src_df = df[df["source"] == "src_a"]
    expected = B.build_split("loro", src_df, task_families=_families_from_frame(df, "src_a"))
    assert isinstance(expected, Split)
    B.build_all(df, tmp_path)
    on_disk = json.loads((tmp_path / "loro_src_a.json").read_text())
    assert on_disk == expected.to_dict()
    jsonschema.validate(on_disk, SCHEMA)


def test_loso_emitted_exactly_once(tmp_path: Path, monkeypatch) -> None:
    df = _frame()
    monkeypatch.setattr(B, "task_family_map", lambda src: _families_from_frame(df, src))
    out = B.build_all(df, tmp_path)
    loso_files = [p for p in out if p.name.startswith("loso_")]
    assert len(loso_files) == 1
    assert loso_files[0].name == "loso_all.json"


def test_tb_live_v2_ltfo_groups_by_exact_task_id_not_coarse_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim:
        `task_family_map("tb_live_v2")` uses exact `task_id`, so
        same-task multi-arm replications stay in one LTFO group.

    Plausible wrong implementations:
        - return `rec.task_family` for tb_live_v2, collapsing all
          `validation_new_work_*` tasks into one group
        - return `run_id`, splitting different arms of the same task
        - silently ignore the tb_live_v2 special case
    """
    runs_root = tmp_path / "tb_live_v2"
    for rid in ("task_a__armA", "task_a__armB", "task_b__armA"):
        run_dir = runs_root / rid
        run_dir.mkdir(parents=True)
        (run_dir / "ledger.jsonl").write_text("{}\n", encoding="utf-8")

    fake_runs = {
        "task_a__armA": SimpleNamespace(
            run_id="task_a__armA",
            task_id="task_a",
            task_family="validation_new_work",
        ),
        "task_a__armB": SimpleNamespace(
            run_id="task_a__armB",
            task_id="task_a",
            task_family="validation_new_work",
        ),
        "task_b__armA": SimpleNamespace(
            run_id="task_b__armA",
            task_id="task_b",
            task_family="validation_new_work",
        ),
    }

    monkeypatch.setattr(B, "runs_root", lambda _src: runs_root)
    monkeypatch.setattr(B, "load_run", lambda _src, rid: fake_runs[rid])

    groups = B.task_family_map("tb_live_v2")
    assert groups["task_a__armA"] == "task_a"
    assert groups["task_a__armB"] == "task_a"
    assert groups["task_b__armA"] == "task_b"
    assert len(set(groups.values())) == 2
