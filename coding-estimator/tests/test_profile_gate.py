"""F11 profiling go/no-go gate (Workstream F11).

Claim:
    Five binary checks decide whether the data is ready to train on.

    C1 collapses replicated checkpoints to per-run BEFORE counting
    class balance (`y_success_eventual` is run-constant). Both classes
    must reach >= 5 on the SAME source.

    C2 counts wallclock checkpoints PER source.

    C3 enforces missingness <= 95% only on features whose
    `populated_on` covers all sources in the frame AND whose
    missingness_semantic is NOT `UNKNOWN_DUE_TO_MISSING_ARTIFACT`
    (which is structurally null by contract).

    C4 fails iff `find_forbidden(columns) != []`.

    C5 requires at least one closure feature whose worst-pair KS is
    < 0.5 across sources. A column with no valid pair (one source
    all-null) MUST NOT qualify.

    `write_gate_report` writes the markdown report THEN raises
    `GateFailedError` on FAIL. The report must exist on disk before
    the exception fires.

Plausible wrong implementations:
    - C1 counts checkpoints not runs: 5 successes × 5 ckpts = 25,
      passes the >= 5 gate even though there are only 5 runs.
    - C1 OR'd succ-on-source-A with fail-on-source-B: passes when
      no source has both classes (a real false-positive risk).
    - C2 counts globally instead of per-source.
    - C3 forgets to exempt UNKNOWN_DUE_TO_MISSING_ARTIFACT and
      fails on `agent_scaffold` 100% null.
    - C5 returns `qualifying=[col]` when worst stays at -1 (no pairs
      had data) — a sentinel that wasn't checked before qualification.
    - `write_gate_report` raises BEFORE writing the report, so a
      forensic CI run has no artifact to inspect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.checkpoints.features.missingness import Missingness
from coding_estimator.checkpoints.features.registry import Feature
from coding_estimator.profile import gate as gate_module
from coding_estimator.profile.gate import (
    GateCheck,
    GateFailedError,
    GateResult,
    check_closure_cross_source_ks,
    check_global_feature_missingness,
    check_no_forbidden_columns,
    check_terminal_class_balance,
    check_wallclock_coverage,
    run_gate,
    write_gate_report,
)


def _label_row(
    run_id: str, value: float, source: str = "tb_live", checkpoint_step: int = 0
) -> dict:
    return {
        "run_id": run_id,
        "source": source,
        "checkpoint_id": f"{run_id}::{checkpoint_step}",
        "checkpoint_step": checkpoint_step,
        "is_terminal_checkpoint": False,
        "target_name": "y_success_eventual",
        "target_family": "success",
        "target_horizon_units": "terminal",
        "target_horizon_value": None,
        "label_value": value,
        "is_masked": False,
        "mask_reason": None,
        "label_available": True,
        "schema_version": "0.1.0",
    }


# ---- C1: terminal class balance ----


def test_c1_collapses_per_run_not_per_checkpoint() -> None:
    """REGRESSION: 3 successful runs each replicated 5 times +
    6 failed runs each replicated 3 times produces 15 + 18 = 33
    label rows. Counting at row level would say n_succ=15, n_fail=18
    (passes >= 5). Counting at run level (correct) says n_succ=3,
    n_fail=6 — fails because n_succ < 5."""
    rows = []
    for i in range(3):
        for ck in range(5):  # 5 checkpoints per run
            rows.append(_label_row(f"succ_{i}", 1.0, checkpoint_step=ck))
    for i in range(6):
        for ck in range(3):
            rows.append(_label_row(f"fail_{i}", 0.0, checkpoint_step=ck))
    df = pd.DataFrame(rows)
    out = check_terminal_class_balance(df)
    assert out.passed is False, out.detail
    # Detail must reflect run-level counts.
    assert "positives=3" in out.detail
    assert "negatives=6" in out.detail


def test_c1_passes_when_at_least_one_source_has_5_of_each() -> None:
    rows = []
    for i in range(5):
        rows.append(_label_row(f"s_{i}", 1.0))
    for i in range(5):
        rows.append(_label_row(f"f_{i}", 0.0))
    df = pd.DataFrame(rows)
    out = check_terminal_class_balance(df)
    assert out.passed is True


def test_c1_fails_when_classes_split_across_sources() -> None:
    """REGRESSION: 10 successes on source A and 10 failures on
    source B must NOT pass. The contract is BOTH classes on the
    SAME source. A naive implementation that ANDs across sources
    would wrongly pass here."""
    rows = []
    for i in range(10):
        rows.append(_label_row(f"a_{i}", 1.0, source="src_a"))
    for i in range(10):
        rows.append(_label_row(f"b_{i}", 0.0, source="src_b"))
    df = pd.DataFrame(rows)
    out = check_terminal_class_balance(df)
    assert out.passed is False


def test_c1_off_by_one_at_threshold() -> None:
    """Threshold is `>= 5`. 4 successes + 5 failures must FAIL.
    5 + 5 must PASS. Catches a `> 5` boundary regression."""
    rows4_5 = [_label_row(f"s_{i}", 1.0) for i in range(4)] + [
        _label_row(f"f_{i}", 0.0) for i in range(5)
    ]
    rows5_5 = [_label_row(f"s_{i}", 1.0) for i in range(5)] + [
        _label_row(f"f_{i}", 0.0) for i in range(5)
    ]
    assert check_terminal_class_balance(pd.DataFrame(rows4_5)).passed is False
    assert check_terminal_class_balance(pd.DataFrame(rows5_5)).passed is True


# ---- C2: wallclock coverage ----


def _ckpt_row(source: str, run_id: str, step: int, **kw) -> dict:
    base = {
        "source": source,
        "run_id": run_id,
        "checkpoint_id": f"{run_id}::{step}",
        "checkpoint_step": step,
    }
    base.update(kw)
    return base


def test_c2_counts_wallclock_per_source_not_global() -> None:
    """Two sources with 30 wallclock-non-null rows each (60 globally)
    must FAIL the per-source >= 50 check. A global-count regression
    would pass."""
    rows = []
    for i in range(30):
        rows.append(_ckpt_row("src_a", f"a_{i}", 0, elapsed_wall_time=1.0))
    for i in range(30):
        rows.append(_ckpt_row("src_b", f"b_{i}", 0, elapsed_wall_time=1.0))
    df = pd.DataFrame(rows)
    assert check_wallclock_coverage(df).passed is False


def test_c2_passes_when_one_source_has_50_wallclock() -> None:
    rows = [
        _ckpt_row("src_a", f"a_{i}", 0, elapsed_wall_time=1.0)
        for i in range(50)
    ] + [
        _ckpt_row("src_b", f"b_{i}", 0, elapsed_wall_time=None)
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    assert check_wallclock_coverage(df).passed is True


# ---- C3: global feature missingness ----


def test_c3_exempts_unknown_due_to_missing_artifact_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: a feature with semantic=UNKNOWN_DUE_TO_MISSING_ARTIFACT
    and 100% null cells must NOT trip C3. The semantic explicitly
    means "missing-by-contract"."""
    fake = Feature(
        column_name="model_name",
        dtype="str",
        group="source_task",
        populated_on=("tb_live",),
        upstream_source=None,
        missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT,
        run_constant_flag=True,
    )
    monkeypatch.setattr(gate_module, "all_features", lambda: [fake])
    df = pd.DataFrame({
        "source": ["tb_live"] * 4,
        "run_id": [f"r{i}" for i in range(4)],
        "model_name": [None, None, None, None],  # 100% null
    })
    out = check_global_feature_missingness(df)
    assert out.passed is True


def test_c3_fires_on_applicable_never_observed_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature with semantic=APPLICABLE_NEVER_OBSERVED_IN_RUN that
    is 100% null DOES trip C3. Contrast with the previous test:
    only UNKNOWN_DUE_TO_MISSING_ARTIFACT is exempt."""
    fake = Feature(
        column_name="probe_signal",
        dtype="bool",
        group="closure",
        populated_on=("tb_live",),
        upstream_source=None,
        missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN,
        run_constant_flag=False,
    )
    monkeypatch.setattr(gate_module, "all_features", lambda: [fake])
    df = pd.DataFrame({
        "source": ["tb_live"] * 4,
        "run_id": [f"r{i}" for i in range(4)],
        "probe_signal": [None, None, None, None],
    })
    out = check_global_feature_missingness(df)
    assert out.passed is False
    assert "probe_signal" in out.detail


def test_c3_skips_features_not_populated_on_all_sources_in_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature whose `populated_on` does NOT cover every source in
    the frame is structurally null elsewhere; C3 must skip it. (E.g.
    `elapsed_wall_time` is wallclock-sources-only.)"""
    wallclock_only = Feature(
        column_name="elapsed_wall_time",
        dtype="float",
        group="time_budget",
        populated_on=("tb_live",),  # NOT in 'src_other'
        upstream_source=None,
        missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN,
        run_constant_flag=False,
    )
    monkeypatch.setattr(gate_module, "all_features", lambda: [wallclock_only])
    df = pd.DataFrame({
        "source": ["tb_live"] * 2 + ["src_other"] * 2,
        "run_id": ["r1", "r2", "r3", "r4"],
        "elapsed_wall_time": [None, None, None, None],
    })
    out = check_global_feature_missingness(df)
    assert out.passed is True


# ---- C4: forbidden columns ----


def test_c4_passes_on_clean_frame() -> None:
    df = pd.DataFrame({
        "source": ["tb_live"],
        "run_id": ["r1"],
        "coding_progress": [0.5],
    })
    assert check_no_forbidden_columns(df).passed is True


def test_c4_fails_when_a_forbidden_token_is_present() -> None:
    """A column named `final_success` (in the forbidden exact list)
    must trip C4."""
    df = pd.DataFrame({
        "source": ["tb_live"],
        "run_id": ["r1"],
        "final_success": [True],
    })
    out = check_no_forbidden_columns(df)
    assert out.passed is False
    assert "final_success" in out.detail


# ---- C5: closure-feature cross-source KS ----


def test_c5_single_source_returns_passed_with_short_circuit() -> None:
    """KS is undefined with a single source. The check must pass
    with detail "only 1 source" rather than crashing or returning
    False."""
    df = pd.DataFrame({
        "source": ["tb_live"] * 4,
        "run_id": [f"r{i}" for i in range(4)],
        "coding_progress": [0.0, 0.5, 0.5, 1.0],
        "completed_leaf_count": [0, 1, 1, 2],
        "validation_progress": [0.0, 0.0, 1.0, 1.0],
        "product_progress": [0.0, 0.5, 0.5, 1.0],
        "investigation_progress": [0.0, 0.0, 0.5, 0.5],
    })
    out = check_closure_cross_source_ks(df)
    assert out.passed is True
    assert "only 1 source" in out.detail


def test_c5_does_not_qualify_column_with_no_valid_pair() -> None:
    """REGRESSION: closure column has data on src_a but is
    all-null on src_b. The pair (src_a, src_b) is skipped. With
    only one source pair total, n_pairs=0, worst stays at -1 (sentinel),
    and the column must NOT be added to qualifying. Without the
    sentinel guard, a `worst < 0.5` comparison would silently pass
    a column with no comparison data."""
    df = pd.DataFrame({
        "source": ["src_a", "src_a", "src_b", "src_b"],
        "run_id": ["a1", "a2", "b1", "b2"],
        # src_a has data, src_b all null on every closure column.
        "coding_progress": [0.5, 0.5, None, None],
        "completed_leaf_count": [1, 2, None, None],
        "validation_progress": [0.0, 1.0, None, None],
        "product_progress": [0.5, 0.5, None, None],
        "investigation_progress": [0.0, 0.5, None, None],
    })
    out = check_closure_cross_source_ks(df)
    assert out.passed is False, out.detail
    # Detail must mention the skip reason.
    assert "skipped" in out.detail


def test_c5_passes_when_distributions_are_close() -> None:
    """Two sources with identical closure distributions have KS = 0,
    which is < 0.5 — the check must pass with the column qualifying."""
    df = pd.DataFrame({
        "source": ["src_a"] * 4 + ["src_b"] * 4,
        "run_id": [f"r{i}" for i in range(8)],
        "coding_progress": [0.0, 0.25, 0.5, 1.0] * 2,
        "completed_leaf_count": [0, 1, 2, 3] * 2,
        "validation_progress": [0.0, 0.0, 0.5, 1.0] * 2,
        "product_progress": [0.0, 0.25, 0.5, 1.0] * 2,
        "investigation_progress": [0.0, 0.5, 0.5, 1.0] * 2,
    })
    out = check_closure_cross_source_ks(df)
    assert out.passed is True


def test_c5_fails_when_all_closure_features_diverge() -> None:
    """If every closure feature has worst-pair KS = 1.0 (perfect
    divergence: src_a all 0, src_b all 1), no feature qualifies and
    C5 fails."""
    df = pd.DataFrame({
        "source": ["src_a"] * 4 + ["src_b"] * 4,
        "run_id": [f"r{i}" for i in range(8)],
        "coding_progress": [0.0] * 4 + [1.0] * 4,
        "completed_leaf_count": [0] * 4 + [10] * 4,
        "validation_progress": [0.0] * 4 + [1.0] * 4,
        "product_progress": [0.0] * 4 + [1.0] * 4,
        "investigation_progress": [0.0] * 4 + [1.0] * 4,
    })
    out = check_closure_cross_source_ks(df)
    assert out.passed is False


# ---- write_gate_report hard-fail ----


def test_write_gate_report_writes_artifact_then_raises_on_fail(tmp_path: Path) -> None:
    """REGRESSION: `write_gate_report` must hard-fail on any failing
    check, BUT the markdown report must exist on disk before the
    exception fires (forensic-friendly). A `raise then write` order
    would leave no artifact behind."""
    failing = GateResult(checks=(
        GateCheck(name="C_fake", passed=False, detail="synthetic failure"),
    ))
    out_dir = tmp_path / "reports"
    with pytest.raises(GateFailedError) as excinfo:
        write_gate_report(failing, out_dir)
    target = out_dir / "F_profiling_go_no_go.md"
    # The artifact must be written before the exception.
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "FAIL" in text
    assert "C_fake" in text
    assert "C_fake" in str(excinfo.value)


def test_write_gate_report_does_not_raise_on_pass(tmp_path: Path) -> None:
    passing = GateResult(checks=(
        GateCheck(name="C_fake", passed=True, detail="ok"),
    ))
    target = write_gate_report(passing, tmp_path)
    assert target.is_file()
    assert "PASS" in target.read_text(encoding="utf-8")


def test_run_gate_assembles_all_five_checks() -> None:
    """`run_gate` must invoke all five checks. A regression that
    drops one check would silently weaken the gate."""
    labels = pd.DataFrame([
        _label_row(f"s_{i}", 1.0) for i in range(5)
    ] + [
        _label_row(f"f_{i}", 0.0) for i in range(5)
    ])
    ckpts = pd.DataFrame([
        _ckpt_row("tb_live", f"r{i}", 0, coding_progress=0.5,
                  elapsed_wall_time=1.0, completed_leaf_count=1,
                  validation_progress=0.5, product_progress=0.5,
                  investigation_progress=0.0)
        for i in range(60)
    ])
    result = run_gate(labels_df=labels, checkpoints_df=ckpts)
    names = {c.name for c in result.checks}
    assert names == {
        "C1_terminal_class_balance",
        "C2_wallclock_coverage",
        "C3_global_feature_missingness",
        "C4_no_forbidden_columns",
        "C5_closure_cross_source_ks",
    }


def test_gate_result_passed_property_requires_all_checks_pass() -> None:
    all_pass = GateResult(checks=(
        GateCheck(name="A", passed=True, detail=""),
        GateCheck(name="B", passed=True, detail=""),
    ))
    one_fail = GateResult(checks=(
        GateCheck(name="A", passed=True, detail=""),
        GateCheck(name="B", passed=False, detail=""),
    ))
    assert all_pass.passed is True
    assert one_fail.passed is False
    assert one_fail.failed[0].name == "B"
