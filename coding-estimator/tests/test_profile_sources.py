"""F1 source-level profile (Workstream F1).

Claim:
    `source_profile_rows(manifest_df, checkpoints_df=None)` returns one
    row per source, with the per-source counts partitioned exactly:
        n_succ + n_fail + n_unres == n_runs
    A run is "unresolvable" iff its `final_success` is null OR its
    `final_success_source` is the literal "missing" enum value (set
    by C5 when the loader raises). The two predicates are OR'd via a
    boolean mask so a row that satisfies BOTH is counted once.

    Run-length quantiles are taken from `checkpoint_step.max()` per
    `run_id` when a checkpoints frame is supplied, else from the
    manifest's `ledger_event_count`.

Plausible wrong implementations:
    - n_unres computed by arithmetic
        n_unres = n_runs - n_succ - n_fail
      this papers over an inconsistent row where final_success=True
      AND final_success_source='missing': it gets absorbed into n_succ
      and disappears from n_unres. The boolean-mask approach surfaces
      it.
    - n_succ = (fs == True).sum() with no `& ~unres_mask` guard
      WRONGLY counts an inconsistent (True, "missing") row as a
      success.
    - quantile from `len(checkpoints)` per run instead of
      `max(checkpoint_step)` per run; differs by 1 (steps start at 0).
    - sort by source-id alphabetical order: a regression that
      preserves dict-iteration order would be tested for stability.
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.profile.sources import source_profile_rows


def _manifest(rows: list[dict]) -> pd.DataFrame:
    """Build a manifest DataFrame with the columns
    `source_profile_rows` reads."""
    base = []
    for r in rows:
        base.append({
            "source": r.get("source", "tb_live"),
            "run_id": r["run_id"],
            "final_success": r["final_success"],
            "final_success_source": r.get("final_success_source", "verifier_exit"),
            "ledger_event_count": r.get("ledger_event_count", 10),
            "has_real_wallclock": r.get("has_real_wallclock", False),
        })
    return pd.DataFrame(base)


def test_partition_invariant_n_succ_n_fail_n_unres_equals_n_runs() -> None:
    """For ANY manifest, the three counts must sum to n_runs. The
    runtime assertion in source_profile_rows enforces this — if a
    future regression breaks the partition (e.g. counts a row twice),
    the function raises AssertionError instead of silently returning."""
    rows = _manifest([
        {"run_id": "a", "final_success": True},
        {"run_id": "b", "final_success": False},
        {"run_id": "c", "final_success": None},
    ])
    [out] = source_profile_rows(rows)
    assert out.n_successful + out.n_failed + out.n_label_unresolvable == out.n_runs == 3


def test_inconsistent_row_true_with_missing_source_is_unresolvable_not_success() -> None:
    """REGRESSION: a row with final_success=True AND
    final_success_source='missing' is logically inconsistent. The
    arithmetic approach (n_unres = n_runs - n_succ - n_fail) would
    silently count it as a success and lose the anomaly. The
    boolean-OR approach must classify it as unresolvable."""
    rows = _manifest([
        {"run_id": "a", "final_success": True, "final_success_source": "verifier_exit"},
        {"run_id": "b", "final_success": True, "final_success_source": "missing"},
    ])
    [out] = source_profile_rows(rows)
    assert out.n_successful == 1
    assert out.n_failed == 0
    assert out.n_label_unresolvable == 1


def test_null_final_success_classified_as_unresolvable() -> None:
    rows = _manifest([
        {"run_id": "a", "final_success": None, "final_success_source": "manual"},
    ])
    [out] = source_profile_rows(rows)
    assert out.n_label_unresolvable == 1
    assert out.n_successful == 0
    assert out.n_failed == 0


def test_failure_when_source_not_missing_and_fs_false() -> None:
    rows = _manifest([
        {"run_id": "a", "final_success": False, "final_success_source": "verifier_exit"},
    ])
    [out] = source_profile_rows(rows)
    assert out.n_failed == 1
    assert out.n_successful == 0
    assert out.n_label_unresolvable == 0


def test_per_source_split_counts_independently() -> None:
    """Counts must be per-source, not global. Two sources each with
    1 success and 1 failure must produce two rows of (n_succ=1,
    n_fail=1), NOT (n_succ=2, n_fail=2)."""
    rows = _manifest([
        {"run_id": "a1", "source": "src_a", "final_success": True},
        {"run_id": "a2", "source": "src_a", "final_success": False},
        {"run_id": "b1", "source": "src_b", "final_success": True},
        {"run_id": "b2", "source": "src_b", "final_success": False},
    ])
    out = source_profile_rows(rows)
    assert len(out) == 2
    for r in out:
        assert r.n_successful == 1
        assert r.n_failed == 1
        assert r.n_runs == 2


def test_run_length_quantile_uses_checkpoint_step_max_when_supplied() -> None:
    """When a checkpoints frame is supplied, run length is the MAX
    `checkpoint_step` per run, not the manifest's
    `ledger_event_count`. A 5-step run (steps 0..5) has length 5,
    not 6 (count) and not 10 (ledger_event_count if multi-event-per-step).
    """
    manifest = _manifest([
        {"run_id": "r1", "final_success": True, "ledger_event_count": 999},
        {"run_id": "r2", "final_success": True, "ledger_event_count": 999},
    ])
    # r1 has steps 0..5 (6 checkpoints, max step = 5)
    # r2 has steps 0..3 (4 checkpoints, max step = 3)
    ckpts = pd.DataFrame([
        {"source": "tb_live", "run_id": "r1", "checkpoint_step": s} for s in range(6)
    ] + [
        {"source": "tb_live", "run_id": "r2", "checkpoint_step": s} for s in range(4)
    ])
    [out] = source_profile_rows(manifest, checkpoints_df=ckpts)
    assert out.p50_run_length_steps == 4.0  # median of (3, 5)
    assert out.p25_run_length_steps == 3.5
    assert out.p75_run_length_steps == 4.5
    # The 999 ledger_event_count must NOT bleed in.
    assert out.p50_run_length_steps != 999.0


def test_n_checkpoints_and_wallclock_counts_only_set_when_frame_supplied() -> None:
    """Without a checkpoints frame, n_checkpoints and
    n_wallclock_checkpoints are None (not 0) — the report renders "—"
    for those, distinguishing "absent" from "zero"."""
    manifest = _manifest([
        {"run_id": "r1", "final_success": True},
    ])
    [no_ck] = source_profile_rows(manifest, checkpoints_df=None)
    assert no_ck.n_checkpoints is None
    assert no_ck.n_wallclock_checkpoints is None

    ckpts = pd.DataFrame([
        {"source": "tb_live", "run_id": "r1", "checkpoint_step": 0,
         "elapsed_wall_time": 1.0},
        {"source": "tb_live", "run_id": "r1", "checkpoint_step": 1,
         "elapsed_wall_time": None},
    ])
    [with_ck] = source_profile_rows(manifest, checkpoints_df=ckpts)
    assert with_ck.n_checkpoints == 2
    assert with_ck.n_wallclock_checkpoints == 1


def test_rows_sorted_by_source_id() -> None:
    """Stable sort by source name. A change to the underlying groupby
    that returned dict-insertion order would diff against this test."""
    manifest = _manifest([
        {"run_id": "z1", "source": "z_src", "final_success": True},
        {"run_id": "a1", "source": "a_src", "final_success": True},
        {"run_id": "m1", "source": "m_src", "final_success": True},
    ])
    out = source_profile_rows(manifest)
    assert [r.source for r in out] == ["a_src", "m_src", "z_src"]


def test_partition_invariant_holds_across_random_input() -> None:
    """Property-style: for any combination of fs / source, the
    partition invariant must hold. Hand-built values that hit the
    documented edges, including both `None` and `'missing'` set."""
    cases = [
        (True, "verifier_exit"),
        (False, "verifier_exit"),
        (True, "missing"),
        (False, "missing"),
        (None, "verifier_exit"),
        (None, "missing"),
        (None, "manual"),
    ]
    rows = _manifest([
        {"run_id": f"r{i}", "final_success": fs, "final_success_source": src}
        for i, (fs, src) in enumerate(cases)
    ])
    [out] = source_profile_rows(rows)
    assert out.n_successful + out.n_failed + out.n_label_unresolvable == out.n_runs


def test_assertion_fires_on_corrupted_internal_state() -> None:
    """Sanity check on the runtime assertion. We can't easily corrupt
    the function's internals, but we can confirm that with valid input
    the assertion passes silently. (This protects against an
    accidental `assert False` regression.)"""
    rows = _manifest([
        {"run_id": "a", "final_success": True},
        {"run_id": "b", "final_success": False},
    ])
    # Must not raise.
    source_profile_rows(rows)
