"""
Claim:
sample_swe_agent_pilot.py deterministically draws balanced success /
failure rows from the A3 inventory, applying the I1-I7 funnel and a
4-level fallback ladder, deduping on instance_id by lowest dataset
index, and emitting a CSV whose pilot_ids depend only on which rows
were selected (not on the seed). Same inventory + flags + seed yields
byte-identical output regardless of inventory row order.

Plausible wrong implementations:
- bool parsed via Python's bool(s) (where bool('False') is True),
  contaminating filters and split.
- dataset index compared lexically ('10' < '2'), so dedupe keeps the
  wrong row when indices cross a digit boundary.
- dedupe keeps last-seen / highest-index instead of lowest.
- sampling without first sorting the pool by instance_id, so two CSV
  read orders that produce the same rows give different picks for the
  same seed.
- pilot_ids assigned in RNG-pick order rather than sorted instance_id
  order, so two seeds that select the same rows produce different ids.
- final output sorted by something other than pilot_id, breaking the
  documented "f_* before s_*" output order.
- fallback ladder applies the wrong relaxation level (e.g. halves
  before dropping the model restriction or the trajectory threshold).
- I6 trajectory_length boundary uses '>' instead of '>=' (off-by-one
  at the exact threshold).
- _format_cell renders bools via the int subclass branch (True -> "1").
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_swe_agent_pilot import (
    CSV_COLUMNS,
    _apply_filters,
    _build_output_rows,
    _dedupe_by_instance,
    _format_pilot_id,
    _parse_bool,
    _parse_dataset_index,
    _sample_side,
    _select_with_fallbacks,
    _split_by_success,
    main,
)
import random


def _inv(
    *,
    instance_id,
    model_name="swe-agent-llama-70b",
    final_success=True,
    traj_len=20,
    idx=0,
    parse_status="ok",
    traj_avail=True,
    fs_avail=True,
    patch_avail=True,
    eval_avail=True,
):
    fs_cell = "True" if final_success is True else ("False" if final_success is False else "")
    return {
        "source_id": f"nebius:{instance_id}:{model_name}",
        "instance_id": instance_id,
        "model_name": model_name,
        "trajectory_available": "True" if traj_avail else "False",
        "trajectory_length": str(traj_len),
        "final_success_available": "True" if fs_avail else "False",
        "final_success": fs_cell,
        "patch_available": "True" if patch_avail else "False",
        "eval_log_available": "True" if eval_avail else "False",
        "repo_name": "owner/repo",
        "issue_id": "1",
        "raw_path_or_dataset_index": f"nebius/SWE-agent-trajectories:train:{idx}",
        "parse_status": parse_status,
        "parse_error": "",
    }


# ---------- primitives ----------


def test_parse_bool_is_strict_against_python_bool_coercion():
    # bool('False') is True in Python; the strict parser must reject it.
    assert _parse_bool("True") is True
    assert _parse_bool("False") is False
    assert _parse_bool("") is None
    assert _parse_bool("TRUE") is None
    assert _parse_bool("true") is None
    assert _parse_bool("1") is None
    assert _parse_bool("0") is None


def test_parse_dataset_index_returns_int_not_string():
    # Must compare numerically downstream — '10' < '2' lexically.
    assert _parse_dataset_index("nebius/SWE-agent-trajectories:train:10") == 10
    assert _parse_dataset_index("nebius/SWE-agent-trajectories:train:2") == 2
    assert _parse_dataset_index("garbage") is None
    assert _parse_dataset_index("") is None


# ---------- dedupe ----------


def test_dedupe_keeps_lowest_dataset_index_numerically_not_lexically():
    # Same instance_id at indices 2, 10, 7 — numeric min is 2;
    # lexical min would be '10'.
    rows = [
        _inv(instance_id="repo__r-1", idx=10),
        _inv(instance_id="repo__r-1", idx=2),
        _inv(instance_id="repo__r-1", idx=7),
    ]
    deduped, n_unique = _dedupe_by_instance(rows)
    assert n_unique == 1
    assert len(deduped) == 1
    assert deduped[0]["raw_path_or_dataset_index"].endswith(":2")


def test_dedupe_drops_rows_with_blank_instance_id():
    rows = [_inv(instance_id="", idx=0), _inv(instance_id="r__r-1", idx=1)]
    deduped, n_unique = _dedupe_by_instance(rows)
    assert n_unique == 1
    assert deduped[0]["instance_id"] == "r__r-1"


# ---------- filter funnel ----------


def test_filter_traj_length_boundary_is_inclusive_at_threshold():
    rows = [
        _inv(instance_id="r__r-1", traj_len=10, idx=0),  # at threshold
        _inv(instance_id="r__r-2", traj_len=9, idx=1),   # just below
    ]
    kept, _funnel = _apply_filters(rows, require_model=True, min_trajectory_length=10)
    kept_ids = {r["instance_id"] for r in kept}
    assert kept_ids == {"r__r-1"}


def test_each_filter_drops_a_dedicated_row():
    # One row violates each of I1..I7; one row passes all.
    pool = [
        _inv(instance_id="r__bad-parse", parse_status="error", idx=0),
        _inv(instance_id="r__no-traj", traj_avail=False, traj_len=0, idx=1),
        _inv(instance_id="r__no-fs", fs_avail=False, final_success=None, idx=2),
        _inv(instance_id="r__no-patch", patch_avail=False, idx=3),
        _inv(instance_id="r__no-eval", eval_avail=False, idx=4),
        _inv(instance_id="r__short-traj", traj_len=3, idx=5),
        _inv(instance_id="r__wrong-model", model_name="swe-agent-llama-8b", idx=6),
        _inv(instance_id="r__keeper", idx=7),
    ]
    kept, _ = _apply_filters(pool, require_model=True, min_trajectory_length=10)
    assert [r["instance_id"] for r in kept] == ["r__keeper"]


def test_model_filter_only_applied_when_require_model_true():
    pool = [
        _inv(instance_id="r__70b", model_name="swe-agent-llama-70b", idx=0),
        _inv(instance_id="r__8b", model_name="swe-agent-llama-8b", idx=1),
    ]
    strict, _ = _apply_filters(pool, require_model=True, min_trajectory_length=10)
    relaxed, _ = _apply_filters(pool, require_model=False, min_trajectory_length=10)
    assert {r["instance_id"] for r in strict} == {"r__70b"}
    assert {r["instance_id"] for r in relaxed} == {"r__70b", "r__8b"}


def test_split_by_success_keeps_only_explicit_true_false_rows():
    rows = [
        _inv(instance_id="a__a-1", final_success=True),
        _inv(instance_id="a__a-2", final_success=False),
        _inv(instance_id="a__a-3", final_success=None),  # missing
    ]
    success, failure = _split_by_success(rows)
    assert [r["instance_id"] for r in success] == ["a__a-1"]
    assert [r["instance_id"] for r in failure] == ["a__a-2"]


# ---------- sampling determinism ----------


def test_sample_side_is_byte_deterministic_across_input_order_permutations():
    pool_forward = [_inv(instance_id=f"r__r-{i:02d}", idx=i) for i in range(10)]
    pool_reverse = list(reversed(pool_forward))

    picks_forward = _sample_side(pool_forward, 4, random.Random(0))
    picks_reverse = _sample_side(pool_reverse, 4, random.Random(0))

    assert [r["instance_id"] for r in picks_forward] == [
        r["instance_id"] for r in picks_reverse
    ]
    # Sanity: the sample is a strict subset of size n with no duplicates.
    assert len(picks_forward) == 4
    assert len(set(r["instance_id"] for r in picks_forward)) == 4


def test_sample_side_returns_all_rows_sorted_when_pool_smaller_than_target():
    pool = [
        _inv(instance_id="r__r-3", idx=2),
        _inv(instance_id="r__r-1", idx=0),
        _inv(instance_id="r__r-2", idx=1),
    ]
    picks = _sample_side(pool, 5, random.Random(0))
    assert [r["instance_id"] for r in picks] == ["r__r-1", "r__r-2", "r__r-3"]


# ---------- pilot id assignment ----------


def test_format_pilot_id_widens_to_three_digits_when_total_exceeds_99():
    assert _format_pilot_id("s", 1, 10) == "swe_agent_pilot_s_01"
    assert _format_pilot_id("s", 1, 99) == "swe_agent_pilot_s_01"
    assert _format_pilot_id("s", 1, 100) == "swe_agent_pilot_s_001"
    assert _format_pilot_id("f", 42, 200) == "swe_agent_pilot_f_042"


def test_pilot_ids_assigned_by_sorted_instance_id_not_rng_pick_order():
    # Pass picks in shuffled (non-sorted) order; the output must
    # assign s_01 to the lexicographically smallest instance_id.
    success_picks = [
        _inv(instance_id="z__z-1", idx=0),
        _inv(instance_id="a__a-1", idx=1),
        _inv(instance_id="m__m-1", idx=2),
    ]
    failure_picks = []
    out = _build_output_rows(success_picks, failure_picks, "primary_balanced_3_0")
    by_id = {r["pilot_id"]: r["instance_id"] for r in out}
    assert by_id["swe_agent_pilot_s_01"] == "a__a-1"
    assert by_id["swe_agent_pilot_s_02"] == "m__m-1"
    assert by_id["swe_agent_pilot_s_03"] == "z__z-1"


def test_final_output_orders_failures_before_successes_by_pilot_id():
    success_picks = [_inv(instance_id="a__a-1", final_success=True, idx=0)]
    failure_picks = [_inv(instance_id="z__z-1", final_success=False, idx=1)]
    out = _build_output_rows(success_picks, failure_picks, "primary_balanced_1_1")
    assert [r["pilot_id"] for r in out] == [
        "swe_agent_pilot_f_01",
        "swe_agent_pilot_s_01",
    ]


# ---------- fallback ladder ----------


def _success(instance_id, **kw):
    return _inv(instance_id=instance_id, final_success=True, **kw)


def _failure(instance_id, **kw):
    return _inv(instance_id=instance_id, final_success=False, **kw)


def test_fallback_ladder_uses_primary_when_strict_pool_is_sufficient():
    rows = (
        [_success(f"a__a-{i}", idx=i) for i in range(2)]
        + [_failure(f"b__b-{i}", idx=10 + i) for i in range(2)]
    )
    s, f, reason, level, _funnel, _uniq = _select_with_fallbacks(
        rows, n_success=2, n_failure=2, seed=0
    )
    assert level == "primary"
    assert reason.startswith("primary")
    assert len(s) == 2 and len(f) == 2


def test_fallback_ladder_descends_to_fallback1_when_only_other_models_complete_pool():
    rows = [
        _success("a__a-1", model_name="swe-agent-llama-70b", idx=0),
        _success("a__a-2", model_name="swe-agent-llama-8b", idx=1),
        _failure("b__b-1", model_name="swe-agent-llama-70b", idx=2),
        _failure("b__b-2", model_name="swe-agent-llama-70b", idx=3),
    ]
    _s, _f, _reason, level, _funnel, _uniq = _select_with_fallbacks(
        rows, n_success=2, n_failure=2, seed=0
    )
    assert level == "fallback1"


def test_fallback_ladder_descends_to_fallback2_when_short_traj_completes_pool():
    rows = [
        _success("a__a-1", traj_len=20, idx=0),
        _success("a__a-2", traj_len=7, idx=1),  # only visible at fallback2
        _failure("b__b-1", traj_len=20, idx=2),
        _failure("b__b-2", traj_len=20, idx=3),
    ]
    _s, _f, _reason, level, _funnel, _uniq = _select_with_fallbacks(
        rows, n_success=2, n_failure=2, seed=0
    )
    assert level == "fallback2"


def test_fallback_ladder_descends_to_fallback3_when_halving_targets_completes_pool():
    rows = [
        _success("a__a-1", idx=0),  # only one success in the entire universe
        _failure("b__b-1", idx=1),
        _failure("b__b-2", idx=2),
    ]
    _s, _f, _reason, level, _funnel, _uniq = _select_with_fallbacks(
        rows, n_success=2, n_failure=2, seed=0
    )
    assert level == "fallback3"


# ---------- end-to-end byte identity through main() ----------


def _write_inventory(path: Path, rows):
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _build_inventory_rows():
    rows = []
    for i in range(8):
        rows.append(_inv(instance_id=f"s__s-{i:02d}", final_success=True, idx=i))
    for i in range(8):
        rows.append(
            _inv(instance_id=f"f__f-{i:02d}", final_success=False, idx=100 + i)
        )
    return rows


def test_main_is_byte_deterministic_across_two_runs_same_seed(tmp_path):
    inv_rows = _build_inventory_rows()
    inv = tmp_path / "inv.csv"
    _write_inventory(inv, inv_rows)

    out1 = tmp_path / "p1.csv"
    out2 = tmp_path / "p2.csv"
    rc1 = main(
        [
            "--inventory-csv", str(inv),
            "--output", str(out1),
            "--n-success", "3",
            "--n-failure", "3",
            "--seed", "0",
        ]
    )
    rc2 = main(
        [
            "--inventory-csv", str(inv),
            "--output", str(out2),
            "--n-success", "3",
            "--n-failure", "3",
            "--seed", "0",
        ]
    )
    assert rc1 == 0 and rc2 == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_main_output_is_invariant_under_inventory_row_permutation(tmp_path):
    inv_rows = _build_inventory_rows()
    inv_a = tmp_path / "inv_a.csv"
    inv_b = tmp_path / "inv_b.csv"
    _write_inventory(inv_a, inv_rows)
    # Reverse the inventory row order on disk; sampling should still pin
    # to sorted-instance_id pools, producing byte-identical output.
    _write_inventory(inv_b, list(reversed(inv_rows)))

    out_a = tmp_path / "out_a.csv"
    out_b = tmp_path / "out_b.csv"
    main([
        "--inventory-csv", str(inv_a),
        "--output", str(out_a),
        "--n-success", "3", "--n-failure", "3", "--seed", "0",
    ])
    main([
        "--inventory-csv", str(inv_b),
        "--output", str(out_b),
        "--n-success", "3", "--n-failure", "3", "--seed", "0",
    ])
    assert out_a.read_bytes() == out_b.read_bytes()


def test_main_output_changes_when_seed_changes(tmp_path):
    # Sanity: the seed must actually influence selection. A wrong impl
    # that ignores --seed would produce identical bytes here.
    inv = tmp_path / "inv.csv"
    _write_inventory(inv, _build_inventory_rows())

    out0 = tmp_path / "out0.csv"
    out1 = tmp_path / "out1.csv"
    main([
        "--inventory-csv", str(inv), "--output", str(out0),
        "--n-success", "3", "--n-failure", "3", "--seed", "0",
    ])
    main([
        "--inventory-csv", str(inv), "--output", str(out1),
        "--n-success", "3", "--n-failure", "3", "--seed", "1",
    ])
    assert out0.read_bytes() != out1.read_bytes()


def test_main_output_csv_header_matches_documented_columns(tmp_path):
    inv = tmp_path / "inv.csv"
    _write_inventory(inv, _build_inventory_rows())
    out = tmp_path / "p.csv"
    main([
        "--inventory-csv", str(inv), "--output", str(out),
        "--n-success", "3", "--n-failure", "3", "--seed", "0",
    ])
    first_line = out.read_bytes().split(b"\n", 1)[0].decode("utf-8")
    assert first_line == ",".join(CSV_COLUMNS)
