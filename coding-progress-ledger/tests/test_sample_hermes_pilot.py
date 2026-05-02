"""Hermes pilot sampler tests (HP2)."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_hermes_pilot import (
    CSV_COLUMNS,
    _apply_filters,
    _format_pilot_id,
    main,
    select_pilots,
)


def _inv(*, instance_id, model_name="kimi", category="Terminal & Coding",
         subcategory="x", traj_len=10, traj_avail=True, idx=0):
    return {
        "source_id": f"hermes:{model_name}:{instance_id}",
        "instance_id": instance_id,
        "model_name": model_name,
        "trajectory_available": "True" if traj_avail else "False",
        "trajectory_length": str(traj_len),
        "final_success_available": "False",
        "final_success": "",
        "patch_available": "False",
        "eval_log_available": "False",
        "category": category,
        "subcategory": subcategory,
        "raw_path_or_dataset_index": f"lambda/hermes-agent-reasoning-traces:{model_name}:train:{idx}",
        "parse_status": "ok",
        "parse_error": "",
    }


def test_I1_filters_non_terminal_coding_category():
    rows = [
        _inv(instance_id="a", category="Terminal & Coding"),
        _inv(instance_id="b", category="Agent Tools"),
    ]
    filtered, funnel = _apply_filters(rows)
    assert funnel["after_I1_category"] == 1
    assert filtered[0]["instance_id"] == "a"


def test_I2_filters_non_kimi_config():
    rows = [
        _inv(instance_id="a", model_name="kimi"),
        _inv(instance_id="b", model_name="glm-5.1"),
    ]
    filtered, funnel = _apply_filters(rows)
    assert funnel["after_I2_config"] == 1


def test_I3_min_conv_length_inclusive_at_threshold():
    rows = [
        _inv(instance_id="a", traj_len=5),
        _inv(instance_id="b", traj_len=6),
        _inv(instance_id="c", traj_len=20),
    ]
    filtered, funnel = _apply_filters(rows)
    assert funnel["after_I3_min_len"] == 2
    iids = [r["instance_id"] for r in filtered]
    assert "a" not in iids
    assert "b" in iids


def test_I4_traj_available_false_excluded():
    rows = [
        _inv(instance_id="a", traj_avail=True),
        _inv(instance_id="b", traj_avail=False),
    ]
    filtered, funnel = _apply_filters(rows)
    assert funnel["after_I4_traj_available"] == 1


def test_I5_dedupe_on_instance_id():
    rows = [
        _inv(instance_id="dup", idx=0),
        _inv(instance_id="dup", idx=1),
        _inv(instance_id="other", idx=2),
    ]
    filtered, funnel = _apply_filters(rows)
    assert funnel["after_I5_dedupe"] == 2


def test_select_pilots_returns_first_n_in_id_sorted_order():
    rows = [
        _inv(instance_id="ddd"),
        _inv(instance_id="aaa"),
        _inv(instance_id="ccc"),
        _inv(instance_id="bbb"),
        _inv(instance_id="eee"),
        _inv(instance_id="fff"),
    ]
    out, _ = select_pilots(rows, n=5, seed=0)
    assert [r["instance_id"] for r in out] == ["aaa", "bbb", "ccc", "ddd", "eee"]


def test_pilot_ids_are_id_sorted_not_input_order():
    rows = [
        _inv(instance_id="zzz"),
        _inv(instance_id="aaa"),
    ]
    out, _ = select_pilots(rows, n=2, seed=0)
    assert out[0]["pilot_id"] == "hermes_pilot_01"
    assert out[0]["instance_id"] == "aaa"
    assert out[1]["pilot_id"] == "hermes_pilot_02"


def test_format_pilot_id_zero_padded_two_digits():
    assert _format_pilot_id(1) == "hermes_pilot_01"
    assert _format_pilot_id(5) == "hermes_pilot_05"


def test_byte_determinism_across_input_permutations(tmp_path):
    rows_a = [_inv(instance_id=c) for c in ["ccc", "aaa", "bbb", "ddd", "eee", "fff"]]
    rows_b = [_inv(instance_id=c) for c in ["fff", "eee", "aaa", "bbb", "ccc", "ddd"]]
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    _write_inventory(rows_a, out_a.with_suffix(".inv.csv"))
    _write_inventory(rows_b, out_b.with_suffix(".inv.csv"))
    main(["--inventory-csv", str(out_a.with_suffix(".inv.csv")), "--out-csv", str(out_a),
          "--n-pilots", "5", "--seed", "0"])
    main(["--inventory-csv", str(out_b.with_suffix(".inv.csv")), "--out-csv", str(out_b),
          "--n-pilots", "5", "--seed", "0"])
    assert out_a.read_bytes() == out_b.read_bytes()


def _write_inventory(rows, path):
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def test_main_emits_5_rows_with_correct_columns(tmp_path):
    rows = [_inv(instance_id=c) for c in ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]]
    inv = tmp_path / "inv.csv"
    out = tmp_path / "pilot.csv"
    _write_inventory(rows, inv)
    rc = main(["--inventory-csv", str(inv), "--out-csv", str(out), "--n-pilots", "5", "--seed", "0"])
    assert rc == 0
    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out_rows = list(reader)
    assert len(out_rows) == 5
    assert set(out_rows[0].keys()) == set(CSV_COLUMNS)
    assert [r["pilot_id"] for r in out_rows] == [f"hermes_pilot_{i:02d}" for i in range(1, 6)]
