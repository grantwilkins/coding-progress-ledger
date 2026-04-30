"""
Claim:
swe_agent_inventory.py converts streamed HF rows into a deterministic
manifest CSV. Per-row metadata (presence flags, lengths, parsed
repo/issue) is captured without retaining trajectory content, and the
output is byte-identical for the same input regardless of streaming
order.

Plausible wrong implementations:
- _parse_repo_and_issue splits on the first "-" instead of the last,
  garbling repos that contain dashes (e.g. pydantic-core).
- _row_to_record conflates target=False with missing target (uses
  truthiness instead of isinstance(target, bool)), losing the
  available/value distinction.
- _row_to_record marks an empty-list trajectory as available (uses
  `is not None` instead of `len > 0`).
- _row_to_record marks patch/eval available based on truthiness of any
  value (e.g. accepting a list or empty string).
- _format_cell handles bool via the int branch (since bool subclasses
  int), rendering True/False as "1"/"0".
- Missing-bool cell renders as "None" or "False" instead of empty.
- _write_csv trusts insertion / streaming order rather than sorting,
  breaking byte-identity across reruns with reshuffled rows.
- _write_csv sorts on the wrong key (model first, or by hf_index).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.swe_agent_inventory import (
    CSV_COLUMNS,
    _format_cell,
    _parse_repo_and_issue,
    _row_to_record,
    _write_csv,
)


def _make_row(
    *,
    instance_id="astropy__astropy-12345",
    model_name="swe-agent-llama-70b",
    trajectory=None,
    target=True,
    generated_patch="diff --git a/x b/x",
    eval_logs="pytest output",
):
    if trajectory is None:
        trajectory = [{"role": "assistant"}, {"role": "user"}]
    return {
        "instance_id": instance_id,
        "model_name": model_name,
        "trajectory": trajectory,
        "target": target,
        "generated_patch": generated_patch,
        "eval_logs": eval_logs,
    }


def test_repo_with_dashes_is_parsed_via_last_dash_not_first():
    repo, issue, warn = _parse_repo_and_issue("pydantic-core__pydantic-core-1234")
    assert warn is None
    assert repo == "pydantic-core/pydantic-core"
    assert issue == "1234"


def test_repo_without_double_underscore_warns_but_keeps_issue():
    repo, issue, warn = _parse_repo_and_issue("weirdrepo-99")
    assert repo == ""
    assert issue == "99"
    assert warn == "no double-underscore in head"


def test_target_false_records_available_with_value_false():
    rec = _row_to_record(_make_row(target=False), hf_index=0)
    assert rec["final_success_available"] is True
    assert rec["final_success"] is False


def test_missing_target_records_unavailable_with_blank_value():
    rec = _row_to_record(_make_row(target=None), hf_index=0)
    assert rec["final_success_available"] is False
    assert rec["final_success"] == ""


def test_empty_trajectory_list_marks_unavailable_with_zero_length():
    rec = _row_to_record(_make_row(trajectory=[]), hf_index=0)
    assert rec["trajectory_available"] is False
    assert rec["trajectory_length"] == 0


def test_patch_available_requires_nonempty_string():
    assert _row_to_record(_make_row(generated_patch=""), hf_index=0)["patch_available"] is False
    assert _row_to_record(_make_row(generated_patch=None), hf_index=0)["patch_available"] is False
    assert _row_to_record(_make_row(generated_patch=["not", "a", "string"]), hf_index=0)["patch_available"] is False
    assert _row_to_record(_make_row(generated_patch="x"), hf_index=0)["patch_available"] is True


def test_format_cell_renders_bools_as_literal_words_not_ints():
    # bool is a subclass of int — a wrong impl using the int branch
    # would render as "1"/"0".
    assert _format_cell(True) == "True"
    assert _format_cell(False) == "False"
    assert _format_cell(1) == "1"
    assert _format_cell(0) == "0"


def test_format_cell_missing_bool_renders_blank_not_none_or_false():
    assert _format_cell("") == ""
    assert _format_cell(None) == ""


def _read_csv_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_write_csv_is_byte_identical_across_record_order_permutations(tmp_path):
    rec_a = _row_to_record(_make_row(instance_id="a__a-1"), hf_index=0)
    rec_b = _row_to_record(_make_row(instance_id="b__b-2"), hf_index=1)
    rec_c = _row_to_record(_make_row(instance_id="c__c-3"), hf_index=2)

    out1 = tmp_path / "in_order.csv"
    out2 = tmp_path / "shuffled.csv"
    _write_csv([rec_a, rec_b, rec_c], out1)
    _write_csv([rec_c, rec_a, rec_b], out2)

    assert _read_csv_bytes(out1) == _read_csv_bytes(out2)


def test_write_csv_sorts_by_instance_id_then_model_name(tmp_path):
    rec1 = _row_to_record(
        _make_row(instance_id="z__z-1", model_name="swe-agent-llama-8b"), hf_index=0
    )
    rec2 = _row_to_record(
        _make_row(instance_id="a__a-1", model_name="swe-agent-llama-70b"), hf_index=1
    )
    rec3 = _row_to_record(
        _make_row(instance_id="a__a-1", model_name="swe-agent-llama-405b"), hf_index=2
    )

    out = tmp_path / "sorted.csv"
    _write_csv([rec1, rec2, rec3], out)

    rows = list(csv.DictReader(out.open("r", encoding="utf-8", newline="")))
    pairs = [(r["instance_id"], r["model_name"]) for r in rows]
    # Lexicographic by instance_id first, then model_name.
    assert pairs == [
        ("a__a-1", "swe-agent-llama-405b"),
        ("a__a-1", "swe-agent-llama-70b"),
        ("z__z-1", "swe-agent-llama-8b"),
    ]


def test_write_csv_emits_lf_line_terminator_and_header_row(tmp_path):
    rec = _row_to_record(_make_row(instance_id="a__a-1"), hf_index=0)
    out = tmp_path / "lf.csv"
    _write_csv([rec], out)
    raw = out.read_bytes()
    # No CR characters: byte-identity across platforms requires LF only.
    assert b"\r" not in raw
    # Header is exactly the CSV_COLUMNS tuple in declared order.
    first_line = raw.split(b"\n", 1)[0].decode("utf-8")
    assert first_line == ",".join(CSV_COLUMNS)
