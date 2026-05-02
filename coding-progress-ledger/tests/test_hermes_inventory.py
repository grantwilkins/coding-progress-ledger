"""Hermes inventory tests (HP2). Mirrors test_swe_agent_inventory.py."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hermes_inventory import (
    CSV_COLUMNS,
    _format_cell,
    _row_to_record,
    _write_csv,
)


def _make_row(*, id="0c699abf-bc77-454a-8197-d56a2294098a", category="Terminal & Coding",
              subcategory="X", conversations=None):
    if conversations is None:
        conversations = [{"from": "system", "value": "sp"}, {"from": "human", "value": "hi"}]
    return {
        "id": id,
        "task": "do x",
        "tools": "[]",
        "category": category,
        "subcategory": subcategory,
        "conversations": conversations,
    }


def test_record_has_every_csv_column():
    rec = _row_to_record(_make_row(), config="kimi", row_index=0)
    for col in CSV_COLUMNS:
        assert col in rec


def test_final_success_always_unavailable_and_blank():
    rec = _row_to_record(_make_row(), config="kimi", row_index=0)
    assert rec["final_success_available"] is False
    assert rec["final_success"] == ""


def test_patch_and_eval_log_always_false():
    rec = _row_to_record(_make_row(), config="kimi", row_index=0)
    assert rec["patch_available"] is False
    assert rec["eval_log_available"] is False


def test_empty_conversations_list_marks_unavailable():
    rec = _row_to_record(_make_row(conversations=[]), config="kimi", row_index=0)
    assert rec["trajectory_available"] is False
    assert rec["trajectory_length"] == 0


def test_trajectory_length_counts_conversation_entries():
    rec = _row_to_record(_make_row(conversations=[{"from": "system", "value": "x"}] * 7),
                         config="kimi", row_index=0)
    assert rec["trajectory_length"] == 7


def test_model_name_is_the_config():
    rec = _row_to_record(_make_row(), config="glm-5.1", row_index=42)
    assert rec["model_name"] == "glm-5.1"
    assert "glm-5.1" in rec["raw_path_or_dataset_index"]


def test_format_cell_renders_bools_as_words():
    assert _format_cell(True) == "True"
    assert _format_cell(False) == "False"
    assert _format_cell(None) == ""
    assert _format_cell("x") == "x"


def test_write_csv_byte_identical_under_permutation(tmp_path):
    a = _row_to_record(_make_row(id="aaaa"), config="kimi", row_index=0)
    b = _row_to_record(_make_row(id="bbbb"), config="kimi", row_index=1)
    c = _row_to_record(_make_row(id="cccc"), config="kimi", row_index=2)
    out1 = tmp_path / "a.csv"
    out2 = tmp_path / "b.csv"
    _write_csv([a, b, c], out1)
    _write_csv([c, a, b], out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_write_csv_sorts_by_model_then_instance_id(tmp_path):
    r1 = _row_to_record(_make_row(id="zzzz"), config="kimi", row_index=0)
    r2 = _row_to_record(_make_row(id="aaaa"), config="glm-5.1", row_index=1)
    r3 = _row_to_record(_make_row(id="bbbb"), config="kimi", row_index=2)
    out = tmp_path / "x.csv"
    _write_csv([r1, r2, r3], out)
    rows = list(csv.DictReader(out.open("r", encoding="utf-8", newline="")))
    pairs = [(r["model_name"], r["instance_id"]) for r in rows]
    assert pairs == [("glm-5.1", "aaaa"), ("kimi", "bbbb"), ("kimi", "zzzz")]


def test_write_csv_lf_terminator_no_cr(tmp_path):
    rec = _row_to_record(_make_row(), config="kimi", row_index=0)
    out = tmp_path / "x.csv"
    _write_csv([rec], out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    first = raw.split(b"\n", 1)[0].decode("utf-8")
    assert first == ",".join(CSV_COLUMNS)


def test_columns_include_category_subcategory_no_repo_or_issue():
    assert "category" in CSV_COLUMNS
    assert "subcategory" in CSV_COLUMNS
    assert "repo_name" not in CSV_COLUMNS
    assert "issue_id" not in CSV_COLUMNS
