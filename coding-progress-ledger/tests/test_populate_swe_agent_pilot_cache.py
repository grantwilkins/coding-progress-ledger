"""
Claim:
populate_swe_agent_pilot_cache.py walks a streaming HF iterator and
writes a per-pilot raw-row JSON to the cache. It validates each
matched row's instance_id and model_name against the pilot CSV so a
silent shift in HF streaming order would fail loudly. By default it
skips pilot_ids whose cache files already exist (idempotent reruns);
--overwrite re-fetches.

Plausible wrong implementations:
- collect_matches accepts whatever row sits at the wanted index
  without checking instance_id / model_name, so an HF re-order would
  silently corrupt the pilot cache.
- collect_matches stops at max_wanted instead of max_wanted +
  max_extra_rows, missing the row exactly at the boundary if the
  stream pads.
- collect_matches treats a duplicate wanted index as the second-wins,
  silently dropping the first.
- write_matches overwrites existing files by default, breaking
  idempotency and ignoring expensive cache state.
- Not-found pilots are silently skipped; only matched pilots get
  reported, so a partial population looks like success.
- _parse_dataset_index falls into the same lexical-vs-numeric trap as
  B2's parser.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.populate_swe_agent_pilot_cache import (
    PilotTarget,
    _parse_dataset_index,
    collect_matches,
    load_targets,
    write_matches,
)


# ---------- helpers ----------


def _row(instance_id, model_name="swe-agent-llama-70b", target=True, traj=None, patch="d", logs="l"):
    return {
        "instance_id": instance_id,
        "model_name": model_name,
        "target": target,
        "trajectory": traj or [],
        "exit_status": "x",
        "generated_patch": patch,
        "eval_logs": logs,
    }


def _stream(*rows):
    for i, r in enumerate(rows):
        yield i, r


def _stream_at(pairs):
    """Yield (hf_index, row) pairs from explicit indices."""
    for i, r in pairs:
        yield i, r


# ---------- _parse_dataset_index ----------


def test_parse_dataset_index_returns_int_for_well_formed_path():
    assert _parse_dataset_index("nebius/SWE-agent-trajectories:train:42") == 42
    assert _parse_dataset_index("nebius/SWE-agent-trajectories:train:0") == 0


def test_parse_dataset_index_returns_none_for_bad_input():
    assert _parse_dataset_index("") is None
    assert _parse_dataset_index("garbage") is None


# ---------- load_targets ----------


def _write_csv(path: Path, rows):
    cols = [
        "pilot_id", "source_id", "instance_id", "model_name",
        "final_success", "trajectory_length", "patch_available",
        "eval_log_available", "raw_path_or_dataset_index", "selection_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _csv_row(pilot_id, instance_id, idx, model="swe-agent-llama-70b"):
    return {
        "pilot_id": pilot_id,
        "source_id": f"nebius:{instance_id}:{model}",
        "instance_id": instance_id,
        "model_name": model,
        "final_success": "True",
        "trajectory_length": "10",
        "patch_available": "True",
        "eval_log_available": "True",
        "raw_path_or_dataset_index": f"nebius/SWE-agent-trajectories:train:{idx}",
        "selection_reason": "primary_balanced_1_1",
    }


def test_load_targets_returns_pilot_targets_in_csv_order(tmp_path):
    csv_path = tmp_path / "p.csv"
    _write_csv(csv_path, [
        _csv_row("swe_agent_pilot_s_01", "a__a-1", 100),
        _csv_row("swe_agent_pilot_f_01", "b__b-2", 50),
    ])
    ts = load_targets(csv_path)
    assert [t.pilot_id for t in ts] == ["swe_agent_pilot_s_01", "swe_agent_pilot_f_01"]
    assert [t.dataset_index for t in ts] == [100, 50]


def test_load_targets_raises_on_unparseable_index(tmp_path):
    csv_path = tmp_path / "p.csv"
    bad = _csv_row("swe_agent_pilot_s_01", "a__a-1", 0)
    bad["raw_path_or_dataset_index"] = "garbage"
    _write_csv(csv_path, [bad])
    with pytest.raises(ValueError):
        load_targets(csv_path)


# ---------- collect_matches: validation ----------


def test_collect_matches_caches_only_indices_that_validate_instance_id():
    targets = [PilotTarget("swe_agent_pilot_s_01", "expected__id-1", "swe-agent-llama-70b", 1)]
    stream = _stream_at([
        (0, _row("zeroth")),
        (1, _row("WRONG__id-99")),  # mismatch on the wanted index
        (2, _row("third")),
    ])
    matches, errors = collect_matches(stream, targets, max_extra_rows=2)
    assert matches == {}
    assert any("expected__id-1" in e and "WRONG__id-99" in e for e in errors)


def test_collect_matches_caches_only_indices_that_validate_model_name():
    targets = [PilotTarget("swe_agent_pilot_s_01", "a__a-1", "swe-agent-llama-70b", 1)]
    stream = _stream_at([
        (0, _row("zeroth")),
        (1, _row("a__a-1", model_name="swe-agent-llama-8b")),
        (2, _row("after")),
    ])
    matches, errors = collect_matches(stream, targets, max_extra_rows=2)
    assert matches == {}
    assert any("70b" in e and "8b" in e for e in errors)


def test_collect_matches_records_match_when_instance_id_and_model_match():
    targets = [
        PilotTarget("swe_agent_pilot_s_01", "a__a-1", "swe-agent-llama-70b", 1),
        PilotTarget("swe_agent_pilot_s_02", "b__b-2", "swe-agent-llama-70b", 3),
    ]
    stream = _stream_at([
        (0, _row("zeroth")),
        (1, _row("a__a-1")),
        (2, _row("middle")),
        (3, _row("b__b-2")),
        (4, _row("after")),
    ])
    matches, errors = collect_matches(stream, targets, max_extra_rows=10)
    assert errors == []
    assert set(matches.keys()) == {"swe_agent_pilot_s_01", "swe_agent_pilot_s_02"}
    assert matches["swe_agent_pilot_s_01"]["instance_id"] == "a__a-1"
    assert matches["swe_agent_pilot_s_02"]["instance_id"] == "b__b-2"


# ---------- collect_matches: stop conditions ----------


def test_collect_matches_streams_past_max_wanted_by_max_extra_rows_inclusive():
    # max_wanted = 5; max_extra_rows = 0 means cutoff at index 5 inclusive.
    # With max_extra_rows = 2, stream may yield up to index 7 then break.
    targets = [PilotTarget("p1", "a__a-1", "swe-agent-llama-70b", 5)]
    rows = [(i, _row(f"row_{i}")) for i in range(0, 5)]
    rows.append((5, _row("a__a-1")))
    rows.extend([(i, _row(f"row_{i}")) for i in range(6, 20)])
    stream = _stream_at(rows)
    matches, errors = collect_matches(stream, targets, max_extra_rows=2)
    assert errors == []
    assert "p1" in matches


def test_collect_matches_reports_not_found_pilots_when_stream_ends_early():
    targets = [
        PilotTarget("p1", "a__a-1", "swe-agent-llama-70b", 1),
        PilotTarget("p2", "b__b-2", "swe-agent-llama-70b", 1000),  # never streamed
    ]
    stream = _stream_at([(0, _row("zeroth")), (1, _row("a__a-1"))])
    matches, errors = collect_matches(stream, targets, max_extra_rows=10)
    assert "p1" in matches
    assert "p2" not in matches
    assert any("p2" in e and "not matched" in e for e in errors)


def test_collect_matches_raises_on_duplicate_dataset_index():
    targets = [
        PilotTarget("p1", "a__a-1", "m", 5),
        PilotTarget("p2", "b__b-2", "m", 5),
    ]
    with pytest.raises(ValueError):
        collect_matches(iter([]), targets)


def test_collect_matches_returns_empty_when_targets_is_empty():
    matches, errors = collect_matches(_stream(_row("a")), [])
    assert matches == {} and errors == []


# ---------- write_matches ----------


def test_write_matches_writes_one_json_per_pilot_id(tmp_path):
    matches = {
        "swe_agent_pilot_s_01": _row("a__a-1"),
        "swe_agent_pilot_f_01": _row("b__b-2", target=False),
    }
    skipped = []
    written = write_matches(matches, tmp_path, overwrite=False, skipped_existing=skipped)
    assert sorted(written) == ["swe_agent_pilot_f_01", "swe_agent_pilot_s_01"]
    s = json.loads((tmp_path / "swe_agent_pilot_s_01.json").read_text(encoding="utf-8"))
    assert s["instance_id"] == "a__a-1" and s["target"] is True
    f = json.loads((tmp_path / "swe_agent_pilot_f_01.json").read_text(encoding="utf-8"))
    assert f["instance_id"] == "b__b-2" and f["target"] is False
    assert skipped == []


def test_write_matches_default_skips_existing_files(tmp_path):
    (tmp_path / "swe_agent_pilot_s_01.json").write_text('{"existing": true}', encoding="utf-8")
    matches = {"swe_agent_pilot_s_01": _row("a__a-1")}
    skipped = []
    written = write_matches(matches, tmp_path, overwrite=False, skipped_existing=skipped)
    assert written == []
    assert skipped == ["swe_agent_pilot_s_01"]
    # Existing content untouched.
    assert (tmp_path / "swe_agent_pilot_s_01.json").read_text(encoding="utf-8") == '{"existing": true}'


def test_write_matches_overwrite_replaces_existing_files(tmp_path):
    (tmp_path / "swe_agent_pilot_s_01.json").write_text('{"existing": true}', encoding="utf-8")
    matches = {"swe_agent_pilot_s_01": _row("a__a-1")}
    skipped = []
    written = write_matches(matches, tmp_path, overwrite=True, skipped_existing=skipped)
    assert written == ["swe_agent_pilot_s_01"]
    assert skipped == []
    s = json.loads((tmp_path / "swe_agent_pilot_s_01.json").read_text(encoding="utf-8"))
    assert s["instance_id"] == "a__a-1"
