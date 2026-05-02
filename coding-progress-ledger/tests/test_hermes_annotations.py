"""
Claim:
The Hermes annotation specs under annotations/hermes_pilot/*.json
faithfully describe the corresponding normalized traces in
runs/hermes_pilot/<pilot_id>/. Specifically:

  (a) Every leaf's completion / block evidence cites a step index that
      exists in the normalized trace. An off-by-one or wrong-pilot
      copy-paste must fail this.
  (b) No evidence quotes thought-only content from a `<think>` block
      (Pitfall H2). Evidence must come from action/observation, not
      thought, on the cited step.
  (c) The exported ledger.jsonl's coding-categories progress matches
      what the spec's leaf statuses imply: complete / total over
      CODING_CATEGORIES leaves only.
  (d) Pitfall H1 — multi-tool-call gpt turns must be split. Total
      assistant steps in the normalized trace must equal the sum of
      <tool_call> blocks across gpt turns plus thought-only gpt turns.
      Merging would yield strictly fewer assistant steps.

Plausible wrong annotations:
- Evidence step cites a non-existent index (typo / wrong pilot).
- Evidence string quotes the cited step's `thought` (annotator
  thought-only content) instead of its `action`/`observation`.
- A blocked or in_progress leaf is silently treated as complete in
  ledger.jsonl, inflating coding-progress.
- The normalizer (or a downstream patch) merged a multi-tool turn,
  hiding stuck-loop fidelity.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger_progress import from_jsonl  # noqa: E402
from ledger_progress.queries import CODING_CATEGORIES  # noqa: E402
from ledger_progress.scoring import score  # noqa: E402

ANNOT_DIR = ROOT / "annotations" / "hermes_pilot"
RUNS_DIR = ROOT / "runs" / "hermes_pilot"

PILOT_IDS = sorted(p.stem for p in ANNOT_DIR.glob("hermes_pilot_*.json"))

STEP_RE = re.compile(r"step\s+(\d+)", re.IGNORECASE)


def _load_spec(pid: str) -> dict:
    return json.loads((ANNOT_DIR / f"{pid}.json").read_text())


def _load_trace(pid: str) -> dict:
    return json.loads((RUNS_DIR / pid / "normalized_trace.json").read_text())


def _load_source(pid: str) -> dict:
    return json.loads((RUNS_DIR / pid / "source_trace.json").read_text())


def _load_ledger(pid: str):
    return from_jsonl(str(RUNS_DIR / pid / "ledger.jsonl"))


def _evidence_strings(event: dict) -> list:
    ev = event.get("evidence")
    if ev is None:
        return []
    if isinstance(ev, str):
        return [ev]
    return list(ev)


def _cited_steps(s: str) -> list:
    return [int(m) for m in STEP_RE.findall(s)]


def test_pilot_specs_present():
    assert PILOT_IDS == [f"hermes_pilot_0{i}" for i in range(1, 6)], PILOT_IDS


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_every_evidence_string_cites_existing_step(pid):
    spec = _load_spec(pid)
    trace = _load_trace(pid)
    max_idx = max(e["step_index"] for e in trace["events"])
    for ev in spec["events"]:
        for s in _evidence_strings(ev):
            cited = _cited_steps(s)
            assert cited, f"{pid}: evidence string lacks 'step N' citation: {s!r}"
            for n in cited:
                assert 0 <= n <= max_idx, (
                    f"{pid}: evidence cites step {n} but trace ends at {max_idx}"
                )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_no_evidence_cites_thought_only_step(pid):
    """Pitfall H2: thought-only assistant steps are zero-evidence."""
    spec = _load_spec(pid)
    trace = _load_trace(pid)
    by_idx = {e["step_index"]: e for e in trace["events"]}
    for ev in spec["events"]:
        for s in _evidence_strings(ev):
            for n in _cited_steps(s):
                step = by_idx[n]
                if step["role"] == "assistant" and step.get("action") is None:
                    pytest.fail(
                        f"{pid}: evidence cites step {n} which is a "
                        f"thought-only assistant step (action=None); "
                        f"violates Pitfall H2"
                    )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_evidence_substring_not_drawn_from_thought(pid):
    """Stronger H2 check: when evidence quotes a phrase, that phrase
    must appear in the cited step's action/observation, NOT only in
    its thought field. A copy-paste from `thought` would pass test
    (b) above but fail this one."""
    spec = _load_spec(pid)
    trace = _load_trace(pid)
    by_idx = {e["step_index"]: e for e in trace["events"]}
    for ev in spec["events"]:
        for s in _evidence_strings(ev):
            quoted = re.findall(r"'([^']{8,})'", s)
            quoted += re.findall(r'"([^"]{8,})"', s)
            for n in _cited_steps(s):
                step = by_idx[n]
                action = str(step.get("command") or "") + " " + str(step.get("action") or "")
                obs = str(step.get("observation") or "")
                thought = str(step.get("thought") or "")
                haystack = action + " " + obs
                for q in quoted:
                    if q in thought and q not in haystack:
                        pytest.fail(
                            f"{pid} step {n}: evidence quote {q!r} appears "
                            f"only in `thought`, not in action/observation"
                        )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_coding_progress_matches_spec_leaf_states(pid):
    spec = _load_spec(pid)
    ledger = _load_ledger(pid)
    coding_categories = {c.value for c in CODING_CATEGORIES}

    leaf_status = {}
    leaf_category = {}
    for ev in spec["events"]:
        if ev["op"] == "add":
            leaf_category[ev["id"]] = ev["category"].lower()
            leaf_status[ev["id"]] = "not_started"
        elif ev["op"] == "complete":
            leaf_status[ev["id"]] = "complete"
        elif ev["op"] == "block":
            leaf_status[ev["id"]] = "blocked"
        elif ev["op"] == "start":
            leaf_status[ev["id"]] = "in_progress"
        elif ev["op"] == "invalidate":
            leaf_status[ev["id"]] = "invalidated"

    coding_leaves = [
        sid for sid, cat in leaf_category.items() if cat in coding_categories
    ]
    coding_active = [
        sid for sid in coding_leaves if leaf_status[sid] != "invalidated"
    ]
    coding_complete = [
        sid for sid in coding_active if leaf_status[sid] == "complete"
    ]
    expected = (
        len(coding_complete) / len(coding_active) if coding_active else 0.0
    )

    actual = score(ledger, categories=CODING_CATEGORIES).progress
    assert actual == pytest.approx(expected, abs=1e-9), (
        f"{pid}: ledger coding_progress={actual}, spec implies {expected}"
    )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_multi_tool_call_split_invariant(pid):
    """Pitfall H1: total assistant steps == sum of <tool_call> blocks
    across gpt turns + count of thought-only gpt turns. A merger
    would yield strictly fewer assistant steps."""
    src = _load_source(pid)
    trace = _load_trace(pid)
    expected_assistant_steps = 0
    for c in src["conversations"]:
        if c.get("from") != "gpt":
            continue
        n_calls = len(re.findall(r"<tool_call>", c.get("value", "")))
        expected_assistant_steps += n_calls if n_calls > 0 else 1
    actual_assistant = sum(
        1 for e in trace["events"] if e["role"] == "assistant"
    )
    assert actual_assistant == expected_assistant_steps, (
        f"{pid}: assistant steps={actual_assistant}, expected "
        f"{expected_assistant_steps} (sum of tool_call blocks + "
        f"thought-only gpt turns); H1 split invariant violated"
    )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_every_complete_or_block_event_carries_evidence(pid):
    spec = _load_spec(pid)
    for ev in spec["events"]:
        if ev["op"] in ("complete", "block"):
            evs = _evidence_strings(ev)
            assert evs, f"{pid}: {ev['op']} on {ev['id']} has no evidence"
            for s in evs:
                assert _cited_steps(s), (
                    f"{pid}: evidence on {ev['id']} lacks step citation: {s!r}"
                )


@pytest.mark.parametrize("pid", PILOT_IDS)
def test_add_event_steps_are_monotone_per_id_assignment(pid):
    """The driver requires id S1, S2, ... order matches insertion order;
    if `add` events were reordered without renumbering, the spec would
    silently target the wrong subtask. Catch that here."""
    spec = _load_spec(pid)
    expected_n = 0
    for ev in spec["events"]:
        if ev["op"] == "add":
            expected_n += 1
            assert ev["id"] == f"S{expected_n}", (
                f"{pid}: add event order mismatch: got {ev['id']} expected S{expected_n}"
            )
