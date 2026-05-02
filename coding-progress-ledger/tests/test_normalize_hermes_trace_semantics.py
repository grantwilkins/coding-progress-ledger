"""
Claim:
normalize_hermes_trace.normalize_row implements six locked decisions
from docs/HERMES_TRACE_SCHEMA.md:
  (1) Multi-tool-call SPLIT — N <tool_call> blocks in one gpt turn
      produce exactly N assistant steps (not 1, not N+1).
  (2) <think> COLLAPSE — contents move into `thought`, tags stripped;
      <think> is NOT emitted as its own step.
  (3) Thought-only retention — a gpt turn with no <tool_call> is still
      emitted as exactly one assistant step with action=None.
  (4) tool_call_id pairing — when ids are present, each tool_call is
      paired with the tool_response carrying its id, regardless of
      response order. When ids are absent, fall back to positional.
  (5) Free-text-thought attachment — text between </think> and the
      first <tool_call> attaches to the FIRST split step's thought
      only; subsequent split steps carry thought=None.
  (6) Top-level final_success is ALWAYS None — Hermes ships no
      success label, and adding a `target` / `resolved` / `success`
      field to the upstream row must not change this.

Plausible wrong implementations:
- Multi-tool-call turn merged into one assistant step (loses ledger
  granularity for stuck-loop / REOPEN timing).
- Splitter emits an extra "wrapper" assistant step around the calls
  (N+1 steps instead of N).
- <think> retained verbatim in `thought` so "<think>" / "</think>"
  bleeds through.
- <think> treated as its own role/step, exploding step counts.
- Thought-only gpt turns silently dropped (zero events for that turn).
- tool_call_id pairing implemented as positional even when ids exist,
  so a B-then-A response order stays B-then-A in the events list
  instead of being re-paired to the A/B call order.
- tool_response with no matching id falls through to nothing,
  dropping observation content.
- Free-text-thought duplicated onto all N split steps, or attached
  to the LAST instead of the FIRST.
- final_success silently mirrored from row['target'] / row['resolved']
  / row['success'] / row['exit_status'], leaking outcome into the
  channel.
- Empty thought rendered as "" instead of None, breaking thought-vs-
  no-thought distinction in downstream features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalize_hermes_trace import normalize_row


# ---------- helpers ----------


def _gpt(value):
    return {"from": "gpt", "value": value}


def _tool(value):
    return {"from": "tool", "value": value}


def _system(value):
    return {"from": "system", "value": value}


def _human(value):
    return {"from": "human", "value": value}


def _assistant_steps(norm):
    return [ev for ev in norm["events"] if ev["role"] == "assistant"]


def _tool_steps(norm):
    return [ev for ev in norm["events"] if ev["role"] == "tool"]


def _multi_call(n, *, with_ids=False):
    """Build a single gpt turn with n tool_calls."""
    parts = ["<think>plan</think>", "Free text response."]
    for i in range(n):
        if with_ids:
            parts.append(
                '<tool_call>\n'
                f'{{"id": "call_{i}", "name": "tool_{i}", "arguments": {{"k": {i}}}}}\n'
                '</tool_call>'
            )
        else:
            parts.append(
                '<tool_call>\n'
                f'{{"name": "tool_{i}", "arguments": {{"k": {i}}}}}\n'
                '</tool_call>'
            )
    return "\n".join(parts)


# ---------- (1) Multi-tool-call SPLIT ----------


def test_split_four_calls_yield_exactly_four_assistant_steps():
    # The schema doc's headline example: a 4-tool-call mega-turn must
    # become 4 steps. Not 1 (merged), not 5 (wrapper added).
    row = {"id": "x", "conversations": [_gpt(_multi_call(4))]}
    norm = normalize_row(row, model_name="kimi")
    asst = _assistant_steps(norm)
    assert len(asst) == 4
    # Each split step carries its own action / tool_name distinct from
    # its neighbors — proves they're real splits, not duplicates.
    assert [ev["tool_name"] for ev in asst] == ["tool_0", "tool_1", "tool_2", "tool_3"]


def test_split_one_call_yields_exactly_one_assistant_step_not_two():
    # Boundary: a single-call turn must not accidentally fire a
    # "wrapper + call" pair. N=1 is the canonical case where N+1
    # impostors are easiest to ship.
    row = {"id": "x", "conversations": [_gpt(_multi_call(1))]}
    norm = normalize_row(row, model_name="kimi")
    asst = _assistant_steps(norm)
    assert len(asst) == 1
    assert asst[0]["tool_name"] == "tool_0"


def test_split_count_metamorphic_n_calls_yield_n_steps():
    # Metamorphic: vary N and assert step count tracks N exactly.
    # Catches off-by-one wrappers and merges in one shot.
    for n in (1, 2, 3, 4, 7):
        row = {"id": "x", "conversations": [_gpt(_multi_call(n))]}
        asst = _assistant_steps(normalize_row(row, model_name="kimi"))
        assert len(asst) == n, f"N={n} produced {len(asst)} steps"


def test_split_decomposition_one_quad_turn_equals_four_singles():
    # Metamorphic decomposition: one turn with 4 calls produces the
    # SAME assistant tool_name sequence as 4 turns with 1 call each.
    # Also catches subtle re-orderings within the split.
    quad = {"id": "x", "conversations": [_gpt(_multi_call(4))]}
    singles = {
        "id": "x",
        "conversations": [_gpt(_multi_call(1).replace("tool_0", f"tool_{i}").replace('"k": 0', f'"k": {i}'))
                          for i in range(4)],
    }
    quad_names = [ev["tool_name"] for ev in _assistant_steps(normalize_row(quad, model_name="kimi"))]
    sing_names = [ev["tool_name"] for ev in _assistant_steps(normalize_row(singles, model_name="kimi"))]
    assert quad_names == sing_names == ["tool_0", "tool_1", "tool_2", "tool_3"]


# ---------- (2) <think> COLLAPSE ----------


def test_think_block_does_not_emit_a_separate_step():
    # The wrong-impl risk: treat <think> as its own role, exploding
    # step counts. A turn with one <think> + one tool_call must yield
    # exactly ONE step, not two.
    value = '<think>reasoning</think>\n<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    row = {"id": "x", "conversations": [_gpt(value)]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert len(asst) == 1


def test_think_tags_stripped_from_thought_content():
    # The literal substrings "<think>" and "</think>" must not appear
    # in the thought field. The reasoning text MUST appear.
    row = {"id": "x", "conversations": [_gpt("<think>inner reason</think>\nVisible.")]}
    ev = _assistant_steps(normalize_row(row, model_name="kimi"))[0]
    assert "<think>" not in (ev["thought"] or "")
    assert "</think>" not in (ev["thought"] or "")
    assert "inner reason" in ev["thought"]


def test_think_block_with_no_other_content_still_yields_one_step():
    # Boundary: a gpt turn that is ONLY <think>...</think> must not
    # produce zero events (which would silently drop visible turns).
    row = {"id": "x", "conversations": [_gpt("<think>only thinking</think>")]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert len(asst) == 1
    assert "only thinking" in (asst[0]["thought"] or "")


def test_multiple_think_blocks_in_one_turn_concatenated_not_dropped():
    # Two <think> blocks in the same turn: a wrong impl might keep
    # only the first or only the last. Both must be visible in
    # thought.
    value = "<think>first part</think>\nbridge\n<think>second part</think>"
    row = {"id": "x", "conversations": [_gpt(value)]}
    ev = _assistant_steps(normalize_row(row, model_name="kimi"))[0]
    assert "first part" in ev["thought"]
    assert "second part" in ev["thought"]


# ---------- (3) Thought-only retention ----------


def test_thought_only_turn_is_retained_as_one_step_with_null_action():
    # A gpt turn with no <tool_call> at all must still appear as
    # exactly one assistant step. Wrong impls drop these turns
    # ("nothing happened") and lose visible work.
    row = {"id": "x", "conversations": [_gpt("Just thinking, no tool call.")]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert len(asst) == 1
    ev = asst[0]
    assert ev["action"] is None
    assert ev["command"] is None
    assert ev["tool_name"] is None
    assert ev["thought"] is not None
    assert "Just thinking" in ev["thought"]


def test_thought_only_turn_in_middle_of_trace_does_not_collapse_neighbors():
    # Sandwich a thought-only gpt turn between two tool-using turns.
    # Wrong impl: drop the thought-only step and merge its neighbors,
    # corrupting step indices downstream.
    row = {
        "id": "x",
        "conversations": [
            _gpt(_multi_call(1)),
            _tool('<tool_response>\n{"tool_call_id": "x", "name": "tool_0", "content": "r1"}\n</tool_response>'),
            _gpt("Pure reflection, no call."),
            _gpt(_multi_call(1).replace("tool_0", "tool_z")),
            _tool('<tool_response>\n{"tool_call_id": "y", "name": "tool_z", "content": "r2"}\n</tool_response>'),
        ],
    }
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert len(asst) == 3
    assert asst[0]["tool_name"] == "tool_0"
    assert asst[1]["tool_name"] is None
    assert asst[1]["thought"] is not None and "reflection" in asst[1]["thought"]
    assert asst[2]["tool_name"] == "tool_z"


# ---------- (4) tool_call_id pairing ----------


def test_pairing_follows_id_when_response_order_is_reversed():
    # The strongest test of id-based pairing: send the responses in
    # the OPPOSITE order from the calls. A positional impl will
    # cross-wire the observations.
    gpt_val = (
        '<tool_call>\n{"id": "call_A", "name": "alpha", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"id": "call_B", "name": "beta", "arguments": {}}\n</tool_call>'
    )
    tool_val = (
        '<tool_response>\n{"tool_call_id": "call_B", "name": "beta", "content": "B-content"}\n</tool_response>\n'
        '<tool_response>\n{"tool_call_id": "call_A", "name": "alpha", "content": "A-content"}\n</tool_response>'
    )
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    norm = normalize_row(row, model_name="kimi")
    tool_evs = _tool_steps(norm)
    # Tool steps appear in CALL order (one per call), each carrying
    # its matched response content.
    assert len(tool_evs) == 2
    assert tool_evs[0]["tool_name"] == "alpha"
    assert tool_evs[0]["observation"] == "A-content"
    assert tool_evs[1]["tool_name"] == "beta"
    assert tool_evs[1]["observation"] == "B-content"


def test_pairing_robust_when_response_id_does_not_match_any_call():
    # Three calls, three responses, but the response ids don't match
    # any call id. Pairing must fall back to positional order, not
    # drop content. Wrong impl: only keep responses whose id matches.
    gpt_val = (
        '<tool_call>\n{"id": "call_A", "name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"id": "call_B", "name": "b", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"id": "call_C", "name": "c", "arguments": {}}\n</tool_call>'
    )
    tool_val = (
        '<tool_response>\n{"tool_call_id": "mismatched_1", "name": "a", "content": "first"}\n</tool_response>\n'
        '<tool_response>\n{"tool_call_id": "mismatched_2", "name": "b", "content": "second"}\n</tool_response>\n'
        '<tool_response>\n{"tool_call_id": "mismatched_3", "name": "c", "content": "third"}\n</tool_response>'
    )
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    tool_evs = _tool_steps(normalize_row(row, model_name="kimi"))
    assert [ev["observation"] for ev in tool_evs] == ["first", "second", "third"]


def test_pairing_partial_id_match_uses_id_for_matched_falls_back_for_rest():
    # Three calls with ids; only middle response has a matching id.
    # The middle pair must lock by id; the remaining two must fall
    # back positionally over the remaining unused responses.
    gpt_val = (
        '<tool_call>\n{"id": "A", "name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"id": "B", "name": "b", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"id": "C", "name": "c", "arguments": {}}\n</tool_call>'
    )
    tool_val = (
        '<tool_response>\n{"tool_call_id": "nope1", "name": "a", "content": "r1"}\n</tool_response>\n'
        '<tool_response>\n{"tool_call_id": "B", "name": "b", "content": "B_correct"}\n</tool_response>\n'
        '<tool_response>\n{"tool_call_id": "nope3", "name": "c", "content": "r3"}\n</tool_response>'
    )
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    tool_evs = _tool_steps(normalize_row(row, model_name="kimi"))
    # Call B must be paired with the id-matched response, not r1 or r3.
    b_step = [ev for ev in tool_evs if ev["tool_name"] == "b"][0]
    assert b_step["observation"] == "B_correct"
    # The remaining two unused responses are r1 and r3; they pair
    # positionally with calls A and C (in that order).
    a_step = [ev for ev in tool_evs if ev["tool_name"] == "a"][0]
    c_step = [ev for ev in tool_evs if ev["tool_name"] == "c"][0]
    assert {a_step["observation"], c_step["observation"]} == {"r1", "r3"}


def test_pairing_no_response_dropped_when_count_matches():
    # Invariant: when N calls and N responses, no response content is
    # silently dropped, regardless of ids.
    gpt_val = "\n".join(
        f'<tool_call>\n{{"id": "id_{i}", "name": "t{i}", "arguments": {{}}}}\n</tool_call>'
        for i in range(4)
    )
    tool_val = "\n".join(
        f'<tool_response>\n{{"tool_call_id": "id_{i}", "name": "t{i}", "content": "obs_{i}"}}\n</tool_response>'
        for i in range(4)
    )
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    tool_evs = _tool_steps(normalize_row(row, model_name="kimi"))
    observed = {ev["observation"] for ev in tool_evs}
    assert observed == {"obs_0", "obs_1", "obs_2", "obs_3"}


# ---------- (5) Free-text thought attachment ----------


def test_free_text_thought_on_first_split_step_only():
    # Free text between </think> and the first <tool_call> attaches
    # to step 0 of the split. Steps 1..N-1 must have thought=None,
    # NOT a duplicate of the free text.
    value = (
        "<think>plan</think>\n"
        "Free-text response that should NOT be duplicated.\n"
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "c", "arguments": {}}\n</tool_call>'
    )
    row = {"id": "x", "conversations": [_gpt(value)]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert len(asst) == 3
    # First split step carries the free text.
    assert asst[0]["thought"] is not None
    assert "Free-text response" in asst[0]["thought"]
    # Subsequent split steps carry None — NOT duplicated, NOT empty
    # string, NOT the free text.
    assert asst[1]["thought"] is None
    assert asst[2]["thought"] is None


def test_free_text_not_attached_to_last_step_when_first_is_correct():
    # Direction test: if free text appeared on the LAST split step
    # instead of the FIRST, this would fail. Catches a swapped-end
    # impl that "looks reasonable" because it still preserves the
    # text somewhere.
    value = (
        "Pre-call commentary.\n"
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    row = {"id": "x", "conversations": [_gpt(value)]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    assert "Pre-call commentary" in (asst[0]["thought"] or "")
    assert "Pre-call commentary" not in (asst[1]["thought"] or "")


def test_free_text_combined_with_think_on_first_step_not_split_across_steps():
    # Both <think> content and free text must surface on step 0;
    # neither should leak onto step 1+.
    value = (
        "<think>plan content</think>\n"
        "Surface commentary.\n"
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    row = {"id": "x", "conversations": [_gpt(value)]}
    asst = _assistant_steps(normalize_row(row, model_name="kimi"))
    t0 = asst[0]["thought"] or ""
    assert "plan content" in t0
    assert "Surface commentary" in t0
    assert asst[1]["thought"] is None


# ---------- (6) final_success ALWAYS None ----------


def test_final_success_none_with_no_label_fields():
    norm = normalize_row({"id": "x", "conversations": []}, model_name="kimi")
    assert norm["final_success"] is None


def test_final_success_none_even_when_target_true_present_in_row():
    # A wrong impl might have copy-pasted the SWE-agent rule of
    # "mirror row['target'] if bool". Hermes must never do that.
    norm = normalize_row(
        {"id": "x", "conversations": [], "target": True},
        model_name="kimi",
    )
    assert norm["final_success"] is None


def test_final_success_none_even_when_target_false_present():
    # Symmetric to the True case — False must not flip final_success
    # to False either.
    norm = normalize_row(
        {"id": "x", "conversations": [], "target": False},
        model_name="kimi",
    )
    assert norm["final_success"] is None


def test_final_success_none_with_resolved_or_success_keys():
    # Other plausible label fields a wrong impl might pick up by name.
    for k in ("resolved", "success", "exit_status", "passed"):
        norm = normalize_row(
            {"id": "x", "conversations": [], k: True},
            model_name="kimi",
        )
        assert norm["final_success"] is None, f"key {k!r} leaked into final_success"


def test_final_success_none_invariant_under_full_trace():
    # Even with a complete-looking trace, no upstream signal should
    # promote final_success above None.
    row = {
        "id": "x",
        "category": "Terminal & Coding",
        "subcategory": "Build",
        "target": True,
        "resolved": True,
        "exit_status": "submitted",
        "conversations": [
            _system("sp"),
            _human("issue"),
            _gpt(_multi_call(2)),
            _tool(
                '<tool_response>\n{"tool_call_id": "x", "name": "tool_0", "content": "ok"}\n</tool_response>\n'
                '<tool_response>\n{"tool_call_id": "y", "name": "tool_1", "content": "ok"}\n</tool_response>'
            ),
        ],
    }
    norm = normalize_row(row, model_name="kimi")
    assert norm["final_success"] is None
    assert norm["exit_status"] is None
