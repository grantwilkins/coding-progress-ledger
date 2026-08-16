import json
from pathlib import Path

import pytest

import service_admission_transition_campaign as campaign
import service_headroom_campaign as headroom


EVIDENCE = Path(__file__).parents[1] / "outputs/service-headroom-a100-20260815"


def source_inputs():
    core = headroom.read_plan(EVIDENCE / "plan.json")
    rates = headroom.read_rates(
        EVIDENCE / "normalization.json", "a100", headroom.digest(core))
    scout = json.loads((EVIDENCE / "scout.json").read_text())
    confirmation = headroom.read_plan(EVIDENCE / "confirmation-plan.json")
    confirmed = json.loads((EVIDENCE / "confirmed.json").read_text())
    return core, rates, scout, confirmation, confirmed


def test_plan_is_nine_discrete_transition_cells():
    plan = campaign.make_plan(*source_inputs())

    assert len(plan["cells"]) == 9
    assert set(row["direction"] for row in plan["cells"]) \
        == set(campaign.DIRECTIONS)
    assert set(row["block"] for row in plan["cells"]) == {6, 7, 8}
    assert plan["target_rho"] == .5
    assert plan["planner_usable"] is False
    assert plan["evidence_status"] == "transition_confirmation_only"
    assert set(plan["recipes"]) == set(campaign.DIRECTIONS)
    assert all(abs(recipe["offered_rho"] - .5) < .005
               for recipe in plan["recipes"].values())
    campaign.validate_plan(plan)


def test_trace_has_incumbents_before_and_new_sessions_only_after_admission():
    core, rates, scout, confirmation, confirmed = source_inputs()
    plan = campaign.make_plan(core, rates, scout, confirmation, confirmed)
    trace = campaign.offered_trace(plan, rates, "balanced")

    assert {row["population"] for row in trace
            if row["offset_s"] < plan["warmup_s"]} == {"incumbent"}
    post = headroom.measurement_rows(plan, trace)
    assert {row["population"] for row in post} == {"incumbent", "balanced"}
    assert abs(headroom.offered_rho(plan, trace) - .5) < .005
    assert all(row["offset_s"] >= plan["warmup_s"]
               for row in trace if row["population"] == "balanced")


def decision_row(plan):
    return {"cell_id": "cell", "direction": "prefill_heavy", "block": 6,
            "admission_error": "", "initial_prewarm_tokens": 8 * 3840,
            "windows": {
                name: {"stable": True, "tpot_reportable": True,
                       "p90_ttft_s": .2, "p90_mean_tpot_s": .04,
                       "cache_mismatch_count": 0}
                for name in ("baseline", "transition", "post_admission")
            },
            "new_cohort": {"offered": 10, "completion_rate": 1,
                           "exact_timing_coverage": 1,
                           "p90_ttft_s": .3, "p90_mean_tpot_s": .05}}


def prewarm_rows(epoch_ns, *, late_s=0):
    return [{"session_id": f"prefill_heavy-{index}", "status": 200,
             "error": "", "done": True, "finish_reason": "length",
             "prompt_tokens": 2048, "output_tokens": 1,
             "planned_output_tokens": 1, "recorded_output_tokens": 1,
             "cached_tokens": 0,
             "start_ns": epoch_ns + int((60 + late_s) * 1e9) + index,
             "end_ns": epoch_ns + int((61 + late_s) * 1e9) + index}
            for index in range(8)]


def test_admission_failure_is_a_valid_failed_decision_not_an_invalid_cell():
    core, rates, scout, confirmation, confirmed = source_inputs()
    plan = campaign.make_plan(core, rates, scout, confirmation, confirmed)
    row = decision_row(plan)
    row["admission_error"] = "RuntimeError: materialization failed"

    decision = campaign.cell_decision(plan, row, [], 1_000_000_000)

    assert not decision["pass"]
    assert not decision["checks"]["admission_materialized"]


def test_new_cohort_slo_and_actual_prewarm_start_are_hard_gates():
    plan = campaign.make_plan(*source_inputs())
    epoch_ns = 1_000_000_000
    row = decision_row(plan)
    healthy = campaign.cell_decision(
        plan, row, prewarm_rows(epoch_ns), epoch_ns)
    assert healthy["pass"]

    row["new_cohort"]["p90_ttft_s"] = 1.01
    slow = campaign.cell_decision(
        plan, row, prewarm_rows(epoch_ns), epoch_ns)
    assert not slow["checks"]["new_cohort_slo"]

    row["new_cohort"]["p90_ttft_s"] = .3
    late = campaign.cell_decision(
        plan, row, prewarm_rows(epoch_ns, late_s=.06), epoch_ns)
    assert not late["checks"]["admission_launch_on_time"]


def test_request_trace_join_rejects_an_altered_recipe():
    core, rates, scout, confirmation, confirmed = source_inputs()
    plan = campaign.make_plan(core, rates, scout, confirmation, confirmed)
    trace = campaign.offered_trace(plan, rates, "decode_heavy")
    epoch_ns = 10_000_000_000
    requests = [{"population": row["population"],
                 "offset_s": row["offset_s"],
                 "session_id": row["session_id"],
                 "request_index": row["request_index"],
                 "input_tokens": row["append_tokens"],
                 "prefix_tokens": row["prefix_tokens"],
                 "planned_output_tokens": row["output_tokens"],
                 "scheduled_ns": epoch_ns + int(row["offset_s"] * 1e9)}
                for row in trace]
    campaign.validate_request_trace(trace, requests, epoch_ns)
    requests[-1]["input_tokens"] += 1
    try:
        campaign.validate_request_trace(trace, requests, epoch_ns)
    except RuntimeError as exc:
        assert "frozen treatment" in str(exc)
    else:
        raise AssertionError("altered request trace was accepted")


def test_completed_cell_checkpoint_is_plan_bound(tmp_path):
    core, rates, scout, confirmation, confirmed = source_inputs()
    plan = campaign.make_plan(core, rates, scout, confirmation, confirmed)
    cell = plan["cells"][0]
    cell_root = tmp_path / cell["cell_id"]
    attempt = cell_root / "attempt-0001"
    attempt.mkdir(parents=True)
    path = attempt / "result.json"
    path.write_text(json.dumps({
        "status": "complete", "plan_sha256": headroom.digest(plan),
        "normalization_sha256": plan["normalization_sha256"], **cell,
    }))

    assert campaign._cell_complete(cell_root, plan, cell)
    assert (cell_root / "selected.json").is_file()
    assert campaign.selected_attempt(cell_root, plan, cell) == attempt
    path.write_text(json.dumps({
        "status": "complete", "plan_sha256": "0" * 64,
        "normalization_sha256": plan["normalization_sha256"], **cell,
    }))
    assert not campaign._cell_complete(cell_root, plan, cell)


def test_invalid_attempt_directory_is_preserved_when_next_attempt_is_selected(tmp_path):
    plan = campaign.make_plan(*source_inputs())
    cell = plan["cells"][0]
    cell_root = tmp_path / cell["cell_id"]
    invalid = cell_root / "attempt-0001"
    complete = cell_root / "attempt-0002"
    invalid.mkdir(parents=True)
    complete.mkdir()
    (invalid / "result.json").write_text('{"status":"invalid"}\n')
    (invalid / "requests.json").write_text('[{"kept":true}]\n')
    result = {"status": "complete", "plan_sha256": headroom.digest(plan),
              "normalization_sha256": plan["normalization_sha256"], **cell}
    (complete / "result.json").write_text(json.dumps(result))

    campaign.select_attempt(cell_root, plan, cell, complete)

    assert campaign.selected_attempt(cell_root, plan, cell) == complete
    assert json.loads((invalid / "requests.json").read_text()) == [{"kept": True}]


def test_driver_status_preserves_preflight_error(tmp_path, monkeypatch):
    plan = campaign.make_plan(*source_inputs())
    rates = source_inputs()[1]
    monkeypatch.setattr(
        campaign, "run_cell",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("runtime identity mismatch")),
    )

    with pytest.raises(RuntimeError, match="runtime identity mismatch"):
        campaign.run_all(
            plan, rates, object(), tmp_path / "runs", tmp_path / "summary.json",
            [], 0, 1,
        )

    status = json.loads((tmp_path / "runs/status.json").read_text())
    assert status["state"] == "failed"
    assert status["last_error"] == "RuntimeError: runtime identity mismatch"
