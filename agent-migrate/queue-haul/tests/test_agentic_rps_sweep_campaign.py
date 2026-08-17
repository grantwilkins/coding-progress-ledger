"""Tests for the intentionally small, non-gating agentic RPS sweep."""

import csv
import json

import pytest

import agentic_rps_sweep_campaign as campaign


def test_plan_is_fixed_shape_open_loop_and_runs_every_rate():
    plan = campaign.make_plan(seed=7)

    assert plan["request_shape"] == {
        "prompt_tokens": 3920,
        "output_tokens": 1024,
        "source": "fixed compact shape derived from the OpenHands coding trace",
    }
    assert plan["rates_rps"] == [
        .125, .25, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8,
    ]
    assert plan["rates_rps_by_model"]["openai/gpt-oss-20b"] == [
        .125, .25, .5, 1, 2, 3, 4, 5, 6, 7, 8,
    ]
    assert plan["rates_rps_by_model"]["Qwen/Qwen3.8-27B"] == [
        .125, .25, .5, .6, .7, .8, .9, 1, 2, 4, 8,
    ]
    assert plan["requests_per_point"] == 32
    assert plan["semantics"]["open_loop_poisson"]
    assert plan["semantics"]["max_concurrency"] is None
    assert plan["semantics"]["run_all_rates_after_violation"]
    assert not plan["semantics"]["slo_is_control_flow"]
    assert plan["semantics"]["refinement_points_predeclared"]
    assert plan["slo"]["fixed"]["google/gemma-4-26B-A4B-it"] == {
        "p90_ttft_s": 2,
        "p90_tpot_s": .2,
    }
    assert plan["semantics"]["tpot_definition"] \
        == "p90_of_all_exact_post_first_token_intervals"
    assert plan["slo"]["relative_models"] == ["Qwen/Qwen3.8-27B"]


def test_h100_plan_changes_only_hardware():
    a100 = campaign.make_plan(seed=7)
    h100 = campaign.make_plan(seed=7, hardware="h100")

    assert h100 == {**a100, "hardware": "h100",
                    "runtime": {**a100["runtime"], "enforce_eager": False}}
    assert campaign.model_config("openai/gpt-oss-20b").enforce_eager
    assert not campaign.model_config(
        "openai/gpt-oss-20b", "h100").enforce_eager


def test_trace_forces_long_output_and_unique_private_prompts():
    plan = campaign.make_plan(seed=3)
    model = "Qwen/Qwen3.8-27B"
    trace = campaign.prepared_trace(plan, model, .25, 0)
    bodies = [json.loads(row["prepared"]["body"]) for row in trace]

    assert len(trace) == 32
    assert all(body["max_tokens"] == 1024 for body in bodies)
    assert all(body["ignore_eos"] and body["temperature"] == 0
               for body in bodies)
    assert all(len(body["prompt"]) == 3920 for body in bodies)
    assert len({row["prepared"]["prompt_sha256"] for row in trace}) == 32
    assert all(body["kv_transfer_params"]["qh_bypass_lmcache"]
               for body in bodies)
    assert [row["offset_s"] for row in trace] == sorted(
        row["offset_s"] for row in trace)


def synthetic_result(plan, model, rate, repeat, ttft, tpot):
    return {
        "schema": campaign.SCHEMA,
        "plan_sha256": campaign.digest(plan),
        **campaign.cell_spec(model, rate, repeat),
        "status": "recorded",
        "offered": 32,
        "completed": 32,
        "failed": 0,
        "exact_timing": 32,
        "p90_ttft_s": ttft,
        "p90_tpot_s": tpot,
    }


def write_result(root, row):
    path = root / "cells" / row["cell_id"] / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))


def populate_results(plan, root):
    for model in campaign.MODELS:
        violation = .8 if model == "Qwen/Qwen3.8-27B" else 6
        rates = campaign.model_rates(plan, model)
        predecessor = rates[rates.index(violation) - 1]
        for rate in rates:
            ttft = 1.0 if rate < violation else 3.0 + rate / 100
            tpot = .04
            write_result(root, synthetic_result(
                plan, model, rate, 0, ttft, tpot,
            ))
        for rate in (predecessor, violation):
            for repeat, delta in ((1, -.05), (2, .05)):
                write_result(root, synthetic_result(
                    plan, model, rate, repeat,
                    (1.0 if rate < violation else 3.0 + rate / 100)
                    + delta,
                    .04,
                ))


def test_reduction_repeats_only_observed_boundary_and_never_gates(tmp_path):
    plan = campaign.make_plan(seed=5)
    populate_results(plan, tmp_path)

    summary = campaign.reduce(plan, tmp_path)

    assert not summary["campaign_gate"]
    assert len(summary["rows"]) == 45
    for model, result in summary["models"].items():
        violation = .8 if model == "Qwen/Qwen3.8-27B" else 6
        predecessor = .7 if model == "Qwen/Qwen3.8-27B" else 5
        assert result["repeated_boundary_rates"] == [predecessor, violation]
        assert result["first_confirmed_violation_rps"] == violation
        assert next(row for row in result["curve"]
                    if row["offered_rps"] == violation)["repeats"] == 3
        if model == "openai/gpt-oss-20b":
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_tpot_s"] == .1
        elif model == "google/gemma-4-26B-A4B-it":
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_tpot_s"] == .2
        else:
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_tpot_s"] == .08


def test_old_mean_tpot_cells_are_not_reused(tmp_path):
    plan = campaign.make_plan(seed=1)
    assert plan["parent"]["plan_sha256"] \
        == campaign.PARENT_PLAN_SHA256
    assert plan["parent"]["reusable_rates_rps"] == []
    model = "openai/gpt-oss-20b"
    cell = campaign.cell_spec(model, 4, 0)
    row = synthetic_result(plan, model, 4, 0, 1.0, .04)
    row.update({"schema": campaign.PARENT_SCHEMA,
                "plan_sha256": campaign.PARENT_PLAN_SHA256})
    write_result(tmp_path, row)

    with pytest.raises(RuntimeError, match="stale or invalid"):
        campaign.read_result(
            plan, cell, campaign.result_path(tmp_path, cell))


def test_historical_cells_are_rereduced_from_all_token_intervals(tmp_path):
    plan = campaign.make_plan(seed=1)
    model = "openai/gpt-oss-20b"
    cell = campaign.cell_spec(model, 4, 0)
    source = tmp_path / "old" / "cells" / cell["cell_id"]
    old = synthetic_result(plan, model, 4, 0, 1.0, .04)
    old.update({
        "schema": "queue-haul-agentic-rps-sweep-v1",
        "plan_sha256": next(
            digest for schema, digest in campaign.HISTORICAL_RESULT_IDENTITIES
            if schema == "queue-haul-agentic-rps-sweep-v1"
        ),
        "peak_running_requests": 7,
    })
    write_result(tmp_path / "old", old)
    requests = [{
        "status": 200, "error": "", "done": True,
        "finish_reason": "length", "output_tokens": 1024,
        "recorded_output_tokens": 1024, "planned_output_tokens": 1024,
        "exact_token_timestamps": True, "ttft_s": 1.0,
        "mean_tpot_s": .05, "token_itls_s": [.01, .02, .03, .20],
        "scheduled_ns": index * 10, "start_ns": index * 10,
        "send_lateness_s": 0,
    } for index in range(3)]
    (source / "requests.json").write_text(json.dumps(requests))

    rows = campaign.rereduce_sources(
        plan, [tmp_path / "old" / "cells"], tmp_path / "new", (model,),
        ["test-node"], ["/persistent/test/cells"],
    )

    assert len(rows) == 1
    assert rows[0]["schema"] == campaign.SCHEMA
    assert rows[0]["p90_tpot_s"] == pytest.approx(.20)
    assert rows[0]["tpot_samples"] == 12
    assert rows[0]["peak_running_requests"] == 7
    assert rows[0]["source_label"] == "test-node"
    assert rows[0]["source_root"] == "/persistent/test/cells"
    assert rows[0]["source_schema"] \
        == "queue-haul-agentic-rps-sweep-v1"


def test_service_failures_remain_curve_data():
    plan = campaign.make_plan()
    cell = campaign.cell_spec(campaign.MODELS[0], .5, 0)
    rows = [{
        "status": 200, "error": "", "done": True,
        "finish_reason": "length", "output_tokens": 1024,
        "recorded_output_tokens": 1024, "planned_output_tokens": 1024,
        "exact_token_timestamps": True, "ttft_s": 1.0,
        "mean_tpot_s": .05, "token_itls_s": [.01, .02, .03, .20],
        "scheduled_ns": index * 10,
        "start_ns": index * 10, "send_lateness_s": 0,
    } for index in range(3)]

    result = campaign.summarize_cell(
        plan, cell, rows, [], False, RuntimeError("one client failed"),
        False, None,
    )

    assert result["status"] == "recorded"
    assert result["completed"] == 3
    assert result["failed"] == 29
    assert result["p90_ttft_s"] == 1
    assert result["p90_tpot_s"] == pytest.approx(.20)
    assert result["tpot_samples"] == 12
    assert result["diagnostic_p90_request_mean_tpot_s"] == .05
    assert "one client failed" in result["client_error"]


def test_csv_exports_pooled_tpot_not_request_mean(tmp_path):
    path = tmp_path / "rps-sweep.csv"
    campaign.write_csv(path, [{
        "p90_tpot_s": .2,
        "diagnostic_p90_request_mean_tpot_s": .05,
    }])

    row = next(csv.DictReader(path.open()))
    assert row == {"p90_tpot_s": "0.2"}
