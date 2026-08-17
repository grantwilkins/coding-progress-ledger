"""Tests for the intentionally small, non-gating agentic RPS sweep."""

import json

import agentic_rps_sweep_campaign as campaign


def test_plan_is_fixed_shape_open_loop_and_runs_every_rate():
    plan = campaign.make_plan(seed=7)

    assert plan["request_shape"] == {
        "prompt_tokens": 3920,
        "output_tokens": 1024,
        "source": "fixed compact shape derived from the OpenHands coding trace",
    }
    assert plan["rates_rps"] == [.125, .25, .5, 1, 2, 4, 8]
    assert plan["requests_per_point"] == 32
    assert plan["semantics"]["open_loop_poisson"]
    assert plan["semantics"]["max_concurrency"] is None
    assert plan["semantics"]["run_all_rates_after_violation"]
    assert not plan["semantics"]["slo_is_control_flow"]
    assert plan["slo"]["fixed"]["google/gemma-4-26B-A4B-it"] == {
        "p90_ttft_s": 2,
        "p90_mean_tpot_s": .2,
    }
    assert plan["slo"]["relative_models"] == ["Qwen/Qwen3.8-27B"]


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
        "p90_mean_tpot_s": tpot,
    }


def write_result(root, row):
    path = root / "cells" / row["cell_id"] / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))


def populate_results(plan, root):
    for model in campaign.MODELS:
        ttfts = (1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0)
        tpots = (.04, .06, .09, .12, .2, .3, .4)
        if model == "openai/gpt-oss-20b":
            ttfts = (1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0)
            tpots = (.04, .06, .08, .12, .2, .3, .4)
        for rate, ttft, tpot in zip(campaign.RATES_RPS, ttfts, tpots):
            write_result(root, synthetic_result(
                plan, model, rate, 0, ttft, tpot,
            ))
        for rate, ttft, tpot in ((.25, 1.5, .06), (.5, 2.6, .091)):
            for repeat, delta in ((1, -.05), (2, .05)):
                write_result(root, synthetic_result(
                    plan, model, rate, repeat, ttft + delta, tpot,
                ))


def test_reduction_repeats_only_observed_boundary_and_never_gates(tmp_path):
    plan = campaign.make_plan(seed=5)
    populate_results(plan, tmp_path)

    summary = campaign.reduce(plan, tmp_path)

    assert not summary["campaign_gate"]
    assert len(summary["rows"]) == 33
    for model, result in summary["models"].items():
        assert result["repeated_boundary_rates"] == [.25, .5]
        assert result["first_confirmed_violation_rps"] == .5
        assert next(row for row in result["curve"]
                    if row["offered_rps"] == .5)["repeats"] == 3
        if model == "openai/gpt-oss-20b":
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_mean_tpot_s"] == .1
        elif model == "google/gemma-4-26B-A4B-it":
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_mean_tpot_s"] == .2
        else:
            assert result["slo"]["p90_ttft_s"] == 2
            assert result["slo"]["p90_mean_tpot_s"] == .08


def test_service_failures_remain_curve_data():
    plan = campaign.make_plan()
    cell = campaign.cell_spec(campaign.MODELS[0], .5, 0)
    rows = [{
        "status": 200, "error": "", "done": True,
        "finish_reason": "length", "output_tokens": 1024,
        "recorded_output_tokens": 1024, "planned_output_tokens": 1024,
        "exact_token_timestamps": True, "ttft_s": 1.0,
        "mean_tpot_s": .05, "scheduled_ns": index * 10,
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
    assert "one client failed" in result["client_error"]
