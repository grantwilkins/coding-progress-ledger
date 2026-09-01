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


def test_error_bar_plans_freeze_shared_adaptive_protocol():
    h100 = campaign.make_slo_plan()
    a100 = campaign.make_slo_plan(hardware="a100")

    assert h100["schema"] == campaign.SLO_SCHEMA
    assert h100["models"] == [campaign.SLO_MODEL]
    assert h100["slo"] == {
        "p90_ttft_s": 1, "p90_tpot_s": .05,
        "source": "fixed-paper-reference",
    }
    assert h100["preflight"]["candidate_rates_rps"] \
        == list(campaign.SLO_SCOUT_RATES_RPS)
    assert h100["preflight"] == a100["preflight"]
    assert h100["preflight"]["fresh_engine_per_cell"]
    assert h100["preflight"]["required_consecutive_violations"] == 3
    assert h100["selection"] == {
        "refinement_intervals": 8,
        "lower_scout_guards": 1,
        "upper_scout_guards": 2,
        "maximum_clear_boundary_steps": 4,
    }
    assert h100["blocks"]["primary"] == 20
    assert h100["blocks"]["maximum"] == 30
    assert h100["blocks"]["rate_order"] \
        == "ascending_sha256(seed:rate:block:rate-order)"
    assert h100["runtime"]["runtime_versions"] == ["0.22.0", "0.5.1"]
    assert h100["runtime"]["max_num_batched_tokens"] == 8192
    assert h100["runtime"]["block_size"] == 16
    assert not h100["runtime"]["enforce_eager"]
    assert h100["runtime"]["stream_interval"] == 1
    assert h100["warmup"]["rate_rps"] == 1
    assert h100["validity"]["max_metric_gap_s"] == 1
    assert h100["statistics"]["per_look_minimum_confidence"] == .975
    assert {"blocks", "validity", "statistics", "semantics",
            "preflight", "selection", "implementation"} \
        <= h100["comparison"].keys()
    assert h100["comparison_sha256"] == a100["comparison_sha256"]
    assert h100 == {**a100, "hardware": "h100"}


def test_exact_median_intervals_use_predeclared_order_statistics():
    primary = campaign.order_statistic_interval(list(range(20)))
    contingency = campaign.order_statistic_interval(list(range(30)))

    assert primary["median"] == 9.5
    assert (primary["lower"], primary["upper"], primary["rank"]) == (4, 15, 5)
    assert primary["confidence"] == pytest.approx(.9881820679)
    assert (contingency["lower"], contingency["upper"],
            contingency["rank"]) == (8, 21, 9)
    assert contingency["confidence"] == pytest.approx(.983875)


def slo_requests(exact=True):
    return [{
        "status": 200, "error": "", "done": True,
        "finish_reason": "length", "output_tokens": 1024,
        "recorded_output_tokens": 1024, "planned_output_tokens": 1024,
        "prompt_tokens": 3920, "planned_prompt_tokens": 3920,
        "cached_tokens": 0, "exact_token_timestamps": exact,
        "ttft_s": .8, "mean_tpot_s": .01,
        "token_itls_s": [.01] * 1023 if exact else [],
        "scheduled_ns": index * 10**9, "start_ns": index * 10**9,
        "end_ns": index * 10**9 + 500_000_000,
        "send_lateness_s": 0,
    } for index in range(32)]


def test_error_bar_cell_requires_complete_exact_uncached_measurement():
    plan = campaign.make_slo_plan()
    cell = campaign.slo_cell_spec(7, 0)
    metrics = [{"monotonic_ns": index * 10**9,
                "vllm:num_requests_running": 0,
                "vllm:num_requests_waiting": 0} for index in range(33)]
    runtime = {"fingerprint_sha256": "fp",
               "shared_fingerprint_sha256": "shared", "git_sha": "git"}

    numeric = campaign.summarize_slo_cell(
        plan, cell, slo_requests(), metrics, True, None, False, None, None,
        runtime)
    inexact = campaign.summarize_slo_cell(
        plan, cell, slo_requests(False), metrics, True, None, False, None,
        None, runtime)
    incomplete = campaign.summarize_slo_cell(
        plan, cell, slo_requests(), metrics[:1], True, None, False, None,
        None, runtime)
    failed_rows = slo_requests()
    failed_rows[0]["status"] = 500
    failure = campaign.summarize_slo_cell(
        plan, cell, failed_rows, metrics, True, None, False, None, None,
        runtime)

    assert numeric["status"] == "numeric"
    assert numeric["exact_timing"] == 32
    assert numeric["tpot_samples"] == 32 * 1023
    assert numeric["realized_rps"] == 1
    assert inexact["status"] == "invalid"
    assert inexact["validity_errors"] == ["exact_token_timing"]
    assert incomplete["validity_errors"] == ["telemetry"]
    assert failure["status"] == "service_failure"
    assert failure["slo_violation"]
    assert failure["p90_ttft_s"] is None

    crashed = campaign.summarize_slo_cell(
        plan, cell, failed_rows, [], False, None, True,
        RuntimeError("metrics endpoint stopped"), "service_error", runtime)
    infrastructure = campaign.summarize_slo_cell(
        plan, cell, failed_rows, [], False, None, True,
        RuntimeError("metrics endpoint stopped"), "infrastructure", runtime)
    assert crashed["status"] == "service_failure"
    assert crashed["validity_errors"] == []
    assert infrastructure["status"] == "invalid"
    assert infrastructure["validity_errors"] == ["infrastructure"]


def write_slo_scout(plan, root, rate, violation, status="numeric"):
    block = plan["blocks"]["maximum"]
    cell = campaign.slo_cell_spec(rate, block, "preflight")
    attempt = root / "preflight" / "cells" / cell["cell_id"] / "attempt-000"
    stack = root / "preflight" / "stacks" / cell["cell_id"]
    row = {
        "schema": campaign.SLO_SCHEMA,
        "plan_sha256": campaign.digest(plan), **cell,
        "status": status, "runtime_fingerprint_sha256": "fp",
        "shared_runtime_sha256": "shared", "launch_git_sha": "git",
        "slo_violation": violation,
        "evidence_path": str(attempt.relative_to(root / "preflight")),
        "stack_path": str(stack.relative_to(root / "preflight")),
        "p90_ttft_s": (None if status == "service_failure" else
                       1.2 if violation else .8),
        "p90_tpot_s": None if status == "service_failure" else .02,
    }
    campaign.write_json(attempt / "requests.json", [])
    campaign.write_json(attempt / "result.json", row)
    campaign.write_json(stack / "runtime-identity.json", {"test": True})
    campaign.write_json(campaign.slo_result_path(root / "preflight", cell), row)
    return row


def write_slo_preflight(plan, root):
    rows = [write_slo_scout(
        plan, root, rate, rate >= .25,
        "service_failure" if rate >= .5 else "numeric",
    ) for rate in (.03125, .0625, .125, .25, .5, 1)]
    record = campaign.make_slo_preflight_record(
        plan, root / "preflight", rows)
    campaign.write_json(root / "preflight" / "complete.json", record)
    return record


def test_error_bar_reduction_allows_narrow_indeterminate_boundary(tmp_path):
    plan = campaign.make_slo_plan()
    preflight = write_slo_preflight(plan, tmp_path)
    for block in range(20):
        for rate in preflight["formal_rates_rps"]:
            cell = campaign.slo_cell_spec(rate, block)
            ttft = .8 if rate <= .1875 else 1 if rate == .203125 else 1.2
            row = {
                "schema": campaign.SLO_SCHEMA,
                "plan_sha256": campaign.digest(plan), **cell,
                "status": "numeric", "runtime_fingerprint_sha256": "fp",
                "shared_runtime_sha256": "shared", "launch_git_sha": "git",
                "selection_sha256": preflight["selection_sha256"],
                "evidence_path": "formal-evidence",
                "stack_path": "formal-stack",
                "slo_violation": ttft > 1,
                "realized_rps": rate * (1 + (block - 9.5) / 1000),
                "p90_ttft_s": (ttft if rate == .203125 else
                                ttft + block / 10000),
                "p90_tpot_s": .02 + block / 10000,
            }
            campaign.write_json(campaign.slo_result_path(tmp_path, cell), row)

    summary = campaign.reduce_slo(plan, tmp_path, 20)
    model = summary["models"][campaign.SLO_MODEL]
    boundary = next(row for row in model["curve"]
                    if row["offered_rps"] == .21875)

    assert model["last_clear_pass_rps"] == .1875
    assert model["first_clear_violation_rps"] == .21875
    assert model["clear_boundary_confirmed"]
    assert model["clear_boundary_width_rps"] == .03125
    assert model["boundary_within_tolerance"]
    assert model["higher_clear_violation_confirmed"]
    assert model["decision"] == "complete"
    assert boundary["classification"] == "clear_fail"
    assert boundary["p90_ttft_s_ci_rank"] == 5
    assert boundary["p90_ttft_s_ci_low"] == pytest.approx(1.2004)
    assert summary["selection_sha256"] == preflight["selection_sha256"]


def test_boundary_rejects_nonmonotonic_clear_evidence():
    curve = [{"offered_rps": index,
              "classification": classification}
             for index, classification in enumerate((
                 "clear_pass", "clear_fail", "clear_pass", "clear_fail"))]

    assert campaign.consistent_slo_boundary(curve) is None


def test_selected_interior_rate_builds_trace_after_selection_reload(tmp_path):
    plan = campaign.make_slo_plan()
    preflight = write_slo_preflight(plan, tmp_path)
    campaign.freeze_plan(tmp_path, plan)
    loaded = campaign.preflight_slo(plan, tmp_path)
    interior = .140625

    trace = campaign.prepared_trace(
        plan, campaign.SLO_MODEL, interior, 0, "formal",
        tuple(loaded["formal_rates_rps"]),
    )

    assert loaded == preflight
    assert interior in loaded["formal_rates_rps"]
    assert len(trace) == 32


def test_preflight_resume_rejects_tampered_scout_evidence(tmp_path):
    plan = campaign.make_slo_plan()
    write_slo_preflight(plan, tmp_path)
    campaign.freeze_plan(tmp_path, plan)
    first = campaign.slo_cell_spec(.03125, plan["blocks"]["maximum"],
                                   "preflight")
    attempt = tmp_path / "preflight" / "cells" / first["cell_id"] \
        / "attempt-000" / "requests.json"
    attempt.write_text("tampered\n")

    with pytest.raises(RuntimeError, match="stale SLO preflight"):
        campaign.preflight_slo(plan, tmp_path)


def test_fresh_formal_block_accepts_preflight_fingerprint(tmp_path, monkeypatch):
    class LaunchReached(Exception):
        pass

    monkeypatch.setattr(campaign.capacity, "stack_commands", lambda *_: {})
    monkeypatch.setattr(
        campaign, "runtime_identity",
        lambda *_: (_ for _ in ()).throw(LaunchReached),
    )
    with pytest.raises(LaunchReached):
        campaign.run_slo_block(
            campaign.make_slo_plan(), tmp_path, 0, (7,), "formal", "fp",
            "selection")


def test_runtime_fingerprint_covers_commands_and_server_config(monkeypatch):
    plan = campaign.make_slo_plan()
    monkeypatch.setattr(campaign.profiler, "git_state", lambda _: ("git", False))
    monkeypatch.setattr(campaign.testbed, "runtime_mode", lambda: "native")
    monkeypatch.setattr(
        campaign.testbed, "runtime_versions", lambda _: ("0.22.0", "0.5.1"))
    monkeypatch.setattr(
        campaign.capacity, "gpu_snapshot", lambda _: {"name": "H100"})
    cfg = campaign.model_config(campaign.SLO_MODEL, "h100")
    left = campaign.runtime_identity(plan, cfg, {"vllm": ["serve", "a"]})
    right = campaign.runtime_identity(plan, cfg, {"vllm": ["serve", "b"]})
    gpu = campaign.runtime_identity(
        plan, cfg, {"vllm": ["CUDA_VISIBLE_DEVICES=GPU-uuid", "serve", "a"]})
    index = campaign.runtime_identity(
        plan, cfg, {"vllm": ["CUDA_VISIBLE_DEVICES=0", "serve", "a"]})
    pid_a = campaign.runtime_identity(
        plan, cfg, {"vllm": ["mkdir -p /tmp/qh-sink-123", "serve", "a"]})
    pid_b = campaign.runtime_identity(
        plan, cfg, {"vllm": ["mkdir -p /tmp/qh-sink-999", "serve", "a"]})
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ambient = campaign.runtime_identity(plan, cfg, {"vllm": ["serve", "a"]})

    assert left["shared_fingerprint_sha256"] != \
        right["shared_fingerprint_sha256"]
    assert gpu["shared_fingerprint_sha256"] == index["shared_fingerprint_sha256"]
    assert pid_a["shared_fingerprint_sha256"] == pid_b["shared_fingerprint_sha256"]
    assert ambient["shared_fingerprint_sha256"] != left[
        "shared_fingerprint_sha256"]
    assert campaign.finalize_runtime_identity(
        left, {"vllm_config": {"block_size": 16}})["fingerprint_sha256"] != \
        campaign.finalize_runtime_identity(
            left, {"vllm_config": {"block_size": 32}})["fingerprint_sha256"]
    assert campaign.finalize_runtime_identity(
        left, {"vllm_config": {"cache_dir": "/tmp/qh-sink-123"}}
    )["fingerprint_sha256"] == campaign.finalize_runtime_identity(
        left, {"vllm_config": {"cache_dir": "/tmp/qh-sink-999"}}
    )["fingerprint_sha256"]
    assert campaign.finalize_runtime_identity(
        left, {"vllm_config": {"instance_id": "launch-a", "block_size": 16}}
    )["fingerprint_sha256"] == campaign.finalize_runtime_identity(
        left, {"vllm_config": {"instance_id": "launch-b", "block_size": 16}}
    )["fingerprint_sha256"]


def test_thirty_blocks_require_unresolved_primary_look(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "reduce_slo", lambda *_: {
        "models": {campaign.SLO_MODEL: {"decision": "complete"}},
    })

    with pytest.raises(RuntimeError, match="unresolved 20-block"):
        campaign.run_slo_campaign(campaign.make_slo_plan(), tmp_path, 30)


def test_preflight_accepts_a_valid_high_rate_service_failure(
        tmp_path, monkeypatch):
    plan = campaign.make_slo_plan()
    calls = []

    def run(plan, root, block, rates, stage, expected_fingerprint=None,
            selection_sha256=None):
        assert len(rates) == 1
        calls.append(rates[0])
        for rate in rates:
            cell = campaign.slo_cell_spec(rate, block, stage)
            attempt = root / "cells" / cell["cell_id"] / "attempt-000"
            stack = root / "stacks" / cell["cell_id"]
            failed = rate >= .25
            row = {
                "schema": campaign.SLO_SCHEMA, "plan_sha256": campaign.digest(plan),
                **cell, "status": "service_failure" if failed else "numeric",
                "runtime_fingerprint_sha256": "fp",
                "shared_runtime_sha256": "shared", "launch_git_sha": "git",
                "slo_violation": failed,
                "evidence_path": str(attempt.relative_to(root)),
                "stack_path": str(stack.relative_to(root)),
                "p90_ttft_s": None if failed else .8,
                "p90_tpot_s": None if failed else .02,
            }
            campaign.write_json(attempt / "requests.json", [])
            campaign.write_json(attempt / "result.json", row)
            campaign.write_json(stack / "runtime-identity.json", {"test": True})
            campaign.write_json(campaign.slo_result_path(root, cell), row)
        return "fp"

    monkeypatch.setattr(campaign, "run_slo_block", run)

    result = campaign.preflight_slo(plan, tmp_path)

    assert result["status"] == "complete"
    assert result["observed_rates_rps"] \
        == [.03125, .0625, .125, .25, .5, 1]
    assert result["bracket"]["observed_pass_rps"] == .125
    assert result["bracket"]["observed_violation_rps"] == .25
    assert len(result["block_orders"]) == 30
    assert calls == result["observed_rates_rps"]


def test_preflight_rejects_a_higher_numeric_pass(tmp_path, monkeypatch):
    plan = campaign.make_slo_plan()

    def run(plan, root, block, rates, stage, expected_fingerprint=None,
            selection_sha256=None):
        rate = rates[0]
        cell = campaign.slo_cell_spec(rate, block, stage)
        attempt = root / "cells" / cell["cell_id"] / "attempt-000"
        stack = root / "stacks" / cell["cell_id"]
        violation = rate == .25
        row = {
            "schema": campaign.SLO_SCHEMA, "plan_sha256": campaign.digest(plan),
            **cell, "status": "numeric", "runtime_fingerprint_sha256": "fp",
            "shared_runtime_sha256": "shared", "launch_git_sha": "git",
            "slo_violation": violation,
            "evidence_path": str(attempt.relative_to(root)),
            "stack_path": str(stack.relative_to(root)),
            "p90_ttft_s": 1.2 if violation else .8,
            "p90_tpot_s": .02,
        }
        campaign.write_json(attempt / "result.json", row)
        campaign.write_json(stack / "runtime-identity.json", {"test": True})
        campaign.write_json(campaign.slo_result_path(root, cell), row)
        return "fp"

    monkeypatch.setattr(campaign, "run_slo_block", run)

    with pytest.raises(RuntimeError, match="not monotone"):
        campaign.preflight_slo(plan, tmp_path)
