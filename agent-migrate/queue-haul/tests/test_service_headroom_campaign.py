"""
Claim:
The service-headroom campaign holds incumbent traffic and KV stock fixed while
varying only scheduled Queue-Haul work in the planner's normalized service unit.

Plausible wrong implementations:
- Measure migrated requests instead of incumbent degradation.
- Change the incumbent trace between load points.
- Normalize both phases with one rate or with completed throughput.
- Use phase directions whose normalized work shares are nearly identical.
- Drop failed offered requests or report an under-sampled P99.
- Accept one lucky restart as the safe boundary.
- Let warmup or post-window drain hide growing active-decode occupancy.
- Accept APC under-hit or append-hot work as the planned normalized load.
- Treat unobservable token gaps as failed service completions.
- Resume a complete cell after its plan, runtime, or normalization changed.
- Rerun an OOM/service exit or accept an Xid/cache mismatch as service evidence.
- Lose successful request rows when one response parser fails.
- Publish a discovery bracket as a confirmed planner constraint.
- Conflate marginal latency quantiles with joint request attainment.
- Reuse discovery blocks as held-out confirmation or omit the balanced check.
- Infer the expected scrape period from a uniformly sparse telemetry trace.
- Confirm a bracket with a different normalization than discovery used.
- Apply an edited confirmation result without its core plan and scout evidence.
- Repeat a direction-free baseline and give the two curves different origins.
- Mix exact-stack identities while reducing a hardware normalizer.
- Ignore an environment-dependent vLLM or cache command in runtime identity.
- Accept a physically executed cell order different from the frozen randomization.
- Rehash the multi-gigabyte serving image in every physical cell.
"""

from __future__ import annotations

import json
import math
import copy
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import service_headroom_campaign as campaign


RATES = {"prefill_tps": 4000, "decode_tps": 1000,
         "kv_capacity_tokens": 1_000_000,
         "balanced_shape": {"prefix_tokens": 3584, "append_tokens": 512,
                            "output_tokens": 128},
         "planned_parked_prefix_tokens": 77_312}
IDENTITY = {"serving_class": "exact"}
IDENTITY_SHA = campaign.digest(IDENTITY)


def evidence(plan: dict, cell: dict, **values) -> dict:
    order = plan["run_order"] if isinstance(plan["run_order"], list) else \
        plan["run_order"][cell["hardware"]][
            "calibration" if cell["kind"] == "calibration" else "measurement"]
    return {"schema": plan["schema"], "plan_sha256": campaign.digest(plan),
            "runtime_identity": IDENTITY,
            "runtime_identity_sha256": IDENTITY_SHA,
            "normalization_sha256": "normalization", "status": "complete",
            "preloaded_kv_usage": .1, "initial_kv_usage": .2,
            "kv_capacity_tokens": 1_000_000,
            "planned_parked_prefix_tokens": 77_312,
            "started_wall_ns": order.index(cell["cell_id"]) + 1,
            "incumbent_exact": 1,
            "incumbent_exact_completion_rate": 1,
            "all_offered_exact_completion_rate": 1,
            **cell, **values}


def complete(row: dict) -> dict:
    return {**row, "status": 200, "error": "", "done": True,
            "finish_reason": "length", "output_tokens": 128,
            "planned_output_tokens": 128, "recorded_output_tokens": 128,
            "exact_token_timestamps": True, "ttft_s": .2,
            "mean_tpot_s": .02, "token_itls_s": [.02],
            "send_lateness_s": 0, "prefix_tokens": 3840,
            "cached_tokens": 3840}


def test_plan_has_three_restart_blocks_for_each_hardware_direction_and_load():
    plan = campaign.make_plan()
    cells = [row for row in plan["cells"] if row["kind"] == "headroom"]
    controls = [row for row in plan["cells"] if row["kind"] == "residency_control"]

    assert len(cells) == 2 * (1 + 2 * 5) * 3
    assert len(controls) == 2 * 3 and not any(row["resident_state"] for row in controls)
    assert {row["target_rho"] for row in cells} == set(campaign.LOADS)
    assert len([row for row in cells if row["direction"] == "baseline"]) == 2 * 3
    ordered = [cell for hardware in campaign.HARDWARE
               for kind in ("calibration", "measurement")
               for cell in plan["run_order"][hardware][kind]]
    assert set(ordered) == {cell["cell_id"] for cell in plan["cells"]}
    campaign.validate_plan(json.loads(json.dumps(plan)))


def test_incumbent_trace_is_paired_and_offered_work_is_context_normalized():
    plan = campaign.make_plan()
    baseline = campaign.offered_trace(plan, RATES, "prefill_heavy", .25, 0)
    loaded = campaign.offered_trace(plan, RATES, "decode_heavy", .95, 0)
    def incumbent(rows):
        return [(r["offset_s"], r["session_id"], r["request_index"])
                for r in rows if r["population"] == "incumbent"]

    assert incumbent(baseline) == incumbent(loaded)
    assert campaign.offered_rho(plan, baseline) == pytest.approx(.25, rel=.01)
    assert campaign.offered_rho(plan, loaded) == pytest.approx(.95, rel=.01)
    assert campaign.phase_share(campaign.SHAPES["prefill_heavy"], RATES) > .9
    assert campaign.phase_share(campaign.SHAPES["decode_heavy"], RATES) < .05
    campaign.validate_rates(RATES)


def test_summary_uses_incumbents_and_keeps_failures_in_denominator():
    plan = campaign.make_plan()
    offered = campaign.offered_trace(plan, RATES, "prefill_heavy", .25, 0)
    selected = campaign.measurement_rows(plan, offered)[:4]
    epoch = 10**12
    requests = []
    for index, row in enumerate(selected):
        request = complete({"scheduled_ns": epoch + int(row["offset_s"] * 1e9),
                            "offset_s": row["offset_s"], "population": "incumbent"})
        if index == 3:
            request["error"] = "timeout"
        requests.append(request)
    metrics = [{"monotonic_ns": epoch + i * 10**9,
                "vllm:num_requests_waiting": 0,
                "vllm:num_requests_running": 0,
                "vllm:gpu_cache_usage_perc": .1} for i in range(100)]
    offered = selected

    result = campaign.summarize(plan, offered, requests, metrics, True)

    assert result["incumbent_exact_completion_rate"] == .75
    assert result["incumbent_service_failure_rate"] == .25
    assert result["all_offered_service_failure_rate"] == .25
    assert result["p90_ttft_s"] == pytest.approx(.2)
    assert result["p99_ttft_s"] is None and not result["p99_reportable"]
    assert not result["stable"]


def test_phase_directions_must_remain_distinct():
    with pytest.raises(ValueError, match="separate"):
        campaign.validate_rates({"prefill_tps": 1000, "decode_tps": 1})
    h100 = {"prefill_tps": 11_400, "decode_tps": 451,
            "kv_capacity_tokens": 1_000_000}
    campaign.validate_rates(h100)
    assert .4 <= campaign.phase_share(campaign.balanced_shape(h100), h100) <= .6


def test_runtime_contract_changes_with_every_semantic_stack_input(monkeypatch):
    monkeypatch.setenv("QH_RUNTIME", "apptainer")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    plan = campaign.make_plan()
    cfg = SimpleNamespace(model=plan["model"], max_model_len=32768,
                          max_num_seqs=256, max_num_batched_tokens=8192)
    gpu = {"name": "NVIDIA A100 80GB PCIe", "uuid": "GPU-x",
           "driver_version": "1", "memory_total_mib": 81920,
           "power_limit_w": 300, "graphics_clock_mhz": 1410,
           "memory_clock_mhz": 1593}
    commands = {"vllm": ["vllm", "serve"], "cache": ["lmcache", "server"],
                "redis": ["redis-server"]}
    first = campaign.runtime_contract(plan, cfg, [], gpu, ("0.22.0+cu129", "0.5.1"),
                                      plan["image_sha256"], "commit", commands)
    changed = SimpleNamespace(**{**vars(cfg), "max_num_batched_tokens": 4096})

    with pytest.raises(RuntimeError, match="serving stack"):
        campaign.runtime_contract(plan, changed, [], gpu,
                                  ("0.22.0+cu129", "0.5.1"),
                                  plan["image_sha256"], "commit", commands)
    with pytest.raises(RuntimeError, match="extra vLLM"):
        campaign.runtime_contract(plan, cfg, ["--foo"], gpu,
                                  ("0.22.0+cu129", "0.5.1"),
                                  plan["image_sha256"], "commit", commands)
    changed_command = campaign.runtime_contract(
        plan, cfg, [], gpu, ("0.22.0+cu129", "0.5.1"), plan["image_sha256"],
        "commit", {**commands, "vllm": ["vllm", "serve", "--changed"]},
    )
    assert first["sha256"] == campaign.digest({key: value for key, value in first.items()
                                                if key != "sha256"})
    assert changed_command["sha256"] != first["sha256"]
    pid_command = campaign.runtime_contract(
        plan, cfg, [], gpu, ("0.22.0+cu129", "0.5.1"), plan["image_sha256"],
        "commit", {**commands, "vllm": ["mkdir", "/tmp/qh-sink-123"]},
    )
    next_pid = campaign.runtime_contract(
        plan, cfg, [], gpu, ("0.22.0+cu129", "0.5.1"), plan["image_sha256"],
        "commit", {**commands, "vllm": ["mkdir", "/tmp/qh-sink-456"]},
    )
    assert pid_command["sha256"] == next_pid["sha256"]

    monkeypatch.setenv("QH_RUNTIME", "native")
    native = campaign.runtime_contract(
        plan, cfg, [], gpu, ("0.22.0", "0.5.1"), None, "commit", commands,
    )
    assert native["runtime_mode"] == "native"
    assert native["image_sha256"] is None
    assert native["sha256"] != first["sha256"]
    with pytest.raises(RuntimeError, match="serving stack"):
        campaign.runtime_contract(
            plan, cfg, [], gpu, ("0.22.0", "0.5.1"), plan["image_sha256"],
            "commit", commands,
        )


def test_image_hash_cache_reuses_only_an_unchanged_file(tmp_path, monkeypatch):
    image, cache = tmp_path / "stack.sif", tmp_path / "image-hash.json"
    image.write_bytes(b"a")
    calls = []
    monkeypatch.setattr(campaign.profiler, "file_hash", lambda _path: (
        calls.append(1) or str(len(calls)) * 64))

    assert campaign.cached_image_hash(image, cache) == "1" * 64
    assert campaign.cached_image_hash(image, cache) == "1" * 64
    image.write_bytes(b"changed")
    assert campaign.cached_image_hash(image, cache) == "2" * 64
    assert len(calls) == 2


def test_resume_requires_the_same_plan_runtime_and_normalization():
    plan = campaign.make_plan()
    cell = next(row for row in plan["cells"] if row["kind"] == "headroom")
    result = evidence(plan, cell)

    campaign.validate_resume(result, plan, cell, IDENTITY, "normalization")
    with pytest.raises(RuntimeError, match="resume"):
        campaign.validate_resume({**result, "normalization_sha256": "stale"},
                                 plan, cell, IDENTITY, "normalization")


def test_normalization_is_bound_to_its_discovery_plan(tmp_path):
    plan = campaign.make_plan()
    rates = {"schema": campaign.SCHEMA, "hardware": "a100",
             "context_tokens": campaign.CONTEXT,
             "plan_sha256": campaign.digest(plan),
             "runtime_identity": IDENTITY,
             "runtime_identity_sha256": IDENTITY_SHA,
             "prefill_tps": 4000, "decode_tps": 1000,
             "kv_capacity_tokens": 1_000_000,
             "balanced_shape": {"prefix_tokens": 3584, "append_tokens": 512,
                                "output_tokens": 128},
             "planned_parked_prefix_tokens": 77_312}
    rates["sha256"] = campaign.digest(rates)
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(rates))

    assert campaign.read_rates(path, "a100", campaign.digest(plan))["sha256"]
    with pytest.raises(ValueError, match="normalization"):
        campaign.read_rates(path, "a100", "different-plan")


def test_service_failure_with_truncated_queue_telemetry_remains_in_denominator():
    plan = campaign.make_plan()
    offered = campaign.measurement_rows(
        plan, campaign.offered_trace(plan, RATES, "prefill_heavy", .25, 0),
    )[:1]
    epoch = 10**12
    request = complete({"scheduled_ns": epoch + int(offered[0]["offset_s"] * 1e9),
                        "offset_s": offered[0]["offset_s"],
                        "population": "incumbent"})
    request["error"] = "engine exited"

    result = campaign.summarize(plan, offered, [request], [{
        "monotonic_ns": epoch, "vllm:num_requests_waiting": 0,
        "vllm:num_requests_running": 0,
        "vllm:gpu_cache_usage_perc": .1,
    }], False)

    assert result["incumbent_exact_completion_rate"] == 0
    assert math.isinf(result["queue_drift_upper_requests_per_s"])
    assert not result["stable"]


def test_summary_uses_measurement_window_and_active_decode_for_stability():
    plan = campaign.make_plan()
    offered = campaign.measurement_rows(
        plan, campaign.offered_trace(plan, RATES, "decode_heavy", .25, 0),
    )[:4]
    epoch = 10**12
    requests = [complete({"scheduled_ns": epoch + int(row["offset_s"] * 1e9),
                          "start_ns": epoch + int(row["offset_s"] * 1e9),
                          "offset_s": row["offset_s"], "population": "incumbent"})
                for row in offered]
    metrics = []
    for second in range(0, 361, 10):
        running = max(0, (second - 60) // 10) if second < 300 \
            else max(0, (360 - second) // 10)
        metrics.append({"monotonic_ns": epoch + second * 10**9,
                        "vllm:num_requests_waiting": 0,
                        "vllm:num_requests_running": running,
                        "vllm:gpu_cache_usage_perc": .2 if second < 300 else .9})

    result = campaign.summarize(plan, offered, requests, metrics, True)

    assert result["initial_in_system_requests"] == 0
    assert result["maximum_kv_usage"] == .2
    assert not result["telemetry_window_complete"]
    assert result["queue_drift_upper_requests_per_s"] > 0
    assert not result["stable"]


def test_cache_contract_requires_only_the_block_rounded_private_prefix():
    valid = complete({})
    under_hit = {**valid, "cached_tokens": 3824}
    append_hot = {**valid, "cached_tokens": 4096}

    assert campaign.cache_mismatches([valid]) == []
    assert campaign.cache_mismatches([under_hit]) == [under_hit]
    assert campaign.cache_mismatches([append_hot]) == [append_hot]


def test_prewarm_proves_private_uncached_prefixes():
    sessions = [SimpleNamespace(session_id="a", prefix_tokens=3840),
                SimpleNamespace(session_id="b", prefix_tokens=2048)]
    rows = [complete({"prompt_tokens": 3840, "output_tokens": 1,
                      "planned_output_tokens": 1, "recorded_output_tokens": 1,
                      "cached_tokens": 0}),
            complete({"prompt_tokens": 2048, "output_tokens": 1,
                      "planned_output_tokens": 1, "recorded_output_tokens": 1,
                      "cached_tokens": 16})]

    with pytest.raises(RuntimeError, match="private-prefix"):
        campaign.validate_prewarm(rows, sessions)


def test_engine_exit_classification_keeps_oom_but_rejects_xid(tmp_path):
    log = tmp_path / "sink.log"
    log.write_text("CUDA out of memory")
    assert campaign.engine_failure_kind(log, True) == "service"
    log.write_text("NVRM: Xid 79, GPU has fallen off the bus")
    assert campaign.engine_failure_kind(log, True) == "infrastructure"
    assert campaign.engine_failure_kind(log, False) is None


def test_measurement_invalidity_is_separate_from_service_failure():
    plan = campaign.make_plan()
    trace = [{"offset_s": 1}]
    requests = [{"send_lateness_s": 0}]
    valid = {"cache_mismatch_count": 0, "tpot_reportable": True,
             "telemetry_window_complete": True}

    assert campaign.invalid_reason(plan, trace, requests, valid, "service") is None
    assert "infrastructure" in campaign.invalid_reason(
        plan, trace, requests, valid, "infrastructure",
    )
    assert "cache" in campaign.invalid_reason(
        plan, trace, requests, {**valid, "cache_mismatch_count": 1}, None,
    )


def test_future_collection_keeps_rows_beside_a_parser_failure():
    good, bad = Future(), Future()
    good.set_result({"request": "retained"})
    bad.set_exception(ValueError("bad SSE"))

    rows, error = campaign.settle_futures([good, bad])

    assert rows == [{"request": "retained"}]
    assert isinstance(error, ValueError)


def test_completion_rate_is_separate_from_token_timing_observability():
    plan = campaign.make_plan()
    offered = campaign.measurement_rows(
        plan, campaign.offered_trace(plan, RATES, "prefill_heavy", .25, 0),
    )[:1]
    epoch = 10**12
    request = complete({"scheduled_ns": epoch + int(offered[0]["offset_s"] * 1e9),
                        "offset_s": offered[0]["offset_s"],
                        "population": "incumbent"})
    request.update({"exact_token_timestamps": False, "mean_tpot_s": None,
                    "token_itls_s": []})
    metrics = [{"monotonic_ns": epoch + second * 10**9,
                "vllm:num_requests_waiting": 0,
                "vllm:num_requests_running": 0,
                "vllm:gpu_cache_usage_perc": .1}
               for second in range(60, 301, 10)]

    result = campaign.summarize(plan, offered, [request], metrics, True)

    assert result["incumbent_exact_completion_rate"] == 1
    assert not result["tpot_reportable"]


def test_dense_measurement_telemetry_can_establish_stability():
    plan = campaign.make_plan()
    offered = campaign.measurement_rows(
        plan, campaign.offered_trace(plan, RATES, "prefill_heavy", .25, 0),
    )[:1]
    epoch = 10**12
    request = complete({"scheduled_ns": epoch + int(offered[0]["offset_s"] * 1e9),
                        "start_ns": epoch + int(offered[0]["offset_s"] * 1e9),
                        "offset_s": offered[0]["offset_s"],
                        "population": "incumbent"})
    metrics = [{"monotonic_ns": epoch + int(second * 1e9),
                "vllm:num_requests_waiting": 0,
                "vllm:num_requests_running": 0,
                "vllm:gpu_cache_usage_perc": .1}
               for second in (60 + index * .25 for index in range(960))]

    result = campaign.summarize(plan, offered, [request], metrics, True)

    assert result["telemetry_window_complete"] and result["stable"]


def test_calibration_reducer_takes_block_peaks_then_median(tmp_path):
    plan = campaign.make_plan()
    cells = [row for row in plan["cells"] if row["hardware"] == "a100"
             and row["kind"] == "calibration"]
    for row in cells:
        path = tmp_path / row["cell_id"]
        path.mkdir()
        value = (100 if row["phase"] == "prefill" else 10) \
            + row["concurrency"] + row["block"]
        (path / "result.json").write_text(json.dumps(evidence(
            plan, row, tokens_per_s=value,
        )))

    result = campaign.reduce_calibration(plan, "a100", tmp_path)

    assert result["prefill_tps"] == 117
    assert result["decode_tps"] == 139
    assert result["normalizer_kind"] == "synchronized_burst_throughput"
    assert result["edge_censored"]
    assert result["balanced_shape"]["append_tokens"] > 0
    assert result["planned_parked_prefix_tokens"] > 0


def test_calibration_reducer_rejects_mixed_runtime_identity(tmp_path):
    plan = campaign.make_plan()
    cells = [row for row in plan["cells"] if row["hardware"] == "a100"
             and row["kind"] == "calibration"]
    for index, row in enumerate(cells):
        path = tmp_path / row["cell_id"]
        path.mkdir()
        values = {"tokens_per_s": 100}
        if index == 0:
            values.update(runtime_identity={"serving_class": "changed"},
                          runtime_identity_sha256=campaign.digest(
                              {"serving_class": "changed"}))
        (path / "result.json").write_text(json.dumps(evidence(plan, row, **values)))

    with pytest.raises(RuntimeError, match="runtime"):
        campaign.reduce_calibration(plan, "a100", tmp_path)


def test_joint_attainment_uses_all_offered_incumbents():
    passing = complete({"population": "incumbent", "offset_s": 61})
    failed = {**passing, "error": "timeout"}

    assert campaign.joint_attainment(
        campaign.make_plan(), [passing, failed], 1, .1,
    ) == .5


def test_boundary_requires_every_restart_and_takes_the_weaker_direction(tmp_path):
    plan = campaign.make_plan()
    for cell in plan["cells"]:
        if cell["hardware"] != "a100" or cell["kind"] == "calibration":
            continue
        path = tmp_path / cell["cell_id"]
        path.mkdir()
        limit = .70
        stable = cell["kind"] == "residency_control" \
            or cell["target_rho"] <= limit
        if cell["kind"] == "headroom" and cell["direction"] == "prefill_heavy" \
                and cell["target_rho"] == .70 and cell["block"] == 2:
            stable = False
        collapse = cell["kind"] == "headroom" \
            and cell["direction"] == "decode_heavy" \
            and cell["target_rho"] == 1.10 and cell["block"] == 0
        (path / "result.json").write_text(json.dumps(evidence(
            plan, cell, stable=stable,
            p90_ttft_s=None if collapse else .2,
            p90_mean_tpot_s=None if collapse else .02,
            tpot_reportable=not collapse, incumbent_exact=0 if collapse else 1,
            incumbent_exact_completion_rate=0 if collapse else 1,
            all_offered_exact_completion_rate=0 if collapse else 1,
            preloaded_kv_usage=.1 if cell["kind"] == "headroom" else .02,
            initial_kv_usage=None if collapse else cell.get("target_rho", 0),
        )))
        if cell["kind"] == "headroom":
            request = complete({"population": "incumbent", "offset_s": 61})
            if collapse:
                request["error"] = "service timeout"
            (path / "requests.json").write_text(json.dumps([request]))

    result = campaign.reduce_headroom(plan, "a100", tmp_path, 1, .1)

    assert {key: value["slo_last_pass"]
            for key, value in result["direction_results"].items()} == {
        "prefill_heavy": .50, "decode_heavy": .70,
    }
    assert result["scout_conservative_bound"] == .50
    assert result["selection_ready"] and not result["planner_usable"]
    assert result["residency_control"]["pass"]
    assert result["residency_control"]["expected_preloaded_kv_usage_delta"] \
        == pytest.approx(77_312 / 1_000_000)
    assert result["residency_control"]["measurement_start_kv_range"] == [
        campaign.BASE_RHO, max(campaign.LOADS),
    ]


def test_unhealthy_residency_control_withholds_confirmation(tmp_path):
    plan = campaign.make_plan()
    for cell in plan["cells"]:
        if cell["hardware"] != "a100" or cell["kind"] == "calibration":
            continue
        path = tmp_path / cell["cell_id"]
        path.mkdir()
        control_failure = cell["kind"] == "residency_control" and cell["block"] == 0
        (path / "result.json").write_text(json.dumps(evidence(
            plan, cell, stable=not control_failure,
            incumbent_exact_completion_rate=.99 if control_failure else 1,
            all_offered_exact_completion_rate=.99 if control_failure else 1,
            p90_ttft_s=.2, p90_mean_tpot_s=.02, tpot_reportable=True,
            preloaded_kv_usage=.1 if cell["kind"] == "headroom" else .02,
        )))
        if cell["kind"] == "headroom":
            (path / "requests.json").write_text(json.dumps([
                complete({"population": "incumbent", "offset_s": 61}),
            ]))

    result = campaign.reduce_headroom(plan, "a100", tmp_path, 1, .1)

    assert not result["residency_control"]["pass"]
    assert not result["selection_ready"]


def test_missing_parked_stock_withholds_confirmation():
    _plan, core, scout = confirmation_artifacts()
    controls = [{**row, "preloaded_kv_usage": .1} for row in scout["controls"]]

    result = campaign.build_scout(
        core, "a100", scout["rows"], controls, scout["targets"],
    )

    assert not result["residency_control"]["stock_match"]
    assert not result["selection_ready"]


def test_reducer_rejects_a_stale_or_mislabeled_cell(tmp_path):
    plan = campaign.make_plan()
    cells = [row for row in plan["cells"] if row["hardware"] == "a100"
             and row["kind"] == "calibration"]
    for index, cell in enumerate(cells):
        path = tmp_path / cell["cell_id"]
        path.mkdir()
        row = evidence(plan, cell, tokens_per_s=100)
        if index == 0:
            row["block"] = 99
        (path / "result.json").write_text(json.dumps(row))

    with pytest.raises(RuntimeError, match="identity"):
        campaign.reduce_calibration(plan, "a100", tmp_path)


def confirmation_artifacts() -> tuple[dict, dict, dict]:
    core = campaign.make_plan()
    rows, controls = [], []
    for cell in core["cells"]:
        if cell["hardware"] != "a100" or cell["kind"] == "calibration":
            continue
        limit = .50 if cell["direction"] == "prefill_heavy" else .70
        stable = cell["kind"] == "residency_control" \
            or cell["direction"] == "baseline" or cell["target_rho"] <= limit
        row = evidence(core, cell, stable=stable, p90_ttft_s=.2,
                       p90_mean_tpot_s=.02, tpot_reportable=True,
                       preloaded_kv_usage=.1 if cell["kind"] == "headroom" else .02)
        (controls if cell["kind"] == "residency_control" else rows).append(row)
    scout = campaign.build_scout(
        core, "a100", rows, controls,
        {"p90_ttft_s": 1, "p90_mean_tpot_s": .1},
    )
    return campaign.make_confirmation_plan(core, scout, "a100"), core, scout


def confirmation_plan() -> dict:
    return confirmation_artifacts()[0]


def test_confirmation_is_three_new_blocks_for_brackets_and_balanced_shape():
    plan = confirmation_plan()

    assert plan["schema"] == campaign.CONFIRM_SCHEMA
    assert len(plan["cells"]) == 3 + 2 * 2 * 3 + 3
    assert {cell["block"] for cell in plan["cells"]} == {3, 4, 5}
    assert {cell["role"] for cell in plan["cells"]} == {
        "baseline", "last_pass", "first_fail", "balanced",
    }
    campaign.validate_plan(json.loads(json.dumps(plan)))
    assert campaign.offered_trace(plan, RATES, "balanced", .50, 3)


def test_confirmation_reuses_the_discovery_runtime_and_normalization():
    plan = confirmation_plan()
    rates = {**RATES, "runtime_identity_sha256": IDENTITY_SHA,
             "sha256": "normalization"}

    campaign.validate_stage_inputs(plan, rates, IDENTITY)
    with pytest.raises(RuntimeError, match="confirmation"):
        campaign.validate_stage_inputs(plan, {**rates, "sha256": "different"},
                                       IDENTITY)


def test_only_confirmation_can_emit_a_planner_usable_bound(tmp_path):
    plan, core, scout = confirmation_artifacts()
    for cell in plan["cells"]:
        path = tmp_path / cell["cell_id"]
        path.mkdir()
        feasible = cell["role"] != "first_fail"
        collapse = cell["role"] == "first_fail" and cell["block"] == 3
        (path / "result.json").write_text(json.dumps(evidence(
            plan, cell, stable=feasible,
            p90_ttft_s=None if collapse else .2,
            p90_mean_tpot_s=None if collapse else .02,
            tpot_reportable=not collapse, incumbent_exact=0 if collapse else 1,
        )))
        request = complete({"population": "incumbent", "offset_s": 61})
        if collapse:
            request["error"] = "service timeout"
        (path / "requests.json").write_text(json.dumps([request]))

    result = campaign.reduce_confirmation(plan, tmp_path, core, scout)

    assert result["planner_usable"]
    assert result["supported_bound"] == .50


def test_confirmation_disagreement_withholds_the_bound(tmp_path):
    plan, core, scout = confirmation_artifacts()
    for cell in plan["cells"]:
        path = tmp_path / cell["cell_id"]
        path.mkdir()
        feasible = cell["role"] != "first_fail" or cell["block"] == 3
        (path / "result.json").write_text(json.dumps(evidence(
            plan, cell, stable=feasible, p90_ttft_s=.2,
            p90_mean_tpot_s=.02, tpot_reportable=True,
        )))
        (path / "requests.json").write_text(json.dumps([
            complete({"population": "incumbent", "offset_s": 61}),
        ]))

    result = campaign.reduce_confirmation(plan, tmp_path, core, scout)

    assert not result["planner_usable"] and result["supported_bound"] is None


def test_confirmation_reduction_requires_the_bound_source_scout(tmp_path):
    plan, core, scout = confirmation_artifacts()
    changed = {**scout, "targets": {"p90_ttft_s": 2,
                                     "p90_mean_tpot_s": .1}}

    with pytest.raises(RuntimeError, match="source scout"):
        campaign.reduce_confirmation(plan, tmp_path, core, changed)


def test_confirmation_plan_recomputes_scout_brackets():
    _plan, core, scout = confirmation_artifacts()
    changed = copy.deepcopy(scout)
    changed["direction_results"]["prefill_heavy"].update(
        slo_last_pass=.70, slo_first_fail=.85,
    )

    with pytest.raises(RuntimeError, match="scout evidence"):
        campaign.make_confirmation_plan(core, changed, "a100")


def test_scout_rejects_execution_outside_the_frozen_order():
    _plan, core, scout = confirmation_artifacts()
    rows = copy.deepcopy(scout["rows"])
    rows[0]["started_wall_ns"], rows[1]["started_wall_ns"] = \
        rows[1]["started_wall_ns"], rows[0]["started_wall_ns"]

    with pytest.raises(RuntimeError, match="run order"):
        campaign.build_scout(core, "a100", rows, scout["controls"],
                             scout["targets"])


def test_supported_bound_loader_rejects_an_edited_result():
    plan, core, scout = confirmation_artifacts()
    result = {"schema": campaign.CONFIRM_SCHEMA, "stage": "confirmation",
              "hardware": "a100", "plan_sha256": campaign.digest(plan),
              "source_plan_sha256": campaign.digest(core),
              "source_scout_sha256": campaign.digest(scout),
              "targets": plan["targets"], "planner_usable": True,
              "supported_bound": .50,
              "runtime_identity": IDENTITY,
              "runtime_identity_sha256": IDENTITY_SHA,
              "normalization_sha256": "normalization",
              "checks": {name: True for name in (
                  "prefill_heavy:baseline", "prefill_heavy:last_pass",
                  "prefill_heavy:first_fail", "decode_heavy:baseline",
                  "decode_heavy:last_pass", "decode_heavy:first_fail", "balanced",
              )},
              "rows": [evidence(
                  plan, cell, stable=cell["role"] != "first_fail",
                  p90_ttft_s=.2, p90_mean_tpot_s=.02, tpot_reportable=True,
              ) for cell in plan["cells"]]}

    assert campaign.supported_bound(result, plan, core, scout) == .50
    with pytest.raises(RuntimeError, match="confirmation result"):
        campaign.supported_bound({**result, "supported_bound": .70},
                                 plan, core, scout)
