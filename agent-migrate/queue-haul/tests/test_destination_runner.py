"""
Claim:
Destination load uses scheduled work, exact session identity, and conservative
normal/emergency/stability classifications.

Plausible wrong implementations:
- Use achieved completions instead of offered tokens.
- Reuse one prefix across nominally distinct sessions.
- Treat an SLO boundary as infeasible or a just-outside value as feasible.
- Declare a growing destination queue stable.
"""

import pytest
import json
from pathlib import Path
from types import SimpleNamespace

import destination_campaign as campaign
import destination_runner as runner


def request(ttft=.1, tpot=.01, error=""):
    return {"status": 200, "error": error, "output_tokens": 2,
            "planned_output_tokens": 2, "ttft_s": ttft, "mean_tpot_s": tpot,
            "input_tokens": 10}


def metrics(slope=0):
    return [{"monotonic_ns": i * 10**9,
             "vllm:num_requests_waiting": 1 + slope * i} for i in range(100)]


def test_schedule_and_session_tokens_are_deterministic_but_isolated():
    assert runner.poisson_schedule(2, 4, 7) == runner.poisson_schedule(2, 4, 7)
    assert runner.uniform_schedule(2, 4, 7) == (0, .5, 1, 1.5)
    assert runner.anchor_rate(100, 20) == 5
    a = runner.Session("a", 4, 2, 3, 100, 7)
    b = runner.Session("b", 4, 2, 3, 100, 7)
    first, forced = a.prompt(0)
    a.commit(first, forced)
    assert a.prompt(1)[0][:4] == first[:4]
    assert len(a.prompt(1)[0]) == 6
    assert b.prompt(0)[0][:4] != first[:4]


def test_offered_work_does_not_depend_on_completion():
    rows = [request(), request(error="failed")]
    assert runner.offered_work(rows, 2) == (10, 2)


def test_slo_boundary_is_inclusive_and_queue_growth_is_not_stable():
    slos = {"normal": {"p90_ttft_s": 2, "p90_mean_tpot_s": .1},
            "emergency": {"p90_ttft_s": 10, "p90_mean_tpot_s": .25}}
    exact = runner.classify([request(2, .1)], metrics(), True, slos)
    outside = runner.classify([request(2.001, .1)], metrics(), True, slos)
    growing = runner.classify([request()], metrics(.1), True, slos)
    assert exact == {"normal": True, "emergency": True, "stable": True}
    assert not outside["normal"] and outside["emergency"]
    assert not growing["stable"]


def test_queue_drift_requires_real_samples():
    with pytest.raises(ValueError, match="sampled"):
        runner.queue_drift_upper(metrics()[:1])


def test_anchor_gate_accepts_improvement_and_fifteen_percent_underdelivery():
    expected = {("prefill", 4096): 100, ("decode", 4096): 50}
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 85},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 57.5},
    ], expected)["within_limit"]
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 200},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 100},
    ], expected)["within_limit"]
    report = runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 84.9},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 50},
    ], expected)
    assert not report["within_limit"]
    with pytest.raises(ValueError, match="incomplete"):
        runner.anchor_gate([{"metric": "prefill", "context_tokens": 4096,
                             "tokens_per_s": 100}], expected)


def test_anchor_gate_uses_independent_run_median_not_last_request():
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": value}
        for value in (100, 100, 1)
    ], {("prefill", 4096): 100})["within_limit"]


def test_anchor_mismatch_recalibrates_central_profile():
    profile = {"cases": {"central": {"prefill_tps": {"1": [[1, 10], [20, 30]]}}}}
    report = {"anchors": [{"metric": "prefill", "context_tokens": 10,
                            "observed_tokens_per_s": 25}]}
    runner.apply_anchor_rates(profile, report)
    assert profile["cases"]["central"]["prefill_tps"]["1"] == [[1, 10], [10, 25], [20, 30]]


def test_profile_rate_interpolates_only_inside_measured_domain():
    profile = {"cases": {"central": {"prefill_tps": {"1": [[10, 100], [20, 200]]}}}}
    assert runner.profile_rate(profile, "prefill", 15) == 150
    with pytest.raises(ValueError, match="outside"):
        runner.profile_rate(profile, "prefill", 9)


def test_repaired_baseline_passes_its_independent_anchor_gate():
    root = Path(runner.__file__).parent
    profile = json.loads((root / "profiles/gpt_oss_20b_a100_tp1.json").read_text())
    rows = json.loads((root / "outputs/destination-anchor-baseline-20260722.json").read_text())["anchors"]
    expected = {(metric, context): runner.profile_rate(profile, metric, context)
                for metric in ("prefill", "decode") for context in (4096, 16384, 24576)}
    assert runner.anchor_gate(rows, expected)["within_limit"]


def test_integrity_preflight_requires_same_but_not_cross_session_cache(monkeypatch):
    cached = iter((0, 4000, 0))
    monkeypatch.setattr(runner, "_completion", lambda *_: {"cached_tokens": next(cached)})
    monkeypatch.setattr(runner.testbed, "http_json", lambda *_: {"count": 2})
    report = runner.integrity_preflight(SimpleNamespace(
        host="h", sink_port=1, model="m"),
        {"image_sha256": campaign.IMAGE_SHA256}, {"acceptance": {"ok": True}}, 100)
    assert report["same_session_cache_hit"] and report["cross_session_cache_hits"] == 0


def test_launch_inputs_are_relative_and_checksum_pinned(tmp_path):
    splits = {job: {split: [f"{job}-{split}-{i}" for i in range(n)]
                    for split, n in (("fit", 12), ("tune", 6), ("validation", 6))}
              for job in campaign.JOB_CLASSES}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest": {"schema": campaign.MANIFEST_SCHEMA,
                                                  "splits": splits},
                                    "traces": [{"session_id": sid, "input_tokens_total": 256}
                                               for values in splits.values()
                                               for ids in values.values() for sid in ids]}))
    bundle = tmp_path / "bundle"; campaign.prepare(manifest, bundle)
    plan, loaded, profile = runner.load_inputs(bundle / "plan.json")
    assert plan["schema"] == campaign.SCHEMA and loaded["manifest"]["splits"] == splits
    assert profile["model"] == "openai/gpt-oss-20b"
    (bundle / "baseline-profile.json").write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        runner.load_inputs(bundle / "plan.json")


def test_run_root_cannot_resume_a_different_commit(tmp_path):
    path = tmp_path / "run.json"
    runner.write_run_metadata(path, {"git_sha": "one"})
    runner.write_run_metadata(path, {"git_sha": "one"})
    with pytest.raises(RuntimeError, match="different campaign or commit"):
        runner.write_run_metadata(path, {"git_sha": "two"})


def test_loaded_scenario_has_one_session_and_one_method():
    scenario = runner.migration_scenario({"id": "s", "job_class": "coding"},
                                         "replay", 16384, 10000, 2)
    assert scenario["concurrency"] == scenario["move_concurrency"] == 1
    assert scenario["moves"] == [{**scenario["sessions"][0], "method": "replay"}]


def test_adaptive_search_brackets_each_nested_boundary():
    boundaries = {"normal": 1, "emergency": 2, "stable": 3}
    found = runner.find_boundaries(
        lambda radius: {mode: radius <= bound for mode, bound in boundaries.items()}
    )
    for mode, bound in boundaries.items():
        assert found[mode][0] <= bound <= found[mode][1]
        assert found[mode][1] - found[mode][0] <= .05 * found[mode][1]


def test_frontier_searches_once_then_repeats_only_boundary_cells(monkeypatch, tmp_path):
    calls = []
    thresholds = {"normal": 1, "emergency": 2, "stable": 3}
    monkeypatch.setattr(runner, "manifest_sessions", lambda *_: [runner.Session("s", 10, 2, 3, 100, 0)])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 100)
    monkeypatch.setattr(runner.testbed, "flush_lmcache", lambda *args: None)
    def probe(*args, **kwargs):
        radius = args[4]; calls.append(radius)
        return {"classification": {mode: radius <= value for mode, value in thresholds.items()}}
    monkeypatch.setattr(runner, "service_probe", probe)
    plan = {"service": {"directions": ["coding"], "initial_repeats": 3,
                        "disagreement_repeats": 5, "radial_resolution": .05,
                        "hold_min_s": 1, "slos": {}}}
    rows, bounds = runner.measure_frontier(plan, {}, {}, SimpleNamespace(
        host="h", sink_port=1, model="m"), object(), tmp_path)
    assert len(rows) == 9 and all(sum(r["mode"] == mode for r in rows) == 3 for mode in thresholds)
    assert bounds["normal"] <= bounds["emergency"] <= bounds["stable"]
    assert (tmp_path / "validation.jsonl").is_file() and len(calls) < 60


def test_rho_uses_token_counter_differences_and_requires_thirty_seconds():
    rows = [
        {"monotonic_ns": 0, "vllm:prompt_tokens_total": 10,
         "vllm:generation_tokens_total": 20},
        {"monotonic_ns": 40_000_000_000, "vllm:prompt_tokens_total": 50,
         "vllm:generation_tokens_total": 60},
    ]
    assert runner.measured_rho(rows, 2, 2) == 1
    assert runner.measured_rho(rows, 2, 2, 2) == .5
    assert runner.require_rho(rows, 1.04, 2, 2) == 1
    with pytest.raises(RuntimeError, match="misses"):
        runner.require_rho(rows, 1.051, 2, 2)
    with pytest.raises(ValueError, match="thirty"):
        runner.measured_rho([rows[0], {**rows[1], "monotonic_ns": 29_000_000_000}], 2, 2)
