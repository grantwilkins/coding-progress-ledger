"""
Claim:
The v5 campaign uses exact 10K idle snapshots, fits width-8 replay as an
unclamped positive capacity factor rather than a width cap, runs six scalable
policies over one manifest while excluding scale-limited greedy_lagrangian,
hard-fails incomplete shard reductions, and censors misses without allowing
them to dominate successful target attainment.

Plausible wrong implementations:
- Reintroduce grouped template replication or an eight-flow ceiling.
- Fit replay from failed, non-10G, or non-width-8 episodes.
- Clamp a measured sub-unit replay factor to one or reject it as a slowdown.
- Accept a nonpositive or above-width replay factor.
- Omit side-case sentinels or one hardware baseline.
- Compare unrelated workload seeds in Pareto dominance.
- Let a deadline miss dominate a successful point.
- Reduce missing, duplicate, or stale shard rows.
- Find target attainment from instantaneous rather than trailing-window power.
- Include greedy_lagrangian or omit an intended scalable baseline.
- Apply an absolute-watt tolerance after normalizing the HiGHS gain row.
- Derive a mixed trace/anchor shard's CSV schema from only its first row.
"""

import csv
import json

import pytest

import simulated_pareto_campaign as campaign
from simulated_pareto_campaign import (
    ANCHORS, MODEL, ROOT, WORKLOADS, attainment_time, file_hash, fit_hardware,
    manifest_rows, pareto_flags, reduce, run_row,
)
from profiles import ModelProfile
from test_execution_simulator import model


def _write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_is_exact_full_grid_plus_sentinel():
    rows = manifest_rows()

    assert len(rows) == 12_336
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert {row["policy"] for row in rows} == set(campaign.POLICIES)
    assert "greedy_lagrangian" not in campaign.POLICIES
    assert {row["case"] for row in rows} == {
        "central", "conservative", "optimistic",
    }
    assert {row["shard"] for row in rows} == set(range(64))
    assert sum(row["case"] == "central" for row in rows) == 11_760
    assert sum(row["case"] != "central" for row in rows) == 576


def test_anchor_contexts_stay_inside_measured_rate_surfaces():
    case = ModelProfile.load(MODEL).case()

    for context in ANCHORS:
        assert case.prefill.rate(context, 1) > 0
        assert case.decode.rate(context, 1) > 0


def test_write_csv_unions_mixed_row_fields(tmp_path):
    path = tmp_path / "mixed.csv"

    campaign._write_csv(path, [{"kind": "trace"},
                               {"kind": "anchor", "anchor_tokens": 1998}])

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"kind": "trace", "anchor_tokens": ""},
                    {"kind": "anchor", "anchor_tokens": "1998"}]


def test_highs_max_gain_fallback_handles_trace_power_scale():
    profile = ModelProfile.load(MODEL)
    manifest = {
        "sessions": 10_000,
        "model": {"path": str(MODEL.relative_to(ROOT)), "sha256": file_hash(MODEL)},
        "workloads": {
            str(path.relative_to(ROOT)): {"sha256": file_hash(path)}
            for path in WORKLOADS
        },
        "fits": fit_hardware(profile),
    }
    result = run_row({
        "episode_id": "interactive_coding-seed-1", "kind": "trace",
        "workload": "profiles/interactive_coding.json", "seed": 1,
        "case": "central", "bandwidth_mbps": 10_000, "deadline_s": 60,
        "target_fraction": .5, "policy": "queue_haul",
    }, manifest)

    assert result["sessions"] == 10_000
    assert result["admitted_moves"] > 0


def test_width8_fit_uses_only_complete_10g_eight_move_episodes(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    scenarios, stages = [], []
    for method, prefix in (("replay", "r"), ("kv_transfer", "k")):
        for episode in range(10):
            scenario = f"{prefix}{episode}"
            scenarios.append({
                "scenario_id": scenario, "kind": "migration", "status": "complete",
                "bandwidth_mbps": 10000, "concurrency": 8, "method": method,
                "source_added_power_w": 4 if method == "kv_transfer" else 1,
                "destination_added_power_w": 20 if method == "kv_transfer" else 200,
            })
            if method == "replay":
                stages.extend({
                    "scenario_id": scenario, "method": "replay", "success": "true",
                    "phase": "initial", "measured_prompt_tokens": 10,
                    "start_ns": 0, "destination_ready_ns": 200_000_000,
                } for _ in range(8))
    _write(evidence / "scenarios.csv", scenarios)
    _write(evidence / "migration_stages.csv", stages)

    fitted = fit_hardware(model(tmp_path, tp=1), evidence)

    assert fitted["replay_episodes"] == 10
    assert fitted["cases"]["central"]["replay_speedup"] == pytest.approx(4)
    assert fitted["cases"]["central"]["source_power_w"] == {
        "replay": 1, "kv_transfer": 4,
    }


def test_width8_fit_hard_fails_above_width(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    scenarios = [{
        "scenario_id": f"r{i}", "kind": "migration", "status": "complete",
        "bandwidth_mbps": 10000, "concurrency": 8, "method": "replay",
        "source_added_power_w": 1, "destination_added_power_w": 1,
    } for i in range(10)] + [{
        "scenario_id": f"k{i}", "kind": "migration", "status": "complete",
        "bandwidth_mbps": 10000, "concurrency": 8, "method": "kv_transfer",
        "source_added_power_w": 1, "destination_added_power_w": 1,
    } for i in range(10)]
    stages = [{
        "scenario_id": f"r{i}", "method": "replay", "success": "true",
        "phase": "initial", "measured_prompt_tokens": 10,
        "start_ns": 0, "destination_ready_ns": 50_000_000,
    } for i in range(10) for _ in range(8)]
    _write(evidence / "scenarios.csv", scenarios)
    _write(evidence / "migration_stages.csv", stages)

    with pytest.raises(ValueError, match="outside"):
        fit_hardware(model(tmp_path, tp=1), evidence)


def test_actual_stage_span_evidence_preserves_measured_subunit_factors():
    fitted = fit_hardware(ModelProfile.load(MODEL))
    factors = fitted["cases"]

    assert fitted["replay_episodes"] == 36
    assert .95 < factors["conservative"]["replay_speedup"] < 1
    assert .95 < factors["central"]["replay_speedup"] < 1
    assert 1 < factors["optimistic"]["replay_speedup"] < 1.05


def test_trailing_window_attainment_finds_first_crossing():
    power = ((0, 100, 0), (2, 120, 0), (4, 50, 0), (10, 50, 0))

    assert attainment_time(power, 75, 2, 10) == pytest.approx(37 / 7)
    assert attainment_time(power, 40, 2, 10) is None


def test_all_seven_policies_run_same_exact_idle_episode():
    fit = {
        "replay_speedup": 1.1,
        "source_power_w": {"replay": 1, "kv_transfer": 4},
        "destination_power_w": {"replay": 230, "kv_transfer": 23},
    }
    manifest = {
        "sessions": 10,
        "model": {"path": str(MODEL.relative_to(ROOT)), "sha256": file_hash(MODEL)},
        "workloads": {
            str(path.relative_to(ROOT)): {"sha256": file_hash(path)}
            for path in WORKLOADS
        },
        "fits": {"cases": {"central": fit}},
    }
    base = {
        "row_id": "test", "shard": 0, "episode_id": "coding-seed-0",
        "kind": "trace", "workload": "profiles/coding.json", "seed": 0,
        "case": "central", "bandwidth_mbps": 10000, "deadline_s": 60,
        "target_fraction": .25,
    }

    rows = [run_row({**base, "policy": policy}, manifest)
            for policy in campaign.POLICIES]

    assert {row["policy"] for row in rows} == set(campaign.POLICIES)
    assert all(row["sessions"] == 10 and row["committed_moves"] == row["admitted_moves"]
               for row in rows)
    assert all({"packing_repair_count", "packing_repair_s",
                "deadline_repair_count", "deadline_repair_s"} <= row.keys()
               for row in rows)


def test_censored_miss_cannot_dominate_success():
    rows = [
        {"episode_id": "a", "bandwidth_mbps": 1000, "case": "central",
         "target_attained": True, "target_attainment_s": 8,
         "attained_shed_fraction": .5},
        {"episode_id": "a", "bandwidth_mbps": 1000, "case": "central",
         "target_attained": False, "target_attainment_s": "",
         "attained_shed_fraction": .9},
    ]

    pareto_flags(rows)

    assert rows[0]["pareto"]
    assert not rows[1]["pareto"]

    misses = [dict(rows[1]), dict(rows[1])]
    pareto_flags(misses)
    assert not any(row["pareto"] for row in misses)


def test_reduce_hard_fails_missing_shards(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": campaign.SCHEMA, "shards": 2, "rows": [],
    }))

    with pytest.raises(FileNotFoundError, match="missing shards"):
        reduce(tmp_path)


def test_reduce_hard_fails_manifest_row_mutation(tmp_path):
    expected = {
        "row_id": "v4-0", "shard": 0, "episode_id": "coding-seed-0",
        "kind": "trace", "workload": "profiles/coding.json", "seed": 0,
        "case": "central", "bandwidth_mbps": 1000, "deadline_s": 60,
        "target_fraction": .25, "policy": "queue_haul",
    }
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": campaign.SCHEMA, "shards": 1, "rows": [expected],
    }))
    altered = {**expected, "policy": "greedy", "sessions": 10,
               "attained_shed_fraction": .2, "target_attained": False,
               "censored": True}
    _write(tmp_path / "shard-00.csv", [altered])

    with pytest.raises(ValueError, match="manifest"):
        reduce(tmp_path)
