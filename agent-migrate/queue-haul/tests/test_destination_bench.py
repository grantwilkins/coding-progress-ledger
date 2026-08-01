"""
Claim:
The two-site bench creates reproducible trace-driven populations, freezes equal
source/sink A100 inventory, varies sink pressures independently, and never labels
out-of-domain sensitivity as supported evidence.

Plausible wrong implementations:
- Treat the full prompt as resident history or lose the newly appended work.
- Round aggregate KV instead of each private session.
- Resize hardware while varying pressure or multiply shared WAN by GPU count.
- Duplicate replicas or route capacity when splitting inventory into pools.
- Couple service and KV baselines.
- Reverse a minimum-capacity threshold or miss a censored boundary.
- Drop context or bandwidth extrapolation from the result label.
"""

from types import SimpleNamespace

import pytest

from destination_bench import (
    KV_BLOCK_TOKENS,
    Pressure,
    architecture,
    boundary,
    evidence,
    extrapolate_replay,
    pack_source,
    parse_pool_counts,
    parse_solvers,
    sample_sessions,
    scenario,
    trace_shapes,
)
from profiles import ModelProfile
from simulate import SimSession


def model():
    return ModelProfile.load(
        __import__("pathlib").Path(__file__).parents[1]
        / "profiles/gpt_oss_20b_a100_tp1.json"
    )


def manifest():
    return {
        "manifest": {"splits": {"coding": {
            "fit": ["a"], "tune": [], "validation": [],
        }}},
        "traces": [{
            "session_id": "a", "turn": 2, "input_tokens_total": 5000,
            "newly_append_tokens": 1000, "output_tokens": 200,
        }],
    }


def test_reference_bench_accepts_additive_highs_backend():
    assert parse_solvers(
        "lp,lp_highs,lp_column_generation,lp_column_generation_persistent,"
        "lp_column_generation_lazy,lp_column_generation_native",
    ) == (
        "lp", "lp_highs", "lp_column_generation",
        "lp_column_generation_persistent", "lp_column_generation_lazy",
        "lp_column_generation_native",
    )


def test_pool_count_parser_and_split_preserve_frozen_inventory():
    assert parse_pool_counts("1,2,4,8") == (1, 2, 4, 8)
    sessions = (SimSession("a", "source-0", 16_384, 1, 1, 32_768),)

    split = architecture(model(), sessions, 8, Pressure(), pool_count=4)

    replicas = [r.replica_id for pool in split.pools for r in pool.replicas]
    assert len(split.pools) == 4
    assert sorted(replicas) == [f"dest-{i}" for i in range(8)]
    assert all(pool.route == ("source-egress", "wan", "destination-ingress")
               for pool in split.pools)


def test_trace_shape_separates_resident_history_from_next_request():
    shape = trace_shapes(manifest(), "coding")[0]

    assert (shape.context_tokens, shape.prompt_tokens, shape.output_tokens) == (
        4000, 1000, 200,
    )


def test_population_sampling_is_seeded_and_uses_declared_reference_rate():
    shapes = trace_shapes(manifest(), "coding")

    first = sample_sessions(shapes, 2, 7, 2)
    second = sample_sessions(shapes, 2, 7, 2)

    assert first == second
    assert first[0].expected_f == pytest.approx(1000 / 180)
    assert first[0].expected_growth_tokens_per_s == pytest.approx(1200 / 180)


def test_source_packing_rounds_private_kv_before_summing():
    sessions = tuple(
        SimSession(str(i), "x", 481_577, 0, 0, 1) for i in range(2)
    )

    _, replicas = pack_source(sessions, model())

    assert replicas == 2


def test_topology_freezes_equal_inventory_and_one_shared_wan():
    sessions = (SimSession("a", "source-0", 4000, 1, 1, 8000,
                           expected_growth_tokens_per_s=1),)
    built = scenario(model(), sessions, 3, Pressure())

    assert sum(i.instance_id.startswith("source-") for i in built.instances) == 3
    assert sum(i.instance_id.startswith("dest-") for i in built.instances) == 3
    assert [link.link_id for link in built.links].count("wan") == 1
    assert built.sessions[0].context_tokens == 4180
    assert built.sessions[0].expected_growth_tokens_per_s == 0


def test_service_and_kv_baselines_are_independent():
    sessions = (SimSession("a", "source-0", 16_384, 1, 1, 32_768),)
    service = architecture(model(), sessions, 1, Pressure(service=.5))
    kv = architecture(model(), sessions, 1, Pressure(kv=.25))
    service_replica, kv_replica = service.pools[0].replicas[0], kv.pools[0].replicas[0]

    assert sum(service_replica.baseline_work) == pytest.approx(.5 * .096953)
    assert service_replica.baseline_kv_tokens == 0
    assert kv_replica.baseline_work == (0, 0)
    assert kv_replica.baseline_kv_tokens // KV_BLOCK_TOKENS \
        == int(963_152 / KV_BLOCK_TOKENS * .25)


def test_boundary_finds_minimum_feasible_value_and_reports_censoring():
    value, state = boundary(lambda x: x >= 4, 1, 8, True, iterations=20)
    censored, censored_state = boundary(lambda _x: False, 1, 8, True)

    assert value == pytest.approx(4, abs=.025)
    assert state == "crossing"
    assert censored == 8 and censored_state == "censored"


def test_evidence_never_hides_context_extrapolation():
    sessions = (SimSession("a", "source-0", 8000, 0, 0, 16_000),)
    arch = architecture(model(), sessions, 1, Pressure())
    move = SimpleNamespace(session_id="a", method="replay")

    status, reasons, fraction = evidence(
        arch, sessions, (move,), Pressure(),
    )

    assert status == "unsupported_extrapolation"
    assert reasons == "context"
    assert fraction == 0


def test_replay_extrapolation_uses_minimum_measured_rate_without_mutating_profile():
    profile = model()
    session = SimSession("a", "source-0", 31_000, 0, 0, 1,
                         expected_growth_tokens_per_s=10)

    extended = extrapolate_replay(profile, (session,), 100)

    original = profile.case().replay.by_concurrency[1]
    changed = extended.case().replay.by_concurrency[1]
    assert changed[0][-1] == 32_000
    assert changed[1][-1] == min(original[1])
    assert original[0][-1] == 31_562
