"""
Claim:
The fleet frontier provisions each source GPU under both the calibrated power
range and the measured offered-RPS service envelope, derives that envelope from
the re-reduced pooled sweep, samples only contexts the rate curves cover, and
requests an attainable fraction of removable power at every deadline.

Plausible wrong implementations:
- Read the envelope from the schema-v2 sweep whose pooled TPOT metric the repo
  disqualifies, or take the first violating rate instead of the last passing one.
- Report a right-censored emergency envelope as an identified boundary.
- Pack the source to max_ell alone, leaving the fleet above its service envelope.
- Pack to the service envelope alone, leaving instance load outside the
  calibrated power curve.
- Sample contexts past the measured rate curves and silently clamp the rate.
- Request the idle floor instead of an attainable fraction, which drives the
  target-first LP into its infeasible fallback.
- Derive migration headroom without subtracting the load the pools absorb.
- Take the requested fraction as the planner's credit target, so a request
  is answered with whatever shed that credit model happens to imply.
- Override the fitted regional replay factors with a scalar floor, which
  triples the prediction error against the artifact they were fitted on.
"""

from __future__ import annotations

import json

import pytest

import fleet_shed_frontier_campaign as campaign
from planner import source_power
from power_model import ExpectedPower
from destination import ProfileRateLimit
from profiles import ModelProfile, WorkloadProfile


@pytest.fixture(scope="module")
def profile():
    return ModelProfile.load(campaign.MODEL)


@pytest.fixture(scope="module")
def workload():
    return WorkloadProfile.load(campaign.WORKLOADS[campaign.HEADLINE_WORKLOAD])


def test_envelope_uses_the_last_passing_rate_and_flags_censoring():
    normal, normal_censored = campaign.envelope_rps(campaign.NORMAL_TTFT_SLO_S)
    emergency, emergency_censored = campaign.envelope_rps(
        campaign.EMERGENCY_TTFT_SLO_S)

    assert normal == 5.0 and not normal_censored
    # No swept rate violates the 10 s emergency tier, so its bound is a lower
    # bound set by the grid, not an identified boundary.
    assert emergency == 8.0 and emergency_censored
    assert "pooled-p90-tpot" in str(campaign.ENVELOPE)


def test_envelope_stops_at_the_first_violating_rate(monkeypatch, tmp_path):
    # A non-monotone curve: 6 RPS violates, 7 RPS passes again.  Taking a global
    # maximum over passing rates would wrongly certify 7.
    artifact = tmp_path / "summary.json"
    artifact.write_text(json.dumps({
        "schema": campaign.ENVELOPE_SCHEMA,
        "models": {campaign.MODEL_ID: {
            "slo": {"p90_ttft_s": 2.0, "p90_tpot_s": 0.1},
            "curve": [
                {"offered_rps": r, "p90_ttft_s_median": t,
                 "p90_tpot_s_median": 0.03}
                for r, t in ((1.0, 0.5), (5.0, 1.4), (6.0, 2.9), (7.0, 1.9))],
        }},
    }))
    monkeypatch.setattr(campaign, "ENVELOPE", artifact)

    assert campaign.envelope_rps(2.0) == (5.0, False)


def test_envelope_rejects_the_disqualified_schema(monkeypatch, tmp_path):
    artifact = tmp_path / "summary.json"
    artifact.write_text(json.dumps({"schema": "queue-haul-agentic-rps-sweep-v2",
                                    "models": {}}))
    monkeypatch.setattr(campaign, "ENVELOPE", artifact)

    with pytest.raises(RuntimeError, match="expected"):
        campaign.envelope_rps(2.0)


def test_service_bound_matches_the_profile_rate_limit_conversion(profile, workload):
    case = profile.case()
    contexts = sorted({record.context_tokens for record in workload.records})
    bound = 5.0 * campaign.request_work(case).sum()
    fits = json.loads(campaign.TIMING.read_text())["fits"]
    architecture = campaign.build_architecture(
        profile, 1, {"normal": bound, "emergency": bound, "stable": bound},
        fits, campaign.RHOS[2], 0.05, contexts)
    limit = ProfileRateLimit(
        f"{campaign.MODEL_ID}-a100-tp1/east", campaign.REF_CONTEXT,
        campaign.PROMPT, campaign.OUTPUT, campaign.RHOS[2] * 5.0, 5.0)

    conversion = limit.conversion(architecture.types[0])

    assert conversion["safe_service_bound"] == pytest.approx(bound)
    assert conversion["baseline_work"] == pytest.approx(
        tuple(campaign.RHOS[2] * 5.0 * campaign.request_work(case)))


def test_source_packing_respects_power_and_service_limits(profile, workload):
    case = profile.case()
    bound = 5.0 * campaign.request_work(case).sum()

    scenario, replicas, demand, _ = campaign.build_fleet(
        profile, workload, 400, 1001, 120.0, bound, "natural")

    loads, work = {}, {}
    for session in scenario.sessions:
        ell = session.expected_f / case.F + session.expected_g / case.G
        served = (session.expected_f / case.prefill.rate(session.context_tokens, 1)
                  + session.expected_g / case.decode.rate(session.context_tokens, 1))
        loads[session.source_instance] = loads.get(session.source_instance, 0.0) + ell
        work[session.source_instance] = work.get(session.source_instance, 0.0) + served

    assert replicas == len(loads)
    assert max(loads.values()) <= profile.max_power_load + 1e-9
    assert max(work.values()) <= bound + 1e-9
    assert demand == pytest.approx(sum(work.values()))


def test_sampled_contexts_stay_inside_the_measured_rate_curves(profile, workload):
    case = profile.case()
    covered = case.prefill.by_concurrency[1][0]

    for record in workload.records:
        assert min(covered) <= record.context_tokens <= max(covered)
        # Rates must be interpolated, never clamped by extrapolation.
        case.decode.rate(record.context_tokens, 1)
        case.replay.rate(record.context_tokens, 1)


def test_scarcity_grid_is_the_headline_axis_and_stays_admissible(profile,
                                                                 workload):
    """rho sets how much room the destinations have to accept migration work at
    all, so it is the axis the multi-action claim lives on; past the absorption
    cap the scenario is infeasible by construction, not by policy."""
    rows = campaign.manifest_rows()

    assert rows and len({row["row_id"] for row in rows}) == len(rows)
    headline = [row for row in rows if row["headline"]]
    assert {row["rho"] for row in headline} == set(campaign.RHOS)
    assert {row["policy"] for row in headline} == set(campaign.POLICIES)
    # The headline runs one workload; every other mixture is sensitivity, so
    # the claim is never read off a workload chosen after seeing the gap.
    assert {row["workload"] for row in headline} == {campaign.HEADLINE_WORKLOAD}
    assert {row["workload"] for row in rows} == set(campaign.WORKLOADS)
    # Executed shed is invariant above a few thousand sessions because packing
    # pins the sessions per node pipe, so the headline runs on that plateau and
    # a sensitivity block re-measures the same cells at the full fleet.
    assert {row["sessions"] for row in headline} == {campaign.SESSIONS}
    invariance = [row for row in rows
                  if row["sessions"] == campaign.INVARIANCE_SESSIONS]
    assert invariance and not any(row["headline"] for row in invariance)
    assert {row["policy"] for row in invariance} == set(campaign.POLICIES)
    assert len(headline) == (len(campaign.DEADLINES_S) * len(campaign.POLICIES)
                             * len(campaign.RHOS) * len(campaign.SEEDS))
    assert campaign.POLICIES["queue_haul"] == "lp_work_first"

    case = profile.case()
    bound = 5.0 * campaign.request_work(case).sum()
    _, replicas, demand, _ = campaign.build_fleet(
        profile, workload, 1500, 1001, 300.0, bound, "natural")
    # Measured on the headline workload, whose long contexts absorb more
    # destination capacity than a short mixture and so cap rho lower.
    headrooms = [campaign.migration_headroom(rho, demand, replicas, bound)
                 for rho in campaign.RHOS]
    # Every swept rho leaves room to migrate, and scarcity is monotone in rho.
    assert all(value > 0 for value in headrooms)
    assert headrooms == sorted(headrooms, reverse=True)
    with pytest.raises(RuntimeError, match="cannot absorb"):
        campaign.migration_headroom(1.0, demand, replicas, bound)


def test_migration_headroom_subtracts_both_baseline_and_absorbed_load():
    # Two pools of 10 replicas at bound 2.0 hold 40.0; a demand of 16.0 is
    # absorbed as 0.4 of that capacity, on top of the 0.45 baseline.
    headroom = campaign.migration_headroom(0.45, 16.0, 10, 2.0)

    assert headroom == pytest.approx(1.0 - 0.45 - 0.4)
    with pytest.raises(RuntimeError, match="cannot absorb"):
        campaign.migration_headroom(0.45, 44.0, 10, 2.0)


def test_replay_uses_the_fitted_regional_completion_factors(profile, workload):
    case = profile.case()
    contexts = sorted({record.context_tokens for record in workload.records})
    fits = json.loads(campaign.TIMING.read_text())["fits"]
    bound = 5.0 * campaign.request_work(case).sum()

    architecture = campaign.build_architecture(
        profile, 1, {"normal": bound, "emergency": bound, "stable": bound},
        fits, campaign.RHOS[2], 0.05, contexts)

    # A scalar prefill floor was tried and rejected: it tripled the error
    # against outputs/timing-power-validation-20260814, the artifact these
    # factors were fitted on.
    for region, destination_type in zip(campaign.REGIONS, architecture.types):
        assert destination_type.migration["replay"].compute_completion_factor \
            == pytest.approx(fits[region]["replay_compute_completion_factor"])
        assert destination_type.migration["kv_transfer"].compute_completion_factor == 1


def test_max_shed_reports_the_largest_contract_respecting_shed(monkeypatch):
    """A cell reports shed it actually executed inside every contract, not the
    largest request some probe happened to meet."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Snapshot:
        power_limit_w: float

    asks = []

    def stepped(scenario, *args, **kwargs):
        asks.append(100.0 - scenario.power_limit_w)
        return SimpleNamespace(moves=())

    monkeypatch.setattr(campaign, "plan", stepped)

    def evaluate(planned, ask):
        # Shed tracks the ask, but past 60 W the plan misses its deadline, so
        # the largest lawful shed is the one just below that edge.
        return {"realized_shed_w": ask, "within_contract": ask <= 60.0}

    _, outcome, ask, probes = campaign.max_shed_plan(
        Snapshot(0.0), None, None, "greedy", 1, "normal", None, 100.0, evaluate)

    assert outcome["within_contract"] and ask <= 60.0
    assert outcome["realized_shed_w"] == pytest.approx(60.0, abs=100.0 / 2 ** 7)
    assert probes == campaign.MAX_SHED_STEPS
    assert max(asks) > 60.0, "the search must probe past the edge to find it"


def test_max_shed_returns_the_last_probe_when_nothing_is_lawful(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Snapshot:
        power_limit_w: float

    monkeypatch.setattr(campaign, "plan",
                        lambda *a, **k: SimpleNamespace(moves=()))
    _, outcome, _, _ = campaign.max_shed_plan(
        Snapshot(0.0), None, None, "greedy", 1, "normal", None, 100.0,
        lambda planned, ask: {"realized_shed_w": ask, "within_contract": False})

    assert not outcome["within_contract"]


def test_source_nodes_own_their_egress_and_reach_only_their_pools(
        profile, workload):
    """One pipe per region starves a fleet by the node count: the measured rate
    is instance-to-instance, so egress must scale with the fleet."""
    case = profile.case()
    bound = 5.0 * campaign.request_work(case).sum()
    scenario, replicas, demand, fits = campaign.build_fleet(
        profile, workload, 400, 1001, 300.0, bound, "natural")
    architecture = campaign.build_architecture(
        profile, replicas, {m: bound for m in ("normal", "emergency", "stable")},
        fits, 0.45,
        campaign.migration_headroom(0.45, demand, replicas, bound), None)

    nodes = -(-replicas // profile.gpus_per_node)
    assert len(architecture.pools) == nodes * len(campaign.REGIONS)
    assert len(scenario.links) == nodes * len(campaign.REGIONS)
    # Every pool is reachable only from its own node's source instances, and
    # every source instance reaches exactly one pool per region.
    reach = {}
    for pool in architecture.pools:
        assert pool.source_affinity
        for instance in pool.source_affinity:
            reach.setdefault(instance, []).append(pool.pool_id)
    assert {len(v) for v in reach.values()} == {len(campaign.REGIONS)}
    assert len(reach) == replicas
    # Each pipe carries only its own node's sessions, so aggregate fleet egress
    # scales with the node count rather than being pinned to one pipeline.
    rates = {link.link_id: link.bytes_per_s for link in scenario.links}
    for pool in architecture.pools:
        assert rates[pool.route[0]] == pytest.approx(
            fits[pool.pool_id.split("/")[1]]["effective_pipeline_mbps"]["natural"]
            * 125_000)


# Claims: each cell reports the largest shed executed inside every contract,
# seeds aggregate by median, and the advantage table measures what multiple
# actions buy over the best single-action baseline.  Plausible wrong
# implementations: taking the max over seeds so one lucky seed decides the
# headline; counting shed from rows that broke a contract; comparing the
# flexible policy against the mean rather than the best single action.
def _mini_campaign(tmp_path, rows):
    manifest = {
        "schema": campaign.SCHEMA, "claim": "test", "sessions": 4,
        "shards": 1, "window_s": 5, "source_site": "s", "sites": {"e": "e"},
        "envelope": {"normal": {"rps": 5.0, "ttft_slo_s": 2.0,
                                "right_censored": False},
                     "emergency": {"rps": 8.0, "ttft_slo_s": 10.0,
                                   "right_censored": True}},
        "inputs": {}, "git_sha": "cafe" * 10,
        "rows": [{"row_id": i, "deadline_s": 300.0,
                  "policy": row.get("policy", "greedy"),
                  "mode": "normal", "tier": "natural",
                  "workload": campaign.HEADLINE_WORKLOAD,
                  "sessions": campaign.SESSIONS,
                  "rho": row.get("rho", 0.45), "seed": row["seed"],
                  "headline": row.get("headline", True)}
                 for i, row in enumerate(rows)],
    }
    (tmp_path / "plan.json").write_text(json.dumps(manifest))
    full = [{**manifest["rows"][i], "git_sha": manifest["git_sha"],
             "executed_shed_w": 1000 * row["executed_shed_fraction"],
             "destination_offered_rps": 4.0, "within_envelope": True,
             "committed_kv_fraction": row.get("committed_kv_fraction", 0.0),
             **{f"{region}_{method}": 0 for region in campaign.REGIONS
                for method in ("replay", "kv_transfer")},
             **row}
            for i, row in enumerate(rows)]
    campaign.write_csv(tmp_path / "shard-00.csv", full)
    return manifest


def test_reduce_medians_executed_shed_and_ignores_broken_contracts(tmp_path):
    _mini_campaign(tmp_path, [
        {"seed": 1, "executed_shed_fraction": 0.50, "within_contract": True},
        {"seed": 2, "executed_shed_fraction": 0.90, "within_contract": True},
        {"seed": 3, "executed_shed_fraction": 0.99, "within_contract": False},
    ])

    campaign.reduce(tmp_path)

    row = campaign._csv(tmp_path / "frontier.csv")[0]
    # Seed 3 broke a contract, so its 0.99 counts as nothing shed; the headline
    # is the median of (0.50, 0.90, 0.00), never the 0.99.
    assert float(row["median_executed_shed_fraction"]) == pytest.approx(0.50)
    assert float(row["max_executed_shed_fraction"]) == pytest.approx(0.90)
    assert int(row["contracts_met"]) == 2


def test_reduce_scores_multi_action_against_the_best_single_action(tmp_path):
    _mini_campaign(tmp_path, [
        {"seed": 1, "policy": "queue_haul", "executed_shed_fraction": 0.70,
         "within_contract": True, "committed_kv_fraction": 0.4},
        {"seed": 1, "policy": "replay_only", "executed_shed_fraction": 0.53,
         "within_contract": True},
        {"seed": 1, "policy": "kv_only", "executed_shed_fraction": 0.58,
         "within_contract": True},
    ])

    campaign.reduce(tmp_path)

    row = campaign._csv(tmp_path / "multi_action_advantage.csv")[0]
    # Gain is measured against the BEST single action (kv_only 0.58), not the
    # weaker replay_only or an average of the two.
    assert float(row["best_single_action"]) == pytest.approx(0.58)
    assert float(row["best_flexible"]) == pytest.approx(0.70)
    assert float(row["multi_action_gain"]) == pytest.approx(0.12)
    assert row["best_flexible_policy"] == "queue_haul"


def test_reduce_rejects_stale_or_mixed_shards(tmp_path):
    rows = [
        {"seed": 1, "executed_shed_fraction": 0.5, "within_contract": True},
        {"seed": 2, "executed_shed_fraction": 0.4, "within_contract": True},
    ]
    _mini_campaign(tmp_path, rows)
    shard = tmp_path / "shard-00.csv"
    good = shard.read_text()

    shard.write_text(good.replace("cafe" * 10, "dead" * 10))
    with pytest.raises(RuntimeError, match="commit"):
        campaign.reduce(tmp_path)

    lines = good.splitlines()
    shard.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        campaign.reduce(tmp_path)

    shard.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="every headline row"):
        campaign.reduce(tmp_path)

    shard.write_text(good.replace("greedy", "queue_haul"))
    with pytest.raises(RuntimeError, match="match the manifest"):
        campaign.reduce(tmp_path)


def test_reduce_accepts_absent_but_not_partial_sensitivity(tmp_path):
    rows = [
        {"seed": 1, "executed_shed_fraction": 0.5, "within_contract": True},
        {"seed": 1, "executed_shed_fraction": 0.2, "within_contract": True,
         "headline": False},
        {"seed": 1, "executed_shed_fraction": 0.6, "within_contract": True,
         "headline": False},
    ]
    _mini_campaign(tmp_path, rows)
    shard = tmp_path / "shard-00.csv"
    lines = shard.read_text().splitlines()

    # Headline complete, sensitivity absent: a valid first phase.
    shard.write_text("\n".join(lines[:2]) + "\n")
    assert campaign.reduce(tmp_path)["rows"] == 1

    # One of two sensitivity rows is a mixed, partial reduction.
    shard.write_text("\n".join(lines[:3]) + "\n")
    with pytest.raises(RuntimeError, match="sensitivity"):
        campaign.reduce(tmp_path)
