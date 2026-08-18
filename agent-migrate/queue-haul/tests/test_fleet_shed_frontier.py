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
    return WorkloadProfile.load(campaign.WORKLOAD)


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
        fits, campaign.RHO_DEST, 0.05, contexts)
    limit = ProfileRateLimit(
        f"{campaign.MODEL_ID}-a100-tp1/east", campaign.REF_CONTEXT,
        campaign.PROMPT, campaign.OUTPUT, campaign.RHO_DEST * 5.0, 5.0)

    conversion = limit.conversion(architecture.types[0])

    assert conversion["safe_service_bound"] == pytest.approx(bound)
    assert conversion["baseline_work"] == pytest.approx(
        tuple(campaign.RHO_DEST * 5.0 * campaign.request_work(case)))


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


def test_every_row_requests_an_attainable_fraction_not_the_idle_floor():
    rows = campaign.manifest_rows()

    assert rows and len({row["row_id"] for row in rows}) == len(rows)
    assert {row["requested_fraction"] for row in rows} == set(campaign.TARGETS)
    # Requesting the whole removable band puts the limit at the idle floor,
    # which is the infeasible-fallback case this campaign exists to avoid.
    assert max(row["requested_fraction"] for row in rows) < 1.0
    assert {row["policy"] for row in rows} == set(campaign.POLICIES)
    # The target-first LP is only meaningful against an attainable request.
    assert campaign.POLICIES["queue_haul"] == "lp_work_first"


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
        fits, campaign.RHO_DEST, 0.05, contexts)

    # A scalar prefill floor was tried and rejected: it tripled the error
    # against outputs/timing-power-validation-20260814, the artifact these
    # factors were fitted on.
    for region, destination_type in zip(campaign.REGIONS, architecture.types):
        assert destination_type.migration["replay"].compute_completion_factor \
            == pytest.approx(fits[region]["replay_compute_completion_factor"])
        assert destination_type.migration["kv_transfer"].compute_completion_factor == 1


def test_calibration_delivers_the_requested_shed(profile, workload):
    """A request is answered with that much power, not with whatever the
    credit target happens to imply."""
    case = profile.case()
    bound = 5.0 * campaign.request_work(case).sum()
    contexts = sorted({record.context_tokens for record in workload.records})
    scenario, replicas, demand, fits = campaign.build_fleet(
        profile, workload, 400, 1001, 600.0, bound, "natural")
    architecture = campaign.build_architecture(
        profile, replicas, {"normal": bound, "emergency": bound, "stable": bound},
        fits, campaign.RHO_DEST,
        campaign.migration_headroom(campaign.RHO_DEST, demand, replicas, bound),
        contexts)
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    removable = initial - source_power(
        scenario, profile, [s.session_id for s in scenario.sessions])
    goal = 0.25 * removable

    _, certified, ask, steps = campaign.calibrated_plan(
        scenario, profile, architecture, "lp_work_first", 1, "normal", power,
        initial, goal)

    assert certified == pytest.approx(goal, rel=campaign.CALIBRATION_TOLERANCE)
    assert 1 <= steps <= campaign.CALIBRATION_STEPS
    # The uncalibrated request would have been answered with a different shed.
    assert ask != pytest.approx(goal, rel=1e-6) or steps == 1


def test_calibration_bisects_through_a_local_plateau(monkeypatch):
    """Two equal outcomes inside the ladder are a step of the policy's answer,
    not proof the goal is unreachable; stopping there marks a reachable target
    missed while a higher request would still pass."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Snapshot:
        power_limit_w: float

    def stepped(scenario, *args, **kwargs):
        ask = 100.0 - scenario.power_limit_w
        shed = 0 if ask < 10 else 48 if ask < 60 else 50
        return SimpleNamespace(
            moves=tuple(SimpleNamespace(session_id=str(k))
                        for k in range(shed)))

    monkeypatch.setattr(campaign, "plan", stepped)
    power = SimpleNamespace(drain_gain=lambda ids: float(len(list(ids))))

    _, shed, ask, steps = campaign.calibrated_plan(
        Snapshot(0.0), None, None, "greedy", 1, "normal", power, 100.0, 50.0)
    assert shed == 50.0 and ask >= 60
    assert steps <= campaign.CALIBRATION_STEPS

    # An unreachable goal still exits early once the ask hits the ceiling.
    _, shed, _, steps = campaign.calibrated_plan(
        Snapshot(0.0), None, None, "greedy", 1, "normal", power, 100.0, 80.0)
    assert shed == 50.0 and steps < campaign.CALIBRATION_STEPS


# Claims: the frontier headline is the median over seeds of each seed's
# largest executed, envelope-compliant shed attained by the deadline, and the
# reduction refuses stale or mixed shards.  Plausible wrong implementations:
# taking the max over all rows so one lucky seed decides the headline;
# counting envelope-breaching rows; accepting shards from another commit or
# rows that disagree with the manifest.
def _mini_campaign(tmp_path, rows):
    manifest = {
        "schema": campaign.SCHEMA, "claim": "test", "sessions": 4,
        "shards": 1, "window_s": 5, "source_site": "s", "sites": {"e": "e"},
        "envelope": {"normal": {"rps": 5.0, "ttft_slo_s": 2.0,
                                "right_censored": False},
                     "emergency": {"rps": 8.0, "ttft_slo_s": 10.0,
                                   "right_censored": True}},
        "inputs": {}, "git_sha": "cafe" * 10,
        "rows": [{"row_id": i, "deadline_s": 300.0, "policy": "greedy",
                  "requested_fraction": row["requested_fraction"],
                  "mode": "normal", "tier": "natural", "rho": 0.45,
                  "seed": row["seed"],
                  "headline": row.get("headline", True)}
                 for i, row in enumerate(rows)],
    }
    (tmp_path / "plan.json").write_text(json.dumps(manifest))
    full = [{**manifest["rows"][i], "git_sha": manifest["git_sha"],
             "realized_shed_w": 1000 * row["realized_shed_fraction"],
             "destination_offered_rps": 4.0, **row}
            for i, row in enumerate(rows)]
    campaign.write_csv(tmp_path / "shard-00.csv", full)
    return manifest


def test_reduce_medians_per_seed_executed_shed_not_the_lucky_seed(tmp_path):
    _mini_campaign(tmp_path, [
        {"seed": 1, "requested_fraction": 0.5, "realized_shed_fraction": 0.5,
         "target_met": True, "within_envelope": True},
        {"seed": 1, "requested_fraction": 0.95, "realized_shed_fraction": 0.95,
         "target_met": False, "within_envelope": False},
        {"seed": 2, "requested_fraction": 0.9, "realized_shed_fraction": 0.9,
         "target_met": True, "within_envelope": True},
    ])

    campaign.reduce(tmp_path)

    frontier = campaign._csv(tmp_path / "frontier.csv")
    assert len(frontier) == 1
    row = frontier[0]
    # Seed 1's compliant best is 0.5 (its 0.95 breached the envelope), seed
    # 2's is 0.9: the headline is their median 0.7, not the 0.95 or the 0.9.
    assert float(row["median_executed_shed_fraction"]) == pytest.approx(0.7)
    assert float(row["min_executed_shed_fraction"]) == pytest.approx(0.5)
    assert float(row["max_executed_shed_fraction"]) == pytest.approx(0.9)
    assert float(row["median_max_requested_met"]) == pytest.approx(0.7)


def test_reduce_rejects_stale_or_mixed_shards(tmp_path):
    rows = [
        {"seed": 1, "requested_fraction": 0.5, "realized_shed_fraction": 0.5,
         "target_met": True, "within_envelope": True},
        {"seed": 2, "requested_fraction": 0.5, "realized_shed_fraction": 0.4,
         "target_met": False, "within_envelope": True},
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
        {"seed": 1, "requested_fraction": 0.5, "realized_shed_fraction": 0.5,
         "target_met": True, "within_envelope": True},
        {"seed": 1, "requested_fraction": 0.25, "realized_shed_fraction": 0.2,
         "target_met": True, "within_envelope": True, "headline": False},
        {"seed": 1, "requested_fraction": 0.75, "realized_shed_fraction": 0.6,
         "target_met": True, "within_envelope": True, "headline": False},
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
