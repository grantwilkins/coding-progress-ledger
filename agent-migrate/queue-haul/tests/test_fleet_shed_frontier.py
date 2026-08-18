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
