"""
Claim:
Paired fixed-width A100 episodes identify a relative Replay endpoint slowdown
under prefill-heavy destination load without replacing the regional idle anchor.

Plausible wrong implementations:
- Weight episodes instead of load cells and let one outlier set the slope.
- Fit planner-selected variable-width rows above the fixed-eight support.
- Apply the width-eight intercept to concurrency-one regional timing.
- Drop adverse 1-Gbit/s validation episodes or duplicate 10-Gbit/s cells.
- Claim a positive KV load effect when its paired bootstrap spans zero.
"""

import math
from copy import deepcopy

import pytest

from loaded_service_model import (fit_evidence, fit_log_cells, load_evidence,
                                  validate_model)


def test_cell_median_fit_recovers_endpoint_factor_despite_outlier():
    rows = []
    route, work, switch, kappa, beta = 2, 4, .1, .75, .3
    for rho in (0, .5, 1):
        commit = route + kappa * math.exp(beta * rho) * work + switch
        values = (commit, commit, commit * 20 if rho == 1 else commit)
        rows.extend({"rho": rho, "commit_s": value} for value in values)

    fitted = fit_log_cells(rows, "commit_s", route, work, switch)

    assert fitted == pytest.approx((kappa, beta))


def test_retained_hardware_evidence_is_fixed_width_unique_and_adverse():
    training, validation, provenance = load_evidence()

    assert len(training) == 160 and len(validation) == 440
    assert {row["method"] for row in training + validation} == {
        "replay", "kv_transfer",
    }
    assert max(row["rho"] for row in training) < .9
    assert min(row["bandwidth_mbps"] for row in validation) == 1000
    assert len({row["scenario_id"] for row in validation}) == 440
    assert not {row["scenario_id"] for row in training} & {
        row["scenario_id"] for row in validation
    }
    assert any(row["bandwidth_mbps"] == 1000
               and row["method"] == "replay"
               and row["resume_s"] > 25 for row in validation)
    assert provenance["training_episodes"] == 160
    assert provenance["validation_episodes"] == 440


def test_real_fit_preserves_idle_anchor_and_identifies_only_replay():
    model, _ = fit_evidence(samples=100, seed=7)
    selected = model["selected_commit_log_slope_per_rho"]
    fitted = model["fitted_commit_log_slope_per_rho"]
    bootstrap = model["bootstrap"]

    assert .25 < selected["replay"] < .35
    assert selected["kv_transfer"] == 0
    assert 0 < fitted["kv_transfer"] < .1
    assert bootstrap["replay"]["positive_fraction"] == 1
    assert bootstrap["kv_transfer"]["p05"] < 0 < bootstrap[
        "kv_transfer"]["p95"]
    assert model["slowdown_at_rho_0"]["replay"] == 1
    assert model["slowdown_at_rho_0"]["kv_transfer"] == 1
    assert model["slowdown_at_rho_0_95"]["replay"] > 1.25
    assert model["slowdown_at_rho_0_95"]["kv_transfer"] == 1
    validation = model["width8_relative_factor_validation"]
    assert validation["replay"]["p90_absolute_percentage_error"] < .1
    assert validation["replay"]["false_feasible_25s"] == 0
    assert validation["kv_transfer"]["false_feasible_25s"] == 0


def test_artifact_validator_rejects_a_tampered_curve():
    model, provenance = fit_evidence(samples=20, seed=7)
    curve = deepcopy(model)
    curve["slowdown"]["replay"][1] = 1

    with pytest.raises(ValueError, match="loaded-service model"):
        validate_model({**curve, "provenance": provenance})

    slope = deepcopy(model)
    slope["selected_commit_log_slope_per_rho"]["replay"] = .01
    slope["slowdown"]["replay"] = [
        math.exp(.01 * rho) for rho in slope["rho_grid"]]
    with pytest.raises(ValueError, match="provenance"):
        validate_model({**slope, "provenance": provenance})

    support = deepcopy(model)
    support["fit_context_tokens"] = [1, 1_000_000]
    with pytest.raises(ValueError, match="loaded-service model"):
        validate_model({**support, "provenance": provenance})

    roundoff = deepcopy(model)
    roundoff["selected_commit_log_slope_per_rho"]["replay"] += 1e-15
    roundoff["slowdown"]["replay"] = [
        math.exp(roundoff["selected_commit_log_slope_per_rho"]["replay"] * rho)
        for rho in roundoff["rho_grid"]]
    validate_model({**roundoff, "provenance": provenance})
