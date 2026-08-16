"""
Claim:
The headline curve reports, at each measured total-work level, the worst
directional median for each latency metric divided by that metric's own SLO.

Plausible wrong implementations:
- Average directions and hide a TPOT violation in one workload mix.
- Use one direction's values for both TTFT and TPOT.
- Divide TPOT by the TTFT target or subtract the nominal baseline.
- Pool rows across different total-work levels.
"""

import pytest

import plot_service_headroom as plotter


def test_aggregate_keeps_raw_block_range():
    rows = []
    for direction in ("baseline", "prefill_heavy", "decode_heavy"):
        rho = .25 if direction == "baseline" else .5
        for block, ttft in enumerate((.1, .2, .3)):
            rows.append({
                "direction": direction, "target_rho": rho,
                "offered_rho": rho + block / 1000,
                "p90_ttft_s": ttft, "p90_mean_tpot_s": ttft / 10,
                "stable": block != 2 or direction != "decode_heavy",
            })
    aggregated = plotter.aggregate({"hardware": "a100", "rows": rows})
    prefill = next(row for row in aggregated
                   if row["direction"] == "prefill_heavy"
                   and row["target_rho"] == .5)
    decode = next(row for row in aggregated
                  if row["direction"] == "decode_heavy"
                  and row["target_rho"] == .5)

    assert prefill["p90_ttft_s_median"] == .2
    assert prefill["p90_ttft_s_minimum"] == .1
    assert prefill["p90_ttft_s_maximum"] == .3
    assert prefill["physically_feasible"]
    assert not decode["physically_feasible"]


def test_pool_slo_envelope_keeps_opposite_direction_violations():
    rows = [
        {"direction": "prefill_heavy", "target_rho": .25,
         "measured_rho_median": .20, "p90_ttft_s_median": .1,
         "p90_mean_tpot_s_median": .02},
        {"direction": "decode_heavy", "target_rho": .25,
         "measured_rho_median": .26, "p90_ttft_s_median": .2,
         "p90_mean_tpot_s_median": .03},
        {"direction": "prefill_heavy", "target_rho": .5,
         "measured_rho_median": .54, "p90_ttft_s_median": .8,
         "p90_mean_tpot_s_median": .12},
        {"direction": "decode_heavy", "target_rho": .5,
         "measured_rho_median": .56, "p90_ttft_s_median": 1.2,
         "p90_mean_tpot_s_median": .04},
    ]

    pooled = plotter.pool_slo_envelope(
        rows, {"p90_ttft_s": 1, "p90_mean_tpot_s": .1})

    assert pooled[0]["added_rho"] == 0
    assert pooled[1]["added_rho"] == pytest.approx(.32)
    assert pooled[1]["p90_ttft_s_slo_ratio"] == 1.2
    assert pooled[1]["p90_mean_tpot_s_slo_ratio"] == 1.2


def test_main_plot_renders_pooled_slo_curve(tmp_path):
    rows = [{
        "direction": direction, "target_rho": rho,
        "measured_rho_median": rho,
        "p90_ttft_s_median": .4,
        "p90_mean_tpot_s_median": .04 if rho == .25 else .12,
    } for rho in (.25, .5)
        for direction in ("prefill_heavy", "decode_heavy")]
    scout = {"targets": {"p90_ttft_s": 1, "p90_mean_tpot_s": .1}}
    out = tmp_path / "service-headroom"

    plotter.plot(rows, scout, out)

    assert out.with_suffix(".pdf").is_file()
    assert out.with_suffix(".png").is_file()


def test_transition_uses_worse_cohort_and_observed_block_range():
    rows = []
    for block, incumbent, admitted in (
            (6, .20, .22), (7, .21, .19), (8, .18, .23)):
        rows.append({
            "direction": "prefill_heavy", "block": block,
            "offered_coordinates": {"offered_rho": .5},
            "windows": {"post_admission": {
                "p90_ttft_s": incumbent, "p90_mean_tpot_s": incumbent / 5,
            }},
            "new_cohort": {
                "p90_ttft_s": admitted, "p90_mean_tpot_s": admitted / 5,
            },
        })
    for direction in ("balanced", "decode_heavy"):
        rows.extend({**row, "direction": direction} for row in rows[:3])
    result = {"campaign_pass": True, "planner_usable": False, "rows": rows}

    reduced = plotter.aggregate_transition(result, .25)
    prefill = next(row for row in reduced
                   if row["direction"] == "prefill_heavy")

    assert prefill["added_work"] == .25
    assert prefill["p90_ttft_s_median"] == .22
    assert prefill["p90_ttft_s_minimum"] == .21
    assert prefill["p90_ttft_s_maximum"] == .23
