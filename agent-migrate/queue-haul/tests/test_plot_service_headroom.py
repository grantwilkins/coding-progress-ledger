"""
Claim:
The headline figure follows one prefill-heavy workload ray: raw TTFT is plotted
against prefill work and raw TPOT against decode work, with the first measured
SLO violation identified by the total-work parameter rho.

Plausible wrong implementations:
- Plot the decode-heavy ray, which does not exhibit the measured TPOT miss.
- Put total rho on both x axes or swap the prefill/decode coordinates.
- Normalize latency by the SLO instead of preserving raw seconds.
- Mark a later violating sample instead of the first measured miss.
"""

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


def test_workload_panels_keep_raw_phase_coordinates_and_first_miss():
    rows = [
        {"direction": "prefill_heavy", "target_rho": .25,
         "offered_prefill_rho_median": .1,
         "offered_decode_rho_median": .2,
         "p90_ttft_s_median": .2, "p90_mean_tpot_s_median": .02},
        {"direction": "prefill_heavy", "target_rho": .5,
         "offered_prefill_rho_median": .4,
         "offered_decode_rho_median": .3,
         "p90_ttft_s_median": .8,
         "p90_mean_tpot_s_median": .12},
        {"direction": "prefill_heavy", "target_rho": .75,
         "offered_prefill_rho_median": .6,
         "offered_decode_rho_median": .35,
         "p90_ttft_s_median": 1.2, "p90_mean_tpot_s_median": .2},
        {"direction": "decode_heavy", "target_rho": .25,
         "offered_prefill_rho_median": 9,
         "offered_decode_rho_median": 8,
         "p90_ttft_s_median": 7, "p90_mean_tpot_s_median": 6},
    ]

    ttft, tpot = plotter.workload_panels(
        rows, {"p90_ttft_s": 1, "p90_mean_tpot_s": .1})

    assert ttft["x"] == [.1, .4, .6]
    assert ttft["y"] == [.2, .8, 1.2]
    assert ttft["first_miss"]["target_rho"] == .75
    assert tpot["x"] == [.2, .3, .35]
    assert tpot["y"] == [.02, .12, .2]
    assert tpot["first_miss"]["target_rho"] == .5


def test_main_plot_renders_raw_phase_slices(tmp_path):
    rows = [{
        "direction": direction, "target_rho": rho,
        "offered_prefill_rho_median": rho,
        "offered_decode_rho_median": rho / 2,
        "p90_ttft_s_median": .4 if rho == .25 else 1.2,
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
