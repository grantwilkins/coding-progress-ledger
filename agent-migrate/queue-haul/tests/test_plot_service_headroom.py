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


def test_main_plot_accepts_heldout_repeat_ranges(tmp_path):
    row = {
        "direction": "prefill_heavy", "measured_rho_median": .85,
        "offered_prefill_rho_median": .63,
        "offered_decode_rho_median": .22,
        "p90_ttft_s_median": .4, "p90_ttft_s_minimum": .3,
        "p90_ttft_s_maximum": .5, "p90_mean_tpot_s_median": .105,
        "p90_mean_tpot_s_minimum": .102,
        "p90_mean_tpot_s_maximum": .108,
        "physically_feasible": True, "evidence_feasible": False,
    }
    scout = {"targets": {"p90_ttft_s": 1, "p90_mean_tpot_s": .1}}
    out = tmp_path / "service-headroom"

    plotter.plot([], [row], scout, {"planner_usable": False}, out)

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
