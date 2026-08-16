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
