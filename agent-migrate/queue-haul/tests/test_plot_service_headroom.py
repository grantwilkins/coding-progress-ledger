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
