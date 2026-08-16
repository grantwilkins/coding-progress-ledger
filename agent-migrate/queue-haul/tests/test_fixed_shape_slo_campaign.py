"""The fixed-shape sweep preserves load and boundary semantics."""

import pytest

import fixed_shape_slo_campaign as campaign


def result(rate, replicate=0, violation=False, ttft=.5, tpot=.05):
    return {"offered_rps": rate, "replicate": replicate,
            "slo_violation": violation, "exact_completion_rate": 1,
            "p90_ttft_s": ttft, "p90_mean_tpot_s": tpot}


def test_plan_freezes_exact_shape_poisson_rates_and_unlimited_concurrency():
    plan = campaign.make_plan("openai/gpt-oss-20b", .8, .08, 7)

    assert plan["rates_rps"] == [.125, .25, .5, 1, 2, 4, 8]
    assert (plan["input_tokens"], plan["output_tokens"],
            plan["requests_per_point"], plan["max_concurrency"]) == (3920, 1024, 32, None)
    assert [row["poisson_seed"] for row in plan["base_cells"]] == list(range(7, 14))
    assert len(campaign.offered_trace(.125, 7)) == 32
    assert campaign.offered_trace(.125, 7) == campaign.offered_trace(.125, 7)


def test_boundary_uses_first_combined_slo_violation_and_predecessor():
    rows = [result(rate, violation=rate >= 2) for rate in campaign.RATES]
    assert campaign.first_boundary(rows) == (1, 2)
    with pytest.raises(RuntimeError, match="not bracketed"):
        campaign.first_boundary([result(rate) for rate in campaign.RATES])
    with pytest.raises(RuntimeError, match="not bracketed"):
        campaign.first_boundary([result(rate, violation=True) for rate in campaign.RATES])


def test_boundary_aggregation_reports_three_run_min_max_whiskers():
    plan = campaign.make_plan("openai/gpt-oss-20b")
    rows = [result(rate, violation=rate >= 2,
                   ttft=1.3 if rate == 2 else .5,
                   tpot=.13 if rate == 2 else .05)
            for rate in campaign.RATES]
    rows += [result(1, repeat, ttft=value, tpot=value / 10)
             for repeat, value in ((1, .4), (2, .6))]
    rows += [result(2, repeat, True, value, value / 10)
             for repeat, value in ((1, 1.2), (2, 1.4))]

    curve = campaign.aggregate(rows, plan)

    predecessor = next(row for row in curve if row["offered_rps"] == 1)
    violation = next(row for row in curve if row["offered_rps"] == 2)
    assert predecessor["replicates"] == violation["replicates"] == 3
    assert (predecessor["p90_ttft_s_min"], predecessor["p90_ttft_s_max"]) == (.4, .6)
    assert (violation["p90_ttft_s_min"], violation["p90_ttft_s_max"]) == (1.2, 1.4)
