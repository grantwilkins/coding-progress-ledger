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
    assert [[row["poisson_seed"] for row in cells]
            for cells in plan["lower_bracket_cells"]] == [
                [28, 29, 30], [31, 32, 33], [34, 35, 36]]
    assert len(campaign.offered_trace(.125, 7)) == 32
    assert campaign.offered_trace(.125, 7) == campaign.offered_trace(.125, 7)


def test_boundary_uses_first_combined_slo_violation_and_predecessor():
    rows = [result(rate, violation=rate >= 2) for rate in campaign.RATES]
    assert campaign.first_boundary(rows) == (1, 2)
    with pytest.raises(RuntimeError, match="not bracketed"):
        campaign.first_boundary([result(rate) for rate in campaign.RATES])
    assert campaign.first_boundary([
        result(rate, violation=True) for rate in campaign.RATES
    ]) == (None, .125)


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


def test_unbracketed_summary_marks_lowest_pair_as_whiskers(tmp_path):
    plan = campaign.make_plan("openai/gpt-oss-20b")
    rows = [result(rate, violation=True) for rate in (.015625, .03125, *campaign.RATES)]
    summary = campaign.write_summary(tmp_path, plan, rows, (.015625, .03125), {}, False)

    assert summary["boundary"] is None
    assert summary["boundary_bracketed"] is False
    assert summary["whisker_rates_rps"] == [.015625, .03125]


def test_upper_reduction_uses_only_base_rates_without_slo_claim(tmp_path):
    plan = campaign.make_plan("openai/gpt-oss-20b")
    identity = {"sha256": campaign.service.digest({})}
    for expected in plan["base_cells"]:
        path = tmp_path / "base" / expected["cell_id"]
        path.mkdir(parents=True)
        row = {**result(expected["offered_rps"]), **expected,
               "schema": campaign.SCHEMA, "status": "complete",
               "plan_sha256": campaign.service.digest(plan),
               "runtime_identity": identity,
               "runtime_identity_sha256": campaign.service.identity_sha(identity)}
        (path / "result.json").write_text(__import__("json").dumps(row))

    summary = campaign.reduce_upper(tmp_path, plan)

    assert len(summary["curve"]) == len(campaign.RATES)
    assert summary["boundary"] is None and summary["whisker_rates_rps"] == []
    assert summary["ttft_slo_s"] is summary["tpot_slo_s"] is None


def test_knee_plan_requires_bound_explosion_and_freezes_dense_rates(tmp_path):
    plan = campaign.make_plan("openai/gpt-oss-20b")
    (tmp_path / "plan.json").write_text(__import__("json").dumps(plan))
    for rate, ttft, peak in ((.125, 1., 4), (8., 4., 32)):
        path = tmp_path / "base" / campaign.cell(rate, 0, 0, plan["seed"])["cell_id"]
        path.mkdir(parents=True)
        (path / "result.json").write_text(__import__("json").dumps({
            "offered_rps": rate, "status": "complete", "drained": True,
            "exact_completions": 32, "p90_ttft_s": ttft,
            "max_in_system_requests": peak}))

    evidence = campaign.explosion_evidence(tmp_path)
    knee = campaign.make_knee_plan(plan["model"], evidence)

    assert knee["rates_rps"] == list(campaign.KNEE_RATES[plan["model"]])
    assert knee["explosion_evidence"] == evidence
    assert not knee["lower_bracket_cells"] and knee["boundary_repeats"] == 0

    high = tmp_path / "base" / "rps8-rep0" / "result.json"
    row = __import__("json").loads(high.read_text())
    row["p90_ttft_s"] = 3.99
    high.write_text(__import__("json").dumps(row))
    with pytest.raises(RuntimeError, match="does not meet"):
        campaign.explosion_evidence(tmp_path)
