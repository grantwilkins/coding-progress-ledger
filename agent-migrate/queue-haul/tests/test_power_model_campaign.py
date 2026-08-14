from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import power_model_campaign as campaign


def test_grid_separates_discovery_and_unseen_confirmation_cells():
    rows = campaign.cells(4)
    discovery = [row for row in rows if row.stage == "discovery"]
    held = [row for row in rows if row.stage == "confirmation"]

    assert len(discovery) == 90
    assert len(held) == 18
    assert rows[0].stage == rows[91].stage == rows[-1].stage == "idle"
    assert {row.concurrency for row in discovery} == {1, 2, 4, 8, 16}
    assert {row.concurrency for row in held} == {3, 6, 12}
    assert (604, 64) in {(row.prompt_tokens, row.output_tokens) for row in held}


def test_followup_randomizes_three_calibration_and_validation_reps_with_idle_brackets():
    rows = campaign.followup_cells(4)

    assert len(rows) == 37
    assert all(row.stage == "idle" for row in rows[::2])
    work = rows[1::2]
    for stage in ("targeted_calibration", "targeted_validation"):
        selected = [row for row in work if row.stage == stage]
        assert {c: sum(row.concurrency == c for row in selected) for c in (3, 6, 12)} \
               == {3: 3, 6: 3, 12: 3}
    assert work != sorted(work, key=lambda row: (row.stage, row.concurrency, row.replicate))


def test_parse_metrics_sums_engines_and_requires_realized_counters():
    text = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
vllm:prompt_tokens_total{engine="0"} 10
vllm:prompt_tokens_total{engine="1"} 20
vllm:generation_tokens_total{engine="0"} 7
vllm:prompt_tokens_cached_total{engine="0"} 0
"""
    assert campaign.parse_metrics(text) == {
        campaign.PROMPT_COUNTER: 30,
        campaign.DECODE_COUNTER: 7,
        campaign.CACHED_COUNTER: 0,
    }
    with pytest.raises(RuntimeError, match="missing vLLM counters"):
        campaign.parse_metrics("vllm:prompt_tokens_total 1")


def synthetic_rows(alpha=1 / 1000, beta=1 / 500):
    rows = []
    for replicate in range(2):
        for family, f, g in (("prefill", 1000, 1), ("decode", 100, 500),
                             ("campaign", 500, 250)):
            z = alpha * f + beta * g
            rows.append({"stage": "discovery", "family": family,
                         "replicate": replicate, "prompt_tokens": 1,
                         "output_tokens": 1, "concurrency": 1,
                         "realized_prefill_tps": f,
                         "realized_decode_tps": g,
                         "power_mean_w": 80 + 220 * z / (1 + z),
                         "cached_prompt_tokens": 0})
    return rows * 15 + [{"stage": "idle", "power_mean_w": 80,
                         "cached_prompt_tokens": 0}] * 3


def test_fit_uses_realized_rates_and_recovers_saturating_model():
    fit = campaign.saturating_fit(synthetic_rows())

    assert fit["alpha_s_per_prefill_token"] == pytest.approx(1 / 1000, rel=.03)
    assert fit["beta_s_per_decode_token"] == pytest.approx(1 / 500, rel=.03)
    assert fit["power_idle_w"] == pytest.approx(80)
    assert fit["power_max_w"] == pytest.approx(300, rel=.03)
    assert fit["link"] == "P=P0+A*z/(1+z); z=alpha*f+beta*g"
    assert "ell_knee" not in fit


def test_fit_result_is_json_serializable():
    rows = synthetic_rows()
    rows += [{**row, "stage": "confirmation"} for row in rows[:18]]

    result = campaign.fit_result(rows)

    json.dumps(result)
    assert all(type(value) is bool for value in result["validation"]["gates"].values())


def test_exponential_provisional_is_json_serializable():
    provisional = campaign.exponential_fit(synthetic_rows())

    json.dumps(provisional)
    assert provisional["status"] == "provisional_reconstructed_after_serialization_failure"


def test_followup_requires_complete_independent_replicates():
    with pytest.raises(RuntimeError, match="three independent reps"):
        campaign.followup_result(synthetic_rows(), [])


def test_offline_refit_hard_fails_incomplete_grid(tmp_path):
    (tmp_path / "cells.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="requires all 111 cells"):
        campaign.complete_rows(tmp_path, 20260814)


def usage(prompt=10, decode=2, cached=0):
    return {"usage": {"prompt_tokens": prompt, "completion_tokens": decode,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def counters(prompt=20, decode=4, cached=0):
    return {campaign.PROMPT_COUNTER: prompt, campaign.DECODE_COUNTER: decode,
            campaign.CACHED_COUNTER: cached}


def test_accounting_accepts_only_complete_exact_realized_batches():
    cell = campaign.Cell("discovery", "campaign", 10, 2, 2, 0)

    result = campaign.accounting(cell, [usage(), usage()], 1, counters())

    assert result["realized_prefill_tokens"] == 20
    assert result["realized_decode_tokens"] == 4


@pytest.mark.parametrize(("requests", "batches", "metric", "match"), (
    ([usage()], 1, counters(10, 2), "completed 1 of 2"),
    ([usage(cached=1), usage()], 1, counters(cached=1), "cached prompt"),
    ([usage(), usage()], 1, counters(30, 4), "counter/API disagreement"),
))
def test_accounting_hard_fails_incomplete_cached_or_disagreeing_work(
        requests, batches, metric, match):
    cell = campaign.Cell("discovery", "campaign", 10, 2, 2, 0)
    with pytest.raises(RuntimeError, match=match):
        campaign.accounting(cell, requests, batches, metric)


def test_nonidle_zero_work_hard_fails():
    cell = campaign.Cell("discovery", "campaign", 0, 0, 1, 0)
    with pytest.raises(RuntimeError, match="zero work"):
        campaign.accounting(cell, [usage(0, 0)], 1, counters(0, 0))


def resume_fixture(tmp_path, monkeypatch):
    plan = campaign.cells(7)
    gpu = {"name": "NVIDIA H100 NVL", "uuid": "GPU-x", "power_limit_w": 400.0}
    sha = "a" * 40
    out = tmp_path / "run"
    power = out / "power"
    power.mkdir(parents=True)
    path = power / f"{campaign.cell_label(0, plan[0])}.csv"
    path.write_text("monotonic_ns,power_w\n" + "".join(f"{i},100\n" for i in range(60)))
    row = {**campaign.asdict(plan[0]), "sequence": 0, "start_ns": 0,
           "end_ns": 12_000_000_000, "window_s": 12, "batches": 0,
           "realized_prefill_tokens": 0, "realized_decode_tokens": 0,
           "realized_prefill_tps": 0, "realized_decode_tps": 0,
           "reported_prompt_tokens": 0, "counter_tolerance_tokens": 1,
           "counter_tolerance_fraction": .001, "power_mean_w": 100,
           "power_p50_w": 100,
           "cached_prompt_tokens": 0, "power_samples": 60,
           "request_count": 0, "completed_requests": 0, "power_path": str(path)}
    (out / "cells.jsonl").write_text(json.dumps(row) + "\n")
    (out / "requests.jsonl").write_text("")
    (out / "metadata.json").write_text(json.dumps({"gpu": gpu, "git_sha": sha,
        "minimum_window_s": 12, "warmup": "one complete batch", "cooldown_s": 2,
        "seed": 7}))
    model = tmp_path / "model"
    (out / "server.log").write_text(str(model))
    orphan = power / f"{campaign.cell_label(1, plan[1])}.csv"
    orphan.write_text("partial")
    args = SimpleNamespace(out=out, model=model, expected_sha=sha, window_s=12,
                           cooldown_s=2, seed=7, discard_orphan_sequences=[1])
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_a, **_k: "b" * 40 + "\n")
    return args, gpu, plan, orphan


def test_resume_validates_prefix_and_replaces_only_explicit_orphan(tmp_path, monkeypatch):
    args, gpu, plan, orphan = resume_fixture(tmp_path, monkeypatch)
    before = [(args.out / name).read_bytes() for name in ("metadata.json", "cells.jsonl",
                                                           "requests.jsonl")]

    rows = campaign.validate_resume(args, gpu, plan)

    assert len(rows) == 1
    assert not orphan.exists()
    assert before == [(args.out / name).read_bytes() for name in
                      ("metadata.json", "cells.jsonl", "requests.jsonl")]
    assert json.loads((args.out / "resumes.jsonl").read_text())["next_sequence"] == 1


def test_resume_rejects_mismatched_prefix_or_unlisted_artifact(tmp_path, monkeypatch):
    args, gpu, plan, _ = resume_fixture(tmp_path, monkeypatch)
    row = json.loads((args.out / "cells.jsonl").read_text())
    row["family"] = "wrong"
    (args.out / "cells.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="deterministic prefix"):
        campaign.validate_resume(args, gpu, plan)

    args, gpu, plan, _ = resume_fixture(tmp_path / "second", monkeypatch)
    (args.out / "power" / "unknown.csv").write_text("x")
    with pytest.raises(RuntimeError, match="power artifacts"):
        campaign.validate_resume(args, gpu, plan)


def test_committed_request_evidence_must_stay_inside_boundary_and_uncached():
    row = {"start_ns": 10, "end_ns": 20, "request_count": 1,
           "realized_prefill_tokens": 10, "realized_decode_tokens": 2}
    evidence = [{"start_ns": 10, "end_ns": 20, **usage()}]
    campaign.validate_request_evidence("cell", row, evidence)

    evidence[0]["end_ns"] = 21
    with pytest.raises(RuntimeError, match="crossed a committed boundary"):
        campaign.validate_request_evidence("cell", row, evidence)
    evidence[0]["end_ns"] = 20
    evidence[0]["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    with pytest.raises(RuntimeError, match="token/cache"):
        campaign.validate_request_evidence("cell", row, evidence)


def test_committed_row_recomputes_window_rates_counters_and_power():
    row = {"sequence": 1, "start_ns": 0, "end_ns": 2_000_000_000,
           "window_s": 2, "realized_prefill_tokens": 20,
           "realized_decode_tokens": 4, "realized_prefill_tps": 10,
           "realized_decode_tps": 2, "reported_prompt_tokens": 20,
           "counter_tolerance_tokens": 1, "counter_tolerance_fraction": .001,
           "power_mean_w": 20, "power_p50_w": 20}
    campaign.validate_row_numbers(row, [10, 20, 30])

    row["reported_prompt_tokens"] = 22
    with pytest.raises(RuntimeError, match="counter/API"):
        campaign.validate_row_numbers(row, [10, 20, 30])
    row["reported_prompt_tokens"] = 20
    row["power_mean_w"] = 21
    with pytest.raises(RuntimeError, match="power reduction"):
        campaign.validate_row_numbers(row, [10, 20, 30])
