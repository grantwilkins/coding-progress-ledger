"""
Claim:
Mean prefill/decode work does not determine request-shape latency, and the
current request simulator cannot validate decode hold or token-gap ITL. The
empirical reference reports joint request attainment over offered requests and
keeps per-request TPOT separate from streamed-event gaps.

Plausible wrong implementations:
- Change total tokens or average power between an A/B pair.
- Split individual requests, rather than contexts, across holdout folds.
- Count queued requests as active decodes.
- Treat aggregate saturation throughput as a single-request token rate.
- Label streamed-event gaps or their mean as per-request TPOT/TBT.
- Drop failed requests from the SLO denominator.
- Pool token gaps before computing the per-request TPOT quantile.
"""

import json
from pathlib import Path

import pytest

import service_holdout_analysis as analysis
from profiles import ModelProfile


ROOT = Path(__file__).parents[1]
PREFILL = ROOT / "outputs/stage1_gpt_oss_20b_a100_tp1_eager_quick2_prefill_rho.csv"
DECODE = ROOT / "outputs/stage1_gpt_oss_20b_a100_tp1_eager_quick2_decode_context.csv"
PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"


def test_matched_shapes_expose_only_serial_ttft_queueing():
    result = analysis.simulation_ab(ModelProfile.load(PROFILE))
    rows = {row["name"]: row for row in result["rows"]}

    assert result["matched_p_d"] and result["matched_modeled_power"]
    assert rows["burst_uniform"]["p95_ttft_ms"] > 100 * rows["smooth_uniform"]["p95_ttft_ms"]
    assert {row["peak_active_decode"] for row in rows.values()} == {1}
    assert len({round(row["p95_modeled_decode_ms_per_output_token"], 12)
                for row in rows.values()}) == 1
    assert rows["burst_long_first"]["p95_ttft_ms"] > rows["burst_long_last"]["p95_ttft_ms"]


def test_context_retrospective_is_disjoint_and_exposes_leakage():
    rows = analysis.read_rows(DECODE)
    result = analysis.decode_retrospective(rows, ModelProfile.load(PROFILE))
    models = result["models"]
    predictions = models["concurrency_throughput_proxy"]["predictions"]

    assert all(row["heldout_context_tokens"] not in row["train_context_tokens"]
               for row in predictions)
    assert "profile normalization also includes heldout context" in result["limitation"]
    assert models["profile_work_leaky"]["normal"]["false_feasible"] == 4
    assert models["concurrency_throughput_proxy"]["normal"]["false_feasible"] == 1


def test_profile_rate_is_not_single_request_itl_tail_evidence():
    rows = analysis.read_rows(DECODE)
    mismatch = analysis.rate_mismatch(rows, ModelProfile.load(PROFILE))

    assert {row["context_tokens"] for row in mismatch} == {256, 4096, 8192}
    assert min(row["observed_to_modeled_ratio"] for row in mismatch) > 10


def test_staircase_keeps_ttft_and_pooled_itl_in_one_diagnostic_gate():
    summary = analysis.staircase_summary(analysis.read_rows(PREFILL),
                                         analysis.read_rows(DECODE))
    rows = {row["context_tokens"]: row for row in summary["decode"]}

    assert rows[256]["normal_joint"]["safe_cell_concurrency_at_max_output_tps"] == 256
    assert rows[4096]["normal_pooled_itl"]["safe_cell_concurrency_at_max_output_tps"] == 64
    assert rows[4096]["normal_joint"]["safe_cell_concurrency_at_max_output_tps"] == 4
    assert rows[8192]["normal_joint"]["safe_cell_concurrency_at_max_output_tps"] == 2
    safe = rows[4096]["normal_joint"]
    assert safe["planned_token_gaps_if_complete"] == safe["n_requests"] * 511


def test_empirical_slo_uses_offered_requests_and_request_level_tpot(tmp_path):
    path = tmp_path / "cell" / "requests.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "num_prompts": 4, "completed": 3, "request_rate": 1,
        "duration": 4, "input_lens": [10, 20, 30],
        "output_lens": [3, 3, 3], "ttfts": [1, 3, 1],
        "itls": [[.05, .05], [.01, .01], [.2, 0]],
    }))

    row = analysis.empirical_slo_cell(path)

    assert row["joint_request_attainment"] == .5
    assert row["service_failure_rate"] == .25
    assert row["p90_request_mean_tpot_ms"] == pytest.approx(90)
    assert row["p95_stream_event_gap_ms"] == pytest.approx(162.5)
    assert not row["legacy_run_pass"]


def test_empirical_slo_is_invariant_to_trace_duplication(tmp_path):
    raw = {
        "num_prompts": 3, "completed": 2, "request_rate": 1,
        "duration": 3, "input_lens": [10, 20], "output_lens": [3, 3],
        "ttfts": [1, 3], "itls": [[.05, .05], [.01, .01]],
    }
    rows = []
    for copies in (1, 2):
        path = tmp_path / str(copies) / "requests.json"
        path.parent.mkdir()
        payload = {**raw, "num_prompts": raw["num_prompts"] * copies,
                   "completed": raw["completed"] * copies,
                   "input_lens": raw["input_lens"] * copies,
                   "output_lens": raw["output_lens"] * copies,
                   "ttfts": raw["ttfts"] * copies, "itls": raw["itls"] * copies}
        path.write_text(json.dumps(payload))
        rows.append(analysis.empirical_slo_cell(path))

    for key in ("joint_request_attainment", "service_failure_rate"):
        assert rows[0][key] == rows[1][key]


def test_one_token_success_is_ttft_only(tmp_path):
    path = tmp_path / "cell" / "requests.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "num_prompts": 3, "completed": 3, "request_rate": 1,
        "duration": 3, "input_lens": [10, 10, 10],
        "output_lens": [1, 4, 3], "ttfts": [1, 1, 1],
        "itls": [[], [.06, .03], []],
    }))

    row = analysis.empirical_slo_cell(path)

    assert row["decode_metric_eligible_requests"] == 1
    assert row["decode_metric_missing_requests"] == 1
    assert row["joint_request_attainment"] == pytest.approx(2 / 3)
    assert row["p90_request_mean_tpot_ms"] == pytest.approx(30)
    assert not row["legacy_run_pass"]
