"""
Claim:
Mean prefill/decode work does not determine request-shape latency, and the
current request simulator cannot validate decode hold or token-gap ITL.

Plausible wrong implementations:
- Change total tokens or average power between an A/B pair.
- Split individual requests, rather than contexts, across holdout folds.
- Count queued requests as active decodes.
- Treat aggregate saturation throughput as a single-request token rate.
- Label pooled token gaps or mean decode duration as per-request TPOT/TBT.
"""

from pathlib import Path

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
