"""
Claim:
Mean prefill/decode work does not determine request-shape latency, and the
current request simulator cannot validate decode hold or TBT.

Plausible wrong implementations:
- Change total tokens or average power between an A/B pair.
- Split individual requests, rather than contexts, across holdout folds.
- Count queued requests as active decodes.
- Treat aggregate saturation throughput as a single-request token rate.
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
    assert len({round(row["p95_modeled_tbt_ms"], 12) for row in rows.values()}) == 1
    assert rows["burst_long_first"]["p95_ttft_ms"] > rows["burst_long_last"]["p95_ttft_ms"]


def test_context_holdout_is_disjoint_and_hold_proxy_catches_more_normal_failures():
    rows = analysis.read_rows(DECODE)
    result = analysis.decode_holdout(rows, ModelProfile.load(PROFILE))
    models = result["models"]
    predictions = models["observed_iteration_proxy"]["predictions"]

    assert all(row["heldout_context_tokens"] not in row["train_context_tokens"]
               for row in predictions)
    assert models["work_only"]["normal"]["false_feasible"] == 4
    assert models["observed_iteration_proxy"]["normal"]["false_feasible"] == 1


def test_profile_rate_is_not_single_request_tbt_evidence():
    rows = analysis.read_rows(DECODE)
    mismatch = analysis.rate_mismatch(rows, ModelProfile.load(PROFILE))

    assert {row["context_tokens"] for row in mismatch} == {256, 4096, 8192}
    assert min(row["observed_to_modeled_ratio"] for row in mismatch) > 10


def test_staircase_keeps_ttft_and_tbt_in_the_same_slo_gate():
    summary = analysis.staircase_summary(analysis.read_rows(PREFILL),
                                         analysis.read_rows(DECODE))
    rows = {row["context_tokens"]: row for row in summary["decode"]}

    assert rows[256]["normal_joint"]["max_tested_concurrency"] == 256
    assert rows[4096]["normal_tbt"]["max_tested_concurrency"] == 64
    assert rows[4096]["normal_joint"]["max_tested_concurrency"] == 4
    assert rows[8192]["normal_joint"]["max_tested_concurrency"] == 2
