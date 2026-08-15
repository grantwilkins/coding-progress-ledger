"""
Claim:
The action bootstrap uses balanced timing-cell resamples and paired joint power
draws, while every replicate accounts for all 28 fixed-pack sessions.

Plausible wrong implementations:
- Pool regions, methods, bandwidths, or contexts while resampling timing.
- Draw power coefficients independently instead of using a joint tuple.
- Give different calibration draws to matched constraint cases.
- Normalize by moved sessions or lose sessions from the three action classes.
"""

from types import SimpleNamespace

import numpy as np

import bootstrap_action_adaptation as bootstrap
from profiles import ModelProfile


def timing_rows():
    return [{"destination": destination, "method": method,
             "bandwidth": bandwidth, "context_tokens": context}
            for destination in ("east", "germany")
            for method in ("replay", "kv_transfer")
            for bandwidth in ("controlled_40", "controlled_80", "natural")
            for context in ("1536", "7680", "32256")
            for _ in range(3)]


def test_timing_bootstrap_preserves_every_three_repeat_cell():
    sampled = bootstrap.stratified_timing_bootstrap(
        timing_rows(), np.random.default_rng(1))
    cells = {}
    for row in sampled:
        key = tuple(row[name] for name in (
            "destination", "method", "bandwidth", "context_tokens"))
        cells.setdefault(key, []).append(row)
    assert len(sampled) == 108 and len(cells) == 36
    assert {len(rows) for rows in cells.values()} == {3}


def test_action_row_conserves_the_fixed_pack():
    result = SimpleNamespace(
        moves=[SimpleNamespace(destination_instance="east", method="replay"),
               SimpleNamespace(destination_instance="germany", method="kv_transfer")],
        power_shortfall_w=0)
    row = bootstrap.action_row("case", "Case", 0, "joint", result, 28, 10, 3)
    assert (row["replay_count"], row["kv_transfer_count"],
            row["not_moved_count"]) == (1, 1, 26)
    assert row["replay"] + row["kv_transfer"] + row["not_moved"] == 1


def test_summary_uses_hand_checked_distribution_quantiles():
    rows = []
    for mode in bootstrap.MODES:
        for case, label in bootstrap.ACTION_MIX_CASES:
            for replicate, replay in enumerate((0, .25, .5, .75, 1)):
                rows.append({"mode": mode, "case_id": case,
                             "bound_constraint": label, "replicate": replicate,
                             "replay": replay, "kv_transfer": 0,
                             "not_moved": 1 - replay, "target_met": True})
    summary = bootstrap.summarize(rows)
    replay = next(row for row in summary if row["mode"] == "joint"
                  and row["bound_constraint"] == "HBM"
                  and row["action"] == "replay")
    assert replay["median"] == .5
    assert replay["p25"] == .25 and replay["p75"] == .75


def test_power_draw_uses_one_joint_tuple_and_refreshes_maximum():
    profile = ModelProfile.load(bootstrap.PROFILE)
    expected = profile.case().phase_power.bootstrap[17]
    varied = bootstrap.power_draw(profile, 17)
    phase = varied.case().phase_power
    assert (phase.p0_w, phase.delta_w, phase.a_s_per_prefill_token,
            phase.b_s_per_decode_token) == expected
    assert varied.max_power_load == max(
        phase.load(*point) for point in phase.valid_hull)
