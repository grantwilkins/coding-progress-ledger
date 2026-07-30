"""
Claim:
The simulated Pareto campaign maximizes attained shed, minimizes completion
time, compares policies only within paired episodes, and exposes interpolation
or extrapolation instead of presenting it as measured.

Plausible wrong implementations:
- Reverse either Pareto objective.
- Let an equal point dominate another point.
- Compare policies from different episodes in the paired result.
- Label non-anchor contexts or contexts outside the measured range as measured.
"""

from simulated_pareto_campaign import (
    context_evidence, measured_replay_caps, meets_deadline, parallel_profile,
    pareto_flags,
)
from test_execution_simulator import model


def test_pareto_direction_and_pairing():
    rows = [
        {"match": "a", "power_attainment_fraction": .8,
         "completion_deadline_ratio": .8},
        {"match": "a", "power_attainment_fraction": .7,
         "completion_deadline_ratio": 1},
        {"match": "a", "power_attainment_fraction": .9,
         "completion_deadline_ratio": 1.2},
        {"match": "b", "power_attainment_fraction": 1,
         "completion_deadline_ratio": .1},
    ]

    pareto_flags(rows, ("match",))

    assert [row["pareto"] for row in rows] == [True, False, True, True]


def test_context_evidence_marks_nonanchors_and_extrapolation():
    anchors = {2048, 4096, 8192, 16384}

    assert context_evidence((2048, 8192), anchors) == "measured"
    assert context_evidence((4096, 12288), anchors) == "interpolated"
    assert context_evidence((1024, 4096), anchors) == "extrapolated"


def test_deadline_boundary_tolerates_roundoff_but_not_real_misses():
    assert meets_deadline(1 - 1e-12, 10 + 1e-12, 10)
    assert not meets_deadline(.99, 9, 10)
    assert not meets_deadline(1, 11, 10)


def test_width8_contract_does_not_silently_serialize_destination(tmp_path):
    base = model(tmp_path)
    context = base.case().replay.by_concurrency[1][0][0]
    serial = base.case().replay.rate(context, 1)
    profile = parallel_profile(base, 8, {"central": serial / 2})

    assert profile.max_destination_replays == 8
    assert profile.max_destination_kv_streams == 8
    assert all(
        curve.concurrency[-1] == 8
        for case in profile.cases.values()
        for curve in case.action_power_w.values()
    )
    assert all(set(case.replay.by_concurrency) == set(range(1, 9))
               for case in profile.cases.values())
    assert profile.case().replay.rate(context, 1) == serial
    assert 8 * profile.case().replay.rate(context, 8) == serial / 2


def test_replay_cap_uses_aggregate_episode_tokens(tmp_path):
    (tmp_path / "plan.json").write_text("""{
      "scenarios": [
        {"episode": 0, "policy": "control",
         "sessions": [{"initial_tokens": 40}, {"initial_tokens": 60}]},
        {"episode": 1, "policy": "control",
         "sessions": [{"initial_tokens": 80}, {"initial_tokens": 120}]}
      ]
    }""")
    (tmp_path / "policy_episodes.csv").write_text(
        "episode,policy,commit_100_s\n"
        "0,replay_only,10\n1,replay_only,20\n"
    )

    caps, count = measured_replay_caps(tmp_path)

    assert caps == {"central": 10, "faster": 10, "slower": 10}
    assert count == 2
