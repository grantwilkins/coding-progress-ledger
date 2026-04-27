"""
Claim:
Hand-written event logs define stable progress curves for reverse progress.

Plausible wrong implementations:
- Score after every event instead of after each ledger step.
- Fail to replay grouped same-step discoveries before scoring.
- Drift from the intended non-monotonic progress curve.
"""

from pathlib import Path

from conftest import replay_progress_curve
from ledger_progress import load_events_jsonl


def test_reverse_progress_fixture_matches_expected_curve():
    events = load_events_jsonl(str(Path(__file__).parent / "fixtures" / "reverse_progress.jsonl"))
    curve = replay_progress_curve(events)

    assert curve == [
        (0, 0.0, 0.0, 0.0),
        (1, 0.0, 4.0, 0.0),
        (3, 2.0, 4.0, 0.5),
        (4, 2.0, 8.0, 0.25),
        (5, 1.0, 8.0, 0.125),
        (8, 8.0, 8.0, 1.0),
    ]
