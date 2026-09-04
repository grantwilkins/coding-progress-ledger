"""The A100 parity pair must use the complete genuine A100 evidence."""

import statistics
from collections import Counter

import pytest

from plot_a100_parity import POWER, TIMING, load_power, load_timing


def _metrics(rows, predicted, measured):
    estimate = [row[predicted] for row in rows]
    observed = [row[measured] for row in rows]
    residuals = [y - x for x, y in zip(estimate, observed)]
    mean = statistics.fmean(observed)
    return statistics.fmean(map(abs, residuals)), 1 - sum(
        value * value for value in residuals) / sum(
            (value - mean) ** 2 for value in observed)


def test_a100_timing_parity_preserves_training_and_holdout_paths():
    rows = load_timing(TIMING)

    assert len(rows) == 108
    assert Counter(row["cohort"] for row in rows) == {
        "training": 72, "holdout_context": 36}
    assert Counter(row["action"] for row in rows) == {
        "replay": 54, "kv_transfer": 54}
    assert _metrics(rows, "predicted_s", "measured_s") == pytest.approx(
        (.27310911243814245, .9924406229483246))


def test_a100_power_parity_uses_every_settled_post_migration_window():
    rows = load_power(POWER)

    assert len(rows) == 350
    assert Counter(row["family"] for row in rows) == {
        "idle": 42, "sessions": 308}
    assert {row["cohort"] for row in rows} == {"descriptive_in_sample"}
    assert _metrics(rows, "predicted_power_w", "measured_power_w") == \
        pytest.approx((2.2821956605432696, .9559351601579562))
