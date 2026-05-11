"""
Claim:
The belief tracker converts empirical-Bayes and GBM final-work beliefs into the
same remaining-work probability claims, and filtered variants update per trace
without letting GBM override the empirical-Bayes anchor.

Plausible wrong implementations:
- Use current work as the remaining-fraction denominator, or skip the floor on
  discrete final-work thresholds.
- Treat adjacent prefixes as independent evidence instead of using damped log
  pooling.
- Leak filtered belief state across traces.
- Let GBM nudge the mixed filter when quantiles cross or the median contradicts
  the empirical-Bayes band.
- Drop unsupported prefixes instead of emitting explicit no-support rows.
"""

from pathlib import Path

import pytest

from observation_channel.belief_tracker import (
    BeliefTracker,
    BeliefTrackerConfig,
    FinalWorkBelief,
    _log_pool,
    evaluate_belief_tracker,
)
from observation_channel.cli import main
from observation_channel.empirical_bayes import Prediction, PrefixRow
from observation_channel.gbm_trial import GbmQuantilePrediction


def _prefix(trace_key: str = "a", *, step: int = 1, total: int = 5) -> PrefixRow:
    return PrefixRow(
        trace_key=trace_key,
        source="s",
        step=step,
        total=total,
        current_category="PRODUCT",
        current_unit_age=1,
        had_stuck_episode=False,
        recent_error_bucket="clean",
        recent_error_rate=0.0,
        touched_source=False,
        investigation_ratio_bucket="low",
        investigation_ratio=0.0,
    )


def _prediction(values: tuple[int, ...]) -> Prediction:
    return Prediction(
        values=values,
        support_count=len(values),
        retained_fields=("source",),
        fallback_depth=0,
        source_retained=True,
        low_confidence_reasons=(),
    )


class _Lookup:
    def __init__(self, predictions: dict[tuple[str, int], Prediction]) -> None:
        self.predictions = predictions

    def predict(self, row: PrefixRow) -> Prediction:
        prediction = self.predictions.get((row.trace_key, row.step))
        if prediction is None:
            raise ValueError("no supported empirical-Bayes bin for prefix")
        return prediction


class _FakeBundle:
    def predict(self, rows: list[PrefixRow]) -> list[GbmQuantilePrediction]:
        return [GbmQuantilePrediction.from_raw((6, 8, 10, 12, 14), row.total) for row in rows]


def test_remaining_fraction_claim_uses_final_denominator_and_floor_threshold() -> None:
    belief = FinalWorkBelief({6: 0.5, 7: 0.5})
    row = BeliefTracker(_Lookup({("a", 1): _prediction((6, 7))})).update(_prefix())

    assert belief.cdf(6) == 0.5
    assert row["eb_direct_remaining_work_fraction_p50"] == pytest.approx(1 / 6)
    assert row["eb_direct_prob_remaining_work_le_25pct"] == 0.5
    assert row["eb_direct_prob_remaining_work_le_50pct"] == 1.0


def test_log_pool_is_damped_geometric_update_not_linear_average() -> None:
    previous = FinalWorkBelief({10: 0.25, 20: 0.75})
    observation = FinalWorkBelief({10: 0.75, 20: 0.25})

    pooled = _log_pool(previous, observation, alpha=0.25)

    assert pooled.masses[10] == pytest.approx(0.3660254038)
    assert pooled.masses[20] == pytest.approx(0.6339745962)


def test_filter_masks_impossible_final_work_and_resets_between_traces() -> None:
    tracker = BeliefTracker(
        _Lookup(
            {
                ("a", 1): _prediction((5, 8, 12)),
                ("a", 2): _prediction((8, 12)),
                ("b", 1): _prediction((100, 200)),
            }
        )
    )

    first = tracker.update(_prefix("a", step=1, total=5))
    second = tracker.update(_prefix("a", step=2, total=8))
    other_trace = tracker.update(_prefix("b", step=1, total=5))

    assert first["eb_filter_estimated_final_work_p10"] == 5
    assert min(tracker._states["a"].eb_filter.masses) >= 8
    assert second["eb_filter_estimated_final_work_p10"] >= 8
    assert other_trace["eb_filter_estimated_final_work_p50"] == 100


def test_gbm_direct_uses_existing_quantile_cdf_for_probability_claims() -> None:
    tracker = BeliefTracker(_Lookup({("a", 1): _prediction((6, 8, 10, 12, 14))}))
    gbm = GbmQuantilePrediction.from_raw((5, 5, 5, 10, 20), current_total=5)

    row = tracker.update(_prefix(), gbm)

    assert row["gbm_direct_prob_finish_within_8_work_units"] == pytest.approx(0.795)


def test_mixed_filter_rejects_material_crossing_and_median_outside_eb_band() -> None:
    tracker = BeliefTracker(
        _Lookup(
            {
                ("crossed", 1): _prediction((10, 12, 14, 16, 18, 20)),
                ("outside", 1): _prediction((10, 12, 14, 16, 18, 20)),
            }
        ),
        BeliefTrackerConfig(gbm_crossing_tolerance=1.0),
    )

    crossed = tracker.update(_prefix("crossed", total=5), GbmQuantilePrediction.from_raw((20, 10, 14, 16, 18), 5))
    outside = tracker.update(_prefix("outside", total=5), GbmQuantilePrediction.from_raw((30, 31, 32, 33, 34), 5))

    assert crossed["eb_gbm_mixed_filter_gbm_used"] is False
    assert crossed["eb_gbm_mixed_filter_gbm_rejected_reason"] == "gbm_quantile_crossing"
    assert outside["eb_gbm_mixed_filter_gbm_used"] is False
    assert outside["eb_gbm_mixed_filter_gbm_rejected_reason"] == "gbm_median_outside_eb_band"


def test_unsupported_empirical_bayes_prefix_keeps_gbm_row_and_flags_no_support() -> None:
    row = BeliefTracker(_Lookup({})).update(_prefix(), GbmQuantilePrediction.from_raw((6, 8, 10, 12, 14), 5))

    assert row["gbm_direct_estimated_final_work_p50"] == 10.0
    assert row["eb_direct_estimated_final_work_p50"] == ""
    assert row["eb_gbm_mixed_filter_estimated_final_work_p50"] == ""
    assert row["confidence_flags"] == "no_empirical_bayes_support"


def test_belief_tracker_eval_writes_artifacts_with_fake_gbm_bundle(tmp_path: Path) -> None:
    traces = tmp_path / "traces.csv"
    turns = tmp_path / "turns.csv"
    report_dir = tmp_path / "belief"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    traces.write_text(
        "trace_key,source,final_total,total_turns,first_stuck_step,censored_right_tail,parse_error\n"
        "a,s,10,3,,False,\n"
        "b,s,12,3,,False,\n",
        encoding="utf-8",
    )
    turns.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,kind,tool,"
        "recent_error_bucket,recent_error_rate,touched_source,investigation_ratio_bucket,investigation_ratio\n"
        "a,s,a,1,5,0,PRODUCT,1,action,bash,clean,0.0,False,low,0.0\n"
        "b,s,b,1,5,0,PRODUCT,1,action,bash,clean,0.0,False,low,0.0\n",
        encoding="utf-8",
    )

    result = evaluate_belief_tracker(turns, traces, report_dir, model_dir, min_support=1, bundle=_FakeBundle())

    assert result["prefix_rows"] == 1
    assert (report_dir / "progress_beliefs.csv").exists()
    assert (report_dir / "belief_threshold_pairs.csv").exists()
    assert (report_dir / "belief_summary.csv").exists()
    assert (report_dir / "REPORT.md").exists()


def test_belief_tracker_cli_fails_when_raw_gbm_columns_are_missing(tmp_path: Path) -> None:
    traces = tmp_path / "traces.csv"
    turns = tmp_path / "turns.csv"
    traces.write_text(
        "trace_key,source,final_total,total_turns,first_stuck_step,censored_right_tail,parse_error\n"
        "a,s,10,3,,False,\n",
        encoding="utf-8",
    )
    turns.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,kind,tool\n"
        "a,s,a,1,5,0,PRODUCT,1,action,bash\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recent_error_rate, investigation_ratio"):
        main(
            [
                "belief-tracker-eval",
                "--turns-csv",
                str(turns),
                "--traces-csv",
                str(traces),
                "--report-dir",
                str(tmp_path / "report"),
                "--model-dir",
                str(tmp_path / "model"),
            ]
        )
