"""
Claim:
The GBM trial changes only the probability mechanism: it uses continuous raw
features and LightGBM quantile CDFs, but emits calibration pairs at exactly the
same v1.6 support-gated prefix/threshold points.

Plausible wrong implementations:
- Emit GBM probabilities for every future-work grid point instead of applying
  the v1.6 lookup support gate.
- Use row-level early-stopping validation and leak prefixes from the same trace.
- Convert quantiles to a CDF with the wrong boundary convention or duplicate
  quantile handling.
- Sort crossed quantiles silently without reporting how often or how much.
- Fall back to bucket labels when raw continuous features are absent.
"""

from pathlib import Path

import pytest

from observation_channel.empirical_bayes import EmpiricalBayesLookup, PrefixRow, TraceMeta, _prediction_rows
from observation_channel.gbm_trial import (
    GbmQuantilePrediction,
    _progress_tracking_rows,
    gbm_prediction_rows,
    quantile_crossing_summary_rows,
    require_raw_feature_columns,
    trace_level_validation_split,
)


def _trace(key: str, final_total: int, total_turns: int = 20) -> TraceMeta:
    return TraceMeta(trace_key=key, source="s", final_total=final_total, total_turns=total_turns)


def _prefix(key: str, total: int = 5, step: int = 10) -> PrefixRow:
    return PrefixRow(
        trace_key=key,
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


class _FakeBundle:
    def predict(self, rows: list[PrefixRow]) -> list[GbmQuantilePrediction]:
        return [GbmQuantilePrediction.from_raw((6, 8, 10, 12, 14), row.total) for row in rows]


def test_quantile_cdf_clamps_sorts_and_uses_boundary_convention() -> None:
    prediction = GbmQuantilePrediction.from_raw((12, 10, 15, 14, 20), current_total=11)

    assert prediction.crossed is True
    assert prediction.quantiles == (11, 12, 14, 15, 20)
    assert prediction.cdf(10) == 0.0
    assert prediction.cdf(11) == 0.1
    assert prediction.cdf(12) == 0.25
    assert prediction.cdf(13) == 0.375
    assert prediction.cdf(20) == 0.9
    assert prediction.cdf(21) == 1.0


def test_quantile_cdf_collapses_duplicate_locations_to_highest_probability() -> None:
    prediction = GbmQuantilePrediction.from_raw((5, 5, 5, 10, 20), current_total=5)

    assert prediction.cdf(4) == 0.0
    assert prediction.cdf(5) == 0.5
    assert prediction.cdf(10) == 0.75


def test_gbm_eval_emits_exactly_the_lookup_supported_pairs() -> None:
    traces = {"a": _trace("a", 6), "b": _trace("b", 9), "eval": _trace("eval", 20)}
    train_prefixes = [_prefix("a", total=5), _prefix("b", total=5)]
    eval_prefixes = [_prefix("eval", total=5)]
    lookup = EmpiricalBayesLookup.build(train_prefixes, traces, min_support=2)

    lookup_pairs, _ = _prediction_rows(eval_prefixes, traces, lookup, {"eval": "short"})
    gbm_pairs, _, _ = gbm_prediction_rows(eval_prefixes, traces, lookup, _FakeBundle(), {"eval": "short"})

    lookup_keys = {(row["trace_key"], row["step"], row["grid_offset"], row["threshold"]) for row in lookup_pairs}
    gbm_keys = {(row["trace_key"], row["step"], row["grid_offset"], row["threshold"]) for row in gbm_pairs}
    assert gbm_keys == lookup_keys == {("eval", 10, 1, 6), ("eval", 10, 2, 7), ("eval", 10, 4, 9)}
    assert {row["predicted_p"] for row in gbm_pairs} != {row["predicted_p"] for row in lookup_pairs}


def test_trace_level_validation_split_keeps_whole_traces() -> None:
    train, valid = trace_level_validation_split({f"t{i}" for i in range(10)}, validation_fraction=0.2)

    assert train.isdisjoint(valid)
    assert len(valid) == 2
    assert len(train) == 8


def test_raw_feature_columns_are_required(tmp_path: Path) -> None:
    missing = tmp_path / "turns.csv"
    missing.write_text("trace_key,source,step,total\nx,s,1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recent_error_rate, investigation_ratio"):
        require_raw_feature_columns(missing)


def test_quantile_crossing_summary_uses_crossed_rows_only_for_magnitude() -> None:
    rows = quantile_crossing_summary_rows(
        {
            "s": [
                GbmQuantilePrediction.from_raw((1, 2, 3, 4, 5), 0),
                GbmQuantilePrediction.from_raw((1, 4, 3, 6, 7), 0),
                GbmQuantilePrediction.from_raw((5, 1, 2, 3, 4), 0),
            ]
        }
    )

    row = rows[0]
    assert row["crossing_count"] == 2
    assert row["crossing_rate"] == 2 / 3
    assert row["mean_reordering_magnitude_when_crossed"] == 2.5
    assert row["p95_reordering_magnitude_when_crossed"] == 4


def test_progress_tracking_rows_convert_final_quantiles_to_progress_fractions() -> None:
    traces = {"eval": _trace("eval", final_total=20)}
    prefix = _prefix("eval", total=5, step=3)
    prediction = GbmQuantilePrediction.from_raw((10, 12, 20, 25, 50), current_total=5)

    rows = _progress_tracking_rows([{"trace_key": "eval", "step": "3", "length_tercile": "short"}], [prefix], traces, [prediction])

    assert rows == [
        {
            "trace_key": "eval",
            "source": "s",
            "length_tercile": "short",
            "step": 3,
            "current_total": 5,
            "final_total": 20,
            "rho_actual": 0.25,
            "rho_p10": 0.1,
            "rho_p50": 0.25,
            "rho_p90": 0.5,
            "p10_final_total": 10.0,
            "p50_final_total": 20.0,
            "p90_final_total": 50.0,
        }
    ]
