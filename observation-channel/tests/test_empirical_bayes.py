"""
Claim:
The empirical-Bayes estimator returns prefix-safe empirical distributions over
final unit count, and evaluates only non-tautological, training-supported future
thresholds.

Plausible wrong implementations:
- Count final totals below the current prefix total after fallback.
- Drop fallback fields in the wrong order or silently pool sources.
- Emit calibration pairs for thresholds beyond the bin's observed training tail.
- Treat stuck as a one-row event instead of a monotone prefix state.
- Split prefix rows independently instead of keeping whole traces together.
"""

from pathlib import Path

from observation_channel.empirical_bayes import (
    EmpiricalBayesLookup,
    PrefixRow,
    Prediction,
    TraceMeta,
    _bootstrap_bands,
    _prediction_rows,
    eligible_prefixes,
    read_prefixes_csv,
    read_traces_csv,
    source_stratified_split,
)


def _trace(key: str, source: str = "s", final_total: int = 10, total_turns: int = 20) -> TraceMeta:
    return TraceMeta(trace_key=key, source=source, final_total=final_total, total_turns=total_turns)


def _prefix(
    key: str,
    *,
    source: str = "s",
    total: int = 5,
    category: str = "PRODUCT",
    step: int = 12,
    age: int = 3,
    stuck: bool = True,
) -> PrefixRow:
    return PrefixRow(
        trace_key=key,
        source=source,
        step=step,
        total=total,
        current_category=category,
        current_unit_age=age,
        had_stuck_episode=stuck,
    )


def test_lookup_fallback_retains_source_and_filters_current_total_lower_bound() -> None:
    traces = {
        "a": _trace("a", final_total=6),
        "b": _trace("b", final_total=7),
        "low": _trace("low", final_total=4),
        "other": _trace("other", source="other", final_total=99),
    }
    lookup = EmpiricalBayesLookup.build(
        [
            _prefix("a", stuck=True),
            _prefix("b", stuck=False),
            _prefix("low", stuck=False),
            _prefix("other", source="other", stuck=False),
        ],
        traces,
        min_support=2,
    )

    prediction = lookup.predict(_prefix("live", stuck=True))

    assert prediction.values == (6, 7)
    assert prediction.fallback_depth == 1
    assert prediction.retained_fields == ("source", "total", "current_category", "turn_bucket", "age_bucket")
    assert prediction.source_retained is True
    assert "fallback_depth" in prediction.low_confidence_reasons


def test_prediction_cdf_quantile_and_progress_interval_are_hand_checkable() -> None:
    result = Prediction(
        values=(6, 8, 10),
        support_count=3,
        retained_fields=("source",),
        fallback_depth=0,
        source_retained=True,
        low_confidence_reasons=(),
    )

    assert result.cdf(8) == 2 / 3
    assert result.quantile(0.5) == 8
    assert result.progress_interval(5) == (0.5, 5 / 6)


def test_calibration_pairs_require_future_threshold_and_observed_tail_support() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=6)}
    prefixes = [_prefix("a", total=5), _prefix("b", total=5)]
    lookup = EmpiricalBayesLookup.build(prefixes, traces, min_support=2)

    pairs, _ = _prediction_rows([_prefix("a", total=5)], traces, lookup, {"a": "short"})

    assert [row["threshold"] for row in pairs] == [6]
    assert all(row["threshold"] > row["current_total"] for row in pairs)


def test_source_stratified_split_keeps_whole_traces_and_reports_counts() -> None:
    traces = [_trace(f"a{i}", source="a") for i in range(5)] + [_trace(f"b{i}", source="b") for i in range(5)]

    train, eval_, summary = source_stratified_split(traces)

    assert train.isdisjoint(eval_)
    assert {row["source"]: (row["train_traces"], row["eval_traces"]) for row in summary} == {
        "a": (4, 1),
        "b": (4, 1),
    }


def test_csv_prefix_loader_derives_monotone_stuck_state_from_first_stuck_step(tmp_path: Path) -> None:
    traces_csv = tmp_path / "traces.csv"
    turns_csv = tmp_path / "turns.csv"
    traces_csv.write_text(
        "trace_key,source,final_total,total_turns,first_stuck_step,censored_right_tail,parse_error\n"
        "t,s,9,5,3,False,\n",
        encoding="utf-8",
    )
    turns_csv.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,kind,tool\n"
        "t,s,t,2,1,0,PRODUCT,1,action,bash\n"
        "t,s,t,3,1,0,PRODUCT,2,observation,\n"
        "t,s,t,4,1,0,PRODUCT,3,action,bash\n",
        encoding="utf-8",
    )

    prefixes = read_prefixes_csv(turns_csv, read_traces_csv(traces_csv))

    assert [row.had_stuck_episode for row in prefixes] == [False, True, True]


def test_censored_and_terminal_prefixes_are_excluded_from_eligible_training_rows() -> None:
    traces = {
        "ok": _trace("ok", final_total=5, total_turns=4),
        "censored": TraceMeta("censored", "s", 200, 4, censored_right_tail=True),
    }
    prefixes = [_prefix("ok", step=3), _prefix("ok", step=4), _prefix("censored", step=3)]

    assert eligible_prefixes(prefixes, traces) == [_prefix("ok", step=3)]


def test_bootstrap_bands_record_fixed_seed() -> None:
    pairs = [
        {"trace_key": "a", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 0},
        {"trace_key": "a", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 0},
        {"trace_key": "b", "source": "s", "source_length_tercile": "s x long", "predicted_p": 0.8, "outcome": 1},
    ]

    bands = _bootstrap_bands(pairs, resamples=5, seed=123)

    assert bands
    assert {row["seed"] for row in bands} == {123}
    assert {row["bootstrap_resamples"] for row in bands} == {5}
