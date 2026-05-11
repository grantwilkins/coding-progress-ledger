"""
Claim:
The empirical-Bayes estimator returns prefix-safe empirical distributions over
final unit count, and evaluates only non-tautological, training-supported future
thresholds. V1.6 prefix features refine bins but fall back before the original
v1 fields.

Plausible wrong implementations:
- Count final totals below the current prefix total after fallback.
- Drop fallback fields in the wrong order or silently pool sources.
- Emit calibration pairs for thresholds beyond the bin's observed training tail.
- Drop old v1 fields before dropping the new v1.6 refinement features.
- Lose v1.6 feature values when writing prediction artifacts.
- Treat stuck as a one-row event instead of a monotone prefix state.
- Split prefix rows independently instead of keeping whole traces together.
- Compute follow-up bias diagnostics at the wrong grouping level or with the
  observed-minus-predicted sign.
- Claim exact prefix cohorts are informative without comparing their width to
  the pooled same-step marginal.
- Bin interval widths by raw step instead of normalized trace position, or pool
  sources before summarizing.
- Count eval prefixes instead of training prefixes in feature diagnostics, or
  correlate current_total with bucket labels instead of the raw action ratio.
"""

from pathlib import Path

from observation_channel.empirical_bayes import (
    EmpiricalBayesLookup,
    PrefixRow,
    Prediction,
    TraceMeta,
    _bootstrap_bands,
    _bootstrap_bands_preaggregated_python,
    _finer_turn_bucket_support_rows,
    _feature_distribution_rows,
    _five_read_trace_feature_rows,
    _heldout_diagnostics,
    _interval_width_by_trace_position_rows,
    _prediction_rows,
    _prefix_cohort_distribution,
    _rate_bucket_conditional_histograms,
    eligible_prefixes,
    query_json,
    read_prefixes_csv,
    read_traces_csv,
    source_stratified_split,
    turn_bucket,
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
    recent_error_bucket: str = "clean",
    touched_source: bool = False,
    investigation_ratio_bucket: str = "moderate",
) -> PrefixRow:
    return PrefixRow(
        trace_key=key,
        source=source,
        step=step,
        total=total,
        current_category=category,
        current_unit_age=age,
        had_stuck_episode=stuck,
        recent_error_bucket=recent_error_bucket,
        touched_source=touched_source,
        investigation_ratio_bucket=investigation_ratio_bucket,
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
    assert prediction.fallback_depth == 4
    assert prediction.retained_fields == ("source", "total", "current_category", "turn_bucket", "age_bucket")
    assert prediction.source_retained is True
    assert "fallback_depth" in prediction.low_confidence_reasons


def test_lookup_drops_v16_refinements_before_v1_fields() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=7)}
    lookup = EmpiricalBayesLookup.build(
        [
            _prefix("a", recent_error_bucket="clean", investigation_ratio_bucket="low", touched_source=False),
            _prefix("b", recent_error_bucket="heavy", investigation_ratio_bucket="dominant", touched_source=True),
        ],
        traces,
        min_support=2,
    )

    prediction = lookup.predict(_prefix("live", recent_error_bucket="clean", investigation_ratio_bucket="low", touched_source=False))

    assert prediction.values == (6, 7)
    assert prediction.fallback_depth == 3
    assert prediction.retained_fields == ("source", "total", "current_category", "turn_bucket", "age_bucket", "stuck")


def test_query_json_uses_neutral_v16_defaults() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=7)}
    lookup = EmpiricalBayesLookup.build([_prefix("a"), _prefix("b")], traces, min_support=2)

    result = query_json(
        lookup,
        source="s",
        total=5,
        current_category="PRODUCT",
        step=12,
        current_unit_age=3,
        had_stuck_episode=True,
    )

    assert result["retained_fields"] == [
        "source",
        "recent_error_bucket",
        "investigation_ratio_bucket",
        "touched_source",
        "total",
        "current_category",
        "turn_bucket",
        "age_bucket",
        "stuck",
    ]


def test_lookup_current_total_filter_keeps_equal_final_totals() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=7), "low": _trace("low", final_total=5)}
    lookup = EmpiricalBayesLookup.build([_prefix(key, total=6) for key in traces], traces, min_support=2)

    prediction = lookup.predict(_prefix("live", total=6))

    assert prediction.values == (6, 7)


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


def test_turn_bucket_uses_finer_geometric_support_scheme() -> None:
    assert [turn_bucket(step) for step in (1, 2, 3, 4, 5, 7, 8, 11, 12, 15)] == [
        "1-2",
        "1-2",
        "3-4",
        "3-4",
        "5-7",
        "5-7",
        "8-11",
        "8-11",
        "12-15",
        "12-15",
    ]
    assert [turn_bucket(step) for step in (16, 23, 24, 31, 32, 47, 48, 63, 64, 95, 96, 127, 128, 191, 192)] == [
        "16-23",
        "16-23",
        "24-31",
        "24-31",
        "32-47",
        "32-47",
        "48-63",
        "48-63",
        "64-95",
        "64-95",
        "96-127",
        "96-127",
        "128-191",
        "128-191",
        "192+",
    ]


def test_calibration_pairs_require_future_threshold_and_observed_tail_support() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=6)}
    prefixes = [_prefix("a", total=5), _prefix("b", total=5)]
    lookup = EmpiricalBayesLookup.build(prefixes, traces, min_support=2)

    pairs, _ = _prediction_rows([_prefix("a", total=5)], traces, lookup, {"a": "short"})

    assert [row["threshold"] for row in pairs] == [6]
    assert all(row["threshold"] > row["current_total"] for row in pairs)


def test_prediction_artifacts_include_v16_features() -> None:
    traces = {"a": _trace("a", final_total=6), "b": _trace("b", final_total=6)}
    prefixes = [
        _prefix("a", total=5, recent_error_bucket="heavy", touched_source=True, investigation_ratio_bucket="dominant"),
        _prefix("b", total=5, recent_error_bucket="heavy", touched_source=True, investigation_ratio_bucket="dominant"),
    ]
    lookup = EmpiricalBayesLookup.build(prefixes, traces, min_support=2)

    pairs, prefix_predictions = _prediction_rows([prefixes[0]], traces, lookup, {"a": "short"})

    assert pairs[0]["recent_error_bucket"] == "heavy"
    assert pairs[0]["touched_source"] is True
    assert pairs[0]["investigation_ratio_bucket"] == "dominant"
    assert prefix_predictions[0]["recent_error_bucket"] == "heavy"
    assert prefix_predictions[0]["touched_source"] is True
    assert prefix_predictions[0]["investigation_ratio_bucket"] == "dominant"


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
    assert [(row.recent_error_bucket, row.touched_source, row.investigation_ratio_bucket) for row in prefixes] == [
        ("clean", False, "moderate"),
        ("clean", False, "moderate"),
        ("clean", False, "moderate"),
    ]


def test_csv_prefix_loader_reads_v16_feature_columns(tmp_path: Path) -> None:
    traces_csv = tmp_path / "traces.csv"
    turns_csv = tmp_path / "turns.csv"
    traces_csv.write_text(
        "trace_key,source,final_total,total_turns,first_stuck_step,censored_right_tail,parse_error\n"
        "t,s,9,5,,False,\n",
        encoding="utf-8",
    )
    turns_csv.write_text(
        "trace_key,source,instance_id,step,total,done,current_category,current_unit_age,had_stuck_episode,recent_error_bucket,touched_source,investigation_ratio_bucket,kind,tool\n"
        "t,s,t,2,1,0,PRODUCT,1,False,heavy,True,dominant,action,bash\n",
        encoding="utf-8",
    )

    row = read_prefixes_csv(turns_csv, read_traces_csv(traces_csv))[0]

    assert (row.recent_error_bucket, row.touched_source, row.investigation_ratio_bucket) == ("heavy", True, "dominant")


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


def test_preaggregated_bootstrap_preserves_trace_cluster_resampling() -> None:
    pairs = [
        {"trace_key": "a", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 1},
        {"trace_key": "a", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 1},
        {"trace_key": "b", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 0},
        {"trace_key": "b", "source": "s", "source_length_tercile": "s x short", "predicted_p": 0.2, "outcome": 0},
    ]

    bands = _bootstrap_bands_preaggregated_python(pairs, resamples=20, seed=1)

    pooled_bin = next(row for row in bands if row["stratum"] == "pooled" and row["bin"] == 2)
    assert pooled_bin["observed_low"] in {0.0, 0.5, 1.0}
    assert pooled_bin["observed_high"] in {0.0, 0.5, 1.0}
    assert pooled_bin["observed_low"] <= pooled_bin["observed_high"]


def test_heldout_diagnostics_use_grid_offset_category_and_predicted_minus_observed_bias(tmp_path: Path) -> None:
    heldout = tmp_path / "heldout.csv"
    heldout.write_text(
        "trace_key,source,step,current_total,threshold,predicted_p,outcome,length_tercile,source_length_tercile\n"
        "a,swe-agent,10,3,4,0.8,1,short,swe-agent x short\n"
        "a,swe-agent,10,3,5,0.2,0,short,swe-agent x short\n"
        "b,swe-agent,12,3,4,0.7,0,long,swe-agent x long\n"
        "c,hermes,10,3,4,0.1,1,short,hermes x short\n",
        encoding="utf-8",
    )
    categories = {("a", 10): "PRODUCT", ("b", 12): "INVESTIGATION", ("c", 10): "PRODUCT"}

    reliability, category_bias, step_bias = _heldout_diagnostics(heldout, categories)

    offset_one = next(row for row in reliability if row["source_length_tercile"] == "swe-agent x short" and row["grid_offset"] == 1)
    assert offset_one["bin"] == 8
    assert offset_one["mean_bias"] == -0.19999999999999996
    short_product = next(row for row in category_bias if row["source_length_tercile"] == "swe-agent x short")
    assert short_product["current_category"] == "PRODUCT"
    assert short_product["n"] == 2
    assert short_product["mean_bias"] == 0
    assert any(row["source"] == "hermes" and row["length_tercile"] == "short" and row["current_step"] == 10 for row in step_bias)


def test_interval_width_by_trace_position_uses_normalized_position_and_source_bins(tmp_path: Path) -> None:
    predictions = tmp_path / "prefix_predictions.csv"
    predictions.write_text(
        "trace_key,source,step,interval80_width\n"
        "short,s,5,8\n"
        "short,s,9,20\n"
        "long,s,50,12\n"
        "other,other,50,30\n",
        encoding="utf-8",
    )
    traces = {
        "short": _trace("short", source="s", total_turns=10),
        "long": _trace("long", source="s", total_turns=100),
        "other": _trace("other", source="other", total_turns=100),
    }

    rows = _interval_width_by_trace_position_rows(predictions, traces, bins=4)

    source_half = next(row for row in rows if row["source"] == "s" and row["position_bin"] == 2)
    assert source_half["position_low"] == 0.5
    assert source_half["position_high"] == 0.75
    assert source_half["n"] == 2
    assert source_half["mean_interval80_width"] == 10
    assert source_half["p25_interval80_width"] == 8
    assert source_half["p75_interval80_width"] == 12
    assert any(row["source"] == "s" and row["position_bin"] == 3 and row["n"] == 1 for row in rows)
    assert any(row["source"] == "other" and row["position_bin"] == 2 and row["n"] == 1 for row in rows)


def test_feature_distribution_rows_use_training_prefixes_and_raw_action_ratio(tmp_path: Path) -> None:
    turns = tmp_path / "turns.csv"
    turns.write_text(
        "trace_key,source,step,total,current_category,current_unit_age,had_stuck_episode,recent_error_bucket,touched_source,investigation_ratio_bucket,kind,tool\n"
        "train,s,1,0,,0,False,clean,False,low,system,\n"
        "train,s,2,2,INVESTIGATION,1,False,heavy,True,dominant,action,search_dir\n"
        "eval,s,1,99,PRODUCT,1,False,heavy,True,low,action,edit\n",
        encoding="utf-8",
    )
    traces = {"train": _trace("train", source="s", total_turns=6), "eval": _trace("eval", source="s", total_turns=6)}

    rows = _feature_distribution_rows(turns, traces, {"train"})

    clean = next(row for row in rows if row["diagnostic"] == "recent_error_bucket" and row["value"] == "clean")
    heavy = next(row for row in rows if row["diagnostic"] == "recent_error_bucket" and row["value"] == "heavy")
    touched_middle = next(
        row
        for row in rows
        if row["diagnostic"] == "touched_source_by_trace_third" and row["position_third"] == "middle" and row["value"] == "True"
    )
    corr = next(row for row in rows if row["diagnostic"] == "investigation_ratio_current_total_correlation")
    assert clean["n"] == 1
    assert heavy["n"] == 1
    assert touched_middle["fraction"] == 1
    assert corr["pearson_current_total"] == 1


def test_five_read_trace_rows_select_closest_prefix_and_compute_lookup_median(tmp_path: Path) -> None:
    prefix_predictions = tmp_path / "prefix_predictions.csv"
    prefix_predictions.write_text(
        "trace_key,source,step,current_total,recent_error_bucket,touched_source,investigation_ratio_bucket,fallback_depth,support_count,retained_fields,low_confidence_flags,length_tercile,p10,p90,interval80_width\n"
        "target,s,29,1,heavy,False,dominant,0,2,,,,10,20,10\n"
        "target,s,31,1,clean,True,low,0,2,,,,10,20,10\n",
        encoding="utf-8",
    )
    turns = tmp_path / "turns.csv"
    turns.write_text(
        "trace_key,source,step,total,current_category,current_unit_age,had_stuck_episode,recent_error_bucket,touched_source,investigation_ratio_bucket,kind,tool\n"
        "target,s,29,1,INVESTIGATION,1,False,heavy,False,dominant,action,search_dir\n",
        encoding="utf-8",
    )
    traces = {"a": _trace("a", source="s", final_total=10), "b": _trace("b", source="s", final_total=20)}
    lookup = EmpiricalBayesLookup.build([_prefix("a", source="s", total=1), _prefix("b", source="s", total=1)], traces, min_support=1)
    targets = ({"trace_key": "target", "requested_step": 30, "human_read": "hand checked"},)

    rows = _five_read_trace_feature_rows(prefix_predictions, turns, lookup, targets)

    assert rows[0]["selected_step"] == 29
    assert rows[0]["recent_error_bucket"] == "heavy"
    assert rows[0]["investigation_ratio_raw"] == 1
    assert rows[0]["p10"] == 10
    assert rows[0]["p50"] == 10
    assert rows[0]["p90"] == 20


def test_rate_bucket_conditional_histograms_filter_censored_rows(tmp_path: Path) -> None:
    cohorts = tmp_path / "conditional.csv"
    cohorts.write_text(
        "condition_id,requested_step,requested_total,requested_category,selected_step,selected_total,selected_category,support,trace_key,source,final_total,censored_right_tail\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,2,a,s,7,False\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,2,b,s,7,False\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,2,c,s,99,True\n",
        encoding="utf-8",
    )

    rows = _rate_bucket_conditional_histograms(cohorts)

    assert rows == [
        {
            "condition_id": "1",
            "current_category": "PRODUCT",
            "selected_step": "10",
            "selected_total": "3",
            "current_rate": 0.3,
            "final_total": 7,
            "rate_bucket": "0.2-0.3",
            "n": 2,
        }
    ]


def test_exact_prefix_distribution_compares_width_to_same_step_marginal() -> None:
    traces = {
        "a": _trace("a", final_total=5),
        "b": _trace("b", final_total=6),
        "c": _trace("c", final_total=20),
        "d": _trace("d", final_total=30),
    }
    prefixes = [
        _prefix("a", step=10, total=2),
        _prefix("b", step=10, total=2),
        _prefix("c", step=10, total=8),
        _prefix("d", step=10, total=9),
    ]

    hist, summary = _prefix_cohort_distribution(prefixes, traces, step=10, total=2)

    assert summary == [
        {
            "current_step": 10,
            "current_total": 2,
            "pooled_step_n": 4,
            "exact_prefix_n": 2,
            "pooled_step_iqr": 15,
            "exact_prefix_iqr": 1,
            "pooled_step_p90_minus_p10": 25,
            "exact_prefix_p90_minus_p10": 1,
            "conditional_iqr_narrower": True,
        }
    ]
    assert {"group": "exact_prefix", "final_total": 5, "n": 1} in hist


def test_finer_turn_bucket_support_counts_trace_level_support_and_prefix_mass() -> None:
    prefixes = [
        _prefix("a", step=10, total=1, category="PRODUCT"),
        _prefix("a", step=11, total=1, category="PRODUCT"),
        _prefix("b", step=10, total=1, category="PRODUCT"),
        _prefix("c", step=12, total=1, category="PRODUCT"),
    ]

    rows = _finer_turn_bucket_support_rows(prefixes, min_support=2)

    full_key = rows[0]
    assert full_key["fallback_depth"] == 0
    assert full_key["supported_cells"] == 1
    assert full_key["supported_prefixes"] == 3
    assert full_key["max_support"] == 2


def test_rate_bucket_conditional_histograms_can_exclude_near_cap_rows(tmp_path: Path) -> None:
    cohorts = tmp_path / "conditional.csv"
    cohorts.write_text(
        "condition_id,requested_step,requested_total,requested_category,selected_step,selected_total,selected_category,support,trace_key,source,final_total,censored_right_tail\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,3,a,swe-agent,7,False\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,3,b,swe-agent,94,False\n"
        "1,10,3,PRODUCT,10,3,PRODUCT,3,c,hermes,94,False\n",
        encoding="utf-8",
    )

    rows = _rate_bucket_conditional_histograms(cohorts, non_near_cap=True)

    assert {row["final_total"] for row in rows} == {7, 94}
    assert sum(row["n"] for row in rows) == 2
