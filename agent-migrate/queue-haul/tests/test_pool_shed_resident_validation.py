from pool_shed_resident_validation import checks, comparison, window


def test_window_counts_censored_arrivals_and_only_complete_tpot():
    rows = [dict(scheduled_s=60., first_s=62., done=False, end_s=300., mean_tpot_s=None),
            dict(scheduled_s=65., first_s=None, done=False, end_s=None, mean_tpot_s=None),
            dict(scheduled_s=89.9, first_s=90.1, done=True, end_s=91., mean_tpot_s=.02)]
    result = window(rows, 60, 90)
    assert result["arrivals"] == result["unfinished"] == 3
    assert result["known_ttft_violations"] == 2
    assert result["ttft_p90_s"] == 2
    assert result["tpot_p90_s"] is None and result["tpot_samples"] == 0
    assert not comparison(2, None, .2)["pass"]
    assert comparison(None, .2, .2)["pass"] is None
    assert not checks([])["gate_pass"]
