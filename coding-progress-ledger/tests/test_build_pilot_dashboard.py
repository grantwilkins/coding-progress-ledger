from scripts.build_pilot_dashboard import OUT, main


def test_dashboard_has_20_rows_and_10_10_split():
    main()
    text = OUT.read_text()
    body = [ln for ln in text.splitlines() if ln.startswith("| swe_agent_pilot_")]
    assert len(body) == 20
    successes = sum(1 for ln in body if "| True |" in ln)
    failures = sum(1 for ln in body if "| False |" in ln)
    assert successes == 10
    assert failures == 10
