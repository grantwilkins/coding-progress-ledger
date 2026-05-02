import csv
from pathlib import Path

from scripts.plot_progress_curves import OUT_DIR, main

EXPECTED_COLS = ["step", "coding_progress", "overall_progress", "coding_drop_marker"]


def test_s_03_csv_columns_drop_and_monotone_steps():
    main()
    csv_path = OUT_DIR / "swe_agent_pilot_s_03_progress.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert list(rows[0].keys()) == EXPECTED_COLS
    steps = [int(r["step"]) for r in rows]
    assert steps == sorted(steps)
    by_step = {int(r["step"]): r for r in rows}
    assert by_step[22]["coding_drop_marker"] == "1"
