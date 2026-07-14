import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_stage1c_results import load_results, write_plots


def test_stage1c_results_are_paired_and_reproducible(tmp_path: Path):
    rows = load_results()
    by_pair = {(row["condition"], row["replicate"]): row for row in rows}

    assert len(rows) == 6
    assert all(row["deadline_hit"] and not row["power_hit"] for row in rows)
    assert all(by_pair[("mechanism_only", i)]["measured_source_drop_w"] > by_pair[("admission", i)]["measured_source_drop_w"] for i in range(3))
    paths = write_plots(rows, tmp_path)
    assert {path.stem for path in paths} == {"stage1c_completion", "stage1c_power_change", "stage1c_power_levels"}
    assert len(paths) == 6 and all(path.exists() for path in paths)
