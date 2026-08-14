"""
Claim:
The scheduled repair grid starts from one plan, observes a 25%-work event through
the repair ledger, and either applies a target-restoring residual diff or reports
the attainable maximum without replacing the active plan.

Plausible wrong implementations:
- Independently solve every degraded snapshot.
- Omit the corrected regional timing components.
- Change only offered load instead of observed prefill capacity.
- Treat a revised maximum as an applied empty plan.
- Leak one location's disturbance into another grid axis.
"""

import csv
import json

from repair_plan_shift_campaign import CUT_SCALE, LOCATION_STATES, run


def test_scheduled_grid_uses_the_ledger_and_preserves_unapplied_plans(tmp_path):
    report = run(tmp_path)
    bundle = json.loads((tmp_path / "plans.json").read_text())
    diffs = list(csv.DictReader((tmp_path / "plan_diffs.csv").open()))

    assert report["passed"] and report["cells"] == 16
    assert report["applied"] >= 1 and report["revised_maximum"] >= 1
    assert bundle["semantics"] == "one initial plan, ledger observations, residual repair"
    assert bundle["grid"] == {
        "bandwidth_states": list(LOCATION_STATES),
        "prefill_states": list(LOCATION_STATES),
        "cut_scale": CUT_SCALE,
        "trigger_work_fraction": .25,
        "target_shed_fraction": .5,
        "move_concurrency": 4,
    }
    assert len(bundle["cells"]) == len(diffs) == 16
    control = next(row for row in bundle["cells"]
                   if row["bandwidth_state"] == row["prefill_state"] == "none")
    assert control["outcome"] == "unchanged" and control["target_met"]
    assert control["diff"]["changed_sessions"] == 0
    applied = [row for row in bundle["cells"] if row["outcome"] == "applied"]
    assert all(row["repair_requested"] and row["target_met"]
               and row["diff"]["changed_sessions"] > 0 for row in applied)
    assert all(row["repair_direction"]["increased_impaired_actions"] == 0
               and row["repair_direction"]["reduced_impaired_actions"]
               + row["repair_direction"]["removed_from_impaired"] > 0
               for row in applied)
    assert all("before" in row["resource_utilization"]
               and "after" in row["resource_utilization"]
               for row in bundle["cells"])
    revised = [row for row in bundle["cells"]
               if row["outcome"] == "revised_maximum"]
    assert all(not row["target_met"] and row["diff"]["changed_sessions"] == 0
               for row in revised)
    assert all(row["timing_evidence"] == "calibrated"
               for row in bundle["cells"] if row["bandwidth_state"] == "none")
    assert (tmp_path / "repair_grid.png").stat().st_size > 0
    assert (tmp_path / "repair_grid.pdf").stat().st_size > 0
