"""JSON I/O for placement and materialization plans."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .policies import MaterializationDecision, PlacementDecision, Plan


def write_plan(plan: Plan, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    placement_path = out / "placement_plan.json"
    materialization_path = out / "materialization_plan.json"
    placement_path.write_text(json.dumps(
        {"policy": plan.policy, "meta": plan.meta,
         "placements": [asdict(p) for p in plan.placements]},
        indent=2) + "\n")
    materialization_path.write_text(json.dumps(
        {"policy": plan.policy, "meta": plan.meta,
         "materializations": [asdict(m) for m in plan.materializations]},
        indent=2) + "\n")


def read_plan(out_dir: str | Path) -> Plan:
    out = Path(out_dir)
    placement_data = json.loads((out / "placement_plan.json").read_text())
    mat_data = json.loads((out / "materialization_plan.json").read_text())
    if placement_data["policy"] != mat_data["policy"]:
        raise ValueError("placement and materialization files disagree on policy")
    return Plan(
        policy=placement_data["policy"],
        meta=placement_data.get("meta", {}),
        placements=[PlacementDecision(**p) for p in placement_data["placements"]],
        materializations=[MaterializationDecision(**m) for m in mat_data["materializations"]],
    )
