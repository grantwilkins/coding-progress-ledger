"""R3 — emit the per_model_architecture K8 regime sweep.

Runs the K8 aggregate sweep once per model profile in `R3_DEFAULT_PROFILES`
and writes the per_cell flip table plus a flip_count axis breakdown to
`runs/r3_model_sweep/`.

The full default sweep (5 models × 4 N × 5 state_scales × 3 prefill caps
× 4 links = 1200 cells × 6 policies) is intentionally aggregate_only; exact
K4 belongs in V1 re_validation for cells the flip table flags.
"""
from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.r3_model_sweep import (
    R3_DEFAULT_PROFILES,
    run_r3_sweep,
    write_r3_artifacts,
)


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics_by_model = run_r3_sweep(REPO, model_names=R3_DEFAULT_PROFILES)
    write_r3_artifacts(metrics_by_model, REPO / "runs" / "r3_model_sweep")


if __name__ == "__main__":
    main()
