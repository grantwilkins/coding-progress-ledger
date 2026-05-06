"""Run W1/W2/W3 anchors across the R3 model_profile axis.

Triggered by R3's pilot showing architecture flips the dominant bottleneck
label on ≥25% of K8 cells. Per_anchor regime hypothesis must be re_checked
under each profile.
"""
from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.w_under_r3 import run_w_r3_matrix, write_w_r3_artifacts


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = run_w_r3_matrix(REPO)
    write_w_r3_artifacts(rows, REPO / "runs" / "w_under_r3")


if __name__ == "__main__":
    main()
