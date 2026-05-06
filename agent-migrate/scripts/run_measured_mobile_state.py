from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.measured_mobile_state import read_snapshot_index, write_measured_artifacts


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "runs" / "measured_mobile_state"
    snapshot_index = out / "upstream_snapshots" / "raw_snapshot_index.csv"
    snapshots = read_snapshot_index(snapshot_index)
    write_measured_artifacts(snapshots, out, repo)


if __name__ == "__main__":
    main()
