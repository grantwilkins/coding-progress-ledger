"""Run V1 exact validation for selected K8 claim cells."""
from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.k8_validation import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(REPO)
