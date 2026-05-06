#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.claim_cells import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(REPO)
