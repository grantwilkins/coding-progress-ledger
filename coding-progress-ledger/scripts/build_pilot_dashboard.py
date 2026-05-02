"""L2 — Pilot dashboard summarizing all 20 SWE-agent pilots."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "runs" / "swe_agent_pilot"
PLOTS_DIR = PILOT_DIR / "plots"
EVIDENCE_MD = PILOT_DIR / "EVIDENCE_AUDIT.md"
OUT = PILOT_DIR / "PILOT_DASHBOARD.md"

EVIDENCE_ROW = re.compile(r"^\| `(swe_agent_pilot_[a-z0-9_]+)` \| (\w+) \|")


def load_evidence_status() -> dict[str, str]:
    out = {}
    for line in EVIDENCE_MD.read_text().splitlines():
        m = EVIDENCE_ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def pilot_row(pilot_dir: Path, evidence: dict[str, str]) -> dict[str, str]:
    pid = pilot_dir.name
    meta = json.loads((pilot_dir / "source_metadata.json").read_text())
    csv_path = PLOTS_DIR / f"{pid}_progress.csv"
    rows = list(csv.DictReader(csv_path.open()))
    cps = [float(r["coding_progress"]) for r in rows]
    largest_drop = max((cps[i - 1] - cps[i] for i in range(1, len(cps)) if cps[i] < cps[i - 1]), default=0.0)
    aq_path = pilot_dir / "annotation_quality.json"
    aq_time = "n/a"
    if aq_path.exists():
        aq = json.loads(aq_path.read_text())
        aq_time = str(aq.get("annotation_time_minutes", "n/a"))
    return {
        "pilot_id": pid,
        "final_success": str(meta.get("final_success")),
        "final_coding_progress": f"{cps[-1]:.3f}",
        "largest_drop": f"{largest_drop:.3f}",
        "evidence_status": evidence.get(pid, "n/a"),
        "annotation_time": aq_time,
        "progress_csv_path": f"plots/{pid}_progress.csv",
    }


def render(rows: list[dict[str, str]]) -> str:
    cols = ["pilot_id", "final_success", "final_coding_progress", "largest_drop", "evidence_status", "annotation_time", "progress_csv_path"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "# Pilot Dashboard\n\n" + "\n".join([head, sep, *body]) + "\n"


def main() -> None:
    evidence = load_evidence_status()
    pilot_dirs = sorted(p for p in PILOT_DIR.iterdir() if p.is_dir() and p.name.startswith("swe_agent_pilot_"))
    rows = [pilot_row(p, evidence) for p in pilot_dirs]
    OUT.write_text(render(rows))
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
