"""L1 — Per-pilot progress CSVs for runs/swe_agent_pilot."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "runs" / "swe_agent_pilot"
OUT_DIR = PILOT_DIR / "plots"


def emit_pilot_csv(pilot_dir: Path, out_dir: Path) -> Path:
    src = pilot_dir / "progress_by_category.csv"
    rows = list(csv.DictReader(src.open()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pilot_dir.name}_progress.csv"
    prev = None
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "coding_progress", "overall_progress", "coding_drop_marker"])
        for r in rows:
            cp = float(r["coding_progress"])
            op = float(r["overall_progress"])
            drop = 1 if prev is not None and cp < prev else 0
            w.writerow([r["step"], r["coding_progress"], r["overall_progress"], drop])
            prev = cp
    return out


def main() -> None:
    pilot_dirs = sorted(p for p in PILOT_DIR.iterdir() if p.is_dir() and p.name.startswith("swe_agent_pilot_"))
    for p in pilot_dirs:
        emit_pilot_csv(p, OUT_DIR)
    print(f"emitted {len(pilot_dirs)} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
