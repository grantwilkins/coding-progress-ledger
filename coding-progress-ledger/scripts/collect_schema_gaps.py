#!/usr/bin/env python3
"""I1: collect schema gaps from per-pilot run_notes.md and annotation_quality.json.

Output: runs/swe_agent_pilot/SCHEMA_GAPS.md (overwritten on each run).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SECTION_RE = re.compile(r"^###\s+8\.\s+Schema gaps observed\s*$", re.MULTILINE)
NEXT_SECTION_RE = re.compile(r"^###\s+\d+\.\s", re.MULTILINE)


def extract_schema_gap_section(notes_path: Path) -> str:
    text = notes_path.read_text(encoding="utf-8")
    m = SECTION_RE.search(text)
    if not m:
        raise ValueError(f"missing '### 8. Schema gaps observed' in {notes_path}")
    rest = text[m.end():]
    nxt = NEXT_SECTION_RE.search(rest)
    body = rest[: nxt.start()] if nxt else rest
    return body.strip()


def is_none_body(body: str) -> bool:
    head = body.splitlines()[0].lstrip("*-> ").strip().lower() if body else ""
    return head.startswith("none")


def pilot_records(runs_dir: Path) -> list[dict]:
    records = []
    for pilot_dir in sorted(runs_dir.glob("swe_agent_pilot_*")):
        notes = pilot_dir / "run_notes.md"
        quality = pilot_dir / "annotation_quality.json"
        body = extract_schema_gap_section(notes)
        q = json.loads(quality.read_text(encoding="utf-8"))
        records.append({
            "pilot_id": pilot_dir.name,
            "schema_gap_found": bool(q["whether_schema_gap_found"]),
            "evidence_gaps": int(q["number_of_evidence_gaps"]),
            "uncertain_events": int(q["number_of_uncertain_events"]),
            "section_body": body,
            "is_none": is_none_body(body),
        })
    return records


def render_markdown(records: list[dict], extra_findings: list[dict]) -> str:
    flagged = [r for r in records if r["schema_gap_found"]]
    none_pilots = [r for r in records if r["is_none"]]
    other_pilots = [r for r in records if not r["schema_gap_found"] and not r["is_none"]]
    lines = ["# Schema gaps collected from pilot run notes (I1)", ""]
    lines.append(f"Pilots scanned: **{len(records)}**. ")
    lines.append(f"Pilots with `whether_schema_gap_found = true`: **{len(flagged)}**. ")
    lines.append(f"Pilots reporting 'None' in § 8: **{len(none_pilots)}**.")
    lines.append("")
    lines.append("## 1. Pilots that flagged a schema gap")
    lines.append("")
    if not flagged:
        lines.append("_(none)_")
    for r in flagged:
        lines.append(f"### {r['pilot_id']}")
        lines.append("")
        lines.append(r["section_body"])
        lines.append("")
    lines.append("## 2. Pilots that explicitly reported no schema gap")
    lines.append("")
    lines.append(f"{len(none_pilots)} pilots: " + ", ".join(f"`{r['pilot_id']}`" for r in none_pilots))
    lines.append("")
    if other_pilots:
        lines.append("## 3. Pilots with non-None § 8 prose but no quality flag")
        lines.append("")
        for r in other_pilots:
            lines.append(f"### {r['pilot_id']}")
            lines.append("")
            lines.append(r["section_body"])
            lines.append("")
    lines.append("## 4. Cross-workstream findings (post-pilot)")
    lines.append("")
    if not extra_findings:
        lines.append("_(none)_")
    for f in extra_findings:
        lines.append(f"### {f['title']}")
        lines.append("")
        lines.append(f"- **Severity:** {f['severity']}")
        lines.append(f"- **Class:** {f['kind']}")
        lines.append(f"- **Source:** {f['source']}")
        lines.append("")
        lines.append(f["body"])
        lines.append("")
    lines.append("## 5. Classification summary")
    lines.append("")
    lines.append(_classification_table(flagged, extra_findings))
    return "\n".join(lines).rstrip() + "\n"


def _classification_table(flagged: list[dict], extra_findings: list[dict]) -> str:
    rows = [("`f_02` stuck-loop rule covered command loops only", "missing protocol coverage", "annoying", "in-pilot, resolved"),
            ("`f_07` stuck-loop rule ambiguous on cycle length", "missing protocol coverage", "annoying", "in-pilot, resolved")]
    for f in extra_findings:
        rows.append((f["title"], f["kind"], f["severity"], f["source"]))
    out = ["| Gap | Class | Severity | Source/Status |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for (a, b, c, d) in rows]
    return "\n".join(out)


EXTRA_FINDINGS = [
    {
        "title": "v1 inconsistently applied Pitfall #8 across harness-terminated failure pilots",
        "severity": "annoying",
        "kind": "protocol application (not schema)",
        "source": "H4 GATE_RESULT § 7",
        "body": (
            "The HIGH-severity H3 revision (Pitfall #8: bug-fix tasks always carry "
            "an implicit `VALIDATION` leaf) was applied by v1 to `f_01` / `f_04` / `s_04` "
            "(submit-without-test) but not to `f_02` / `f_03` / `f_07` / `f_10` "
            "(harness-forced termination mid-loop). The schema is fine; this is a "
            "protocol-application gap. Fix: re-emit specs for the four pilots adding "
            "a not_started VAL leaf. Estimated effort ~30 min."
        ),
    },
    {
        "title": "Builder reports `category_resolution_mode = mixed` for 181/191 SWE-agent step rows",
        "severity": "note",
        "kind": "category resolution / pipeline",
        "source": "datasets/swe_agent_pilot_observations_step_audit.md",
        "body": (
            "Annotation specs assign categories explicitly on every add/split, yet the "
            "builder records 'mixed' for nearly all step rows and 'native' for only 10. "
            "One run (`s_03`) carries a 'large native/resolved divergence' warning. "
            "This is investigated and resolved in J1; logged here so the gap is "
            "visible in the schema-gap collection."
        ),
    },
    {
        "title": "`final_success` heuristic from `test_output.txt` mis-classified 3 SWE-agent successes",
        "severity": "blocker (resolved)",
        "kind": "label leakage / heuristic drift",
        "source": "datasets/observation_distribution_comparison.md § 3.6",
        "body": (
            "Pre-fix: builder's `resolve_final_success` keyword-scanned `test_output.txt`, "
            "misclassifying `s_03` / `s_06` / `s_09` as failures. Fix (commit 7df39ba): "
            "honor `source_metadata.json:final_success` whenever the importer pinned "
            "an authoritative label. Listed here as a load-bearing pilot finding even "
            "though it was schema-adjacent (heuristic-driven) rather than schema-shaped."
        ),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/swe_agent_pilot", type=Path)
    parser.add_argument("--output", default="runs/swe_agent_pilot/SCHEMA_GAPS.md", type=Path)
    args = parser.parse_args()
    records = pilot_records(args.runs_dir)
    args.output.write_text(render_markdown(records, EXTRA_FINDINGS), encoding="utf-8")
    print(f"wrote {args.output} ({len(records)} pilots, "
          f"{sum(1 for r in records if r['schema_gap_found'])} flagged, "
          f"{len(EXTRA_FINDINGS)} cross-workstream findings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
