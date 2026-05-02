#!/usr/bin/env python3
"""K1: per-pilot evidence-availability audit on the SWE-agent pilot.

Reuses scripts/rescore_suite_by_category.py:audit_completion_evidence
+ classify_evidence. Writes runs/swe_agent_pilot/EVIDENCE_AUDIT.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RESCORE_SPEC = importlib.util.spec_from_file_location(
    "rescore_suite_by_category", ROOT / "scripts" / "rescore_suite_by_category.py"
)
rescore = importlib.util.module_from_spec(_RESCORE_SPEC)
_RESCORE_SPEC.loader.exec_module(rescore)

audit_completion_evidence = rescore.audit_completion_evidence
classify_evidence = rescore.classify_evidence
evidence_level = rescore.evidence_level
STRONG = rescore.STRONG_EVIDENCE_TYPES
LEVELS = ("mechanical", "trace_semantic", "annotator_judgment")
CATS = ("product", "validation", "investigation")


def load_events(ledger_path: Path) -> list[dict]:
    return [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def per_pilot_summary(ledger_path: Path) -> dict:
    events = load_events(ledger_path)
    audit = audit_completion_evidence(events)
    type_counts: Counter = Counter()
    level_by_category = {c: {lv: 0 for lv in LEVELS} for c in CATS}
    categories: dict[str, str] = {}
    completion_events = 0
    completions_with_strong = 0
    completions_manual_only = 0
    for event in events:
        et = event.get("event_type")
        sid = event.get("subtask_id")
        payload = event.get("payload", {})
        if et == "add_subtask":
            categories[sid] = payload.get("category", "product")
        elif et == "split_subtask":
            parent_cat = categories.get(sid, "product")
            for child in payload.get("children", []):
                categories[child["id"]] = child.get("category", parent_cat)
        if et != "update_status" or payload.get("status") != "complete":
            continue
        completion_events += 1
        evidence = payload.get("evidence") or []
        types = classify_evidence(evidence)
        for t in types:
            type_counts[t] += 1
        if STRONG & types:
            completions_with_strong += 1
        if types == {"manual_note"}:
            completions_manual_only += 1
        cat = categories.get(sid, "product")
        if cat in level_by_category:
            level_by_category[cat][evidence_level(types)] += 1
    by_cat = audit["by_category"]
    return {
        "pilot_id": ledger_path.parent.name,
        "audit_status": audit["status"],
        "weak_total": audit["weak_completion_evidence_count"],
        "weak_by_category": {c: by_cat[c]["weak_completion_evidence_count"] for c in by_cat},
        "audited_by_category": {c: by_cat[c]["audited_completion_count"] for c in by_cat},
        "evidence_type_counts": dict(type_counts),
        "level_by_category": level_by_category,
        "completion_events": completion_events,
        "completions_with_strong": completions_with_strong,
        "completions_manual_only": completions_manual_only,
        "weak_subtasks": audit["weak_completion_evidence"],
    }


def aggregate(rows: list[dict]) -> dict:
    weak_by_cat = {c: 0 for c in CATS}
    audited_by_cat = {c: 0 for c in CATS}
    level_by_cat = {c: {lv: 0 for lv in LEVELS} for c in CATS}
    types: Counter = Counter()
    completion_events = 0
    completions_with_strong = 0
    completions_manual_only = 0
    for r in rows:
        for c in CATS:
            weak_by_cat[c] += r["weak_by_category"][c]
            audited_by_cat[c] += r["audited_by_category"][c]
            for lv in LEVELS:
                level_by_cat[c][lv] += r["level_by_category"][c][lv]
        for t, n in r["evidence_type_counts"].items():
            types[t] += n
        completion_events += r["completion_events"]
        completions_with_strong += r["completions_with_strong"]
        completions_manual_only += r["completions_manual_only"]
    level_totals = {lv: sum(level_by_cat[c][lv] for c in CATS) for lv in LEVELS}
    return {
        "pilots": len(rows),
        "weak_by_category": weak_by_cat,
        "audited_by_category": audited_by_cat,
        "level_by_category": level_by_cat,
        "level_totals": level_totals,
        "type_totals": dict(types),
        "completion_events": completion_events,
        "completions_with_strong": completions_with_strong,
        "completions_manual_only": completions_manual_only,
    }


def render_md(rows: list[dict], totals: dict) -> str:
    out = ["# Evidence availability audit (K1)", ""]
    out.append("Reuses `scripts/rescore_suite_by_category.py` "
               "(`audit_completion_evidence`, `classify_evidence`). "
               "Strong evidence = `test_output | diff | file_exists | command_output` "
               "(plus `contract_text` for INVESTIGATION leaves on understanding-style descriptions).")
    out.append("")
    out.append("## 1. Headline")
    out.append("")
    out.append(f"- Pilots audited: **{totals['pilots']}**")
    out.append(f"- Total completion events: **{totals['completion_events']}**")
    out.append(f"- Completions with at least one strong evidence type: **{totals['completions_with_strong']}** ({totals['completions_with_strong']/totals['completion_events']:.0%})")
    out.append(f"- Completions with `manual_note` only: **{totals['completions_manual_only']}** ({totals['completions_manual_only']/totals['completion_events']:.0%})")
    out.append("")
    out.append("## 2. Evidence-type counts (across all completion events)")
    out.append("")
    out.append("| Evidence type | Count |")
    out.append("|---|---:|")
    for t in sorted(totals["type_totals"], key=lambda k: -totals["type_totals"][k]):
        out.append(f"| `{t}` | {totals['type_totals'][t]} |")
    out.append("")
    out.append("## 2b. Evidence levels (K4)")
    out.append("")
    out.append("| Level | product | validation | investigation | total |")
    out.append("|---|---:|---:|---:|---:|")
    for lv in LEVELS:
        row = totals["level_by_category"]
        out.append(f"| {lv} | {row['product'][lv]} | {row['validation'][lv]} | {row['investigation'][lv]} | {totals['level_totals'][lv]} |")
    out.append("")
    out.append("Levels: `mechanical` = test/command output, diff, file_exists, tool_action; "
               "`trace_semantic` = contract_text on understanding-style leaves; "
               "`annotator_judgment` = manual_note fallback. Trace_semantic counts are candidates for live sidecar automation.")
    out.append("")
    out.append("## 3. Weak completions by category")
    out.append("")
    out.append("| Category | Audited completions | Weak | Weak rate |")
    out.append("|---|---:|---:|---:|")
    for c in ("product", "validation", "investigation"):
        a = totals["audited_by_category"][c]
        w = totals["weak_by_category"][c]
        rate = "—" if a == 0 else f"{w/a:.0%}"
        out.append(f"| {c} | {a} | {w} | {rate} |")
    out.append("")
    out.append("## 4. Per-pilot status")
    out.append("")
    out.append("| Pilot | Status | Weak total | weak prod / val / inv |")
    out.append("|---|---|---:|---|")
    for r in sorted(rows, key=lambda x: x["pilot_id"]):
        wbc = r["weak_by_category"]
        out.append(f"| `{r['pilot_id']}` | {r['audit_status']} | {r['weak_total']} | "
                   f"{wbc['product']} / {wbc['validation']} / {wbc['investigation']} |")
    out.append("")
    out.append("## 5. Notes on interpretation")
    out.append("")
    out.append("- Weak evidence is a **signal**, not a replay failure. The framework allows `manual_note` evidence on completed subtasks; this audit measures *how often* that fallback fires.")
    out.append("- A pilot's status is `weak` if any audited completion has only weak evidence; `strong` if all completions have at least one strong evidence type. `not_applicable` if the pilot has zero PRODUCT/VALIDATION/INVESTIGATION completions in scope.")
    out.append("- The `contract_text` carve-out for INVESTIGATION captures \"understanding what the issue asks for\" as legitimate completion evidence on a leaf whose description is about contract-reading.")
    out.append("")
    out.append("## 6. Pointers")
    out.append("")
    out.append("- Classifier: `scripts/rescore_suite_by_category.py:classify_evidence`")
    out.append("- Audit fn: `scripts/rescore_suite_by_category.py:audit_completion_evidence`")
    out.append("- This script: `scripts/audit_pilot_evidence.py`")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/swe_agent_pilot", type=Path)
    parser.add_argument("--output", default="runs/swe_agent_pilot/EVIDENCE_AUDIT.md", type=Path)
    parser.add_argument("--json-out", default="runs/swe_agent_pilot/EVIDENCE_AUDIT.json", type=Path)
    args = parser.parse_args(argv)

    ledger_paths = sorted((args.runs_dir).glob("swe_agent_pilot_*/ledger.jsonl"))
    if not ledger_paths:
        raise FileNotFoundError(f"no ledger.jsonl files under {args.runs_dir}")
    rows = [per_pilot_summary(p) for p in ledger_paths]
    totals = aggregate(rows)
    args.output.write_text(render_md(rows, totals), encoding="utf-8")
    args.json_out.write_text(json.dumps({"per_pilot": rows, "totals": totals}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.output} and {args.json_out} ({totals['pilots']} pilots, "
          f"{totals['completions_with_strong']} strong completions, "
          f"{totals['completions_manual_only']} manual-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
