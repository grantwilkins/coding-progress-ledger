# Schema decision (I2) — SWE-agent retrospective pilot

This satisfies `TASKS.md` § Workstream I, task **I2**. It is the
explicit pilot-end answer to "do we need schema changes before
scaling?". Source for the inputs: `runs/swe_agent_pilot/SCHEMA_GAPS.md`
(I1).

## Decision

**No schema change needed for the pilot. Only annotation-protocol
changes — and those have already landed.**

## What "schema change" would have meant

- A new `EventType`, `Status`, or `SubtaskCategory` enum value in
  `ledger_progress/core.py`.
- A new required field on `Subtask` / `LedgerEvent` / `Ledger`.
- A breaking change to `serialization.from_jsonl` / replay semantics.
- A new top-level CSV column the dataset builder must emit.

None of those is required by the 20-pilot evidence.

## What the pilot evidence actually surfaced

From I1 (`SCHEMA_GAPS.md`):

| Finding | Class | Resolution |
|---|---|---|
| `f_02` stuck-loop rule covered only command loops | annotation protocol (general § 6) | resolved in-pilot — added § 6(b) tool-response-loop variant |
| `f_07` stuck-loop rule ambiguous on cycle length | annotation protocol (general § 6) | resolved in-pilot — refined to "any cycle length ≥ 1" |
| v1 inconsistently applied Pitfall #8 across harness-terminated failure pilots | annotation protocol (addendum § 5) | already-landed H3 Revision 1 codifies the rule; H4 follow-up applies it consistently |
| Builder reports `category_resolution_mode = mixed` for 181/191 SWE-agent step rows | pipeline / category resolution | Workstream **J1** investigates and resolves |
| `final_success` heuristic mis-classified 3 SWE-agent successes | label-leakage / heuristic | resolved in commit 7df39ba — `resolve_final_success` now honors `source_metadata.json` |

Every entry is either:
- a protocol-text refinement (general doc or addendum), or
- a pipeline / heuristic fix in scripts (not in the core data model), or
- a protocol-application gap to be closed by re-emitting specs.

The core schema (`ledger_progress/core.py`) has been **unchanged** since
the start of the pilot. 230+ tests pass against it; 235+ after H4.

## What is explicitly *not* changing

```text
ledger_progress/core.py:Status              — six values, unchanged
ledger_progress/core.py:EventType           — eight values, unchanged
ledger_progress/core.py:SubtaskCategory     — six values, unchanged
ledger_progress/core.py:Subtask             — fields unchanged
ledger_progress/core.py:LedgerEvent         — fields unchanged
ledger_progress/core.py:Ledger              — fields unchanged
ledger_progress/scoring.py:score            — semantics unchanged
ledger_progress/serialization.py:from_jsonl — semantics unchanged
```

## What *is* changing in the protocol docs (already landed)

Already committed via 2656391 (H3 revisions):

- General § 6 stuck-loop rule: cycle-of-1, cycle-of-2, tool-response-loop variant.
- General § 6 wording: "third iteration begins" → assistant-turn step disambiguation.
- General § 9: granularity-is-annotator-latitude acknowledgment.
- Addendum § 5 Pitfall #6 (harness-forced termination ≠ submit).
- Addendum § 5 Pitfall #7 (`final_diff.patch` is state, not action).
- Addendum § 5 Pitfall #8 (bug-fix tasks always have implicit VAL).
- Addendum § 1: `__init__.py` re-exports default rule.

## Outcome line (per § I2 acceptance)

> **No schema change needed for pilot. Only annotation protocol
> changes needed — already landed.**

Compatible with the M1 recommendation (scale retrospective to 100,
gated on H4) and the H4 follow-up (consistent Pitfall #8 application
on `f_02` / `f_03` / `f_07` / `f_10`).

## Pointers

- I1 collection: `runs/swe_agent_pilot/SCHEMA_GAPS.md`
- I1 collector script: `scripts/collect_schema_gaps.py`
- Protocol revisions: `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md`
- Core schema source: `ledger_progress/core.py`
- M1 memo: `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`
- H4 gate: `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`
