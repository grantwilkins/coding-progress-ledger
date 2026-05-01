# LedgerSet protocol (v1)

This document defines `LedgerSet`, the unit of analysis for multi-task /
long-range agentic work. It is source-agnostic: the same protocol covers
a refactor decomposed into N sub-issues, a research agent's experiment
sequence, and a SWE-bench batch viewed as N issues. Source-specific
addenda (e.g. `docs/SWE_AGENT_LEDGER_SET_ADDENDUM.md`) refine how this
protocol lands on a particular trace shape; they do not override.

The single-`Ledger` pipeline is unchanged. This protocol layers above
it; it does not replace it. See
`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` for the per-trace
annotation rules, and `docs/AGENT_USAGE.md` for a single-ledger
workflow walk-through.

Ground truth for the type system is the in-repo source, not this doc:

- `ledger_progress/core.py:Status` — six statuses
- `ledger_progress/core.py:SubtaskCategory` — six categories
- `ledger_progress/queries.py:CODING_CATEGORIES` — the
  `(PRODUCT, VALIDATION, INVESTIGATION)` triple used to compute the
  coding sub-progress curve
- `ledger_progress/session.py:LedgerSession` — the single-task session
  ergonomics that `LedgerSetSession` mirrors one level up

If this protocol drifts from those files, **the files win.**

## 1. Purpose and scope

A `LedgerSet` is a thin, source-agnostic container over an ordered
collection of `Ledger`s. Real long-range agentic work has a coarser
unit of analysis than a single trace: a multi-week refactor decomposed
into 30 sub-issues, a research agent running a sequence of partly-
independent experiments, a SWE-bench run viewed as one issue out of N.
The set layer makes that unit explicit without disturbing single-task
machinery.

The set layer reads finished ledgers and aggregates. It does not
annotate. The discovered-vs-hidden distinction and the anti-narrative
stance are properties of one trace; they do not lift to set level.

## 2. Data model

```text
LedgerSet:
  set_id: str
  members: list[LedgerSetMember]

LedgerSetMember:
  member_id: str
  ledger_ref: path-or-handle      # points to an existing Ledger
  weight: float = 1.0
  status_override: Status | None = None
```

`ledger_ref` points to an existing `Ledger`; the set never owns ledger
bytes. A ledger may belong to multiple sets. `status_override` uses
`Status` from `ledger_progress/core.py` and is only set when a member's
outcome is decided outside its ledger (e.g. a sub-issue declared
out-of-scope after the trace ended).

The existing `Ledger` / `LedgerSession` / `Subtask` types are unchanged.
No fields beyond the four above exist on a member in v1.

## 3. Minimal API

`LedgerSetSession(set_id)` mirrors `LedgerSession` (see
`ledger_progress/session.py`) one level up. It exposes:

- `add_member(ledger_ref, weight=1.0)` — register a finished ledger as
  a member of this set.
- `mark_member(member_id, status)` — set `status_override` when the
  member's outcome is decided outside the ledger. Use sparingly; the
  preferred channel is the member's own ledger.
- `score()` — return set-level progress per § 4.
- `export_jsonl()` — serialize the set to a `set.jsonl` file with the
  same replay/serialization parity that `Ledger` has.

There are no splits, reopens, or reparenting at the set level in v1.
A member's progress shape is the member's ledger's job.

## 4. Aggregation rule (v1)

Set-level progress is the **weight-weighted mean of per-member
coding-progress**:

```text
set_progress =
    sum(weight * member.score(CODING_CATEGORIES).progress)
  / sum(weight)
```

`CODING_CATEGORIES` is the `(PRODUCT, VALIDATION, INVESTIGATION)`
triple from `ledger_progress/queries.py`.

Members with `status_override in {INVALIDATED, DELETED}` (`Status`
values from `ledger_progress/core.py`) drop out of both numerator and
denominator. This matches single-task semantics, where invalidated /
deleted leaves leave the active denominator.

Default `weight = 1.0` gives a uniform mean. Explicit weights are the
annotator's call: leaf-count weighting is rejected because a 30-leaf
member would otherwise dominate a 3-leaf member when both are "one
sub-issue."

## 5. What does NOT change

The following continue to operate on single ledgers, untouched by the
set layer:

- the B2 sampler
- the C3 importer
- the D1 retrospective annotation protocol
- the E1 annotation pass
- the F2 / F3 dataset builders
- the `ledger-run` CLI

Reason: discovered-vs-hidden and the anti-narrative stance
(see `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` § 2–3) are
properties of one trace. Promoting them to set level would invite
retro-fitting cross-task dependencies the traces don't surface.

## 6. Migration

The 20 SWE-agent pilot ledgers each become a `LedgerSet` of size 1 via
a trivial wrapper (T4). Their per-run artifacts are untouched; only a
sibling `set.jsonl` appears at `runs/swe_agent_pilot/<pilot_id>/`. The
suite-level rollup at `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`
becomes the first non-trivial set: one `LedgerSet` of 20 members,
weight 1.0 each.

## 7. Rejected alternatives

1. **Make `Ledger` itself recursive** (a ledger may contain ledgers).
   Overloads `Subtask` semantics — a subtask is a leaf-or-parent unit
   of in-trace discovered work, not a pointer to another trace. Forces
   scoring to traverse mixed leaf / sub-ledger trees, which complicates
   replay and breaks the "one trace, one ledger" property the
   per-trace protocol depends on.
2. **Track sets only as a CSV manifest with no runtime type.** Pushes
   aggregation policy into ad-hoc scripts, loses replay / serialization
   parity with `Ledger`, and prevents the set layer from being a stable
   target for downstream consumers (Q's predictive modeling, source-
   specific addenda).

## 8. Out of scope for v1

The following are deferred to future addenda. Adding any of them
without a real project that needs the signal is out of scope:

- cross-member dependencies / DAG edges
- time windows (set-level timestamps, member start/end ordering)
- cross-member evidence (one member's artifact cited by another)
- set-level reopens / splits
- alternative aggregation rules (leaf-count-weighted, min, max,
  percentile)

## 9. Open questions

These are preserved from the workstream-T discussion in `TASKS.md` and
should be revisited when a real project surfaces evidence:

1. Should a member carry a *contribution-to-set* annotation distinct
   from `weight` (e.g. "blocking" vs "nice-to-have")? Deferred —
   `weight` covers v1; promote only if real projects need a richer
   signal.
2. How does set-level progress interact with members at `BLOCKED`
   (`Status.BLOCKED` in `ledger_progress/core.py`)? v1 treats blocked
   members as in-progress: the member's own ledger score is used as-is
   in the weighted mean, and the set is not itself "blocked" as a
   status. Revisit if a real multi-issue project surfaces a counter-
   example.
3. Workstream Q (predictive modeling) may eventually want set-level
   features. T does not pre-build them; Q can opt in by consuming
   `score_set` once it lands. Per the locked-in framing, Q's
   prediction target is "on-time finish, regardless of failure" —
   set-level progress should support that without privileging the
   upstream success label.
