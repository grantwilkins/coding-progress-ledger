# N4 — Live-vs-retrospective parity report

This compares the two N3 live sidecar ledgers against the retrospective SWE-agent pilot ledgers for the same instances. Shape classes are primary; scalar progress is secondary.

## Verdict

N4 parity gate does not pass yet; do not proceed to N5 live N=20.

| Instance | Success | Retrospective shape | Live shape | Retrospective coding | Live coding | Delta | Scalar within 0.05 |
|---|---:|---|---|---:|---:|---:|---|
| `Melevir__cognitive_complexity-15` | True | `complete_visible_frontier+validation_complete` | `complete_visible_frontier+validation_complete` | 1.000 | 1.000 | +0.000 | yes |
| `WIPACrepo__iceprod-339` | False | `partial_visible_frontier+validation_gap` | `complete_visible_frontier+no_validation_frontier` | 0.667 | 1.000 | +0.333 | no |

## Schema And Shape Parity

| Instance | Retrospective events | Live events | Retrospective categories | Live categories | Retrospective statuses | Live statuses |
|---|---:|---:|---|---|---|---|
| `Melevir__cognitive_complexity-15` | 13 | 43 | `artifact:1, investigation:1, product:2, validation:2` | `artifact:1, investigation:14, product:4, validation:2` | `complete:6` | `complete:21` |
| `WIPACrepo__iceprod-339` | 8 | 17 | `artifact:1, investigation:1, product:1, validation:1` | `artifact:1, investigation:6, product:1` | `complete:3` | `complete:8` |

## Divergences

| Divergence | Instances | Assignment | Consequence |
|---|---|---|---|
| Live sidecar emits one leaf per visible assistant command; retrospective ledgers collapse many commands into semantic work leaves. | both | true semantic ambiguity for raw-step adapter | Event counts differ even when final shape matches. Explicit `ledger_ops` or a smarter adapter is needed for semantic grouping. |
| Retrospective `WIPACrepo__iceprod-339` includes an unstarted validation leaf; live sidecar has no validation leaf because the agent emitted no validation command. | `WIPACrepo__iceprod-339` | missing instrumentation / semantic obligation not emitted | Live coding progress is 1.000 while retrospective coding progress is 0.667; this is the scalar parity failure. |
| Live ledgers have timestamps; retrospective ledgers do not. | both | expected instrumentation difference | Timestamp-aware features can run on N3 live ledgers but cannot be compared to retrospective wall-clock intervals. |

## K2 Evidence-Gap Check

| K2 gap | N4 result | Classification |
|---|---|---|
| Structured edit/submit action evidence | Closed for emitted live actions: `wire_events.jsonl` carries `tool_name`, `command`, observation, and terminal `exit_status`. | closed for N3 emitted actions |
| Agent-vs-harness submit provenance | Partially closed: N3 records terminal `exit_status` on the final emitted assistant action, but the selected pair contains explicit submit-style traces rather than the six harness-forced ambiguous pilots. | partial |
| Baseline failing test output before edits | Not closed: N3 replays normalized traces and does not run pre-fix tests. | open |
| Full command stdout/stderr beyond source truncation | Not closed: N3 uses the same normalized observations available retrospectively. | open |
| Per-edit before/after file state | Not closed: N3 records commands and observations, not file snapshots around every edit. | open |
| Hidden-work/repro validity gap | Not closed by this adapter: the live trace preserves visible commands but does not decide whether a repro exercised the issue. | open |

## Observability Matrix

See `EVENT_OBSERVABILITY_MATRIX.md`. Summary: mechanical events are available for emitted tool actions; validation obligations, blocked states, reopens, invalidations, and semantic splits remain annotation-only or weakly inferable unless the agent emits explicit `ledger_ops`.

## Timestamp Realism

Every N3 live ledger event has a non-null timestamp, while the retrospective pilot ledgers have none. The intervals are replay-time timestamps from normalized traces, not real SWE-agent wall-clock durations; they are sufficient to exercise timestamp plumbing but not to calibrate deadline models.
