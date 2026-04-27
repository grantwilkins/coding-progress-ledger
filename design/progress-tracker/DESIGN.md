# Coding Progress Ledger: Design

## 1. Purpose

This project implements the first primitive for a coding-agent progress observer.

The long-term goal is a 0–1 observation channel for coding tasks:

> Given the agent's current task decomposition, trajectory, code artifacts, and evidence, estimate how much of the currently discovered required work is complete.

This repository should **not** implement a policy, scheduler, controller, intervention mechanism, or agent orchestration framework. It should only implement the substrate needed to observe progress.

The key design idea is a **mutable discovered-work ledger**. A coding task begins with an initial set of subtasks. As the agent works, it may discover new required work, split vague subtasks into concrete ones, reopen previously completed work, or invalidate work that turned out to be wrong. Therefore, progress is allowed to decrease.

This is intentional.

Progress is not elapsed time. Progress is not the fraction of tokens consumed. Progress is not the fraction of the original plan completed. Progress is:

```text
completed active discovered work / total active discovered work
```

The denominator can change as the agent learns more about the task.

---

## 2. Non-Goals

Do not implement these in the first version:

- LLM prompting
- learned progress models
- embeddings
- public trace ingestion
- SWE-agent/OpenHands/Codex integration
- browser or terminal sandbox management
- task scheduling or control policies
- UI dashboards
- multi-agent orchestration
- automatic code analysis

The first version should be a small, deterministic, replayable ledger core.

---

## 3. Core Concepts

### 3.1 Root Task

The root task is the user-facing coding goal, for example:

```text
Fix timezone parser so colonless offsets like +0530 are accepted.
```

The root task is not directly scored. It is decomposed into subtasks.

### 3.2 Subtask

A subtask is a concrete unit of discovered work.

A valid subtask should be evidence-checkable. An external observer should be able to determine, from logs, diffs, tests, or other artifacts, whether the subtask is complete.

Good subtask:

```text
Add a regression test for +0530 timezone offset parsing.
```

Bad subtask:

```text
Fix the bug.
```

A subtask may have children. If it has active children, it is treated as an internal organizational node and does not directly count toward progress.

### 3.3 Ledger

The ledger is the current state of the discovered-work decomposition.

It contains:

- the root task
- all subtasks
- an append-only list of events

The ledger must be reconstructable by replaying its event log from scratch.

### 3.4 Ledger Event

Every mutation to the ledger must be represented as an append-only event.

Examples:

- add a subtask
- mark a subtask complete
- add evidence
- split a subtask into smaller subtasks
- reopen completed work
- invalidate work
- delete work from the active denominator while preserving history

The event log is the primary artifact. The current ledger is derived state.

### 3.5 Progress Observation

A progress observation is a deterministic summary of the current ledger:

```json
{
  "step": 8,
  "complete_weight": 2.0,
  "active_weight": 8.0,
  "progress": 0.25,
  "complete_leaf_count": 2,
  "active_leaf_count": 8
}
```

The scalar `progress` is derived from the numerator and denominator. It should never be emitted without the numerator and denominator.

---

## 4. Status Semantics

Each subtask has one status:

```text
not_started
in_progress
blocked
complete
invalidated
deleted
```

### 4.1 Active Statuses

These statuses count in the active denominator:

```text
not_started
in_progress
blocked
complete
```

These statuses do not count in the active denominator:

```text
invalidated
deleted
```

Invalidated and deleted subtasks remain in the event history.

### 4.2 Completion

A subtask may be marked `complete` only if evidence is provided.

Evidence should be a list of strings, for example:

```json
[
  "Opened src/timezone/parser.py and identified parse_offset as the relevant function.",
  "pytest tests/test_timezone.py::test_colonless_offset passed."
]
```

The first version should enforce only that at least one evidence string exists when marking a subtask complete. It does not need to semantically verify the evidence.

---

## 5. Scoring Semantics

Progress is computed over active leaf subtasks.

Definitions:

- A subtask is **active** if its status is neither `invalidated` nor `deleted`.
- A subtask is a **leaf** if it has no active children.
- Only active leaves count toward the denominator.
- Complete active leaves contribute their full weight to the numerator.
- Not-started, in-progress, and blocked active leaves contribute zero to the numerator in the first version.

Formula:

```text
progress = complete_weight / active_weight
```

If `active_weight == 0`, progress should be `0.0`.

### 5.1 Parent Nodes

If a subtask has active children, the parent does not count directly toward progress.

This prevents double-counting.

Example:

```text
S1: Implement parser fix
  S1.1: Add regression test
  S1.2: Modify parser
  S1.3: Run targeted tests
```

If `S1.1`, `S1.2`, and `S1.3` are active, then `S1` is an internal node and does not count toward numerator or denominator.

### 5.2 Reverse Progress

Progress can decrease when:

- new active subtasks are added
- a completed subtask is reopened
- a completed subtask is invalidated
- a completed subtask is split into unfinished children
- existing work is discovered to be incomplete

Example:

```text
Step 0: 0 / 4 = 0.00
Step 5: 2 / 4 = 0.50
Step 8: add 4 new subtasks -> 2 / 8 = 0.25
Step 9: reopen one completed subtask -> 1 / 8 = 0.125
```

This behavior is central to the project.

---

## 6. Event Types

The first version should support these event types:

```text
init
add_subtask
update_status
add_evidence
split_subtask
reopen_subtask
invalidate_subtask
delete_subtask
```

### 6.1 `init`

Initializes a ledger with a root task.

Payload:

```json
{
  "root_task": "Fix timezone parser bug"
}
```

### 6.2 `add_subtask`

Adds a new subtask.

Payload:

```json
{
  "id": "S1",
  "description": "Locate timezone parser implementation",
  "parent_id": null,
  "weight": 1.0,
  "status": "not_started"
}
```

### 6.3 `update_status`

Updates a subtask status.

Payload:

```json
{
  "status": "complete",
  "evidence": ["Opened src/timezone/parser.py and found parse_offset."]
}
```

If the new status is `complete`, evidence is required either in the event payload or already present on the subtask.

### 6.4 `add_evidence`

Adds evidence to a subtask.

Payload:

```json
{
  "evidence": ["pytest tests/test_timezone.py passed"]
}
```

### 6.5 `split_subtask`

Splits a subtask into child subtasks.

Payload:

```json
{
  "children": [
    {
      "id": "S1.1",
      "description": "Add regression test for +0530 offset",
      "weight": 1.0,
      "status": "not_started"
    },
    {
      "id": "S1.2",
      "description": "Modify parse_offset to accept colonless offsets",
      "weight": 1.0,
      "status": "not_started"
    }
  ]
}
```

The original subtask remains in the ledger as the parent. If it now has active children, it no longer directly counts toward progress.

### 6.6 `reopen_subtask`

Changes a completed subtask back to `in_progress`.

Payload:

```json
{
  "reason": "A later test showed the previous parser change was incomplete."
}
```

### 6.7 `invalidate_subtask`

Marks a subtask as `invalidated`.

Payload:

```json
{
  "reason": "This approach was abandoned after discovering the bug is in a different module."
}
```

Invalidated work remains in history but does not count in the active denominator.

### 6.8 `delete_subtask`

Marks a subtask as `deleted`.

Use this for cleanup when a subtask was added by mistake or is no longer relevant. Deleted work remains in history but does not count in the active denominator.

---

## 7. Invariants

The implementation must enforce these invariants:

1. Progress is computed from the current ledger, not from elapsed time.
2. Every ledger mutation is represented by an append-only event.
3. Replaying the event log reconstructs the same ledger and score.
4. A subtask cannot be marked complete without evidence.
5. Only active leaf subtasks count toward progress.
6. Parent subtasks with active children do not count directly.
7. Invalidated and deleted subtasks do not count in the active denominator.
8. Invalidated and deleted subtasks remain visible in history.
9. Progress can decrease when new work is discovered or completed work is reopened.
10. The score exposes numerator and denominator, not only the scalar progress.

---

## 8. Example Progress Trace

Initial task:

```text
Fix timezone parser so colonless offsets like +0530 are accepted.
```

Events:

```jsonl
{"step":0,"event_type":"init","subtask_id":null,"payload":{"root_task":"Fix timezone parser so colonless offsets like +0530 are accepted."},"reason":null}
{"step":1,"event_type":"add_subtask","subtask_id":"S1","payload":{"description":"Understand expected behavior for colonless timezone offsets","status":"not_started","weight":1.0,"parent_id":null},"reason":"Initial decomposition"}
{"step":1,"event_type":"add_subtask","subtask_id":"S2","payload":{"description":"Locate timezone parser implementation","status":"not_started","weight":1.0,"parent_id":null},"reason":"Initial decomposition"}
{"step":1,"event_type":"add_subtask","subtask_id":"S3","payload":{"description":"Modify parser to accept +HHMM and -HHMM offsets","status":"not_started","weight":1.0,"parent_id":null},"reason":"Initial decomposition"}
{"step":1,"event_type":"add_subtask","subtask_id":"S4","payload":{"description":"Run targeted parser tests","status":"not_started","weight":1.0,"parent_id":null},"reason":"Initial decomposition"}
{"step":4,"event_type":"update_status","subtask_id":"S1","payload":{"status":"complete","evidence":["Issue states +0530 should parse as UTC+05:30."]},"reason":"Expected behavior identified"}
{"step":5,"event_type":"update_status","subtask_id":"S2","payload":{"status":"complete","evidence":["Opened src/timezone/parser.py and found parse_offset."]},"reason":"Implementation localized"}
{"step":8,"event_type":"add_subtask","subtask_id":"S5","payload":{"description":"Add regression test for +0530 parsing","status":"not_started","weight":1.0,"parent_id":null},"reason":"Need test coverage for new behavior"}
{"step":8,"event_type":"add_subtask","subtask_id":"S6","payload":{"description":"Check serializer round-trip behavior for colonless offsets","status":"not_started","weight":1.0,"parent_id":null},"reason":"Parser change may affect serialization"}
```

Scores:

```text
After step 1: 0 / 4 = 0.00
After step 4: 1 / 4 = 0.25
After step 5: 2 / 4 = 0.50
After step 8: 2 / 6 = 0.333...
```

This is the desired reverse-progress behavior.

---

## 9. Long-Term Trajectory

This first version is the deterministic core. Later versions should add:

1. Manual CLI for recording ledger updates during coding runs.
2. LLM observer that proposes ledger events from recent trajectory and artifacts.
3. Expressive subtasks with explicit completion criteria.
4. Retrospective annotation of public agent traces.
5. Learned models for ledger update proposal and evidence auditing.
6. Probabilistic progress:

```text
expected completed active work / total active work
```

Eventually, hard status labels can become probabilities:

```text
P(subtask complete | task, trajectory, artifacts, evidence)
```

Then progress becomes:

```text
sum_i weight_i * P(subtask_i complete) / sum_i weight_i
```

But the first implementation should remain deterministic.
