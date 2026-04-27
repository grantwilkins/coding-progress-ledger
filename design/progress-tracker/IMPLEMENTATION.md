# Coding Progress Ledger: Implementation Handoff

## 1. Assignment

Implement the standalone deterministic core for a coding-agent progress ledger.

This is the first brick for a future 0–1 progress observation channel. The goal is to represent discovered subtasks, status transitions, evidence, and reverse progress when new work is added or completed work is invalidated.

Do **not** build an LLM agent, prompt wrapper, trace ingester, model trainer, or policy system.

Build only:

- core data models
- event application
- deterministic scoring
- replay from event log
- JSONL serialization
- unit tests proving the progress semantics

---

## 2. Repository Structure

Create this structure:

```text
coding-progress-ledger/
  ledger_progress/
    __init__.py
    core.py
    serialization.py
    scoring.py
  tests/
    test_progress_semantics.py
    test_replay.py
    test_serialization.py
  examples/
    reverse_progress.jsonl
  pyproject.toml
  README.md
```

Keep the implementation simple. Prefer standard library dataclasses and enums unless there is a strong reason to use Pydantic.

---

## 3. Required Public API

The package must expose these functions:

```python
new_ledger(root_task: str) -> Ledger
apply_event(ledger: Ledger, event: LedgerEvent) -> Ledger
score(ledger: Ledger) -> ProgressObservation
replay(events: list[LedgerEvent]) -> Ledger
to_jsonl(ledger: Ledger, path: str) -> None
from_jsonl(path: str) -> Ledger
```

Optional but useful:

```python
load_events_jsonl(path: str) -> list[LedgerEvent]
write_events_jsonl(events: list[LedgerEvent], path: str) -> None
```

---

## 4. Data Models

Implement in `ledger_progress/core.py`.

### 4.1 Status Enum

```python
class Status(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    INVALIDATED = "invalidated"
    DELETED = "deleted"
```

### 4.2 EventType Enum

```python
class EventType(Enum):
    INIT = "init"
    ADD_SUBTASK = "add_subtask"
    UPDATE_STATUS = "update_status"
    ADD_EVIDENCE = "add_evidence"
    SPLIT_SUBTASK = "split_subtask"
    REOPEN_SUBTASK = "reopen_subtask"
    INVALIDATE_SUBTASK = "invalidate_subtask"
    DELETE_SUBTASK = "delete_subtask"
```

### 4.3 Subtask

```python
@dataclass
class Subtask:
    id: str
    description: str
    status: Status
    evidence: list[str] = field(default_factory=list)
    weight: float = 1.0
    parent_id: str | None = None
    created_at_step: int = 0
    updated_at_step: int = 0
```

Validation rules:

- `id` must be non-empty.
- `description` must be non-empty.
- `weight` must be positive.
- `parent_id`, if present, must refer to an existing subtask at event-application time.

### 4.4 LedgerEvent

```python
@dataclass
class LedgerEvent:
    step: int
    event_type: EventType
    subtask_id: str | None
    payload: dict[str, Any]
    reason: str | None = None
```

Validation rules:

- `step >= 0`
- `payload` is a dictionary
- event type determines required payload fields

### 4.5 Ledger

```python
@dataclass
class Ledger:
    root_task: str
    subtasks: dict[str, Subtask] = field(default_factory=dict)
    events: list[LedgerEvent] = field(default_factory=list)
```

### 4.6 ProgressObservation

```python
@dataclass
class ProgressObservation:
    step: int
    complete_weight: float
    active_weight: float
    progress: float
    complete_leaf_count: int
    active_leaf_count: int
```

---

## 5. Event Application Semantics

`apply_event(ledger, event)` should return an updated ledger. It may mutate the input or return a copy, but behavior must be deterministic and tests must be clear.

Every successful event application must append the event to `ledger.events`.

### 5.1 `new_ledger(root_task)`

Creates a ledger with the root task and an initial `init` event.

Expected behavior:

```python
ledger = new_ledger("Fix parser bug")
assert ledger.root_task == "Fix parser bug"
assert len(ledger.subtasks) == 0
assert len(ledger.events) == 1
assert ledger.events[0].event_type == EventType.INIT
```

### 5.2 `add_subtask`

Required payload fields:

```python
{
    "description": str,
    "status": str | Status,        # default allowed: not_started
    "weight": float,               # default allowed: 1.0
    "parent_id": str | None         # default allowed: None
}
```

Use `event.subtask_id` as the new subtask id.

Rules:

- `subtask_id` is required.
- id must not already exist.
- parent, if given, must exist.
- status defaults to `not_started` if omitted.
- weight defaults to `1.0` if omitted.

### 5.3 `update_status`

Required payload fields:

```python
{
    "status": str | Status,
    "evidence": list[str] | None
}
```

Rules:

- subtask must exist.
- if evidence is provided, append it to the subtask evidence list.
- if new status is `complete`, the subtask must have at least one evidence string after applying the event.
- update `updated_at_step`.

### 5.4 `add_evidence`

Required payload fields:

```python
{
    "evidence": list[str]
}
```

Rules:

- subtask must exist.
- evidence list must be non-empty.
- append evidence strings.
- update `updated_at_step`.

### 5.5 `split_subtask`

Required payload fields:

```python
{
    "children": [
        {
            "id": str,
            "description": str,
            "status": str | Status,  # default allowed: not_started
            "weight": float          # default allowed: 1.0
        }
    ]
}
```

Rules:

- parent subtask must exist.
- each child id must be unique and not already exist.
- each child gets `parent_id` equal to the parent subtask id.
- child status defaults to `not_started`.
- child weight defaults to `1.0`.
- the parent remains in the ledger.
- if the parent now has active children, the parent no longer directly counts toward progress.

This operation may decrease progress if a previously complete parent is split into unfinished children.

### 5.6 `reopen_subtask`

Rules:

- subtask must exist.
- status becomes `in_progress`.
- append reason as evidence only if you want, but do not require it.
- update `updated_at_step`.

This operation should decrease progress if the subtask was previously a complete active leaf.

### 5.7 `invalidate_subtask`

Rules:

- subtask must exist.
- status becomes `invalidated`.
- update `updated_at_step`.
- event remains in history.

Invalidated subtasks do not count in the active denominator.

### 5.8 `delete_subtask`

Rules:

- subtask must exist.
- status becomes `deleted`.
- update `updated_at_step`.
- event remains in history.

Deleted subtasks do not count in the active denominator.

---

## 6. Scoring Semantics

Implement in `ledger_progress/scoring.py`, or in `core.py` if simpler.

### 6.1 Active Subtasks

A subtask is active if:

```python
subtask.status not in {Status.INVALIDATED, Status.DELETED}
```

### 6.2 Active Children

A subtask has active children if any subtask has `parent_id == subtask.id` and is active.

### 6.3 Active Leaves

A subtask is an active leaf if:

```python
is_active(subtask) and not has_active_children(subtask)
```

Only active leaves are scored.

### 6.4 Complete Weight

For each active leaf:

```python
if subtask.status == Status.COMPLETE:
    contributes subtask.weight to complete_weight
else:
    contributes 0 to complete_weight
```

In the first version, `in_progress` and `blocked` contribute zero.

### 6.5 Active Weight

For each active leaf:

```python
active_weight += subtask.weight
```

### 6.6 Progress

```python
if active_weight == 0:
    progress = 0.0
else:
    progress = complete_weight / active_weight
```

Use the latest event step as `ProgressObservation.step`. If there are no events, use `0`.

---

## 7. Serialization

Implement in `ledger_progress/serialization.py`.

Use JSONL as the source-of-truth storage format.

Each line should be one serialized `LedgerEvent`.

Example:

```jsonl
{"step":0,"event_type":"init","subtask_id":null,"payload":{"root_task":"Fix parser bug"},"reason":null}
{"step":1,"event_type":"add_subtask","subtask_id":"S1","payload":{"description":"Locate parser implementation","status":"not_started","weight":1.0,"parent_id":null},"reason":"Initial decomposition"}
```

Required functions:

```python
to_jsonl(ledger: Ledger, path: str) -> None
from_jsonl(path: str) -> Ledger
```

`from_jsonl` should:

1. read all events
2. replay them
3. return the reconstructed ledger

Also implement helpers if useful:

```python
event_to_dict(event: LedgerEvent) -> dict
event_from_dict(data: dict) -> LedgerEvent
```

Enums should serialize to their string values.

---

## 8. Replay Semantics

`replay(events)` should reconstruct the ledger from an event list.

Rules:

- The first event should be `init`.
- The init payload must contain `root_task`.
- Replay should apply each subsequent event in order.
- The reconstructed ledger should include the same event list.
- Scoring the replayed ledger should match scoring the original ledger.

Implementation detail:

Be careful not to duplicate events during replay. A clean approach is:

1. create an empty ledger from the init event without appending through `new_ledger`, or
2. allow `new_ledger` to create the init event and then apply only events after init, ensuring event lists match.

Tests should verify no duplicated init event.

---

## 9. Required Tests

Use `pytest`.

### 9.1 `test_empty_ledger_score`

```python
ledger = new_ledger("Fix bug")
obs = score(ledger)
assert obs.active_weight == 0
assert obs.complete_weight == 0
assert obs.progress == 0.0
```

### 9.2 `test_four_not_started_subtasks_progress_zero`

Create four subtasks. Score should be:

```text
0 / 4 = 0.0
```

### 9.3 `test_complete_two_of_four`

Create four subtasks. Mark two complete with evidence.

Expected:

```text
complete_weight = 2
active_weight = 4
progress = 0.5
```

### 9.4 `test_reverse_progress_when_new_work_added`

After completing two of four, add four more subtasks.

Expected:

```text
complete_weight = 2
active_weight = 8
progress = 0.25
```

### 9.5 `test_reopen_completed_subtask_decreases_progress`

After the previous test, reopen one completed subtask.

Expected:

```text
complete_weight = 1
active_weight = 8
progress = 0.125
```

### 9.6 `test_split_completed_parent_decreases_progress`

Scenario:

1. Add one subtask `S1`.
2. Mark `S1` complete with evidence.
3. Score is `1 / 1 = 1.0`.
4. Split `S1` into three unfinished children.
5. Parent no longer counts because it has active children.
6. Children count as active leaves.

Expected:

```text
complete_weight = 0
active_weight = 3
progress = 0.0
```

### 9.7 `test_complete_without_evidence_raises`

Trying to mark a subtask complete without evidence should raise `ValueError`.

### 9.8 `test_invalidated_subtask_not_counted_but_kept_in_history`

Create two subtasks. Complete one. Invalidate the completed one.

Expected:

- invalidated subtask remains in `ledger.subtasks`
- invalidation event remains in `ledger.events`
- invalidated subtask does not count in active denominator

### 9.9 `test_deleted_subtask_not_counted_but_kept_in_history`

Same as invalidation but with delete.

### 9.10 `test_parent_with_active_children_not_counted`

Add parent and children. Ensure only active leaves are scored.

### 9.11 `test_replay_matches_original_score`

Create a ledger with multiple events. Replay its events.

Expected:

```python
assert score(original) == score(replayed)
```

### 9.12 `test_jsonl_round_trip_matches_score`

Write ledger to JSONL, read it back, score both.

Expected:

```python
assert score(original) == score(loaded)
```

---

## 10. Example File

Create `examples/reverse_progress.jsonl` with an event log showing:

```text
Step 0: init
Step 1: add four subtasks
Step 4: complete two subtasks
Step 8: add four more subtasks
Step 9: reopen one completed subtask
```

The expected progress curve is:

```text
Step 1: 0 / 4 = 0.00
Step 4: 2 / 4 = 0.50
Step 8: 2 / 8 = 0.25
Step 9: 1 / 8 = 0.125
```

---

## 11. README Requirements

The README should explain:

1. This is a deterministic progress ledger, not a task manager.
2. Progress is over active discovered work.
3. Progress can go backward when new work is discovered.
4. The event log is append-only and replayable.
5. The first version is intentionally not an LLM system.

Include a short usage example:

```python
from ledger_progress import new_ledger, apply_event, score, LedgerEvent, EventType

ledger = new_ledger("Fix timezone parser")
ledger = apply_event(ledger, LedgerEvent(
    step=1,
    event_type=EventType.ADD_SUBTASK,
    subtask_id="S1",
    payload={"description": "Locate parser implementation"},
    reason="Initial decomposition",
))
print(score(ledger))
```

---

## 12. Quality Bar

The implementation is successful if:

- all required tests pass
- reverse progress is demonstrated
- JSONL replay is deterministic
- core code is small and readable
- no LLM, agent, or trace-ingestion logic is included
- the design cannot be mistaken for a generic TODO list

The most important behavior to preserve is:

```text
Progress is a function of completed work over currently active discovered work, and the discovered-work denominator can grow.
```
