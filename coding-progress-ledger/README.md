# Coding Progress Ledger

This is a deterministic progress ledger, not a task manager. It records append-only events about discovered coding work and computes progress over active discovered work:

```text
completed active leaf work / total active leaf work
```

Progress can go backward when new work is discovered, completed work is reopened, or work is invalidated. The event log is the source of truth and can be replayed to reconstruct the same ledger and score.

This first version is intentionally not an LLM system. It has no agent, prompt wrapper, trace ingester, trainer, scheduler, or policy logic.

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
