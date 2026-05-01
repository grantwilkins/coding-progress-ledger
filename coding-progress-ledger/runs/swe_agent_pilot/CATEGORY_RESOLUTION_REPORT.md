# J1: native-category resolution on the SWE-agent pilot

This satisfies `TASKS.md` § Workstream J, task **J1**. It explains the
"181 mixed / 10 native" discrepancy reported by the prior step-table
audit (`datasets/swe_agent_pilot_observations_step_audit.md` before
this work) and reports the post-fix state.

## 1. Headline

| State | Native | Mixed | Legacy_inferred | Native/resolved warnings |
|-------|--------|-------|-----------------|--------------------------|
| Before fix (audit at HEAD prior to this commit) | 10 / 191 | 181 / 191 | 0 | 1 (`s_03`) |
| After fix (this commit) | **191 / 191** | 0 | 0 | **0** |

Event-level table moves to `202 / 202 native`, also clean.

## 2. Root cause

`LedgerSession.add()` had a "PRODUCT is the default — don't write it
to the payload" optimization:

```python
# ledger_progress/session.py (pre-fix)
if category is not SubtaskCategory.PRODUCT:
    payload["category"] = category
```

The annotation specs always set `category` explicitly (the protocol
forces it, the spec-driven driver passes it through). But for any
PRODUCT subtask the session helper **stripped the field** before
serialization. Replay still worked because `core.py:_add_subtask`
defaults to `PRODUCT` when category is missing — so the *semantics*
were preserved.

The dataset builder, however, can't distinguish "annotator
explicitly said PRODUCT" from "annotator omitted category and the
core defaulted." `resolve_categories()` counts payload-without-category
as `missing`, and any run with at least one such event becomes
`mixed`. Because every SWE-agent pilot has at least one PRODUCT
subtask, almost every pilot ended up `mixed`. The one outlier
(`s_03`) — the run with the largest divergence warning — was the
single pilot whose specs all happened to use non-PRODUCT categories
on the path the legacy resolver took.

This is purely a serialization-level bug: it does **not** affect
score, replay, or any progress computation. Confirmed: progress
numbers for all 20 SWE-agent pilots are byte-identical before and
after the fix (E1 numbers reproduced verbatim).

## 3. Fix

`ledger_progress/session.py:add()` now always emits `category`:

```python
payload = {
    "description": description,
    "parent_id": parent_id,
    "weight": weight,
    "category": category,
}
```

One pre-existing test (`tests/test_session.py::test_session_calls_match_manual_events_score_and_replay`)
hand-built `LedgerEvent` objects without `category` and asserted
event-equality with session-built events. Updated those manual fixtures
to include `category=SubtaskCategory.PRODUCT` so the event-equality
invariant still holds. No semantic change.

`split()` was inspected: child entries already write `category`
when the spec sets one, and the SWE-agent corpus has zero `split`
events (per F4 § 3.4), so no fix needed there. If future annotators
add splits with default PRODUCT children, the same pattern would
recur — addressed by J2's enforcement script which flags any
add/split-child without an explicit category.

## 4. Re-built artifacts

| File | Before | After |
|------|--------|-------|
| `runs/swe_agent_pilot/swe_agent_pilot_*/ledger.jsonl` | 20 files, mixed payload categories | 20 files, every ADD_SUBTASK payload carries explicit `category` |
| `runs/swe_agent_pilot_v3/swe_agent_pilot_*/ledger.jsonl` | 5 files (H4 v3) | 5 files, native |
| `datasets/swe_agent_pilot_observations_step.csv` | 191 rows (mixed) | 191 rows (native) |
| `datasets/swe_agent_pilot_observations_event.csv` | 202 rows (mixed) | 202 rows (native) |
| `datasets/swe_agent_pilot_observations_*_audit.md` | one warning, mixed totals | clean, all native |

`ledger-run check-run` passes on all 20 pilot run dirs and all 5 v3
runs after re-emission. 247-test suite remains green.

## 5. Why this matters for the observation channel

The mission-critical claim from M1 is that the channel exposes
*category-specific progress* as a first-class feature. With ~95% of
step rows tagged `mixed`, downstream consumers (Q's predictive
modeling, R's writeup) couldn't trust the per-category numerator/
denominator without re-deriving from the raw events themselves —
defeating the purpose of the resolved columns. After the fix, every
category-progress column on the SWE-agent corpus is unambiguously
sourced from explicit annotator decisions.

## 6. Pointers

- Fix commit: this commit
- Builder logic: `scripts/build_ledger_observation_dataset.py:resolve_categories`
- Session source: `ledger_progress/session.py:add`
- J2 enforcement: `scripts/check_native_categories.py`
- Audit outputs: `datasets/swe_agent_pilot_observations_step_audit.md`,
  `datasets/swe_agent_pilot_observations_event_audit.md`
