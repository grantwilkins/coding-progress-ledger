## Run notes — `swe_agent_pilot_s_04` (`joke2k__django-environ-174`)

- annotator: Claude (E1)
- annotation pass: E1
- protocol: `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
- source addendum: `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`
- upstream success label (NOT a feature): `True`

### 1. Initial reading

The issue cites `Env.db_url_config()` and points at the exact line
where the engine is set from `DB_SCHEMES`. The agent should add a
branch that lets a user-supplied custom `engine` parameter survive
the `url.scheme not in DB_SCHEMES` check.

### 3. Checkpoint notes

- step 13: `environ/environ.py` is open at `db_url_config` (goto 352).
- step 14: edit 425:426 (file size 794 -> 796, +2 lines), the fix.
- step 16: submit, **with no preceding test run**.

### 4. Uncertain decisions

None.

### 5. Evidence citations

| subtask id | category        | completed at step | evidence step(s) | one-line citation |
|------------|-----------------|-------------------|------------------|-------------------|
| `S1`       | `INVESTIGATION` | 13                | 11, 13           | search_file finds db_url_config; goto 352 confirms |
| `S2`       | `PRODUCT`       | 15                | 15               | tool ack of edit 425:426; +2 lines |
| `S3`       | `VALIDATION`    | (not reached)     | —                | left at `not_started` (no in-trace test run) |
| `S4`       | `ARTIFACT`      | 16                | 16               | submit issued |

### 6. Known missing evidence

- `S3` (validation) **left at `not_started`**. Same shape as
  `f_01`: the agent submitted without running tests in-trace.
  `test_output.txt` exists post-hoc and the upstream label is `True`,
  but per general § 4.4 a post-hoc artifact cannot complete a
  validation leaf the agent never started. Final progress < 1.0 by
  design — and notably independent of `final_success`.

### 7. Final scope closure

- total leaves: 4
- complete: 3 · in_progress: 0 · blocked: 0 · not_started: 1 · invalidated: 0
- progress (overall): {{PROGRESS_OVERALL}}
- progress (CODING_CATEGORIES = product+validation+investigation): {{PROGRESS_CODING}}

Was there ever a temptation to use `final_success` as evidence? **No.**
This is the canonical "lucky guess" case: upstream `True`, but the
agent never validated in-trace. Same ledger shape as `f_01` despite
opposite upstream labels — exactly what the framework should produce.

### 8. Schema gaps observed

None observed.
