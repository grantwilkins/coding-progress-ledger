# tb_live_v2 — Batch 2 results

_Run 2026-05-05. 5 deliberately harder internal tasks × 3 arms (A=opus, B=sonnet, C=haiku) = 15 runs._

## Tasks scaffolded for Batch 2

The Batch 1B observation that telegraphed-trap descriptions caused
100% pass led to Batch 2's design rule: **withhold the trap from
the description**. Verifiers test edge cases the description does
not name.

| task_id | shape | difficulty | hidden trap |
|---|---|---|---|
| low_progress_success_04_missing_env_export | low_progress_success | easy | `.env` parser rejects `export`/quotes; verifier checks file content |
| progress_drop_05_two_function_integration | progress_drop | medium | csv.DictReader returns strings; type coercion needed for handoff |
| validation_new_work_04_tz_offset_in_log | validation_new_work | hard | `Z` suffix vs explicit `+HH:MM` offsets; spec doesn't mention timezones |
| stuck_blocked_04_module_not_found_pythonpath | stuck_blocked | medium | constraints forbid pip / file moves / `__init__.py`; correct fix is PYTHONPATH |
| high_progress_failure_05_caching_correctness | high_progress_failure | hard | spec mandates correctness-after-mutation; obvious id-based caches break |

Each was sanity-checked: solution passes verifier; seed without
solution fails it.

## Outcome table

| task | arm | model | final_success | num_events |
|---|---|---|---|---|
| hpf_05  | A | opus   | True |  6 |
| hpf_05  | B | sonnet | True |  6 |
| hpf_05  | C | haiku  | True |  6 |
| lps_04  | A | opus   | True | 10 |
| lps_04  | B | sonnet | True |  6 |
| lps_04  | C | haiku  | True | 10 |
| pd_05   | A | opus   | True |  4 |
| pd_05   | B | sonnet | True |  4 |
| pd_05   | C | haiku  | True |  6 |
| sb_04   | A | opus   | True |  4 |
| sb_04   | B | sonnet | True |  6 |
| sb_04   | C | haiku  | True | 10 |
| vnw_04  | A | opus   | True |  6 |
| vnw_04  | B | sonnet | True |  6 |
| vnw_04  | C | haiku  | True |  6 |

**Tally: 15/15 pass.** All three arms: 5/5.

## What this tells us

**The trap-withholding hypothesis was wrong.** Batch 2 was designed
to surface failures by hiding the verifier's strict checks from the
description. Result: still 100% pass.

What's actually happening:
1. Even when the spec doesn't mention timezones, all three models
   wrote tz-aware parsers because "ISO-8601 timestamp with optional
   offset" is a well-trained pattern — they handle Z and +HH:MM by
   default.
2. Even when the spec says "fast under repeats" (suggesting cache),
   all three picked the simplest correct design (`dict.get`) — the
   "id-based cache" trap requires actively designing a wrong
   solution, which they didn't.
3. The pd_05 type-coercion trap was preempted because all three
   models wrote `int(row["qty"]) * float(row["unit_price"])`
   reflexively when consuming CSV data.
4. The sb_04 PYTHONPATH constraint was followed exactly — opus
   used `PYTHONPATH=lib`, sonnet/haiku same.
5. lps_04 `.env` convention: opus and haiku both wrote
   `DATA_DIR=data` to `.env` AND modified `report.py` to load it
   (overkill but not violating the verifier's parser).

The corpus is hitting **the ceiling of "tasks Claude can do in 20-30
actions with a deterministic verifier"**. The tasks we'd need to
generate failures from current Claude tier-1/2/3 are tasks that
*aren't well-represented in training data* — e.g., domain-specific
business rules, novel APIs, multi-file refactors with cross-cutting
constraints, or genuine ambiguity that requires user clarification.

## Cumulative corpus state (n=60)

```
Per-batch:
  Batch 0:  10/15 pass (33% fail)
  Batch 1A: 11/15 pass (27% fail)
  Batch 1B: 15/15 pass ( 0% fail)
  Batch 2:  15/15 pass ( 0% fail)

Cumulative: 51/60 pass (15% fail).

Per-arm:
  A (opus):   20/20
  B (sonnet): 16/20
  C (haiku):  15/20  (+1 no_done_record from B0 hpf/C)

Per-shape (n=12 each):
  high_progress_failure: 10/12
  low_progress_success:  12/12
  progress_drop:           9/12
  stuck_blocked:         12/12
  validation_new_work:    8/12
```

The corpus is now at 60 runs but the failure rate is **15% — well
below the 0.25-0.60 sampling-policy band**. All 9 failures are from
Batches 0 and 1A, and 4 of those were the path-interpretation
confound (mis-shaped failure mode) rather than the intended trap.

## Next-step decision

The current task pool can't generate enough failures from current
Claude. Options to recover failure rate:

1. **Switch to harder substrate**: real Terminal-Bench 2 tasks
   (npm/cargo builds, multi-step CTF, cross-file SWE-style tasks)
   instead of single-file Python.
2. **Use weaker agents**: re-run with deliberately small budgets
   (e.g., 5 actions) to force premature termination, generating
   `no_done_record` shape rather than verifier_fail.
3. **Add adversarial constraints**: tasks where the verifier asks
   for unusual output formats, exotic libraries, or cross-process
   coordination.
4. **Stop expanding tasks**, accept the 60-run corpus, and use it
   as-is to test the estimator pipeline. The bottleneck shifts from
   data collection to feature/model work — but we already showed in
   v0 that prefix-only ledger features predict process dynamics, so
   this corpus is sufficient to test that line.

Recommend (4) as the next move: rebuild checkpoints/labels on the
60-run corpus, run G5 process-dynamics evaluation, and only return
to data collection if the estimator hits a genuine signal-to-noise
floor.
