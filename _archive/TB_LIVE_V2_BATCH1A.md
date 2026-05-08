# tb_live_v2 — Batch 1A results

_Run 2026-05-05. 5 newly-scaffolded internal tasks × 3 arms (A=opus, B=sonnet, C=haiku) = 15 runs._

## New tasks scaffolded for Batch 1A

| task_id | shape | difficulty |
|---|---|---|
| low_progress_success_03_typo_in_string | low_progress_success | easy |
| validation_new_work_05_quoted_field_in_tsv | validation_new_work | medium |
| stuck_blocked_02_ambiguous_path | stuck_blocked | medium |
| high_progress_failure_02_partial_solution_passes_smoke | high_progress_failure | hard |
| progress_drop_03_lint_clean_logic_wrong | progress_drop | medium |

Each was sanity-checked locally: solution.sh produces a workspace state
the verifier accepts; the unfixed seed state fails the verifier.

## Outcome table

| task | arm | model | final_success | termination_reason | num_events |
|---|---|---|---|---|---|
| hpf_02 | A | claude-opus-4-7   | True  | verifier_pass    |  6 |
| hpf_02 | B | claude-sonnet-4-6 | True  | verifier_pass    |  6 |
| hpf_02 | C | claude-haiku-4-5  | True  | verifier_pass    | 10 |
| lps_03 | A | claude-opus-4-7   | True  | verifier_pass    |  6 |
| lps_03 | B | claude-sonnet-4-6 | True  | verifier_pass    |  4 |
| lps_03 | C | claude-haiku-4-5  | True  | verifier_pass    |  6 |
| pd_03  | A | claude-opus-4-7   | True  | verifier_pass    |  6 |
| pd_03  | B | claude-sonnet-4-6 | False | verifier_fail    |  6 |
| pd_03  | C | claude-haiku-4-5  | False | verifier_fail    | 10 |
| sb_02  | A | claude-opus-4-7   | True  | verifier_pass    |  4 |
| sb_02  | B | claude-sonnet-4-6 | True  | verifier_pass    |  6 |
| sb_02  | C | claude-haiku-4-5  | True  | verifier_pass    |  6 |
| vnw_05 | A | claude-opus-4-7   | True  | verifier_pass    |  6 |
| vnw_05 | B | claude-sonnet-4-6 | False | verifier_fail    |  6 |
| vnw_05 | C | claude-haiku-4-5  | False | verifier_fail    |  6 |

**Tally:** 11 pass / 4 fail (n=15). 27% failure rate.

Per-arm:
- Arm A (opus):   5/5 pass
- Arm B (sonnet): 3/5 pass
- Arm C (haiku):  3/5 pass

Cumulative across Batch 0 + 1A: **21/30 pass, 9 fail (30% failure rate)**.

## Failure mode discovered: literal `/app/` path interpretation

Two of the four pd_03/vnw_05 failures have the same root cause: Sonnet
and Haiku interpreted the task description's `/app/<file>.py` as a
literal path and created a workspace subdirectory `app/`, while Opus
correctly interpreted it as the Docker convention and wrote to
`<workspace>/<file>.py`.

The verifier expects `WS / <file>.py` (workspace root). It fails
when the file is at `WS / app / <file>.py`.

This is **not a bug** — it's a measurement: smaller models follow
literal path instructions more rigidly. The Batch 0 hpf_01 task had
the same `/app/server.py` phrasing and all three models wrote to
workspace root, suggesting the failure mode is task-content-dependent
(the existence of seed files at workspace root may be the cue).

For future tasks, the task description convention should be revised:
either drop `/app/` entirely or make seed.sh produce a placeholder
file at the expected location to anchor the model.

## Combined corpus state

```
n = 30 runs
shapes covered: 5 (one + one each)
arms: 3 (opus, sonnet, haiku)
pass: 21
fail:  9
unresolved: 0  (1 no_done_record from Batch 0 hpf/C re-resolved
                as verifier_fail post-fix)

per-shape coverage (n=6 per shape):
  high_progress_failure: 4 pass / 2 fail
  low_progress_success:  6 pass / 0 fail
  progress_drop:         4 pass / 2 fail
  stuck_blocked:         6 pass / 0 fail
  validation_new_work:   3 pass / 3 fail
```

## Caveats

1. n=30 still well below sampling-policy minimum (60). Batches 1B+
   need to ship the remaining 15 spec tasks from MANIFEST.md.
2. The `/app/` path-interpretation failure mode introduces a confound
   with intended task shape. The vnw_05 / pd_03 failures might have
   been produced by the actual edge cases (quoted tab, edge-case
   logic) had the file been written at the right place — we cannot
   distinguish.
3. Stuck-blocked has 0% failure across both batches; the 5 designs
   appear too easy. Future stuck_blocked specs should add traps that
   genuinely block recovery within budget.

## Next

- Batch 1B: ship 5 more tasks (the second-priority spec rows) and
  run another 15 runs to bring the corpus to n=45.
- Revisit `/app/` path convention in task descriptions before next
  scaffold round.
- Consider adding a sanity step: after the agent declares done, the
  driver could `find <workspace> -name '<expected_file>' -type f`
  and warn if the file is at an unexpected location. This is a
  shape-classification aid, not a verifier change.
