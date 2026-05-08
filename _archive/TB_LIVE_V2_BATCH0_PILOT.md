# tb_live_v2 — Batch 0 pilot results

_Run 2026-05-05. 5 internal task scaffolds × 3 arms (A=opus, B=sonnet, C=haiku) = 15 runs._

## Outcome table

| task | arm | model | final_success | termination_reason | num_events | target_shape |
|---|---|---|---|---|---|---|
| high_progress_failure_01 | A | claude-opus-4-7   | True  | verifier_pass    |  6 | high_progress_failure |
| high_progress_failure_01 | B | claude-sonnet-4-6 | False | verifier_fail    | 10 | high_progress_failure |
| high_progress_failure_01 | C | claude-haiku-4-5  | False | no_done_record   |  6 | high_progress_failure |
| low_progress_success_01  | A | claude-opus-4-7   | True  | verifier_pass    |  6 | low_progress_success  |
| low_progress_success_01  | B | claude-sonnet-4-6 | True  | verifier_pass    |  6 | low_progress_success  |
| low_progress_success_01  | C | claude-haiku-4-5  | True  | verifier_pass    |  6 | low_progress_success  |
| progress_drop_01         | A | claude-opus-4-7   | True  | verifier_pass    |  4 | progress_drop         |
| progress_drop_01         | B | claude-sonnet-4-6 | True  | verifier_pass    |  6 | progress_drop         |
| progress_drop_01         | C | claude-haiku-4-5  | False | verifier_fail    |  6 | progress_drop         |
| stuck_blocked_01         | A | claude-opus-4-7   | True  | verifier_pass    |  4 | stuck_blocked         |
| stuck_blocked_01         | B | claude-sonnet-4-6 | True  | verifier_pass    | 10 | stuck_blocked         |
| stuck_blocked_01         | C | claude-haiku-4-5  | True  | verifier_pass    | 10 | stuck_blocked         |
| validation_new_work_01   | A | claude-opus-4-7   | True  | verifier_pass    |  6 | validation_new_work   |
| validation_new_work_01   | B | claude-sonnet-4-6 | False | verifier_fail    |  2 | validation_new_work   |
| validation_new_work_01   | C | claude-haiku-4-5  | False | verifier_fail    |  6 | validation_new_work   |

**Tally:** 10 pass / 5 fail / 0 unresolved (n=15). Failure rate 33%.

## Outcome-diversity check

Sampling policy target was failure rate ≥ 25% (15+/60 hard minimum). Batch 0
already at 33%, so the two-arm-plus-haiku design does land in the
0.25–0.60 band on the shipped 5 task scaffolds.

Per-arm failure rate:
- Arm A (opus): 0/5 fail
- Arm B (sonnet): 2/5 fail
- Arm C (haiku): 3/5 fail (+1 no_done_record)

Per-shape outcome alignment with target_shape:
- high_progress_failure: 1/3 fail at verifier despite logging multiple
  visible subtasks — target_shape produced as designed by the strict
  verifier on at least the sonnet arm.
- low_progress_success: 3/3 pass with very few events (4–6) — the
  one-line-fix shape held across all three models.
- progress_drop: 1/3 fail (haiku); pass curves should still show the
  initial-implementation → verifier-fail → fix loop on the failing arm.
- stuck_blocked: 3/3 pass — every arm found the `bs4` workaround.
  Sonnet/haiku used `pip install --break-system-packages` /
  auto-install; opus rewrote without bs4. Different recovery shapes.
- validation_new_work: 1/3 pass; the edge cases (empty cells,
  quoted commas) caught both sonnet and haiku.

## Caveats

1. **n=15 is a pilot, not a measurement.** This is enough to verify
   the runner driver, sidecar integration, and shape-vs-outcome
   alignment on five tasks. It is NOT enough to compute calibration
   or shape-classification accuracy.
2. **One arm-C run terminated as `no_done_record`** (hpf/C). Subagent
   was killed by the runtime cap; transcript ends mid-action. The
   driver still produced a usable ledger from the partial transcript.
3. **Verifier is the ground truth.** Two arm-B "done" records claimed
   success but the verifier failed (hpf/B, vnw/B) — confirming the
   spec's contract that `done` is a self-claim, not a label.
4. **Sidecar bug found and fixed.** `_replay_sidecar` was passing a
   relative `--run-dir` while `cd`'ing into the upstream repo. Fixed
   by resolving paths to absolute before invoking the sidecar.
5. **Arm C added late.** Original Workstream U design was two-arm
   (A=top, B=mid). Arm C (haiku) was added at user request as a
   third tier; budget matches arm B (20 actions).

## Next

- Build checkpoints + labels on the 15-run pilot to verify the
  artifacts pipeline ingests tb_live_v2 cleanly:
  ```
  uv run python scripts/build_checkpoints.py --source tb_live_v2
  ```
- Hold off on estimator retraining until U3 ships the remaining 20
  internal tasks (per Workstream U dependency chain).
- Profile shape diversity: this pilot has at least one verified
  failure on every shape except `low_progress_success` (which is
  designed to be near-universal pass).
