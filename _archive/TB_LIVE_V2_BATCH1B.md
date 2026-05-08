# tb_live_v2 — Batch 1B results

_Run 2026-05-05. 5 newly-scaffolded internal tasks × 3 arms (A=opus, B=sonnet, C=haiku) = 15 runs._

## New tasks scaffolded for Batch 1B

| task_id | shape | difficulty |
|---|---|---|
| low_progress_success_02_config_flag_decisive | low_progress_success | easy |
| validation_new_work_02_silent_io_format_drift | validation_new_work | medium |
| stuck_blocked_03_perm_denied_chmod | stuck_blocked | medium |
| high_progress_failure_04_idempotent_required | high_progress_failure | hard |
| progress_drop_04_yaml_valid_schema_invalid | progress_drop | medium |

These descriptions deliberately avoid the `/app/` Docker convention
that confused Sonnet/Haiku in Batch 1A; they say "in the workspace"
or use unqualified relative paths.

## Outcome table

| task | arm | model | final_success | num_events |
|---|---|---|---|---|
| hpf_04  | A | claude-opus-4-7   | True |  4 |
| hpf_04  | B | claude-sonnet-4-6 | True |  6 |
| hpf_04  | C | claude-haiku-4-5  | True |  6 |
| lps_02  | A | claude-opus-4-7   | True |  6 |
| lps_02  | B | claude-sonnet-4-6 | True |  6 |
| lps_02  | C | claude-haiku-4-5  | True |  6 |
| pd_04   | A | claude-opus-4-7   | True |  6 |
| pd_04   | B | claude-sonnet-4-6 | True |  2 |
| pd_04   | C | claude-haiku-4-5  | True |  6 |
| sb_03   | A | claude-opus-4-7   | True |  4 |
| sb_03   | B | claude-sonnet-4-6 | True |  2 |
| sb_03   | C | claude-haiku-4-5  | True |  2 |
| vnw_02  | A | claude-opus-4-7   | True |  6 |
| vnw_02  | B | claude-sonnet-4-6 | True |  6 |
| vnw_02  | C | claude-haiku-4-5  | True |  6 |

**Tally:** 15/15 pass (n=15). 0% failure rate.

## Why all-pass?

Two factors stack to produce the 0% failure rate:

1. **Path convention fix.** The Batch 1A failures on `pd_03` and
   `vnw_05` were path-interpretation confounds, not the intended
   trap. With `/app/` removed from descriptions, Sonnet and Haiku
   now write to workspace root like Opus does.

2. **Tasks were less adversarial than expected.** Specifically:
   - `hpf_04` (idempotency) was a hard target but all three models
     wrote idempotent code on the first attempt — the "guard before
     mutate" pattern is well-rehearsed.
   - `pd_04` (yaml schema) was straightforward conformance once the
     schema was read carefully; agents read first, wrote complete
     config.
   - `vnw_02` (json-lines drift) requires explicit try/except, but
     the spec explicitly mentioned "silently skip", which removed
     the surprise edge.

Compare expected_pass_rate vs observed:
- hpf_04 expected 0.40, observed 1.00 (+0.60)
- pd_04  expected 0.55, observed 1.00 (+0.45)
- vnw_02 expected 0.55, observed 1.00 (+0.45)
- lps_02 expected 0.85, observed 1.00 (+0.15)
- sb_03  expected 0.55, observed 1.00 (+0.45)

The expected_pass_rate calibration was set against weaker agents.
For the current Claude lineup, these tasks are too easy.

## Cumulative corpus state (n=45)

```
Per-batch:
  Batch 0:  10/15 pass (33% fail)
  Batch 1A: 11/15 pass (27% fail)
  Batch 1B: 15/15 pass ( 0% fail)

Cumulative: 36/45 pass (20% fail).

Per-arm cumulative:
  A (opus):   15/15
  B (sonnet): 11/15
  C (haiku):  10/15  (+1 no_done_record from B0 hpf/C)

Per-shape cumulative (target_shape, n=9 each):
  high_progress_failure: 7/9
  low_progress_success:  9/9
  progress_drop:         6/9
  stuck_blocked:         9/9
  validation_new_work:   5/9
```

The corpus is now at 36/45 = 80% pass rate, which is **drifting out
of the sampling-policy target band** (0.40–0.60 outcome diversity).
The Batch-0/1A failures keep us above the hard minimum for failures
(15) only via the path-confound failures, which are arguably
mis-shaped for this corpus.

## Next steps

1. **Harden remaining task specs.** Future batches should target
   harder versions:
   - high_progress_failure: stricter strict-mode (e.g., binary
     compatibility, byte-for-byte stable output, security
     guarantees).
   - validation_new_work: spec withholds the trap until the
     verifier exposes it (e.g., specifically don't mention
     malformed-line handling).
   - progress_drop: require multi-iteration discovery (current
     designs are single-iteration).
2. **Down-weight the `_02`/`_03`/`_04` Batch 1B set** when computing
   shape balance: they pass uniformly, so they contribute little
   variance and could swamp the per-shape calibration.
3. **Address sb_03 anomaly:** num_events for B/C dropped to 2,
   meaning the agents ran one chmod + ran ./build.sh and stopped
   without a separate verification step. The transcripts logged but
   the leaf count is unusually low. Worth eyeballing whether the
   ledger reflects the chmod recovery.
4. **Continue corpus expansion.** 5 more spec tasks per shape remain
   (10 total tasks left in MANIFEST.md). At current pace, two more
   batches reach n=75.
