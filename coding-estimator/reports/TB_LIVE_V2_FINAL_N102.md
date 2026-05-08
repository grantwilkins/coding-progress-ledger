# tb_live_v2 — final report for the n=102 live corpus

_Drafted 2026-05-05 from the existing batch reports and
`TB_LIVE_V2_N102_MILESTONE.md`. This document synthesizes the completed
internal `tb_live_v2` collection; it does not recompute metrics from raw
run artifacts._

## Executive summary

The final `tb_live_v2` corpus contains **102 completed runs** across
three arms and **25 unique internal task scaffolds**, with **9 selected
tasks replicated** to measure stability on both failure-heavy and
all-pass cases.

Headline outcome:

```text
n              = 102 runs
pass           = 81
fail           = 21
unresolved     = 0
failure rate   = 20.6%
```

This corpus **meets the total-run target** and **exceeds the hard
minimum failure count** from the collection plan. It does **not** meet
the intended outcome-diversity target: the final failure count is below
the 25-failure target, and the observed failure rate is below the
25-60% policy band cited in the milestone note.

Bottom line for downstream estimator work:

- `tb_live_v2` is now large enough to support **checkpoint rebuilds,
  label generation, and process-dynamics evaluation** on a live corpus.
- It should **not** be treated as an unbiased estimate of absolute
  terminal-task success or as a policy-grade completion-risk benchmark.
- Further attempts to raise failure count by adding more single-file
  internal Python scaffolds are unlikely to pay off; the batch record
  shows clear ceiling effects.

## Corpus composition

The final corpus is built from five 5-task scaffold batches plus one
replicate batch:

| component | design | runs |
|---|---|---:|
| Batch 0 pilot | 5 tasks x 3 arms | 15 |
| Batch 1A | 5 tasks x 3 arms | 15 |
| Batch 1B | 5 tasks x 3 arms | 15 |
| Batch 2 | 5 tasks x 3 arms | 15 |
| Batch 3 | 5 tasks x 3 arms | 15 |
| Replicate batch | 9 selected tasks x 3 arms | 27 |
| **Total** |  | **102** |

The 25 unique tasks cover the five intended trajectory-shape families:

- `low_progress_success`
- `stuck_blocked`
- `high_progress_failure`
- `progress_drop`
- `validation_new_work`

The replicate batch was intentionally selective rather than random. It
mixed:

- high-failure tasks to confirm reproducibility of known failure modes
- all-pass or near-all-pass tasks as controls

That makes the replicate batch useful for stability checks, but it means
the final `n=102` corpus is **not** an iid sample over task space.

## Final outcomes

### Per arm

| arm | model | pass | total | pass rate |
|---|---|---:|---:|---:|
| A | claude-opus-4-7 | 33 | 34 | 97.1% |
| B | claude-sonnet-4-6 | 24 | 34 | 70.6% |
| C | claude-haiku-4-5 | 24 | 34 | 70.6% |

The arm split is informative for estimator work: nearly all failures are
concentrated in the mid-tier and small-model arms, while Opus is close
to ceiling on this task substrate.

### Per shape

| shape | pass | total | pass rate |
|---|---:|---:|---:|
| low_progress_success | 21 | 21 | 100% |
| stuck_blocked | 18 | 18 | 100% |
| high_progress_failure | 16 | 21 | 76.2% |
| progress_drop | 14 | 21 | 66.7% |
| validation_new_work | 12 | 21 | 57.1% |

Interpretation:

- `validation_new_work` remained the highest-yield failure family.
- `progress_drop` produced useful but still limited failure mass.
- `low_progress_success` and `stuck_blocked` were effectively solved out.

### Task-level concentration of failures

Most tasks ended at ceiling. Failures concentrated in a small subset:

| task | final pass/total |
|---|---:|
| `validation_new_work_03_unicode_normalization` | 1/3 |
| `high_progress_failure_01_subtasks_done_verifier_strict` | 2/6 |
| `progress_drop_03_lint_clean_logic_wrong` | 2/6 |
| `validation_new_work_05_quoted_field_in_tsv` | 2/6 |
| `progress_drop_01_lint_then_runtime_failure` | 3/6 |
| `validation_new_work_01_test_reveals_edge_case` | 3/6 |
| all remaining tasks | 3/3, 5/6, or 6/6 |

The one partial-exception above ceiling was
`high_progress_failure_02_partial_solution_passes_smoke` at **5/6**.

## What the batch sequence established

The batch reports show a clear collection story rather than a flat
steady-state sample:

1. **Batch 0 and Batch 1A generated most of the early failures.**
   These runs established that the live runner, ledger, and verifier
   stack worked, and they produced real disagreement between model
   self-claims and verifier truth.
2. **Batch 1B and Batch 2 both went all-pass.**
   The intended traps were largely preempted by current Claude default
   behavior on familiar, single-file Python tasks.
3. **Batch 3 produced only one clean new trap.**
   The Unicode-normalization task fired exactly as intended; the other
   four new tasks were solved by all three arms.
4. **Replicates added failure evidence, but only selectively.**
   The replicate batch confirmed that several failure modes were
   reproducible rather than one-off transcript noise, but it did not
   change the larger ceiling-effect conclusion.

This is the most important scientific result of the collection:
**additional task count on the same substrate does not buy proportionate
failure diversity**.

## Replication findings

The replicate batch was useful. It established that several observed
failures were stable across repeat runs:

- `high_progress_failure_01` moved from `1/3` to `2/6`, with repeated
  failures on the smaller arms.
- `progress_drop_03` moved from `1/3` to `2/6`, again reproducing the
  B/C-arm failures.
- `validation_new_work_05` moved from `1/3` to `2/6`, showing the edge
  case was not a one-run accident.
- `validation_new_work_01` moved from `1/3` to `3/6`, showing higher
  variance than the other replicated failure tasks.
- Control tasks such as `low_progress_success_01`, `low_progress_success_03`,
  and `stuck_blocked_01` stayed at ceiling.

The replicate batch therefore improved confidence in the **reality** of
the failure modes, but not in the corpus as a representative sample of
open-ended terminal work.

## Batch 3 trap-design result

Batch 3 was the best direct test of whether one more round of hidden
verifier traps would recover the desired failure rate. It mostly did
not:

| task | observed outcome | trap fired? |
|---|---:|---|
| `low_progress_success_05_quote_glob` | 3/3 | no |
| `validation_new_work_03_unicode_normalization` | 1/3 | **yes** |
| `stuck_blocked_05_make_target_typo` | 3/3 | no |
| `high_progress_failure_03_url_encode_strict` | 3/3 | no |
| `progress_drop_02_currency_format_thousands` | 3/3 | no |

The report record supports a strong conclusion: on this scaffold style,
current Claude models often write the strict-correct implementation
without needing the verifier to teach the hidden edge case.

## Threats to validity

Several caveats materially affect how this corpus should be used.

### 1. `solution.sh` leak inflates success

The milestone note records that multiple agent runs read
`solution.sh` from the seeded workspace. Because the runner excluded
`tests/` but not `solution.sh`, agents could receive a strong hint about
the intended fix. This affects all batches, including replicates.

Implication: the observed pass rate is **optimistic**.

### 2. Early `/app/` path confounds are real verifier fails but weak task-shape evidence

Batch 1A surfaced a path-interpretation failure mode in which smaller
models treated `/app/<file>.py` literally and wrote into a workspace
subdirectory rather than the intended root. Later batches removed that
description pattern and the issue disappeared.

Implication: some early failures are genuine end-to-end misses, but they
are less informative about the intended semantic trap than later failures.

### 3. The final corpus is shaped, not representative

The final dataset is dominated by:

- internal, deliberately scaffolded tasks
- equalized multi-arm repeats
- targeted replication of chosen tasks

Implication: the corpus is appropriate for **estimator stress-testing**
and live-ledger analysis, not for broad claims about real-world
Terminal-Bench difficulty.

### 4. The collection drifted from the original two-arm expectation

The batch reports note that Arm C was added after the original two-arm
design. The final corpus is internally consistent, but it should be
described as a **three-arm collected corpus**, not as a strict execution
of the earlier two-arm expectation.

## Recommendation for downstream estimator use

Recommended:

- Freeze `tb_live_v2` at `n=102` as the current live corpus.
- Rebuild checkpoints and combined labels on the full set.
- Use this corpus for process-dynamics tasks:
  `future_progress_drop`, `validation_new_work`, stuck-loop analysis,
  and within-corpus calibration checks.
- Keep explicit caveats in any writeup that uses terminal success as a
  label.

Not recommended:

- Treating the 81/102 pass rate as a deployment-facing success prior.
- Using this corpus alone to set hard completion-risk thresholds.
- Spending more collection budget on the same family of single-file
  internal scaffolds in the hope of reaching the original 25-failure
  target.

## Immediate next actions

1. Rebuild checkpoints and labels on the completed corpus.
2. Run the next estimator evaluation pass on `tb_live_v2`, focusing on
   process-dynamics prediction rather than absolute success calibration.
3. Fix the runner to exclude `solution.sh` from seeded workspaces before
   any future live collection.
4. If more failures are required later, switch substrate rather than
   iterating on the current one:
   multi-file tasks, translated TB2 tasks, unfamiliar libraries, or
   genuinely cross-cutting constraints.

## Final judgment

`tb_live_v2` is complete enough to unblock the next estimator stage.
It is a **useful live-process corpus** with real failures, meaningful
arm separation, and enough scale for dynamics work. It is **not** yet a
clean final benchmark for absolute terminal-success estimation, because
the failure spectrum is too narrow and the `solution.sh` leak makes the
observed success rate optimistic.
