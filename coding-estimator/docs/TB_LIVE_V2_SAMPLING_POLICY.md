# tb_live_v2 sampling policy (U2)

_Generated 2026-05-05. Closes U2 in TASKS.md (Workstream U). Companion to
`reports/TB_TASK_SPACE_REVIEW.md` (U1)._

## Goal

Collect a live, **outcome-diverse** Terminal-Bench corpus that supports
real terminal-success measurement for the estimator. The corpus must
have enough failures to test completion-risk gates, and enough
trajectory-shape diversity to stress the observation channel — without
tuning the agent or task set to push pass rate toward a target.

## Hard requirements (lifted from TASKS.md U2 acceptance)

```text
total_runs                  target = 100   minimum = 60 (if budget limited)
failures                    target = 25    hard minimum = 15
real wall-clock timestamps  required
sidecar / ledger protocol   same as tb_live v0
checkpoint schema           same as tb_live v0
label schema                same as tb_live v0
no agent / task tuning to   force pass rate higher
```

## Pre-registered configuration

The configuration below is **pre-registered**: changing any of these
values mid-collection requires a written justification in
`reports/tb_live_v2_protocol_changes.md`. This is the U2 anti-tuning
rule.

### Two-arm collection

| field | Arm A (degraded top model) | Arm B (mid-tier model) |
|---|---|---|
| agent | `terminus` (TB harness) | `terminus` |
| model | `claude-opus-4-7` (or `glm-5.1`) | `step-3.5-flash` (or `qwen3.5-27b`) |
| `max_agent_timeout_sec` | 90 (TB2 default 180; deliberate cut) | 180 (TB2 default) |
| `max_iter` | 20 (~33% cut) | 30 (default) |
| network policy | per-task default | per-task default |
| reuse shell history | false | false |
| target task count | 50 TB2 tasks + 25 internal | 50 TB2 tasks + 25 internal |
| expected pass rate | 0.40–0.55 | 0.40–0.55 |

Total expected runs: ~150 (75 per arm). Failures expected: ~70 across
both arms — well above the 25-failure floor.

If runtime budget forces a cut, drop Arm B first (it is cleaner-shape
but topically redundant with Arm A). Minimum viable: Arm A only,
60 runs, ~25 failures.

### Source task pools

| pool | rule | expected count contributed |
|---|---|---|
| Terminal-Bench 2.0 | sample stratified by category + difficulty (§ "Category balance" below). All 89 are eligible. | up to 50 per arm |
| original Terminal-Bench | only used to fill the easy-difficulty slot (TB2 has only 3 easy tasks; we want ≥10). | ~7 per arm |
| internal U3 tasks (`tasks/tb_live_v2/`) | designed in U3 for trajectory-shape diversity (progress drop, validation-new-work, stuck/blocked, high-progress failure, low-progress success). | 25 per arm |
| Terminal-Bench Pro | **excluded from v2**. Re-evaluate after v2 lands. | 0 |

### Inclusion criteria (per task)

A task enters the `tb_live_v2` candidate set only if **all** hold:

1. Verifier is deterministic pytest assertions.
2. Verifier does not depend on the network unless the task is
   explicitly about a network service.
3. Task `descriptions[base].description` does not leak verifier
   internals.
4. Task is reproducible from a public (or internal-versioned) Docker
   image — no manual setup.
5. Task `max_agent_timeout_sec ≤ 600`. (Longer tasks are out of scope
   for v2.)

### Exclusion criteria (per task)

A task is excluded if **any** hold:

1. Requires GPU. (v2 is CPU-only for portability.)
2. Verifier accesses live external services (e.g., GitHub API rate
   limits, paid APIs).
3. Verifier is non-deterministic across runs (clock-dependent,
   randomness without seed).
4. License unclear or task author has not opted into derivative use.
5. Task statement is duplicated by another task in the candidate set.

### Category balance (across both arms combined)

```text
software engineering            : 25–35 %
data science / ML               : 20–30 %
security / systems              : 15–25 %
file/data manipulation          : 10–20 %
miscellaneous terminal tasks    : remainder
```

Implementation: TB2 categories are mapped to these buckets in
`docs/TB_CATEGORY_MAP.md` (to be produced in U3 alongside task
selection).

### Difficulty balance

```text
easy    : 20 %    # ~20 tasks; sourced primarily from original TB
medium  : 40 %    # ~40 tasks; sourced from TB2 medium + internal medium
hard    : 40 %    # ~40 tasks; sourced from TB2 hard + internal hard
```

### Trajectory-shape goals (across both arms combined)

These are *minimums*. Internal U3 tasks are designed to deliberately
produce these shapes; TB2 tasks contribute opportunistically.

```text
runs with progress drops      : ≥ 15
runs with validation-new-work : ≥ 10
runs with stuck/blocked       : ≥ 10
runs with high-progress fail  : ≥  5
runs with low-progress succ   : ≥  5
runs with late recovery       : ≥  5
```

The shape labels are computed post-hoc from the ledger via the
existing `coding_estimator/labels/dynamics.py` and `shapes.py`
modules — no manual tagging.

## Sampling procedure (deterministic, reproducible)

1. Build the candidate set: TB2 ∩ inclusion ∩ ¬exclusion → `C_tb2`.
2. Augment with original-TB easy tasks that pass the same filters →
   `C_easy_aug`.
3. Augment with internal U3 tasks → `C_internal`.
4. Stratified sample 50 tasks from `C_tb2 ∪ C_easy_aug` per the
   category and difficulty balances above. Use `numpy.random` with
   `seed=20260505` and record the seed in `run_manifest.json`.
5. Take all 25 internal U3 tasks (no sampling — internal tasks are
   designed in fixed quantities).
6. Per arm: run each selected task once. Record: model, agent, max
   iter, max timeout, network policy, real wall-clock start and end,
   pass/fail per verifier.

## What to do if the first batch is all successes

Per TASKS.md U2: "*if first 30 runs have < 5 failures, stop and resample
harder tasks*."

Concrete rule: after every 30 completed runs in Arm A, compute pass
rate. If pass rate ∉ [0.30, 0.70]:

- pass rate > 0.70 → next 30-run batch must replace its medium-difficulty
  tasks with hard-difficulty tasks (resample within `C_tb2`); do **not**
  swap models, lower budget, or rewrite tasks.
- pass rate < 0.30 → next 30-run batch must replace its hard-difficulty
  tasks with medium-difficulty tasks; same constraint.

Document each rebalance in `reports/tb_live_v2_protocol_changes.md`
with the timestamp, the pass rate that triggered it, and the resulting
candidate-set diff.

## Anti-tuning rules (active throughout collection)

1. **Do not** select only tasks the current agent can solve.
2. **Do not** tune prompts, model weights, decoding settings, or task
   set during a batch.
3. **Do not** swap models mid-batch.
4. **Do not** retry failed runs as successes.
5. **Do not** drop runs that are "interesting failures" but
   "uninteresting successes" or vice versa.

If a run produces a malformed ledger (broken JSON, missing required
fields), it may be re-run **once** with the same configuration. If
the second run also fails to produce a valid ledger, the task is
flagged as `infrastructure_failure` and is not counted toward the
100-run target nor toward the 25-failure floor.

## Per-run required artifacts (lifted from TASKS.md U4)

```text
runs/tb_live_v2/<run_id>/ledger.jsonl
runs/tb_live_v2/<run_id>/progress.csv
runs/tb_live_v2/<run_id>/progress_by_category.csv
runs/tb_live_v2/<run_id>/summary_by_category.json
runs/tb_live_v2/<run_id>/live_instrumentation.json
runs/tb_live_v2/<run_id>/run_manifest.json
runs/tb_live_v2/<run_id>/terminal_output.log
runs/tb_live_v2/<run_id>/verifier_output.txt
runs/tb_live_v2/<run_id>/run_notes.md
```

Required `run_manifest.json` fields:

```json
{
  "task_id": "...",
  "task_family": "tb2 | tb_original | internal",
  "difficulty": "easy | medium | hard",
  "category": "software_engineering | ...",
  "agent_scaffold": "terminus",
  "model_name": "claude-opus-4-7",
  "model_revision": "...",
  "arm": "A | B",
  "start_time": "ISO-8601 UTC",
  "end_time": "ISO-8601 UTC",
  "timeout_seconds": 90,
  "max_iter": 20,
  "final_success": true | false,
  "final_success_source": "tb_verifier",
  "termination_reason": "verifier_pass | verifier_fail | timeout | iter_limit | infrastructure_failure",
  "num_ledger_events": 0,
  "has_real_wallclock": true,
  "tb_dataset_version": "..."
}
```

`final_success_source = "tb_verifier"` is the value that distinguishes
this corpus from Hermes — the verifier is deterministic, so the label
is upstream-grounded, not annotated.

## Acceptance (U2 gate)

- This file states source pools, inclusion/exclusion criteria, target
  outcome mix, target category mix, target shape mix, and the
  fallback when the first batch is all successes. ✅
- Pre-registered configuration is fixed before any run is collected. ✅
- Anti-tuning rules are explicit. ✅

## Cross-references

- `reports/TB_TASK_SPACE_REVIEW.md` — task pool, leaderboard, two-arm rationale.
- `tasks/tb_live_v2/<task_id>/` — internal task design, U3.
- `coding_estimator/labels/dynamics.py`, `shapes.py` — post-hoc shape labels.
- `TASKS.md § Workstream U` — backlog and gate.
- `TASKS.md § Y` — anti-tuning, anti-controller guardrails.
