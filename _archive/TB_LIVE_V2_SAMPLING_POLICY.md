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

Subagents are spawned via the Claude Code Agent tool. No upstream
harness, no Docker, no direct API calls. See
`docs/TB_LIVE_V2_RUNNER_SPEC.md` for the runner contract.

| field | Arm A (top model) | Arm B (mid-tier model) |
|---|---|---|
| subagent | Agent tool, `subagent_type: general-purpose` | same |
| model | `claude-opus-4-7` | `claude-sonnet-4-6` |
| `budget_lines` | 30 (soft cap, enforced by prompt + transcript count) | 20 |
| isolation | fresh tempdir + fresh `python -m venv` | same |
| target task count | 25 internal + a translated subset of TB2 (verifier-pytest only) | same |
| expected pass rate | 0.45–0.60 | 0.30–0.50 |

Total expected runs: ~100 across both arms (≈50 tasks × 2 arms; some
tasks run only in Arm A if Arm B's smaller model fails to even attempt
them). Failures expected: ~40–50 — well above the 25-failure floor.

The Arm B model is **smaller**, not "degraded by budget cut", because
the runner does not control wall-clock — it controls the action-budget
through the prompt. Smaller-model is the cleaner outcome-diversity
lever in this setup.

If runtime budget forces a cut, drop Arm B first (Arm A alone delivers
the failure floor with better trajectory variety).

### Source task pools

| pool | rule | expected count contributed |
|---|---|---|
| internal U3 tasks (`tasks/tb_live_v2/`) | designed for trajectory-shape diversity. Primary pool — these are guaranteed verifier-pytest, no Docker, no GPU. | 25 per arm |
| Terminal-Bench 2.0 (translated subset) | Only TB2 tasks with: pytest verifier in `tests/test_outputs.py`, no Docker-only setup, no GPU. Translate verbatim into `tasks/tb_live_v2/<id>/` (the directory shape is identical). | up to 25 per arm, after audit |
| original Terminal-Bench | use to fill the easy-difficulty slot if the internal pool under-represents easy. | ~5 per arm |
| Terminal-Bench Pro | **excluded from v2**. Re-evaluate after v2 lands. | 0 |
| TB tasks that need Docker (apt-get, system services, multi-container) | flagged `requires_docker: true` in their `shape.yaml` and skipped. | 0 |

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
  "subagent_type": "general-purpose",
  "model_name": "claude-opus-4-7",
  "arm": "A | B",
  "budget_lines": 30,
  "start_time": "ISO-8601 UTC",
  "end_time": "ISO-8601 UTC",
  "final_success": true | false,
  "final_success_source": "internal_verifier",
  "termination_reason": "verifier_pass | verifier_fail | no_done_record | subagent_limit | infrastructure_failure",
  "num_ledger_events": 0,
  "has_real_wallclock": true
}
```

`final_success_source = "internal_verifier"` is the value that
distinguishes this corpus from Hermes — the pytest verifier exit is
deterministic, so the label is upstream-grounded, not annotated.

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
