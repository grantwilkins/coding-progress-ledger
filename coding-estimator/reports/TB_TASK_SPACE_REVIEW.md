# Terminal-Bench task-space review (U1)

_Generated 2026-05-05. Closes U1 in TASKS.md (Workstream U). Scope-bounded
literature review: enough information to design `tb_live_v2`, no more._

> Warning per TASKS.md U1: do **not** turn this review into a survey of
> every benchmark. The goal is an outcome-diverse, verifier-backed live
> terminal-agent corpus — not coverage breadth.

## TL;DR

- Terminal-Bench 2.0 (TB2) is the right base pool: 89 verifier-backed
  tasks with public Docker images and pytest verifiers, manually
  reviewed by three humans.
- For ≥25 failures across ~100 runs, the agent must land in a
  **40–60% pass-rate regime**. Top models (GPT-5.5, Claude Mythos,
  Claude Opus 4.7, GLM-5.1) all sit at 0.65–0.83 and would not produce
  enough failures. Pick agents from the *middle of the leaderboard*
  (e.g. Qwen3.5-27B 0.416, Step-3.5-Flash 0.510), or run a top model
  with **deliberate degradations** (lower max-iter budget, smaller
  context window, no shell history).
- Tasks ship as `task.yaml` + `Dockerfile` + `tests/test_outputs.py` +
  `solution.sh`. Pass/fail is determined by pytest assertions on
  container state. This makes outcome ground truth deterministic,
  which is exactly what Hermes lacked.
- The internal task pool (U3) only needs to add diversity that TB2
  cannot supply by itself: deliberately-shaped trajectory dynamics
  (progress drop, validation-new-work, stuck/blocked, high-progress
  failure, low-progress success). 5–10 internal tasks per dynamic
  shape is sufficient — TB2 covers the topical breadth.

## 1. Available task pools

### 1.1 Terminal-Bench 2.0 (primary pool)

| field | value |
|---|---|
| task count | **89** |
| difficulty | 3 easy / 57 medium / 29 hard |
| categories (24 total) | software engineering (12), data science/processing (8), system administration (7), security (7), scientific computing (6), file operations (5), debugging (4), machine learning (4), mathematics (3), compilation/build tools (3), coding (3), + 16 other domains |
| verifier | pytest in `tests/test_outputs.py`; pass = all assertions pass within `max_test_timeout_sec` (default 30s) |
| environment | Ubuntu/Debian Docker; harness orchestrates multi-container; `tmux` + `asciinema` required |
| harness | `tb run --agent <name> --model <id> --dataset-name terminal-bench-core` |
| license / use | public; tasks human-authored and triple-reviewed |
| reference example tasks | `fix-git`, `prove-plus-comm`, `llm-inference-batching-scheduler`, `torch-pipeline-parallelism`, MuJoCo tuning, SPARQL knowledge-graph queries, protein assembly |

### 1.2 Original Terminal-Bench (~100 tasks, beta)

Older, broader pool. Smaller per-task triple-review guarantee than
TB2 but still verifier-backed. Useful for **easier** tasks that TB2
de-emphasizes (TB2 has only 3 easy tasks; original TB has more).

### 1.3 Terminal-Bench Pro (claimed 400 tasks)

Per upstream description, a systematic extension by Alibaba. Treat
as **secondary** — license, verifier coverage, and review depth not
yet audited. Use only if TB2 + original TB cannot hit category mix.

### 1.4 Out of scope for v2

- **SWE-bench / SWE-bench Verified.** Different shape: patches vs.
  terminal sessions. Useful for *future* cross-source comparison; not
  for `tb_live_v2`.
- **OSWorld / desktop benchmarks.** Different observation channel.
- **APEX / professional-workflow.** Different agent stack.

## 2. Outcome-diversity calculus

To hit `≥25 failures` across ~100 runs (TASKS.md U2 hard minimum), the
agent's expected pass rate must be in **[0.40, 0.60]** — outside that
band the binomial distribution either over-saturates with successes
(top models) or under-samples interesting near-success behavior
(weak models).

### 2.1 Public TB2 leaderboard snapshot (top 10, 2026-05-05)

| rank | model | TB2 pass rate |
|----:|---|---:|
| 1 | GPT-5.5 | 0.827 |
| 2 | Claude Mythos Preview | 0.820 |
| 3 | GPT-5.3 Codex | 0.773 |
| 4 | GPT-5.4 | 0.751 |
| 5 | Claude Opus 4.7 | 0.694 |
| 6 | GLM-5.1 | 0.690 |
| 7 | Gemini 3.1 Pro | 0.685 |
| 8 | DeepSeek-V4-Pro-Max | 0.679 |
| 9 | Kimi K2.6 | 0.667 |
| 10 | Claude Opus 4.6 | 0.654 |

### 2.2 In-band candidates

| model | TB2 pass | notes |
|---|---:|---|
| Qwen3.6-27B | 0.593 | borderline upper |
| Step-3.5-Flash | 0.510 | center of band |
| Qwen3.5-27B | 0.416 | lower band, ~58% failures |
| NVIDIA Nemotron 3 Super | 0.310 | below band; over-samples failures |

### 2.3 Top-model + deliberate-degradation track

Running a top model with degraded budget produces **interesting**
failures (high-progress failures, late-recovery, stuck-loop) that a
weak model would not produce, because the trajectory shapes differ.

Recommended degradations:
- `max_agent_timeout_sec` cut to 25–50% of TB2 default.
- `max_iter` (turn count) cut by ~33%.
- No tmux shell-history reuse.
- No tool-use auto-retries.

A top model under tight budget pushes pass rate from ~0.69 toward
~0.40–0.55 while preserving topical reach. This is preferable to
running a weak model on the same tasks because it stresses the
**observation channel**, not just the agent.

### 2.4 Recommendation

`tb_live_v2` should be a **two-arm collection**:

- **Arm A (primary, ~70 runs).** Top model (Claude Opus 4.7 or GLM-5.1)
  with deliberate budget tightening. Produces high-progress failures,
  late-recovery, and stuck/blocked patterns that the estimator most
  needs to learn.
- **Arm B (secondary, ~30 runs).** Mid-tier model (Step-3.5-Flash or
  Qwen3.5-27B) at default settings. Produces clean failure shapes
  and easier-to-grade success/failure signal.

Both arms run on the same TB2 task set. Each task is run once per arm.
Total runs: ~100 if both arms cover the same 50 tasks. Failures:
expected ~30 in Arm A (degraded top model) + ~25 in Arm B
(mid-tier model), well above the 25-failure floor.

## 3. Task-package format (canonical)

```text
tasks/<task_id>/
  Dockerfile                           # Ubuntu/Debian; install tmux, asciinema; WORKDIR /app
  docker-compose.yaml                  # use TB env vars; client image required
  task.yaml                            # see schema below
  solution.sh                          # human oracle (reference), one-shot
  tests/
    test_outputs.py                    # pytest; deterministic verifier
  run-tests.sh                         # optional; defaults provided by harness
```

Required `task.yaml` keys (per upstream docs):

```yaml
descriptions:
  - key: base
    description: |
      Plain natural-language statement of the task. No verifier hints.
author_email: contact@example.com
difficulty: easy | medium | hard
tags: [category, subcategory]                # optional but recommended
max_agent_timeout_sec: 180                   # default
max_test_timeout_sec: 30                     # default
test_scripts: [setup-uv-pytest.sh, run-uv-pytest.sh]
run_tests_in_same_shell: true
```

Hard constraints:

- **No test dependencies in the agent Docker image.** Test setup runs
  via `run-tests.sh` after the agent finishes; otherwise the agent
  could pip-introspect the verifier.
- **`task.yaml::descriptions[base].description` must not leak verifier
  internals** (e.g., specific test names, expected output values that
  are not part of the task statement).
- **Verifier exits deterministic.** `pytest --tb=no` with a fixed seed;
  no clock-dependent or network-dependent assertions unless the task
  is explicitly about a network service.
- **Solutions human-authored.** Per upstream review process, AI-drafted
  solutions are allowed only with human verification.

## 4. What the TB pool is missing for an estimator corpus

TB2 is built to grade *task completion*, not to expose *belief-relevant
trajectory dynamics*. Specifically, TB2 under-represents:

| dynamic | why under-represented | mitigation |
|---|---|---|
| progress drop | tasks are designed so the obvious approach works; no false-start traps | internal U3 tasks add false-start traps |
| validation-new-work | tests are usually one-shot; agent does not re-run mid-trajectory | internal tasks where test failure reveals a new requirement |
| stuck/blocked | TB2 timeouts trim long stuck loops | internal tasks with deliberate dependency confusion |
| high-progress failure | rare; most TB2 tasks are pass-or-fail-clean | internal tasks where 80% of subtasks are completable but the verifier requires the last 20% |
| low-progress success | rare; TB2 tasks reward thorough work | internal tasks solvable by a single decisive command |

**Implication.** U3 (internal tasks) should target the five dynamics
above, not topical breadth. ~5 tasks per dynamic = 25 internal tasks,
combined with ~50 TB2 tasks per arm, gives `tb_live_v2` ~125 task
slots.

## 5. Risks and constraints

- **License / fair-use.** TB2 task statements are public; running them
  produces logs that are derivative. The estimator does not redistribute
  task statements; it ships only ledger-shaped traces from runs we
  produced. Audit before public release of `datasets/checkpoints_tb_live_v2.parquet`.
- **Determinism.** TB2 verifiers use Docker isolation but agent network
  access is task-dependent. For reproducibility, pin the agent's
  network policy per task in our run manifest.
- **Cost.** ~100 runs × top-model budget ≈ a non-trivial bill. Estimate
  before U4.b and confirm with maintainer.
- **Outcome-tuning trap.** Per § Y in TASKS.md and the U2 sampling rule:
  do **not** swap models or tasks mid-collection to push pass rate
  toward a target. Pre-register the model + budget configuration; if
  the first 30 runs land outside [0.30, 0.65] pass rate, **resample
  harder/easier tasks**, not the agent.
- **Stratification.** Aim to hit category and difficulty mix from
  Section 1.1 with a budget tracker that rebalances each batch.

## 6. Action items into U2/U3

- U2 (sampling policy) inherits the two-arm design and the
  outcome-diversity calculus from § 2.4.
- U3 (internal task design) inherits the dynamics-targeted task list
  from § 4.
- U4 (batch run) inherits the resample-tasks-not-agent rule from § 5.
- U5 (artifact build) is gated on U4 completion.

## Sources

- [Terminal-Bench](https://www.tbench.ai/)
- [Terminal-Bench 2.0 task list](https://www.tbench.ai/benchmarks/terminal-bench-2)
- [Terminal-Bench task overview / spec](https://www.tbench.ai/docs/task-overview)
- [Terminal-Bench GitHub (laude-institute)](https://github.com/laude-institute/terminal-bench)
- [Terminal-Bench Hard leaderboard (Artificial Analysis)](https://artificialanalysis.ai/evaluations/terminalbench-hard)
- [Terminal-Bench 2.0 leaderboard (llm-stats.com)](https://llm-stats.com/benchmarks/terminal-bench-2)
- [Terminal-Bench paper (ICLR 2026)](https://openreview.net/pdf?id=a7Qa4CcHak)
