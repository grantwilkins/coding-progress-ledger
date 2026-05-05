# TASKS — coding-estimator

This file is the working backlog for `coding-estimator`, the **belief-state layer** that consumes append-only ledger histories produced by `../coding-progress-ledger` and outputs calibrated, checkpoint-level beliefs over latent remaining work, completion-by-horizon, and process-dynamics events.

The goal is **not** to redefine progress, replace the ledger, or build a controller. It is to answer one question, on live ledger histories, with calibration:

> Given the prefix-only ledger history `H_t` of a long-horizon coding task at checkpoint `t`, output calibrated probabilities over (a) eventual success, (b) success-by-horizon, (c) remaining time conditional on success, and (d) near-future progress-dynamics events (drops, reopens, validation surprises, stuck loops, scope discovery).

---

## Current scientific state: v0 boundary

v0 has established a measurement boundary.

```text
Positive result:
  Prefix-only ledger features predict near-future process dynamics.
  In particular, they predict progress drops within a short horizon
  substantially better than elapsed time.

Negative result:
  Prefix-only ledger features do not yet improve terminal success
  prediction over elapsed time at the current sample size.

Interpretation:
  The observation channel measures work-frontier dynamics before it
  becomes a reliable completion-risk estimator.

Consequence:
  The next phase is not model polish, online inference, semantic features,
  or scheduling. The next phase is targeted data work:
    1. collect a much larger, outcome-diverse live Terminal-Bench corpus
       (tb_live_v2: 100+ verified runs, deliberately mixed outcomes);
    2. use Hermes HF retrospectively for *process-dynamics* scaling only
       (not for terminal-success gates — Hermes ships no verifier);
    3. rerun the estimator only after those data foundations exist.
```

`not_safe_for_control = true` remains stamped on the v0 estimator. The v0 verdict is `indeterminate` because of data gaps, not leakage or code defects. Full evidence: `reports/V0_FINDINGS.md`, `reports/REVIEWER_BRIEFING.md`, `reports/ESTIMATOR_GO_NO_GO.md`, `reports/NOT_READY_FOR_SCHEDULING.md`.

### Primary v0 target family

```text
Primary v0 target family (process dynamics):
  near-future process dynamics:
    - y_future_progress_drop_h5
    - y_validation_new_work_h5
    - later, if base rates permit:
      y_stuck_loop_h5, y_product_reopen_h5, y_blocked_within_h5

Secondary / negative-control target:
  y_success_eventual

Audit / negative-control target:
  y_submit_without_validation
```

The v0 headline is process-dynamics prediction. Terminal success is a secondary, currently negative target — re-tested only when the data can answer the question (Workstream X).

### What the next phase is NOT

```text
- Building a scheduler, controller, monitor, or online inference surface.
- Adding semantic / text features (Workstream Q remains deferred).
- Loosening the +0.02 O7 threshold to get a pass.
- Promoting a tb_live result obtained on the all-success cohort.
- More modeling complexity before the data foundation lands.
```

---

## v0 shipped surface (Workstreams A–P)

Every v0 workstream below is `done`, `shipped`, or `deferred` per § 0.0β. The detailed plan that produced these artifacts is preserved in git history (commits up to `59b44af`). Re-open a section here only if a v0 artifact is being modified.

```text
A. Repository scaffold and conventions ................ done
B. Estimator interface freeze (schemas) ............... done
C. Source ingestion and unification ................... done
D. Checkpoint dataset construction (D0–D5.5) .......... done
E. Label construction (E1, E4, E6, E7, E8) ............ done
   E2/E3/E5 horizon, wall-clock, survival labels ...... deferred
F. Profiling and distribution analysis (F1–F4, F11) ... done
   F5 descriptive rollups ............................. not started (low priority)
G. Baseline ladder (G1, G2, G4, G7) ................... done
   G3, G6 ............................................. deferred
   G5 ledger-dynamics ................................. shipped (post-feedback round)
   G8 human baseline scaffold ......................... shipped (Workstream V runs it)
H. Splits and evaluation protocol (H1–H4, H6) ......... done
I. Model ladder (I0, I1) .............................. done
   I2 GBM, I3 hazard, I4 sequence, I5 semantic ........ deferred
J. Calibration (J1–J5) ................................ shipped
K. Live Terminal-Bench-specific evaluation (K1, K3) ... shipped
   K2 online feasibility .............................. deferred
L. Live vs retrospective parity (L3) .................. shipped
   L1 blocked on upstream NTB6; L2/L4 deferred
M. Online inference ................................... deferred (gated on P)
N. Estimator artifacts and model cards (N1–N3) ........ shipped
O. Failure-mode tests:
   O1 progress overconfidence ......................... PASS (median 0.578)
   O5 source leakage .................................. shipped (vacuous)
   O7 timeout bias .................................... FAIL on swe_agent_pilot (Δ Brier −0.009)
   O2/O3/O4/O6 ........................................ deferred (small-N slices)
P. Go/no-go gate:
   P1 v0 no-regression gate ........................... INDETERMINATE
        P1.a PASS (G4 wins or ties G2 on 6/8 cells)
        P1.b INDETERMINATE (single-class y on tb_live)
        P1.c INDETERMINATE (hermes labels missing)
        P1.d INDETERMINATE (single-class y on tb_live)
        P1.e PASS (no forbidden columns)
        P1.f PASS (zero run-constant feature/target pairs)
        P1.g INDETERMINATE (D5 audit shipped post-verdict)
        P1.h PASS (winners span multiple targets)
   P2 sign-off package (ledger_basic_v0.1) ............ shipped
   P3 downstream readiness ............................ shipped (NOT_READY)
   P-future aspirational gate ......................... deferred (N > 200 / live > 50)
Q. Semantic features and sequence models .............. deferred (gated on Y2)
R. Scheduling consumer ................................ deferred (out of scope)
```

The corresponding artifacts on disk are stable and reproducible from `scripts/run_*.py` against the pinned upstream SHA in `datasets/manifests/upstream_commit.json`.

---

## § 0. Project rules (apply to all workstreams)

```text
0.1  Do not mutate any code under ../coding-progress-ledger/.
0.2  Do not redefine ledger semantics. Progress is what coding-progress-ledger
     says it is. We add features and labels around it, never inside it.
0.3  Do not use any ledger event with step > S when constructing features at
     checkpoint S. The replayer must be prefix-only.
0.4  Do not use any post-run artifact when constructing features at S unless
     that artifact was visible at step S in the live trace.
0.5  Do not split train/test by checkpoint; always split by run_id.
0.6  Do not train large neural models in the first ladder.
0.7  Do not output policy actions. The estimator outputs belief state only.
0.8  Do not equate progress with success. Progress measures visible discovered
     work, not correctness.
0.9  Do not fabricate data when an upstream artifact is missing.
0.10 Hard fail over silent fallback.
0.11 Calibration is a first-class output.
0.12 Every model artifact ships with a model card. No exceptions.
```

**Status markers:** `not started` · `in progress` · `blocked` · `done` · `deferred`.

---

## § Workstream S — Lock the v0 boundary (v1 framing)

Goal: Make the current state impossible to misunderstand and align the project framing with what the data actually supports.

### S1. Update project framing
Status: done (2026-05-04)

Outputs (shipped):
```text
TASKS.md top banner — already states the four points (no edit needed).
README.md — rewritten with the v0-boundary paragraph and cross-links.
reports/V0_FINDINGS.md — cross-linked from README.
reports/NOT_READY_FOR_SCHEDULING.md — cross-linked from README.
```

Acceptance: met. README now states the four points in one paragraph
(positive, negative, not_safe_for_control, next phase = Hermes + tb_live_v2).

### S2. Make process dynamics the primary v0 target
Status: done (2026-05-04)

Outputs (shipped):
```text
reports/g5/g5_eval.md — added headline-framing paragraph; primary table
  is y_future_progress_drop_h5 / y_validation_new_work_h5; the success
  table is preserved as a "secondary / negative" result.
reports/sign_off_ledger_basic_v0.1.md — already states process dynamics
  primary, terminal success secondary/negative (no edit needed).
TASKS.md current scientific state section — primary v0 target family
  is process dynamics; y_success_eventual is secondary/negative.
```

Acceptance: met. No model changes. Headline framing aligned with what
the data supports.

---

## § Workstream T — Hermes annotation unblock — DEFERRED (2026-05-05)

> **Reframed by user direction (2026-05-05).** Hermes HF is not a labeled
> outcome source. The published HF schema (14.7k tool-calling trajectories
> across Kimi and GLM configs, with conversations, tools, categories,
> subcategories, task descriptions) **does not** expose `final_success`,
> `verifier_pass`, `eval_log`, or any benchmark outcome field. Annotating
> terminal success on Hermes traces would either (a) derive `final_success`
> from agent self-claims (fabrication, § 0.9), or (b) require an external
> verifier that does not exist in the dataset.
>
> Therefore:
>
> - **Do not** derive `final_success` from self-claims.
> - **Do not** treat final assistant completion as success.
> - **Do not** use Hermes HF for terminal-success gates unless a
>   human-reviewed or external verifier label is added.
> - **Do** use Hermes HF to scale observation-channel ingestion and
>   process-dynamics analysis (visible discovered work, progress drops,
>   validation events, stuck/blocked patterns). That use is consistent
>   with the dataset's schema.
>
> P1.c "combined retrospective" gate is **not** the right unblock target
> for Hermes. The right unblock target is `tb_live_v2` (Workstream U) —
> outcome-diverse Terminal-Bench tasks with deterministic verifiers.
>
> **What stays.** The conservative grader and proposal harness ship
> as-is and remain useful as a *process-dynamics* annotation aid (it
> never asserts `final_success=true`, only flags BLOCK / budget-exhaust
> as failure-shaped trajectories, which is a *trajectory-shape* signal,
> not a terminal-outcome signal). Reframed handoff is in
> `reports/HERMES_ANNOTATION_PROPOSAL.md`.
>
> **What is deferred.** Upstream commit of any `final_success` label on
> `hermes_pilot_h5_v2`. Workstream T sub-tasks T2 and T3 are deferred
> until either an external verifier exists for Hermes traces, or a
> human-reviewed annotation pass is explicitly funded.

The original Workstream T plan is preserved below as historical record.

---

### T1. Confirm upstream Hermes annotation state
Status: done (2026-05-05)

Outputs (shipped):
```text
reports/HERMES_ANNOTATION_PRECHECK.md
```

Verdict: **Path A — upstream annotation missing.** All 30/30 runs
have ledger.jsonl present, all 30/30 have `source_metadata.final_success
= null`, `final_success_source = "missing"`, `annotation_mode =
"not_annotated"`. Local loader returns 0 labeled rows, classifying all
30 as `unresolvable` (correct behavior; not malformed). No local wiring
bug. T2 must materialize labels upstream.

### T2. Annotate 30 Hermes runs upstream
Status: **DEFERRED** (2026-05-05) — Hermes HF is not a labeled outcome source; see Workstream T banner above. The harness below is preserved as a process-dynamics aid but does not run as a terminal-success annotator.

What landed in this repo:
```text
coding_estimator/labels/hermes_terminal_grader.py  # conservative deterministic grader
scripts/build_hermes_annotation_proposal.py        # proposal generator
reports/HERMES_ANNOTATION_PROPOSAL.md              # 30-run summary
reports/hermes_annotation_proposals/<run_id>.md    # 30 per-run proposals
```

Heuristic verdict distribution (30 runs):
```text
failure              (low/medium confidence): 11   # budget-exhausted or BLOCK pattern
success_self_claim   (low confidence):         6   # terminal tool call, no verifier
ambiguous            (no terminating signal): 13
```

Why end-to-end T2 was not finished autonomously:
```text
Hermes traces ship NO upstream eval log. final_success cannot be
machine-derived from the trace alone, only inferred from self-claim or
budget-exhaust signals. Setting final_success=true from a self-claim
fabricates data (rule § 0.9). The 25/30 usable-labels acceptance threshold
is therefore unreachable without a human read of trajectory_summary.md
and final_diff.patch per run. The grader exists to make that human pass
fast; it is not a substitute for it.
```

Threshold arithmetic (flagged by critic 2026-05-05):
```text
The harness yields at most 11 confident-failure + 6 self-claim = 17
runs where the trace alone provides any evidence at all. The remaining
13 are ambiguous (no terminating signal). If the upstream annotator
cannot resolve some of those 13 by reading final_diff.patch, T2's 25/30
acceptance is unreachable even with human review.

Mitigation options the upstream annotator may choose:
  (a) drop the threshold to >= 17/30 with explicit per-run notes;
  (b) introduce annotation_mode = "inconclusive_no_verifier" for runs
      that cannot be resolved even after human review, and accept those
      as a documented null label rather than a missing label;
  (c) collect a re-run with a deterministic verifier (Hermes v3) so
      final_success is machine-derivable.
```

Goal: Materialize valid retrospective ledgers and final labels for all `hermes_pilot_h5_v2` runs. **Work happens in `../coding-progress-ledger`, not here.**

Required upstream outputs per run:
```text
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/ledger.jsonl
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/progress.csv
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/progress_by_category.csv
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/summary_by_category.json
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/source_metadata.json
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/run_notes.md
../coding-progress-ledger/runs/hermes_pilot_h5_v2/<run_id>/annotation_quality.json
```

Required metadata:
```text
source_metadata.final_success != null
source_metadata.final_success_source != missing
annotation_mode != not_annotated
```

Annotation constraints:
```text
- Do not use terminal outcome to decide intermediate progress.
- Preserve visible discovered work vs hidden work.
- Completion requires evidence.
- Do not force failed runs to low progress or successful runs to high progress.
- Record uncertainty in run_notes.md.
```

Acceptance:
```text
At least 25/30 Hermes runs have usable final labels.
At least 25/30 replay without error.
All missing labels are explained per-run.
```

### T3. Rebuild estimator artifacts with Hermes
Status: **DEFERRED** (2026-05-05) — gated on T2 which is deferred per the Workstream T banner. Hermes does not contribute terminal-success labels under the current dataset definition.

Commands:
```bash
uv run python scripts/build_checkpoints.py --source hermes_pilot_h5_v2
uv run python -c "
from pathlib import Path
from coding_estimator.labels.build import write_combined_labels
write_combined_labels(Path('datasets'))
"
uv run python scripts/run_baselines.py
uv run python scripts/run_g5_eval.py
uv run python scripts/run_failure_modes.py
uv run python scripts/run_go_no_go.py
```

Outputs:
```text
datasets/checkpoints_hermes_pilot_h5_v2.parquet
datasets/labels_hermes_pilot_h5_v2.parquet
datasets/labels_all.parquet
reports/g5/g5_eval.md
reports/ESTIMATOR_GO_NO_GO.md
reports/V0_FINDINGS.md
```

Acceptance:
```text
P1.c is no longer indeterminate due to missing Hermes labels.
The combined retrospective pool contains SWE-agent + Hermes labels.
```

---

## § Workstream U — Terminal-Bench live v2: 100+ outcome-diverse tasks

The most important new data collection task. Current `tb_live` (12 successes, 0 failures) cannot test completion risk. **The next live dataset must be deliberately designed for outcome diversity.**

```text
tb_live_v2:
  >= 100 live runs
  >= 25 failures minimum
  real wall-clock timestamps
  same sidecar / ledger protocol
  same checkpoint schema
  same label schema
  no tuning the agent to force all tasks to succeed
```

The 100+ target (vs. 30) is to enable slicing by task type, difficulty, trajectory shape, and failure mode — a measurement study, not just a gate-unblock.

### U1. Study the Terminal-Bench ecosystem
Status: done (2026-05-05) — `reports/TB_TASK_SPACE_REVIEW.md` shipped. TB2 (89 tasks, 3/57/29 easy/medium/hard, 24 categories) is the primary pool; original TB fills the easy slot; TB-Pro deferred. Two-arm collection design (degraded top model + mid-tier baseline) selected to land in the 0.40–0.60 outcome-diversity band.

Where to look:
```text
Terminal-Bench official site:    https://www.tbench.ai/
Terminal-Bench 2.0 task list:    https://www.tbench.ai/benchmarks/terminal-bench-2
Terminal-Bench GitHub:           https://github.com/harbor-framework/terminal-bench
Terminal-Bench 2 GitHub:         https://github.com/harbor-framework/terminal-bench-2
Terminal-Bench Pro:              https://github.com/alibaba/terminal-bench-pro
```

Facts to record:
```text
- Terminal-Bench 2.0 lists 89 high-quality tasks across software
  engineering, machine learning, security, data science, and more.
- The original Terminal-Bench benchmark has 80 tasks.
- Terminal-Bench GitHub describes the benchmark as currently having
  roughly 100 tasks and seeking new challenging tasks.
- Terminal-Bench Pro claims a larger systematic extension with 400 tasks.
```

Outputs: `reports/TB_TASK_SPACE_REVIEW.md`.

Acceptance:
```text
Report lists:
  - available task sources;
  - task categories;
  - licensing / usage notes if visible;
  - feasibility of running locally;
  - whether tasks expose verifiers;
  - rough difficulty labels if available.
```

#### U1.a Research sources for task design

```text
Terminal-Bench official site:
  Use for task categories, available task list, task format, current scope.
Terminal-Bench GitHub:
  Use for implementation structure, contribution format, verifier
  conventions, Docker/task layout.
Terminal-Bench 2.0:
  Use for modern task examples and category balance.
Terminal-Bench Pro:
  Use as inspiration for scaling and limitations of the original benchmark.
SWE-bench / SWE-bench Verified:
  Use for software repair task design and patch-verifier structure.
SWE-bench Pro / long-horizon SWE tasks:
  Use for harder software-engineering tasks if available.
METR time-horizon work:
  Use for human-time estimates and task duration as a capability axis.
OSWorld:
  Use later for non-terminal, desktop-oriented task inspiration.
APEX / professional workflow benchmarks:
  Use later for long-horizon cross-tool tasks outside coding.
```

Warning:
```text
Do not turn tb_live_v2 into a literature survey. The goal is not to
cover every benchmark. The goal is to construct an outcome-diverse,
verifiable, live terminal-agent dataset that stresses the observation
channel.
```

### U2. Define the tb_live_v2 sampling policy
Status: done (2026-05-05) — `docs/TB_LIVE_V2_SAMPLING_POLICY.md` shipped. Pre-registered two-arm config (Arm A = top model with degraded budget; Arm B = mid-tier model at default), TB2 + original-TB-easy + 25 internal tasks, anti-tuning rules explicit, rebalance rule for out-of-band batches.

Outputs: `docs/TB_LIVE_V2_SAMPLING_POLICY.md`.

Sampling target:
```text
Total runs:
  target = 100
  minimum = 60 if runtime budget is limited

Outcome diversity:
  target failures >= 25
  hard minimum failures >= 15
  if first 30 runs have < 5 failures, stop and resample harder tasks

Category balance:
  software engineering:           25–35%
  data science / ML:              20–30%
  security / systems:             15–25%
  file/data manipulation:         10–20%
  miscellaneous terminal tasks:   remainder

Difficulty balance:
  easy:    20%
  medium:  40%
  hard:    40%

Trajectory-shape goals:
  at least 15 runs with progress drops
  at least 10 runs with validation-new-work
  at least 10 stuck/blocked runs
  at least  5 high-progress failures if possible
  at least  5 low-progress successes if possible
```

Sampling rule:
```text
Do not select only tasks the current agent can solve.
Do not tune prompts, model, or task set to maximize success.
The dataset needs failures to measure completion risk.
```

Acceptance:
```text
Sampling policy states source task pools, inclusion/exclusion criteria,
target outcome mix, target category mix, and what to do if the first
batch is all successes.
```

### U3. Design new Terminal-Bench-style tasks if needed
Status: 20 of 25 spec rows shipped (4 per shape). Batch-0: lps_01, vnw_01, sb_01, hpf_01, pd_01. Batch-1A: lps_03, vnw_05, sb_02, hpf_02, pd_03. Batch-1B: lps_02, vnw_02, sb_03, hpf_04, pd_04. Batch-2: lps_04, vnw_04, sb_04, hpf_05, pd_05. 5 remaining (lps_05, vnw_03, sb_05, hpf_03, pd_02). **Decision:** further single-file Python tasks won't generate failures from current Claude lineup — Batch 1B and Batch 2 both hit 100% pass even with hidden traps. Future task expansion should switch substrate (real TB2 tasks, multi-file refactors, exotic libraries) rather than continue the spec list. Shared docker-compose template under `tasks/tb_live_v2/docker-compose.template.yaml`.

Goal: If the public task pool does not provide enough diversity, design additional internal tasks following Terminal-Bench conventions.

Task design requirements (per task):
```text
- clear starting environment;
- clear success criterion;
- executable verifier;
- bounded runtime;
- no internet dependency unless explicitly part of the benchmark;
- logs sufficient for ledger observation;
- expected human-time estimate or rough difficulty bucket;
- category label;
- known failure modes if available.
```

Design tasks to create these dynamics:
```text
progress drop:           initial obvious route fails validation.
validation-new-work:     tests reveal a missing edge case.
stuck/blocked:           missing dependency, ambiguous file layout, misleading error.
high-progress failure:   visible subtasks complete while hidden verifier still fails.
low-progress success:    small decisive action solves the task before bookkeeping closes.
late recovery:           agent appears stuck but finds a workaround.
```

Outputs (per task):
```text
tasks/tb_live_v2/<task_id>/task.yaml
tasks/tb_live_v2/<task_id>/Dockerfile or environment spec
tasks/tb_live_v2/<task_id>/verifier.sh or verifier.py
tasks/tb_live_v2/<task_id>/README.md
```

Acceptance:
```text
At least 20 new internal tasks exist if public tasks are insufficient.
Every task runs locally in isolation.
Every task has a deterministic verifier.
Every task has a category and expected difficulty.
```

### U4. Run tb_live_v2 in batches
Status: Batches 0 + 1A + 1B + 2 shipped (2026-05-05) — 20 task scaffolds × 3 arms = 60 runs. 51 pass / 9 fail / 0 unresolved (**15% failure rate; well below 0.25–0.60 sampling band**). Per-arm: A=20/20, B=16/20, C=15/20. Per-shape: lps 12/12, sb 12/12, hpf 10/12, pd 9/12, vnw 8/12. **Batch 2 used hidden traps and still hit 100% pass — the substrate (single-file Python tasks with deterministic verifiers) does not generate failures from current Claude lineup.** Reports: `reports/TB_LIVE_V2_BATCH0_PILOT.md`, `BATCH1A.md`, `BATCH1B.md`, `BATCH2.md`. Recommended next move: stop expanding tasks; rebuild checkpoints/labels on n=60 corpus and run process-dynamics evaluation. Further failures require switching substrate.

Batch plan:
```text
Batch 0: pilot, 10 tasks. Verify sidecar/ledger protocol. Check outcome diversity.
Batch 1: 30 tasks. If failures < 5, resample harder tasks.
Batch 2: 30 more tasks. Aim for shape diversity.
Batch 3: 30 more tasks. Fill missing categories/failure modes.
Final target: 100+ runs.
```

Per-run required artifacts:
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

Required metadata:
```text
task_id, task_family, difficulty, agent_scaffold, model_name,
start_time, end_time, timeout_seconds, final_success,
final_success_source, termination_reason, num_ledger_events,
has_real_wallclock = true
```

Acceptance:
```text
After each batch:
  - rebuild checkpoints and labels;
  - profile success/failure balance;
  - profile shape labels;
  - decide whether to continue, resample harder tasks, or stop.
```

### U5. Build tb_live_v2 estimator artifacts
Status: blocked on U4

Commands:
```bash
uv run python scripts/build_checkpoints.py --source tb_live_v2
uv run python scripts/run_d5_audit.py
uv run python scripts/run_baselines.py
uv run python scripts/run_g5_eval.py
uv run python scripts/run_tb_live_eval.py
uv run python scripts/run_failure_modes.py
uv run python scripts/run_go_no_go.py
```

Outputs:
```text
datasets/checkpoints_tb_live_v2.parquet
datasets/labels_tb_live_v2.parquet
datasets/profiles/tb_live_v2.md
reports/tb_live_v2_eval.md
reports/tb_live_v2_shape_profile.md
reports/tb_live_v2_failure_modes.md
```

Acceptance:
```text
tb_live_v2 supports:
  - process dynamics evaluation;
  - terminal success evaluation;
  - calibration slices by phase;
  - failure-mode slices.
```

---

## § Workstream V — Human baseline

The human baseline answers whether the ledger is readable as a belief signal. Scaffolding (`coding_estimator/eval/human_baseline.py`, `scripts/run_human_baseline.py prepare|compare`) is shipped.

### V1. Run human baseline on existing prompts
Status: not started

Inputs: `reports/human_baseline/prompts/`.

Tasks:
```text
- One human reads each midpoint-prefix prompt.
- Human records probabilities for:
  y_future_progress_drop_h5
  y_validation_new_work_h5
  y_success_eventual
- Human should not see terminal outcome.
```

Outputs:
```text
reports/human_baseline/human_predictions.csv
reports/human_baseline/comparison.md
```

Acceptance:
```text
Report states:
  - whether human beats elapsed time;
  - whether G4/G5 match human on dynamics;
  - whether the ledger is readable as a belief signal.
```

### V2. Repeat human baseline on tb_live_v2
Status: blocked on U4

Acceptance:
```text
Human baseline includes both successes and failures.
Human baseline includes process-dynamics positives and negatives.
```

---

## § Workstream W — Process-dynamics result package

This is the main evaluation package for the current observation channel.

### W1. Build trajectory case studies
Status: not started

Choose examples:
```text
true positive progress drop
false positive progress drop
false negative progress drop
quiet run correctly predicted low-risk
validation-new-work positive
stuck/blocked trajectory
```

Each case study includes:
```text
progress curve
predicted P(progress_drop_h5)
actual progress drops
validation events
reopen/block markers
short interpretation
```

Outputs:
```text
reports/process_dynamics_case_studies.md
reports/figures/process_dynamics_<run_id>.png
```

Acceptance:
```text
At least 4 case studies exist.
Each case uses prefix-only predictions.
```

### W2. Write process-dynamics summary
Status: not started

Outputs: `reports/PROCESS_DYNAMICS_RESULT.md`.

Required sections:
```text
- what the target means;
- why elapsed time is the baseline;
- G2/G4/G5 comparison;
- source-specific results;
- case studies;
- limitations;
- why this is not yet terminal completion risk.
```

Acceptance:
```text
A reader can understand the positive result without reading code.
```

---

## § Workstream X — Completion-risk re-test

Do not keep rerunning terminal success on the same data and calling it progress.

### X1. Define retest prerequisites
Status: not started

Retest terminal success only if at least one is true:
```text
Hermes labels are annotated and combined retrospective pool has:
  >= 45 labeled runs
  >= 15 failures

OR

tb_live_v2 has:
  >= 60 live runs
  >= 15 failures
```

Acceptance:
```text
TASKS.md says terminal-success retest is blocked until prerequisites hold.
```

### X2. Re-run O7 after prerequisites
Status: blocked on X1

Compare:
```text
G2 time-only
G4 ledger-basic
G5 dynamics
G4+G5
```

Targets:
```text
y_success_eventual
future success-by-horizon targets if label support exists
```

Report:
```text
Brier delta vs G2
AUROC
ECE
run-level bootstrap CI
high-progress failure slice
low-progress success slice
```

Acceptance:
```text
Completion-risk result is updated only when the dataset is large and
diverse enough.
```

---

## § Workstream Y — Defer scope creep

Active guardrails. Any PR adding these items must fail review unless it updates TASKS.md with explicit approval.

### Y1. Keep online inference deferred
Status: active guardrail

```text
Do not build online predictor, monitor_run.py, or scheduler-facing API
until P passes or a separate monitoring-only approval is written.
```

### Y2. Keep semantic features deferred
Status: active guardrail

```text
Do not add transcript embeddings, command text embeddings, LLM judges,
or sequence models until:
  - Hermes labels or tb_live_v2 lands;
  - process-dynamics result is written;
  - completion-risk retest prerequisites are defined.
```

### Y3. Keep controller deferred
Status: active guardrail

```text
No task pausing, throttling, power control, API-cost control, or
model-effort policy in this repo. This repo outputs belief estimates only.
```

---

## § 1. Recommended execution order (v1 next phase)

Strict dependencies:

```text
S1 → S2                         done — framing locked
T1                              done — verdict deferred (Hermes is not an outcome source)
T2, T3                          DEFERRED — see Workstream T banner
U1 → U2 → U3 → U4 → U5         tb_live_v2 collection (now the critical path)
V1                              human baseline on existing prompts, can run anytime
V2                              blocked on U4
W1, W2                          process dynamics package, can run after S2
X1 → X2                         completion-risk retest, blocked on U5
Y1, Y2, Y3                      active guardrails, no work — review-time only
```

Parallelism windows:
```text
- V1, W1, W2 can run in parallel with U work.
- U1–U2 (research + sampling policy) and U3 (internal task design) can run
  in parallel once U1 has produced the task-space review.
- X1 can be defined as soon as S2 is done; X2 is gated on U5.
```

---

## § 2. Final instruction to the coding agent

The next phase should not make the estimator more complex. It should make the data more decisive.

If you are tempted to add a model, ask:

> "Would this still matter if Hermes labels and tb_live_v2 landed tomorrow?"

If the answer is no, do not build it.

The immediate priority is:
1. design and collect tb_live_v2 with 100+ outcome-diverse, **verified** tasks (Workstream U);
2. use Hermes HF as a process-dynamics scaling source only — not for terminal-success gates (Workstream T deferred);
3. run the human baseline (Workstream V);
4. package the process-dynamics result (Workstream W);
5. re-test completion risk only after the data can actually answer it (Workstream X).

---

## § 3. Mission-aligned summary

> The estimator is a belief layer over live coding-progress ledgers. The ledger records the evolution of visible discovered work: what the agent has found, attempted, completed, reopened, invalidated, blocked on, and validated. The estimator does not redefine progress and does not decide actions. At each checkpoint, it consumes prefix-only ledger features and outputs calibrated probabilities over successful completion by future horizons, remaining time, and near-future progress dynamics. v0 has shown that prefix-only ledger features predict near-future process dynamics but do not yet improve terminal success prediction over elapsed time. The next phase resolves that limitation through targeted data work — Hermes annotation and an outcome-diverse Terminal-Bench live v2 corpus — not through additional modeling.
