# TASKS - Agent Migrate

This is the working backlog for Agent Migrate. It is now organized around
hypotheses, not workstream codes. If reality diverges, update this file.

## Thesis

Agent Migrate studies clean restart boundaries for LLM-backed tasks.

At a clean boundary, progress can be represented as:

```text
trace
compact trace / summary
environment delta
execution cache
```

The system chooses whether to:

```text
reuse
copy
replay
rebuild
```

The core artifact is:

```text
cut point x package type x structural validity x bytes x restart time
```

This is not primarily a planner project. Planner and regime-map machinery only
matter after restart packages show real resource pressure.

## Current Answer

The strongest current story is:

```text
boundary -> representation -> structural validity -> materialization cost
         -> measured prevalence -> batch pressure
```

As of the 2026-05-06 artifacts:

- Clean cut points exist in recorded traces.
- Static validation can filter structurally invalid packages.
- A trace-derived fixture shows a >10x representation byte gap.
- The retained measured corpus mostly has tiny dirty mobile state.
- Measured-derived batch replay is dominated by prefill pressure, not
  workspace or network bytes.

The negative measured-state result is central: for this retained coding-agent
corpus, the dominant restart strategy may be strong reuse plus small
correctness-critical deltas, not full environment migration.

## Hard Rules

Status markers: `not started`, `in progress`, `done`, `deferred`.

- Import `ledger_progress`; do not fork it.
- Ride on `LedgerEvent`; do not invent a new event class.
- Do not run real harnesses, models, tools, tests, or verifiers from Agent
  Migrate.
- Do not use semantic correctness, `verifier_success`, or task success as an
  Agent Migrate metric.
- Do not add online scheduling, admission control, FIFO queues, priorities, or
  packet-level simulation.
- Use the exact fluid reconstitution simulator for batch claims.
- Do not tune policies until measured restart packages show pressure.
- Do not score a policy podium on a single episode.
- Keep random diversification in batch comparisons.
- Do not expand the public state taxonomy unless the restart-package table
  needs it.

If a task requires violating a rule, stop and escalate.

## Naming Rule

New implementation must be named for behavior, not task codes.

Use semantic names in source, scripts, tests, docs, and generated artifacts:

```text
cut_point_extraction
resume_package_ablation
measured_mobile_state
regime_map_sweep
small_n_oracle
model_profile_regime_sweep
oracle_policy_diff
```

Do not create new names like `h6`, `k10`, `mse5`, or `run_c5.py`.

Historical task-code names may stay in old docs, old run directories, fixture
filenames, and compatibility wrappers. New writing should introduce the
semantic name first and parenthesize the historical code only when needed:

```text
resume-package ablation table (historical C4)
aggregate regime sweep (historical K8)
small-N exact oracle (historical K9)
```

## Core Path

### Restart Boundaries Exist

**Status:** `done`

Agent workflows expose deterministic cut points where no action is in flight
and the next step is a model/tool boundary.

Keep:

- `src/agent_migrate_agent/cut_points.py`
- session-scoped cuts;
- no open subtask;
- next event is an LLM-call node;
- prior declared state must have non-empty content hashes;
- phase classification within the session.

Evidence:

- `swe_agent_pilot_s_07.json`: 11 LLM calls -> 10 cut points.
- `examples/traces/h5a_multi_trajectory_swe.jsonl`: 5 sessions x 2 LLM calls
  -> 5 cut points.

Do not expand to mid-action migration yet.

### Restart Packages Can Be Structurally Validated

**Status:** `done`

At a cut point, more than one representation may be structurally sufficient to
resume.

Keep:

- `src/agent_migrate_agent/resume_packages.py`
- `src/agent_migrate_agent/resume_validator.py`
- `src/agent_migrate_agent/resume_ablation.py`

Package shapes:

```text
prompt_transcript_only
trace_plus_harness_state
base_repo_plus_diff
full_workspace_snapshot
minimal_required_state
```

The validator is a structural filter. It checks transcript prefix, state
coverage, content hashes, workspace digests, harness schema, and diff
applicability. It does not claim semantic correctness.

### Representation Choice Changes Restart Cost

**Status:** `in progress`

If two structurally valid packages differ by >10x in bytes or materially differ
in restart time, representation-aware restart is real.

Current evidence:

- Trace-derived fixture: `base_repo_plus_diff` is 212 bytes and
  `full_workspace_snapshot` is about 1 GB.
- Measured corpus: no non-empty final diffs, so the measured table does not yet
  show two valid representations with a >10x gap.

Existing partial implementation:

- `src/agent_migrate_agent/restart_gate.py` already emits
  `model_resume_s`, `environment_resume_s`, and `task_resume_s`.
- `src/agent_migrate_agent/measured_mobile_state.py` emits the same fields for
  measured package rows.
- `src/agent_migrate_agent/resume_ablation.py` still lacks those fields.

Next required task: promote restart-time fields into the general
resume-package ablation table and regenerate:

```text
runs/restart_packages/restart_package_table.csv
```

Preserve but do not extend as canonical:

```text
runs/c4_ablation/
runs/restart_representation/
```

### Real Runs May Mostly Have Small Mobile State

**Status:** first corpus `done`; upstream coverage `in progress`

Measure what retained agent runs actually leave behind: dirty diffs, touched
files, read files, tool outputs, test logs, build artifacts, dependency caches,
retrieved documents, and retained workspace snapshots.

Keep:

- `src/agent_migrate_agent/measured_mobile_state.py`
- upstream exporter:
  `coding-data-collection/src/coding_data_collection/mobile_state.py`
- artifacts: `runs/measured_mobile_state/`

Current result from 232 claim-usable retained runs:

- dirty payload never exceeds 1 MB;
- dependency/build/cache never exceeds 100 MB;
- tool/test output never exceeds 10 MB;
- no row crosses the SWE-bench, medium, monorepo, or large-artifact thresholds;
- largest retained full workspace snapshot is about 115 MB, but that is a
  representation cost until unchanged base bytes can be separated.

Claim boundary:

```text
In retained coding-agent runs, mobile dirty state is usually tiny; the dominant
restart strategy is reconstructing or reusing a base environment and moving
small correctness-critical deltas.
```

Next data needs:

- initial-workspace or base-archive manifests;
- full tool-output artifacts when retained;
- workloads that create large caches, build artifacts, retrieved documents,
  generated outputs, or non-empty dirty diffs.

### Batch Restart Pressure Is Regime-Dependent

**Status:** `in progress`

Many-task restart matters when valid restart packages create pressure on:

```text
prefill / replay
network transfer
workspace hydration
cache residency
```

Keep:

- exact fluid simulator: `src/agent_migrate_agent/fluid_sim.py`
- resource model: `episode.py`, `resources.py`, `warmness.py`
- policies: `src/agent_migrate_agent/reconstitution.py`
- measured replay helper: `measured_restart_pressure(...)`

Current measured-derived result:

- exact replay on the 20 largest measured dirty rows is dominated by
  trace-derived prompt tokens;
- `mixed_min_pressure` beats strong reuse and random on p50 in that episode;
- this is landing/prefill pressure, not evidence for large workspace or network
  pressure.

Next batch artifact, after restart-package timing is canonical:

```text
runs/batch_restart_pressure/exact_batch_restart_table.csv
```

Metrics:

```text
p50/p90/p99 task_resume_s
dominant bottleneck
strong reuse sufficiency
value of richer materialization over strong reuse
comparison against random diversification
```

Use "restart bottleneck cells" or "materialization regimes," not "policy
winners."

### Architecture Changes The Execution-Cache Tradeoff

**Status:** `deferred`

Model architecture changes the replay-vs-cache-transfer tradeoff. Keep the
model profiles and KV/replay arithmetic, but only make a figure if it supports
a specific claim.

Optional future artifact:

```text
runs/architecture_cache_tradeoff/kv_replay_phase.csv
runs/architecture_cache_tradeoff/kv_replay_phase.pdf
```

## Immediate Tasks

### Promote Restart Timing Into Resume Ablation

**Status:** `not started`

Add to `resume_ablation.py`:

```text
model_resume_s
environment_resume_s
task_resume_s
resume_metric_kind
```

Do not silently mix byte-estimate timing with older exact-fluid proxy columns.
The metric kind must make that distinction explicit.

### Regenerate Canonical Package Tables

**Status:** `not started`

Run the canonical package table on:

- at least three trace-derived cut points;
- the measured retained snapshots already in
  `runs/measured_mobile_state/measured_restart_package_table.csv`;
- any new upstream rows with non-empty dirty diffs or better base-workspace
  separation.

Answer:

```text
How often is prompt/transcript enough?
How often is base+diff enough?
How often is full workspace necessary?
How large is the byte gap?
How large is the task_resume_s gap?
```

### Promote The Measured Small-State Result

**Status:** `not started`

Write or update a short memo so the measured negative result is not buried
behind synthetic regime maps.

Claim boundary:

- valid for the retained coding-agent corpus measured on 2026-05-06;
- not evidence that all agent workloads are small-state;
- large-state mobility remains a stress-regime mechanism until upstream
  captures larger dirty workspaces or artifacts.

### Replay Batch Pressure From Canonical Package Costs

**Status:** `deferred`

Resume only after the canonical package table has `task_resume_s`.

Compare:

```text
strong_site_reuse
random_diversification
mixed_min_pressure
```

Report restart pressure under measured package distributions, not planner wins.

## Semantic Naming Cleanup Plan

**Status:** planning `done`; implementation `not started`

Subagents inspected the codebase. Land cleanup incrementally with compatibility
wrappers.

Implementation order:

1. Add semantic script names and output directories; keep old scripts as
   wrappers.
2. Add semantic source/API aliases for public helpers.
3. Rename source modules and tests once aliases are green.
4. Preserve historical run directories and reports as audit records.

Source/API targets:

| Current | Semantic target |
| --- | --- |
| `k8_regime.py` | `regime_map.py` |
| `K8_POLICIES` | `REGIME_MAP_POLICIES` |
| `run_k8_cell` | `run_regime_cell_exact` |
| `estimate_k8_cell` | `estimate_regime_cell_aggregate` |
| `run_k8_sweep` | `run_regime_sweep` |
| `k8_validation.py` | `regime_validation.py` |
| `run_k8_validation` | `run_exact_regime_validation` |
| `k9_oracle.py` | `small_n_oracle.py` |
| `r3_model_sweep.py` | `model_profile_regime_sweep.py` |
| `run_r3_sweep` | `run_model_profile_regime_sweep` |
| `w_under_r3.py` | `workload_anchor_model_profile_matrix.py` |
| `run_w_r3_matrix` | `run_anchor_model_profile_matrix` |
| `oracle_diff.py` | `oracle_policy_diff.py` |

Script targets:

| Current | Semantic target |
| --- | --- |
| `scripts/run_c1_cut_points.py` | `scripts/run_cut_point_extraction.py` |
| `scripts/run_c4_ablation.py` | `scripts/run_resume_package_ablation.py` |
| `scripts/run_k8_k9.py` | `scripts/run_regime_map_and_oracle.py` |
| `scripts/run_k8_validation.py` | `scripts/run_regime_validation.py` |
| `scripts/run_r3_model_sweep.py` | `scripts/run_model_profile_regime_sweep.py` |
| `scripts/run_w_under_r3.py` | `scripts/run_workload_anchor_model_profile_matrix.py` |
| `scripts/run_o2.py` | `scripts/run_oracle_policy_diff.py` |
| `scripts/run_s2_audit.py` | `scripts/run_mobile_state_audit.py` |

Future writers should prefer:

```text
runs/resume_package_ablation/
runs/regime_map_sweep/
runs/small_n_exact_oracle/
runs/exact_claim_cell_validation/
runs/model_profile_regime_sweep/
runs/oracle_policy_diff/
runs/workload_anchor_model_sensitivity/
runs/batch_restart_pressure/
```

Preserve these historical directories:

```text
runs/c4_ablation/
runs/k7_gauntlet/
runs/k8_regime_map/
runs/k8_validation/
runs/k9_oracle/
runs/r3_model_sweep_pilot/
runs/o2_oracle_diff/
runs/w_under_r3/
```

## Supporting Infrastructure

Keep, but do not make these the public story:

- trace -> manifest pipeline:
  `events.py`, `manifest.py`, `manifest_io.py`, adapters;
- state layers and roles:
  `state_layers.py`, `workspace.py`;
- materialization-mode registry:
  `materialization_modes.py`;
- strong site reuse baseline:
  use `strong_site_reuse` in writing and keep `cache_reuse` for compatibility;
- random diversification baseline:
  keep `random_mode` in batch comparisons.

## Exploratory Or Deferred

Aggregate regime sweep:

- status: `done`, exploratory only;
- historical implementation: `src/agent_migrate_agent/k8_regime.py`;
- historical artifacts: `runs/k8_regime_map/`;
- use as search hints only.

Exact claim-cell validation:

- status: `done`;
- historical implementation: `src/agent_migrate_agent/k8_validation.py`;
- historical artifacts: `runs/k8_validation/`;
- result: all seven selected aggregate cells needed exact simulation before
  timing or bottleneck claims.

Small-N exact oracle:

- status: `done`, diagnostic only;
- historical implementation: `src/agent_migrate_agent/k9_oracle.py`;
- historical artifacts: `runs/k9_oracle/`;
- oracle gaps matter only after package costs are real.

Oracle policy difference and quantile-aware planning:

- status: `deferred`;
- historical artifacts: `runs/o2_oracle_diff/`;
- do not resume until measured restart packages create enough pressure to
  justify planner-quality work.

Workload anchors:

- status: `done`, stress fixtures only;
- artifacts: `runs/workload_anchors/`, `runs/w_under_r3/`;
- current bytes are synthetic hypothesis fixtures, not measured prevalence
  evidence.

Heuristic policy work:

- status: `deferred`;
- the one-step-lookahead negative result is enough for now.

Simulated end-to-end restart demo:

- status: `deferred`;
- must remain structural and simulated if resumed.

## Historical Result Index

- The original trace/manifest/cost/policy pipeline is complete and useful.
- Strong per-site reuse collapsed the original grouping gap on linear real
  traces.
- Synthetic 1 GB workspace fixtures showed state-locality pressure can exist.
- Real working-tree byte sums collapsed that synthetic gap.
- The mobility-episode gauntlet showed prefill stampede and multi-resource
  stress, but not a universal planner claim.
- Aggregate maps are exploratory; exact simulation is required for claims.
- The small-N oracle is a ceiling diagnostic, not a product planner.
- The measured mobile-state corpus is reuse/small-dirty-state dominated.
- The measured package tradeoff is negative/partial because non-empty dirty
  diffs and initial/base manifests are missing.

Historical docs:

- `docs/WEEK1_REPORT.md`
- `docs/K7_gauntlet_results.md`
- `docs/K8_K9_regime_map_and_oracle.md`
- `docs/K8_exact_validation.md`
- `docs/R4_regime_discovery_memo.md`
- `docs/strong_site_reuse_baseline.md`
- `kv-transfer-early-experiment/FINDINGS.md`

## Repo Grounding

Read before changing behavior:

- `AGENTS.md`
- `CLAUDE.md`
- `agent-migrate_agent_repo_implementation_plan.md` as historical reference;
- `../coding-progress-ledger/ledger_progress/{core,session,serialization,queries,sidecar}.py`;
- `../coding-data-collection/` before changing measured-state capture.
