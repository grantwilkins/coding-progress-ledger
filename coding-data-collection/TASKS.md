# TASKS - coding-data-collection

This repository is the experiment harness and data factory for
long-horizon coding-agent traces.

It is **not** a modeling repo and **not** a ledger-semantics repo.

```text
orchestration/adapters/artifacts -> coding-data-collection
progress semantics/replay/scoring -> coding-progress-ledger
features/labels/models/eval      -> coding-estimator
```

Status markers: `not started`, `scaffolded`, `in progress`, `blocked`,
`done`, `deferred`.

## Current State - 2026-05-05

This repo has a first scaffold, contract-hardening pass, and completed
feasibility spike. It does **not** yet run pilot Terminal-Bench collection.

Shipped scaffold:

```text
README.md
pyproject.toml
.python-version
docs/RUN_PROTOCOL.md
docs/ARTIFACT_LAYOUT.md
docs/OBSERVATION_EVENT_CONTRACT.md
docs/BENCHMARK_ADAPTERS.md
docs/BENCHMARK_DATA_POLICY.md
docs/PILOT_GATES.md
policies/COLLECTION_BUDGET_POLICY.md
policies/TERMINAL_BENCH_PILOT_POLICY.md
policies/PROTOCOL_CHANGES.md
src/coding_data_collection/
scripts/
tests/
```

Critic-review fixes already applied:

```text
- prepare_run.py now excludes nested hidden paths and skips symlinks.
- leakage scanning now flags tests/, verifier/oracle path components, and
  content markers such as oracle solution / gold patch / test_patch.
- artifact completeness is status-specific:
  completed_* requires full terminal artifacts;
  agent_* requires process-dynamics artifacts;
  verifier_* also requires verifier_output.txt;
  infra/setup/leakage statuses require only partial failure artifacts.
- finalize_run.py no longer marks missing verifier results as terminal
  model failures; it records infrastructure_failure.
- semantic tests now cover sparse-step verifier timing, visibility flags,
  path normalization, nested hidden files, status-specific artifact sets,
  task-scoring arithmetic, and ledger wire event shape.
```

Verification:

```bash
env UV_PROJECT_ENVIRONMENT=.venv312 uv run pytest tests
# 54 passed
```

Known limitation:

```text
The scripts are still protocol scaffolds. Feasibility proved the
`hf_archive_custom` path and the smoke corpus now builds estimator artifacts
through `coding-estimator`, but production HF archive extraction and richer
native transcript capture are still not complete. Do not start a pilot from
this repo until Workstream I hardening and the K preflight pass.
```

## 0. Feasibility Spike - Harbor vs HF Archive

Status: done

Goal: choose the Terminal-Bench execution path before building the pilot.

Tasks:

- Run one easy/medium Terminal-Bench task through Harbor oracle.
- Run one task through `ia03/terminal-bench` archive extraction if feasible.
- For each path, produce a run-shaped directory with:
  ```text
  task.md
  task_metadata.json
  environment_manifest.json
  protocol_manifest.json
  transcript.jsonl
  observation_events.jsonl
  events.jsonl
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  run_manifest.json
  verifier_output.txt
  run_notes.md
  ```
- Explicitly answer:
  ```text
  Can oracle/test/verifier internals be hidden from the agent phase?
  Can per-step transcript events be captured without fighting Harbor?
  Can native observation_events.jsonl be emitted at the right step?
  Can events.jsonl replay through coding-progress-ledger sidecar?
  Can the verifier be rerun from a clean verifier phase with same result?
  Does Harbor expose enough hooks, or is HF archive custom execution needed?
  ```

Acceptance:

- `reports/FEASIBILITY_SPIKE.md` chooses exactly one primary path:
  `harbor_native`, `hf_archive_custom`, or `hybrid`.
- The report includes one artifact tree listing per attempted path.
- The report lists blockers and a go/no-go decision for the 24-run pilot.
- `scripts/run_pilot.py` remains blocked until this is done.

Result:

- `reports/FEASIBILITY_SPIKE.md` chooses `hf_archive_custom`.
- Harbor local migration/oracle succeeded on `aimo-airline-departures` with
  reward `1.0`; remote Harbor registry lookup timed out.
- HF archive custom extraction hid oracle/verifier/canary-bearing files from
  the agent workspace and reran hidden tests in a clean Docker verifier phase.
- Run-shaped artifact directories:
  `runs/feasibility/harbor_native_aimo_run/` and
  `runs/feasibility/hf_archive_custom_aimo_run/`.
- 24-run pilot remains no-go until Workstreams H-J complete.

## A. Repo Scaffold and Dependency Wiring

Status: done

Shipped:

- Package scaffold under `src/coding_data_collection/`.
- `pyproject.toml` with editable sibling dependencies:
  `../coding-progress-ledger`, `../coding-estimator`.
- Python pinned to `3.12` / `>=3.11,<3.13` to match estimator constraints
  and avoid Python 3.14 pyarrow builds.
- `.gitignore` excludes local virtualenvs, pytest cache, bytecode, and
  generated egg-info.

Acceptance: met.

## B. Shared Run Protocol and Artifact Layout

Status: done

Shipped:

- `ProtocolVersions`
- `RunStatus`
- required / estimator-stage / conditional artifact lists
- status-specific artifact completeness logic
- analysis inclusion flags for terminal success, process dynamics, and
  artifact quality

Files:

```text
src/coding_data_collection/protocol.py
src/coding_data_collection/artifacts.py
docs/RUN_PROTOCOL.md
docs/ARTIFACT_LAYOUT.md
```

Completed tasks:

- Add JSON Schema files for:
  ```text
  run_manifest.json
  task_metadata.json
  environment_manifest.json
  protocol_manifest.json
  observation_events.jsonl
  ledger wire events
  ```
- Add validators that fail before pilot collection.
- Decide whether `artifact_incomplete` should preserve any partially
  produced run artifacts for process-dynamics audits.

Result:

- Schemas added under `schemas/`.
- `src/coding_data_collection/validation.py` and `scripts/validate_run.py`
  validate run manifests, task/environment/protocol manifests,
  `observation_events.jsonl`, and ledger wire events.
- `artifact_incomplete` preserves produced artifacts for artifact-quality and
  harness-debug audits only; it is excluded from terminal-success and
  process-dynamics analysis.

Acceptance:

- Every emitted artifact declares compatible schema/protocol versions.
- Completed runs cannot pass artifact completeness without full required
  artifacts.
- Infrastructure/setup failures cannot be silently included in
  terminal-success analysis.

## C. Benchmark Registry and Data Policy

Status: done

Shipped:

- Inspect-only adapters:
  ```text
  TerminalBenchHFAdapter
  HarborTerminalBenchAdapter
  SWEBenchProAdapter
  ```
- CSV registry writer.
- `scripts/inspect_benchmark.py`.
- `docs/BENCHMARK_DATA_POLICY.md`.

Current boundary:

```text
Adapters inspect local rows or provided task IDs.
They do not fetch remote datasets.
They do not extract archives.
They do not run Harbor.
They do not pull Docker images.
```

Completed tasks:

- Add real source-fetch commands or documented manual export path.
- Write `reports/BENCHMARK_SOURCE_AUDIT.md`.
- Record license/usage terms and canary-handling notes per source.
- Confirm `ia03/terminal-bench` field coverage from real rows.
- Confirm Harbor registry/task IDs from current Harbor output.
- Confirm SWE-bench Pro inspect fields from real dataset rows.

Result:

- Manual/source export paths documented in `docs/BENCHMARK_ADAPTERS.md` and
  `docs/BENCHMARK_DATA_POLICY.md`.
- `reports/BENCHMARK_SOURCE_AUDIT.md` records source fields, license/usage
  notes, and canary/gold/verifier handling.
- Redacted real samples:
  `datasets/source_samples/terminal_bench_hf_rows.jsonl` and
  `datasets/source_samples/swe_bench_pro_rows.jsonl`.
- Current Harbor task IDs recorded in
  `manifests/harbor_terminal_bench_tasks.json`.

Acceptance:

- At least 3 real tasks/instances from each adapter can be inspected.
- Raw archives, hidden tests, oracle files, verifier internals, and gold
  patches are never copied into committed artifacts.

## D. Docker / Container Substrate

Status: done

What exists today:

- `scripts/prepare_run.py` creates an agent workspace from a local task dir.
- It recursively skips hidden path components and symlinks.
- `src/coding_data_collection/docker_substrate.py` defines phase-specific
  Docker command construction and manifest metadata.
- `scripts/run_docker_substrate_smoke.py` runs a no-op substrate smoke.
- `runs/d_smoke/noop_aimo/` is a validated completed-failure smoke where the
  agent phase cannot see hidden tests and the verifier fails cleanly from a
  clean copied workspace.
- `runs/d_smoke/oracle_hello_world/`,
  `runs/d_smoke/oracle_grid_pattern_transform/`, and
  `runs/d_smoke/oracle_aimo_airline_departures/` are validated
  completed-success privileged oracle smokes.
- Oracle smoke workspaces preserve `oracle_workspace_snapshot/` before the
  verifier workspace is cleaned, so oracle-produced product files remain
  auditable without exposing oracle files to the agent phase.
- Agent crash/timeout exits finalize as `agent_crash` / `agent_timeout`
  instead of falling through to verifier success, and timeout paths use named
  containers for cleanup.
- Transcript rows retain both a display command and exact `command_argv`.

Tasks:

- Build isolated agent and verifier phases.
- Mount agent workspace writable.
- Mount task spec read-only.
- Do not mount hidden tests/verifier/oracle/gold during agent phase.
- Disable network by default; require explicit task metadata exceptions.
- Record:
  ```text
  image digest
  Dockerfile hash
  task archive hash
  CPU limit
  memory limit
  disk limit
  wall-clock limit
  network policy
  ```
- Run verifier in a clean verifier phase or clean reset of workspace.

Acceptance:

- Hidden-file exposure test passes from inside agent-visible environment.
- A no-op agent fails cleanly and emits required failure artifacts.
- Oracle smoke run passes for at least 3 Terminal-Bench tasks.

Result:

- Hidden-file exposure and no-op failure acceptance met for
  `aimo-airline-departures`.
- Resource, hash, and network-policy metadata are recorded in
  `environment_manifest.json`.
- Agent network defaults to disabled; verifier network exceptions are explicit.
- Three-task oracle smoke passed for `hello-world`,
  `grid-pattern-transform`, and `aimo-airline-departures`.
- All four D smoke run directories validate with `scripts/validate_run.py`.

## E. Terminal-Bench Smoke Tests

Status: done

What exists today:

- The chosen path is `hf_archive_custom`, from
  `reports/FEASIBILITY_SPIKE.md`.
- D smoke artifacts provide the required HF archive oracle/no-op evidence:
  ```text
  runs/d_smoke/noop_aimo/
  runs/d_smoke/oracle_hello_world/
  runs/d_smoke/oracle_grid_pattern_transform/
  runs/d_smoke/oracle_aimo_airline_departures/
  ```
- `scripts/verify_verifier_determinism.py` reruns a verifier from a preserved
  workspace and compares semantic verifier signatures rather than raw Docker
  logs.
- `runs/d_smoke/oracle_hello_world/verifier_determinism_report.json`
  records a two-trial deterministic rerun check.

Tasks:

- Run Harbor oracle smoke for at least 3 tasks if Harbor path is chosen.
- Run HF archive oracle/no-op smoke if archive path is chosen.
- Verify deterministic verifier rerun on a sample.

Acceptance:

- Oracle passes for at least 3 tasks.
- No-op fails cleanly.
- Verifier rerun reproduces the same outcome.

Result:

- Harbor was not selected as the primary path, so Harbor oracle smoke is not
  required for E.
- HF archive no-op failure is validated by `runs/d_smoke/noop_aimo/`.
- HF archive oracle success is validated for `hello-world`,
  `grid-pattern-transform`, and `aimo-airline-departures`.
- Deterministic verifier rerun on `oracle_hello_world` reproduced:
  ```text
  exit_code=0
  collected=2
  passed=2
  failed=0
  errors=0
  warnings=1
  failed_tests=[]
  ```
  across both rerun trials.

## F. Native Observation Events

Status: done

Shipped:

- `src/coding_data_collection/observation.py`
- `src/coding_data_collection/observation_quality.py`
- `docs/OBSERVATION_EVENT_CONTRACT.md`
- `scripts/audit_observation_quality.py`
- `reports/OBSERVATION_F_QUALITY_REPORT.json`

Implemented event extraction from transcript-shaped rows:

```text
validation_attempt
validation_pass_observed
validation_fail_observed
error_observed
error_repeated
environment_blocked
product_file_written
product_file_edited
agent_claims_done
verifier_pass
verifier_fail
verifier_disagreement
expected_file_missing
oracle_artifact_read
```

Hardening already tested:

```text
terminal verifier events use max(transcript.step) + 1
transcript-derived events are visible_to_agent = true
verifier/missing-file terminal events are visible_to_agent = false
expected path matching normalizes ./ prefixes
hidden test reads are treated as oracle_artifact_read
```

Remaining tasks:

- None for F. Later pilot work still needs richer live terminal capture beyond
  transcript-shaped shell rows.

Acceptance:

- Terminal verifier events are never visible to preterminal checkpoints.
- Observation event quality gates can be computed for a pilot report.

Result:

- `scripts/validate_run.py` validates `observation_events.jsonl` against
  `schemas/observation_event.schema.json` and rejects terminal verifier events
  emitted at or before the final transcript step.
- `scripts/run_docker_substrate_smoke.py` emits observation events from real
  HF Docker smoke transcripts.
- `scripts/audit_observation_quality.py` computes shell-row coverage for:
  ```text
  exit_code
  stdout_snippet / obs_snippet
  stderr_snippet
  hidden_phase_events_visible_to_agent
  terminal_events_visible_to_agent
  observation schema validity
  ```
- F audit over the four HF smoke runs passed with:
  ```text
  run_count=4
  shell_rows=11
  shell_exit_code_coverage=1.0
  shell_stdout_snippet_coverage=1.0
  shell_stderr_snippet_coverage=1.0
  hidden_phase_events_visible_to_agent=0 for every run
  terminal_events_visible_to_agent=0 for every run
  median_observation_events_per_run=4.0
  ```
- The smoke corpus does not satisfy the later pilot gate
  `median_observation_events_per_run >= 10`; the audit computes this as
  `pilot_gates.median_observation_events_per_run_passed=false`.

## G. Ledger Sidecar Replay Integration

Status: done

Shipped:

- `src/coding_data_collection/ledger.py`
- `src/coding_data_collection/ledger_sidecar_audit.py`
- `scripts/audit_ledger_sidecar.py`
- `reports/LEDGER_G_SIDECAR_AUDIT.json`
- `reports/LEDGER_G_REPLAY_AUDIT.json`
- transcript-shaped rows convert into ledger wire events with schema `1.0.0`.
- wrapper calls `python -m ledger_progress.sidecar`.

Policy:

```text
Pilot runs use a hybrid ledger event source:

- explicit row-level ledger_ops are preserved when present;
- otherwise transcript-shaped rows fall back to conservative inferred add+start
  ops by category, with only the agent `done` boundary completed.
```

Remaining tasks:

- None for G. Later pilot work may replace some inferred transcript ops with
  richer explicit agent-emitted ledger ops.

Acceptance:

- Sidecar replay succeeds on at least one Phase 0 run without local
  reimplementation of ledger scoring.

Result:

- Existing HF smoke finalization replays through `../coding-progress-ledger`
  sidecar and emits:
  ```text
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  ```
- `runs/g_sidecar/oracle_hello_world_replay/` is a fresh G-only replay from
  `runs/d_smoke/oracle_hello_world/events.jsonl`; sidecar returned `0`.
- The G replay directory is collection-validator compatible as an
  `artifact_incomplete` / `sidecar_replay_only` artifact-quality run.
- `scripts/audit_ledger_sidecar.py` verifies sidecar artifact presence,
  non-empty progress outputs, `summary_by_category.json::generator`, and
  `source_ledger_sha256` consistency without recomputing ledger scoring. It
  also includes collection validation issues in the audit result.
- Both the four HF smoke runs and the fresh G replay passed ledger sidecar
  audits.

## H. Estimator Artifact Production

Status: done

Shipped:

- `src/coding_data_collection/estimator_artifacts.py`
- `scripts/build_artifacts.py`
- `reports/H_ESTIMATOR_ARTIFACT_STATUS.md`
- `reports/H_ESTIMATOR_ARTIFACT_REPORT.json`
- Smoke estimator artifacts under:
  ```text
  datasets/d_smoke_estimator/checkpoints.parquet
  datasets/d_smoke_estimator/labels.parquet
  datasets/d_smoke_estimator/estimator_predictions.parquet
  datasets/d_smoke_estimator/checkpoint_feature_manifest.json
  datasets/d_smoke_estimator/estimator_source_manifest.json
  ```

Estimator-side bridge:

- `coding-estimator` now exposes `terminal_bench_pilot` as a non-canonical
  first-party live source.
- `coding-estimator/scripts/build_collection_artifacts.py` owns checkpoint,
  label, prediction, and feature-manifest artifact construction.
- `coding-data-collection` stages run directories into
  `../coding-estimator/runs/terminal_bench_pilot/` as symlinks and validates
  the resulting artifact contract.

Tasks:

- Define how new corpus IDs are exposed to `coding-estimator`. done
- Build checkpoints, labels, estimator predictions, and feature manifests. done
- Copy or reference outputs into `runs/<corpus>/<run_id>/` or
  `datasets/<corpus>/` according to `docs/ARTIFACT_LAYOUT.md`.
- Validate prefix provenance:
  ```text
  max_ledger_step_used
  max_observation_step_used
  ```
  done

Acceptance:

- No estimator model logic lives in this repo. met
- Estimator artifacts are produced by `coding-estimator`. met
- Prefix provenance exists for 100% of checkpoint rows. met

Result:

- `scripts/build_artifacts.py --corpus-id d_smoke_estimator ...` produced
  estimator artifacts for the four D smoke runs.
- `reports/H_ESTIMATOR_ARTIFACT_REPORT.json` passed with:
  ```text
  checkpoint_rows=19
  prefix_provenance_complete=true
  issues=[]
  ```

## I. Audits and Hardening

Status: done

Shipped:

- `scan_agent_workspace_for_leakage`
- `prefix_safety_report`
- `run_quality_report`
- `scripts/audit_prefix_safety.py`
- `scripts/audit_corpus_artifacts.py`
- visibility-aware redaction audit for task prompts, visible transcript rows,
  visible ledger event payloads, and visible observation payloads.
- validation-attempt precision and miss-rate sampling.
- corpus artifact completeness/schema audit.
- semantic tests for leakage, prefix safety, status artifacts, and
  finalization status.
- `scripts/verify_verifier_determinism.py`
- `scripts/audit_observation_quality.py`
- `reports/I_HARDENING_AUDIT_REPORT.json`
- `reports/I_HARDENING_STATUS.md`

Remaining tasks:

- None for pre-K hardening. Future pilot runs must rerun the same audit over
  the actual 24-run corpus.

Acceptance:

- Audits must run and pass before Workstream K. met for the current smoke
  corpus.

Result:

- `scripts/audit_corpus_artifacts.py runs/d_smoke ...` passed with:
  ```text
  run_count=4
  redaction.leakage_incidents=0
  artifact_completeness.passed=true
  validation_attempt_precision.sample_precision=1.0
  validation_attempt_precision.recall_miss_rate=0.0
  ```

## J. Candidate Task Scoring

Status: done

Shipped:

- `src/coding_data_collection/task_scoring.py`
- exact arithmetic tests for trajectory richness, operational risk, and
  pilot priority.
- `scripts/score_pilot_candidates.py`
- `manifests/pilots/terminal_bench_candidate_calibration.csv`
- `manifests/pilots/terminal_bench_candidate_scores.csv`
- `reports/J_CANDIDATE_TASK_SCORING_STATUS.md`

Current scoring formula:

```text
trajectory_richness =
  runtime_bucket
  + 2 * validation_visibility
  + file_edit_complexity
  + environment_complexity
  + 2 * failure_modes

operational_risk =
  3 * oracle_test_leakage_risk
  + (5 - docker_feasibility)
  + 3 if requires_internet
  + 2 if large_download_or_build

pilot_priority = trajectory_richness - operational_risk
```

Remaining tasks:

- None for pre-pilot selection. If K remains strictly `hf_archive_custom`,
  replace the two selected Harbor contingency tasks with HF-extractable tasks
  before running agents.

Acceptance:

- Pilot sample selection is documented before any agent run. met

Result:

- 14 Terminal-Bench candidates scored.
- 12 tasks selected for pilot priority across 9 categories.
- The output manifest includes all scoring inputs, computed richness/risk,
  `pilot_priority`, `selected_for_pilot`, and calibration notes.

## K. Terminal-Bench Pilot

Status: V10 pre-pilot passed L; targeted sample still no-go on visible validation failures

Plan:

```text
12 tasks
2 arms
24 runs
```

Acceptance:

- Every run emits required artifacts.
- No run with leakage enters estimator training.
- Pilot gate report is generated.

Shipped:

- `scripts/run_pilot.py` now performs a fail-closed preflight instead of
  exiting as an opaque stub.
- `src/coding_data_collection/pilot_plan.py` builds a 12-task / 2-arm /
  24-run dry-run plan from `terminal_bench_candidate_scores.csv`.
- `manifests/pilots/terminal_bench_pilot_plan.json` records the current
  blocked plan and exact runnable commands for tasks that do pass preflight.
- Typed pilot arms now distinguish `shell_command` protocol-smoke backends
  from `model_tool_loop` real-agent backends.
- `scripts/run_model_agent_pilot.py` runs a host-side model controller against
  a network-disabled Docker task sandbox using scripted/mock model actions.
- `src/coding_data_collection/agents/model_client.py` defines the provider
  neutral model-client protocol plus:
  ```text
  ScriptedModelClient       deterministic integration tests
  ProviderModelClient       external provider adapter command via stdin/stdout
  ```
- `scripts/audit_real_model_run.py` provides the one-run go/no-go audit before
  any 3-task mini-pilot.
- `scripts/openai_model_client.py` provides an OpenAI Responses API provider
  adapter command using `OPENAI_API_KEY` and strict JSON-schema action output.
- `scripts/openai_compatible_model_client.py` provides an OpenAI-compatible
  Chat Completions adapter for OpenRouter and similar routers using
  `OPENROUTER_API_KEY` by default.
- `docs/MODEL_AGENT_PROMPT_CONTRACT.md` documents the prompt/tool boundary.
- `docs/MODEL_PROVIDER_ROUTING.md` documents provider/model routing policy and
  L-eligibility constraints for OpenRouter-style routed calls.
- Added backend/recording/sandbox modules:
  ```text
  src/coding_data_collection/agents/
  src/coding_data_collection/sandbox/
  src/coding_data_collection/recording/
  ```

Current preflight blockers:

```text
None for the scripted scout plan.
None for the one-run OpenAI milestone.
V10 pre-pilot passed L, but the broader targeted sample did not. Next step is
to revise the full 12-task provider-backed plan toward visible-validation-fail
tasks; do not run Workstream M until the L gate passes.
```

Result:

- Replaced the two Harbor contingency tasks with HF-extractable tasks:
  `blind-maze-explorer-algorithm`, `classifier-debug`, and
  `nginx-request-logging` entered calibration; the balanced selector retained
  12 strict-HF tasks.
- Extracted selected HF archives into untracked `/private/tmp/houdini_tb_hf/`
  scratch.
- Added compose-style `client/Dockerfile` support for `extract-safely`.
- Ran a 24-run scripted scout/validation collection under
  `runs/terminal_bench_pilot/`.
- Built estimator artifacts under `datasets/terminal_bench_pilot_estimator/`.
- Added one verifier determinism rerun report for
  `classifier-debug__scout`.
- This is not yet a final model-agent pilot because the two arms were scripted
  shell commands, not real coding-agent arms.
- Added the first implementation milestone for the real-agent architecture:
  host-side `model_tool_loop` backend, Docker sandbox executor, incremental
  transcript recorder, typed arm planning, and deterministic mocked-model tests.
  This proves the artifact path without requiring model API credentials.
- Added strict action parsing, invalid JSON/action retry recording, provider
  usage metadata, and L-eligibility separation:
  ```text
  shell_command           protocol smoke, L-ineligible
  scripted model_tool_loop integration smoke, L-ineligible
  provider model_tool_loop real pilot, L-eligible
  ```
- Added controller-side early-done guard for provider-backed model loops:
  ```text
  min_steps_before_done default: 15
  require_validation_before_done default: true
  allow_blocked_done default: true
  rejected done action event: early_done_denied
  ```
- Completed guarded provider-backed OpenAI mini-pilot under
  `runs/real_model_mini3_guarded/`; summary:
  `reports/REAL_MODEL_MINI3_GUARDED_STATUS.md`.
- Valid guarded mini-pilot artifacts:
  ```text
  runs/real_model_mini3_guarded/classifier-debug__openai_gpt51
  runs/real_model_mini3_guarded/nginx-request-logging__openai_gpt51
  runs/real_model_mini3_guarded/adaptive-rejection-sampler__openai_gpt51
  reports/real_model_mini3_guarded_classifier_debug_audit.json
  reports/real_model_mini3_guarded_nginx_request_logging_audit.json
  reports/real_model_mini3_guarded_adaptive_rejection_sampler_audit.json
  ```
- Completed one provider-backed OpenAI run:
  ```text
  run_dir=runs/real_model_one/classifier-debug__openai_gpt51_v2
  audit=reports/real_model_one_classifier_debug_v2_audit.json
  audit_passed=true
  transcript_rows=25
  observation_event_rows=9
  events_rows=25
  ledger_rows=51
  terminal_event_steps=[26, 26]
  run_status=completed_failure
  verifier_result=task failure, expected A but model wrote I
  ```
- The first live attempt also produced
  `runs/real_model_one/classifier-debug__openai_gpt51`, but verifier bootstrap
  needed a network exception; the v2 run is the milestone artifact.
- Completed a provider-backed OpenAI mini-pilot under
  `runs/real_model_mini3/`; summary:
  `reports/REAL_MODEL_MINI3_STATUS.md`.
- Valid audited mini-pilot artifacts:
  ```text
  runs/real_model_mini3/classifier-debug__openai_gpt51
  runs/real_model_mini3/nginx-request-logging__openai_gpt51_v2
  runs/real_model_mini3/adaptive-rejection-sampler__openai_gpt51_v2
  reports/real_model_mini3_classifier_debug_audit.json
  reports/real_model_mini3_nginx_request_logging_v2_audit.json
  reports/real_model_mini3_adaptive_rejection_sampler_v2_audit.json
  ```
- Mini-pilot no-go result:
  ```text
  all audits passed
  no hidden-file leakage
  all runs reached verifier failure/disagreement
  transcript rows: 22, 10, 10
  failed criterion: at least 2/3 runs with transcript rows >= 15
  ```
- The first `nginx-request-logging` and `adaptive-rejection-sampler` mini-pilot
  attempts exposed a YAML prompt materialization bug for `instruction: |-`;
  `scripts/run_model_agent_pilot.py` now handles `instruction: |*` block
  scalars, and the `_v2` runs are the valid artifacts.
- After the no-go mini-pilot, `ModelToolLoopBackend` was updated to reject
  premature `done` actions until the configured depth and validation policy are
  satisfied, unless visible transcript evidence supports a genuinely blocked
  run.
- Guarded mini-pilot pass result:
  ```text
  all audits passed
  no hidden-file leakage
  all runs reached verifier failure/disagreement
  transcript rows: 22, 16, 12
  at least 2/3 runs with transcript rows >= 15: pass
  validation/shell-check coverage: pass
  product write/edit coverage: pass
  ```
- Reran the 3-task mini-pilot with two provider-backed OpenAI arms:
  `gpt-5.4` and `gpt-5.4-mini`. Valid run root:
  `runs/real_model_mini3_gpt54_vs_mini_v3/`; summary:
  `reports/REAL_MODEL_MINI3_GPT54_VS_MINI_STATUS.md`.
- Valid audited GPT-5.4 / GPT-5.4-mini artifacts:
  ```text
  runs/real_model_mini3_gpt54_vs_mini_v3/classifier-debug__openai_gpt54
  runs/real_model_mini3_gpt54_vs_mini_v3/nginx-request-logging__openai_gpt54
  runs/real_model_mini3_gpt54_vs_mini_v3/adaptive-rejection-sampler__openai_gpt54
  runs/real_model_mini3_gpt54_vs_mini_v3/classifier-debug__openai_gpt54mini
  runs/real_model_mini3_gpt54_vs_mini_v3/nginx-request-logging__openai_gpt54mini
  runs/real_model_mini3_gpt54_vs_mini_v3/adaptive-rejection-sampler__openai_gpt54mini
  reports/real_model_mini3_gpt54_vs_mini_v3_classifier_debug_gpt54_audit.json
  reports/real_model_mini3_gpt54_vs_mini_v3_nginx_request_logging_gpt54_audit.json
  reports/real_model_mini3_gpt54_vs_mini_v3_adaptive_rejection_sampler_gpt54_audit.json
  reports/real_model_mini3_gpt54_vs_mini_v3_classifier_debug_gpt54mini_audit.json
  reports/real_model_mini3_gpt54_vs_mini_v3_nginx_request_logging_gpt54mini_audit.json
  reports/real_model_mini3_gpt54_vs_mini_v3_adaptive_rejection_sampler_gpt54mini_audit.json
  ```
- GPT-5.4 / GPT-5.4-mini mini-pilot result:
  ```text
  all audits passed
  no hidden-file leakage
  transcript rows: 28, 32, 54, 18, 22, 40
  validation/shell-check coverage: pass
  product write/edit coverage: pass
  terminal outcome mix: fail for scale; 6/6 verifier failures
  ```
- The GPT-5.4 rerun exposed and fixed two provider-loop issues:
  `scripts/openai_model_client.py` now parses multi-chunk Responses API output
  without concatenating independent JSON actions, and `ProviderModelClient`
  no longer turns provider adapter failures into accepted `done` actions.
- The rerun also exposed a scale-readiness issue: tool output snippets are
  tail-biased, which caused repeated long-file inspection on
  `adaptive-rejection-sampler`. Add head+tail snippets or chunked file reads
  before the 12-task pilot.
- Completed the failure-triage and first tool-affordance hardening pass:
  ```text
  reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.md
  reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.csv
  reports/TOOL_AFFORDANCE_GAPS.md
  reports/HARNESS_REASSESSMENT.md
  scripts/triage_model_failures.py
  ```
- Corrected the failure taxonomy: the old `environment_network` label was too
  broad and hid harness readiness problems. The six GPT-5.4 / GPT-5.4-mini
  failures currently classify as:
  ```text
  agent_image_missing_runtime=3
  no_network_install_mismatch=2
  tool_affordance=1
  ```
  with repeated long-output/truncation evidence preserved separately.
- Added first-class model-loop tools:
  ```text
  find_files
  grep
  read_file(start_line, end_line)
  apply_patch
  ```
  plus head+tail output snippets, visible network/dependency blocked
  messages, and observation events for:
  ```text
  network_blocked
  dependency_missing
  tool_output_truncated
  repeated_file_inspection
  chunked_file_read
  ```
- Added agent readiness preflight to `scripts/run_model_agent_pilot.py`.
  Future no-network model-agent runs fail before model calls with:
  ```text
  run_status=environment_setup_failure
  termination_reason=agent_readiness_preflight_failed
  eligible_for_L_gate=false
  ```
  when the image lacks task-required runtimes/dependencies or the task requires
  solve-time package installation that is incompatible with a no-network
  sandbox.
- Reran the same GPT-5.4 / GPT-5.4-mini mini-pilot through readiness preflight
  only under:
  ```text
  runs/real_model_mini3_gpt54_vs_mini_v4_preflight/
  reports/REAL_MODEL_MINI3_GPT54_VS_MINI_V4_PREFLIGHT_STATUS.md
  ```
  The provider-backed rerun command was blocked by the approval reviewer as
  potential external data exfiltration, so the rerun used no-provider readiness
  mode against the same image tags. All six task/image pairs stopped before
  model calls with `environment_setup_failure`, `eligible_for_L_gate=false`:
  ```text
  classifier-debug: python_imports_available failed (numpy/torch missing)
  adaptive-rejection-sampler: r_runtime_available failed
  nginx-request-logging: nginx_available failed and solve-time install flagged
  ```
  All six partial run directories validate.
- Started the approved provider-backed 12-task / 24-run pilot under:
  ```text
  runs/terminal_bench_real_pilot_gpt54_vs_mini/
  manifests/pilots/terminal_bench_real_pilot_gpt54_vs_mini_plan.json
  ```
  and aborted it after detecting hidden-artifact leakage on
  `blind-maze-explorer-algorithm__gpt54`. The prepared agent workspace exposed
  `protected/` files containing Terminal-Bench canary text, and the model read
  `protected/maze_server.py`.
- Quarantined the leaked run and wrote:
  ```text
  reports/TERMINAL_BENCH_REAL_PILOT_ABORTED_LEAKAGE.md
  ```
- Hardened preparation and leakage scanning:
  ```text
  scripts/prepare_run.py now hides protected/
  scripts/prepare_run.py sanitizes visible copied canary marker lines
  src/coding_data_collection/audits.py flags protected/ and terminal-bench-canary
  ```
  Do not resume provider collection from the aborted run root. Regenerate the
  plan, run no-provider workspace/preflight over all 24 runs, and only then
  restart provider calls in a fresh run root.
- After the `prepare_run.py` hardening, regenerated and scanned agent
  workspaces for all 12 selected pilot tasks without provider calls:
  ```text
  runs/preflight_workspace_leakage_after_abort/workspace_leakage_report.json
  task_count=12
  failed=[]
  ```
- Regenerated the approved provider-backed 12-task / 24-run pilot in a fresh
  root after leakage and tool/preflight hardening:
  ```text
  runs/terminal_bench_real_pilot_gpt54_vs_mini_v9/
  manifests/pilots/terminal_bench_real_pilot_gpt54_vs_mini_v9_plan.json
  reports/TERMINAL_BENCH_REAL_PILOT_V9_STATUS.md
  reports/terminal_bench_real_pilot_gpt54_vs_mini_v9_corpus_audit.json
  reports/TERMINAL_BENCH_REAL_PILOT_V9_GATE_REPORT.json
  reports/TERMINAL_BENCH_REAL_PILOT_V9_FAILURE_ANALYSIS.md
  datasets/terminal_bench_real_pilot_gpt54_vs_mini_v9_estimator/
  ```
- Regenerated V9 derived artifacts after V10 validation/ledger hardening.
  Current V9 corpus result:
  ```text
  run_count=24
  completed_success=10
  completed_failure=6
  environment_setup_failure=8
  eligible_for_L_gate=16
  terminal_failure_rate=0.375
  checkpoint_rows=385
  prefix_provenance_complete=true
  leakage_incidents=0
  ```
- The 8 environment setup failures are fail-closed preflight exclusions before
  model calls:
  ```text
  adaptive-rejection-sampler: hidden/protected image artifact or runtime mismatch
  blind-maze-explorer-algorithm: hidden /protected image artifact readable
  classifier-debug: missing task-required Python imports
  nginx-request-logging: nginx unavailable and solve-time install incompatible
  ```
- V9 L gate remains a scale no-go, but no longer because of all-failure model
  outcomes or missing progress-drop signal:
  ```text
  passed:
    median_transcript_steps=18.0
    validation_attempt_run_fraction=0.5625
    terminal_failure_rate=0.375
    progress_drop_run_fraction=1.0
    high_progress_failure_or_disagreement_count=9
    shell snippet coverage=1.0
    prefix provenance
    zero leakage
    verifier determinism
  failed:
    median_observation_events_per_run=9.5
    validation_fail_observed_run_fraction=0.0625
  ```
- Prepared V10 targeted iteration:
  ```text
  reports/V9_VALIDATION_ATTEMPT_AUDIT.md
  reports/V9_LEDGER_DISCOVERY_AUDIT.md
  reports/V9_SETUP_FAILURE_TRIAGE.md
  manifests/pilots/terminal_bench_v10_candidate_scores.csv
  reports/TERMINAL_BENCH_V10_TASK_SELECTION.md
  ```
- Added explicit transcript-to-ledger bridge hardening: concrete successful
  tool rows complete their own leaves, failed tool/controller rows block their
  leaves, and final verifier failures reopen prior visible work when they show
  it was incomplete. Scoring remains delegated to `coding-progress-ledger`.
- Added regeneration and artifact-builder ergonomics:
  ```text
  scripts/regenerate_derived_artifacts.py
  scripts/build_artifacts.py --run-root
  ```
- Ran V10 4-task / 2-arm pre-pilot:
  ```text
  runs/terminal_bench_v10_prepilot/
  reports/TERMINAL_BENCH_V10_PREPILOT_GATE_REPORT.json
  reports/terminal_bench_v10_prepilot_corpus_audit.json
  datasets/terminal_bench_v10_prepilot_estimator/
  ```
  Result:
  ```text
  run_count=8
  completed_success=3
  completed_failure=5
  median_transcript_steps=29.5
  median_observation_events_per_run=17.5
  validation_attempt_run_fraction=0.75
  validation_fail_observed_run_fraction=0.625
  progress_drop_run_fraction=1.0
  terminal_failure_rate=0.625
  high_progress_failure_or_disagreement_count=6
  L gate passed=true
  ```
- Ran V10 9-task / 2-arm targeted sample:
  ```text
  runs/terminal_bench_v10_targeted/
  reports/TERMINAL_BENCH_V10_TARGETED_GATE_REPORT.json
  reports/TERMINAL_BENCH_V10_TARGETED_FAILURE_ANALYSIS.md
  reports/TERMINAL_BENCH_V10_TARGETED_STATUS.md
  reports/terminal_bench_v10_targeted_corpus_audit.json
  datasets/terminal_bench_v10_targeted_estimator/
  ```
  Result:
  ```text
  run_count=18
  completed_success=11
  completed_failure=7
  median_transcript_steps=21.5
  median_observation_events_per_run=10.0
  validation_attempt_run_fraction=0.6111111111
  validation_fail_observed_run_fraction=0.0555555556
  progress_drop_run_fraction=1.0
  terminal_failure_rate=0.3888888889
  high_progress_failure_or_disagreement_count=11
  L gate passed=false
  failed_gate=validation_fail_observed_coverage
  ```
  The targeted sample diluted the visible-failure pre-pilot tasks with
  easy-success and hidden-verifier-disagreement tasks. Do not run Workstream M;
  revise the full 12-task plan toward visible-validation-failure likelihood.
- Full verification after V10 harness changes:
  ```bash
  env UV_PROJECT_ENVIRONMENT=.venv312 uv run pytest tests
  # 103 passed
  ```
- Added OpenRouter / OpenAI-compatible provider routing support:
  ```text
  scripts/openai_compatible_model_client.py
  docs/MODEL_PROVIDER_ROUTING.md
  reports/OPENROUTER_PROVIDER_ROUTING_STATUS.md
  ```
  The adapter records requested model, resolved model, alias/routing changes,
  fallback usage, provider routing policy, token usage, and provider cost
  credits. Policy:
  fixed model IDs with `allow_fallbacks=false` may be L-eligible; `openrouter/auto`
  and fallback chains are scouting/debug only unless a future protocol defines
  routed-arm analysis.
- Live OpenRouter adapter smoke passed after refreshing `.zshrc`:
  ```text
  model=baidu/cobuddy:free
  resolved_model=baidu/cobuddy-20260430:free
  action_type=read_file
  tokens_in=207
  tokens_out=84
  cost_credits=0
  ```
  This remains smoke-only and should not be used as a benchmark-quality pilot
  arm.
- Ran an end-to-end OpenRouter free-model harness smoke:
  ```text
  reports/OPENROUTER_FREE_SMOKE_STATUS.md
  runs/openrouter_free_smoke/hello-world__cobuddyfree
  runs/openrouter_free_smoke/hello-world__cobuddyfree_v2
  reports/openrouter_free_smoke_corpus_audit.json
  ```
  Result:
  ```text
  v1 completed_failure because verifier bootstrap had no network/curl/uv
  v2 completed_success with verifier-only network exception
  agent sandbox network=disabled
  pilot_type=openrouter_free_smoke
  eligible_for_L_gate=false
  resolved_model=baidu/cobuddy-20260430:free
  fallback_call_count=0
  total_cost_credits=0.0
  valid_run_dir=true
  ```
  The smoke corpus audit has artifact/redaction pass and zero leakage, but
  validation-attempt precision is false because this toy run did not produce a
  recognized visible validation attempt. This is acceptable for L-ineligible
  smoke and should not be treated as pilot data.
- Full verification after OpenRouter adapter changes:
  ```bash
  env UV_PROJECT_ENVIRONMENT=.venv312 uv run pytest tests
  # 109 passed
  ```
- Prepared the OpenRouter L1 fixed-model gate plan:
  ```text
  manifests/pilots/terminal_bench_v10_openrouter_l1_plan.json
  run_root=runs/terminal_bench_v10_openrouter_l1/
  tasks=grid-pattern-transform, broken-python, attention-mil, csv-to-parquet
  arms=openai/gpt-5.4, openai/gpt-5.4-mini via OpenRouter
  provider.allow_fallbacks=false
  provider.data_collection=deny
  eligible_for_L_gate=true
  ```
- Live OpenRouter L1 execution was blocked by approval review before any
  provider calls because the run would send local task/workspace contents to
  an external service. The plan remains ready, but do not treat this as an L
  result until the live provider run is explicitly approved and completed.
- After explicit user approval, the live OpenRouter L1 execution was retried
  and blocked again by tenant-policy external-disclosure review. No provider
  calls were made from this environment.
- User ran the approved OpenRouter L1 command outside the restricted tool
  runner and completed the 8-run collection:
  ```text
  reports/terminal_bench_v10_openrouter_l1_execution.json
  reports/terminal_bench_v10_openrouter_l1_corpus_audit.json
  reports/TERMINAL_BENCH_V10_OPENROUTER_L1_ESTIMATOR_REPORT.json
  reports/TERMINAL_BENCH_V10_OPENROUTER_L1_GATE_REPORT.json
  reports/TERMINAL_BENCH_V10_OPENROUTER_L1_FAILURE_ANALYSIS.md
  datasets/terminal_bench_v10_openrouter_l1_estimator/
  runs/terminal_bench_v10_openrouter_l1/
  ```
  Result:
  ```text
  run_count=8
  nonzero_count=0
  completed_success=0
  completed_failure=8
  eligible_for_L_gate=8
  fallback_call_count=0
  checkpoint_rows=44
  prefix_provenance_complete=true
  leakage_incidents=0
  L gate passed=false
  median_transcript_steps=3.0
  median_observation_events_per_run=4.0
  validation_attempt_run_fraction=0.0
  validation_fail_observed_run_fraction=0.0
  progress_drop_run_fraction=0.125
  terminal_failure_rate=1.0
  high_progress_failure_or_disagreement_count=0
  verifier_determinism_passed=false
  ```
  The run is not benchmark-quality L data. Seven of eight runs recorded only
  model-parse-error trajectories with zero provider usage metadata; one
  `openai/gpt-5.4-mini` run resolved to `openai/gpt-5.4-mini-20260317`, made
  valid provider calls, then also degraded into invalid JSON model output.
- Added provider-route fail-closed hardening before another OpenRouter L
  attempt:
  ```text
  ProviderModelClient returns typed provider_adapter_error payloads.
  ModelToolLoopBackend records provider_adapter_error transcript events instead
  of collapsing adapter failures into model_parse_error.
  scripts/execute_pilot_plan_continue.py preflights each unique provider arm
  once before launching planned runs and refuses corpus execution on failure.
  scripts/run_model_agent_pilot.py preflights the provider route before the
  agent loop and records provider_route_preflight.json.
  run manifests only keep eligible_for_L_gate=true when real agent-loop
  provider calls record nonzero usage, resolved model metadata, and zero
  fallback calls.
  ```
  Targeted regression tests cover adapter stderr preservation, route preflight
  rejection, provider metadata requirements, and pre-execution arm preflight.
- The first rerun after provider-route preflight correctly stopped before the
  corpus and exposed an OpenRouter adapter parser bug: `openai/gpt-5.4` returned
  two valid action JSON objects in one message, and the adapter exited before
  emitting usage/provider metadata. Fixed
  `scripts/openai_compatible_model_client.py` to scan JSON-ish content and
  return the first valid action object, including repeated JSON-object content.
- Rolled the L1 path back to direct OpenAI after OpenRouter adapter instability:
  ```text
  scripts/openai_model_client.py now emits normalized provider metadata
  adapter=openai_responses
  fallback_used=false
  requested_model / resolved_model / response_id recorded
  manifests/pilots/terminal_bench_v10_openai_l1_plan.json
  run_root=runs/terminal_bench_v10_openai_l1/
  ```
  The local tool runner blocked launching this direct-OpenAI collection because
  it would send task/workspace/transcript context to an external service. The
  plan is ready to run from a user terminal with `OPENAI_API_KEY`.
- Ran the same OpenRouter L1 plan in no-provider preflight mode:
  ```text
  manifests/pilots/terminal_bench_v10_openrouter_l1_preflight_plan.json
  reports/terminal_bench_v10_openrouter_l1_preflight_execution.json
  reports/terminal_bench_v10_openrouter_l1_preflight_corpus_audit.json
  runs/terminal_bench_v10_openrouter_l1_preflight/
  run_count=8
  nonzero_count=0
  artifact_completeness_passed=true
  leakage_incidents=0
  ```
  The preflight corpus audit is intentionally not a passing L audit because
  preflight-only runs produce no transcript validation attempts.

## L. Pilot Gate Report

Status: gate machinery done; protocol-smoke runs excluded from real L metrics

Do not scale unless all gates in `docs/PILOT_GATES.md` pass.

If any gate fails:

```text
write reports/PILOT_FAILURE_ANALYSIS.md
do not run Workstream M
recommend task resampling or harness fixes
```

Shipped:

- `src/coding_data_collection/pilot_gates.py`
- `scripts/build_pilot_gate_report.py`
- `reports/PILOT_GATE_REPORT.json`
- `reports/PILOT_FAILURE_ANALYSIS.md`
- `reports/L_PILOT_GATE_STATUS.md`
- L gate metrics now count only runs whose manifest metrics include:
  ```text
  pilot_type=real_agent_pilot
  eligible_for_L_gate=true
  ```
  Shell-command smoke runs are retained for artifact audits but are not treated
  as real agent trajectories.

Result on current scripted scout corpus:

- Gate report is fail-closed and writes failure analysis automatically.
- Estimator checkpoints are cross-checked against run observation events for
  prefix safety, and labels/predictions must reference known checkpoint IDs.
- Current scripted scout corpus remains no-go for scale and is now classified
  as protocol smoke rather than real pilot data:
  ```text
  total_run_count=24
  eligible_run_count=0 after smoke manifests are regenerated
  median_transcript_steps=3.0
  median_observation_events_per_run=9.0
  terminal_failure_rate=1.0
  failed_gates:
    median_transcript_steps
    terminal_failure_rate
    median_observation_events_per_run
    artifact_hardening
  ```

## M. Scale Batch

Status: blocked until real `model_tool_loop` pilot passes L

Plan:

```text
40 tasks
2 arms
80 runs
```

Non-goal:

```text
Do not scale merely because the pilot completed. Scale only if the
event-rate and observation-quality gates pass.
```

Result:

- Re-ran `scripts/build_pilot_gate_report.py` on the current scripted scout
  corpus.
- Scale remains blocked because `reports/PILOT_GATE_REPORT.json` is
  fail-closed:
  ```text
  run_count=24
  median_transcript_steps=3.0
  median_observation_events_per_run=9.0
  terminal_failure_rate=1.0
  failed_gates:
    median_transcript_steps
    terminal_failure_rate
    median_observation_events_per_run
    artifact_hardening
  ```
- `reports/PILOT_FAILURE_ANALYSIS.md` was regenerated and still says not to
  run Workstream M until all gates pass.

## N. SWE-bench Pro Inspect Adapter

Status: done

Shipped:

- `SWEBenchProAdapter` emits future command plans with:
  ```text
  repo
  base_commit
  dockerhub_tag
  docker_image
  problem_statement
  visible_test_route
  hidden_evaluation_route
  patch_output_path
  expected_diff_format
  ```

Remaining tasks:

- None for inspect mode. Do not run SWE-bench Pro collection until Workstream
  M passes.

Result:

- Inspected 3 real redacted SWE-bench Pro rows from
  `datasets/source_samples/swe_bench_pro_rows.jsonl`.
- Regenerated `manifests/swe_bench_pro_registry.csv`.
- Source audit section is already present in
  `reports/BENCHMARK_SOURCE_AUDIT.md`.

## O. SWE-bench Pro Pilot

Status: deferred

Plan:

```text
10 instances
10-20 runs
```

Gated after Terminal-Bench pilot and scale gates pass.

## P. Mobile-State Snapshot Export

Status: done

Purpose:

Support Agent Migrate's measured mobile-state question without running new
harnesses from Agent Migrate.

Shipped:

- `coding_data_collection.mobile_state.snapshot_run(...)` measures retained
  post-run artifacts from existing run directories.
- `scripts/export_mobile_state_snapshots.py` recursively scans nested
  `runs/<batch>/<run>/run_manifest.json` layouts and emits per-run
  `*.mobile_state.json` plus `raw_snapshot_index.csv`.
- The exporter labels final diffs as patch-file bytes, transcript tool output
  as lower-bound snippet bytes, and missing initial workspace state explicitly.
- Symlinks are skipped; hardlinks are inode-deduped; protected/leakage rows are
  marked unusable for claims.

Result:

- Exported 238 retained Terminal-Bench-style run manifests to
  `../agent-migrate/runs/measured_mobile_state/raw_snapshot_index.csv`.
- 232 rows are usable by Agent Migrate after workspace-retention and
  leakage/protected-artifact filtering.
- The measured dirty/mobile payload is small in this corpus; the largest
  retained workspace is about 115 MB, but initial workspace manifests are
  missing, so final workspace size is not a dirty-state headline.

Remaining tasks:

- Add initial workspace/base archive byte manifests if future Agent Migrate
  claims need clean-base vs dirty-state separation.
- Capture full tool-output artifacts if the runner retains them; current
  snapshot bytes are transcript-snippet lower bounds.
