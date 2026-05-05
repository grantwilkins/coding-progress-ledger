# TASKS — coding-estimator

This file is the working backlog for `coding-estimator`, the **belief-state layer** that consumes append-only ledger histories produced by `../coding-progress-ledger` and outputs calibrated, checkpoint-level beliefs over latent remaining work, completion-by-horizon, and process-dynamics events.

The goal is **not** to redefine progress, replace the ledger, or build a controller. It is to answer one question, on live ledger histories, with calibration:

> Given the prefix-only ledger history `H_t` of a long-horizon coding task at checkpoint `t`, output calibrated probabilities over (a) eventual success, (b) success-by-horizon, (c) remaining time conditional on success, and (d) near-future progress-dynamics events (drops, reopens, validation surprises, stuck loops, scope discovery).

The repository it sits next to (`../coding-progress-ledger`) already contains:

- Append-only ledger semantics (`ledger_progress/`).
- Replay/scoring and JSONL serialization.
- Retrospective ingestion for SWE-agent and Hermes (`scripts/import_*`, `scripts/normalize_*`, `scripts/auto_annotate_hermes.py`, `scripts/annotate_pilots_from_spec.py`).
- Live sidecar/run-manager primitives (`ledger_progress/sidecar.py`, `ledger_progress/run_manager.py`).
- Terminal-Bench live tooling: `tasks/tb_live/` (12 task specs), `runs/tb_live/` (12 live runs with real wall-clock timestamps), `scripts/tb_emit.py`, `scripts/validate_tb_run.py`.
- Observation/checkpoint dataset builders: `scripts/build_ledger_observation_dataset.py`, `scripts/build_estimator_checkpoints.py`, `scripts/build_q_labels.py`, `scripts/label_observation_shapes.py`, `scripts/q_baselines.py`, `scripts/smoke_test_completion_prediction.py`.
- Existing checkpoint tables in `datasets/`: SWE-agent (191 rows, 20 runs), Hermes (multiple variants), TB live (per-run summaries).
- Q1 channel-native targets (`docs/Q_TARGETS.md`) — `future_progress_drop`, `product_reopened_after_completion`, `validation_exposes_new_work`, `stuck_loop_next_window`, `submit_without_validation_state`.

This repo is downstream of those artifacts. We **read** them; we do **not** rewrite them.

---

## § 0.0β v0 reality check — recentered (READ AFTER § 0.0)

The v0 measurement is in (commit 6e12dfc / 96d6558 / and the post-feedback hardenings that follow). It is honest and informative: the pipeline tells the operator the estimator is not ready, and *why*. The publishable shape of the v0 result has been recentered:

```text
Primary v0 claim:
  Prefix ledger features predict near-future progress dynamics
  (y_future_progress_drop_h5, y_validation_new_work_h5).

Secondary / negative v0 claim:
  Prefix ledger features do not improve terminal success prediction
  (y_success_eventual) over elapsed time on small retrospective data.

Interpretation:
  The observation channel measures work-frontier dynamics before it
  becomes a reliable completion estimator.
```

Full evidence, with numbers: `reports/V0_FINDINGS.md`. Per-target G2/G4/G5 comparison: `reports/g5/g5_eval.md`.

### What the next phase is

In strict order. Do *not* re-run the entire pipeline before (1) and (2) land — the gate is currently bottlenecked on data, not models.

```text
1. Annotate the 30 hermes_pilot_h5_v2 runs upstream (currently
   final_success: null, annotation_mode: not_annotated for all 30).
   Diagnosis: reports/HERMES_LABEL_DIAGNOSIS.md.
   Unblocks P1.c on ~50 retrospective runs.
2. Collect tb_live_v2 with outcome diversity (≥ 30 runs / ≥ 10
   failures / real wall-clock / same protocol). Without failures
   tb_live cannot test any success-prediction gate. Unblocks
   P1.b, P1.d, and the tb_live half of O7.
3. Run the human baseline (scripts/run_human_baseline.py is
   ready). One human reads 6 midpoint TB ledger prefixes and
   predicts; the comparison answers whether the ledger is
   readable as a belief signal.
4. Re-run the full pipeline only after (1)–(3); regenerate every
   report deterministically.
5. Optionally: revisit G5 dynamics features once the data
   defects are fixed. They already help on the dynamics targets;
   the question is whether they help on the recentered headline
   under more diverse data.
```

### What the next phase is NOT

```text
- Building a scheduler, controller, monitor, or online inference
  surface (Workstreams M, R remain explicitly deferred).
- Adding semantic / text features (Workstream Q remains deferred —
  would help prediction but muddy the "ledger-native" claim).
- Loosening the +0.02 O7 threshold to get a pass. The strict gate
  is the informative gate; the failure is signal, not noise.
- Promoting a tb_live result obtained on the all-success cohort.
  Re-evaluate only against tb_live_v2.
```

### Workstreams added since the original plan

```text
G5 (ledger-dynamics, post-processing layer)
   coding_estimator/checkpoints/dynamics.py — attach_g5_features
   coding_estimator/baselines/ledger_dynamics.py — LEDGER_DYNAMICS
   scripts/run_g5_eval.py → reports/g5/
D5 audit (structured JSON)
   coding_estimator/leakage/d5_audit.py — required by P1.g
   scripts/run_d5_audit.py → reports/d5_audit.{md,json}
Human baseline scaffolding
   coding_estimator/eval/human_baseline.py
   scripts/run_human_baseline.py prepare|compare → reports/human_baseline/
V0 findings memo
   reports/V0_FINDINGS.md
   reports/REVIEWER_BRIEFING.md
Hermes label-gap diagnosis
   reports/HERMES_LABEL_DIAGNOSIS.md
```

---

## § 0. Project rules for all agents

These rules apply to every workstream. They are stricter than the upstream ledger rules because the estimator is the most leakage-prone surface in the project.

```text
0.1  Do not mutate any artifact under ../coding-progress-ledger/.
0.2  Do not redefine ledger semantics. Progress is what coding-progress-ledger
     says it is. We add features and labels around it, never inside it.
0.3  Do not use any ledger event with step > S when constructing features at
     checkpoint S. The replayer must be prefix-only.
0.4  Do not use any post-run artifact when constructing features at S unless
     that artifact was visible at step S in the live trace.
     Examples of forbidden post-run leakage as features: final_diff.patch,
     final test_output.txt, final eval_logs, summary_by_category.json,
     verifier_exit_code, final_success.
0.5  Do not split train/test by checkpoint; always split by run_id.
0.6  Do not train large neural models in the first ladder. Logistic
     regression and calibrated GBMs come first.
0.7  Do not output policy actions ("pause", "stop", "throttle"). The
     estimator outputs belief state only.
0.8  Do not equate progress with success. Progress is a *measurement of
     visible discovered work*, not correctness. The estimator's job is to
     learn which progress patterns *imply* completion risk.
0.9  Do not fabricate data when an upstream artifact is missing. Skip the
     run, record it in the profile report, and move on.
0.10 Hard fail over silent fallback. If a feature is undefined for a
     checkpoint (e.g. wall-clock timestamp missing on a retrospective
     row), emit `null`/`NaN` and let the leakage audit surface it.
0.11 Calibration is a first-class output. A ranked-but-uncalibrated model
     is not shippable, even if AUROC looks good.
0.12 Every model artifact must ship with a model card stating intended
     use, data sources, splits, and known failure modes. No exceptions.
```

**Status markers:** `not started` · `in progress` · `blocked` · `done` · `deferred`. These are plain text — they are *not* ledger events. The ledger is for runs we measure, not for tracking estimator work.

---

## § 0.0 Data-budget reality check (READ FIRST)

This plan describes a layered estimator. Before any modeling, every agent must internalize how thin the data actually is.

```text
Source                         Runs   Live wall-clock?   Annotation provenance
─────────────────────────────────────────────────────────────────────────────
swe_agent_pilot                  20   no                 retrospective (LLM/human)
swe_agent_pilot_v3               20   no                 retrospective (revised protocol)
swe_agent_live                   20   synthetic          retrospective replayed via sidecar
swe_agent_live_wallclock         20   synthetic-backfill retrospective replayed via sidecar
hermes_pilot* (h5, h5_v2)       ~30   no                 retrospective (LLM annotation)
tb_live (TB-12)                  12   YES (real)         live, agent-emitted, real timestamps
─────────────────────────────────────────────────────────────────────────────
                          total ≈ 80–100 runs, but only 12 are first-party live.
```

The headline live source has 12 runs with median ~13–30 events per run; the lightest run (`markdown-to-html-cli`) ran 88 seconds with 13 events.

**Implications that override naive ladders:**

```text
- Targets with positive rate < 0.03 at N=20 cannot be evaluated under
  LORO at N=12. Don't even add them as v0 targets.
- Bootstrap CIs at N=12 (run-level resampling, the only legitimate kind)
  are wide. "G4 beats G2 with CI excluding zero" is not a v0 gate.
- Wall-clock features are populated on ~12 runs. Any model that depends
  on them is a tb_live-only model.
- Retrospective ledgers were annotated knowing the outcome. Annotator
  outcome knowledge is the deepest leakage class on swe_agent and hermes
  sources, and it is unfixable here (lives upstream). It must be named,
  not hidden.
- The model ladder must collapse: G1 (constant), G2 (time-only), G4
  (ledger-basic) are mandatory v0; G3, G5, G6, I2 (GBM), I3 (hazard) are
  deferred until N > 200 runs total.
```

This section pins our north-star: **the v0 estimator is a small, calibrated, no-regression demonstration on a tiny dataset, not a polished classifier.**

**Empirical findings (2026-05-04, end of Workstream G):**

```text
- hermes_pilot_h5_v2: ALL 30 runs are unannotated (annotation_mode ==
  "not_annotated", source_metadata.final_success == null). Label build
  emits 0 rows. P1.c's "swe_agent_pilot u hermes_pilot_h5_v2 ~ 50 runs"
  premise is BROKEN as of this snapshot. Either annotated hermes runs
  must land upstream, or P1.c must be restated. The non-canonical
  swe_agent_live (20 labeled) is the most likely substitute pool.
- tb_live y_success_eventual is 12/0 (twelve successes, zero failures
  across the snapshot). P1.a "G4 >= G2 on at least one source" is
  unsatisfiable on tb_live for that target — the gate must be evaluated
  on swe_agent_pilot (with annotation-leakage caveat stamped).
- LOSO swe_agent_pilot -> tb_live transfer for y_future_progress_drop_h5
  is positive: G4 stays near 0 on drop-quiet runs and spikes at the
  correct step on the 3 positive cases (gut-check, not statistical).
  This is evidence the LORO tracking on swe_agent_pilot is not pure
  annotation leakage. See reports/trajectory_confidence_loso_swe_to_tb.png.
- Per-checkpoint targets (y_future_progress_drop_h5) produce far more
  legible trajectory plots than run-constant targets. The headline
  figure for the v0 sign-off package (Workstream N) should be a
  per-checkpoint trajectory plot, not a run-constant one.
```

---

## § 0.1 Repository scope and relationship to coding-progress-ledger

```text
../coding-progress-ledger          coding-estimator (here)
─────────────────────────────      ───────────────────────────────
ledger_progress/             →     ingestion only (read-only import)
scripts/build_estimator_*    →     re-used as artifact source
runs/<source>/<run_id>/      →     read-only inputs
datasets/*.csv               →     re-used or rebuilt with stricter
                                   forbidden-column gates
                              ←    estimator outputs (predictions,
                                   reports, models) live HERE
```

Concrete contract:

```text
Inputs we read (never write):
  ../coding-progress-ledger/runs/swe_agent_pilot/**/ledger.jsonl
  ../coding-progress-ledger/runs/swe_agent_live*/**/ledger.jsonl
  ../coding-progress-ledger/runs/hermes_pilot*/**/ledger.jsonl
  ../coding-progress-ledger/runs/tb_live/**/ledger.jsonl
  ../coding-progress-ledger/runs/<run>/test_output.txt          (final label only)
  ../coding-progress-ledger/runs/<run>/run_manifest.json        (final label only)
  ../coding-progress-ledger/runs/<run>/live_instrumentation.json (timestamp metadata)

Outputs we write (only here):
  datasets/checkpoints_<source>.parquet
  datasets/labels_<source>.parquet
  datasets/features_<source>.parquet
  datasets/profiles/*.md
  datasets/splits_<source>.json
  models/<estimator_id>/{model.pkl,model_card.md,calibration.json,...}
  reports/*.md
```

Two repositories means two `pyproject.toml`s and two test suites. Ledger semantic tests stay upstream; estimator schema/leakage/calibration tests live here.

---

## § 0.2 Definitions

```text
checkpoint        a row in the estimator dataset, indexed by (run_id, t)
                  where t is either step index or wall-clock seconds.
prefix history    H_t = all ledger events with step <= t (or wall-time <= t)
                  in run run_id.
horizon           τ ∈ {steps, wall-clock seconds} forward from checkpoint t.
target            a label evaluated at H_{terminal} or events in (t, t+τ].
feature           any column derivable from H_t alone.
forbidden column  any column that contains H_{>t} information.
shape class       categorical tag describing trajectory pattern over a run
                  (e.g. high-progress failure, validation-induced reopen).
calibration       Brier, ECE, reliability. Required for every probability
                  output. Ranked-but-uncalibrated is not acceptable.
source            one of: swe_agent, hermes, tb_live (and future live).
```

---

## § 0.3 What we will not build

Even after live Terminal-Bench ledgers exist, these are **out of scope** for this repo:

```text
power-aware controller
task pausing / throttling policy
online RL or bandit controllers
model-effort modulation
automatic task termination
LLM-judge as primary estimator
large neural sequence estimators (deferred until § Workstream Q)
new ledger event types
new SubtaskCategory values
```

The immediate goal is **belief-state estimation** with honest calibration, not control. § Workstream P enforces a go/no-go gate before any control work elsewhere.

---

## § Workstream A — Repository scaffold and conventions

Everything else depends on this. Do these in order.

### A1. Initialize project layout
Status: done

Goal: Create the on-disk layout for the estimator project.

Outputs:
```text
coding-estimator/
├── README.md              (one-page: what this repo is, what it isn't)
├── AGENTS.md              (mirrors the upstream rules + § 0)
├── pyproject.toml         (uv-managed; sklearn, pandas, pyarrow, numpy,
                            scipy, lightgbm/xgboost optional, pytest, ruff)
├── uv.lock
├── coding_estimator/
│   ├── __init__.py
│   ├── ingest/            (Workstream C)
│   ├── checkpoints/       (Workstream D)
│   ├── labels/            (Workstream E)
│   ├── profile/           (Workstream F)
│   ├── baselines/         (Workstream G)
│   ├── splits/            (Workstream H)
│   ├── models/            (Workstream I)
│   ├── calibration/       (Workstream J)
│   ├── eval/              (Workstream H/J shared evaluation harness)
│   ├── leakage/           (Workstream D5 + § 0.4)
│   └── io.py              (parquet/csv writers, deterministic ordering)
├── scripts/               (CLIs that wrap library code)
├── docs/
├── datasets/              (gitignored except small reference manifests)
├── models/                (gitignored)
├── reports/               (versioned)
└── tests/
```

Acceptance:
```text
`uv run pytest -q` runs (even if zero tests).
`uv run python -c "import coding_estimator"` succeeds.
README.md states (a) what this is, (b) what it isn't, (c) one-line run
of the smoke pipeline once it exists.
.gitignore excludes datasets/*, models/*, reports/htmlcov/, .ipynb_checkpoints/.
```

### A2. Lock dependencies and tooling
Status: done

Goal: One toolchain. No surprise version drift between agent runs.

Outputs:
```text
pyproject.toml with pinned major versions:
  python = ">=3.11,<3.13"
  pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm,
  matplotlib, pytest, pytest-cov, ruff, jsonschema
.python-version
ruff config (line-length 100, target-version py311)
```

Acceptance:
```text
`uv sync` succeeds clean.
`uv run ruff check .` succeeds (rules: E, F, W, I, UP, B, SIM).
`uv run pytest -q` succeeds.
```

### A3. Define the upstream-artifact path resolver
Status: done

Goal: Single function that knows where the ledger lives, the runs live, and what an `ingest_root` looks like — so every script can be pointed at a sibling checkout, a CI mount, or a future fork.

Outputs:
```text
coding_estimator/ingest/paths.py
```

API:
```python
def ledger_root() -> Path                 # default: ../coding-progress-ledger
def runs_root(source: str) -> Path        # source ∈ {swe_agent, hermes, tb_live}
def run_dir(source: str, run_id: str) -> Path
def list_run_ids(source: str) -> list[str]
```

Behavior:
```text
- LEDGER_ROOT env var overrides default.
- Hard fail if ledger_root() does not exist or is missing `ledger_progress/`.
- list_run_ids is deterministic (sorted).
```

Acceptance:
```text
tests/test_paths.py passes against a fixture ledger_root.
Real ../coding-progress-ledger resolves without errors.
```

### A4. Reproducibility primitives
Status: done

Goal: All randomness is seeded, all I/O is byte-stable.

Outputs:
```text
coding_estimator/io.py:
  set_global_seed(seed: int)
  write_parquet(df, path) (sorts columns, sorts rows by stable key)
  write_csv(df, path) (utf-8, lf, no index, sorted columns)
  write_json(obj, path) (sort_keys=True, indent=2)
docs/REPRODUCIBILITY.md: one page on seeds, sort keys, and what not to
                        rely on (e.g. dict iteration order, set ordering).
```

Acceptance:
```text
tests/test_io_byte_stability.py: writing twice produces identical bytes.
tests/test_seeding.py: a tiny pipeline run twice with the same seed
produces identical predictions.
```

### A5. AGENTS.md
Status: done

Goal: Drop-in guidance for any subagent or human contributor.

Outputs:
```text
AGENTS.md:
  - Every change adds tests.
  - Hard fails over try/except.
  - Run `uv run pytest` after every change.
  - At end of each task, update TASKS.md status markers.
  - Project rules from § 0 are reproduced verbatim.
  - Pointer to upstream AGENTS.md for ledger-side rules.
```

### A6. Upstream artifact pinning
Status: done

Goal: Make upstream drift visible. Every dataset build pins the upstream commit and the upstream W3 table digest used.

Outputs:
```text
datasets/manifests/upstream_commit.json
coding_estimator/ingest/pinning.py
```

`upstream_commit.json` per build:
```text
{
  "ledger_repo_path": "../coding-progress-ledger",
  "ledger_commit_sha": "<git rev-parse HEAD upstream>",
  "ledger_dirty": false,
  "w3_table_path": "datasets/swe_agent_estimator_checkpoints.csv",
  "w3_table_sha256": "<hash>",
  "q_labels_table_path": "datasets/swe_agent_q_labels.csv",
  "q_labels_table_sha256": "<hash>",
  "captured_at": "<iso-8601>"
}
```

Acceptance:
```text
tests/test_upstream_pinning.py asserts that running the pipeline at
the same upstream SHA produces the same digests; running with an
upstream dirty tree fails fast unless --allow-dirty.
```

### A7. Data-budget snapshot
Status: done

Goal: One-page artifact, regenerated on every dataset build, showing per-(target, source) effective N.

Outputs:
```text
datasets/profiles/data_budget.md
coding_estimator/profile/budget.py
```

Per cell `(target, source, split_scheme)`:
```text
runs, checkpoints, positives, negatives, masked,
LORO-feasible (positives ≥ 5 AND negatives ≥ 5 in EACH fold),
LOSO-feasible (same on the held-out source).
```

Cells flagged as not-feasible are skipped by the evaluation harness (G7) and surface in every report as "n/a (insufficient data)" — never silently zero or NaN.

Acceptance:
```text
tests/test_data_budget.py loads a synthetic ledger of 5 runs and
asserts that 4 of 5 hand-crafted targets are flagged not-feasible.
```

---

## § Workstream B — Estimator interface freeze (Phase A of the plan)

Freeze schemas before writing any modeling code. This is the cheapest place to fix mistakes.

### B1. Checkpoint schema
Status: done

Goal: Specify exactly what one checkpoint row contains across all sources.

Outputs:
```text
docs/ESTIMATOR_INTERFACE.md
schemas/checkpoint_schema.json (jsonschema; loadable by tests)
```

Required keys (per § 4 of the plan):
```text
identity:        run_id, source, checkpoint_id, checkpoint_step,
                 checkpoint_event_index, checkpoint_wall_time,
                 checkpoint_elapsed_seconds, checkpoint_fraction_timeout,
                 is_terminal_checkpoint
features:        every column listed in B3, prefix-only at t
labels:          every column listed in B2 (pivoted to long-form table E2)
provenance:      ledger_path, schema_version, builder_commit_sha,
                 source_protocol_version
```

Acceptance:
```text
schemas/checkpoint_schema.json is valid jsonschema.
tests/test_checkpoint_schema.py instantiates one row per source and
validates against the schema.
ESTIMATOR_INTERFACE.md states which columns are required vs nullable
per source and which are always required across sources.
```

### B2. Targets and label schema
Status: done

Goal: Enumerate every prediction target with its definition, units, horizon, and provenance. **v0 ships four headline targets only**; everything else is deferred to B2.bis until N > 100 runs.

Outputs:
```text
docs/ESTIMATOR_TARGETS.md
schemas/label_schema.json
coding_estimator/labels/registry.py    (one Target dataclass per target)
```

#### v0 headline targets (must ship)

```text
1. y_success_eventual         binary, terminal
                              the trivial-but-useful target. Most likely
                              to be uninformative beyond elapsed-time;
                              its job is to anchor the no-regression gate.

2. y_future_progress_drop_h5  binary, horizon = 5 ledger steps
                              == upstream Q1 future_progress_drop.
                              Highest positive rate of the Q1 family
                              (~0.30 at N=20), best-supported v0 target.

3. y_validation_new_work_h5   binary, horizon = 5 steps
                              == upstream Q1 validation_exposes_new_work.
                              Low positive rate (~0.02) but tied to the
                              validation pillar — kept for diagnostic.

4. y_submit_without_validation binary, terminal, run-constant
                              == upstream Q1 submit_without_validation_state.
                              EXPLICITLY NOTED: this is a run-level
                              constant. Any high score is a data property,
                              not skill (see § 0.4 q_baselines context).
                              Reported as a calibration sanity target.
```

For each headline target the registry must record:
```text
name, family, horizon_units, horizon_value, definition, window_kind
("strict-future" | "terminal" | "regression"), source_signal,
mask_rule, upstream_q_target_id, run_constant_flag, base_rate_estimate.
```

Mask rules (mandatory):
```text
- y_*_h5 targets: mask checkpoints whose t + 5 > finish_step (label
  would equal terminal label trivially). is_terminal_checkpoint must
  also be masked for horizon labels.
- y_stuck_loop_h5: additionally mask rows where the stuck flag is
  already true at t (per upstream Q_TARGETS.md).
- y_success_by_h_seconds_*: mask if timestamp_quality != "real"
  for that run.
```

#### B2.bis Deferred targets (do not implement in v0)

These are listed only so they aren't lost. Re-evaluate when N > 100 runs:

```text
y_success_by_h_steps_{5,10,25,50}, y_success_by_timeout
y_success_by_h_seconds_{300,900,1800,runtimeout}
y_remaining_steps_if_success, y_remaining_seconds_if_success
y_finish_step, y_finish_seconds
y_product_reopen_h5, y_stuck_loop_h5
y_blocked_within_h5, y_new_scope_within_h5, y_validation_failure_within_h5
```

#### Shape labels are NOT prediction targets in v0

Upstream `label_observation_shapes.py` produces *post-hoc, run-level descriptors* of an entire trajectory. Predicting them at non-terminal `t` is essentially predicting `final_success` with extra steps and is a leakage hazard. In v0 they are used for:
```text
- profiling (F5, descriptive rollup)
- evaluation slicing (H4, per-shape AUROC/Brier)
- failure-mode case studies (O*)
NOT as prediction targets. The y_shape_* registry entries are deferred.
```

Acceptance:
```text
ESTIMATOR_TARGETS.md exists, one section per v0 target plus a "deferred"
appendix.
schemas/label_schema.json validates.
tests/test_targets_registry.py asserts: every v0 target has a definition,
window_kind, mask_rule, and a deterministic computation function
reference; every deferred target has its v0 status set.
```

### B3. Feature groups schema
Status: done

Goal: Enumerate every feature column, mapped to a feature group (per § 4 of the plan), with its data type, fill semantics, and per-source availability.

Outputs:
```text
docs/ESTIMATOR_FEATURE_GROUPS.md
schemas/feature_schema.json
coding_estimator/checkpoints/features/registry.py
```

Each entry in the registry must specify:
```text
column_name, dtype, group, prefix_only=True,
derivable_from_ledger ∈ {yes, no, requires_transcript, requires_source_trace},
populated_on ⊆ {swe_agent, hermes, tb_live},
upstream_source (which build_*.py already emits it, if anywhere — re-use!),
fill_when_missing (null vs 0 vs sentinel),
run_constant_flag (true if value cannot change across t within a run),
feature_or_label.
```

#### v0 feature groups (ledger-derivable only)

These are the only groups any v0 model trains on. **All columns are constructible from `ledger.jsonl` alone**, mirroring upstream `build_estimator_checkpoints.py` and extending it with a few prefix-only additions.

```text
frontier
  - active_leaf_count, active_coding_leaf_count, active_validation_leaf_count

closure  (re-used verbatim from upstream W3 schema)
  - completed_leaf_count, coding_progress, validation_progress
  - product_progress, investigation_progress       (from upstream)

discovery
  - num_adds_so_far, num_splits_so_far,
    denominator_growth_so_far, steps_since_new_subtask
  - new_leaf_count_last_{1,3,5}_steps              (new; computed locally)

instability
  - num_reopens_so_far, num_invalidations_so_far,
    num_deletes_so_far, largest_progress_drop_so_far,
    num_progress_drops_so_far, steps_since_last_drop

stalling (ledger-only)
  - blocked_leaf_count, blocked_coding_leaf_count,
    blocked_validation_leaf_count
  - steps_since_completion, steps_since_progress_increase,
    steps_since_status_change, steps_since_evidence
  - repeated_observation_loop_flag                  (from upstream W3)
  - no_progress_window_{5,10}                       (new; ledger-derived)

validation
  - validation_leaf_exists, validation_started, validation_complete,
    validation_failed, validation_blocked, validation_in_progress
  - num_validation_attempts, num_validation_failures, num_validation_successes
  - steps_since_last_validation
  - submit_without_validation_so_far               (prefix-only flag —
    NOT the terminal column. True iff at t no validation events seen.)

evidence (re-uses upstream rescore_suite_by_category.classify_evidence)
  - strong_completion_count, manual_only_completion_count,
    weak_product_completion_count
  - strong_evidence_fraction, manual_only_evidence_fraction
  - latest_completion_evidence_type

time_budget (step-based; wall-clock SUBSET only on tb_live)
  - elapsed_steps                                  (always)
  - elapsed_wall_time                              (tb_live only)
  - fraction_timeout_consumed, remaining_timeout_budget   (tb_live only)
  - completion_rate_recent_steps                   (always)
```

#### Source/task group (REPORT SEPARATELY, not used in headline G4 baseline)

Listed but used only in G6 to measure how much of any "win" comes from source identity:

```text
source_task
  - source, agent_scaffold, model_name
  - task_family_hash, repo_family_hash    (run-constant, leakage-prone)
  - initial_prompt_length, initial_files_count
```

#### Excluded from v0 (not constructible from ledger.jsonl alone)

These cannot be built without new ingestion adapters that read source traces or transcripts. The upstream ledger does not contain command-level events, tool calls, or tokens.

```text
elapsed_agent_turns, elapsed_tool_calls, elapsed_commands,
elapsed_tokens_if_available
repeated_command_count, repeated_observation_count,
same_error_loop_flag, two_command_oscillation_flag
average_wall_time_per_completion, tool_call_rate_recent
```

Adding any of the above requires a separate workstream that ingests `transcript.md` or `trajectory_steps.jsonl`. Not in scope for v0.

#### Stretch group (deferred to § Workstream Q)

```text
semantic / wall-clock-stalling-cross-source / cross-validation-features
```

Acceptance:
```text
ESTIMATOR_FEATURE_GROUPS.md is exhaustive.
schemas/feature_schema.json validates.
tests/test_feature_registry.py asserts every checkpoint column
appears in exactly one group, every column is marked prefix_only=True,
and no column overlaps with the forbidden column list (B4).
```

### B4. Forbidden columns
Status: done

Goal: Machine-readable list of columns that must never be features.

Outputs:
```text
schemas/forbidden_columns.json
coding_estimator/leakage/guard.py
```

Initial forbidden list:
```text
y_*                                   (all label columns)
label_*                               (legacy upstream label prefix)
final_success, final_success_source
finish_step, finish_seconds
verifier_exit_code, verifier_pass
test_output_*, eval_log_*
final_diff_*
summary_by_category_*
shape_label, shape_tags               (these are labels, not features)
checkpoint_event_index_at_terminal
final_artifact_without_validation     (terminal property; the prefix-only
                                       analogue lives in B3 as
                                       submit_without_validation_so_far)
```

Acceptance:
```text
schemas/forbidden_columns.json validates against jsonschema.
guard.assert_no_forbidden(df) raises on any forbidden column,
suffix-match included.
tests/test_forbidden_columns.py exercises the guard against synthetic
checkpoint frames containing label leakage.
```

### B4.5. Run-constant feature register
Status: done

Goal: Catalog every feature that is constant within a run. These are *not* forbidden, but they are dangerous: under loro on small N, a run-constant feature paired with a run-constant label is a perfect predictor that learns nothing transferable.

Outputs:
```text
schemas/run_constant_features.json
coding_estimator/leakage/run_constancy.py
```

Known run-constants from B3:
```text
source, agent_scaffold, model_name, task_family_hash, repo_family_hash,
initial_prompt_length, initial_files_count
```

Audit emitted at every dataset build (Workstream D): per (feature, target), assert that not BOTH are run-constant in the training fold. Fail-loud if so.

Acceptance:
```text
tests/test_run_constancy.py: a synthetic frame with `source` as a
feature and `y_submit_without_validation` as a label triggers the
audit.
```

### B5. Splits protocol
Status: done

Goal: Specify the canonical split schemes, with run-level disjointness as the invariant.

Outputs:
```text
docs/SPLITS_PROTOCOL.md
schemas/split_schema.json
coding_estimator/splits/protocol.py
```

Required schemes:
```text
loro                       leave-one-run-out (per-source)
ltfo                       leave-one-task-family-out (per-source)
loso                       leave-one-source-out (combined)
holdout                    fixed train/val/test by run, seed=0
temporal                   train on earliest k% of runs by start_time,
                           test on the rest (live sources only; warn if
                           timestamps are synthetic)
```

Acceptance:
```text
SPLITS_PROTOCOL.md states the invariant: no run_id appears in more than
one split partition.
tests/test_splits_disjoint.py asserts disjointness on every scheme
and every source.
schemas/split_schema.json validates.
```

### B6. Estimator output schema
Status: done

Goal: One row per (run_id, checkpoint_id, target_name, model_id) with the probability, optionally a regression value, calibration bucket, and uncertainty.

Outputs:
```text
docs/ESTIMATOR_OUTPUT.md
schemas/estimator_output_schema.json
```

Columns (v0 required):
```text
run_id, source, checkpoint_id, model_id, model_version,
target_name, target_family, target_horizon,
probability, prediction_kind ("binary" only in v0),
calibration_bucket, calibration_source,
estimator_commit_sha, schema_version
```

Reserved nullable columns (populated only when corresponding model
ladder un-defers):
```text
regression_value, regression_units      (I3 hazard / regression)
lower_ci, upper_ci                       (CI-emitting models)
top_feature_attributions (json)          (interpretability tooling)
```

Acceptance:
```text
schema validates.
tests/test_output_schema.py round-trips a synthetic prediction frame
through parquet and back.
```

### B7. Schema-only smoke test
Status: done

Goal: Wire up the entire schema layer end-to-end against an empty pipeline so later workstreams have a known-good frame template.

Outputs:
```text
tests/test_schema_pipeline_smoke.py
```

Behavior:
```text
- Builds a 1-row checkpoint frame matching B1.
- Builds a 1-row label frame matching B2.
- Builds a 1-row prediction frame matching B6.
- Validates each against jsonschema.
- Runs forbidden-column guard on the checkpoint frame.
- Runs split disjointness on a 2-run synthetic split.
```

---

## § Workstream C — Source ingestion and unification

Each source has a different on-disk shape and a different protocol generation. We expose them through a single `RunRecord` interface so downstream code is source-agnostic.

### C1. Source registry
Status: done

Goal: Central enumeration of all sources, their roots, and their schema-version policy.

Outputs:
```text
coding_estimator/ingest/sources.py
docs/SOURCES.md
```

Sources at v0:
```text
swe_agent_pilot           retrospective, 20 runs (10s/10f), step-only
swe_agent_pilot_v3        retrospective, revised protocol
swe_agent_live            live-via-replay, synthetic timestamps
swe_agent_live_wallclock  live-via-replay with timestamps backfilled
hermes_pilot              retrospective Hermes
hermes_pilot_h5           retrospective Hermes, larger v
hermes_pilot_h5_v2        retrospective Hermes, revised
tb_live                   live, real wall-clock, 12 runs (TB-12)
```

For each source, record:
```text
source_id, runs_dir (relative to ledger_root), default_split,
timestamp_quality (real|synthetic|none), label_field_path,
protocol_doc, schema_version, known_caveats,
canonical_for_v0 (true for ONE swe_agent variant and ONE hermes variant).
```

Canonical-source decisions (must be locked in SOURCES.md):
```text
- swe_agent canonical: swe_agent_pilot (20 runs, original protocol).
  swe_agent_pilot_v3 is reserved for parity-of-protocol comparisons
  in Workstream L; swe_agent_live is reserved for sidecar-feasibility
  testing only.
- hermes canonical: hermes_pilot_h5_v2 (most recent, revised
  annotation protocol).
- live canonical: tb_live (12 first-party runs).
- swe_agent_live_wallclock: do NOT mix into headline pools; its
  wall-clock is back-filled synthetic per upstream
  WORKSTREAM_N_TB_PLAN.md.
```

Annotation-leakage acknowledgment (mandatory in SOURCES.md):
```text
Retrospective sources (swe_agent_pilot, hermes_pilot*) were
annotated post-hoc with knowledge of the run's outcome. This means
event-categorization decisions may carry annotator-outcome
information that is unfixable at the estimator layer. Any model
trained or evaluated on retrospective sources inherits this leakage;
it must be named in every report and treated as an upper bound on
"realistic" performance, not a faithful estimate.
```

Acceptance:
```text
tests/test_sources_registry.py asserts every entry resolves under
ledger_root and contains at least one run; exactly one swe_agent
source and exactly one hermes source are flagged canonical_for_v0.
SOURCES.md links to upstream protocol docs and includes the
annotation-leakage note verbatim.
```

### C2. RunRecord and ledger reader
Status: done

Goal: A single immutable representation of one run that downstream code consumes.

Outputs:
```text
coding_estimator/ingest/run_record.py
```

Dataclass:
```text
RunRecord:
  run_id: str
  source: str
  ledger_path: Path
  events: list[dict]                    (sorted by step, then by file order)
  has_real_wallclock: bool
  start_wall_time: datetime | None
  end_wall_time: datetime | None
  task_id: str | None
  task_family: str | None
  agent_scaffold: str | None
  model_name: str | None
  raw_metadata: dict                    (whatever run_manifest.json had)
```

Behavior:
```text
- Hard fails if ledger.jsonl is missing or unreadable.
- Validates ledger event schema against ../coding-progress-ledger
  schema (re-uses upstream serialization).
- Records timestamp_quality from live_instrumentation.json when present.
- Does NOT load test_output.txt or final_diff.patch except for label
  resolution (label loader is a separate function).
```

Acceptance:
```text
tests/test_run_record.py loads one run per source and asserts events
are sorted, ledger_path exists, has_real_wallclock matches the
live_instrumentation.json claim.
```

### C3. Final-label loader
Status: done

Goal: Resolve `final_success` for every run, separately from feature construction. Re-use upstream `ledger_progress.run_manager.resolve_final_success` where possible.

Outputs:
```text
coding_estimator/ingest/labels.py
```

API:
```python
def load_final_label(run: RunRecord) -> FinalLabel
```

Returned struct:
```text
final_success: bool | None
final_success_source: "verifier_exit" | "swe_agent_target" |
                      "hermes_resolved" | "manual" | "missing"
finish_step: int | None
finish_seconds: float | None          (wall-clock if available)
timeout: bool
termination_reason: str | None
```

Acceptance:
```text
tests/test_label_loader.py covers each source.
Hard fails for runs whose label cannot be resolved (caller must
explicitly skip them).
Final labels are NEVER joined into the feature-only frame.
```

### C4. Per-source ingestion adapters
Status: done

Per-source code lives under `coding_estimator/ingest/<source>.py`. Each adapter:

```text
- Lists run_ids deterministically.
- Loads RunRecord per run_id.
- Records metadata caveats (timestamp_quality, missing fields).
- Emits a per-source manifest at datasets/manifests/<source>.csv.
```

Sub-tasks (canonical sources only — non-canonical variants are deferred to
Workstream L for parity comparisons and are not built in v0):
```text
C4a. swe_agent_pilot
C4b. hermes_pilot_h5_v2
C4c. tb_live (12 runs)
```

Acceptance:
```text
tests/test_ingest_<source>.py loads ≥ 1 run per source and asserts
the manifest covers every run found on disk.
A missing or malformed run is reported (not silently dropped).
```

### C5. Combined manifest
Status: done

Goal: One manifest covering all sources for downstream profiling.

Outputs:
```text
datasets/manifests/all_runs.csv
```

Columns:
```text
run_id, source, ledger_path, ledger_event_count,
has_real_wallclock, start_wall_time, end_wall_time, task_id,
task_family, agent_scaffold, model_name, final_success,
final_success_source, timeout, finish_step, finish_seconds,
notes.
```

Acceptance:
```text
Manifest has ≥ N_runs rows where N_runs = sum over sources.
tests/test_combined_manifest.py asserts byte-stability and validates
final_success_source ∈ allowed enum.
```

---

## § Workstream D — Checkpoint dataset construction

This is the heart of the estimator's input pipeline. The plan was tightened
after a senior critique: **the checkpoint builder silently determines the
scientific validity of everything downstream**. We treat D as a
measurement-system validation project — the winning v0 outcome is not a
trained classifier but a prefix-only checkpoint dataset that provably does
not leak future state.

The locked ordering is:

```text
D0  Golden semantic fixture           (executable definition of feature semantics)
D1  Checkpoint policy
D2  Prefix replay engine              (validated against D0 + future-mutation tests)
D2.5 Audit skeleton                   (structural before features land)
D3  Feature builders                  (each tested against D0)
D4  Build CLI
D5  Checkpoint construction audit gate (blocks Workstream G)
```

### D0. Golden semantic fixture
Status: done

Goal: A hand-authored ledger plus hand-authored expected checkpoint states.
Built BEFORE the prefix replay engine so the engine cannot become the de
facto semantics by accident.

Outputs:
```text
tests/fixtures/golden_run/ledger.jsonl
tests/fixtures/golden_run/expected_checkpoints.json
tests/fixtures/golden_run/README.md
tests/test_golden_fixture.py
```

The fixture must include at least one event of each kind:
```text
init, add_subtask, update_status (in_progress / complete / blocked),
add_evidence, split_subtask, reopen_subtask, invalidate_subtask,
validation pass (validation leaf -> complete),
validation fail (validation leaf -> invalidate / blocked),
a strict progress drop (reopen of a previously-completed leaf),
a "future" event past the chosen mid-step `t_mid` whose presence/absence
must NOT affect any feature value at `t <= t_mid`.
```

For the fixture, expected checkpoint states are authored by hand for at
least these aggregates at every step: `active_leaf_count`,
`completed_leaf_count`, `num_adds_so_far`, `num_splits_so_far`,
`num_reopens_so_far`, `num_invalidations_so_far`,
`num_progress_drops_so_far`, `largest_progress_drop_so_far`. The full
feature set lands as builders come online (D3), each pinned to the same
fixture.

Acceptance:
```text
tests/test_golden_fixture.py:
- every required event_type appears in ledger.jsonl
- the fixture parses through upstream load_events_jsonl
- expected_checkpoints.json has one entry per step in the ledger
- the future-mutation invariant fixture (a paired ledger that diverges
  past t_mid) is byte-identical to the canonical ledger up to t_mid
```

### D1. Checkpoint policy
Status: done

Goal: Decide where in each run a checkpoint row exists.

Outputs:
```text
docs/CHECKPOINT_POLICY.md
```

v0 ships only `P_step` (one checkpoint per ledger step), for parity with
upstream `build_estimator_checkpoints.py`. The policy enum (`P_event`,
`P_kstep`, `P_wallclock_grid`, `P_terminal_only`) is documented for the
record but not implemented; reintroduce when there's a concrete reason
to compare fidelity.

Acceptance:
```text
CHECKPOINT_POLICY.md documents the enum and the default.
tests/test_checkpoint_policy.py asserts P_step produces a strictly
increasing checkpoint index on a synthetic run.
```

### D2. Prefix replay engine
Status: done

Goal: Given a RunRecord and a checkpoint t, return the replayed ledger state and an `events_so_far` list whose every entry has step ≤ t.

Outputs:
```text
coding_estimator/checkpoints/replay.py
```

API:
```python
def prefix_replay(run: RunRecord, t_step: int) -> ReplayState
def prefix_replay_at_event(run: RunRecord, event_index: int) -> ReplayState
def prefix_replay_at_wallclock(run: RunRecord, t_seconds: float) -> ReplayState
```

ReplayState:
```text
score_obs (from upstream `score(ledger, categories=CODING_CATEGORIES)`)
ledger state at t (subtask map, status map, evidence map)
events_so_far list, prefix-only invariant assertable
```

Behavior:
```text
- Re-uses upstream `replay()` and `score()` to avoid drift.
- Hard fails if any event with step > t_step leaks into events_so_far.
```

Acceptance:
```text
tests/test_prefix_replay.py asserts no future leakage on a fixture
run with steps [0..10] queried at t=5.
Cross-checks `coding_progress` at terminal step against upstream
build_estimator_checkpoints output (parity test).
```

### D3. Feature extractors
Status: done

Goal: One module per feature group from B3. Each module is a pure function `(ReplayState, RunRecord, t) → dict[str, Any]`.

Sub-tasks (one feature module each):
```text
D3a frontier
D3b closure                     (re-uses upstream score())
D3c discovery
D3d instability
D3e stalling                    (includes wall-clock variants when available)
D3f validation
D3g evidence                    (re-uses scripts/rescore_suite_by_category.classify_evidence
                                 helper from upstream — copy a snapshot,
                                 do not mutate upstream)
D3h time_budget
```

`source_task` features are emitted by C5's combined manifest already; no
v0 feature module needed because no v0 baseline (G1/G2/G4) consumes them.
Reintroduce a D3i module if/when G6 is un-deferred.

Each module ships:
```text
coding_estimator/checkpoints/features/<group>.py
tests/test_features_<group>.py
```

Each test asserts:
```text
- Features are computed from prefix only (no peek-ahead).
- For one fixture run with hand-computed expected values, every column
  matches.
- Source-disjoint features (e.g. wall-clock features on retrospective
  rows) emit `null`, not 0.
```

### D4. Build CLI
Status: done

Goal: Single command to build per-source and combined checkpoint datasets.

Outputs:
```text
scripts/build_checkpoints.py
```

Usage:
```text
uv run python scripts/build_checkpoints.py \
    --source tb_live \
    --policy P_step \
    --out datasets/checkpoints_tb_live.parquet
```

Outputs:
```text
datasets/checkpoints_<source>.parquet
datasets/checkpoints_<source>_summary.md
datasets/checkpoints_all.parquet           (concatenation)
```

Acceptance:
```text
Running on each source produces the expected number of rows
(per-run counts agree with upstream W3 numbers within ±1 for swe_agent).
Running twice byte-stable (parquet has stable encoding).
tests/test_build_checkpoints.py covers a 2-run synthetic source.
```

### D5. Leakage gate
Status: done

Goal: Every output of D4 passes a structural and behavioral leakage audit.

Outputs:
```text
coding_estimator/leakage/audit.py
scripts/audit_checkpoints.py
datasets/audits/checkpoints_<source>_audit.md
```

Audits:
```text
structural:  all columns are in feature_schema or are identity columns.
             no column overlaps forbidden_columns.json.
behavioral:  for each run, recompute features at t = mid_step using a
             ledger truncated to step ≤ mid_step. Assert frame equality
             with the row built from the full ledger. This catches
             "I accidentally used events[t+1] in my feature" bugs.
shuffle test: shuffle each future-only label column and confirm
             feature columns do not change.
```

Acceptance:
```text
Audit emits zero structural failures.
Behavioral audit: zero diff on a 5-run sample.
tests/test_leakage_audit.py exercises both audits on a synthetic
ledger that injects a future event into one feature module.
```

### D5.5. Within-run constancy audit
Status: done

Goal: Catch the failure mode where a constant-within-run feature pairs with a constant-within-run label and yields a perfect-but-meaningless predictor.

Outputs:
```text
coding_estimator/leakage/run_constancy.py        (consumed by D5)
datasets/audits/run_constancy_<source>.md
```

Algorithm:
```text
For each (feature, target) pair:
  - feature_constant_in_run[r]  = (feature.std_within(r) == 0)
  - target_constant_in_run[r]   = (target.std_within(r) == 0)
  - if (mean over r of feature_constant_in_run) > 0.99 AND
       (mean over r of target_constant_in_run) > 0.99:
    flag as joint run-constant.
```

Joint run-constant pairs are forbidden in v0 modeling — the harness skips them and the audit lists them with a recommended fix (e.g. "drop feature `source` when modeling `y_submit_without_validation`").

Acceptance:
```text
tests/test_run_constancy_audit.py covers a synthetic frame where
`source` is the only feature and `y_submit_without_validation` is
the only label; audit must flag the pair.
```

### D6. Removed
Folded into F2 (checkpoint-distribution profile). The per-source rollup
emitted by D4 already covers run/checkpoint counts; deeper profiling
lives in Workstream F so leakage and label profiling stay co-located.

---

## § Workstream E — Label construction

### E1. Terminal labels
Status: done

Outputs:
```text
coding_estimator/labels/terminal.py
```

Targets: `y_success_eventual`, `y_finish_step`, `y_finish_seconds`,
`y_timeout`, `y_submit_without_validation`.

Tests:
```text
tests/test_terminal_labels.py
```

Cross-check against upstream `label_final_success` columns for parity.

### E2. Horizon labels (steps)
Status: deferred (per § B2.bis). Schema slot reserved in
`labels/registry.py`; no implementation in v0.

### E3. Horizon labels (wall-clock)
Status: deferred (per § B2.bis). Wall-clock-conditioned targets are not
in the v0 headline set; tb_live is the only source that could populate
them and N=12 is too thin.

### E4. Process-dynamics labels
Status: done

v0 targets only: `y_future_progress_drop_h5`, `y_validation_new_work_h5`.
Re-use upstream `scripts/build_q_labels.py` definitions; do not
re-derive.

Other Q1 targets (`y_product_reopen_h5`, `y_stuck_loop_h5`,
`y_blocked_within_h5`, `y_new_scope_within_h5`,
`y_validation_failure_within_h5`) are deferred per § B2.bis — schema
slots reserved, no v0 implementation.

Tests:
```text
tests/test_dynamics_labels.py asserts parity with upstream Q1 outputs
for the two v0 targets on the swe_agent_pilot frame.
```

### E5. Survival / regression labels
Status: deferred (per § B2.bis). Conditional-regression targets are not
v0; reintroduce when N > 100 runs.

### E6. Shape labels (slicing only)
Status: done

Shape labels are NOT prediction targets in v0 (per § B2). They are
attached to runs purely for evaluation slicing (H4) and case studies.
Re-use upstream `scripts/label_observation_shapes.py` directly; do not
re-derive.

Outputs:
```text
coding_estimator/labels/shapes.py    (thin wrapper around upstream)
datasets/shapes_<source>.parquet     (run_id → shape tags)
```

Tests:
```text
tests/test_shape_labels.py asserts shape labels agree with upstream
`*_shape_labels.csv` outputs on shared sources.
```

### E7. Combined label table
Status: done

Goal: Long-form table `(run_id, checkpoint_id, target_name, target_value, target_horizon, target_units, label_available, label_source)`.

Outputs:
```text
datasets/labels_<source>.parquet
datasets/labels_all.parquet
```

Tests:
```text
tests/test_label_table.py asserts: every label column from B2 is
present, no row has both target_value=null and label_available=true.
```

### E8. Label balance audit
Status: done

Outputs:
```text
datasets/profiles/labels_<source>_balance.md
```

Per target:
```text
positives, negatives, masked, positive rate by source, by progress
bucket, by elapsed-fraction bucket, by shape class.
```

Flag any (target, source) cell where positives < 5 or negatives < 5 — those targets cannot be trained on that source alone.

---

## § Workstream F — Profiling and distribution analysis

This phase **must** complete before any modeling. It is the cheapest insurance against training on broken data.

### F1. Source-level profile
Status: done

Outputs: `datasets/profiles/sources.md` covering every metric in plan § 6.1.

### F2. Checkpoint-distribution profile
Status: done

Outputs: `datasets/profiles/checkpoints_distribution.md` covering plan § 6.2 (progress histogram, elapsed fraction, leaf counts, validation/blocked state distributions).

### F3. Label-balance profile
Status: done

Outputs: `datasets/profiles/labels_balance.md`. Aggregates E8 into a single artifact.

### F4. Leakage profile
Status: done

Outputs:
```text
datasets/profiles/leakage_audit.md
schemas/forbidden_columns.json     (kept up to date)
```

Per feature:
```text
available_at_checkpoint, derived_only_from_prefix, contains_final_success,
contains_eval, contains_post_run_artifact, contains_finish_time,
constant_or_near_constant, high_cardinality_id.
```

### F5. Descriptive rollups
Status: not started

Single optional artifact `datasets/profiles/descriptive.md` that
covers: cross-source feature parity (mean/std/KS), shape-class
distribution, phase analysis (early/middle/late), non-monotonicity
counts, validation timelines, and tb_live stuck-loop windows.

These are descriptive, not gating. F11 does not depend on F5. Useful
input to model cards (Workstream N) and case studies (Workstream O).

Note: the original F5/F6/F7/F8/F9/F10 split was intentionally collapsed
— at N≈80 total runs and N=12 first-party live, six independent profile
docs is more reading than the data supports.

### F11. Profiling go/no-go gate
Status: done

Goal: Single rollup that says "the data is ready to train on" or "no, fix X first".

Outputs: `reports/F_profiling_go_no_go.md`.

Gate criteria:
```text
- ≥ 1 source has ≥ 5 successes and ≥ 5 failures for terminal labels.
- ≥ 1 source has ≥ 50 checkpoints with valid wall-clock features.
- No feature has > 95% missingness across all sources combined.
- No forbidden column appears in feature schema.
- Cross-source KS for at least the closure features is < 0.5.
```

If any criterion fails, halt at this gate and fix upstream.

---

## § Workstream G — Baseline ladder

Each baseline trains on the combined checkpoint table and is evaluated under split schemes from B5. **v0 ships only G1, G2, G4** under loro per source plus loso to tb_live; G3, G5, G6 are kept as deferred for parity reporting only.

### G1. Constant baseline (v0 mandatory)
Status: done

```text
predict training-set positive rate for each target.
```

Outputs: `coding_estimator/baselines/constant.py`, `reports/baseline_constant.md`.

### G2. Time-only baseline (v0 mandatory)
Status: done

Features: `elapsed_steps`, plus `elapsed_wall_time` and `fraction_timeout_consumed` only on tb_live where populated.

Tests: parity with upstream `q_baselines.py::elapsed_only` on swe_agent canonical source.

### G3. Progress-only baseline
Status: deferred — diagnostic only; not on v0 critical path.
Features: `coding_progress`, `validation_progress`, `product_progress`.

### G4. Ledger-basic baseline (v0 mandatory — the headline)
Status: done

Features: closure + frontier + instability + discovery (no dynamics, no semantics, no source).
This is the v0 headline model. Its no-regression vs G2 is the gate (§ Workstream P).

### G5. Ledger-dynamics baseline
Status: deferred until N > 100 runs. Features: G4 + stalling +
validation + evidence.

### G6. Source/task baseline
Status: deferred. Used only by O5 (source-leakage diagnostic), which is
itself deferred. Not consumed by P1.

### G7. Evaluation harness (v0 mandatory)
Status: done (v0 binary targets only; continuous regression deferred)

Outputs:
```text
coding_estimator/eval/harness.py
coding_estimator/eval/bootstrap.py
scripts/run_baselines.py
reports/baseline_results.md
reports/baseline_calibration.md
```

Reported metrics per (target, model, split scheme, source slice):
```text
AUROC (if both classes present), Brier, log-loss, ECE,
positive rate (data), predicted positive rate (model),
bootstrap 95% CI on Brier (run-level resampling, B=1000),
n_runs_train, n_runs_test, n_checkpoints_test.
```

Resampling rule (mandatory):
```text
Bootstrap resamples are taken at the RUN level, not checkpoint
level. Checkpoint-level resampling inflates power because rows
within a run are highly correlated. Any report that uses
checkpoint-level CIs is wrong and must be regenerated.
```

Acceptance:
```text
G1, G2, G4 run end-to-end.
Harness emits one row per (target, model, split, source slice) with
the metrics above; cells flagged not-feasible by A7 are emitted as
"n/a (insufficient data)".
G2 parity with upstream q_baselines elapsed_only within ±1e-6 Brier
on the swe_agent canonical source.
```

### G8. Human ledger baseline (v0 strongly recommended)
Status: not started

Goal: One person looks at a held-out tb_live ledger for ≤ 5 minutes and predicts the v0 headline targets by eye. This is the ceiling-baseline for "is the channel readable at all by something simpler than a model".

Outputs:
```text
reports/human_baseline.md
datasets/human_baseline_predictions.csv
```

Procedure:
```text
- Pick 6 of 12 tb_live runs at random (seed=0).
- Show the human only events with step ≤ midpoint.
- Human predicts P(success_eventual) and P(progress_drop_h5).
- Compare to G2 and G4 by Brier and ECE.
```

Acceptance: a single human predictor logs at least one prediction per held-out run. If the human cannot beat G2, the channel is not human-readable at midpoint; this is a finding, not a failure.

---

## § Workstream H — Splits and evaluation protocol

### H1. Split builder
Status: done

Outputs: `coding_estimator/splits/builder.py`, `datasets/splits/<scheme>_<source>.json`.

For each scheme in B5, emit a JSON file:
```text
{ "scheme": "loro", "source": "tb_live", "folds": [
  {"name": "fold_0", "train_run_ids": [...], "test_run_ids": ["markdown-to-html-cli"]},
  ...
]}
```

Tests: H1 ⇒ B5 disjointness invariant.

### H2. Within-source evaluation
Status: done

Run v0 baselines (G1, G2, G4) under `loro` and `ltfo` per source; emit per-source result tables.

### H3. Cross-source evaluation
Status: done

Run v0 baselines (G1, G2, G4) under `loso`. Headline cross-source test: train on `swe_agent_pilot` + `hermes_pilot_h5_v2`, test on `tb_live`.

### H4. Slice-specific evaluation
Status: done

Slice every test fold by (a) phase (early/middle/late) and (b) shape
class. Report per-slice AUROC, Brier, ECE. Slices with < 5 positives
or < 5 negatives in the test fold are emitted as "n/a (insufficient
data)" rather than computed.

### H6. Reporting templates
Status: done

Outputs: `reports/templates/eval_report.md.j2` (jinja), used by every model run.

Next:
- Wire upstream producer to populate `fraction_timeout_consumed` and
  fully-fill `steps_since_*` counters at canonical fill semantics so
  the eval-side `_apply_canonical_fills` shim can be retired.
- I0 / I1 (model ladder) and J1+ (calibration metrics) now unblocked.

---

## § Workstream I — Model ladder

### I0. Empirical-bin model
Status: done

Goal: Smoke test labels and splits.

```text
bin by progress quartile × elapsed-fraction quartile, predict
empirical positive rate on training fold, evaluate on test fold.
```

### I1. Logistic regression
Status: done

Outputs:
```text
coding_estimator/models/logreg.py
models/logreg_v0/{model.pkl, model_card.md, calibration.json}
```

For every v0 headline target in B2:
```text
fit on G4 features under the holdout split, plus loro for
within-source numbers. (G5/G6 feature groups stay deferred.)
```

Tests:
```text
tests/test_logreg.py asserts feature-importance signs are
plausible (e.g. `validation_complete` raises P(success), not
lowers it).
```

Hardenings (post sonnet-critic + research-test-creator pass):
- `predict_proba` clips to `OUTPUT_CLIP=(0.001, 0.999)` on both models
  so the pickled artifact behaves identically to the in-eval pipeline.
- Empirical-bin: `checkpoint_fraction_timeout` short-circuits the
  elapsed signal when fully finite (no priority-inversion ValueError);
  NaN `coding_progress` at predict time hard-fails per AGENTS.md policy.
- `evaluate_model_cell` annotates `EvalCell.note`:
  `run_constant_target` for run-constant V0 headline targets (driven
  by `V0_TARGETS[name].run_constant_flag`); a `single_class` substring
  whenever any train fold collapses to a single class.
- `save_model_bundle` keeps `calibration.json["targets"]` a strict
  subset of the pickled model dict and propagates `cell.note`.
- `calibration_source = "constant"` iff some holdout cell carries a
  `single_class` note; infeasibility notes (e.g. `no joined rows`) do
  not flip it. `render_model_card` tolerates infeasible cells.
- Test coverage extended from 4 → 20 cases, including clipping
  invariants, elapsed-signal priority, NaN hard-fail, registry-driven
  run-constant flag, single-class fold annotation, calibration.json /
  model.pkl alignment, and a single-driver coefficient-dominance check
  to guard against column-shift bugs.

### I2. Calibrated GBM (deferred until N > 200)
Status: deferred

Reason: 80–100 runs total, 12 first-party live; isotonic calibration on a held-out fold of 12 will overfit. Re-evaluate when the dataset grows.

Skeleton when un-deferred:
```text
coding_estimator/models/gbm.py
models/gbm_v0/...
isotonic on validation fold; report 3-bin ECE.
```

### I3. Discrete-time hazard model (deferred until live runs > 50)
Status: deferred

Reason: median tb_live run length is ~13–30 events; hazard-rate estimates per t-bucket are unstable past t=5. The model is correct in form but will not have stable per-t intercepts at this N.

Skeleton when un-deferred:
```text
coding_estimator/models/hazard.py
models/hazard_v0/...
For every (run, t) row, define h_t = P(finish at t+1 | survived to t, H_t).
Single logistic regression over (features, t_index) with per-t
intercepts; cap t-index at 5 if median run length stays small.
```

### I4. Sequence model — DEFERRED
Status: deferred

Defer until I1+I2 stable, calibration acceptable, and dataset has ≥ 100 runs total.

### I5. Semantic model — DEFERRED
Status: deferred

Defer until ledger-only ladder is calibrated; semantic features are bonus, not core.

---

## § Workstream J — Calibration

### J1. Calibration metrics
Status: shipped

Outputs: `coding_estimator/calibration/metrics.py`.

API: `brier(y, p)`, `expected_calibration_error(y, p, n_bins)`, `reliability_table(y, p, n_bins)`.

### J2. Reliability diagrams
Status: shipped

Markdown reliability per (model, source) under
`reports/calibration/calibration_<model>_<source>.md`. Driver:
`scripts/run_calibration.py`. PNG rendering remains deferred.

### J3. Calibration slices
Status: shipped

Output: `reports/calibration/calibration_slices.md`. Slice axes: source,
target_horizon, phase (early/middle/late), shape class, progress
bucket, validation state.

### J4. Recalibration methods
Status: shipped

Outputs: `coding_estimator/calibration/recalibrate.py` —
`PlattRecalibrator` (unregularized logit MLE), `IsotonicRecalibrator`,
`SourceIsotonicRecalibrator` (per-source + global fallback).
`kfold_recalibrated_predictions` in `calibration/report.py` does
honest run-level k-fold so the recalibrator never trains on its own
test row's run.

### J5. Calibration report
Status: shipped

Output: `reports/calibration/calibration_v0.md`. Gate: any (model,
source, target) with `ECE > 0.1` after isotonic recalibration is
flagged `not_safe_for_control` in the rollup. Annotation propagates
into model cards via the existing `render_model_card` machinery.

---

## § Workstream K — Live Terminal-Bench-specific evaluation

### K1. TB-only checkpoint evaluation
Status: shipped

Outputs: `reports/tb_live/tb_live_eval.md`,
`reports/tb_live/tb_live_metrics.csv`. Driver:
`scripts/run_tb_live_eval.py`. Per-target metrics for G2 and G4 on
`tb_live` under LORO with run-level bootstrap CIs and `ECE_3bin`
(10-bin unestimable at N=12). Single-class y is annotated.

### K2. TB online-feasibility test
Status: deferred (depends on Workstream M, which is itself deferred).
The D5 behavioral leakage audit covers v0's leakage gate — see P1.g.

### K3. TB qualitative rollup
Status: shipped

Output: `reports/tb_live/tb_live_qualitative.md`. One artifact across
the TB-12 cohort: phase/shape distribution, first stuck-loop precursor
checkpoint (`no_progress_window_5 >= 5`), repeated-loop flag,
validation timeline (first attempt / first success / first failure),
and max prediction-jump step (using G4 predictions on
`y_success_eventual`).

---

## § Workstream L — Live vs retrospective parity

### L1. Per-source retrospective re-annotation
Status: blocked on upstream NTB6
(`../coding-progress-ledger/docs/WORKSTREAM_N_TB_PLAN.md` § NTB6).
Un-block when NTB6 ships ≥ 6 of 12 paired re-annotations.

### L2. Event-observability matrix
Status: deferred — descriptive doc, not gating. Reintroduce when L1
unblocks.

### L3. Retrospective-vs-live model transfer (v0 headline)
Status: shipped

Output: `reports/retro_to_live_transfer.md`. Driver:
`scripts/run_retro_to_live.py`. Trains G4 on `swe_agent_pilot ∪
hermes_pilot_h5_v2` and evaluates on `tb_live`, with per-group
ablation (`g4_minus_{closure,frontier,instability,discovery}`) and a
G2 reference. Annotation-leakage caveat from § C1 stamped on the report.

### L4. Source-specific calibration
Status: deferred — at N=12, per-source isotonic on tb_live overfits.
Reintroduce when live N > 50.

---

## § Workstream M — Online inference (DEFERRED until P passes)

Status: deferred.

The upstream `ledger_progress.sidecar` already provides streaming primitives. Building a parallel streaming surface in this repo before any model has cleared the no-regression gate is premature scaffolding.

When un-deferred (post-P1), build:

```text
M1. coding_estimator/online/streamer.py — OnlinePredictor.
M2. scripts/monitor_run.py — pretty-print live ledger predictions.
M3. Online-vs-offline parity test (every tb_live checkpoint).
```

Until then, the offline pipeline (predict_at via prefix-replay over a frozen ledger.jsonl) is the only supported inference path. § L3's "transfer to live" evaluation runs purely offline against the existing tb_live ledgers.

---

## § Workstream N — Estimator artifacts and model cards

### N1. Model card schema
Status: shipped

Outputs: `docs/MODEL_CARD_TEMPLATE.md`,
`schemas/model_card_schema.json`. Schema enforces every plan § 15.5
field plus `failure_mode_results` and an optional `go_no_go_gate`
pointer. `coding_estimator/models/cards.py::validate_card` is the
runtime guard.

### N2. Model artifact bundle
Status: shipped

Outputs: `models/<estimator_id>/{model.pkl, calibration.json,
feature_schema.json, target_schema.json, model_card.md,
model_card.json, training_report.md}`. `save_model_bundle` validates
`model_card.json` against the schema before writing.
Acceptance: `tests/test_model_card.py` (34 tests; rejects every
required-field omission, enforces estimator_id and commit_sha regexes,
round-trips written sidecars).

### N3. Versioning policy
Status: shipped

Output: `docs/VERSIONING.md`. Format
`<model_family>_v<major>.<minor>[_<source-slice>]`; immutability rules;
sign-off requires N1 schema + Workstream O results + Workstream P
verdict on the card. Pre-v1 estimators carry
`not_safe_for_control = true` regardless of test results.

---

## § Workstream O — Failure-mode tests

These are *adversarial* tests that the estimator must survive before consideration for any downstream consumer.

### O1. Progress-overconfidence test
Status: shipped — **PASS** (median 0.578)

`evaluate_o1` slices on (final_success == 0) AND (per-row
coding_progress ≥ 0.8). On the v0 G4 LORO predictions across all
sources, median predicted P(success) on the slice is 0.578 < 0.7
threshold.

### O2. Late-discovery surprise test
Status: deferred — slice has < 5 runs at current N. Reintroduce when N
grows.

### O3. Stuck-loop ambiguity test
Status: deferred — recovery vs terminal-failure slices each have < 3
runs at current N.

### O4. Validation-shock test
Status: deferred — slice has < 3 runs at current N.

### O5. Source-leakage test
Status: shipped (vacuous on v0)

`coding_estimator/eval/failure_modes.py::evaluate_o5` compares G4
against G4 + numeric source_task columns under LOSO → tb_live. v0
checkpoints don't carry the source_task numeric columns yet, so the
comparison is identity and the result is `indeterminate`. Re-runnable
once the columns are built into the checkpoint frame.

### O6. Retrospective/live mismatch test
Status: deferred — overlaps L3 (retro→live transfer with feature-group
ablation). Reintroduce only if L3 surfaces feature-level findings that
warrant per-feature LOFO.

### O7. Timeout-bias test
Status: shipped — **FAIL on swe_agent_pilot (v0)**

`evaluate_o7` per-source LORO comparison of G4 vs G2. On the largest
canonical source (`swe_agent_pilot`, ~600 checkpoints), Brier delta is
**−0.009** — G4 is *worse* than G2, well under the +0.02 gate. tb_live
is `indeterminate` (single-class y on `y_success_eventual`). Output:
`reports/failure_modes/failure_modes.md`. This is the strongest
scientific finding from O — adding the deferred dynamics group (G5) is
the cheapest next experiment.

---

## § Workstream P — Go/no-go gate (Phase K of the plan)

The decision point that determines whether the estimator is ready to be consumed by anything beyond an evaluation harness. Given § 0.0 (data-budget), this gate is intentionally a **no-regression** gate, not a "wins" gate. The original aspirational gate is preserved as P-future (below).

### P1. v0 no-regression gate
Status: shipped — **verdict: INDETERMINATE** at v0

Outputs: `reports/ESTIMATOR_GO_NO_GO.md`,
`reports/ESTIMATOR_GO_NO_GO.json`. Driver:
`scripts/run_go_no_go.py`. Module: `coding_estimator/eval/go_no_go.py`.
v0 verdict at current data:
- P1.a PASS (G4 wins or ties G2 on 6/8 cells)
- P1.b INDETERMINATE (single-class y on tb_live)
- P1.c INDETERMINATE (hermes labels missing — combined retrospective
  not testable as plan defines it)
- P1.d INDETERMINATE (single-class y on tb_live)
- P1.e PASS (no forbidden columns; prefix/suffix/exact all checked)
- P1.f PASS (zero run-constant feature/target pairs in G4 folds)
- P1.g INDETERMINATE (D5 audit not provided; bare `clean: true`
  rejected — must include schema_version, n_runs_audited,
  n_checkpoints_audited, findings)
- P1.h PASS (winners span multiple targets)
Overall: indeterminate, gate cannot be cleared until tb_live
collects failures, hermes labels ship, and D5 audit lands.

Pass conditions (all must hold):
```text
P1.a  G4 (ledger-basic) point-estimate Brier ≥ G2 (time-only) on at
      least one v0 headline target on at least one canonical source
      under LORO. ("Wins or ties on point estimate" — CIs are too
      wide at N=12 to demand exclusion.)
P1.b  ECE_3bin (3-bin coarse) does NOT increase by > 0.05 going from
      G2 to G4 on tb_live under loro after isotonic recalibration.
      (10-bin ECE is unestimable at N=12.)
P1.c  Combined-retrospective headline test: on
      swe_agent_pilot ∪ hermes_pilot_h5_v2 (~50 runs, ~400+ checkpoints)
      under loro, G4 beats G2 on `y_success_eventual` Brier with
      run-level bootstrap 95% CI excluding zero. This is the
      *one* CI-exclusion result we ask for, on the largest available
      retrospective pool — but the result is reported with the
      annotation-leakage caveat from § C1 stamped on it.
P1.d  loso to tb_live: point-estimate Brier on tb_live under loso
      ≤ within-source loro Brier on tb_live + 0.05 absolute. (No CI
      requirement — sample is too small.)
P1.e  Forbidden-column audit: zero hits.
P1.f  Run-constancy audit (D5.5): zero joint run-constant
      (feature, target) pairs in any G4 training fold.
P1.g  Online-vs-offline parity: not required for v0 (Workstream M is
      deferred). Replaced by D5 behavioral leakage audit passing.
P1.h  Submit-without-validation caveat: if the headline win is
      driven by `y_submit_without_validation`, the report must
      explicitly state that this is a data property, not skill, per
      upstream q_baselines § "Reading these numbers".
```

### P-future. Aspirational gate (preserved for when N grows)
Status: deferred

When N > 200 runs total or live N > 50:
```text
P-future.a  G4 beats G2 with run-level bootstrap 95% CI excluding zero
            on at least one headline target on tb_live under loro.
P-future.b  10-bin ECE ≤ 0.1 on the headline target on tb_live after
            recalibration.
P-future.c  loso to tb_live within 50% Brier of within-source loro.
P-future.d  G5 (ledger-dynamics) beats elapsed-only by > 0.02 absolute
            Brier on the headline target.
P-future.e  Online-vs-offline parity (Workstream M) zero diff.
```

### P2. Sign-off package
Status: shipped (v0 estimator: `ledger_basic_v0.1`)

Outputs: `models/ledger_basic_v0.1/{model_card.json, model_card.md}`
+ `reports/sign_off_ledger_basic_v0.1.md`. Driver:
`scripts/run_sign_off.py`. Module:
`coding_estimator/eval/sign_off.py`. The card embeds the gate verdict
and every failure-mode result so a consumer can read the full sign-off
from a single artifact. `not_safe_for_control = true` because P1.c is
indeterminate, P1.b/P1.d are indeterminate, P1.g is indeterminate,
and O7 fails on swe_agent_pilot — none of those clear with margin.

### P3. Downstream readiness
Status: shipped — `reports/NOT_READY_FOR_SCHEDULING.md`

`coding_estimator/eval/sign_off.py::write_p3_report` emits
`READY_FOR_SCHEDULING.md` iff `gate.verdict == "pass"`, else
`NOT_READY_FOR_SCHEDULING.md`. v0 verdict is `indeterminate`, so the
NOT_READY artifact is on disk with prioritized recommendations
(BLOCKING / DATA / AUDIT) for the cheapest next experiment per
blocking condition.

---

## § Workstream Q — DEFERRED: semantic features and sequence models

Out of scope until P passes for the ledger-only estimator. Tracked here so they aren't lost.

```text
Q1. Trace-text features: command embeddings, error embeddings,
    file-touch counts.
Q2. Sequence model on event stream (small transformer or GRU).
Q3. LLM-judge as auxiliary feature (NOT primary estimator).
Q4. Multi-task learning across all targets.
Q5. Active acquisition: which checkpoint to re-annotate by hand
    next, given current calibration gaps.
```

---

## § Workstream R — DEFERRED: scheduling consumer

Explicitly out of scope here. Lives in a future repo when P1 passes.

```text
R1. Belief-state-conditioned policy.
R2. Pause/resume controller.
R3. Effort modulation (model-tier swap).
R4. Online RL or contextual bandit.
```

These are listed only to make the boundary explicit. **Do not start any of them in this repo.**

---

## § 1. Recommended execution order (v0)

Strict dependencies, with deferred items removed from the critical path:

```text
A (incl. A6, A7) → B → C (incl. canonical-source decisions) → D → E (v0 targets only)
   → F (gate F11) → G (G1, G2, G4 only) → H → I (I0, I1 only) → J
   → N → O → P (gate P1)
                                  ↘
                                   K, L2, L3, L4 in parallel after G7 ships
                                   (L1 stays blocked on upstream NTB6)
```

Parallelism windows:

```text
- C4a/C4b/C4c can run in parallel after C2.
- D3a–D3h can run in parallel after D2.
- F1–F4 can run in parallel after E* completes.
- G1, G2, G4 can run in parallel after F11 passes.
- O1, O5, O7 can run in parallel after I1+J ship.
```

Deferred (post-P1):
```text
- E2/E3/E5 horizon, wall-clock, and survival labels
- G3, G5, G6 baselines
- I2 (GBM), I3 (hazard), I4 (sequence), I5 (semantic)
- K2 (online feasibility); L2/L4 (parity/per-source calibration);
  O2/O3/O4/O6 (small-N adversarial slices)
- M (online inference)
- Q (semantic features)
```

---

## § 2. Mission-aligned summary

The plan above operationalizes one paragraph:

> The estimator is a belief layer over live coding-progress ledgers. The ledger records the evolution of visible discovered work: what the agent has found, attempted, completed, reopened, invalidated, blocked on, and validated. The estimator does not redefine progress and does not decide actions. At each checkpoint, it consumes prefix-only ledger features and outputs calibrated probabilities over successful completion by future horizons, remaining time, and near-future progress dynamics. The first goal is to determine whether live ledger histories contain decision-relevant signal beyond elapsed time and scalar progress. The second goal is to identify which trajectory shapes — high-progress failures, low-progress successes, validation shocks, stuck loops, and scope discovery — remain difficult. Only after this belief layer is calibrated on live Terminal-Bench and retrospective coding-agent traces should it be considered for scheduling or modulation.

The go/no-go gate (§ Workstream P) is the only place this repo declares whether that paragraph has been satisfied.
