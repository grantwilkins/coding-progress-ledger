# TASKS — SWE-agent Retrospective Pilot and Long-Horizon Backlog

This file is the working backlog for taking the `coding-progress-ledger` framework from toy/control runs into a real **SWE-agent retrospective pilot**, and beyond.

The immediate goal:

> Take real SWE-agent trajectories, convert them into run directories, retrospectively annotate ledgers from visible evidence, build event/step observation tables, and rerun the smoke-test completion pipeline on a small balanced sample.

We are **not** building a controller, training large models, or claiming predictive performance. We are checking whether the ledger schema and the observation pipeline survive contact with real agent traces.

Repo grounding (read these before starting any task):

- `docs/AGENT_USAGE.md` — protocol for using `LedgerSession` while coding.
- `ledger_progress/core.py` — event/status/category enums and replay engine.
- `ledger_progress/session.py` — `LedgerSession` helper (`add`, `start`, `complete`, `block`, `reopen`, `invalidate`, `split`, `score`, `export_jsonl`, `export_curve_csv`).
- `ledger_progress/queries.py` — `CODING_CATEGORIES = (PRODUCT, VALIDATION, INVESTIGATION)`.
- `ledger_progress/run_manager.py` — `ledger-run` CLI (`init`, `export-run`, `capture-tests`, `capture-diff`, `check-run`, `summarize-run`).
- `scripts/build_ledger_observation_dataset.py`, `scripts/audit_ledger_observation_dataset.py`, `scripts/smoke_test_completion_prediction.py`, `scripts/rescore_suite_by_category.py`.
- `datasets/completion_prediction_smoke_report.md` — the "before" state. The next scientific test it asks for is the SWE-agent retrospective pilot, which is what this file plans.

**Status markers** on each task: `not started` · `in progress` · `blocked` · `done` · `deferred`. These are plain text — they are *not* ledger events. The ledger is for runs, not for tracking pilot work.

## § 0. Project rules for all agents

Every agent and contributor must respect these constraints, in every workstream, before starting any task:

```text
Do not change ledger scoring semantics.
Do not mutate source SWE-agent traces.
Do not rewrite existing historical ledger.jsonl files.
Do not add observer automation yet.
Do not train large models.
Do not claim predictive performance.
Do not infer completion from progress.
Do not use final outcome as a feature.
Do not use future trace events when constructing checkpoint features.
```

Every output must distinguish:

```text
source trace        = immutable input
retrospective ledger = annotation artifact
observation dataset  = derived replay artifact
completion label     = final success/failure metadata
```

If a task seems to require violating one of these rules, stop and escalate — don't quietly relax them.

---

## § Workstream A — Source data acquisition and inventory

### A1. Create external data area
Status: not started

Goal: Set up an isolated directory for raw SWE-agent traces, separate from the curated toy/control runs.

Inputs:
```text
(none — fresh directory layout)
```

Outputs:
```text
external_data/swe_agent/raw/
external_data/swe_agent/manifests/
external_data/swe_agent/samples/
external_data/swe_agent/README.md
```

Acceptance:
```text
external_data/swe_agent/raw/ exists
external_data/swe_agent/manifests/ exists
external_data/swe_agent/samples/ exists
README.md exists and explains that raw traces are immutable and out of scope for git
.gitignore (or .gitattributes) decision documented for raw/
```

### A2. Decide first source format
Status: not started

Goal: Lock in the exact SWE-agent trajectory source we'll use first.

**Pre-recommendation (do not redo source discovery — verify and confirm):**
- Primary: `nebius/SWE-agent-trajectories` on Hugging Face. ~80k trajectories, parquet, fields known to include `instance_id` (str), `trajectory` (list of role/content), `model_name` (str), `target` (bool — success label co-located), `generated_patch` (str), `eval_logs` (str), `exit_status` (str). Access pattern: `datasets.load_dataset('nebius/swe-agent-trajectories')`.
- Fallback: `SWE-bench/SWE-smith-trajectories` (~5k Claude 3.7 Sonnet traces; fields `instance_id`, `messages`, `resolved`, `patch`, `traj_id`, model `Claude 3.7 Sonnet`).
- Reference only (do not use as primary): `github.com/SWE-agent/SWE-agent/trajectories/demonstrations` (`.traj` schema reference, too small for sample), `github.com/swe-bench/experiments` (mostly patch-only).

A2's job is to **verify by inspecting one row** that the schema matches expectations, and to **resolve the license question** before bulk download.

Inputs:
```text
HF dataset page: https://huggingface.co/datasets/nebius/SWE-agent-trajectories
Fallback page:   https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories
```

Outputs:
```text
external_data/swe_agent/SOURCE_FORMAT.md
external_data/swe_agent/raw/sample_row.json   # one decoded row, for schema reference
```

Acceptance:
```text
SOURCE_FORMAT.md exists and chooses ONE concrete primary source
SOURCE_FORMAT.md lists every field name verbatim from a real row, with type
SOURCE_FORMAT.md explicitly marks fields whose meaning is unconfirmed
SOURCE_FORMAT.md states success/failure label field, patch field, eval log field, trajectory field
SOURCE_FORMAT.md records license/usage constraints discovered (or notes "unknown — proceed only with internal/research use")
sample_row.json contains exactly one trace, raw, never edited
```

### A3. Build raw manifest script
Status: not started

Goal: Scan downloaded raw data and emit a deterministic manifest, without mutating raw inputs.

Inputs:
```text
external_data/swe_agent/raw/
```

Outputs:
```text
scripts/swe_agent_inventory.py
external_data/swe_agent/manifests/swe_agent_inventory.csv
```

CSV columns:
```text
source_id
instance_id
model_name
trajectory_available
trajectory_length
final_success_available
final_success
patch_available
eval_log_available
repo_name
issue_id
raw_path_or_dataset_index
parse_status
parse_error
```

Acceptance:
```text
script runs without mutating raw data
CSV contains one row per trace
parse failures are recorded in parse_status/parse_error, not fatal unless all fail
script is deterministic (same raw → same CSV byte-for-byte)
```

Tests: `tests/test_swe_agent_inventory.py` covers `_parse_repo_and_issue` (last-dash split, dashes-in-repo), `_row_to_record` (False vs missing target, empty trajectory list, patch/eval string-non-empty), `_format_cell` (bool not int subclass branch), and `_write_csv` byte-identity / sort key.

### A4. Count success/failure availability
Status: not started

Goal: Summary report so a human can decide if a balanced pilot sample is reachable.

Inputs:
```text
external_data/swe_agent/manifests/swe_agent_inventory.csv
```

Outputs:
```text
external_data/swe_agent/manifests/swe_agent_inventory_summary.md
```

Report:
```text
total traces
traces with usable trajectory
traces with success labels
success count
failure count
missing-label count
patch availability
eval-log availability
top model names
top repos
```

Acceptance:
```text
summary tells us whether a balanced pilot sample (10 success / 10 failure) is possible
summary flags imbalances (e.g. "only 4 failures in this dump") explicitly
```

---

## § Workstream B — Sampling strategy

### B1. Define pilot sample policy
Status: not started

Goal: Write down inclusion/exclusion criteria, fallbacks, and seeding before any selection happens.

Outputs:
```text
external_data/swe_agent/PILOT_SAMPLING_POLICY.md
```

Initial target:
```text
20 traces total
10 successful
10 failed
```

Fallbacks (if balance not achievable):
```text
fallback 1: 10 success / 10 failure from any model
fallback 2: 5 success / 5 failure if parsing is difficult
fallback 3: record why balance failed; do not silently rebalance
```

Sampling constraints:
```text
prefer one model/scaffold first if possible
prefer tasks with patch/eval logs
avoid duplicate instance_id unless explicitly comparing models
avoid extremely short malformed traces (< N steps; N stated in policy)
avoid traces with missing trajectory
```

Acceptance:
```text
policy states inclusion/exclusion criteria
policy states random seed (e.g. seed=0)
policy states fallback rules in order
```

### B2. Build sampler script
Status: not started

Goal: Deterministic sampler reading the inventory CSV, producing a pilot sample CSV.

Inputs:
```text
external_data/swe_agent/manifests/swe_agent_inventory.csv
--n-success 10
--n-failure 10
--seed 0
```

Outputs:
```text
scripts/sample_swe_agent_pilot.py
external_data/swe_agent/manifests/swe_agent_pilot_sample.csv
```

CSV columns:
```text
pilot_id
source_id
instance_id
model_name
final_success
trajectory_length
patch_available
eval_log_available
raw_path_or_dataset_index
selection_reason
```

Acceptance:
```text
balanced sample produced if possible
selection is deterministic under --seed
sample excludes malformed traces per B1 criteria
re-running with same seed produces byte-identical CSV
```

Tests: `tests/test_sample_swe_agent_pilot.py` covers strict `_parse_bool` (rejects `bool("False")`), numeric-not-lexical dataset-index dedupe, the I1-I7 funnel (each filter drops a dedicated row, traj-length boundary inclusive), `_sample_side` byte-determinism across input-order permutations, pilot_id assignment from sorted instance_id (not RNG order), failure-before-success final ordering, the four-level fallback ladder, and end-to-end `main()` byte-identity (including invariance to inventory row permutation and a sanity check that seed actually changes output).

Open follow-up: `selection_reason` strings emitted by the script (`primary_balanced_*`, `fallback1_all_models_*`, `fallback2_short_traj_*`, `fallback3_half_targets_*`) diverge from the policy doc strings (`primary_balanced_10_10`, `fallback_any_model`, `fallback_short_traj`, `fallback_5_5`). Tests assert the level identifier returned by `_select_with_fallbacks`, not the policy string. Reconcile script ↔ policy doc before B3 audits cite either.

### B3. Create sample audit
Status: not started

Goal: Human-readable audit of what got selected, before annotation effort starts.

Outputs:
```text
external_data/swe_agent/manifests/swe_agent_pilot_sample_summary.md
```

Report:
```text
success/failure counts
model distribution
repo distribution
trajectory length distribution
patch/eval availability
known caveats
```

Acceptance:
```text
human can inspect whether sample is reasonable before annotation begins
```

---

## § Workstream C — Trace import and run directory conversion

### C1. Define SWE-agent normalized trace schema
Status: done — `docs/SWE_AGENT_TRACE_SCHEMA.md` (commit c93d182)

Goal: A small, well-documented internal schema for a normalized agent trace, decoupled from any one upstream source.

Outputs:
```text
docs/SWE_AGENT_TRACE_SCHEMA.md
ledger_progress/swe_agent_schema.py    # OPTIONAL — only if dataclasses help
```

Normalized event shape (per step):
```json
{
  "step_index": 0,
  "role": "assistant|tool|environment|system|unknown",
  "thought": "...",
  "action": "...",
  "observation": "...",
  "tool_name": "...",
  "command": "...",
  "files_touched": [],
  "timestamp": null
}
```

Acceptance:
```text
schema can represent thought/action/observation traces
schema tolerates missing fields
schema preserves raw fields under raw_metadata (no silent drops)
schema doc shows worked example from one nebius row and one SWE-smith row (for portability)
```

### C2. Implement trace normalizer
Status: done — `scripts/normalize_swe_agent_trace.py` + `tests/test_normalize_swe_agent_trace.py` (commit e2281af)

Goal: Convert one raw row/file into the normalized trace + a human summary.

Inputs:
```text
one raw trajectory row/file (path or HF dataset index)
```

Outputs:
```text
scripts/normalize_swe_agent_trace.py
<run_dir>/normalized_trace.json
<run_dir>/trajectory_summary.md
```

Acceptance:
```text
preserves original raw trace separately as <run_dir>/source_trace.json (never modified)
does not drop unrecognized fields without storing them in raw_metadata
handles at least 3 sample traces (success, failure, ambiguous) end-to-end
trajectory_summary.md is short, scannable, and lists per-step thought/action/observation tags
```

### C3. Implement import-to-run script
Status: done — `scripts/import_swe_agent_trace.py` + `tests/test_import_swe_agent_trace.py` (commit 801fbf3). Cache populated by `scripts/populate_swe_agent_pilot_cache.py` (network) into `external_data/swe_agent/pilot_cache/` (gitignored). All 20 pilot run dirs materialized at `runs/swe_agent_pilot/` (gitignored) and `--verify-only` reports `verify ok: 20 run dirs`.

**D4 follow-up resolved:** the importer now writes `test_output.txt` (the framework's standard artifact name; `ledger-run check-run` requires it) sourced from upstream `eval_logs`. The earlier `eval_output.txt` mirrored the upstream field name unnecessarily. Generalizable rule: importers map `<upstream-field-name>` → `<framework-artifact-name>`; the framework does not learn upstream-specific names. Pilot-zero annotation script's alias workaround removed; re-import + re-annotation reproduce the same progress numbers (s_01: 1.00 / f_01: 0.75).

Goal: Bulk-convert the pilot sample into per-run directories. Critically, **does not produce `ledger.jsonl` yet** — annotation is its own workstream.

Inputs:
```text
--sample-csv external_data/swe_agent/manifests/swe_agent_pilot_sample.csv
--runs-dir runs/swe_agent_pilot
```

Outputs (per pilot trace):
```text
runs/swe_agent_pilot/<pilot_id>/
  task.md
  source_trace.json
  normalized_trace.json
  trajectory_summary.md
  final_diff.patch
  eval_output.txt
  run_notes.md
  source_metadata.json
```

`source_metadata.json` shape:
```json
{
  "source": "swe_agent",
  "pilot_id": "...",
  "instance_id": "...",
  "model_name": "...",
  "final_success": true,
  "final_success_source": "source_label",
  "patch_available": true,
  "eval_log_available": true,
  "trajectory_length": 42,
  "annotation_mode": "not_annotated"
}
```

Acceptance:
```text
all pilot sample traces get run directories
source_trace.json preserves raw input byte-equivalently
normalized_trace.json is machine-readable (loads with json.load without errors)
task.md contains issue/task text if available; otherwise explicitly notes absence
run_notes.md is initialized from a template with TODO sections
NO ledger.jsonl is generated by default
```

### C4. Pre-annotation verification (folded into C3)
Status: done — `--verify-only` mode in `import_swe_agent_trace.py`; rejects missing artifacts, empty patch/eval when their flags are True, and unexpected `ledger.jsonl` files (commit pending)

Goal: A pre-annotation check that doesn't need `ledger.jsonl` (which `ledger-run check-run` requires).

**Deviation from brief:** the brief proposes a standalone `scripts/check_swe_agent_imports.py`. That duplicates work `import_swe_agent_trace.py` (C3) should do itself, and creates a parallel CLI alongside `ledger-run`. Instead:

- Have C3's importer self-verify its outputs at the end of every per-run import (raise on missing/empty artifacts).
- Add `import_swe_agent_trace.py --verify-only --runs-dir runs/swe_agent_pilot` as a re-runnable mode.
- **Do not** create a separate `check_swe_agent_imports.py` script unless C3's verify mode turns out to be insufficient.

Acceptance:
```text
C3's importer reports missing source artifacts (task.md, source_trace.json, normalized_trace.json, final_diff.patch if expected, eval_output.txt if expected, run_notes.md, source_metadata.json)
verify mode does NOT expect ledger artifacts before annotation
verify mode exits non-zero if any pre-annotation artifact missing; logs which run_dir/which artifact
no parallel script unless explicitly justified
```

---

## § Workstream D — Retrospective ledger annotation protocol

### D1. Write annotation guidelines
Status: done — split into two files so the protocol is reusable across trace sources:
- `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` — the **general** binding protocol (rules, categories, statuses, event types, procedure, pitfalls). Source-agnostic. Cross-references `ledger_progress/core.py` and `ledger_progress/queries.py:CODING_CATEGORIES`.
- `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md` — a **thin SWE-agent addendum** that specializes the general protocol: shell-vocabulary→category map, run-dir artifact list, two real worked examples (s_01 / f_01) of good and bad annotations, SWE-agent-specific pitfalls. The general doc wins on any conflict.

**Pre-E1 stress-test refinements (from f_03 walk, 113-step stuck loop):**
- General protocol § 6 `blocked` now carries a stuck-loop rule: same N≥3 commands verbatim, no query variation, no new tool output → mark `blocked` at the third iteration; cite both endpoints.
- SWE-agent addendum § 5 pitfall #6: harness-forced termination (`exit_status='submitted (exit_context)'` etc.) does NOT generate an `ARTIFACT` leaf when the agent never issued a literal `submit`.
- SWE-agent addendum § 5 pitfall #7: `final_diff.patch` is a state diff, not an action diff — investigation/repro residue (e.g. created `reproduce.py`) shows up there even when no fix was attempted.
- f_03 annotated as a third pilot-zero validation run; ends at progress=0.50 / coding=0.50 (1 complete + 1 blocked) under the refined guidance.

Goal: A protocol document that constrains annotation to visible trace evidence and prevents narrative reconstruction.

Outputs:
```text
docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md
```

Core principles:
```text
Annotate only visible trace evidence.
Do not force non-monotonicity.
Do not make failed traces reach 1.0 unless the discovered work was actually complete.
Do not use final_success to decide intermediate completion.
Use final eval only as final validation evidence.
Preserve uncertainty in run_notes.md.
```

Categories (cross-reference: `ledger_progress/core.py:SubtaskCategory`, `ledger_progress/queries.py:CODING_CATEGORIES`):
```text
investigation:
  understanding issue, inspecting repo, localizing relevant files
product:
  editing implementation, changing tests if required by task, fixing code
validation:
  running tests, interpreting failures, checking final patch/eval
environment:
  dependency installation, setup errors, path/package issues unrelated to product
artifact:
  patch export, final answer/report generation
documentation:
  docs or explanatory changes when task requires them
```

Status guidance (matches `ledger_progress/core.py:Status`):
```text
not_started:  discovered but no action yet
in_progress:  partial action visible
blocked:      cannot proceed due to missing info/tool/env
complete:     supported by concrete trace evidence
invalidated:  approach/subtask no longer active but should remain in history
deleted:      rarely use; prefer invalidated
```

Acceptance:
```text
protocol includes good/bad examples (at least 2 of each)
protocol explains progress is NOT success probability
protocol explains discovered work vs true hidden work
protocol cross-references the in-repo enum source files so annotators don't drift
```

### D2. Create annotation template
Status: done — `docs/LEDGER_ANNOTATION_TEMPLATE.md` (general; not SWE-agent-specific). Sections: initial reading, initial ledger proposal, checkpoint notes, uncertain decisions, evidence citations, known missing evidence, final scope closure, schema gaps. Header lists every `LedgerSession` method so the notes-to-events link is explicit.

Goal: A copy-pasteable scaffold for `run_notes.md` so annotators record the same kinds of evidence in the same shape.

Outputs:
```text
docs/SWE_AGENT_LEDGER_ANNOTATION_TEMPLATE.md
```

Sections:
```text
Initial ledger proposal
Checkpoint notes
Uncertain decisions
Evidence citations
Final scope closure
Known missing evidence
```

Acceptance:
```text
annotators can copy template into run_notes.md as a starting point
template references LedgerSession method names (add/complete/split/etc.) so the link from notes to events is explicit
```

### D3. Manual annotation helper (deferred by default)
Status: deferred — D4 confirmed snippets are sufficient. The pilot-zero annotations were hand-encoded directly against `LedgerSession` in `scripts/annotate_swe_agent_pilot_zero.py`; no boilerplate-paste friction surfaced that a generic CLI helper would address. Per § D3 acceptance: helper not built; closing this section. Re-open only if E1 (full N=20) reveals concrete repeated friction.

Goal: Make annotation easier without automating semantic decisions.

**Deviation from brief:** the brief proposes building `scripts/annotate_swe_agent_run.py`, with a hedge that it can be dropped. Given that `LedgerSession`'s API is already ergonomic (one-line `session.add(...)`, `session.complete(...)`, `session.split(...)`), and that we have ~2 unannotated traces (D4) before the full 20, the right call is the inverse:

- **Default: do NOT build this helper.** Use Python snippets / a Jupyter notebook / `python -i` against `LedgerSession` directly during D4.
- **Only build it if D4 reveals concrete repeated friction** (e.g. annotators are pasting the same boilerplate to load source_trace.json and view next chunk on every run). In that case, build the smallest helper that addresses the specific friction, not a generic CLI.

If built later, the candidate functions are:
```text
show-task               # prints task.md
show-next-chunk         # prints next N steps of normalized_trace.json
show-score              # prints scoring.score(ledger) result
append-event            # thin wrapper around LedgerSession methods
export-ledger           # delegates to LedgerSession.export_jsonl
export-progress         # delegates to `ledger-run export-run`
```

Acceptance (if built):
```text
helper NEVER suggests subtasks
helper NEVER marks completion automatically
helper only records human/agent-entered events
helper does not duplicate any function already on LedgerSession or ledger-run
if not built, this section is closed with a note in D4's run_notes.md saying "snippets sufficient — helper not needed"
```

### D4. Annotate 2 traces by hand as pilot-zero
Status: done — three pilot-zero annotations (s_01 / f_01 / f_03) drive an E1-ready pipeline:
- `annotations/swe_agent_pilot/<pilot_id>.json` — declarative event spec (committed; the canonical annotation record).
- `annotations/swe_agent_pilot/<pilot_id>.notes.md` — run_notes prose with `{{PROGRESS_OVERALL}}` / `{{PROGRESS_CODING}}` placeholders (committed).
- `scripts/annotate_pilots_from_spec.py` — source-agnostic driver that replays specs into ledger.jsonl + run_notes.md + annotation_quality.json under each `runs/.../<pilot_id>/`. Idempotent.
- `tests/test_annotate_pilots_from_spec.py` — 12 tests covering op routing (block != complete), split-without-invalidate, id-mismatch errors, unknown-op rejection, placeholder substitution, and missing-notes-file failure.

Final progress: s_01 = 1.00 / 1.00; f_01 = 0.75 / 0.67; f_03 = 0.50 / 0.50. All three pass `ledger-run check-run`.

Goal: Catch protocol problems with a tiny sample before scaling to 20.

Selection:
```text
1 successful trace
1 failed trace
```

Outputs (per run):
```text
runs/swe_agent_pilot/<pilot_id>/ledger.jsonl
runs/swe_agent_pilot/<pilot_id>/progress.csv
runs/swe_agent_pilot/<pilot_id>/progress_by_category.csv
runs/swe_agent_pilot/<pilot_id>/summary_by_category.json
runs/swe_agent_pilot/<pilot_id>/run_notes.md   # extended from template
```

Acceptance:
```text
ledger-run check-run passes, OR missing artifacts are explained in run_notes.md and source_metadata.json
annotations follow the D1 protocol
human reviews both before scaling to 20 traces
```

### D5. Annotation quality checklist
Status: done — `annotation_quality.json` emitted by the same pilot-zero script. `whether_schema_gap_found=True` for both runs (real gap surfaced: framework's `test_output.txt` vs C3's `eval_output.txt`; aliased at annotation time and recorded in run_notes § 8). `whether_final_success_used_only_at_end=True` and `whether_progress_forced=False` for both — the upstream label was never used as ledger evidence during the walk.

Goal: Per-run quality metadata so we can later audit annotation drift, not just code drift.

Outputs (per run):
```text
runs/swe_agent_pilot/<pilot_id>/annotation_quality.json
```

Fields:
```json
{
  "annotation_time_minutes": 0,
  "number_of_subtasks": 0,
  "number_of_uncertain_events": 0,
  "number_of_evidence_gaps": 0,
  "whether_final_success_used_only_at_end": true,
  "whether_progress_forced": false,
  "whether_schema_gap_found": false
}
```

Acceptance:
```text
quality metadata exists for every annotated pilot run
honesty: "whether_progress_forced" and "whether_final_success_used_only_at_end" reflect the annotator's actual experience, not aspiration
```

---

## § Workstream E — Retrospective annotation at pilot scale

### E1. Annotate 20 traces
Status: not started

Goal: Apply the D1 protocol to the full pilot sample.

Outputs (per run):
```text
runs/swe_agent_pilot/<pilot_id>/ledger.jsonl
runs/swe_agent_pilot/<pilot_id>/progress.csv
runs/swe_agent_pilot/<pilot_id>/progress_by_category.csv
runs/swe_agent_pilot/<pilot_id>/summary_by_category.json
runs/swe_agent_pilot/<pilot_id>/annotation_quality.json
runs/swe_agent_pilot/<pilot_id>/run_notes.md
```

Acceptance:
```text
20 runs annotated OR failures explained per-run in run_notes.md
no source_trace.json modified
all ledger.jsonl files replay (via ledger_progress.serialization.from_jsonl + replay)
```

### E2. Run run-manager exports
Status: not started

Goal: Use the existing tooling to regenerate derived artifacts from the source-of-truth ledger.

Per run:
```bash
ledger-run export-run runs/swe_agent_pilot/<pilot_id>
ledger-run check-run  runs/swe_agent_pilot/<pilot_id>
```

If `final_diff.patch` / `test_output.txt` are unavailable from the upstream source, record the missingness explicitly in `source_metadata.json` and `run_notes.md` rather than fabricating placeholders.

Acceptance:
```text
derived outputs regenerated from ledger.jsonl
source hash preserved
missing artifacts are documented per-run, never invented
```

### E3. Write pilot annotation summary
Status: not started

Outputs:
```text
runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md
```

Report:
```text
number annotated
success/failure split
average annotation time
median subtasks
median final coding progress
high-progress failures
low-progress successes
non-monotonic coding runs
schema gaps (severity counts)
common evidence gaps
category distribution
annotation uncertainty distribution
```

Acceptance:
```text
summary states whether schema changes were needed
summary states whether annotation felt like trace-backed observation or narrative reconstruction (honest qualitative judgment)
```

---

## § Workstream F — Observation dataset integration

### F1. Extend dataset builder to include SWE-agent pilot
Status: not started

Goal: Verify (and only modify if needed) that the existing builder picks up `runs/swe_agent_pilot/**`. The current builder already scans `runs/**/ledger.jsonl`, so this is mostly a sanity check.

Inputs:
```text
runs/swe_agent_pilot/**/ledger.jsonl
```

Possible filter additions to scripts/build_ledger_observation_dataset.py:
```text
--include "runs/swe_agent_pilot/**"
--exclude archived fixture repos
```

Acceptance:
```text
event and step datasets include SWE-agent pilot rows
run_id preserves swe_agent_pilot prefix
no toy/control runs are accidentally re-categorized as SWE-agent
```

### F2. Generate SWE-agent-only observation tables
Status: not started

Command:
```bash
python scripts/build_ledger_observation_dataset.py \
  --runs-dir runs/swe_agent_pilot \
  --output-event-csv datasets/swe_agent_pilot_observations_event.csv \
  --output-step-csv  datasets/swe_agent_pilot_observations_step.csv \
  --summary-md       datasets/swe_agent_pilot_observations_summary.md
```

Acceptance:
```text
SWE-agent-only event table generated
SWE-agent-only step table generated
summary generated
no rows from runs/task_*, runs/control_*, runs/negative_control_*
```

### F3. Audit SWE-agent observation dataset
Status: not started

Command:
```bash
python scripts/audit_ledger_observation_dataset.py \
  --input-csv  datasets/swe_agent_pilot_observations_step.csv \
  --output-md  datasets/swe_agent_pilot_observations_step_audit.md \
  --output-json datasets/swe_agent_pilot_observations_step_audit.json
```

Acceptance:
```text
integrity passes (weight/leaf sums consistent, progress in [0, 1], deltas consistent)
unknown labels are zero or documented
native/resolved category rates reported
event-vs-step differences reported
```

### F4. Compare toy/live vs SWE-agent distributions
Status: not started

Outputs:
```text
datasets/observation_distribution_comparison.md
```

Compare across the two populations:
```text
number of rows per run
final coding progress
final overall progress
non-monotonicity rate
drop-source distribution
category distribution
weak evidence rate
success/progress quadrants
active denominator size
split/reopen rates
```

Acceptance:
```text
report says whether SWE-agent traces are more diverse than toy/live runs (and how)
report flags any toy/live property that doesn't survive on real traces
```

---

## § Workstream G — Completion-prediction smoke test on SWE-agent pilot

### G1. Run existing smoke script on SWE-agent-only step table
Status: not started

Note: the current `scripts/smoke_test_completion_prediction.py` uses `--input-csv`, `--predictions-csv`, `--report-md`. **This file uses the existing flag names** — earlier draft language using `--output-report` / `--output-predictions` was a brief artifact, not a code change. Don't rename.

Command:
```bash
python scripts/smoke_test_completion_prediction.py \
  --input-csv       datasets/swe_agent_pilot_observations_step.csv \
  --predictions-csv datasets/swe_agent_pilot_completion_smoke_predictions.csv \
  --report-md       datasets/swe_agent_pilot_completion_smoke_report.md
```

Acceptance:
```text
leave-one-run-out by run_id (already implemented)
no leakage (no future events, no final_success used as feature)
predictions generated for all three model variants: progress_only, ledger_basic, elapsed_only
disclaimer included in report
```

### G2. Add SWE-agent smoke report interpretation
Status: not started

Goal: Make the report explicitly answer the questions that motivate the pilot.

Outputs (extend `datasets/swe_agent_pilot_completion_smoke_report.md`):
```text
Do failed runs still get high predicted probabilities?
Do high-progress failures exist naturally in SWE-agent traces?
Does ledger_basic differ from progress_only?
Does elapsed_only remain competitive?
Do evidence gaps dominate the signal?
```

Acceptance:
```text
report does NOT claim predictive performance
report says whether data is suitable for a larger retrospective study
report cross-references which case studies (G3) illustrate which finding
```

### G3. Case-study extraction
Status: not started

Outputs:
```text
datasets/swe_agent_pilot_case_studies.md
```

Pick 4 examples:
```text
1. successful high-progress normal run
2. successful non-monotonic run
3. failed high-progress run
4. failed low-progress / stuck run
```

For each:
```text
task summary
final outcome
progress curve summary
key ledger events (with step indices)
evidence gaps
why it matters
```

Acceptance:
```text
case studies are trace-backed (every claim cites a ledger event or trace step)
no unsupported claims
each case study is one page or less
```

---

## § Workstream H — Inter-annotator reliability

> **Scheduling note:** H is methodologically important but heavy (5 extra annotations at ~30 min each). Run H **after** E2 lands and only if M leans toward "scale". If M leans "pause" or "schema-change-needed", skip H and revisit when annotation effort is justified by a real scale-out plan.

### H1. Duplicate-annotate 5 traces
Status: deferred until after E2 · _scheduling adjusted from brief_

Selection:
```text
2 successful
2 failed
1 ambiguous / high-progress failure if available
```

Independent directories (annotators do NOT see each other's ledger):
```text
runs/swe_agent_pilot_reannotation/<pilot_id>__annotator_a/
runs/swe_agent_pilot_reannotation/<pilot_id>__annotator_b/
```

Acceptance:
```text
two independent ledgers per selected trace
same source_trace.json shared by both annotators (immutable)
```

### H2. Compare annotations
Status: not started

Outputs:
```text
scripts/compare_ledger_annotations.py
runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md
```

Metrics:
```text
final coding progress difference
final overall progress difference
number of subtasks difference
category distribution difference
number of splits/reopens difference
largest drop difference
success/progress quadrant agreement
evidence audit agreement
```

Acceptance:
```text
agreement report identifies where ledger is stable vs subjective
report distinguishes "different ledger, same conclusions" from "different conclusions"
```

### H3. Decide if annotation protocol needs changes
Status: not started

Outputs:
```text
docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md
```

Acceptance:
```text
if no changes needed: one paragraph saying so, with evidence
if changes needed: list MINIMAL changes, ranked by severity
```

---

## § Workstream I — Schema-gap review

### I1. Collect schema gaps from run notes
Status: not started

Outputs:
```text
scripts/collect_schema_gaps.py
runs/swe_agent_pilot/SCHEMA_GAPS.md
```

Searches:
```text
runs/swe_agent_pilot/*/run_notes.md
runs/swe_agent_pilot/*/annotation_quality.json
```

Classify each gap:
```text
missing category
ambiguous evidence
unable to represent partial validation
environment/product boundary unclear
trace lacks command output
trace lacks patch state
subtask too coarse
other
```

Acceptance:
```text
all gaps surfaced in run_notes.md are listed
each carries severity: blocker / annoying / note
```

### I2. Decide no-change vs schema-change
Status: not started

Outputs:
```text
runs/swe_agent_pilot/SCHEMA_DECISION.md
```

Possible outcomes:
```text
No schema change needed for pilot.
Only annotation protocol changes needed.
Need new non-breaking metadata field.
Need core schema change before scaling.
```

Acceptance:
```text
decision is explicit
no schema/code changes made silently — every change has a corresponding outcome line
```

---

## § Workstream J — Native-category quality

### J1. Measure category resolution for SWE-agent pilot
Status: not started

Goal: New annotations should be `category_resolution_mode = native` (i.e. category set explicitly on every subtask).

Report:
```text
native rows
mixed rows
legacy_inferred rows
```

Acceptance:
```text
if any new SWE-agent rows are not native, explain why per-run in run_notes.md
existing toy/control runs are exempt — they predate the native-category convention
```

### J2. Enforce native categories for new annotations
Status: not started

Outputs:
```text
scripts/check_native_categories.py
```

Inputs:
```text
runs/swe_agent_pilot/*/ledger.jsonl
```

Output:
```text
non-native / missing-category events listed by (run_id, step, subtask_id, event_type)
```

Acceptance:
```text
new SWE-agent annotations use explicit categories
legacy old runs exempted via path filter, not by silently passing
```

---

## § Workstream K — Evidence quality

### K1. Evidence availability audit
Status: not started

Goal: Quantify weak vs strong evidence on the new annotations using the existing classifier.

Reuse: `scripts/rescore_suite_by_category.py:audit_completion_evidence` and `:classify_evidence`. Strong evidence types: `{test_output, diff, file_exists, command_output}` (plus `contract_text` as a special case for INVESTIGATION).

Outputs:
```text
runs/swe_agent_pilot/EVIDENCE_AUDIT.md
```

Report:
```text
completion events with test_output evidence
completion events with diff evidence
completion events with command_output evidence
completion events with manual-only evidence
weak product completions
weak validation completions
weak investigation completions
```

Acceptance:
```text
weak evidence is quantified
weak evidence is NOT treated as replay failure (it's a known signal, not a bug)
```

### K2. Source trace evidence-gap report
Status: not started

Goal: What would live instrumentation need to capture that retrospective annotation can't reconstruct?

Outputs:
```text
runs/swe_agent_pilot/SOURCE_EVIDENCE_GAPS.md
```

Report where source data lacks:
```text
baseline failing test output
final passing/failing eval output
patch
file-open context
command output
tool observations
```

Acceptance:
```text
report tells us what a future live SWE-agent integration would need to instrument
report distinguishes "this source omits X" from "no source could reconstruct X"
```

---

## § Workstream L — Visualization and qualitative review

### L1. Generate progress plots
Status: not started

**Pre-flight:** check `pyproject.toml` for matplotlib (or any other plotting lib) before writing the script. If no plotting dep exists, the first sub-step is "add matplotlib to pyproject.toml dev dependencies" — don't pull in plotly/seaborn/etc., matplotlib is the minimal addition. If the team prefers no new deps, fall back to writing per-run progress data to `runs/swe_agent_pilot/plots/<pilot_id>_progress.csv` and let the user plot externally.

Outputs:
```text
scripts/plot_progress_curves.py
runs/swe_agent_pilot/plots/<pilot_id>_progress.png    # if matplotlib added
runs/swe_agent_pilot/plots/<pilot_id>_progress.csv    # always emitted, even without matplotlib
```

Each plot shows:
```text
coding_progress over step
overall_progress over step
drop markers (where coding_progress decreases)
final success/failure marker
```

Acceptance:
```text
plots generated for all annotated traces (or CSVs if no plotting lib)
no exotic style dependency
no new dep added without an explicit pyproject.toml edit in the same commit
```

### L2. Generate pilot dashboard summary
Status: not started

Outputs:
```text
runs/swe_agent_pilot/PILOT_DASHBOARD.md
```

Include:
```text
small table of all traces (pilot_id, success, final coding progress, largest drop, evidence status, annotation time)
relative paths to progress plots
```

Acceptance:
```text
human can inspect the whole pilot at a glance
table renders as proper markdown
```

---

## § Workstream M — SWE-agent scale-up decision

### M1. Write go/no-go memo
Status: not started

Outputs:
```text
runs/swe_agent_pilot/GO_NO_GO_MEMO.md
```

Answer in order:
```text
Did the ledger schema represent real SWE-agent traces?
Was retrospective annotation feasible?
Were evidence gaps tolerable?
Did failed traces provide useful diversity?
Did high-progress failures appear?
Did native category annotation work?
Did smoke prediction plumbing still work?
Should we scale to 100 traces?
Should we instead instrument live SWE-agent runs?
```

Acceptance:
```text
memo gives ONE recommendation:
  scale retrospective
  collect live instrumented traces
  revise schema/protocol
  pause
memo states the cost of being wrong about the recommendation
```

### M2. Define next sample size
Status: not started

Outputs:
```text
runs/swe_agent_pilot/NEXT_SAMPLE_PLAN.md
```

If go (retrospective):
```text
100 traces
50 success / 50 failure
or model/repo-balanced sample (specify split)
```

If go (live instrumented):
```text
defer to Workstream N
```

If no-go:
```text
state which blocker must be fixed first, with owner
```

Acceptance:
```text
clear next action
no ambiguity about who does what next
```

---

## Long-horizon extensions

> **All of the following workstreams are out of pilot scope; revisit after M.** They are listed here so the team has a shared mental map of the road past the pilot, but they should not compete with A–M for attention. Each is intentionally less detailed than A–M.

### § Workstream N — Live SWE-agent instrumentation
Status: not started · _out of pilot scope; revisit after M_

Goal: Wrap a live SWE-agent run with a `LedgerSession` callback so events emit during execution, removing retrospective bias.

Sketch:
```text
N1. Decide sidecar vs in-agent instrumentation
    - sidecar:    parses agent stdout/json events post-step, no agent code changes
    - in-agent:   patch SWE-agent to call LedgerSession directly
    - tradeoff:   fidelity vs maintenance burden vs upstream PR friction
N2. Build minimal hook (or sidecar) for one SWE-agent invocation
N3. Run it on 1 known-success and 1 known-failure instance from SWE-bench
N4. Compare live ledger to retrospective ledger for the same instance — schema/evidence parity check
N5. Decide whether to extend to a live N=20 batch
```

Acceptance:
```text
parity report exists comparing live vs retrospective annotation of the same instance
parity report says whether live instrumentation closes the K2 (SOURCE_EVIDENCE_GAPS) gaps
```

### § Workstream O — Scale-out retrospective study (100+ traces)
Status: not started · _out of pilot scope; revisit after M_

Goal: Move from a 20-trace pilot to a 100+ trace retrospective study, only if M recommends it.

Sketch:
```text
O1. Revised sampling policy with model/repo balance
O2. Batch annotation tooling (improvements based on D3 + E1 friction)
O3. Annotation budget tracking (annotation_quality.json roll-up)
O4. Automated quality gates (J2 + K1 in CI-like pipeline)
O5. Scale-out audit
O6. Updated GO_NO_GO at N=100
```

### § Workstream P — Cross-model / cross-scaffold comparison
Status: not started · _out of pilot scope; revisit after M_

Goal: Take the same set of SWE-bench instances and compare progress/failure shapes across different agent models/scaffolds (e.g. SWE-agent + Llama-70B vs SWE-agent + Claude vs OpenHands + Qwen vs Agentless).

Sketch:
```text
P1. Pick 10 instances each solved/attempted by ≥3 distinct model/scaffold pairs
P2. Annotate or instrument each
P3. Compare progress curves and failure modes per instance, not per model
P4. Report whether failure shape is more about instance difficulty or about scaffold
```

### § Workstream Q — Predictive modeling pass
Status: not started · _out of pilot scope; revisit after M_

Goal: Move beyond the smoke test: real evaluation, real disclaimers.

Sketch:
```text
Q1. Leave-one-repo-out evaluation (not just leave-one-run-out)
Q2. Calibration curves
Q3. Feature ablation (progress only, ledger features only, both)
Q4. Comparison against trivial baselines (always-predict-mean, elapsed-only)
Q5. Explicit non-leakage proofs (no future events, no final_success leakage)
Q6. RESULTS_DISCLAIMERS.md template — what we can and cannot claim
```

### § Workstream R — External write-up / paper draft
Status: not started · _out of pilot scope; revisit after M_

Goal: If the pilot succeeds and Q produces a real result, package it for external readers.

Sketch:
```text
R1. Methods writeup (annotation protocol, evidence audit, scoring semantics)
R2. Threats to validity (annotation drift, retrospective bias, evidence gaps)
R3. Dataset card (for any released annotated artifacts)
R4. Ethics/license review (esp. for nebius/SWE-smith trajectory licenses)
R5. Reproducibility appendix (commands, seeds, dataset versions)
```

### § Workstream S — Open research questions
Status: ongoing · _living document_

Append to this list as work proceeds. Each entry should be one or two sentences with a pointer to the run/case study that surfaced it.

Initial questions:
```text
Is `coding_progress` measurable on agentic tasks where validation evidence is partial?
Do high-progress failures cluster by repo or by model?
Does retrospective annotation systematically over- or under-estimate progress relative to live instrumentation?
Are non-monotonic progress drops a useful signal, or noise?
Does evidence type (test_output vs diff vs command_output) predict downstream success?
Is the PRODUCT/VALIDATION/INVESTIGATION split sufficient, or do we need a new category for tool-use / search?
```

---

## Suggested parallelization

Workstream-to-agent assignment (one suggestion, not a hard contract):

| Agent | Owns |
|-------|------|
| Agent 1 | A (inventory + source format + sampler) — **blocking input** for Agents 2 and 4 |
| Agent 2 | C (import + normalization scripts) |
| Agent 3 | D1, D2, D3 (annotation protocol + templates) |
| Agent 4 | D4, E1 (pilot-zero + scale annotation) |
| Agent 5 | F (dataset integration + audit scripts) |
| Agent 6 | G (smoke-test args + SWE-agent report) |
| Agent 7 | I, K (schema-gap and evidence reports) |
| Agent 8 | L (visualization/dashboard) |
| Agent 9 | H (inter-annotator reliability) |
| Agent 10 | M (final go/no-go memo) |

---

## Minimal first batch

If you want a smaller crunch target before unleashing everyone, do only this:

```text
1. Inventory source traces.            (A1, A3)
2. Sample 2 successful + 2 failed.     (B1 lite — n_success=2, n_failure=2)
3. Import them into run directories.   (C1, C2, C3)
4. Manually annotate ledgers.          (D1 lite, D4)
5. Build step observations.            (F2 on N=4)
6. Run audit.                          (F3 on N=4)
7. Write a 1-page feasibility memo.    (proto-M1)
```

This is the smallest useful SWE-agent pilot. If anything breaks here, fix it before scaling to 20.

---

## Definition of done for "ready to scale"

You are ready to scale beyond the 20-trace pilot only if:

```text
20 traces imported
20 traces annotated or failures explained
source traces never mutated
ledger.jsonl files replay cleanly
step observation table generated
audit integrity passes
native categories are used for new annotations
evidence gaps are quantified
at least one failed high-progress or ambiguous run exists, OR its absence is noted
annotation burden is measured
go/no-go memo recommends a next step
```

That is the point where a larger SWE-agent retrospective study becomes a real experiment rather than a tooling exercise.
