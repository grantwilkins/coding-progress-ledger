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
Do not train large models.
Do not claim predictive performance.
Do not infer completion from progress.
Do not use final outcome as a feature.
Do not use future trace events when constructing checkpoint features.
```

> **Note (post-CRITIC_AUDIT, 2026-04-30):** the previous rule "Do not add observer automation yet" has been **removed**. It served the pilot phase by keeping retrospective annotation honest; it no longer reflects the mission. Live-instrumentation work (§ N, § U) is now the priority.

Every output must distinguish:

```text
source trace        = immutable input
retrospective ledger = annotation artifact (pilot phase, complete)
live ledger         = events emitted by a running agent (target phase, building)
observation dataset  = derived replay artifact
completion label     = final success/failure metadata
```

If a task seems to require violating one of these rules, stop and escalate — don't quietly relax them.

---

## § 0.1 Mission audit (post-K, post-critic-review)

The user's stated mission, verbatim:

> *"an automated way to check and query progress for long range agentic tasks with a ledger design."*

Four critic agents audited the framework against this mission on 2026-04-30. The unanimous verdict, with the surgical fixes already shipped and the strategic gaps still open, is documented in **`runs/swe_agent_pilot/CRITIC_AUDIT.md`**.

**Surgical fixes shipped 2026-04-30 (commit `5bdcab6`):**

- `LedgerEvent.timestamp` (optional ISO-8601) — round-tripped through serialization, auto-stamped by `LedgerSession`. Unlocks deadline-aware progress modeling.
- Six new query functions on `ledger_progress/queries.py`: `current_step`, `active_blocked_leaves`, `reopens_since`, `newly_discovered_since`, `last_validation_event`, `stalled_for`. Implements the "check and query" mission verb at the package level.
- 17 new columns on the observation channel: per-category progress (`product_progress` / `validation_progress` / `investigation_progress`), per-step event windows (`step_added_subtasks`, `step_*_completes`), evidence strength (`step_strong_completions` / `cum_*` — K1 classifier wired in), stalled intervals (`steps_since_*`). Mission features 1–10 are now all first-class.

**Strategic re-scoping per CRITIC_AUDIT § 4:**

- **PROMOTED to current priority:** § Workstream N (live instrumentation), § Workstream W (observation-channel sharpening — new), § Workstream U (live query CLI / monitor), § Workstream T1 (LedgerSet protocol doc, the only multi-task-scope unblocker), § Workstream V (time-aware features, consumes the timestamp field).
- **DEFERRED indefinitely:** § Workstream R (paper write-up; premature — locks in the retrospective framing as the product), § Workstream O (100-trace retrospective scale-out; the smoke test runs at chance by design and trace 21+ is debt), § Workstream P / P1–P3 (cross-source pilots; doubles annotation surface for a hypothesis with no live consumer).

**Pilot phase (A–M) is closed.** The infrastructure produced real value (clean schema, 20 high-fidelity annotations, 5/5 quadrant inter-annotator agreement, mission features now first-class). Everything *forward* should be live-instrumentation-shaped, not annotation-shaped.

## § 0.2 Documentation handoff

Status: done — `IMPLEMENTATION_v0.md` added as the detailed handoff document for the SWE-agent retrospective methodology, source provenance, and broad validation surface. It incorporates a critic pass and explicitly separates current revised protocol from the pre-cleanup v1 corpus where implicit-validation handling remains caveated.

Next tasks:

- Keep `IMPLEMENTATION_v0.md` synchronized if the four Pitfall #8 cleanup pilots (`f_02`, `f_03`, `f_07`, `f_10`) are re-materialized.
- Add a short architecture diagram only if a future reviewer asks for a visual companion; the current handoff is intentionally prose-first.
- When N3 lands, add a separate live-instrumentation implementation doc rather than expanding the SWE-agent retrospective methodology document.

## § 0.3 Observation-channel thesis (post-handoff critique)

Status: current steering constraint.

The scientific object is **the temporal evolution of the visible work frontier**, not the final progress scalar and not final success prediction. The ledger observes:

```text
discovery
completion
reopening
invalidation
blocking
validation state
evidence strength
category-local progress
```

Inventory, sampling, import, normalization, and annotation are trust infrastructure. The variables above are the channel. Future work must make those variables sharper and more automatic, especially in N3/N4 and Workstream W.

Rules this adds:

```text
Do not expand SubtaskCategory for estimator convenience; derive state features instead.
Do not treat exact scalar progress as high precision when protocol sensitivity can move it.
Prefer shape classes over absolute progress values in claims and reports.
Do not train a better final-success classifier before building the estimator checkpoint table.
Treat the retrospective corpus as a parity benchmark for live instrumentation, not as the product.
```

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

**Pre-E1 stress-test refinements (from f_03 113-step + f_07 183-step walks):**
- General protocol § 6 `blocked` now carries a stuck-loop rule that handles cycles of **any length**, including a single command repeated and 2-command oscillations. Mark `blocked` at the **earliest** step where any pattern hits its third iteration.
- SWE-agent addendum § 5 pitfall #6: harness-forced termination (`exit_status='submitted (exit_context)'` etc.) does NOT generate an `ARTIFACT` leaf when the agent never issued a literal `submit`.
- SWE-agent addendum § 5 pitfall #7: `final_diff.patch` is a state diff, not an action diff — investigation/repro residue shows up there even when no fix was attempted (or alongside a real fix, as in f_07).
- General § 7 `SPLIT_SUBTASK` corrected: the parent's status is unchanged by split; it just stops being a leaf and drops out of the progress denominator. (Earlier draft incorrectly said "parent invalidated automatically".)
- Four pilots now span four shapes: s_01 (1.00 success), f_01 (0.75, no validation), f_03 (0.50, investigation blocked), f_07 (0.67, validation blocked after real PRODUCT edit). The progress signal discriminates between failure modes — exactly the contract the framework was supposed to deliver.

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
Status: done — all 20 pilots annotated via the spec-driven driver (`annotations/swe_agent_pilot/<pilot_id>.{json,notes.md}`). Every run passes `ledger-run check-run`. F2/F3 ingest the full 20-pilot dataset cleanly: integrity checks (`completed_exceeds_active`, `delta_mismatches`) all empty. One audit warning surfaced (`s_03` shows a 0.33 native-vs-resolved coding-progress divergence around the step-22 REOPEN); recorded as a downstream-audit signal, not blocking.

**Progress shape across the 20 pilots:**
- 9 of 10 successes end at 1.00; one (`s_04`) ends at 0.75 — the "validated-by-chance" submit-without-test shape, identical to `f_01`/`f_04` despite opposite upstream labels.
- Failures span 0.50–1.00, discriminating cleanly between failure modes:
  - 1.00 (`f_06`): all discovered work completed; failure sits entirely in undiscovered hidden work (the agent's repro never actually triggered the bug). Canonical "framework-allowed-positive-progress-on-failure" case.
  - 0.83 (`f_09`): validation re-opened because final edit happened post-pytest.
  - 0.75 (`f_01`, `f_04`): submitted without in-trace validation.
  - 0.71 (`f_08`): validation in_progress, fields.py investigation blocked in scroll loop.
  - 0.67 (`f_07`, `f_10`): PRODUCT done with validation blocked / PRODUCT blocked in syntax-error stuck loop.
  - 0.60 (`f_05`): PRODUCT and VALIDATION both blocked mid-edit-cycle.
  - 0.50 (`f_02`, `f_03`): investigation blocked; never reached PRODUCT.

The progress signal is highly discriminating between failure modes — exactly the contract D1 promised — and is genuinely independent of `final_success`.

**Pre-E2 protocol refinement** (forced by `f_02`'s 509-step thesaurus loop): general § 6 `blocked` rule split into `(a) command-loop` (same command verbatim ≥3×) and `(b) tool-response-loop` (identical tool response ≥3× regardless of query variation). f_02 didn't trigger (a) because the agent varied keywords; (b) was needed.

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
Status: done — `ledger-run export-run` was invoked by `scripts/annotate_pilots_from_spec.py` for each of the 20 pilots during E1 annotation. `ledger-run check-run` reports "all required artifacts present" on every pilot. `final_diff.patch` and `test_output.txt` are present for all 20 (sourced by C3 from upstream `generated_patch` / `eval_logs`); no fabricated placeholders.

Goal (original): Use the existing tooling to regenerate derived artifacts from the source-of-truth ledger.

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
Status: done — `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` (committed via gitignore exception; the dir is otherwise gitignored to keep upstream traces local-only). Reports per-pilot table, headline numbers, notable shapes (high-progress failure `f_06`, low-progress success `s_04`, three REOPEN runs, six BLOCKED runs), four protocol refinements forced by the pilot, common evidence-gap patterns, category distribution, annotation uncertainty distribution, and an honest qualitative judgment ("~85% observation, ~15% narrative-reconstruction risk concentrated in test-edits-as-PRODUCT and validation-as-implicit-discovered-work calls").

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
Status: done — verified the existing `scripts/build_ledger_observation_dataset.py` accepts `--runs-dir runs/swe_agent_pilot` and emits the SWE-agent-only event/step/summary CSVs without modification. `run_id` preserves the `swe_agent_pilot_*` prefix for every row; no toy/control rows are mixed in when the dataset is built with `--runs-dir runs/swe_agent_pilot`.

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
Status: done — generated at the spec'd paths: `datasets/swe_agent_pilot_observations_event.csv` (202 rows), `datasets/swe_agent_pilot_observations_step.csv` (191 rows), `datasets/swe_agent_pilot_observations_summary.md`. No rows from `runs/task_*`, `runs/control_*`, or `runs/negative_control_*` (verified by `--runs-dir runs/swe_agent_pilot` filter).

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
Status: done — `datasets/swe_agent_pilot_observations_step_audit.{md,json}` and `_event_audit.{md,json}`. Integrity checks: `completed_exceeds_active=[]`, `delta_mismatches=[]`, `first_delta_nonzero=[]`, `invalid_progress=[]`. One residual warning (`s_03` shows a 0.33 native/resolved coding-progress divergence around the step-22 REOPEN) — recorded as a downstream-audit signal, not blocking. Note: 6 spec files (`f_01`, `f_04`, `f_05`, `f_09`, `s_04`, `s_10`) were stable-sorted to step-monotonic order during F3 because non-monotonic step ordering caused 68 spurious `delta_mismatch` warnings; semantics unchanged (same final progress numbers as before).

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
Status: done — `datasets/observation_distribution_comparison.md`. SWE-agent traces are demonstrably more diverse: they populate all four success/progress quadrants (toy/live populates only 2 of 4); they exercise BLOCKED status (toy/live does not); they have richer drop-source distribution (4 categories vs 3, with INVESTIGATION drops only on SWE-agent); and 100% of SWE-agent runs are non-monotonic vs 78% of toy/live. The doc also surfaces a real builder bug: `resolve_final_success` infers from `test_output.txt` and misclassifies 3 SWE-agent successes (`s_03`, `s_06`, `s_09`) as failures because SWE-bench eval logs format pass markers differently from toy/live's pytest output. Authoritative upstream label is `source_metadata.json:final_success`. Builder fix is a follow-up before any G claim about prediction performance.

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
Status: done — `datasets/swe_agent_pilot_completion_smoke_predictions.csv` (574 rows, 191 steps × 3 model variants ≈ 573 + 1 header) and `datasets/swe_agent_pilot_completion_smoke_report.md`. Leave-one-run-out by `run_id`; no leakage (no future events, no `final_success` as feature). All three model variants (`progress_only`, `ledger_basic`, `elapsed_only`) produced predictions. Disclaimer included via the auto-generated report header and reinforced in G2's appended interpretation.

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
Status: done — `datasets/swe_agent_pilot_completion_smoke_report.md` extended with eight subsections (G2.1-G2.8). The report explicitly does NOT claim predictive performance (G2.1 disclaimer). High-progress failures exist naturally per upstream label: `f_06` (G2.3). Builder-classified high-progress failures include 3 misclassified upstream successes (`s_03`, `s_06`, `s_09`); only `f_06` is a real upstream-failure-at-1.00. `ledger_basic` does not improve over `progress_only` on this small/noisy a sample (G2.4); `elapsed_only` is anti-correlated because long SWE-agent traces are stuck-loop failures (G2.5). Verdict (G2.7): data is suitable for a larger retrospective study, conditional on the builder's `resolve_final_success` heuristic being fixed and Workstream H's inter-annotator pass.

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
Status: done — `datasets/swe_agent_pilot_case_studies.md`. Four trace-backed case studies populating all four success/progress quadrants: `s_01` (clean success), `s_03` (non-monotonic success with REOPEN), `f_06` (high-progress failure with hidden-work gap), `f_03` (stuck-loop failure with investigation blocked). Every claim cites a ledger event or trace step; no unsupported claims; each case is one page or less.

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
Status: done — second-annotator pass by an Opus subagent on `s_01`, `s_03`, `f_01`, `f_06`, `f_03` (all four success/progress quadrants). v2 specs at `annotations/swe_agent_pilot_v2/`. Subagent was forbidden from reading v1 annotations / run_notes / pilot summary; only read protocol docs, addendum, template, core enums, and per-pilot trajectory files. Caveat: both annotators are LLM passes (correlated biases); a real human re-pass would expand the agreement signal. Layout deviation from brief: `annotations/swe_agent_pilot_v2/` instead of per-pilot reannotation subdirs because the spec-driven driver consumes specs from `annotations/`.

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
Status: done — `scripts/compare_annotations.py` (10 tests) + `datasets/h_inter_annotator_report.md` (raw metrics) + `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md` (analysis). **Quadrant agreement: 5 / 5.** Mean absolute coding-progress delta: 0.10. 4 of 5 pilots are "different ledger, same conclusions"; only `f_01` shows different conclusions (0.67 vs 1.00) due to the implicit-validation gap. Verdict distribution: 1 high, 2 moderate, 2 low.

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
Status: done — `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md` lists 3 revisions (1 HIGH, 2 LOW) plus 1 no-change acknowledgment of granularity latitude. All three revisions applied to the protocol docs in the same commit:
- **Revision 1 (HIGH):** addendum § 5 pitfall #8 — bug-fix tasks always have implicit validation work. Closes the f_01 disagreement.
- **Revision 2 (LOW):** general § 6 — tightened "third iteration begins" wording to mean the assistant-turn step. Closes the f_03 step-count disagreement.
- **Revision 3 (LOW):** addendum § 1 — `__init__.py` / package-wiring default is PRODUCT (issue-required) vs ENVIRONMENT (purely setup). Resolves the s_03 disagreement.
- **Acknowledgment:** general § 9 — granularity is annotator latitude.

A future re-pass under the revised protocol is the empirical test of whether the changes close the gaps.

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
Status: done — `scripts/collect_schema_gaps.py` walks the 20 pilot run dirs, parses § 8 of each `run_notes.md`, and cross-references `whether_schema_gap_found` from `annotation_quality.json`. Output at `runs/swe_agent_pilot/SCHEMA_GAPS.md`: 2 pilots flagged (`f_02`, `f_07`), 18 explicitly None. Three cross-workstream findings appended (v1's inconsistent Pitfall #8, J1's mixed/native discrepancy, the resolved `final_success` heuristic). Tests at `tests/test_collect_schema_gaps.py` (12 tests) lock in the collector invariants and the core-enum value sets the I2 decision depends on.

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
Status: done — `runs/swe_agent_pilot/SCHEMA_DECISION.md`. **Outcome: No schema change needed for pilot. Only annotation protocol changes needed — already landed.** All 5 pilot-surfaced findings are protocol-text refinements (general § 6, addendum § 5 / § 1) or pipeline / heuristic fixes; none touch `ledger_progress/core.py`. Status / EventType / SubtaskCategory enum value sets unchanged since pilot start (verified by I1's invariant tests). Compatible with M1 + H4 follow-up.

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
Status: done — `runs/swe_agent_pilot/CATEGORY_RESOLUTION_REPORT.md`. **Root cause:** `LedgerSession.add()` had a "PRODUCT-is-default — don't write it to payload" optimization that stripped the `category` field from every PRODUCT subtask's serialization. The dataset builder couldn't distinguish "explicitly PRODUCT" from "missing category", so almost every SWE-agent run resolved to `mixed`. **Fix:** `add()` now always emits `category` in the payload (`ledger_progress/session.py`); one fixture in `tests/test_session.py` updated to match. **Result:** all 191 step rows and 202 event rows are now `native`; zero `mixed`, zero `legacy_inferred`, zero native/resolved warnings. Progress numbers byte-identical before and after — the bug was serialization-layer only.

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
Status: done — `scripts/check_native_categories.py`. Walks `runs/swe_agent_pilot` and `runs/swe_agent_pilot_v3` by default; reports any ADD_SUBTASK or SPLIT child whose payload lacks `category`. Exits non-zero on any violation. Legacy toy/control/live runs are exempted via explicit `--legacy-root` path filter (not silent passing). On the current corpus: 25 SWE-agent runs (20 pilot + 5 H4 v3) all native, zero violations; 18 legacy runs path-filtered. Tests at `tests/test_native_category_invariants.py` (6 tests) cover the J1 invariant (`add()` always emits category) and J2's offender detection on hand-crafted JSONL fixtures.

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
Status: done — `runs/swe_agent_pilot/EVIDENCE_AUDIT.md` (+ `.json`). Script `scripts/audit_pilot_evidence.py` reuses `rescore_suite_by_category.py`'s classifier on the 20 SWE-agent pilots. Across 81 completion events: **51 strong (63%) / 30 manual-only (37%)**. Per category: PRODUCT 24 audited / 20 weak (83%); VALIDATION 12 / 1 (8%); INVESTIGATION 31 / 1 (3%). Most "weak" PRODUCT completions are short edit-acks that don't match stdout/test-output heuristics; K2 proposes a cheap classifier extension that closes ~60% of those without re-annotation. Weak evidence is a signal, not a replay failure.

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

Invariant tests: `tests/test_pilot_evidence_audit.py` locks in row decomposition, totals = sum-over-pilots, classifier fallback semantics, STRONG_EVIDENCE_TYPES scope, and CODING_CATEGORIES exclusion of ARTIFACT/DOCUMENTATION.

### K2. Source trace evidence-gap report
Status: done — `runs/swe_agent_pilot/SOURCE_EVIDENCE_GAPS.md`. Classifies each evidence-source gap as recoverable retrospectively / closed by live instrumentation / structurally unrecoverable. Headline finding: ~60% of K1's `manual_note` completions can be closed by extending the classifier with a `tool_action` strong-evidence type (~30 min, no re-annotation). Live instrumentation (Workstream N) is justified primarily for closing hidden-work-gap visibility (`f_06`-style), agent-vs-harness submit provenance (6 pilots), and pre-fix baseline test runs (none captured in the source).

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

### K3. Extend `classify_evidence` with `tool_action` strong type (post-CRITIC_AUDIT)
Status: not started · _cheap channel improvement; runs in parallel with N1_

Goal: K2 found that ~60% of K1's 30 manual-only completions are short edit/submit/goto/search tool acks that the classifier currently treats as `manual_note`. Adding a `tool_action` strong-evidence type closes them without re-annotation.

Outputs:
```text
scripts/rescore_suite_by_category.py   # add `tool_action` to STRONG_EVIDENCE_TYPES + EVIDENCE_PATTERNS
tests/test_rescore_suite_by_category.py # extend
runs/swe_agent_pilot/EVIDENCE_AUDIT.md  # regenerate via scripts/audit_pilot_evidence.py
```

Acceptance:
```text
EVIDENCE_PATTERNS gains a tuple ("tool_action", ("edit ", "submit ", "goto ", "search_file ", "search_dir ", "tool ack"))
STRONG_EVIDENCE_TYPES includes "tool_action"
manual-only completion count on the SWE-agent pilot drops from 30 to ≤ 12
test invariants in tests/test_pilot_evidence_audit.py still pass
```

### K4. Split manual-only evidence into semantic levels
Status: not started · _feeds W3 and N4; do not change ledger core_

Goal: `manual_note` currently mixes trace-visible but parser-missed evidence with pure annotator judgment. Split evidence audit output into three levels so live instrumentation can distinguish what is automatable from what is inherently semantic.

Evidence levels:
```text
mechanical evidence       # test output, command output, diff, file existence
trace_semantic evidence   # visible action/observation clearly implies visible-subtask completion
annotator_judgment        # completion inferred from context with weak grounding
```

Outputs:
```text
scripts/rescore_suite_by_category.py        # add level classifier on top of existing evidence types
scripts/audit_pilot_evidence.py             # report counts by level and category
runs/swe_agent_pilot/EVIDENCE_AUDIT.md      # regenerate with the 3-level table
tests/test_pilot_evidence_audit.py          # lock level counts on toy fixtures
```

Acceptance:
```text
existing strong/manual-only fields remain backward compatible
K1 headline counts are still reproducible
PRODUCT weak completions are split into trace_semantic vs annotator_judgment
report states which trace_semantic patterns are candidates for live sidecar automation
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
Status: done — `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`. Recommendation: **scale retrospective to 100 traces, gated on a 5-pilot re-annotation under the H3-revised protocol that closes the `f_01` conclusion gap (new task H4 below); defer live instrumentation (N) to the next go/no-go.** Memo synthesizes A–H artifacts; cost-of-being-wrong table quantifies each alternative path. The H4 gate is what makes the recommendation cheap to be wrong about: ~3 h of re-annotation buys an empirical answer to the single methodology gap H surfaced (the implicit-validation rule, `f_01` 0.67 vs 1.00).

### H4. Re-test H3 protocol revisions on the 5 H pilots (gate to M2 / scale)
Status: done — `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`. **Gate PASSES.** Cold-pass v3 specs by an Opus subagent at `annotations/swe_agent_pilot_v3/` (read-restricted to revised protocol + per-pilot trajectories; forbidden from v1, v2, pilot summary, H2 report). Materialized to `runs/swe_agent_pilot_v3/`. Quadrant agreement v1↔v3: **5/5**. f_01 v1↔v3 coding-progress delta: **0.00** (the gate's load-bearing condition: 0.667 vs 0.667). Tests at `tests/test_h4_gate_invariants.py` (5 tests) lock in the gate-condition invariants. Incidental finding: v1 was inconsistent in applying Pitfall #8 across the 4 harness-terminated failure pilots (`f_02`, `f_03`, `f_07`, `f_10`); a focused E1-pass cleanup on those four (~30 min) is recommended before scaling to 100. Does **not** block M2.

Goal: Empirically test whether the HIGH-severity H3 revision (bug-fix tasks always carry implicit validation work) closes the `f_01` 0.67-vs-1.00 disagreement, and whether the LOW-severity revisions close the `f_03` step-count and `s_03` `__init__.py` ENVIRONMENT-vs-PRODUCT gaps.

Inputs:
```text
docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md   # the three revisions
docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md   # already updated in commit 2656391
annotations/swe_agent_pilot_v2/                   # H pass under pre-revision protocol
```

Outputs:
```text
annotations/swe_agent_pilot_v3/                    # cold-pass under post-revision protocol
runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md
```

Acceptance:
```text
re-annotate s_01, s_03, f_01, f_06, f_03 under the v3-revised protocol
gate pass: f_01 produces the same conclusion (final coding-progress within 0.05) in v3 vs both v1 and v2 readings under the new implicit-validation rule
gate fail: H3 revisions are insufficient; do NOT proceed to M2 scale; re-open H3
report cites which v3 leaf each revision touched and whether the gap closed
```

### M2. Define next direction (post-CRITIC_AUDIT pivot)
Status: done — `runs/swe_agent_pilot/NEXT_DIRECTION_MEMO.md`. Pivot from retrospective scale-out to live instrumentation. Defines the eight forward deliverables (N1–N4, U1–U2, T1, V1) and a four-clause gating criterion for proceeding to live N=20 (parity within 0.05 on ≥2 instances; ≥1 K2 gap closed; no agent code changes; sidecar latency < 100ms/step). Cost-of-being-wrong table justifies the pivot's cheapness. Workstream R becomes meaningful only after N4 ships.

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

> **Post-CRITIC_AUDIT note:** M2's framing (retrospective scale-out gated on H4) is **superseded** by the audit. The right next action is **Workstream N (live instrumentation)**, not a 100-trace retrospective batch. Re-write M2 as a one-page memo that records the pivot and references `runs/swe_agent_pilot/CRITIC_AUDIT.md` § 4.

---

## Forward priorities (post-CRITIC_AUDIT)

The workstreams in this section serve the mission verbs. They are **not** "out of pilot scope" — they are the work.

### § Workstream N — Live SWE-agent instrumentation
Status: **PROMOTED to current priority** (post-CRITIC_AUDIT, 2026-04-30). Concretized below; the original sketch was a 5-line aside.

Goal: Wrap a live SWE-agent run so `LedgerEvent`s emit during execution (with real wall-clock `timestamp`s), removing retrospective bias and unlocking K2's hidden-work-gap visibility / submit-provenance / pre-fix-baseline gaps.

The new `LedgerSession.observe(...)` ergonomics (timestamps + clock override) and the new query API (Workstream U's consumer) make this much smaller than M1's "weeks of engineering" estimate.

#### N1. Decide sidecar vs in-agent instrumentation
Status: done — `docs/LIVE_INSTRUMENTATION_DECISION.md`. **Chosen branch: hybrid sidecar with stable wire-format protocol v1.0.** Agents emit JSONL (raw step records via `agent_step` field; or explicit ledger ops via `ledger_ops` field; or both). The sidecar applies `ledger_ops` verbatim when present, else runs a per-framework adapter's `infer_events()` heuristic. Two-tier fidelity, single code path, framework-agnostic — works for SWE-agent, Claude Code, LangGraph, OpenAI Assistants, custom RL, etc., as adopters write a ~50–150 line adapter module. Replay-safety is invariant (idempotent re-feed produces byte-identical ledger.jsonl). N2 acceptance criteria are explicitly enumerated in the decision doc § 8.

#### N2. Build the live ledger sidecar
Status: done — `ledger_progress/sidecar.py`, `ledger_progress/adapters/generic.py`, `ledger_progress/adapters/swe_agent.py`, and `tests/test_sidecar.py`. Ships `python -m ledger_progress.sidecar --run-dir X --adapter {generic,swe_agent}` with stdin JSONL input plus finite `--input-file` batch input. The sidecar validates wire-format v1.x events, rejects unknown major versions, enforces one `run_id` per run dir, applies explicit `ledger_ops` verbatim when present, otherwise routes `agent_step` through the selected adapter. It writes timestamped `ledger.jsonl`, regenerates `progress.csv`, `progress_by_category.csv`, and `summary_by_category.json`, and scaffolds the minimal run artifacts needed for `ledger-run check-run`.

Outputs:
```text
ledger_progress/sidecar.py        # python -m ledger_progress.sidecar --run-dir X
                                  # consumes JSONL on stdin → ledger.jsonl on disk
tests/test_sidecar.py             # synthetic-input integration test
```

Design:
```text
class LedgerSidecar:
    accepts: structured agent step (action, observation, optional file edits,
             optional thought, exit_status)
    routes:  to LedgerSession.add / complete / start / block / split
             via heuristic event inferrer (same vocabulary as the SWE-agent
             addendum's category map)
    emits:   ledger.jsonl with real timestamps (default clock = UTC now)
    exposes: in-memory LedgerSession so Workstream U's queries can read it live
```

Acceptance:
```text
sidecar consumes a synthetic 5-step JSONL stream and emits a timestamped ledger.jsonl
all events carry a non-None timestamp
ledger-run check-run passes on the resulting run dir
```

N2 test coverage also locks replay-equality, per-event timestamp authority, explicit-op bypass, explicit id/category/parent/weight preservation, additive v1.x wire-format compatibility, single-`run_id` invariants, SWE-agent vocabulary categories, no completion from command/exit-status alone, scope-change ops (`split` / `reopen` / `invalidate`), `add_evidence`, CLI input-file mode, and <100ms/event synthetic latency. Full suite after test hardening: `uv run pytest` → 296 passed.

#### N3. Hook one SWE-agent run
Status: done — `scripts/run_swe_agent_live_sidecar.py`, `docs/LIVE_SWE_AGENT_HOOK.md`, and two generated run dirs under `runs/swe_agent_live/`. The hook converts normalized SWE-agent assistant/tool turns into the N1 JSONL wire format, emits fresh wall-clock timestamps, streams the events through `LedgerSidecar(adapter="swe_agent")`, and writes `wire_events.jsonl` plus the standard run artifacts. It hard-fails rather than overwriting an existing live `ledger.jsonl`.

Interpretation: N3 proves the live sidecar path can ingest SWE-agent-shaped events and emit timestamped ledgers. It does **not** prove semantic parity with retrospective annotation. N4 owns the sharper question: which events are mechanically observable, which are weakly inferable, and which remain annotation-only.

Outputs:
```text
runs/swe_agent_live/<instance_id>/...   # one known-success and one known-failure
                                         # SWE-bench instance, run live with sidecar
runs/swe_agent_live/<instance_id>/wire_events.jsonl
```

Acceptance:
```text
two run dirs, each with a real-time ledger.jsonl
each event has a wall-clock timestamp (so Workstream V's time-aware features fire)
wire_events.jsonl retained exactly as consumed by the sidecar
each live run maps to a known retrospective pilot instance
```

Generated N3 runs:
```text
runs/swe_agent_live/Melevir__cognitive_complexity-15   # known success; 21 wire events, 43 ledger events
runs/swe_agent_live/WIPACrepo__iceprod-339             # known failure; 8 wire events, 17 ledger events
```

Both pass `uv run ledger-run check-run <run_dir>`, and every `ledger.jsonl` event has a non-null timestamp.

#### N4. Live-vs-retrospective parity report
Status: done — `scripts/build_live_parity_report.py`, `runs/swe_agent_live/PARITY_REPORT.md`, and `runs/swe_agent_live/EVENT_OBSERVABILITY_MATRIX.md`. Frontier policy resolved: raw-step live instrumentation does **not** invent discovered-but-unattempted validation obligations. Submit-without-validation is represented as `complete_visible_frontier+no_validation_frontier` unless the agent emits explicit `ledger_ops` for validation work. Verdict: **N4 policy-adjusted parity gate passes; N5 may proceed under the no-validation-frontier policy.** Scalar progress still differs on `WIPACrepo__iceprod-339` (`1.000` live vs `0.667` retrospective), and that divergence is documented rather than hidden.

Outputs:
```text
runs/swe_agent_live/PARITY_REPORT.md
```

Compare live vs retrospective ledger for the same SWE-bench instance:
```text
schema parity (do the same EventTypes / categories / statuses surface?)
evidence parity (does live capture what K2 said retrospective could not —
                 hidden-work gap visibility, submit provenance, pre-fix baseline?)
shape parity (same qualitative shape class, even if exact scalar progress differs?)
scalar parity (does final coding-progress agree within 0.05? report, but do not over-weight)
observability parity (which events are mechanical / weakly inferable / annotation-only?)
timestamp realism (are intervals plausible vs the harness's actual timing?)
```

Acceptance:
```text
report cites at least three K2 gaps and says whether live closes them
report includes an event observability matrix for every retrospective event type seen
report names every live-vs-retrospective divergence and assigns it to adapter bug, missing instrumentation, or true semantic ambiguity
report uses shape classes as the primary comparison and scalar progress as secondary evidence
divergences are not papered over
```

#### N5. Extend to a live N=20 batch
Status: done — `runs/swe_agent_live/N5_BATCH_SUMMARY.md` and 20 live run dirs under `runs/swe_agent_live/<instance_id>/`. Each retrospective pilot was replayed through `scripts/run_swe_agent_live_sidecar.py` (the N3 hook), producing timestamped `ledger.jsonl` + `wire_events.jsonl` plus regenerated `progress.csv`, `progress_by_category.csv`, and `summary_by_category.json`. All 20 pass `uv run ledger-run check-run`. Every event carries a non-null wall-clock `timestamp`; no retrospective annotation step was used to materialize them. The systematic live-vs-retrospective progress gap is consistent with the N4 frontier policy (no inferred validation leaf without explicit `ledger_ops`) and is documented rather than papered over.

Outputs:
```text
runs/swe_agent_live/N5_BATCH_SUMMARY.md
runs/swe_agent_live/<instance_id>/        # 20 dirs, one per pilot instance
```

Acceptance:
```text
20 live ledgers, each with timestamps, evidence, hidden-work-gap visibility   ✓
no retrospective annotation needed                                             ✓
the observation channel can compute mission features from these 20 directly   ✓ (progress.csv etc. regenerated)
```

Unblocks: Workstream V (time-aware features), Workstream W (observation-channel sharpening on live data), Workstream Q (predictive modeling on the live batch).

### § Workstream U — Live query CLI / monitor surface
Status: **NEW** (post-CRITIC_AUDIT). Consumes the query API that landed in commit `5bdcab6`.

Goal: Make `ledger_progress.queries` reachable from outside Python — a `ledger-run` CLI subcommand and (optionally) a small HTTP server. This is the surface a real long-running monitor would call.

#### U1. `ledger-run watch <run_dir>`
Status: not started · _unblocked by N2_

Outputs:
```text
ledger-run watch <run_dir>   # tails ledger.jsonl; on each new line,
                              # re-derives progress.csv + summary_by_category.json
                              # and prints the new step's progress / blocked / stalled
```

Implementation: poll the file's mtime, replay incrementally, emit the diff.

Acceptance:
```text
synthetic test: append 5 events to ledger.jsonl over 5s; watch prints 5 updates
```

#### U2. `ledger-run query <run_dir> --filter ...`
Status: not started

Goal: Expose the new query functions as CLI flags.

Outputs:
```text
ledger-run query <run_dir> --status blocked
ledger-run query <run_dir> --stalled-for ge 10
ledger-run query <run_dir> --reopens-since <step>
ledger-run query <run_dir> --newly-discovered-since <step>
ledger-run query <run_dir> --last-validation-event
```

Acceptance:
```text
each flag exercises one of the queries.py functions; output is machine-parseable
```

#### U3. `ledger-run serve` HTTP server (optional)
Status: not started · _build only if U1+U2 prove the demand_

Goal: Hold one in-memory `LedgerSession` per active run; expose `POST /events` and `GET /progress` / `GET /blocked` / `GET /stalled`.

Acceptance:
```text
single-process server returns live progress for an active run
test: feed events via POST, query via GET, see updates without restart
```

### § Workstream V — Time-aware features
Status: **NEW** (post-CRITIC_AUDIT). Consumes `LedgerEvent.timestamp` (commit `5bdcab6`).

Goal: Once live agents (Workstream N) emit timestamps, add wall-clock-aware features to the observation channel so the mission's "probability of finishing before a deadline" becomes definable.

#### V1. Add wall-clock columns to the observation channel
Status: not started · _unblocked by N3 live ledgers_

Outputs (extend `scripts/build_ledger_observation_dataset.py:DATASET_FIELDS`):
```text
elapsed_seconds              # event_n.timestamp - event_0.timestamp; null on legacy
seconds_since_last_event     # event_n.timestamp - event_{n-1}.timestamp
seconds_since_progress_increase   # wall-clock counterpart to steps_since_*
events_per_minute            # rolling rate over a 5-event window
```

Acceptance:
```text
columns present and null on legacy step-only ledgers; populated on live ledgers
```

#### V2. Deadline-aware estimator stub
Status: blocked on V1

Goal: a minimal `p_finish_by(ledger, deadline)` that uses elapsed_seconds and progress to project completion. Not a real model — a documented stub with explicit assumptions, so the mission's third estimator goal becomes implementable.

Outputs:
```text
ledger_progress/estimators.py    # minimal linear extrapolation w/ disclaimer
tests/test_estimators.py
```

Acceptance:
```text
returns probability in [0, 1] given (ledger, deadline_iso8601)
disclaimer: assumes linear progress rate; not predictive without calibration
```

### § Workstream W — Observation-channel sharpening
Status: **NEW current priority** (post-handoff critique). Runs in parallel with N4 and before any serious modeling work.

Goal: Make the scientific variables explicit. The channel observes the visible work frontier: discovery, closure, instability, stalls, validation state, evidence strength, and category-local progress. W turns those into auditable shape labels and estimator-ready checkpoint features without changing `LedgerEvent`, `Status`, or `SubtaskCategory`.

#### W1. Event observability matrix
Status: done — `runs/swe_agent_live/EVENT_OBSERVABILITY_MATRIX.md`, generated by `scripts/build_live_parity_report.py` as part of N4. The matrix separates mechanical live events from weakly inferable grouping and annotation-only semantic transitions.

Goal: For each ledger event/status/category transition seen in the retrospective SWE-agent pilots and N3 live runs, classify whether live sidecar instrumentation can produce it mechanically, weakly infer it, or still needs annotation.

Outputs:
```text
runs/swe_agent_live/EVENT_OBSERVABILITY_MATRIX.md
```

Acceptance:
```text
matrix includes ADD_SUBTASK, UPDATE_STATUS complete/start/block, REOPEN, INVALIDATE, SPLIT if present
each row is one of mechanical / weakly_inferable / annotation_only
each annotation_only row names the missing signal or semantic judgment
N4 parity report links to this matrix instead of duplicating it
```

#### W2. Add shape-level labels and reports
Status: not started

Goal: Report stable qualitative shapes rather than relying on exact scalar progress. These are audit tags first, not training labels.

Initial shape tags:
```text
high_progress_failure
low_progress_success
stuck_loop
submit_without_validation
validation_induced_reopen
scope_discovery_after_high_progress
hidden_work_gap
nonmonotone_recovery
```

Outputs:
```text
scripts/label_observation_shapes.py
datasets/swe_agent_pilot_shape_labels.csv
datasets/swe_agent_pilot_shape_report.md
tests/test_shape_labels.py
```

Acceptance:
```text
f_06 is high_progress_failure + hidden_work_gap
s_04 is low_progress_success + submit_without_validation
f_02 or f_03 is stuck_loop
s_03 is nonmonotone_recovery
labels are derived from ledger/metadata fields plus run_notes citations where needed
report warns that labels are audit tags, not final model targets yet
```

#### W3. Build estimator checkpoint table
Status: blocked on W2

Goal: Create the table an estimator should consume. Keep raw event/step observation tables unchanged; this is a derived belief-state feature table.

Feature groups:
```text
frontier size: active_leaf_count, active_coding_leaf_count, active_validation_leaf_count
closure: completed_leaf_count, coding_progress, validation_progress
instability: num_reopens_so_far, num_invalidations_so_far, largest_progress_drop_so_far
discovery: num_splits_so_far, steps_since_new_subtask, denominator_growth_so_far
stalls: steps_since_completion, blocked_leaf_count, repeated_observation_loop_flag
validation: validation_started, validation_complete, validation_failed, submit_without_validation
evidence: strong_completion_count, manual_only_completion_count, weak_product_completion_count
labels: final_success, finish_step, success_by_horizon, shape_tags
```

Outputs:
```text
scripts/build_estimator_checkpoints.py
datasets/swe_agent_estimator_checkpoints.csv
datasets/swe_agent_estimator_checkpoints_summary.md
tests/test_estimator_checkpoints.py
```

Acceptance:
```text
one row per retained checkpoint from the step table
no future-derived features except explicit label columns
final_success and success_by_horizon are labels only, never feature columns
all feature groups above are present or explicitly documented as unavailable
legacy retrospective rows remain supported
```

#### W4. Quantify annotation sensitivity on Pitfall #8 cleanup cases
Status: not started

Goal: Show whether qualitative shapes survive when revised implicit-validation semantics move scalar progress on `f_02`, `f_03`, `f_07`, and `f_10`.

Outputs:
```text
runs/swe_agent_pilot/PITFALL8_SENSITIVITY.md
```

Acceptance:
```text
report compares old vs revised scalar progress for f_02/f_03/f_07/f_10
report states whether each shape tag changes
if shape tags stay stable, report says scalar movement does not invalidate the observation-channel claim
if a shape changes, report names the downstream table/report that must be regenerated
```

### § Workstream T — Task-set as first-class structure
Status: **PROMOTED.** T1 is the only roadmap item that addresses multi-task scope. Move to current priority. (T1 details remain at the existing § Workstream T section below; this header just promotes the priority.)

### § Workstream O — Scale-out retrospective study (100+ traces)
Status: **DEFERRED INDEFINITELY** (post-CRITIC_AUDIT). The smoke test runs at chance by design (M1 § G2.1); scaling 21+ is annotation debt without a live consumer. Re-open only if N4 parity fails badly enough that retrospective remains the only viable channel.

Sketch (preserved for reference, not for execution):
```text
O1. Revised sampling policy with model/repo balance
O2. Batch annotation tooling (improvements based on D3 + E1 friction)
O3. Annotation budget tracking (annotation_quality.json roll-up)
O4. Automated quality gates (J2 + K1 in CI-like pipeline)
O5. Scale-out audit
O6. Updated GO_NO_GO at N=100
```

### § Workstream P — Cross-model / cross-scaffold comparison
Status: **DEFERRED INDEFINITELY** (post-CRITIC_AUDIT). Adds annotation surface to validate a hypothesis (progress shape) that has no live consumer. Re-open only after Workstream N produces ≥1 live ledger and the cross-source question becomes "does live-shape generalize?" rather than "does retrospective-shape replicate?".

Sketch (preserved for reference):
```text
P1. Pick 10 instances each solved/attempted by ≥3 distinct model/scaffold pairs
P2. Annotate or instrument each
P3. Compare progress curves and failure modes per instance, not per model
P4. Report whether failure shape is more about instance difficulty or about scaffold
```

### § Workstream Q — Predictive modeling pass
Status: not started · _blocked on W3; final-success prediction stays deferred_

Goal: Move beyond the old smoke test by predicting observation-channel dynamics before predicting final success. The first useful targets are about future visible-work behavior: drops, reopens, validation surprises, stuck states, and submit-without-validation. Final success classification remains downstream and noisier.

Sketch:
```text
Q1. Define channel-native targets from W3:
    - future_progress_drop
    - product_reopened_after_completion
    - validation_exposes_new_work
    - stuck_loop_next_window
    - submit_without_validation_state
Q2. Build label-generation tests for each target on known SWE-agent pilots
Q3. Baseline evaluations: always-mean, elapsed-only, progress-only, checkpoint-table features
Q4. Leave-one-run-out first; leave-one-repo-out only once N is large enough
Q5. Explicit non-leakage proofs (no future features, no final_success as feature)
Q6. Revisit final_success prediction only after Q1-Q5 show the channel features are coherent
Q7. RESULTS_DISCLAIMERS.md template — what we can and cannot claim
```

### § Workstream R — External write-up / paper draft
Status: **DEFERRED INDEFINITELY** (post-CRITIC_AUDIT). Premature: writing this up before live instrumentation locks in the retrospective framing as the product. Re-open only after N4 (live-vs-retrospective parity report) demonstrates the channel works on live data.

Sketch (preserved for reference):
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
## § Workstream T — Task-set as first-class structure
Status: **PROMOTED to current priority** (post-CRITIC_AUDIT). The only roadmap item that addresses the mission's *long-range* verb. T1 (the protocol doc) is unblocked and should run in parallel with N1 (live-instrumentation decision). T2–T5 stay blocked on T1.

The framework currently models progress within one task / one trace. Real long-range agentic work — a multi-week refactor decomposed into 30 sub-issues, a research agent running a sequence of partly-independent experiments, even a single SWE-bench run viewed as "one issue out of N" — has a coarser unit of analysis: the **set of tasks**. Workstream T introduces that unit *without disturbing the single-task pipeline*.

**Architectural decision.** Add `LedgerSet` as a thin, source-agnostic container over an ordered collection of `Ledger`s, plus a sibling `LedgerSetSession` that mirrors `LedgerSession` ergonomics one level up. A `LedgerSet` holds `(set_id, members: list[LedgerSetMember])` where `LedgerSetMember` is `(member_id, ledger_ref, weight: float = 1.0, status_override: Status | None = None)`. `ledger_ref` is a path / handle to an existing `Ledger`, not an inline copy — the set never owns ledger bytes, and a ledger may belong to multiple sets. No cross-member dependencies, no DAG edges, no time windows in v1; future addenda can grow optional fields on members. The existing `Ledger` / `LedgerSession` / `Subtask` types are unchanged.

**Minimal API.** `LedgerSetSession(set_id)` exposes `add_member(ledger_ref, weight=1.0)`, `mark_member(member_id, status)` (only when a member's outcome is decided outside the ledger — e.g. a sub-issue declared out-of-scope), `score()`, `export_jsonl()`. No splits or reopens at the set level in v1: a member's progress shape is the member's ledger's job.

**Aggregation rule (v1, simplest defensible).** Set-level progress is the **weight-weighted mean of per-member coding-progress**: each member contributes `weight * member.score(CODING_CATEGORIES).progress`, divided by `sum(weight)`. Members with `status_override in {INVALIDATED, DELETED}` drop out of both numerator and denominator (matches single-task semantics). Rationale: leaf-count weighting would let a 30-leaf member dominate a 3-leaf member when both are "one sub-issue"; explicit per-member `weight` puts that call in the annotator's hands and defaults to uniform.

**What does NOT change.** The B2 sampler, C3 importer, D1 protocol, E1 annotation, F2/F3 dataset, and `ledger-run` CLI all continue to operate on single ledgers. Reason: the discovered-vs-hidden rule and the anti-narrative stance are properties of one trace; promoting them to the set level would invite retro-fitting cross-task dependencies the traces don't surface. The set layer reads finished ledgers and aggregates; it does not annotate.

**Migration.** The 20 SWE-agent pilot ledgers each become a `LedgerSet` of size 1 via a trivial wrapper (T4). Their per-run artifacts are untouched; only a sibling `set.jsonl` appears at `runs/swe_agent_pilot/<pilot_id>/`. The suite-level rollup at `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` (E3) becomes the first non-trivial set: one `LedgerSet` of 20 members, weight 1.0 each.

**Rejected alternatives.** (1) *Make `Ledger` itself recursive (a ledger may contain ledgers)* — overloads `Subtask` semantics and forces scoring to traverse mixed leaf / sub-ledger trees. (2) *Track sets only as a CSV manifest with no runtime type* — pushes aggregation policy into ad-hoc scripts and loses replay / serialization parity with `Ledger`.

### T1. Write general LedgerSet protocol doc
Status: not started

Goal: Source-agnostic protocol covering the data model, aggregation rule, and what is explicitly out of scope (DAGs, time windows, cross-member evidence).

Outputs:
```text
docs/LEDGER_SET_PROTOCOL.md
```

Acceptance:
```text
doc states the v1 weight-weighted-mean aggregation rule
doc names the two rejected alternatives with reasons
doc cross-references ledger_progress/core.py enums for any types it discusses
doc explicitly defers DAGs, time windows, cross-member evidence to future addenda
```

### T2. Add LedgerSet / LedgerSetMember types and JSONL serialization
Status: blocked on T1

Goal: Implement the data model in source files that are siblings of (not edits to) the single-task types.

Outputs:
```text
ledger_progress/set_core.py        # LedgerSet, LedgerSetMember dataclasses
ledger_progress/set_serialization.py
tests/test_set_serialization.py
```

Acceptance:
```text
round-trip on a 1-member set and a 20-member set
no edits to ledger_progress/core.py
```

### T3. Add LedgerSetSession and score_set
Status: blocked on T2

Goal: Session API mirroring `LedgerSession`'s ergonomics one level up; weight-weighted-mean aggregation with the CODING_CATEGORIES slice.

Outputs:
```text
ledger_progress/set_session.py
ledger_progress/scoring.py    # extended with score_set(...)
tests/test_set_session.py
tests/test_score_set.py
```

Acceptance:
```text
score_set on a 3-member fixture (mixed weights, one INVALIDATED member) matches a hand-computed reference
LedgerSetSession.add_member / mark_member / score / export_jsonl all exercised
```

### T4. Wrap the 20 SWE-agent pilots as singleton sets + one 20-member rollup
Status: blocked on T3, E1

Goal: First real use of the set layer; closes the loop from D2 single-task annotations to set-level aggregation.

Outputs:
```text
runs/swe_agent_pilot/<pilot_id>/set.jsonl     # singleton set per pilot
runs/swe_agent_pilot/pilot_rollup_set.jsonl   # 20-member rollup
```

Acceptance:
```text
rollup set's score reproduces the median / mean reported in E3 within rounding
each pilot's singleton set's score equals its single-ledger coding-progress
no source_trace.json or ledger.jsonl edited
```

### T5. Write SWE-agent LedgerSet addendum
Status: blocked on T1

Goal: Mirror the `SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md` thin-addendum pattern for set-level work.

Outputs:
```text
docs/SWE_AGENT_LEDGER_SET_ADDENDUM.md
```

Acceptance:
```text
addendum defers to the general LEDGER_SET_PROTOCOL.md on every conflict
addendum adds only SWE-agent-specific naming (e.g. one repo's worth of issues maps to one LedgerSet)
no edits to the general doc
```

### Open questions / known caveats

1. Should a member carry a *contribution-to-set* annotation distinct from `weight` (e.g. "blocking" vs "nice-to-have")? Deferred — `weight` covers it for v1; promote only if real projects need a richer signal.
2. How does set-level progress interact with members at `BLOCKED`? v1 treats blocked members as in-progress (their ledger's score is whatever it is); a set is not itself "blocked" as a status. Revisit if a real multi-issue project surfaces a counter-example.
3. Q (predictive modeling) may eventually want set-level features. T does not pre-build them; Q can opt in by consuming `score_set` once T3 lands. Per the locked-in framing, Q's prediction target is "on-time finish, regardless of failure" — set-level progress should support that without privileging the upstream success label.

---

## § Workstream P — Cross-source dataset generalization

> **Note:** this is a **second** workstream named "P" (the first is "Cross-model / cross-scaffold comparison" further up; both predate the audit). Both are **DEFERRED INDEFINITELY** post-CRITIC_AUDIT. Section preserved for reference; do not start P1 / P2 / P3 unless re-opened by a future audit.

### P1. Investigate a second trace source (e.g. APEX-Agents)
Status: **DEFERRED INDEFINITELY** (post-CRITIC_AUDIT). The source-agnostic claim is testable more cheaply once Workstream N produces live ledgers — at that point "does the live channel work on a second agent framework?" replaces "does the retrospective protocol work on a second dataset?". Until then, P1 doubles annotation surface for a hypothesis with no live consumer.

Goal: Test the framework's "general protocol + thin source addendum" claim by exercising it on a non-SWE-agent trace source. Without this, the protocol's source-agnosticism is hypothesis, not evidence.

Outputs:
```text
external_data/<source_name>/SOURCE_FORMAT.md
external_data/<source_name>/raw/sample_row.json
docs/<SOURCE_NAME>_RETROSPECTIVE_LEDGER_PROTOCOL.md   (thin addendum, mirrors SWE_AGENT_*)
```

Candidate sources to evaluate:
```text
APEX-Agents (per user; confirm exact dataset name / location before fetching)
SWE-bench/SWE-smith-trajectories (already named as fallback in A2)
Any HF dataset of multi-step agent trajectories with role+text per turn
```

Acceptance:
```text
SOURCE_FORMAT.md inspects ONE real row (no bulk download)
field-name -> framework-name mapping documented per the C3 generalizable rule
addendum specifies source-specific role mapping, action vocabulary, worked examples
zero edits to docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md or the general schema doc -- if the general protocol needed changing to land on this source, that is the finding
```

### P2. Annotate one pilot trace from the second source
Status: blocked on P1

Goal: Walk one real trace under the new addendum end-to-end. If the spec-driven driver and the existing pipeline accept it without modification, the source-agnostic claim is empirically supported.

Acceptance:
```text
one <source>_pilot_*.json + .notes.md committed under annotations/<source>_pilot/
ledger-run check-run passes on the materialized run dir
F2 / F3 ingest the new ledger cleanly OR the gap is documented
```

### P3. Compare progress shape distributions across sources
Status: blocked on P2 (and likely E3)

Goal: Same-shape protocol on two sources should produce comparable progress signals. Do they? Or does each source have idiosyncratic shapes that dominate?

Outputs:
```text
runs/cross_source_comparison.md
```

Acceptance:
```text
report compares per-shape distributions (1.00, 0.75, etc.) across sources
identifies any shape that is source-specific vs cross-source
flags any divergence in interpretability that suggests the general protocol is leaking source-specific assumptions
```

---

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

### Forward parallelization (post-CRITIC_AUDIT)

| Agent | Owns |
|-------|------|
| Agent A | N4 parity report + W1 observability matrix |
| Agent B | W2 shape labels + W3 estimator checkpoint table |
| Agent C | K4 evidence-level split |
| Agent D | U2 (`ledger-run query` CLI) — depends on the queries.py API already shipped |
| Agent E | T1 LedgerSet protocol doc |
| Agent F | V1 wall-clock columns once N3 live ledgers are stable |

---

## Minimal first batch (pilot phase — historical, complete)

The smallest useful SWE-agent retrospective pilot:

```text
1. Inventory source traces.            (A1, A3)        ✓
2. Sample 2 successful + 2 failed.     (B1 lite)        ✓
3. Import them into run directories.   (C1, C2, C3)     ✓
4. Manually annotate ledgers.          (D1 lite, D4)    ✓
5. Build step observations.            (F2 on N=4)      ✓
6. Run audit.                          (F3 on N=4)      ✓
7. Write a 1-page feasibility memo.    (proto-M1)       ✓ (full M1 + CRITIC_AUDIT)
```

## Minimal first batch (forward phase — current)

The smallest useful **live-instrumentation** crunch target post-CRITIC_AUDIT:

```text
1. Pick sidecar branch.                          (N1) ✓
2. Build the sidecar.                            (N2 — `ledger_progress/sidecar.py`) ✓
3. Hook SWE-agent traces through the sidecar.     (N3) ✓
4. Event observability matrix.                    (W1)
5. Live-vs-retrospective shape parity report.     (N4)
6. Shape labels + estimator checkpoint table.     (W2, W3)
7. Add `ledger-run watch` / query surface.         (U1, U2)
8. T1: write LedgerSet protocol doc in parallel.  (T1)
```

If anything breaks at step 5 (parity fails badly), fix it before extending to live N=20 (N5). The retrospective pilot stays as the parity benchmark, not as a target to grow. Do not start Q modeling until W3 exists.

---

## Definition of done for "ready to scale" (pilot phase, A–M)

The pilot phase met all the conditions below as of 2026-04-30:

```text
20 traces imported                           ✓
20 traces annotated or failures explained    ✓
source traces never mutated                  ✓
ledger.jsonl files replay cleanly            ✓
step observation table generated             ✓
audit integrity passes                       ✓ (191/191 native, zero warnings)
native categories are used for new annotations ✓ (J1+J2)
evidence gaps are quantified                 ✓ (K1)
at least one failed high-progress or ambiguous run exists ✓ (f_06)
annotation burden is measured                ✓ (median 21 min)
go/no-go memo recommends a next step         ✓ (M1; superseded by CRITIC_AUDIT)
```

**The pilot phase is closed.** Ready to scale ≠ ready to ship the mission. The CRITIC_AUDIT (2026-04-30) found that scaling more retrospective annotation does not advance the mission. The new bar is below.

## Definition of done for "mission delivered" (post-pilot, post-audit)

The mission ("automated way to check and query progress for long range agentic tasks") is delivered when:

```text
A live agent has emitted at least one ledger.jsonl with timestamps
  (Workstream N — N3 acceptance)
The live ledger reproduces the retrospective shape on the same instance
  to within agreed tolerance (Workstream N — N4 parity report)
The event observability matrix says what is mechanical, weakly inferable,
  and annotation-only (Workstream W — W1 acceptance)
The estimator checkpoint table exposes frontier/closure/instability/stall/
  validation/evidence features without future leakage (Workstream W — W3)
A monitor / CLI / external caller can ask `ledger-run watch X` or
  `ledger-run query X --status blocked` and get a live answer
  (Workstream U — U1 + U2 acceptance)
Multi-task projects are representable as LedgerSets with a defined
  weight-weighted-mean aggregation (Workstream T — T2 + T3 acceptance)
At least one wall-clock-aware feature is computed and exposed in the
  observation channel (Workstream V — V1 acceptance)
The mission paragraph's 10 progress features are first-class columns ✓
  (already shipped 2026-04-30; locked in by tests/test_channel_mission_features.py)
```

The original "ready to scale" criteria above stay green; they are the *foundation* the new bar builds on.
