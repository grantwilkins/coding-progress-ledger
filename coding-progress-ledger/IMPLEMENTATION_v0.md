# IMPLEMENTATION v0: SWE-agent Methodology, Source, and Validation Overview

This document describes the SWE-agent work in this repository at the level a
new reader needs in order to understand the methodology, source provenance, and
validation surface. It intentionally does not present this as a predictive
performance result. The work here is a data and methodology pipeline:

```text
upstream SWE-agent trajectory
  -> deterministic inventory
  -> deterministic balanced pilot sample
  -> local pilot cache and byte-preserving run copy
  -> normalized trace schema
  -> retrospective progress-ledger annotation
  -> replayed progress artifacts
  -> event/step observation datasets
  -> audits, agreement checks, and smoke tests
```

The central methodological claim is narrow: a deterministic append-only ledger
can represent visible progress through real SWE-agent traces while keeping
progress separate from final success. The ledger records discovered work, not
all hidden work required to solve an issue. That distinction explains why a
failed run can legitimately end at full progress and why a successful run can
legitimately end below full progress.

## 1. Repository Orientation

The core package is `ledger_progress/`. It implements the generic ledger model,
not SWE-agent-specific logic.

The SWE-agent-specific work is spread across four areas:

| Area | Purpose |
| --- | --- |
| `external_data/swe_agent/` | Source-format documentation, immutable raw-data policy, inventory and sample manifests. |
| `scripts/*swe_agent*` and related dataset scripts | Inventory, sampling, trace normalization, run import, annotation materialization, audits. |
| `annotations/swe_agent_pilot*/` | Spec files and notes used to materialize retrospective ledgers. |
| `datasets/`, `runs/swe_agent_pilot/`, and `runs/swe_agent_pilot_reannotation/` reports | Derived observation tables, audit reports, agreement reports, smoke-test reports, and synthesis memos. |

The run directories under `runs/swe_agent_pilot/` are the materialized
per-pilot artifacts. Large upstream raw traces and pilot cache files are kept
out of git by policy; the committed source documentation and manifests are the
auditable pointers back to the source rows.

## 2. Core Ledger Semantics

The type system lives in `ledger_progress/core.py`.

The ledger is an append-only event log. The event log is the source of truth;
derived CSVs and summaries are regenerated from it.

The event model has three important enums:

| Enum | Values |
| --- | --- |
| `Status` | `not_started`, `in_progress`, `blocked`, `complete`, `invalidated`, `deleted` |
| `EventType` | `init`, `add_subtask`, `update_status`, `add_evidence`, `split_subtask`, `reopen_subtask`, `invalidate_subtask`, `delete_subtask` |
| `SubtaskCategory` | `product`, `validation`, `investigation`, `environment`, `artifact`, `documentation` |

`LedgerEvent` currently carries:

```text
step
event_type
subtask_id
payload
reason
timestamp
```

The public helper API is `ledger_progress/session.py:LedgerSession`. The
annotation scripts use this helper rather than manually writing event JSON. The
main operations are `add`, `start`, `complete`, `block`, `reopen`,
`invalidate`, `split`, `score`, and export methods.

Progress scoring lives in `ledger_progress/scoring.py`:

```text
progress = complete active leaf weight / total active leaf weight
```

This has several load-bearing consequences:

- Only leaf subtasks count. If a parent is split into children, the parent is no
  longer in the active denominator.
- `invalidated` and `deleted` subtasks are excluded from active work.
- Completed subtasks require evidence.
- New discovered work can lower progress because the active denominator grows.
- Reopened completed work can lower progress because completed weight drops.
- Progress is not a final-success probability.

For coding-oriented progress, the repository uses
`ledger_progress/queries.py:CODING_CATEGORIES`:

```python
(SubtaskCategory.PRODUCT, SubtaskCategory.VALIDATION, SubtaskCategory.INVESTIGATION)
```

This is why most SWE-agent reports distinguish `coding_progress` from
`overall_progress`. `ARTIFACT`, `ENVIRONMENT`, and `DOCUMENTATION` are still
real categories, but they are not part of the default coding-progress curve.

## 3. Source Provenance

The chosen upstream source is documented in
`external_data/swe_agent/SOURCE_FORMAT.md`.

The primary source is:

```text
nebius/SWE-agent-trajectories
```

It is accessed with Hugging Face streaming:

```python
from datasets import load_dataset
ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
```

The source-format document was written from inspection of a real decoded row,
not from memory or inferred documentation. The top-level upstream fields used by
the pilot are:

| Upstream field | Role in this repo |
| --- | --- |
| `instance_id` | SWE-bench-style issue identifier. |
| `model_name` | Agent model/scaffold identifier. |
| `target` | Upstream success/failure label; mirrored as `final_success`. |
| `trajectory` | Ordered SWE-agent conversation/tool trajectory. |
| `exit_status` | Upstream termination status. |
| `generated_patch` | Final patch artifact; imported as `final_diff.patch`. |
| `eval_logs` | Upstream evaluation output; imported as `test_output.txt`. |

The upstream `trajectory` entries use roles such as `system`, `user`, and `ai`.
The normalizer maps these to the internal role vocabulary described below.

The raw-data policy is documented in `external_data/swe_agent/README.md`:

- `external_data/swe_agent/raw/` is immutable.
- Raw traces are out of scope for git.
- Manifest and summary artifacts are committed because they are small derived
  artifacts.
- Raw traces should not be redistributed from this repo.

License and usage notes are recorded in `SOURCE_FORMAT.md`. The pilot treats the
raw trajectory data as internal/research-use input and shares manifests and
summaries rather than raw trajectory dumps.

## 4. Inventory Methodology

The inventory builder is `scripts/swe_agent_inventory.py`.

Its job is to stream upstream rows and emit a deterministic manifest without
retaining or rewriting full trajectory content. The manifest columns are:

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

Important implementation details:

- The script streams rows and stores only per-row metadata.
- It does not write into `external_data/swe_agent/raw/`.
- It parses repo and issue from SWE-bench-style `instance_id` values by
  splitting on the final dash, then splitting owner/repo on `__`.
- It records parse warnings in `parse_status` and `parse_error` instead of
  silently dropping rows.
- It writes booleans as literal `True` / `False`, missing booleans as blank
  cells, and uses LF line endings.
- It sorts before writing so the output is byte-deterministic.

The committed inventory summary is
`external_data/swe_agent/manifests/swe_agent_inventory_summary.md`.

The key corpus facts from that summary are:

| Metric | Value |
| --- | ---: |
| Total rows | 80,036 |
| Usable trajectory rows | 80,036 |
| Rows with success label | 80,036 |
| Successes | 13,389 |
| Failures | 66,647 |
| Patch available | 70,742 |
| Eval log available | 70,639 |
| `parse_status=ok` | 80,036 |

The inventory also surfaced an important sampling issue: the dataset has many
duplicate trajectories for the same underlying issue/model pair. It contains
80,036 rows but only 3,591 unique `instance_id` values and 4,219 unique
`(instance_id, model_name)` pairs. The pilot therefore could not sample
directly at row level without risking repeated versions of the same underlying
task.

## 5. Pilot Sampling Methodology

The sampling policy is written before selection in
`external_data/swe_agent/PILOT_SAMPLING_POLICY.md`.

The deterministic sampler is `scripts/sample_swe_agent_pilot.py`.

The canonical pilot target is:

```text
10 successful traces
10 failed traces
20 total traces
seed = 0
model_name = swe-agent-llama-70b
```

The strict inclusion criteria are:

- `parse_status == "ok"`
- trajectory present
- final success label present
- non-empty patch present
- non-empty eval log present
- `trajectory_length >= 10`
- `model_name == "swe-agent-llama-70b"`

The dedupe rule is by `instance_id`, not by raw row. When multiple rows survive
for the same `instance_id`, the sampler keeps the lowest
`raw_path_or_dataset_index`. This gives one trajectory per underlying issue in
the pilot.

The sampler is deterministic by construction:

- It parses booleans strictly from the literal strings `True` and `False`.
- It sorts eligible pools before sampling.
- It uses `random.Random(seed)`, not module-global randomness.
- It samples successes and failures with fixed target sizes.
- It assigns pilot IDs after sampling by sorted `instance_id`.
- It writes a fixed CSV header and LF line endings.

The output is
`external_data/swe_agent/manifests/swe_agent_pilot_sample.csv`.

The audit is
`external_data/swe_agent/manifests/swe_agent_pilot_sample_summary.md`.

The selected sample has:

| Property | Value |
| --- | ---: |
| Total selected | 20 |
| Successes | 10 |
| Failures | 10 |
| Model | 20 / 20 `swe-agent-llama-70b` |
| Distinct repositories | 20 |
| Patch available | 20 / 20 |
| Eval log available | 20 / 20 |
| Fallback level used | none |

The pilot IDs are stable names:

```text
swe_agent_pilot_f_01 ... swe_agent_pilot_f_10
swe_agent_pilot_s_01 ... swe_agent_pilot_s_10
```

Failures sort before successes in the sample CSV because the output is sorted
by pilot ID.

## 6. Raw Cache and Run Directory Import

The pilot cache populator is `scripts/populate_swe_agent_pilot_cache.py`.

Its role is deliberately separate from the importer:

- It streams the upstream dataset.
- It matches rows by `raw_path_or_dataset_index`.
- It validates that the streamed row matches the sample CSV's `instance_id` and
  `model_name`.
- It writes one JSON row per pilot into a local cache directory.

The importer is `scripts/import_swe_agent_trace.py`.

The importer is offline and deterministic. It assumes one cached raw JSON row
exists per pilot. It then materializes a framework-shaped run directory without
creating a ledger. This separation matters: import is source preservation and
normalization; annotation is a later, explicit methodology step.

The byte-preservation guarantee is relative to the local pilot cache:
`source_trace.json` is copied byte-for-byte from `<pilot_cache>/<pilot_id>.json`.
The cache file itself is a JSON serialization of a decoded Hugging Face row, not
a claim about preserving upstream parquet transport bytes.

Each imported run directory contains:

| File | Meaning |
| --- | --- |
| `task.md` | Issue text extracted from the leading environment turn. |
| `source_trace.json` | Byte-equivalent copy of the cached upstream row. |
| `normalized_trace.json` | Internal normalized trace schema. |
| `trajectory_summary.md` | Human-readable per-step summary for annotators. |
| `final_diff.patch` | Upstream `generated_patch`. |
| `test_output.txt` | Upstream `eval_logs`, renamed to the framework-standard artifact name. |
| `run_notes.md` | Annotation notes template before annotation; extended notes after annotation. |
| `source_metadata.json` | Pilot metadata, including authoritative upstream label provenance. |

The importer verifies that all pre-annotation artifacts exist. If the sample CSV
says a patch or eval log is available, the corresponding imported artifact must
be non-empty. In `--verify-only` mode it checks already materialized run
directories without touching the raw cache.

The importer deliberately rejects unexpected pre-annotation `ledger.jsonl`
files. A run should not have a ledger until the annotation workstream creates
one.

## 7. Normalized Trace Schema

The schema contract is documented in `docs/SWE_AGENT_TRACE_SCHEMA.md`.

The implementation is `scripts/normalize_swe_agent_trace.py`.

The normalized trace has `schema_version = 1` and these top-level fields:

```text
schema_version
source
instance_id
model_name
exit_status
final_success
trajectory_length
issue_text
system_prompt
events
raw_metadata
```

Each normalized event has:

```text
step_index
role
thought
action
observation
tool_name
command
files_touched
timestamp
raw
```

Role mapping is source-specific but lossless because the upstream entry remains
under `raw`:

| Upstream role | Internal role |
| --- | --- |
| `system` | `system` |
| `ai` | `assistant` |
| first non-system `user` | `environment` |
| later `user` | `tool` |
| anything else | `unknown` |

Assistant turns in the upstream SWE-agent data usually contain reasoning text
plus a fenced command. The normalizer extracts the first fenced code block as
the command:

- text before the fence becomes `thought`;
- fence body becomes `command` and `action`;
- first token of `command` becomes `tool_name`;
- malformed or absent fences are recorded in `raw.parse_warnings`.

The normalizer is intentionally tolerant of partial upstream data:

- missing trajectory is treated as an empty event list;
- missing or non-bool `target` becomes `final_success = null`;
- unknown top-level keys are recorded by name in `raw_metadata`;
- every original event dict is preserved under `raw`.

The normalizer hard-fails only on corrupt input shapes that violate the schema
contract, such as a non-dict source row or a non-list trajectory.

## 8. Retrospective Annotation Methodology

The general annotation rules are in
`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`.

The SWE-agent-specific addendum is in
`docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`.

The annotations themselves live in `annotations/swe_agent_pilot/` as JSON
specs plus notes. The materialization driver is
`scripts/annotate_pilots_from_spec.py`.

The annotation methodology is built around one distinction:

```text
discovered work = work visible/nameable from the trace
hidden work     = work actually required to solve the issue but not surfaced by the trace
```

The ledger tracks discovered work. It does not pretend to know all hidden work.
This is the reason progress remains separate from outcome.

The hard annotation rules are:

- Annotate visible trace evidence only.
- Do not use the upstream success label to decide intermediate completion.
- Do not force monotonic progress.
- Every completion must cite evidence.
- Preserve uncertainty in `run_notes.md` rather than hiding it.
- Use final artifacts only in the limited way the protocol allows.

The final-artifact rule is especially important for SWE-agent:

- `final_diff.patch` is a final-state artifact, not proof that the agent
  intentionally completed every visible edit.
- `test_output.txt` is upstream post-hoc eval output. It can corroborate a
  validation subtask only when validation was surfaced in the trace.
- If the agent submitted without running validation in-trace, the validation
  leaf stays incomplete even if the post-hoc eval exists.
- Harness-forced termination is not the same as an agent-issued `submit`.

The SWE-agent addendum maps shell vocabulary to categories:

| SWE-agent action shape | Default category |
| --- | --- |
| `find_file`, `search_dir`, `search_file`, `grep`, `ls`, `open`, `goto`, scrolling | `INVESTIGATION` |
| `edit`, `create` | `PRODUCT` |
| `pytest`, `tox`, repro scripts, ad-hoc checks | `VALIDATION` |
| dependency or setup fixes | `ENVIRONMENT` |
| `submit` | `ARTIFACT` |
| docs edits only when task requires docs | `DOCUMENTATION` |

The revised protocol now says bug-fix tasks carry an implicit `VALIDATION` leaf.
If the agent never validates, that leaf remains `not_started` or `in_progress`.
This rule was tightened after inter-annotator disagreement around
submit-without-validation traces. The published 20-run v1 corpus predates full
cleanup of this rule: `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`
identifies `f_02`, `f_03`, `f_07`, and `f_10` as harness-terminated failures
whose absolute progress would likely shift downward after consistent
application. The quadrant-level conclusions are unchanged, but those absolute
values should not be treated as final high-precision measurements.

The protocol also defines stuck-loop handling. A `blocked` leaf is appropriate
when a command cycle or repeated tool-response pattern visibly shows the agent
is no longer making progress. This was refined because real SWE-agent failures
include single-command loops, two-command oscillations, and long sequences of
varied commands that all receive the same unhelpful response.

## 9. Annotation Materialization

The annotation specs are not the final data product by themselves. They are
inputs to `scripts/annotate_pilots_from_spec.py`.

The driver:

1. Reads a pilot spec.
2. Replays declared operations through `LedgerSession`.
3. Checks asserted `add` IDs against deterministic session-generated IDs.
   Omitted `add` IDs are allowed; asserted mismatches hard-fail.
4. Writes `ledger.jsonl`.
5. Regenerates derived run artifacts with the same machinery used elsewhere.
6. Writes or updates annotation quality metadata.

The supported spec operations mirror `LedgerSession`:

```text
add
start
complete
block
reopen
invalidate
split
```

The full pilot has 20 annotated runs. The summary is
`runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`.

Headline annotation facts from that summary:

| Metric | Value |
| --- | ---: |
| Pilots annotated | 20 |
| Success / failure split | 10 / 10 |
| Median subtasks per pilot | 5 |
| Total annotation time | 492 minutes |
| Median annotation time | 21 minutes |
| Median final overall progress | 0.92 |
| Median success coding-progress | 1.00 |
| Median failure coding-progress | 0.69 |
| Pilots with at least one `BLOCKED` leaf | 6 / 20 |
| Pilots with at least one `REOPEN` | 3 / 20 |

The pilot intentionally preserved shapes that are easy to erase by accident:

- high-progress failure: `swe_agent_pilot_f_06` reaches coding-progress 1.00
  while upstream `final_success=False`;
- low-progress success: `swe_agent_pilot_s_04` succeeds despite no in-trace
  validation and ends at coding-progress 0.67;
- non-monotonic runs: `s_03`, `s_05`, and `f_09` contain trace-backed reopens;
- stuck-loop failures: multiple failures end with `BLOCKED` investigation or
  product leaves.

These are not bugs in scoring. They are the intended result of separating
visible discovered work from hidden success conditions.

## 10. Derived Run Artifacts

The run-manager CLI is `ledger_progress/run_manager.py` and is exposed as
`ledger-run`.

For a materialized run, the important commands are:

```text
ledger-run export-run <run_dir>
ledger-run check-run <run_dir>
ledger-run summarize-run <run_dir>
```

`export-run` regenerates derived artifacts such as:

```text
progress.csv
progress_by_category.csv
summary_by_category.json
```

`summary_by_category.json` records category-specific scores, source ledger hash,
success metadata, and related summary values. Downstream dataset builders treat
the ledger as immutable input and raise if a build mutates `ledger.jsonl`.

The label provenance issue matters here. SWE-agent labels are authoritative
when `source_metadata.json` says:

```json
{
  "final_success_source": "source_label",
  "final_success": true
}
```

Dataset rows report this resolved provenance as `source_metadata.target`.

The builder now short-circuits to source metadata for such runs rather than
trying to infer success from `test_output.txt`. This fixed a real problem:
SWE-bench eval logs can contain misleading strings such as warnings, errors, and
passed-test text interleaved in ways that a toy pytest-output heuristic can
misread.

## 11. Observation Dataset Methodology

The dataset builder is `scripts/build_ledger_observation_dataset.py`.

It can be pointed at any runs directory. For the SWE-agent pilot, it was run
against `runs/swe_agent_pilot` to produce SWE-agent-only tables:

```text
datasets/swe_agent_pilot_observations_event.csv
datasets/swe_agent_pilot_observations_step.csv
datasets/swe_agent_pilot_observations_summary.md
```

The event table has one row per `LedgerEvent` prefix. It preserves replay
fidelity: each row answers "what would the score be if replay stopped after this
event?"

The step table keeps the final event state for each `(run_id, step)` and
recomputes deltas across retained step rows. It is the more natural table for
plotting and simple checkpoint-style modeling.

The builder emits both native and resolved category metrics:

- native metrics use the event categories exactly as serialized;
- resolved metrics can fill legacy missing categories from summaries;
- current SWE-agent rows are `native`, meaning categories are present in the
  event payloads themselves.

The main observation fields include:

```text
run_id
step
event_index
event_type
subtask_id
coding_progress
overall_progress
active/completed coding weights
active/completed overall weights
active/completed coding leaves
active/completed overall leaves
num_splits_so_far
num_reopens_so_far
num_invalidations_so_far
delta_coding_progress
delta_overall_progress
drop sources
final_success
final_success_source
native_* mirrors
category_resolution_mode
product_progress
validation_progress
investigation_progress
step event-count features
strong/manual-only completion counts
steps-since-progress/completion/subtask-added features
```

Current SWE-agent dataset summary:

| Metric | Value |
| --- | ---: |
| Runs | 20 |
| Event rows | 202 |
| Step rows | 191 |
| Successes | 10 |
| Failures | 10 |
| Category resolution | 202 event rows and 191 step rows `native` |
| Unknown success labels | 0 |
| Sanity warnings | none |

The audit in `datasets/swe_agent_pilot_observations_step_audit.md` reports:

| Check | Result |
| --- | ---: |
| Rows | 191 |
| Runs | 20 |
| Invalid progress values | 0 |
| Completed greater than active failures | 0 |
| Delta mismatches | 0 |
| First-row nonzero deltas | 0 |
| Missing identifiers | 0 |
| Invalid success metadata | 0 |
| Unknown success metadata | 0 |
| Native/resolved mismatch | none |

The step audit also reports all four success/progress quadrants:

- success + high progress;
- success + low progress;
- failure + high progress;
- failure + low progress.

That quadrant coverage is a central reason the SWE-agent pilot is more
methodologically informative than the original toy/control corpus.

## 12. Evidence Quality Methodology

Evidence strength is audited in `scripts/audit_pilot_evidence.py`, using the
classifier in `scripts/rescore_suite_by_category.py`.

The evidence audit report is `runs/swe_agent_pilot/EVIDENCE_AUDIT.md`.

The evidence classifier distinguishes strong evidence types such as:

```text
test_output
diff
file_exists
command_output
```

from manual-note-only evidence. Manual evidence is not illegal; the audit
measures how often the annotation depends on it.

The K1 audit found:

| Metric | Value |
| --- | ---: |
| Pilots audited | 20 |
| Completion events audited | 81 |
| Strong completion evidence | 51 / 81 |
| Manual-only completion evidence | 30 / 81 |

Weak evidence is concentrated in `PRODUCT` completions. This is expected because
many product completions are short edit acknowledgments that are clear to a
human annotator but do not match the current strong-evidence classifier. The
report treats weak evidence as a signal and follow-up opportunity, not as a
replay failure.

## 13. Inter-Annotator Methodology

The first 20-pilot annotation pass is not treated as infallible. The repository
contains a second-annotator study and a post-revision gate.

The comparison report is
`runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`, with raw metrics in
`datasets/h_inter_annotator_report.md`.

The second pass covered five pilots chosen to span the load-bearing shapes:

```text
s_01: clean success
s_03: success with reopen
f_01: failure with submit-without-validation ambiguity
f_06: high-progress failure / hidden-work gap
f_03: stuck-loop failure
```

The second annotator was instructed to read protocol docs and per-pilot source
artifacts, but not the original annotations or pilot summary.

Headline agreement facts:

| Metric | Value |
| --- | ---: |
| Pairs compared | 5 |
| Mean coding-progress delta | +0.10 |
| Mean absolute coding-progress delta | 0.10 |
| Mean leaf-count delta | +1.0 |
| Same success/progress quadrant | 5 / 5 |

The most important disagreement was `f_01`: one reading added an implicit
validation leaf left incomplete, while the other treated validation as hidden
work and omitted the leaf. That disagreement motivated a protocol revision:
bug-fix tasks carry implicit validation work, and skipped validation should be
represented as incomplete progress.

The post-revision gate is documented in
`runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`.

The v3 cold pass under the revised protocol preserved quadrant agreement on all
five pilots and reproduced the intended `f_01` conclusion. It also surfaced a
useful caveat: earlier annotations had applied the implicit-validation rule
inconsistently to some harness-terminated failures. That is documented as a
cleanup recommendation rather than hidden.

The inter-annotator methodology has an explicit limitation: the annotators are
LLM passes, so their biases may be correlated. The study is best read as a test
of protocol clarity, not as a definitive human/AI annotation reliability result.

## 14. Distribution and Smoke-Test Methodology

The distribution comparison is
`datasets/observation_distribution_comparison.md`.

It compares:

- original toy/control/live runs in `datasets/ledger_observations_v0_step.csv`;
- SWE-agent pilot runs in `datasets/swe_agent_pilot_observations_step.csv`.

The headline methodological result is that SWE-agent traces are more diverse in
the ways that matter for a progress channel:

- toy/control/live populates only two success/progress quadrants;
- SWE-agent populates all four;
- SWE-agent exercises `BLOCKED` leaves;
- SWE-agent exercises investigation-sourced drops;
- SWE-agent contains high-progress failures and low-progress successes.

The completion-prediction smoke report is
`datasets/swe_agent_pilot_completion_smoke_report.md`.

The smoke test is intentionally modest. It verifies plumbing, not predictive
science.

The method is:

- input: SWE-agent step table;
- split: leave-one-run-out by `run_id`;
- estimator: deterministic binned success-rate baseline;
- feature sets:
  - `progress_only`;
  - `ledger_basic`;
  - `elapsed_only`;
- leakage exclusions:
  - `run_id` used only for grouping;
  - `final_success` used only as label;
  - source/provenance fields and final-outcome artifacts excluded from features;
  - future trace events are not used to construct checkpoint features.

The observed AUROCs are near chance. This is not a failure of the methodology.
It is expected: the framework says progress is decoupled from outcome, so a
small pass/fail predictor using progress features should not be interpreted as
a performance claim.

## 15. Broad Test Coverage

The canonical test command for this repo is:

```bash
uv run pytest
```

The SWE-agent-related test surface is broad and mostly targets invariants, not
snapshotting current outputs.

### 15.1 Inventory Tests

`tests/test_swe_agent_inventory.py` validates:

- repo/issue parsing with dashes in repository names;
- warning behavior for malformed `instance_id` values;
- preservation of `target=False` as an available false label rather than a
  missing label;
- missing target handling;
- empty trajectory handling;
- patch/eval-log availability as non-empty string checks;
- deterministic CSV writing across input-order permutations;
- stable sorting and LF line terminators.

### 15.2 Sampling Tests

`tests/test_sample_swe_agent_pilot.py` validates:

- strict boolean parsing;
- numeric parsing of dataset indices;
- dedupe keeping the lowest numeric dataset index;
- dropping blank instance IDs;
- inclusive trajectory-length threshold;
- each policy filter dropping the intended rows;
- optional model filter behavior;
- success/failure splitting only on explicit true/false labels;
- deterministic sampling under row permutation;
- pilot ID assignment independent of RNG pick order;
- output ordering and CSV header;
- fallback ladder behavior;
- seed-sensitive output changes.

These tests are important because a sampling bug could silently contaminate the
pilot with duplicates, missing labels, or non-reproducible row choices.

### 15.3 Raw Cache Tests

`tests/test_populate_swe_agent_pilot_cache.py` validates the source-to-cache
bridge:

- selected dataset indices are parsed and matched correctly;
- cached rows must match the sample CSV's `instance_id` and `model_name`;
- duplicate requested indices are rejected;
- missing selected rows are reported;
- existing cache files are not overwritten by default;
- forced overwrites are explicit;
- cache output is deterministic enough for the importer to copy forward.

These tests matter because the cache is the boundary between remote Hugging
Face streaming and the local byte-preserving import step.

### 15.4 Normalization Tests

`tests/test_normalize_swe_agent_trace.py` validates:

- assistant fenced-command parsing;
- no-fence, null-text, and unterminated-fence warnings;
- tool-name extraction;
- first user turn becoming `environment`;
- later user turns becoming `tool`;
- unknown upstream roles being preserved;
- target true/false/null behavior;
- preservation of raw upstream keys;
- recording unknown top-level keys;
- empty and missing trajectories;
- hard failures on non-dict rows and non-list trajectories;
- dense zero-based step indices;
- schema version/source fields;
- raw metadata length semantics;
- sample-row round-trip invariants;
- CLI writing of normalized trace and summary;
- summary truncation marker behavior.

### 15.5 Import Tests

`tests/test_import_swe_agent_trace.py` validates:

- creation of every pre-annotation artifact;
- byte-equivalent `source_trace.json` copies, including unusual JSON
  formatting;
- `task.md` behavior with and without issue text;
- run-notes template initialization;
- source metadata fields and false-vs-missing label preservation;
- idempotent import output;
- hard failure on missing raw cache files;
- required `--raw-cache-dir` outside verify-only mode;
- verify-only artifact checks;
- rejection of empty patch/eval artifacts when the manifest says present;
- rejection of unexpected pre-annotation `ledger.jsonl`.

### 15.6 Annotation Driver and Pilot Annotation Tests

`tests/test_annotate_pilots_from_spec.py` validates the spec driver:

- add/complete behavior;
- blocked status routing;
- reopen-induced progress drops;
- split child categories and parent removal from leaves;
- invalidation exclusion from active progress;
- hard failures for unknown ops and categories;
- ID mismatch detection;
- optional ID recovery when unique;
- quality-file writing;
- missing notes/run-dir hard failures.

`tests/test_swe_agent_pilot_annotations.py` validates the pilot corpus itself:

- exactly 20 pilot specs;
- unique pilot IDs and instance IDs;
- every completion event carries evidence;
- subtask counts match add events;
- event IDs match deterministic session numbering;
- nonnegative event steps;
- at least one reopen exists;
- at least one blocked event exists;
- at least one validation leaf remains unstarted or in progress;
- pilot IDs match source metadata;
- cited steps fall within trajectory bounds;
- replayed quality metadata matches sessions;
- at least one failure ends at full progress;
- at least one success ends below full progress;
- failure progress spans a range.

These tests encode the methodological shapes the pilot is supposed to preserve.
They make it harder to accidentally "clean up" the very cases that prove
progress and outcome are not the same.

### 15.7 Observation Dataset and Audit Tests

`tests/test_ledger_observation_dataset.py` validates:

- deterministic dataset generation;
- no mutation of `ledger.jsonl`;
- event-level one-row-per-event output;
- adjacent delta calculation;
- step-level grouping by `(run_id, step)`;
- recomputation of step deltas;
- drop-source materiality logic;
- coding vs overall divergence;
- high-progress wrong-solution behavior;
- native, legacy, and mixed category-resolution modes;
- CLI artifact writing.

`tests/test_ledger_observation_audit.py` validates:

- invalid progress detection;
- completed-greater-than-active detection;
- adjacent delta mismatch detection;
- native/resolved mismatch reporting;
- event-vs-step comparison;
- success/progress quadrant classification;
- audit CLI output without mutating input CSV.

`tests/test_channel_mission_features.py` validates the newer observation-channel
features:

- all mission-feature columns are present;
- per-category progress is bounded;
- step-windowed event counts are nonnegative;
- cumulative evidence counts are monotone;
- stalled interval columns are nonnegative;
- evidence-strength totals are in the expected range;
- `f_03` retains the stuck-loop shape;
- `f_06` retains the high-progress failure shape.

### 15.8 Final-Success Provenance Tests

`tests/test_resolve_final_success_source_metadata.py` validates the fix that
makes source metadata authoritative for SWE-agent:

- SWE-bench-style eval logs with misleading error tokens do not override an
  upstream true label;
- upstream false labels override misleading passed text;
- unrelated source metadata is ignored;
- missing final-success fields are ignored;
- run manifests still take precedence where applicable;
- absent source metadata falls back to legacy heuristic behavior.

This protects the distinction between source labels and heuristic test-output
parsing.

### 15.9 Agreement, Gate, Evidence, and Schema-Gap Tests

`tests/test_compare_annotations.py` validates the inter-annotator comparison
math: category-vector distance, verdict bands, signed deltas, mismatch errors,
and reopen/block count deltas.

`tests/test_h4_gate_invariants.py` validates the post-revision gate:

- `f_01` coding progress is within gate tolerance;
- all five pilots keep quadrant agreement;
- every v3 spec has a validation leaf;
- upstream labels are not used as evidence;
- v3 ledgers replay cleanly.

`tests/test_pilot_evidence_audit.py` validates evidence-audit accounting,
strong/manual evidence classification, category exclusions, and pilot glob
coverage.

`tests/test_collect_schema_gaps.py` validates schema-gap collection and locks
the core enum value sets used by the schema-decision report.

`tests/test_native_category_invariants.py` validates that new events serialize
categories natively and that the SWE-agent corpus has zero missing-category
violations.

### 15.10 Smoke-Prediction Tests

`tests/test_completion_prediction_smoke.py` validates the smoke-test pipeline:

- missing required columns are rejected;
- unknown labels are rejected;
- forbidden features are not used;
- leave-one-run-out never places the same run in train and test;
- predictions CSV contains required columns;
- the report includes the no-predictive-performance disclaimer;
- high-progress wrong-solution controls are reported when present.

The point is not to prove a model. The point is to keep the evaluation plumbing
honest about grouping, leakage, and disclaimers.

## 16. Current Methodological Limits

Several limitations are explicit in the reports:

- The pilot is N=20, not a distributional estimate.
- The main 20-pilot annotation pass is single-annotator.
- The second and third passes are also LLM annotators, so agreement is not a
  human reliability estimate.
- Some absolute progress values can shift under protocol refinements, especially
  around implicit validation in harness-terminated failures.
- Evidence strength is uneven: product completions often rely on manual notes.
- Retrospective annotation cannot recover every live signal, such as exact
  agent-vs-harness submit provenance or pre-fix baseline test intent.
- Live SWE-agent instrumentation is not yet validated on a real run; current
  SWE-agent evidence is retrospective.

The forward-direction docs therefore pivot toward live instrumentation, but the
retrospective SWE-agent work remains the parity benchmark and the methodology
testbed.

## 17. Reading Order for a New Reviewer

For methodology and source, read in this order:

1. `README.md`
2. `external_data/swe_agent/SOURCE_FORMAT.md`
3. `external_data/swe_agent/PILOT_SAMPLING_POLICY.md`
4. `external_data/swe_agent/manifests/swe_agent_inventory_summary.md`
5. `external_data/swe_agent/manifests/swe_agent_pilot_sample_summary.md`
6. `docs/SWE_AGENT_TRACE_SCHEMA.md`
7. `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
8. `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`
9. `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`
10. `datasets/swe_agent_pilot_observations_summary.md`
11. `datasets/swe_agent_pilot_observations_step_audit.md`
12. `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`
13. `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`
14. `runs/swe_agent_pilot/EVIDENCE_AUDIT.md`
15. `datasets/observation_distribution_comparison.md`
16. `datasets/swe_agent_pilot_completion_smoke_report.md`

For implementation source, read:

1. `ledger_progress/core.py`
2. `ledger_progress/scoring.py`
3. `ledger_progress/session.py`
4. `ledger_progress/queries.py`
5. `ledger_progress/run_manager.py`
6. `scripts/swe_agent_inventory.py`
7. `scripts/sample_swe_agent_pilot.py`
8. `scripts/populate_swe_agent_pilot_cache.py`
9. `scripts/normalize_swe_agent_trace.py`
10. `scripts/import_swe_agent_trace.py`
11. `scripts/annotate_pilots_from_spec.py`
12. `scripts/build_ledger_observation_dataset.py`
13. `scripts/audit_ledger_observation_dataset.py`
14. `scripts/audit_pilot_evidence.py`
15. `scripts/smoke_test_completion_prediction.py`
