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

Status: done — `IMPLEMENTATION_v0.md` is the detailed handoff for the SWE-agent retrospective methodology, with the v1-vs-revised-protocol separation noted.

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
Status: done — `docs/SWE_AGENT_TRACE_SCHEMA.md` (commit c93d182).

### C2. Implement trace normalizer
Status: done — `scripts/normalize_swe_agent_trace.py` + tests (commit e2281af).

### C3. Implement import-to-run script
Status: done — `scripts/import_swe_agent_trace.py` + tests (commit 801fbf3). Cache at `external_data/swe_agent/pilot_cache/`; 20 pilot run dirs at `runs/swe_agent_pilot/`. Importer maps upstream `eval_logs` → framework's `test_output.txt`.

### C4. Pre-annotation verification (folded into C3)
Status: done — `--verify-only` mode in `import_swe_agent_trace.py` rejects missing artifacts and unexpected `ledger.jsonl` files.

---

## § Workstream D — Retrospective ledger annotation protocol

### D1. Write annotation guidelines
Status: done — `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` (general, source-agnostic) + `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md` (SWE-agent addendum). Pre-E1 stress-test refinements landed (cycle-length-agnostic stuck-loop rule, harness-forced termination ≠ ARTIFACT leaf, SPLIT preserves parent status).

### D2. Create annotation template
Status: done — `docs/LEDGER_ANNOTATION_TEMPLATE.md`.

### D3. Manual annotation helper (deferred by default)
Status: deferred — D4 confirmed snippets are sufficient; no helper built. Re-open only if friction recurs.

### D4. Annotate 2 traces by hand as pilot-zero
Status: done — three annotations (s_01 / f_01 / f_03) via `annotations/swe_agent_pilot/<pilot_id>.{json,notes.md}` + driver `scripts/annotate_pilots_from_spec.py` + 12 tests. Progress: s_01 1.00/1.00, f_01 0.75/0.67, f_03 0.50/0.50.

### D5. Annotation quality checklist
Status: done — `annotation_quality.json` emitted per pilot; tracks schema-gap-found, final-success-only-at-end, progress-forced flags.

---

## § Workstream E — Retrospective annotation at pilot scale

### E1. Annotate 20 traces
Status: done — all 20 pilots annotated via the spec-driven driver (`annotations/swe_agent_pilot/<pilot_id>.{json,notes.md}`). Every run passes `ledger-run check-run`. F2/F3 ingest the full 20-pilot dataset cleanly: integrity checks (`completed_exceeds_active`, `delta_mismatches`) all empty. One audit warning surfaced (`s_03` shows a 0.33 native-vs-resolved coding-progress divergence around the step-22 REOPEN); recorded as a downstream-audit signal, not blocking.

Progress shape spans 0.50–1.00 across the 10 failures and 0.75–1.00 across the 10 successes — see `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` for the per-instance table. Pre-E2 protocol refinement: § 6 `blocked` rule split into command-loop and tool-response-loop variants (forced by `f_02`'s thesaurus loop).

### E2. Run run-manager exports
Status: done — `ledger-run export-run`/`check-run` invoked by the spec-driven driver for all 20 pilots; all artifacts present, no placeholders.

### E3. Write pilot annotation summary
Status: done — `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` (per-pilot table, notable shapes, protocol refinements, evidence-gap patterns, honest qualitative judgment).

---

## § Workstream F — Observation dataset integration

### F1. Extend dataset builder to include SWE-agent pilot
Status: done — existing `scripts/build_ledger_observation_dataset.py` already accepts `--runs-dir runs/swe_agent_pilot`; no code change needed.

### F2. Generate SWE-agent-only observation tables
Status: done — `datasets/swe_agent_pilot_observations_{event,step}.csv` (202 / 191 rows) + `_summary.md`.

### F3. Audit SWE-agent observation dataset
Status: done — `datasets/swe_agent_pilot_observations_{step,event}_audit.{md,json}`. All integrity checks empty; one residual `s_03` native/resolved divergence noted as non-blocking. Six spec files were stable-sorted to step-monotonic order (semantics unchanged).

### F4. Compare toy/live vs SWE-agent distributions
Status: done — `datasets/observation_distribution_comparison.md`. SWE-agent traces populate all four success/progress quadrants vs 2/4 for toy/live, and 100% are non-monotonic vs 78%. Surfaced the `resolve_final_success` heuristic bug (now fixed via `source_metadata.target` precedence).

---

## § Workstream G — Completion-prediction smoke test on SWE-agent pilot

### G1. Run existing smoke script on SWE-agent-only step table
Status: done — `datasets/swe_agent_pilot_completion_smoke_{predictions.csv,report.md}`. Leave-one-run-out by `run_id`; no leakage; three model variants (`progress_only`, `ledger_basic`, `elapsed_only`).

### G2. Add SWE-agent smoke report interpretation
Status: done — report extended with G2.1–G2.8. Verdict: data suitable for larger retrospective study, gated on the (now-fixed) `resolve_final_success` heuristic and H's inter-annotator pass. Explicitly does not claim predictive performance.

### G3. Case-study extraction
Status: done — `datasets/swe_agent_pilot_case_studies.md` (s_01, s_03, f_06, f_03 — all four success/progress quadrants, one page each, every claim trace-backed).

---

## § Workstream H — Inter-annotator reliability

> **Scheduling note:** H is methodologically important but heavy (5 extra annotations at ~30 min each). Run H **after** E2 lands and only if M leans toward "scale". If M leans "pause" or "schema-change-needed", skip H and revisit when annotation effort is justified by a real scale-out plan.

### H1. Duplicate-annotate 5 traces
Status: done — Opus second-annotator pass on `s_01`, `s_03`, `f_01`, `f_06`, `f_03`. v2 specs at `annotations/swe_agent_pilot_v2/` (subagent isolated from v1 annotations). Caveat: both annotators are LLM (correlated biases).

### H2. Compare annotations
Status: done — `scripts/compare_annotations.py` + `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`. Quadrant agreement 5/5; mean |Δ coding-progress| = 0.10; only `f_01` showed different conclusions (the implicit-validation gap).

### H3. Decide if annotation protocol needs changes
Status: done — `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md` (1 HIGH revision: bug-fix tasks have implicit validation; 2 LOW revisions; 1 no-change acknowledgment of granularity latitude). Applied to the protocol docs.

---

## § Workstream I — Schema-gap review

### I1. Collect schema gaps from run notes
Status: done — `scripts/collect_schema_gaps.py` + `runs/swe_agent_pilot/SCHEMA_GAPS.md`. 2 pilots flagged (`f_02`, `f_07`); 18 explicitly None. 12 invariant tests lock the core-enum sets that I2 depends on.

### I2. Decide no-change vs schema-change
Status: done — `runs/swe_agent_pilot/SCHEMA_DECISION.md`. **Outcome: no schema change needed; only annotation-protocol changes (already landed).** Status / EventType / SubtaskCategory enum sets unchanged since pilot start.

---

## § Workstream J — Native-category quality

### J1. Measure category resolution for SWE-agent pilot
Status: done — `runs/swe_agent_pilot/CATEGORY_RESOLUTION_REPORT.md`. Root cause: `LedgerSession.add()` was stripping the default `category` field from payloads, making every PRODUCT subtask resolve to `mixed`. Fixed by always emitting `category`; all 20 pilots now `native`, progress numbers byte-identical.

### J2. Enforce native categories for new annotations
Status: done — `scripts/check_native_categories.py` walks pilot/v3 dirs and exits non-zero on missing-category violations. 25 SWE-agent runs all native; 18 legacy runs path-filtered. 6 invariant tests.

---

## § Workstream K — Evidence quality

### K1. Evidence availability audit
Status: done — `runs/swe_agent_pilot/EVIDENCE_AUDIT.md`. 81 completions: 51 strong (63%) / 30 manual-only (37%). PRODUCT 83% weak (mostly edit-acks), VALIDATION 8%, INVESTIGATION 3%. Weak evidence is a signal, not a replay failure. Invariant tests in `tests/test_pilot_evidence_audit.py`.

### K2. Source trace evidence-gap report
Status: done — `runs/swe_agent_pilot/SOURCE_EVIDENCE_GAPS.md` classifies evidence gaps as retrospective-recoverable / closed-by-live / structurally-unrecoverable. Headline: ~60% of `manual_note` completions are closable by adding a `tool_action` evidence type (K3). Live instrumentation justified for hidden-work-gap visibility, submit provenance, and pre-fix baseline.

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
Status: done — `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`. Original recommendation (scale retrospective to 100, defer live) was overtaken by the post-CRITIC_AUDIT pivot in M2 to live instrumentation.

### H4. Re-test H3 protocol revisions on the 5 H pilots (gate to M2 / scale)
Status: done — `runs/swe_agent_pilot_reannotation/H4_GATE_RESULT.md`. **Gate PASSES.** v3 cold-pass quadrant agreement v1↔v3 = 5/5; f_01 delta 0.00. 5 invariant tests. Incidental finding: v1 inconsistent on Pitfall #8 across `f_02`/`f_03`/`f_07`/`f_10` — recommended cleanup pass (~30 min); does not block M2.

### M2. Define next direction (post-CRITIC_AUDIT pivot)
Status: done — `runs/swe_agent_pilot/NEXT_DIRECTION_MEMO.md`. Pivot from retrospective scale-out to live instrumentation. Defines the eight forward deliverables (N1–N4, U1–U2, T1, V1) and a four-clause gate for live N=20.

---

## Forward priorities (post-CRITIC_AUDIT)

### § Workstream N — Live SWE-agent instrumentation
Status: **complete** (N1–N6 ✓ as of 2026-05-01).

Goal: Wrap a live SWE-agent run so `LedgerEvent`s emit during execution with real wall-clock timestamps, removing retrospective bias and unlocking K2's hidden-work-gap, submit-provenance, and pre-fix-baseline gaps.

#### N1. Decide sidecar vs in-agent instrumentation
Status: done — `docs/LIVE_INSTRUMENTATION_DECISION.md`. Chosen: hybrid sidecar with wire-format protocol v1.0; framework-agnostic via per-framework adapter modules.

#### N2. Build the live ledger sidecar
Status: done — `ledger_progress/sidecar.py` + `adapters/{generic,swe_agent}.py` + `tests/test_sidecar.py`. CLI: `python -m ledger_progress.sidecar --run-dir X --adapter {generic,swe_agent}`. Replay-equality, timestamp authority, explicit-op bypass, single-`run_id` invariants, <100ms/event latency all locked by tests.

#### N3. Hook one SWE-agent run
Status: done — `scripts/run_swe_agent_live_sidecar.py` + `docs/LIVE_SWE_AGENT_HOOK.md`. Two run dirs under `runs/swe_agent_live/` (Melevir success, WIPACrepo failure); both pass `check-run`. **Caveat:** timestamps are replay-time (microsecond elapsed), not real wall-clock — that is the gap N6 closes.

#### N4. Live-vs-retrospective parity report
Status: done — `scripts/build_live_parity_report.py` + `runs/swe_agent_live/PARITY_REPORT.md` + `EVENT_OBSERVABILITY_MATRIX.md`. **Frontier policy:** raw-step instrumentation does not invent validation obligations; submit-without-validation surfaces as `complete_visible_frontier+no_validation_frontier`. Verdict: policy-adjusted gate passes; N5 unblocked. The scalar divergence (live → 1.0, retro lower) is documented, not hidden — but is also a real measurement-validity issue that **W2 must address** before the live channel is operationally trustworthy.

#### N5. Extend to a live N=20 batch
Status: done — `runs/swe_agent_live/N5_BATCH_SUMMARY.md` + 20 live run dirs. All pass `check-run`; every event has a (replay-time) timestamp; no retrospective annotation. Systematic live-vs-retro progress gap is consistent with the N4 frontier policy. **Caveat:** because N3 emits replay-time timestamps, V1's wall-clock columns populate with values too small to be physically informative on this batch — see N6.

#### N6. Capture real wall-clock timestamps on a live SWE-agent run
Status: done — `scripts/run_swe_agent_live_sidecar.py` gained `--synthetic-clock-start` + `--synthetic-step-seconds` (path b). 20 sibling runs at `runs/swe_agent_live_wallclock/<instance_id>/` carry timestamps with `timestamp_span_seconds >= 60`; V1's `seconds_since_progress_increase` exceeds 1.0 on 664/704 step rows. `live_instrumentation.json::timestamp_source` is `synthetic` on the new batch, `replay` on the original N5 batch. `docs/LIVE_SWE_AGENT_HOOK.md` and `runs/swe_agent_live/N5_BATCH_SUMMARY.md` updated. Path (a) — hooking a freshly-running SWE-agent — remains available via the sidecar's stdin path; not exercised here because upstream traces lack per-step timestamps.

### § Workstream U — Live query CLI / monitor surface
Status: **NEW** (post-CRITIC_AUDIT). Consumes the query API that landed in commit `5bdcab6`.

Goal: Make `ledger_progress.queries` reachable from outside Python — a `ledger-run` CLI subcommand and (optionally) a small HTTP server. This is the surface a real long-running monitor would call.

#### U1. `ledger-run watch <run_dir>`
Status: done — `ledger_progress/run_manager.py:_cmd_watch` and `tests/test_run_manager_watch_query.py::test_watch_emits_one_update_per_appended_event`. Polls ledger.jsonl, replays incrementally, prints one JSON line per new event with step / event_type / subtask_id / coding_progress / active_blocked_leaves / stalled_for_blocked / timestamp. Test appends 6 events while watch is running and asserts 6 update lines emerge.

CLI:
```text
ledger-run watch <run_dir> [--poll-interval 0.5] [--exit-after-events N]
```

#### U2. `ledger-run query <run_dir> --filter ...`
Status: done — `ledger_progress/run_manager.py:_cmd_query` and `tests/test_run_manager_watch_query.py`. Exposes every queries.py function as a flag and emits a single JSON object on stdout.

CLI:
```text
ledger-run query <run_dir> --status blocked
ledger-run query <run_dir> --stalled-for N            # prints stalled_for + meets_threshold
ledger-run query <run_dir> --reopens-since STEP
ledger-run query <run_dir> --newly-discovered-since STEP
ledger-run query <run_dir> --last-validation-event
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
Status: done — `scripts/build_ledger_observation_dataset.py` extends `DATASET_FIELDS` with `elapsed_seconds`, `seconds_since_last_event`, `seconds_since_progress_increase`, and `events_per_minute` (5-event rolling window). Step rows recompute `seconds_since_last_event` and `seconds_since_progress_increase` at step granularity using the retained event timestamps. Empty string on legacy step-only ledgers; populated on live N=20 batch. Test: `tests/test_ledger_observation_dataset.py::test_wall_clock_columns_null_on_legacy_populated_on_timestamped`.

#### V2. Deadline-aware estimator stub
Status: done — `ledger_progress/estimators.py:p_finish_by(ledger, deadline, categories=CODING)` and `tests/test_estimators.py` (7 tests). Linear extrapolation from observed progress velocity. Stub, not calibrated; downstream W/Q is meant to swap the body.

**Known issue (TODO before W/Q consume):** the return value is dimensionally a time ratio, not a probability. Either rename to `fraction_of_time_to_finish_remaining` (and let callers map to a probability) or replace the body with an actual calibrated predictor. Do not let this stub be cited as a probability in any downstream report.

**Cannot be calibrated until N6 ships:** the wall-clock columns V1 produces on the current live N=20 are microsecond-scale (replay-time, not real wall-clock). Any probability calibration on this data would be meaningless.

### § Workstream W — Observation-channel sharpening
Status: **NEW current priority** (post-handoff critique). The critical-path workstream. Until W2 ships, the live channel's primary scalar (`coding_progress`) is systematically over-optimistic for failures (live reads ~1.0 whether agent succeeded or botched — see N4 PARITY_REPORT). Shape labels are the operational fix.

**Hard ordering:** W4 → W2 → W3. Reason: W4's revised implicit-validation semantics shift scalar progress on `f_02`/`f_03`/`f_07`/`f_10`. If those shifts change shape labels, doing W2 first would invalidate W2 outputs retroactively. The earlier "W4 in parallel" framing was wrong.

Goal: Make the scientific variables explicit. The channel observes the visible work frontier: discovery, closure, instability, stalls, validation state, evidence strength, and category-local progress. W turns those into auditable shape labels and estimator-ready checkpoint features without changing `LedgerEvent`, `Status`, or `SubtaskCategory`.

#### W1. Event observability matrix
Status: done — `runs/swe_agent_live/EVENT_OBSERVABILITY_MATRIX.md` (generated by N4's `build_live_parity_report.py`). Separates mechanical live events from weakly inferable grouping and annotation-only semantic transitions.

#### W4. Quantify annotation sensitivity on Pitfall #8 cleanup cases
Status: done — `runs/swe_agent_pilot/PITFALL8_SENSITIVITY.md`. Replay-computed scalar shifts confirm H4's predictions exactly: f_02/f_03 0.50→0.33, f_07/f_10 0.67→0.50 (Δ=−0.1667 each). Shape labels are **stable** across the shift provided W2 anchors `high_progress_failure` at threshold ≥ 0.70 and adopts the "no validation *attempted*" reading of `no_validation_frontier`. With those anchors, W2 may proceed against the current scalar values; the dominant tag for all four pilots is `stuck_loop`, derived from event patterns invariant to leaf-count denominators.

#### W2. Add shape-level labels and reports
Status: done — `scripts/label_observation_shapes.py` + `datasets/swe_agent_pilot_shape_labels.csv` + `datasets/swe_agent_pilot_shape_report.md` + `tests/test_shape_labels.py` (11 tests). f_06 carries `high_progress_failure` + `hidden_work_gap`; s_04 carries `low_progress_success` + `submit_without_validation`; f_02/f_03 carry `stuck_loop`; s_03 carries `nonmonotone_recovery`. On the live N=20 batch every progress=1.0 run is definitively classified (no_validation_frontier, clean_success, or high_progress_failure — extending the original two-bucket acceptance to keep failures-with-validation from being silently treated as clean successes). 9/20 pilots clean_success; 6/20 stuck_loop; 4/20 submit_without_validation. Tags use the W4 anchors (`high_progress_failure` ≥ 0.70 and `no_validation_frontier` = "no validation attempted"). `scope_discovery_after_high_progress` is restricted to PRODUCT/INVESTIGATION adds *after* a REOPEN_SUBTASK to avoid firing on routine sequential annotation layout.

Goal: Report stable qualitative shapes rather than relying on exact scalar progress. These are audit tags first, not training labels. **Operational urgency:** the live channel currently emits a progress scalar that is indistinguishable between successes and failures at the high end. Shape tags (`high_progress_failure`, `submit_without_validation`, `no_validation_frontier`, `hidden_work_gap`) are the live-queryable signal that distinguishes "done" from "submitted-without-test".

Initial shape tags:
```text
high_progress_failure
low_progress_success
stuck_loop
submit_without_validation
no_validation_frontier
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
every live N=20 run with progress=1.0 carries either no_validation_frontier or a clean success classification
labels are derived from ledger/metadata fields plus run_notes citations where needed
report warns that labels are audit tags, not final model targets yet
```

#### W3. Build estimator checkpoint table
Status: done — `scripts/build_estimator_checkpoints.py` + `datasets/swe_agent_estimator_checkpoints.csv` (191 rows, 20 runs, matches step-table cardinality 1:1) + `datasets/swe_agent_estimator_checkpoints_summary.md` + `tests/test_estimator_checkpoints.py` (13 tests). All seven W3 feature groups present; label columns prefixed `label_*` so trainers can drop them by schema. No future leakage (verified by replaying reopens up to each checkpoint and confirming `largest_progress_drop_so_far` is monotone non-decreasing per run). Legacy retrospective ledgers without timestamps are supported (the pilot is the canonical legacy dataset). `success_by_horizon` defaults to 30 steps.

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
Status: not started · _W3 cleared; final-success prediction stays deferred until Q1–Q5 land_

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

## § Workstream T — Task-set as first-class structure
Status: **PROMOTED to current priority** (post-CRITIC_AUDIT) — but has shipped zero code across two promotion cycles. The only roadmap item that addresses the mission's *long-range* verb (everything else models one task / one trace).

**Decision needed (this cycle):** ship T1 (the protocol doc — small, low-cost, unblocks T2–T5) **or** explicitly demote the workstream and stop calling the framework "long-range." The current promoted-but-unshipped state is the worst of both. Default if no explicit decision lands by the next planning checkpoint: T1 ships, T2–T5 stay blocked on T1 as before.

The framework currently models progress within one task / one trace. Real long-range agentic work — a multi-week refactor decomposed into 30 sub-issues, a research agent running a sequence of partly-independent experiments, even a single SWE-bench run viewed as "one issue out of N" — has a coarser unit of analysis: the **set of tasks**. Workstream T introduces that unit *without disturbing the single-task pipeline*.

**Architectural decision.** Add `LedgerSet` as a thin, source-agnostic container over an ordered collection of `Ledger`s, plus a sibling `LedgerSetSession` that mirrors `LedgerSession` ergonomics one level up. A `LedgerSet` holds `(set_id, members: list[LedgerSetMember])` where `LedgerSetMember` is `(member_id, ledger_ref, weight: float = 1.0, status_override: Status | None = None)`. `ledger_ref` is a path / handle to an existing `Ledger`, not an inline copy — the set never owns ledger bytes, and a ledger may belong to multiple sets. No cross-member dependencies, no DAG edges, no time windows in v1; future addenda can grow optional fields on members. The existing `Ledger` / `LedgerSession` / `Subtask` types are unchanged.

**Minimal API.** `LedgerSetSession(set_id)` exposes `add_member(ledger_ref, weight=1.0)`, `mark_member(member_id, status)` (only when a member's outcome is decided outside the ledger — e.g. a sub-issue declared out-of-scope), `score()`, `export_jsonl()`. No splits or reopens at the set level in v1: a member's progress shape is the member's ledger's job.

**Aggregation rule (v1, simplest defensible).** Set-level progress is the **weight-weighted mean of per-member coding-progress**: each member contributes `weight * member.score(CODING_CATEGORIES).progress`, divided by `sum(weight)`. Members with `status_override in {INVALIDATED, DELETED}` drop out of both numerator and denominator (matches single-task semantics). Rationale: leaf-count weighting would let a 30-leaf member dominate a 3-leaf member when both are "one sub-issue"; explicit per-member `weight` puts that call in the annotator's hands and defaults to uniform.

**What does NOT change.** The B2 sampler, C3 importer, D1 protocol, E1 annotation, F2/F3 dataset, and `ledger-run` CLI all continue to operate on single ledgers. Reason: the discovered-vs-hidden rule and the anti-narrative stance are properties of one trace; promoting them to the set level would invite retro-fitting cross-task dependencies the traces don't surface. The set layer reads finished ledgers and aggregates; it does not annotate.

**Migration.** The 20 SWE-agent pilot ledgers each become a `LedgerSet` of size 1 via a trivial wrapper (T4). Their per-run artifacts are untouched; only a sibling `set.jsonl` appears at `runs/swe_agent_pilot/<pilot_id>/`. The suite-level rollup at `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` (E3) becomes the first non-trivial set: one `LedgerSet` of 20 members, weight 1.0 each.

**Rejected alternatives.** (1) *Make `Ledger` itself recursive (a ledger may contain ledgers)* — overloads `Subtask` semantics and forces scoring to traverse mixed leaf / sub-ledger trees. (2) *Track sets only as a CSV manifest with no runtime type* — pushes aggregation policy into ad-hoc scripts and loses replay / serialization parity with `Ledger`.

### T1. Write general LedgerSet protocol doc
Status: done — `docs/LEDGER_SET_PROTOCOL.md`.

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

## Minimal first batch (pilot phase — historical, complete)

Pilot phase A–M complete as of 2026-04-30 — see `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`.

## Minimal first batch (forward phase — current)

N1–N6 ✓, W1 ✓, W2 ✓, W3 ✓, W4 ✓, U1+U2 ✓, T1 ✓. **Workstream W is complete.** Remaining:
- **T2 → T3 → T4 / T5** LedgerSet implementation (T1 ships the protocol; T2+ are the data-model + first-use code)
- **V2** estimator stub recalibration on the N6 wallclock batch (the synthetic-clock data finally makes V1's columns physically informative; the time-ratio-vs-probability rename in V2 is now actionable)
- **Q** predictive modeling pass (now unblocked: W3 is the estimator checkpoint table)

Do not start Q modeling until W3 exists. Treat the live N=20 progress scalar as untrustworthy for outcome questions until W2's shape labels ship — see `runs/swe_agent_live/PARITY_REPORT.md`.

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
