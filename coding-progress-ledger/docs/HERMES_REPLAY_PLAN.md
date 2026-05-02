# Hermes agent reasoning traces — replay feasibility & adaptation plan

Source: [`lambda/hermes-agent-reasoning-traces`](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces)
on Hugging Face. Apache 2.0, public, ~14.7k traces across two model
configs (`kimi`, `glm-5.1`), parquet, ~1.62 GB total.

## Verdict

**Replayable in shape, but no ground-truth outcome label.** The
trajectory + tool-call structure maps cleanly to our retrospective
annotation pipeline. The trace is arguably *richer* than SWE-agent
(explicit `<think>`/`<tool_call>`/`<tool_response>` boundaries give
clean step segmentation). The blocker is that there is **no
`target` / `resolved` / `exit_status` field** — the dataset does not
ship a success label.

This is consistent with our `feedback_progress_vs_outcome_decoupling`
memory: the ledger measures process shape independent of outcome.
Workstream Q's targets are channel-native (drops, reopens, validation
surprises, stuck loops, submit-without-validation), all derivable
without `final_success`. Hermes is therefore a fit for the channel
work *and* a forcing function on the decoupling thesis.

## Schema map: Hermes ↔ SWE-agent

| Need                    | SWE-agent (nebius)         | Hermes                                                 |
|-------------------------|----------------------------|--------------------------------------------------------|
| Task description        | issue text                 | `task` (str)                                           |
| Trajectory              | `trajectory` (turns)       | `conversations` (ShareGPT, ~19–24 avg turns/sample)    |
| Roles                   | `system`, `ai`, `user`     | `system`, `human`, `gpt`, `tool`                       |
| Tool calls / responses  | embedded in turn text      | `<tool_call>` / `<tool_response>` blocks; `tool` role  |
| Reasoning               | implicit                   | explicit `<think>` blocks (~414 words avg)             |
| Tool definitions        | (n/a)                      | `tools` (str of JSON schema)                           |
| Category                | (n/a; one shape)           | `category` (9 levels) + `subcategory`                  |
| Success / outcome       | `target` (bool)            | **missing**                                            |
| Patch / diff            | `generated_patch` (str)    | **missing as field** (may appear inline in tool calls) |
| Eval logs               | `eval_logs` (str)          | **missing**                                            |
| Exit status             | `exit_status` (str)        | **missing**                                            |
| Model name              | `model_name` (str)         | implicit (config = `kimi` or `glm-5.1`)                |
| ID                      | `instance_id` (str)        | `id` (UUID)                                            |

## Category distribution (kimi / glm-5.1)

```
Terminal & Coding         2,010 / 2,237
Agent Tools               1,474 / 2,775
Repository Tasks          1,109 / 1,022
Browser Automation        1,048 /   639
Multi-Tool                  807 /    52
File Operations             757 /   134
Scheduling                  204 /   104
Planning & Organization     201 /    92
Conversational               36 /     0
```

Roughly 5K of 14.7K samples are unambiguously coding-shaped (Terminal
& Coding + Repository Tasks + File Operations). These fit our
PRODUCT/VALIDATION/INVESTIGATION enum without strain. Browser
Automation and Agent Tools fit ARTIFACT/ENVIRONMENT. Conversational and
Scheduling are weakly coding — defer to a later wave.

## Adaptation plan

### Phase 0 — sanity (1 hour)

- Pull one row from each config, save under `external_data/hermes/raw/sample_row.json`.
- Verify the conversation segmentation visually.
- Confirm that `<tool_call>` / `<tool_response>` blocks parse with a
  plain regex (no LLM-aided extraction).

### Phase 1 — schema docs (no code yet)

- `docs/HERMES_TRACE_SCHEMA.md` — mirrors `docs/SWE_AGENT_TRACE_SCHEMA.md`.
  - Role mapping: `system`→`system`, `human`→`user`, `gpt`→`assistant`, `tool`→`tool`.
  - Step extraction: each `gpt` turn = one assistant step; embedded
    `<tool_call>`/`<tool_response>` blocks become tool/environment
    sub-steps within the same step or split into adjacent steps
    (decision recorded in the schema doc).
  - Note explicitly: **no `final_success` field**; downstream
    pipelines must treat the field as `null`.
- `external_data/hermes/SOURCE_FORMAT.md` — verbatim field types from
  the HF datasets-server API; license; per-config row counts;
  retention/citation expectations.
- `external_data/hermes/PILOT_SAMPLING_POLICY.md` — inclusion criteria
  scoped to coding-shaped categories (Terminal & Coding, Repository
  Tasks, File Operations) for the first wave; balance by `category`
  (and per-config) instead of by success/failure (no labels).

### Phase 2 — code (~½ day)

Following the inventory of source-agnostic vs source-specific code,
mirror the SWE-agent files:

- `scripts/hermes_inventory.py` — streams parquet, writes
  `external_data/hermes/manifests/hermes_inventory.csv`. Columns
  match the SWE-agent inventory contract; `final_success_available` /
  `final_success` are always empty. Adds `category` and `subcategory`.
- `scripts/sample_hermes_pilot.py` — deterministic sampler. Balance by
  category × config instead of success × failure. Same fallback ladder
  pattern; same I1–I7 funnel structure.
- `scripts/normalize_hermes_trace.py` — output schema is identical to
  the SWE-agent normalizer's (`schema_version: 1`, `events[]` with
  `step_index`/`role`/`thought`/`action`/`observation`/...). The
  normalizer-private logic (`_internal_role`, action-extraction
  parser) is the only thing that changes.
- `scripts/import_hermes_trace.py` — emits the same eight pre-
  annotation artifacts. Skips `final_diff.patch` and `test_output.txt`
  unless the trace happens to expose a tool-call producing them.
  `source_metadata.json` carries `category`/`subcategory`/`config`.

### Phase 3 — annotation protocol (~½ day)

- `docs/HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md` — thin addendum that
  defers to `RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` on every
  shared rule and only carries Hermes-specific guidance:
  - Action vocabulary → categories: `<tool_call>` opening a
    `read_file` is INVESTIGATION; `write_file` / `edit_file` is
    PRODUCT; `run_tests` / `python script.py` is VALIDATION;
    `submit_answer` / `pull_request` is ARTIFACT.
  - Hermes-specific pitfalls: thought-only turns (skip; not a step
    transition), `<tool_response>` empty-result loops (treat as the
    loop signal Q4's `stuck_loop_next_window` keys on),
    multi-tool-call within one assistant turn (split or merge —
    decision recorded here).
  - **No `final_success` to consult.** Annotators record progress
    shape from visible evidence only. This makes Hermes a clean test
    of the decoupling memory.

### Phase 4 — pilot (decision gate)

- Annotate **5 traces by hand** before scaling (mirrors the D4
  pilot-zero pattern that worked for SWE-agent). Constrain to
  Terminal & Coding category for tightest fit with existing protocol.
- Decision criteria for scaling:
  - Step segmentation is unambiguous on all 5 (no "did the model use
    one tool call or two?" disagreement).
  - All five categories needed (PRODUCT/VALIDATION/INVESTIGATION/
    ARTIFACT/ENVIRONMENT) actually appear at least once across the
    pilot — confirms the existing enum is sufficient.
  - At most one new pitfall per trace — comparable to the SWE-agent
    pilot.
- If any criterion fails, fix the protocol before scaling. Do **not**
  scale to 20 with an ambiguous protocol.

## Hardcoded SWE-agent assumptions to break

From the pipeline inventory, the following will NOT carry over
without edits:

1. Role mapping (`_internal_role`) — Hermes uses `gpt`, not `ai`.
2. Action extraction (`_split_thought_and_command`) — Hermes uses
   `<tool_call>`/`<tool_response>` tags, not triple-backtick fences.
3. Final-success-as-bool assumption — Hermes is `null`. Existing
   downstream tools (`resolve_final_success`, the smoke test) must
   tolerate `null`. Workstream Q labels do not depend on it; W3's
   `label_final_success` will be `null` for Hermes runs.
4. `generated_patch` / `eval_logs` field names — Hermes does not have
   these. The importer should write `final_diff.patch` /
   `test_output.txt` only if the trace contains explicit tool calls
   that produced them.
5. CSV deduplication on `instance_id` — Hermes `id` is a UUID and is
   already unique; dedupe rule simplifies.
6. Sampling policy — balance by `category` × `config`, not by
   success/failure.

## Out of scope for the first pass

- Cross-source predictive comparison (Workstream P) — defer until at
  least one Hermes pilot exists.
- Live Hermes instrumentation (Workstream N analogue) — Hermes is
  retrospective-only by nature; skip.
- Adding new `SubtaskCategory` values for non-coding tasks. The
  six-category enum is intentionally domain-agnostic; we re-label
  the same categories in the Hermes addendum.
- `final_success` prediction (Q6) on Hermes — there is no label.
  Q1–Q5 channel-native targets carry through.

## Files to create (estimated ½ day each)

```
docs/HERMES_TRACE_SCHEMA.md
docs/HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md
external_data/hermes/SOURCE_FORMAT.md
external_data/hermes/PILOT_SAMPLING_POLICY.md
external_data/hermes/raw/sample_row.json
scripts/hermes_inventory.py
scripts/sample_hermes_pilot.py
scripts/normalize_hermes_trace.py
scripts/import_hermes_trace.py
tests/test_hermes_inventory.py
tests/test_normalize_hermes_trace.py
tests/test_import_hermes_trace.py
tests/test_sample_hermes_pilot.py
```

## Open questions

1. Is there any way to reconstruct a success signal post-hoc — e.g.
   inspecting the final `tool_response` for "tests pass" or
   `submit_answer` payload validity? Investigate during Phase 0.
2. Are `<think>` blocks worth surfacing as a separate event type, or
   collapsed into the assistant step's `thought` field? Current
   recommendation: collapse, matching SWE-agent's `thought` slot.
3. Should we annotate both `kimi` and `glm-5.1` configs in the same
   pilot, or pick one? Default: one (kimi, larger Terminal & Coding
   slice); revisit if cross-model comparison becomes the headline.
