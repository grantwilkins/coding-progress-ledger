# Hermes retrospective ledger annotation addendum

This addendum specializes the general protocol
(`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`) to Hermes
agent reasoning traces normalized per `docs/HERMES_TRACE_SCHEMA.md`.
The general protocol wins on every conflict.

This file only:

- maps Hermes tool vocabulary onto the general categories (§ 1),
- names the Hermes input artifacts (§ 2),
- specifies the (absent) upstream success label (§ 3),
- gives Hermes-specific pitfalls (§ 4).

## 1. Hermes tool vocabulary → general categories

Hermes traces use a function-calling tool API. Map by tool name:

| Hermes tool name                                              | General category   |
|---------------------------------------------------------------|--------------------|
| `read_file`, `list_directory`, `find_file`, `grep`, `search`  | `INVESTIGATION`    |
| `web_search`, `fetch_url`, `browse`                           | `INVESTIGATION`    |
| `write_file`, `patch`, `edit_file`, `create_file`             | `PRODUCT`          |
| `run_tests`, `pytest`, `python <repro>`, `bash <test_cmd>`    | `VALIDATION`       |
| `bash` / `shell` for *running* a written script               | `VALIDATION`       |
| `pip install`, `apt-get`, env-setup commands                  | `ENVIRONMENT`      |
| `submit_answer`, `final_response`, `task_complete`            | `ARTIFACT`         |
| `write_file` of `*.md` / `README` when task requires docs     | `DOCUMENTATION`    |

If a tool is not on this list, classify by the **observable effect**:
- modifies file → PRODUCT
- runs file / executes assertions → VALIDATION
- produces information → INVESTIGATION
- changes the runtime/env → ENVIRONMENT
- emits the final user-visible answer → ARTIFACT

The general "subtask = unit of discovered work" rule still holds: 10
read_file calls that locate one definition are ONE `INVESTIGATION`
leaf, not ten.

## 2. Hermes input artifacts

Per pilot run dir at `runs/hermes_pilot/<pilot_id>/`:

```text
task.md                  # the upstream `task` field, verbatim
source_trace.json        # the upstream row, verbatim
normalized_trace.json    # output of scripts/normalize_hermes_trace.py
trajectory_summary.md    # human-readable per-step summary
final_diff.patch         # placeholder; Hermes has no upstream diff
test_output.txt          # placeholder; Hermes has no upstream eval log
run_notes.md             # annotator's free-form notes
source_metadata.json     # source / instance_id / model / category / final_success: null
```

The annotator works from `task.md` + `normalized_trace.json` +
`trajectory_summary.md`. `final_diff.patch` and `test_output.txt`
are placeholders unless the agent's tool calls happen to produce
them inline (e.g., a `patch` tool's diff).

## 3. Upstream success label: ABSENT

Hermes has **no `final_success` / `target` / `resolved` / `exit_status`
field** (verified, see `external_data/hermes/SOURCE_FORMAT.md`).

Annotator implications:

- Do not infer success from "the agent emitted submit_answer." Many
  Hermes traces ARE successful, but the dataset does not certify it.
- Do not infer failure from a missing `submit_answer`.
- The pilot's `source_metadata.json::final_success` is `null`.
- The general protocol's "no upstream-label inference" rule (§ 1)
  becomes load-bearing here: progress is measured purely from
  visible work, never from outcome.

## 4. Hermes-specific pitfalls

### Pitfall H1 — Multi-tool-call assistant turns

A single `gpt` turn can issue 2–4 `<tool_call>` blocks. The
normalizer SPLITs each into its own step (per
`HERMES_TRACE_SCHEMA.md` § "Step extraction"). Annotators must treat
each split step as a distinct opportunity for INVESTIGATION /
PRODUCT / VALIDATION classification — do NOT merge a multi-call turn
into one ledger event.

### Pitfall H2 — `<think>` block content is annotator-facing thought, not ledger evidence

The normalizer collapses `<think>...</think>` content into the
step's `thought` field. **Do not cite `thought` as evidence for
subtask completion.** Evidence must come from `action` /
`observation` (the tool call and its response), per general
protocol § 4.

### Pitfall H3 — Tool-response-loop detection on Hermes responses

Tool responses include a stable `tool_call_id` preamble that varies
per call even when the response *body* is identical. When applying
the tool-response-loop rule (general § 6 tool-response-loop variant),
**compare response bodies (the `content` field), not the wrapping
text.** Three identical `content` payloads in a row = a loop.

### Pitfall H4 — `submit_answer` is ARTIFACT, not VALIDATION

A Hermes agent emitting `submit_answer` (or equivalent terminal
tool call) closes an `ARTIFACT` leaf. It is NOT validation evidence
unless the call payload itself contains a verified test result —
which is rare. If validation never happened, the run carries
`submit_without_validation` in W2 / Q1 terms.

### Pitfall H5 — Browser/shell ambiguity

Tools like `bash` or `terminal` can be either VALIDATION (running a
test script) or PRODUCT (writing via `>>` redirect) or
INVESTIGATION (`ls`, `cat`). Always classify by the *intent
visible in the command*, not by the tool name. A `bash` call running
`pytest tests/` is VALIDATION; a `bash` call running `cat
README.md` is INVESTIGATION.

## 5. Open questions

1. Is there any post-hoc success signal recoverable (e.g., the final
   `tool_response` containing "tests passed" text)? Defer until HP4
   parity report has feasibility data.
2. Should we ever annotate a `<think>` block as a separate
   "thought-only step"? Current rule: NO; the step exists, but no
   leaf is opened on thought alone.
3. Should multi-config (`kimi` + `glm-5.1`) annotation use a single
   protocol? Yes — the tool vocabulary is shared. The protocol does
   not need per-config branches.
