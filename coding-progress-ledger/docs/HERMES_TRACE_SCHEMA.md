# Hermes — normalized trace schema

Mirrors `docs/SWE_AGENT_TRACE_SCHEMA.md`. The normalizer's output
(`normalized_trace.json`) is the same shape; only the *upstream*
parser changes.

## Role mapping

| Hermes role | normalized role |
|-------------|------------------|
| `system`    | `system`         |
| `human`     | `user`           |
| `gpt`       | `assistant`      |
| `tool`      | `tool`           |

## Step extraction — locked decisions

These resolve the open questions called out in
`docs/HERMES_REPLAY_PLAN.md` § "Open questions".

### 1. Multi-tool-call turns: SPLIT

A single `gpt` turn may contain multiple `<tool_call>` blocks. The
normalizer **splits** each `<tool_call>` into its own assistant
step, paired with its corresponding `<tool_response>` (matched on
`tool_call_id`). Rationale:

- Each tool_call has a matching tool_response, so the call/response
  pair is the natural step unit.
- Step granularity matters for ledger semantics (REOPEN /
  INVALIDATE / BLOCK timing). A 4-tool-call mega-turn would lose
  that fidelity.
- Symmetrical to SWE-agent's "one fenced command per assistant
  step" rule (`SWE_AGENT_TRACE_SCHEMA.md` § 5).
- Stuck-loop detection (RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL § 6
  tool-response-loop variant) requires comparing *consecutive*
  identical tool_responses; merging hides this.

The free-text portion of the `gpt` turn (preceding the first
tool_call) becomes the `thought` of the FIRST split step. Subsequent
split steps in the same turn carry an empty `thought`.

### 2. `<think>` blocks: COLLAPSE into `thought`

The contents of any `<think>...</think>` block in a `gpt` turn are
concatenated into the assistant step's `thought` field, with
`<think>` / `</think>` tags stripped. Rationale: matches SWE-agent's
single `thought` slot; surfacing `<think>` as a separate role would
explode the step count without distinguishing semantic transitions.

### 3. Thought-only turns: KEEP as a step with empty `action`

A `gpt` turn with no `<tool_call>` is still emitted as one assistant
step. `action` and `command` are null. `thought` carries the free
text (and any `<think>` content). Rationale: empty-action steps are
visible turns the agent "spent"; the ledger annotator may or may
not assign them a leaf, but the observation channel should not
silently drop them.

## Per-step event shape

Identical to the SWE-agent normalizer's output:

```jsonc
{
  "step_index": 0,
  "role": "system|user|assistant|tool|environment|unknown",
  "thought": "<str or null>",
  "action": "<tool_name or null>",
  "observation": "<tool_response content or null>",
  "tool_name": "<str or null>",
  "command": "<arguments JSON-encoded, or null>",
  "files_touched": [],
  "timestamp": null,
  "raw": { ... verbatim upstream entry, before parsing ... }
}
```

`files_touched` is opportunistic: when `tool_name` ∈ `{write_file,
patch, edit_file, read_file}`, parse the arguments for a `path` /
`file` field and populate the list. Otherwise leave empty.

## Top-level normalized trace

```jsonc
{
  "schema_version": 1,
  "source": "hermes_agent_reasoning",
  "instance_id": "<id>",                  // Hermes UUID
  "model_name": "<config>",                // "kimi" or "glm-5.1"
  "exit_status": null,                     // not provided
  "final_success": null,                   // not provided — see HERMES_REPLAY_PLAN.md
  "trajectory_length": <int>,              // post-split step count
  "issue_text": "<task>",
  "system_prompt": "<conversations[0].value if from==system, else null>",
  "events": [ ... ],
  "raw_metadata": {
    "category": "<row.category>",
    "subcategory": "<row.subcategory>",
    "tool_definitions_length": <int>,
    "upstream_conversation_count": <int>
  }
}
```

## What this schema does NOT decide

- Annotation guidance (action vocabulary → SubtaskCategory mapping)
  → `docs/HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md`.
- Sampling policy → `external_data/hermes/PILOT_SAMPLING_POLICY.md`.
- Whether to use kimi or glm-5.1 → policy doc.
