# Serving Group Manifest — schema

The manifest is a derived artifact built by replaying a vagrant trace. **Do not hand-edit it.** If a manifest looks wrong, fix the trace or the replay logic.

```text
trace.jsonl  --build_manifest()-->  ServingGroupManifest  --write_json-->  manifest.json
```

## Top-level

```json
{
  "workflow_id": "toy_workflow_v1",
  "root_task": "toy coding task",
  "nodes":         { "<node_id>": <WorkNode> },
  "state_objects": { "<state_id>": <StateObject> },
  "edges":         [ <StateEdge>, ... ]
}
```

## `WorkNode`

```json
{
  "node_id":        "S2",
  "node_type":      "subagent",
  "parent_node_id": "S1",
  "workflow_id":    "toy_workflow_v1",
  "label":          "A",
  "status":         "complete",
  "required_state": ["system_prefix", "repo_context", "private_A", "workspace_AC"],
  "produced_state": []
}
```

- `node_type` ∈ `{llm_call, tool_call, subagent, summary, test, unknown}` — recovered from the `add_subtask` event payload (ledger_progress's Subtask doesn't carry it).
- `status` is the **final** status from the replayed ledger.
- `required_state` / `produced_state` are deduplicated `state_id` lists.

## `StateObject`

```json
{
  "state_id":     "repo_context",
  "content_hash": "hash_repo_context_v1",
  "layer":        "prompt_context",
  "lifetime":     "shared",
  "tokens":       8000,
  "bytes":        null,
  "producers":    [],
  "consumers":    ["S1", "S2", "S3", "S4"],
  "invalidated":  false
}
```

- **Primary key:** `state_id`. **Secondary:** `content_hash`. **Any** duplicate `state_declare` for the same `state_id` hard-fails — even with the same hash. MVP does not version state objects; real adapters must dedupe before emitting.
- `layer` ∈ `{model_execution, prompt_context, subagent, workspace, memory, semantic}`.
- `lifetime` ∈ `{persistent, shared, private, ephemeral}`.
- `bytes` is null for token-only state. Workspace state typically has `tokens=0` and a non-null `bytes`.
- `producers` / `consumers` are deduplicated, insertion-ordered.
- `invalidated` flips to `true` if a `state_invalidate` event is seen.

## `StateEdge`

```json
{ "node_a": "S1", "node_b": "S2", "state_id": "repo_context", "weight": 8000 }
```

- **This is a derived view**, not the source of truth. The bipartite source is `StateObject.consumers`. Pairwise edges are emitted for every unordered pair of distinct consumers of the same state object.
- `weight` = `state_object.tokens` (NOT the sum of per-read tokens). Two consumers of an 8000-token object share the *whole* object once; the edge weight is 8000.
- Workspace state has `tokens=0`, so its derived edges have `weight=0`. They are still emitted; policies decide whether to filter at their `tau` threshold.

## Build invariants (all hard-fail if violated)

- Every node referenced as a producer or consumer must have an `add_subtask` event in the trace. Validated after the event loop.
- Every state object referenced by a `state_read` / `state_write` / `state_invalidate` must have a prior `state_declare`.
- `content_hash` referenced on a read/write must match the declared hash.
- No duplicate `state_declare` for the same `state_id` (regardless of hash).
- Mixed `workflow_id` values across `add_subtask` events hard-fail.

## Bipartite source-of-truth contract

`StateObject.consumers` and `StateObject.producers` are authoritative. `WorkNode.required_state` and `WorkNode.produced_state` are mirrors derived by a reconciliation pass after the event loop, so trace event ordering (e.g. `state_read` before its consumer's `add_subtask`) does not leave the manifest inconsistent.

## CLI

```bash
vagrant-manifest build --trace examples/traces/toy_subagent_trace.jsonl \
                        --out  examples/manifests/toy.json
```
