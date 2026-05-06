# Vagrant Agent: Implementation Plan and Repo Sketch

## Implementation stance

Do **not** build a new agent harness or a new LLM serving engine. Build Vagrant Agent as a **state-mobility layer** between agent harnesses and serving/runtime backends.

Use existing systems for:

1. **trace generation** from agent harnesses;
2. **runtime calibration** of prefill/prefix/cache costs;
3. **baselines** such as request-level routing, prefix-aware routing, session affinity, and workflow-aware scheduling.

Build your own primitive:

> **Serving Group Manifest** — a structured representation of work nodes, state objects, sharing relationships, warm-state locations, and materialization options.

The implementation should answer:

> Given an agent workflow graph and a placement-change event, what should move together, what should split, and how should each state object be materialized at the destination?

## How to reuse `ledger_progress`

The provided `ledger_progress` README has several invariants and modules that map cleanly to Vagrant Agent.

### Ledger invariant reuse

Existing invariant:

> Event log is the source of truth.

Vagrant invariant:

> The agent trace event log is the source of truth. Serving groups, state graphs, manifests, metrics, and movement plans are derived by replay.

Existing invariant:

> Progress is about discovered active leaf work, not final correctness.

Vagrant adaptation:

> Active work is the set of discovered runnable leaf nodes in the agent workflow graph: active LLM calls, tool calls, subagents, summaries, tests, and workspace operations. Movement planning should reason over active leaf work and its required state, not final task correctness.

Existing invariant:

> Reopen/split/invalidate semantics are first-class and can reduce progress.

Vagrant adaptation:

> Agent workflows can split into subagents, reopen prior work after failed tests, invalidate summaries or tool outputs, and rewrite workspace artifacts. These events must be first-class because they change state dependencies and serving-group boundaries.

### Module mapping

| `ledger_progress` module | Reuse idea in Vagrant Agent |
|---|---|
| `core.py` | Event/status/category enums; replayable state transitions. |
| `session.py` | Append-only trace writing API for agent events. |
| `serialization.py` | JSONL/dataclass serialization for traces and manifests. |
| `run_manager.py` | Filesystem run layout for experiments and benchmark outputs. |
| `sidecar.py` | Live instrumentation helpers for agent harnesses. |
| `queries.py` | Reusable queries over traces: active nodes, leaf work, subagent trees, shared-state users. |
| `scoring.py` | Replace “progress scoring” with statefulness/mobility metrics. |
| `adapters/` | Harness adapters: OpenHands/SWE-agent/LangGraph/CrewAI/custom. |
| `set_*` | Multi-run experiment sets and sweeps. |

### Recommendation

If `ledger_progress` is available as an internal dependency, create Vagrant as a sibling package that imports its session, serialization, run-manager, and sidecar patterns. If not, port the minimal subset:

- append-only JSONL event log;
- dataclass serialization;
- replay engine;
- run directory manager;
- sidecar instrumentation hooks.

Do not mutate derived state in place. Emit events, replay them, and compute derived structures.

## Repo goals

The repository should support three workflows:

### Workflow 1 — Trace collection

```bash
agent-migrate-trace start --run runs/task_001
# agent harness executes and emits events
agent-migrate-trace summarize runs/task_001/trace.jsonl
```

Output:

```text
trace.jsonl
state_objects.json
manifest.json
statefulness_metrics.csv
```

### Workflow 2 — Offline planning/simulation

```bash
agent-migrate-plan \
  --manifest runs/task_001/manifest.json \
  --sites configs/sites_4site.yaml \
  --models configs/model_profiles.yaml \
  --policy adaptive_serving_group \
  --out runs/task_001/plans/adaptive
```

Output:

```text
placement_plan.json
materialization_plan.json
results.csv
queue_timeseries.csv
state_breakdown.csv
```

### Workflow 3 — Benchmark sweeps

```bash
agent-migrate-bench \
  --trace-set runs/set_001 \
  --site-config configs/sites_4site.yaml \
  --policies request_level,prefix_aware,session_sticky,workspace_sticky,adaptive \
  --out results/bench_001
```

Output:

```text
summary.csv
recovery_cdf.png
state_transfer_breakdown.png
duplication_factor.png
mode_mix.png
queue_timeseries.png
```

## Proposed repo layout

```text
agent-migrate-agent/
  README.md
  pyproject.toml
  configs/
    model_profiles.yaml
    sites_4site.yaml
    policies.yaml
  docs/
    research_plan.md
    trace_schema.md
    serving_group_manifest.md
    optimization_problem.md
    benchmark_plan.md
  examples/
    traces/
      toy_agent_trace.jsonl
    manifests/
      toy_manifest.json
  src/
    agent_migrate_agent/
      __init__.py
      core.py
      schema.py
      serialization.py
      session.py
      replay.py
      segments.py
      state_graph.py
      manifest.py
      materialization.py
      costs.py
      sites.py
      queues.py
      policies.py
      planner.py
      metrics.py
      benchmark.py
      run_manager.py
      cli.py
      adapters/
        __init__.py
        ledger_progress.py
        openhands.py
        swe_agent.py
        langgraph.py
        crewai.py
        synthetic.py
      plots.py
  tests/
    test_schema_roundtrip.py
    test_replay.py
    test_segments.py
    test_materialization_costs.py
    test_metrics.py
    test_planner_toy.py
```

## Core data model

### Event types

Initial event taxonomy:

```python
class EventType(str, Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"

    NODE_DISCOVERED = "node_discovered"
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_FAIL = "node_fail"
    NODE_INVALIDATE = "node_invalidate"

    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"

    SUBAGENT_SPAWN = "subagent_spawn"
    SUBAGENT_RESUME = "subagent_resume"
    SUBAGENT_RETURN = "subagent_return"

    STATE_DECLARE = "state_declare"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    STATE_INVALIDATE = "state_invalidate"

    SEGMENT_OBSERVED = "segment_observed"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    SUMMARY_CREATED = "summary_created"

    PLACEMENT_DECISION = "placement_decision"
    MATERIALIZATION_PLAN = "materialization_plan"
    MIGRATION_START = "migration_start"
    MIGRATION_END = "migration_end"
```

### State layers

```python
class StateLayer(str, Enum):
    MODEL_EXECUTION = "model_execution"      # KV, prefix, decode state
    PROMPT_CONTEXT = "prompt_context"        # transcript, retrieved snippets
    SUBAGENT = "subagent"                    # child-local transcript/state
    WORKSPACE = "workspace"                  # repo, diff, build/test artifacts
    MEMORY = "memory"                        # persistent or vector memory
    SEMANTIC = "semantic"                    # plan, subgoals, checkpoint
```

### Materialization modes

```python
class MaterializationMode(str, Enum):
    WARM_REUSE = "warm_reuse"
    KV_TRANSFER = "kv_transfer"
    CONTEXT_REPLAY = "context_replay"
    TEXT_TRANSFER = "text_transfer"
    ARTIFACT_COPY = "artifact_copy"
    WORKSPACE_HYDRATE = "workspace_hydrate"
    REMOTE_WORKSPACE = "remote_workspace"
    SUMMARY = "summary"
    RESTART = "restart"
```

### Trace event

```python
@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    timestamp_s: float
    event_type: EventType
    workflow_id: str
    node_id: str | None = None
    parent_node_id: str | None = None
    agent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
```

### Prompt segment

```python
@dataclass(frozen=True)
class PromptSegment:
    segment_id: str
    name: str
    layer: StateLayer
    tokens: int
    bytes: int | None
    content_hash: str
    lifetime: str             # persistent/shared/private/ephemeral
    producer_node_id: str | None
```

### State object

```python
@dataclass
class StateObject:
    state_id: str
    layer: StateLayer
    tokens: int = 0
    bytes: int = 0
    content_hash: str | None = None
    producers: set[str] = field(default_factory=set)
    consumers: set[str] = field(default_factory=set)
    materialization_modes: set[MaterializationMode] = field(default_factory=set)
    warm_sites: set[str] = field(default_factory=set)
```

### Work node

```python
@dataclass
class WorkNode:
    node_id: str
    workflow_id: str
    node_type: str             # llm_call/tool_call/subagent/summary/test/etc.
    parent_node_id: str | None
    agent_id: str | None
    status: str
    required_state: set[str] = field(default_factory=set)
    produced_state: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    start_time_s: float | None = None
    end_time_s: float | None = None
```

### Serving Group Manifest

```python
@dataclass
class ServingGroupManifest:
    serving_group_id: str
    workflow_id: str
    nodes: dict[str, WorkNode]
    state_objects: dict[str, StateObject]
    edges: list[StateEdge]
    candidate_boundaries: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Replay pipeline

The replay engine should consume `trace.jsonl` and produce:

```text
WorkflowState
StateObject inventory
StateGraph
ServingGroupManifest
```

Replay responsibilities:

1. Maintain current active nodes.
2. Track parent-child/subagent relationships.
3. Track state object producers/consumers.
4. Track prompt segment reuse via hashes.
5. Track invalidations and summary replacements.
6. Build edges between nodes that share state.
7. Mark clean movement boundaries.

Important invariant:

> The manifest is derived. If it is wrong, fix event collection or replay; do not hand-edit the manifest as source of truth.

## First MVP scope

The first implementation should be intentionally narrow.

### MVP workload representation

Support only:

- LLM call events;
- prompt segments with token counts and hashes;
- tool call start/end;
- subagent spawn/return;
- workspace read/write byte counts;
- summary events.

Skip initially:

- actual KV dumping;
- actual multi-region serving;
- real-time scheduling;
- semantic quality scoring;
- real workspace hydration.

### MVP metrics

Implement:

1. repeated prefix fraction;
2. subagent shared-context fraction;
3. duplication factor;
4. state-layer breakdown;
5. mobility-boundary counts;
6. time between model calls;
7. number of sites per workflow under simulated placement.

### MVP policies

Implement:

1. `request_level`
2. `session_sticky`
3. `prefix_group`
4. `subagent_group`
5. `workspace_group`
6. `adaptive_shared_state`

### MVP materialization costs

Implement:

- KV transfer cost;
- context replay cost;
- text transfer cost;
- artifact copy cost;
- workspace hydrate as byte-copy approximation.

## Cost model details

### Model profile

```yaml
models:
  GLM5_like:
    active_params_b: 40
    kv_bytes_per_token: 1277952
    prefill_tok_s_at_100k: 13300
    notes: "heavy KV, cheap-ish attention"

  DeepSeekV4Pro_like:
    active_params_b: 49
    kv_bytes_per_token: 70656
    prefill_tok_s_at_100k: 1600
    notes: "compact KV, expensive attention"
```

### Site profile

```yaml
sites:
  phoenix:
    prefill_tok_s: 30000
    decode_slots: 500
    ingress_gbps: 100
    egress_gbps: 100
    workspace_io_gbps: 20

  seattle:
    prefill_tok_s: 45000
    decode_slots: 600
    ingress_gbps: 100
    egress_gbps: 100
    workspace_io_gbps: 20

links:
  phoenix->seattle:
    rtt_ms: 30
    effective_gbps: 25
```

### Materialization cost

For state object `s` with `T` tokens:

```python
kv_transfer_s = 8 * T * kv_bytes_per_token / link_bps
context_replay_s = T / destination_prefill_tok_s
text_transfer_s = 8 * text_bytes / link_bps
artifact_copy_s = 8 * artifact_bytes / link_bps
```

For shared state, cost is paid once per destination site where it is materialized, not once per consuming node.

This fixed-cost sharing is the core difference from ordinary request-level load balancing.

## Optimization formulation

Given:

- work nodes `V`;
- state objects `S`;
- sites `K`;
- dependency matrix `A[v, s] = 1` if node `v` needs state `s`;
- materialization modes `M_s` for each state `s`.

Decision variables:

```text
x[v,k]     = 1 if node v is placed at site k
y[s,k]     = 1 if state s is materialized at site k
z[s,k,m]   = 1 if state s is materialized at site k using mode m
```

Constraints:

```text
sum_k x[v,k] = 1
x[v,k] * A[v,s] <= y[s,k]
sum_m z[s,k,m] = y[s,k]
```

Objective:

```text
minimize:
  lambda_time      * recovery_time
+ lambda_network   * network_bytes
+ lambda_prefill   * prefill_gpu_seconds
+ lambda_workspace * workspace_io_bytes
+ lambda_dup       * duplicated_shared_state
+ lambda_fidelity  * semantic_loss
```

Capacity constraints:

```text
network bits per link <= link capacity over horizon
prefill GPU-seconds per site <= prefill capacity over horizon
workspace I/O per site <= workspace capacity over horizon
resident KV/state bytes <= memory capacity
```

Offline oracle can solve a small relaxed version. Online policy should be heuristic.

## Online heuristic policy

### Adaptive Shared-State Policy

1. Build a graph where nodes are work items and edge weights are shared-state cut costs.
2. Estimate cost of cutting each edge under the current architecture and destination options.
3. Agglomeratively merge nodes into serving groups when shared-state cut cost exceeds expected placement benefit from splitting.
4. For each candidate group/site, choose cheapest feasible materialization modes.
5. Place groups by projected bottleneck recovery time:

```text
T_group_site = max(
  network_queue_time,
  prefill_queue_time,
  workspace_queue_time,
  decode_admission_time
)
```

6. Reserve virtual destination resources immediately.
7. Emit a placement and materialization plan.

## Metrics implementation

### Repeated prefix fraction

For each workflow:

```text
sum(tokens of repeated/shared segments) / sum(all input tokens)
```

A segment is repeated if its content hash appears in more than one LLM call.

### Subagent shared-context fraction

For each parent/subagent fanout:

```text
shared_tokens / (shared_tokens + sum(private_tokens_per_subagent))
```

### Duplication factor

For a placement plan:

```text
actual materialized shared-state cost / grouped lower-bound materialized shared-state cost
```

### State-layer breakdown

Aggregate materialization costs by:

```text
model_execution
prompt_context
subagent
workspace
memory
semantic
```

### Time to useful restart

First time at which a moved workflow has enough required state to execute its next LLM/tool action.

### Time to full recovery

Time at which all required state objects for all moved nodes are materialized.

## CLI sketch

### `agent-migrate-trace`

```bash
agent-migrate-trace init --run runs/demo
agent-migrate-trace append runs/demo/trace.jsonl --event '{...}'
agent-migrate-trace summarize runs/demo/trace.jsonl
```

### `agent-migrate-manifest`

```bash
agent-migrate-manifest build \
  --trace runs/demo/trace.jsonl \
  --out runs/demo/manifest.json
```

### `agent-migrate-plan`

```bash
agent-migrate-plan \
  --manifest runs/demo/manifest.json \
  --sites configs/sites_4site.yaml \
  --models configs/model_profiles.yaml \
  --policy adaptive_shared_state \
  --out runs/demo/plans/adaptive
```

### `agent-migrate-bench`

```bash
agent-migrate-bench \
  --trace-set runs/set_001 \
  --policies request_level,prefix_group,session_sticky,adaptive_shared_state \
  --out results/bench_001
```

## First coding-agent task list

Give a coding agent these tasks in order.

### Task 1 — Create skeleton package

- Create `pyproject.toml`.
- Create `src/agent_migrate_agent/`.
- Add `core.py`, `schema.py`, `serialization.py`, `session.py`.
- Add basic JSONL read/write.
- Add tests for dataclass roundtrip.

Acceptance criteria:

```text
pytest passes
trace events can be written to JSONL and read back losslessly
```

### Task 2 — Implement trace session API

- `TraceSession.start_workflow()`
- `TraceSession.record_llm_call()`
- `TraceSession.record_tool_call()`
- `TraceSession.record_subagent_spawn()`
- `TraceSession.record_subagent_return()`
- `TraceSession.record_state_read/write()`
- `TraceSession.close()`

Acceptance criteria:

```text
examples/traces/toy_agent_trace.jsonl can be generated by a script
```

### Task 3 — Implement prompt segment hashing

- Define `PromptSegment`.
- Hash text if provided.
- Allow count-only segments when text is unavailable.
- Compute repeated segment statistics.

Acceptance criteria:

```text
shared segment hashes are detected across multiple LLM calls
```

### Task 4 — Implement replay engine

- Replay trace events into `WorkflowState`.
- Track nodes, state objects, producers, consumers.
- Track active leaf nodes.
- Track invalidations.

Acceptance criteria:

```text
toy trace replays into expected nodes and state objects
```

### Task 5 — Implement manifest builder

- Convert `WorkflowState` into `ServingGroupManifest`.
- Build edges based on shared state consumers.
- Emit JSON manifest.

Acceptance criteria:

```text
manifest contains nodes, state objects, and shared-state edges
```

### Task 6 — Implement metrics

- repeated prefix fraction;
- subagent shared-context fraction;
- duplication factor lower bound;
- state-layer breakdown;
- mobility-boundary counts.

Acceptance criteria:

```text
metrics command produces CSV on toy trace
```

### Task 7 — Implement materialization cost model

- Model profile loader.
- Site/link config loader.
- KV transfer cost.
- Context replay cost.
- Text/artifact copy cost.

Acceptance criteria:

```text
cost tests reproduce known crossover examples
```

### Task 8 — Implement baseline policies

- request-level;
- session-sticky;
- prefix-group;
- subagent-group;
- workspace-group;
- adaptive shared-state heuristic.

Acceptance criteria:

```text
planner emits placement/materialization plans for toy manifest
```

### Task 9 — Implement benchmark runner

- Run multiple policies on same manifest.
- Save `results.csv` and breakdowns.
- Plot recovery CDF and transfer breakdown.

Acceptance criteria:

```text
agent-migrate-bench runs end-to-end on toy traces
```

## Initial toy trace

The first toy trace should intentionally demonstrate serving-group behavior.

Scenario:

```text
workflow: coding task
parent planner call uses shared repo context
planner spawns 3 subagents
all subagents share repo context and tool definitions
subagent A has small private context
subagent B has large private context
subagent C reads/writes workspace artifacts
parent merges results
```

Expected:

- request-level routing duplicates shared repo context across subagents;
- session-sticky pays shared context once but may overconstrain placement;
- adaptive policy groups A/C with shared workspace and allows B to split if private context dominates.

## Adapter strategy

### Synthetic adapter first

Start with `adapters/synthetic.py` to generate traces with controlled shared/private context.

Parameters:

```text
num_subagents
shared_context_tokens
private_context_distribution
workspace_bytes
summary_tokens
fanout_depth
```

This lets you debug metrics and policies before integrating real harnesses.

### Ledger adapter

If `ledger_progress` exists, add `adapters/ledger_progress.py` to import event logs and translate:

```text
progress leaf work -> work nodes
split events -> subagent/work split
reopen events -> invalidation/retry
sidecar events -> tool/workspace/LLM observations
```

### Real harness adapters later

Implement after the synthetic pipeline works:

```text
OpenHands / SWE-agent / LangGraph / CrewAI / custom coding agent
```

Do not block MVP on real harness integration.

## Benchmark outputs

Each benchmark run directory should include:

```text
trace.jsonl
manifest.json
policy_name/
  placement_plan.json
  materialization_plan.json
  results.csv
  queue_timeseries.csv
  state_breakdown.csv
  mode_mix.csv
  plots/
    recovery_cdf.png
    transfer_breakdown.png
    duplication_factor.png
    queue_timeseries.png
```

## Non-goals for initial repo

- No real cloud deployment.
- No live migration of actual KV tensors.
- No packet-level network simulation.
- No power-grid model.
- No claim of semantic correctness preservation.
- No dependence on private/leaked provider internals.

## Definition of a successful MVP

The MVP succeeds if it can:

1. generate or ingest an agent trace;
2. replay it into a state graph;
3. compute statefulness metrics;
4. build a serving-group manifest;
5. estimate materialization costs under multiple model/site profiles;
6. compare fixed group policies and adaptive grouping;
7. report data transferred and time to useful restart for a movement event.

At that point, you can decide whether the traces show a real enough phenomenon for a paper.

