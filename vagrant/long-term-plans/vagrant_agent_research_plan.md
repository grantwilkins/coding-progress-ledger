# Vagrant Agent: Research Plan and Paper Hypotheses

## One-sentence thesis

**Vagrant Agent studies state-boundary-aware movement of agentic serving groups.** An agentic workflow is not just a stream of independent LLM requests; it is a dynamic graph of model calls, tool calls, subagents, memory operations, and workspace artifacts whose placement decisions are coupled by shared state. The central problem is deciding which parts of the graph should move together and which state representation should be transferred, replayed, copied, summarized, or restarted.

## Why this project exists

The initial observation was that KV caches are much larger than prompt/context text, so it seems cheaper to transmit context and recompute KV at a destination site. The single-request study complicated that story: the correct choice depends on model architecture. KV bytes per token and prefill FLOPs per token can push in opposite directions, so the KV-transfer versus context-replay crossover bandwidth varies dramatically across models.

That result is useful, but it is not yet a systems paper by itself. The larger systems problem is that Codex/Claude-Code-like workloads are not single uninterrupted decode sessions. They are long-lived logical jobs composed of many model calls, tool executions, file reads/writes, tests, memory updates, summaries, and often subagent branches. The durable state of such a job is split across several layers:

| State layer | Examples | Movement question |
|---|---|---|
| Model execution state | KV cache, prefix cache, decode state | Transfer KV, reuse warm prefix, or replay context? |
| Prompt/context state | transcript, retrieved snippets, tool outputs, summaries | Move text, replay prompt, compact, or duplicate? |
| Subagent state | child transcript, private tool results, resume handle | Keep with parent, split, or summarize return state? |
| Workspace/artifact state | repo checkout, worktree, diffs, logs, build/test cache | Copy diff, hydrate workspace, remote mount, or pin? |
| Semantic task state | plan, subgoals, success criteria, partial conclusions | Restart, delegate, summarize, or checkpoint? |

The key hypothesis is that the **schedulable unit** of agentic work is not always a request and not always a whole session. It is a **serving group**: a set of work items whose placement should be coupled because they share expensive or correctness-relevant state.

## Relationship to existing work

Vagrant Agent should not claim that workflow-aware agent serving is new. Existing systems already cover large parts of the space:

- Application-level LLM dataflow and semantic variables.
- Program/workflow-aware scheduling of agent calls.
- Agent workflow query plans and cache-aware execution.
- KV/prefix-aware routing and cache management.
- Multi-agent shared KV reuse.
- Stateful workflow/serverless placement.

The intended wedge is narrower:

> Prior work mostly optimizes **how an agent graph executes where it is already running**. Vagrant Agent studies **how an already-running agentic serving group changes placement**, and what state boundary is used to rematerialize that group elsewhere.

Short version:

> Prior work schedules agent graphs. Vagrant Agent migrates, splits, and rematerializes agent graphs.

## Core research questions

1. **How architecture-dependent is model-state mobility?**  
   For a single migrated request, when does KV transfer beat context replay?

2. **Are agentic workloads actually stateful serving groups?**  
   How much context, artifact, memory, workspace, and subagent state is shared across calls?

3. **What is the right schedulable unit?**  
   Should placement operate at the request, prefix group, subagent, session, workspace, or adaptive serving-group level?

4. **What state should move versus be recomputed?**  
   For each state object, should the runtime transfer KV, replay context, copy artifacts, hydrate a workspace, summarize, or restart?

5. **What happens when many groups move at once?**  
   Does independent redispatch duplicate shared state? Does full stickiness overconstrain placement? Can state-aware serving groups reduce restart time and data movement?

## Hypotheses

### H1 — Single-request rematerialization is architecture-dependent

For a single request, the choice between KV transfer and context replay is governed by an architecture/link exchange rate, not only by context length.

For context length `T`, KV bytes per token `s_kv`, effective bandwidth `B`, and prefill rate `r_prefill`:

```text
t_KV     = 8 * T * s_kv / B
t_replay = T / r_prefill
```

Replay wins when:

```text
B < 8 * s_kv * r_prefill
```

`T` cancels under the simple full-transfer/full-replay model. This means the decision is governed by KV bytes per token, prefill throughput, and link bandwidth. Your initial analysis already suggests crossover bandwidth spans more than two orders of magnitude across model architectures.

### H2 — Agentic jobs are not independent request streams

Coding-agent-like tasks repeatedly reuse prompt prefixes, project instructions, tool definitions, retrieved files, plans, subagent instructions, and workspace artifacts. If this repeated state is materialized independently at many sites, the system duplicates prefill and data transfer.

### H3 — Full stickiness is also wrong

Pinning the whole workflow/session to one site preserves locality, but loses flexibility and may overload a destination during redispatch. A whole session may be too coarse: some subagents or branches can move independently if their private state dominates shared state.

### H4 — The right serving-group boundary depends on state structure and model architecture

The same agent graph may prefer different placement policies under different architectures. A heavy-KV/cheap-prefill model may favor replaying shared context at several destinations; a compact-KV/expensive-prefill model may favor preserving or transferring KV/prefix locality.

### H5 — Movement should be evaluated by useful restart, not just transfer completion

The user-visible metric is not simply “all bytes moved.” A workflow can sometimes resume useful work after a subset of state is available. Vagrant should measure both:

- **time to useful restart**: when the workflow can produce its next meaningful action;
- **time to full recovery**: when all required state is available at the new placement.

## The serving group abstraction

A serving group is a dynamic state-coupled unit of agentic work.

Formally:

```text
G = (V, E, S, M)
```

where:

- `V` are work nodes: LLM calls, tool calls, subagents, memory ops, summaries, tests/builds.
- `E` are dependencies and sharing edges: parent-child dependency, shared prefix, shared artifact, shared workspace, subagent return, memory dependency.
- `S` are state objects: KV/prefix, transcript, retrieved docs, tool outputs, workspace diffs, persistent memory, semantic plan.
- `M` are materialization modes: warm reuse, KV transfer, context replay, text transfer, artifact copy, workspace hydrate, summary, restart.

The optimizer chooses:

1. a partition of `V` into one or more movable groups;
2. placement of each group at destination sites;
3. materialization modes for each state object at each site.

## Big experimental phases

### Phase 0 — Freeze the KV/context result

Purpose: establish the motivating observation.

Deliverables:

- Single-request KV-transfer versus context-replay model.
- Architecture table: KV bytes/token, prefill FLOPs/token, derived prefill rate, crossover bandwidth.
- Figures: crossover bandwidth by model; transmission/replay time by bandwidth.

This should be treated as **Observation 1**, not the whole paper.

### Phase 1 — Define the state taxonomy

Purpose: define what can move.

Deliverables:

- State layer taxonomy.
- State object schema.
- Materialization mode taxonomy.
- “Who owns/reads/writes/moves this state?” table.

### Phase 2 — Build trace harness

Purpose: measure real or realistic agent data flows.

The trace harness should log:

- LLM calls and prompt segments.
- Tool calls and results.
- File reads/writes and workspace diffs.
- Subagent spawn/resume/return.
- Memory reads/writes.
- Summaries and compaction events.
- Parent-child workflow structure.

For privacy and portability, the harness can store segment hashes and token/byte counts rather than raw prompt text.

### Phase 3 — Run controlled agent workloads

Purpose: obtain traces with different agent structures.

Modes:

- Single-agent.
- Planner-worker.
- Multi-subagent fanout.
- Reviewer/implementer/tester.
- Shared memory enabled.
- Aggressive summarization.
- Workspace-heavy coding tasks.

Task types:

- Bug fix.
- Test repair.
- Multi-file refactor.
- Documentation update.
- Feature implementation.
- Benchmark/evaluation.
- Dependency/config fix.
- Large repo search and edit.

### Phase 4 — Measure whether serving groups are real

Purpose: falsify or support the serving-group premise.

Metrics:

1. **Repeated prefix fraction**

   ```text
   repeated/shared prompt tokens / total input tokens
   ```

2. **Subagent shared-context fraction**

   ```text
   shared tokens / (shared tokens + private subagent tokens)
   ```

3. **Duplication factor under independent routing**

   ```text
   materialized shared-state cost under independent placement
   / minimum materialized shared-state cost under grouped placement
   ```

4. **State-layer dominance**

   Breakdown of model-state, context-state, artifact-state, workspace-state, and semantic-state costs.

5. **Mobility-boundary frequency**

   Count clean movement points: before/after LLM calls, after tools, after tests, after subagent completion, after summary/compaction.

Kill criterion: if shared state is small and duplication factors are near 1, serving groups may not matter.

### Phase 5 — Build state graph extractor

Purpose: turn traces into analyzable graphs.

Nodes:

- LLM call.
- Tool call.
- Subagent.
- Memory operation.
- Workspace operation.
- Summary/compaction.
- Test/build run.

Edges:

- Parent-child dependency.
- Shared prefix.
- Shared retrieved docs.
- Shared tool outputs.
- Shared workspace.
- Shared memory.
- Subagent return.
- File dependency.

Edge weights:

- Shared tokens.
- KV-equivalent bytes.
- Artifact bytes.
- Workspace bytes.
- Expected replay FLOPs.
- Communication frequency.
- Semantic dependency strength.

### Phase 6 — Define group baselines

Compare fixed group levels:

| Policy | Description |
|---|---|
| Request-level | Every LLM call independent. |
| Prefix group | Calls sharing long prefix are grouped. |
| Subagent group | Each subagent is a unit. |
| Session group | Whole user task/session is a unit. |
| Workspace group | All work sharing a workspace/worktree is grouped. |
| Adaptive serving group | Graph partition based on shared-state weights and placement pressure. |

### Phase 7 — Attach materialization costs

Purpose: make movement concrete.

For each state object `s`, destination `k`, and mode `m`, estimate:

- network bytes;
- prefill GPU-seconds;
- workspace I/O bytes;
- decode admission cost;
- latency;
- fidelity/semantic-risk penalty.

Model state uses the KV/context crossover equations. Artifact/workspace state uses byte-copy, snapshot, diff, or hydration approximations.

### Phase 8 — Simulate redispatch without power

Purpose: study movement independent of grid modeling.

Event:

> A set of active agentic serving groups must be redistributed from one source site to one or more destination sites.

Site resources:

- prefill capacity;
- decode capacity;
- ingress/egress bandwidth;
- workspace I/O capacity;
- artifact storage/cache;
- warm prefixes/workspaces.

Outputs:

- time to useful restart;
- time to full recovery;
- data transferred by state layer;
- prefill GPU-seconds;
- duplicated shared state;
- queue buildup;
- cache hit/loss;
- sites per workflow.

### Phase 9 — Compare policies

Baselines:

- Request-level least-loaded.
- Prefix-aware routing.
- Session/program affinity.
- Workspace stickiness.
- Subagent-independent routing.
- KV-only movement.
- Replay-only movement.
- Offline oracle/relaxed optimizer.
- Vagrant adaptive serving groups.

Primary metrics:

- time to useful restart;
- time to full recovery;
- total bytes transferred;
- KV bytes transferred;
- text/context bytes transferred;
- artifact/workspace bytes transferred;
- prefill GPU-seconds;
- duplication factor;
- destination queue pressure.

### Phase 10 — Herd movement

Purpose: stress correlated migration.

Setup:

- Many workflows/subagents move at once.
- Shared contexts and workspaces create correlated state.
- Destinations have finite prefill, ingress, decode, and workspace capacity.

Expected failure modes:

- Request-level redispatch duplicates shared state.
- Replay-only creates destination prefill storms.
- KV-only creates network/ingress pressure.
- Full session stickiness creates hot destinations.
- State-aware grouping balances shared-state reuse and placement flexibility.

## Central figures for a paper

1. **Architecture crossover plot**  
   KV-transfer versus context-replay crossover bandwidth by model.

2. **Statefulness of agentic traces**  
   CDFs of repeated prefix fraction, subagent shared-context fraction, and duplication factor.

3. **State-layer breakdown**  
   For each workload: model state, transcript/context, artifact, workspace, semantic state.

4. **Group granularity comparison**  
   Request-level, prefix, subagent, session, workspace, adaptive group.

5. **Phase diagram**  
   x-axis: shared-state fraction.  
   y-axis: destination load imbalance.  
   color: best movement strategy.  
   panels: different model architectures.

6. **Workflow movement benchmark**  
   Time to useful restart and total data transferred under each policy.

7. **Herd redistribution**  
   Recovery CDFs and queue buildup under many simultaneous movements.

## Expected contribution list

A plausible paper contribution list:

1. **Architecture-dependent state mobility.**  
   A quantitative model and catalog showing KV-transfer/context-replay crossovers vary substantially across model architectures.

2. **Agentic state-flow measurement.**  
   A trace schema and measurement study showing how context, subagent state, memory, and workspace artifacts are shared across agentic workflows.

3. **Serving group abstraction.**  
   A formalization of dynamic state-coupled groups as the schedulable unit for mobile agentic workloads.

4. **State materialization model.**  
   A model for choosing KV transfer, context replay, artifact copy, workspace hydration, summary, or restart.

5. **Movement planner and benchmark.**  
   A trace-driven evaluation comparing request-level, prefix-aware, session-sticky, workspace-sticky, and adaptive serving-group movement policies.

## Non-goals for the first version

Do not start with:

- packet/NIC-level modeling;
- full power-grid modeling;
- new inference engine;
- new agent framework;
- production multi-region deployment;
- live token-level KV migration as the central mechanism;
- leaked/private source code.

The first version should be a trace-driven state-mobility framework with measured/calibrated costs.

## Go/no-go criteria

Proceed if:

- repeated/shared state is substantial;
- independent routing creates significant duplication;
- different group boundaries win in different regimes;
- architecture changes materialization choices;
- adaptive grouping improves restart time and/or data movement.

Pivot if:

- agent traces are mostly independent calls;
- workspace state dominates so strongly that model-state locality is secondary;
- summarization eliminates most state affinity;
- prefix-aware routing already captures nearly all benefit;
- queueing dominates state materialization in all realistic settings.

