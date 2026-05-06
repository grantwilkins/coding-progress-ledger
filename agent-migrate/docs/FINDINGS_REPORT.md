# Findings and Implementation Report — Vagrant

**Date:** 2026-05-06
**Scope:** What this codebase is, what has been built, and what the experiments have established about state mobility for LLM-agent workloads. Written for a reader who wants the systems-level claims without project shorthand.

---

## 1. What the system is

Vagrant is a derivation pipeline that takes:

1. an append-only event log of an agent workflow (planner LLM call, tool calls, subagent calls, and the prompt/workspace/artifact state those nodes read and write), and
2. a placement-change event ("these workflows must resume on different machines"),

and produces:

1. a **placement plan**: which physical site each compute node runs at,
2. a **materialization plan**: for each piece of state, how it is made available at the destination site (re-prefill from text, transfer the KV cache, copy a workspace artifact, hydrate a working tree from a globally available source, or reuse a copy already resident at the destination), and
3. a **cost estimate**: per-policy seconds-to-resume under a finite-resource model of the destination.

It is not a serving engine, not a scheduler, and not an agent harness. It is a measurement and analysis framework for asking: *given an agentic workload and a forced relocation event, what does the destination actually have to do, how long does that take, and which placement strategy minimizes that time?*

The single experimental claim it tries to demonstrate, on real or realistic traces: **routing requests as if they were independent gives a different (and often worse) plan than routing them as a graph of nodes that share state**.

---

## 2. Implementation overview

The codebase has three layers stacked on each other.

### 2.1 Trace → manifest

A trace is a JSONL event log produced either by a synthetic adapter or by post-processing real SWE-agent rollouts. Events declare state objects, record reads/writes, mark node start/end, and (in the mobility setting) carry source-site / destination-site information. The trace is the immutable input; everything else is derived from it.

Replaying the trace produces a **manifest**: a bipartite graph with compute nodes on one side and state objects on the other, plus a derived "pairwise edge" view between any two nodes that share state. The manifest classifies state by:

- **Layer** — prompt context, workspace bytes, KV cache, intermediate artifact, retrieved document, transcript, etc.
- **Lifetime** — persistent (the system prompt of the model), shared across a session, or ephemeral to a single node.
- **Mobility class** — globally available, cheaply rehydratable, must move, can be recomputed, can be discarded.

Manifests are never edited by hand; if the manifest is wrong, the trace or the replay is wrong.

### 2.2 Cost model and per-state materialization modes

Each piece of state can be reconstituted at a destination via one of:

- **CONTEXT_REPLAY** — ship the source text and re-run prefill at the destination GPU,
- **KV_TRANSFER** — ship the destination the precomputed KV tensors,
- **ARTIFACT_COPY** — copy a built file (a workspace, a parquet shard, a model checkpoint),
- **WORKSPACE_HYDRATE** — instantiate a working tree at the destination from a globally available source (a clone, a registry pull, a CDN-served snapshot),
- **WARM_REUSE** — the destination already holds a usable copy.

The cost model charges, per chosen mode, a closed-form combination of bytes-over-wire, prefill tokens, workspace-hydrate bytes, KV resident bytes, and wall-clock seconds. The token count cancels in the prompt-replay-vs-KV-transfer comparison; the crossover lives entirely in `(link bandwidth, KV bytes per token, destination prefill rate)`.

Four documented simplifications, consciously accepted:

1. Costs are additive over states (real systems pipeline transfer with prefill — biases against grouped policies).
2. Decode time is omitted (cancels in same-trace policy comparisons).
3. KV is charged at raw bytes (no compression — overstates KV transfer by ~3–4× per CacheGen, but stays within the sensitivity sweep range).
4. Faster-prefill bias under unconstrained capacity (a policy that freely picks the fastest site always wins — only meaningful when capacity is finite, which is what the simulator below addresses).

### 2.3 Finite-resource event simulator

For the mobility setting (an entire batch of workflows lands on a finite destination at once), the cost model is too optimistic: it lets every workflow pick the fastest site and pay no contention. The simulator (`fluid_sim.py`, ~360 LOC) replaces that with an event-ordered loop over four resource axes:

- **Network** (bytes/sec, proportionally shared among in-flight transfers),
- **Prefill compute** (tokens/sec at each destination GPU, proportionally shared among concurrent prefills),
- **Workspace hydrate** (bytes/sec at each destination's workspace ingest path),
- **KV memory** (bytes resident at each destination, modeled as a capacity with LRU eviction).

There are no queues, no admission control, no priorities, no preemption. Capacity-bearing resources are modeled fluidly: at any instant each in-flight action gets a proportional share of its bottleneck resource; time advances to the next event boundary. KV memory is the one capacity-with-eviction. This is deliberately the simplest model that still expresses contention — enough to ask "does adding more workflows saturate prefill" without simulating a real serving engine.

The simulator coalesces concurrent materialization of the same `(state, destination)`: if two workflows independently decide to fetch the same artifact, they share the in-flight transfer rather than paying twice.

### 2.4 Policies

Six policies feed the simulator, all producing static per-(workflow, state) plans:

- **min_cost_independent** — every workflow picks its own per-state cheapest mode and destination. The strawman; stampedes the fastest GPU.
- **replay_all** — always re-prefill at the destination from text. Useful as a single-mode bound.
- **kv_all** — always ship the KV cache. The other single-mode bound.
- **cache_reuse (strong per-site reuse)** — for each piece of state, if any destination already has it warm, reuse; otherwise pick the cheapest cold mode; route each workflow to the destination with the most warm hits, breaking ties on lowest cold cost. This is the serious baseline; everything below has to beat it to be interesting.
- **workspace_sticky** — keep workflows whose workspace artifact is large bound to wherever that workspace can hydrate cheapest.
- **mixed_min_pressure** — greedy load-aware planner: order workflows, score each candidate (mode, destination) pair by the predicted maximum across `network_share`, `prefill_share`, `workspace_share`, pick the pair that minimizes the worst predicted resource utilization.
- **random_mode** — sanity baseline; samples mode and destination uniformly. Surprisingly strong on slow-link / large-state cells because it accidentally splits load across destinations.

### 2.5 Auxiliary tooling

- A 5-profile model-architecture catalog (`configs/model_profiles.yaml`) covering the 2025–2026 plausible range: Kimi-K2.6-class compact MLA, Llama-3-70B GQA, DeepSeek-V4-Pro CSA, GLM-5 MLA+DSA, Qwen3-Next-80B-A3B hybrid. Each profile carries a single-stream prefill rate cross-checked against published model configs and a corrected per-token KV byte count. The earlier "GLM-5 1.28 MB/token" number was full-MHA arithmetic on a model that is in fact MLA; corrected.
- A small-N **oracle** that exhaustively enumerates `(destination, prompt-mode, workspace-mode)` per workflow for episodes up to 4 workflows × 2–3 destinations. Used to measure how much headroom the heuristics leave on the table.
- A **regime-map sweep** that runs every policy across `N workflows × workspace scale × prefill capacity × link bandwidth`. The aggregate version uses a closed-form service-time estimator so 1K–10K-workflow cells are tractable; an exact event-driven path is available for any cell that becomes a quantitative claim.
- **Sensitivity sweep** over `kv_bytes_per_token × link_bps`, used to ask whether a reported gap survives or evaporates as the constants move.
- **Workload anchors** for three families chosen because they stress different state layers: large-repo coding (workspace-dominated, ~1 GB per workflow), data/RAG/artifact-heavy (must-move retrieved documents and intermediates, ~2 GB per workflow with 8 GB globally-available vector index shards), and multi-agent fanout/fanin (many concurrent LLM calls, prefill-saturating).

---

## 3. Findings

The findings below are framed as systems claims. They are stated independently of the project's task numbering.

### 3.1 At realistic SWE-agent scales, cache reuse alone is most of the gain

The original thesis was that *grouping nodes by shared state* changes optimal placement. On a constructed adversarial trace (a planner with three subagents that share two prompt contexts and a workspace artifact), it does — by ~2× duplication factor.

Replacing the "no reuse anywhere" baseline with **per-site materialization reuse** (every node placed at a site reuses any state already materialized there) collapses that 2× gap to zero on every linear-session trace. The grouping policy and the cache-reuse policy produce numerically identical plans on:

- the toy trace,
- a mid-sized synthetic g_demo trace,
- a real SWE-agent rollout (s_07 from SWE-bench).

**Implication.** The "shared-state-aware grouping" claim, framed as superiority over a no-reuse baseline, is partly an indictment of that strawman. Once a destination caches state across colocated nodes, grouping adds nothing on linear sessions.

### 3.2 Grouping does help — but only at byte magnitudes above a threshold the typical SWE-bench workload doesn't reach

Constructed multi-session fixtures concatenate three or five distinct sessions, each with a private workspace artifact, distributed across two source sites. With **synthetic 1 GB per-session workspaces**, grouping (which forces all sessions sharing prompt context into a single component placed at one site) loses to per-site reuse by a margin proportional to the cross-site workspace transfer cost — roughly `2 × 1 GB / 5 Gbps ≈ 3.2 s` per cross-site move. The gap survives the full `(KV/token × bandwidth)` sensitivity grid.

Replacing those synthetic workspace bytes with **real working-tree byte sums from the upstream repos at HEAD** (~33 MB total across five sessions) collapses the gap to numerical noise. The mechanism is real — a recovery test that swaps the same trajectories' bytes back to 1 GB reproduces the 3.2 s gap exactly — but it is **byte-magnitude-sensitive** and falls below threshold for SWE-bench-class repositories at HEAD.

**Implication.** The relevant threshold for "grouping matters" in the placement layer is set by `cross_site_workspace_bytes / link_bps` vs the difference in destination prefill rates. A typical SWE-bench instance is sub-threshold against a 5 Gbps inter-region link. A real running agent with installed pip dependencies, pytest artifacts, and persistent caches would have hundreds of MB of workspace state — easily clearing the threshold. The shallow-clone bytes in the negative test are a measurement artifact, not the production regime.

### 3.3 KV-transfer vs prompt-replay is decided by the model architecture, not the link

Independent crossover study (`kv-transfer-early-experiment/`) computed, for every model in the 2025–2026 catalog, the link bandwidth above which shipping the KV cache beats re-running prefill. The crossover spans **>150×** across the catalog (≈1 Gbps to ≈140 Gbps), driven by two architectural levers pulling in opposite directions:

- **KV bytes per token** — full multi-head attention (e.g., 1.25 MB/token for full MHA) is ~50× heavier than hybrid attention (Qwen3-Next-class, 24 KB/token).
- **Attention compute per token** — large `head_dim` (DeepSeek-V4-Pro: `head_dim=512` and `4 · L · n_q · head_dim · T` attention FLOPs) crashes the destination's prefill rate, which makes replay slow even when the KV is small.

Combining the two: GLM-5 has heavy KV but cheap attention → replay wins below ~136 Gbps; DeepSeek-V4-Pro has light KV but expensive attention → KV-transfer wins above ~1 Gbps. They give opposite migration recommendations on the same physical link.

**Implication.** A mobility planner that picks a single global mode (always replay, always KV-transfer) is wrong by design for any heterogeneous fleet. Even a single-model deployment changes its preferred mode across plausible link tiers (intra-rack, intra-DC, cross-region). This is the strongest model-aware result in the codebase.

### 3.4 Mobility under finite resources is at least four regimes, not one problem

When a batch of stateful workflows lands on a destination at once, the bottleneck shifts as the workload changes. The simulator and sweep make four regimes legible:

| Regime | What dominates time-to-resume | Best policy class |
| ------ | ------------------------------ | ----------------- |
| **Reuse** | state is small or already warm | cache reuse is sufficient |
| **State-locality** | large workspace / artifact bytes must cross the link | per-state mode choice (artifact-copy vs hydrate) and destination preference |
| **Landing-pressure** | many workflows concurrently re-prefill, saturating destination GPUs | mode-mixing (some replay, some KV transfer) so prefill is not the only path |
| **Multi-resource** | network, prefill, workspace are all simultaneously finite | richer load-aware planning has a measurable ceiling |

These are not a podium where one policy wins universally. The same workload changes regime under a slow link vs a fast link, and under a compact-KV model vs a heavy-KV model.

A representative single-source evacuation cell (100 workflows, ~500 MB workspaces, three destinations, finite network/prefill/workspace/KV memory) gives time-to-resume:

```
replay_all              67.7 s     single-mode bound
cache_reuse             67.7 s     strong per-site reuse
kv_all                 177.9 s     single-mode bound
workspace_sticky        48.2 s     workspace-locality wins partway
random_mode             50.5 s     accidental load-spreading
mixed_min_pressure      24.3 s     load-aware planner
```

The load-aware planner finishes ~49% faster than the best single-mode and ~28% faster than the workspace-locality heuristic on this cell. That earns the mobility-episode framing as worth carrying. It does not earn a universal "load-aware planning always wins" claim — the same planner wins by 0.7% on a workspace-pressure cell where the workspace-locality heuristic is already near optimal.

### 3.5 Workload regime is model-architecture-dependent

Cross-running each workload anchor across all five model architectures shows the regime label flips for ~25% of the cells:

- Large-repo coding: 5 of 12 non-baseline rows flip among `state_locality`, `multi_resource`, `landing_pressure`.
- Data/RAG-heavy: 3 of 12 flip.
- Multi-agent fanout: 2 of 12 flip — DeepSeek-V4-Pro's compressed KV moves it from `landing_pressure` to `state_locality`.

**Implication.** Statements like "this workload is a workspace-locality regime" must be qualified by which model architecture it was observed under. The same workload is a different problem on Qwen3-Next-class hybrid attention vs Llama-3-class GQA.

### 3.6 Per-site cache reuse is not the same as per-(state, site) reuse

An edge-typed grouping policy that weights connections by `(state.layer, state.lifetime)` was expected to be strictly ≤ the unweighted grouping policy. It is not. On the realistic-bytes fixture it is **worse**, because zeroing out the global `system_prompt` edge fragments the graph into five components, each of which then materializes the system prompt independently — paying it five times instead of once.

The fix is a structural one in the simulator: charge materialization at **per-(state, destination)**, not per-(component, destination). Grouping policies do not get to amortize a state across nodes that happen to land in the same component; they pay once per state per destination. This is now a load-bearing invariant in the resource model and is enforced by tests.

**Implication.** "Grouping" is the wrong abstraction for charging cost; "what is materialized where" is the right one. The codebase enforces this and the rest of the analysis depends on it.

### 3.7 Most workflow state does not have to move

A walk over a real workflow directory, classified by file path and extension, produces a per-layer breakdown across 11 layers (base repo checkout, uncommitted diff, files read, files touched, tool outputs, test logs, build artifacts, dependency cache, retrieved documents, subagent transcripts, summaries/compaction). Of these:

- **Globally available** (base repo, vector index shards, dependency registries) — does not need to be moved; the destination fetches from a CDN, registry, or shared filesystem.
- **Cheaply rehydratable** (dependency cache, build artifacts) — can be reconstructed at the destination; not required to be on the wire.
- **Must move** (uncommitted diff, in-progress tool outputs, subagent transcripts) — the only layers a migration strictly has to ship.
- **Can be recomputed** (test logs, partial summaries) — drop on the floor; redo if needed.
- **Can be discarded** — drop on the floor; not needed.

For a typical SWE-agent workflow, "must move" bytes are dominated by the uncommitted diff (typically <1 MB) and any in-progress tool output buffers. The base repo (tens to hundreds of MB) and the dependency cache (often ~1 GB) are globally available — they should not be on the migration's critical path.

**Implication.** Reporting "migration payload" as `du -sh` of the workflow directory is wrong by an order of magnitude. The relevant payload is bytes in the `must_move` class. Most production "moves" are cheap if the destination has the same registry mirror and clones the same upstream.

### 3.8 Aggregate timing approximations find regimes; they don't quote numbers

Sweeping the full `N × state-scale × prefill × link` grid with the exact event-driven simulator is too slow at 1K–10K workflows. The codebase carries a closed-form aggregate service-time estimator that is fast enough.

Calibration on 36 sampled cells against the exact simulator:

- Best-policy agreement: 24/36 cells (67%).
- Bottleneck-label agreement: 102/216 policy rows (47%).
- Median relative timing error: 49%.
- Maximum relative timing error: 789%.

Of seven named "claim cells" rerun through the exact simulator, all seven are tagged `needs_exact_k4` — the aggregate often agrees on the winner but is too noisy on timing or bottleneck attribution to be quoted directly.

**Operating rule.** The aggregate map is for finding candidate regimes worth investigating. Any cell that becomes a quoted number must be rerun with the exact event-driven simulator. The map and the validation are deliberately separated: heatmaps for discovery, exact runs for claims.

### 3.9 The current heuristic's ceiling is real but uneven

A 4-workflow, 2-destination, single-source evacuation oracle (exhaustive search over destination, prompt-mode, workspace-mode per workflow — 4096 plans per cell) gives:

| Cell | Oracle vs strong-reuse | Oracle vs load-aware planner |
| ---- | ---------------------: | ---------------------------: |
| Tiny prefill pressure | 56% | 35% |
| Medium multi-resource | 81% | 50% |
| Monorepo / workspace pressure | 50% | 0.7% |
| Slow link / network pressure | 97% | 50% |

Three observations:

1. The oracle beats strong cache reuse by 50–97% in the constrained cells. The ceiling above strong reuse is real here.
2. The current load-aware planner is already near-optimal in the workspace-pressure cell and leaves 35–50% on the table in the prefill, multi-resource, and slow-link cells.
3. The headroom comes primarily from per-workflow destination and prompt-mode choice — not from workspace-mode choice. The planner is already making good per-state mode decisions; it is making suboptimal *which-workflow-goes-where* decisions.

Adding a one-step lookahead to the load-aware planner (penalize a candidate whose downstream effect on the next workflow's worst-case pressure is bad) **does not close the gap on median time-to-resume**, although it does improve makespan. The reason is a metric mismatch: the lookahead optimizes a max-pressure proxy that corresponds to makespan, while the median is improved by *deliberately unbalancing* the herd (e.g., two fast workflows on KV transfer, two slow on replay). A median-aware planner would need a quantile-aware scoring function, not better max-pressure prediction.

**Implication.** "Better heuristic" is not a single problem. The right next step is to measure exactly what the oracle does that the planner doesn't, on the cells where the gap is real, before tuning anything.

### 3.10 Random is sometimes accidentally the right thing

Across the regime map, a uniformly-random mode/destination policy wins outright in 7 of 240 cells (medium state, slow link). In several others it is within a few percent of the load-aware planner. The reason is structural: in slow-link / large-state cells, "everything via the cheapest cold mode at one destination" is a worse strategy than "spread the load by accident." Strong cache reuse, which sends every workflow to the destination with the most warm hits, can be *worse than chance* in a regime where there is no warm cache to begin with.

**Implication.** Any policy comparison must include random as a sanity baseline. A heuristic that doesn't beat random in a given regime is not a good heuristic in that regime, regardless of what it beats among the named policies.

---

## 4. What the framework can answer today

- **For a given trace:** what state objects exist, how they connect, and what each one's mobility class is.
- **For a given placement plan and destination capacity:** how long the resume takes under finite network, prefill, workspace-hydrate, and KV-memory resources.
- **For a given workload at a chosen scale:** which resource is the bottleneck, and which class of policy is in range of the optimum.
- **For a given model architecture:** above which link bandwidth KV transfer beats prompt replay, and below which the opposite.
- **For a small batch of workflows:** how much headroom remains over the current load-aware planner.

## 5. What the framework cannot answer today

- Real wall-clock latencies. The cost model omits decode and uses additive (not pipelined) `transfer + prefill` time. Relative orderings between policies are sound; absolute seconds are loose.
- Whether a real harness (OpenHands, LangGraph, CrewAI) sees the same regimes as the synthetic anchors. Only one real-trace adapter (SWE-agent retrospective) has been written.
- Per-stream prefill rates at long context (>32K tokens). Attention is `O(N²)`; the linear-rate approximation breaks above that.
- Heterogeneous-hardware destinations (mixed FP16/FP8 GPU pools, mixed HBM capacities). Sites differ only in their per-stream prefill rate today.
- Online behavior. The simulator runs static plans. No admission control, no preemption, no streaming workload.

---

## 6. Where the project stands

The strongest current claim is structural rather than competitive:

> Migrating an in-flight LLM-agent workflow is a state-reconstitution problem. The state decomposes by layer (prompt, workspace, artifact, KV, retrieved documents, transcripts) and by mobility class (globally available, cheaply rehydratable, must move, can be recomputed). The cost of a migration is the destination's time to make all `must_move` state available under finite network, prefill compute, workspace-hydrate, and KV-memory budgets. Which placement strategy wins depends on which of those resources is dominant — and that depends jointly on workload size, per-workflow state bytes, link bandwidth, and the model's architecture (KV bytes per token and attention compute per token).

A planner-superiority claim (a single policy wins by a meaningful margin across realistic regimes) is **not yet earned**. The current load-aware planner wins in some regimes, ties cache reuse in others, and is sometimes outperformed by random spreading. Earning the planner claim requires:

- exact-simulator validation of any cell quoted as evidence,
- realistic herd traces drawn from the workload anchors at production byte sizes (with installed dependencies, build artifacts, real tool outputs — not shallow clones),
- a planner that is competitive with the oracle on per-workflow destination and prompt-mode choice (the workspace-mode dimension is already near-optimal),
- regime stability under the model-architecture axis.

The honest framing is therefore a **regime-discovery** paper, not a planner paper: state which regime a workload falls into and which policy class is appropriate there, with the calibration and oracle work as the substantive contribution.
