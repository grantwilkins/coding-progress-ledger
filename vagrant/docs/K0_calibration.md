# K0 — Workstream K calibration writeup

**Status:** done, 2026-05-05
**Scope:** define the central concepts (`mobility episode`, `warmness map`, `resource vector`, `reconstitution plan`) precisely; lock in the 4-level abstraction hierarchy; map mobility-episode use-cases to scenario classes; specify the three falsification tests (T1/T2/T3) with pass/fail criteria; pre-accept the A1–A4 audit findings.

## What "mobility episode" means in this project

A **mobility episode** is a batch event with the following structure:

```
input:
  source_sites:       set of sites where workflows currently live
  destination_sites:  set of sites where workflows must reconstitute (≥1)
  workflows:          set of stateful agentic workflows that must move
  state_warmness:     for each state object, which sites currently hold a copy
  capacities:         per-site fluid budgets for network / prefill /
                      workspace hydration / KV memory

output:
  reconstitution_plan: per-(workflow, state_object), how to make that
                      state available at the chosen destination site
                      (mode + dst_site)
  schedule:            wall-clock ordering of materialization actions,
                      determined by the fluid simulator (K4)

metric:
  time_to_useful_resume: per-workflow, time from episode trigger until
                         destination can produce its first decoded token
  episode_summary:       time-to-X%-resume across the herd (X=50 default)
```

This is **not** a per-request scheduler. It is not a routing decision. It does not migrate live KV across the wire (that's the deferred Workstream J). It is a model that lets us ask: *given that N workflows must move, what's the best plan, and which resource saturates?*

## The 4-level hierarchy

Adopting collaborator 2's framing (memo §10):

| Level | Policy class | Concrete representatives | What it captures |
| ----- | ------------ | ------------------------ | ---------------- |
| L0 | No-reuse baseline | D1 (`request_level_no_reuse`) | strawman; every consumer pays its own materialization |
| L1 | Per-site cache reuse | H1 (`request_level_with_site_cache`) | per-(state, site) dedup; per-node placement |
| L2 | Graph grouping | D2 (`shared_state_aware`), D3 (`shared_state_aware_typed`), G1 (`g1_brute_force`) | per-(component, site) dedup; component-level placement |
| L3 | Mobility episodes | K5 reconstitution policies (`replay_all`, `kv_all`, `cache_reuse`, `workspace_sticky`, `mixed_min_pressure`) | resource-vector cost; fluid capacity; herd dynamics |

Per A3 audit: **L2's component-level materialization accounting is structurally weaker than L1 in some regimes** (D3 > D2 on H5b real bytes by 108 ms). L1 ≤ L2 is not an invariant. The right characterization: L1 dominates L2 when colocation-at-one-site is correct; L2 dominates L1 when fragmentation creates real cost savings (which requires home asymmetry that aligns with byte mass, per A2).

L3 introduces a different axis: **resource-vector cost under fluid capacity**, not just scalar wall-clock cost. K7 will determine whether L3 is meaningfully different from L1.

## Mobility-episode usefulness map

From collaborator 2 §2, with scenario-class labels from A2:

### Useful (6)

| Use case | Scenario class | Why mobility episodes matter |
| -------- | -------------- | ---------------------------- |
| Capacity evacuation | single-source-evacuation | Many workflows leave one source within a window; reconstitution at destinations consumes shared resources. |
| Regional failover | single-source-evacuation | A region degrades; sessions must resume elsewhere; herd by definition. |
| Maintenance drain | single-source-evacuation | Cluster being drained; deadlines/slack differ by workload. |
| Spot/preemptible capacity shift | single-source-evacuation | Cheap capacity disappears or appears; batch movement. |
| Cross-site fanout/fanin | distributed-origin → fan-in | Subagents launched at different sites; merge point shares capacity. |
| Large artifact/data workflows | distributed-origin or single-source | Workspaces are big enough that reconstitution mode actually matters. |

### Not useful (6)

| Use case | Why not |
| -------- | ------- |
| Stateless or short chat | Requests independent; no reconstitution needed. |
| Linear small coding sessions | What H5b is; per-site cache reuse handles it. |
| Workspaces in globally-accessible storage | "Home" is weak; every destination can hydrate cheaply. |
| Sticky session security/correctness | Optimizer has no room. |
| Very fast intra-region fabrics | Network ≈ free; no bottleneck unless herd is enormous. |
| Heavy summarization/checkpointing | Mobility becomes a semantic-checkpoint problem. |

**Four out of six "useful" scenarios are single-source-evacuation, which vagrant has not modelled.** The K7 gauntlet's fixtures must include single-source-evacuation variants — see K6.

## Resource-vector consumption table

Per K3's `ResourceCost(network_bytes, prefill_tokens, workspace_bytes, kv_resident_bytes, wallclock_s)`. The five materialization modes (already in `events.py:MATERIALIZATION_MODES`) consume resources as follows:

| Mode | Network | Prefill | Workspace hydration | KV memory (resident) | Notes |
| ---- | ------: | ------: | ------------------: | -------------------: | ----- |
| `kv_transfer` | high (= state KV bytes) | none | none | high (= state KV bytes) | Linear in T × kv_bytes_per_token. Subject to compression (see A4). |
| `context_replay` | low (text tokens) | high (= state tokens) | none | high (= state KV bytes after replay) | Pays prefill rate per token at destination. |
| `artifact_copy` | medium/high (= state bytes) | none | high (= state bytes) | none until consumed | Workspace state typically uses this mode. |
| `warm_reuse` | zero | zero | zero | zero (already resident at dst) | Free; valid only when `warmness.is_warm(state, dst_site)`. |
| `text_transfer` | medium (= state bytes) | none | none | none | Materialized but not yet replayed; cheaper than artifact_copy if not workspace-bound. |

The L3 policies (K5) choose mode per-(workflow, state_object). The K4 fluid simulator sums per-resource consumption across in-flight actions and divides each resource's bandwidth proportionally.

**Resource conservation invariant** (K3 + K4 will pin): for any episode and any plan, `sum(action.network_bytes) == manifest's total cross-site transferred bytes` — no double-counting, no leaks. Same for prefill_tokens, workspace_bytes, kv_resident_bytes.

## The three falsification tests (T1/T2/T3)

These are the pass/fail criteria for the K7 gauntlet. The tentative thesis (mobility episodes is the right project abstraction) is accepted iff **all three pass**.

### T1 — Capacity-free collapse

**Fixture:** `examples/episodes/gauntlet_t1_infinite_capacity.json`. N=100 workflows, all four resource budgets set to `math.inf`, distributed-origin scenario with H5a-like home distribution.

**Run:** all 6 K5 reconstitution policies on K4's fluid simulator.

**Pass criterion:**
```
mixed_min_pressure.makespan == cache_reuse.makespan == H1.placement_total_cost  (within 1e-6)
```

**Why:** under infinite capacity, the "L3 vs L1" distinction must vanish. There's no resource to saturate, no reason to balance bottlenecks, no advantage to mixing modes. If the simulator says L3 is better, **the simulator is smuggling in an effect** — most likely a bug in the cost computation (e.g., charging per-component instead of per-(state, site)).

**Failure mode if violated:** if T1 fails (mixed > L1 OR mixed < L1 by > tolerance), the K4 simulator has a bug. **Resolve before evaluating T2/T3.**

### T2 — Prefill-stampede

**Fixture:** `gauntlet_t2_prefill_only.json`. N=100 workflows, prefill capacity set to `30000 tok/s` per site, network and workspace set to `inf`. Single-source-evacuation scenario.

**Run:** all 6 policies.

**Pass criteria:**
```
replay_all.p50_resume > kv_all.p50_resume                       (replay stampedes prefill)
mixed_min_pressure.p50_resume < replay_all.p50_resume - 10%     (mixed avoids stampede)
```

**Why:** under prefill capacity only, replaying every workflow's context at one destination saturates prefill. The right behavior is to mix: some workflows replay, others KV-transfer to bypass prefill. If `mixed_min_pressure` cannot do better than `replay_all` by ≥10%, the herd idea is weak.

**Failure mode if violated:** if T2 fails, prefill capacity does not differentiate policies in the modeled regime. The mobility-episode framing offers no advantage over L1 for prefill-bound workloads.

### T3 — Multi-resource bottleneck

**Fixture:** `gauntlet_t3_multi_resource.json`. N=100 workflows. Network = 5e9 bps (5 Gbps), prefill = 30000 tok/s per site, workspace_hydrate = 1e9 bps, KV memory = 100 GB per site. Mix of single-source-evacuation and distributed-origin variants.

**Run:** all 6 policies.

**Pass criterion:**
```
mixed_min_pressure.p50_resume <
  min(replay_all, kv_all, cache_reuse, workspace_sticky).p50_resume - 10%
```

**Why:** under multi-resource pressure, no single fixed-mode policy can balance the bottleneck. `replay_all` saturates prefill; `kv_all` saturates network; `cache_reuse` does well only where warm; `workspace_sticky` forbids mode switching. `mixed_min_pressure` should win by ≥10% by intentionally diversifying modes across workflows.

**Failure mode if violated:** if T3 fails, fixed-mode policies are competitive even under saturation. **Mixed planning isn't earning the abstraction.** The honest conclusion is L1 + per-mode operator-controlled choice is sufficient; L3 doesn't add value.

## Pre-accepted audit findings

Workstream K's plans are conditioned on the following findings from A1–A4. Each is cross-referenced; do not silently revisit without updating this section.

- **A1** (`docs/A1_workspace_payload_audit.md`): every measurable workspace-payload interpretation collapses the H1<D2 regime in the H5b shallow-clone setup. Production workloads with `dependency_cache` (~100–500 MB) and `build_artifact` (~10–100 MB) are the regime we should target with K's herd fixtures, not the SWE-bench-pilot 33 MB regime.
- **A2** (`docs/A2_home_site_premise.md`): every existing multi-workflow fixture is *distributed-origin*. Single-source-evacuation, the dominant production motivation, has not been studied. K1 schema includes `source_sites` distinguisher; K6 fixtures must include single-source variants.
- **A3** (`docs/A3_edge_typed_policy_audit.md`): D3 fixes overgrouping but does not close the L1 gap. The L1-vs-L2 distinction is structural (per-(state, site) vs per-(component, site) materialization), not about edge-typing. **K3's `reconstitution_cost` MUST charge per-(state, site)**, not per-(component, site).
- **A4** (`docs/A4_cost_model_audit.md`): the cost model is additive (overstates grouped-plan costs), prefill-bias-prone under infinite capacity (which K4 resolves), decode-omitting (cancels in policy comparisons), and KV-uncompressed (3–4× pessimistic, within sensitivity range).

## What K is not (load-bearing scope guards)

- **Not** a per-request scheduler.
- **Not** a routing decision.
- **Not** an admission-controlled queue.
- **Not** a packet-level network model.
- **Not** a live-migration KV protocol.
- **Not** a winning-policy claim. K's deliverable is a regime map (Phase 3a) or a calibration paper (Phase 3b).

## What's next

K1 is next: implement `MobilityEpisode` + `Workflow` dataclasses with JSON load/dump, plus an adapter that wraps one F2 trace as a 1-workflow episode. K2/K3/K4/K5/K6/K7 follow per the plan. K7's gauntlet decision — pivot or calibration paper — is the project's load-bearing fork.
