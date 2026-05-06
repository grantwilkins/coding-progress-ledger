# TASKS — Vagrant Agent: State-Mobility Layer for Agent Workflows

This file is the working backlog for `vagrant-agent`. It is the authoritative plan; if reality diverges, update this file rather than the implementation plan.

## Project status — calibration result (post-K7 gauntlet, 2026-05-05)

> **The K7 falsification gauntlet has failed.** T1 (correctness check) and T2 (prefill-stampede) pass; **T3 (multi-resource bottleneck) fails** — even with a load-aware bin-packing `mixed_min_pressure`, the herd-level planner beats best fixed-mode (`cache_reuse`) by only ~3.6%, below the 10% bar. `random_mode` matches `mixed_min_pressure` to within 0.4%, indicating the diversification heuristic is no better than chance on the canonical fixture. **The K-pivot is not earned**; the project's path is **Phase 3b — Workstream L (calibration paper)**.
>
> The honest framing of the project's contribution: for SWE-bench-class workloads at observed scales, L1 (per-site cache reuse + per-state intelligent mode dispatch) explains most of the state-locality benefit. Herd-level planning shows measurable but sub-threshold improvement on multi-resource fixtures, and is dominated by a smart per-state policy. The phenomenon claim requires either workloads above the regime-flip threshold (~50 MB minority-home cross-site bytes) OR explicit prefill-only saturation (T2 regime), neither of which is present in the SWE-bench shallow-clone setup.
>
> See `docs/K7_gauntlet_results.md` for the full breakdown and `docs/L1_calibration_paper_draft.md` for the calibration paper outline.

The four-level conceptual hierarchy frames everything below:

| Level | Policy | Status today |
| ----- | ------ | ------------ |
| L0 | No-reuse baseline (D1) | Strawman; gap closes with L1 |
| L1 | Per-site cache reuse (H1) | Important; often sufficient on observed traces |
| L2 | Graph grouping (D2 / D3 / G1) | Conditional; collapses to L1 on every real fixture we hold |
| L3 | Mobility episodes (Workstream K) | Hypothesis under test; pivot iff K7 gauntlet passes |

**Repo grounding** (read these before starting any task):

- `vagrant_agent_repo_implementation_plan.md` — original (longer) design doc; reference, not gospel.
- `../coding-progress-ledger/ledger_progress/{core,session,serialization,queries,sidecar}.py` — vagrant rides on these.
- `AGENTS.md` (this repo and sibling) — coding rules, identical in spirit.
- `docs/TAKEAWAYS_FOR_REVIEW.md` — context for the K/L pivot.

Status markers on each task: `not started` · `in progress` · `blocked` · `done` · `deferred`.

---

## § 0. Project rules for all agents

```text
Do not fork ledger_progress. Import it.
Do not invent a new event class. Ride on LedgerEvent.
Do not add an ILP solver in the MVP.
Do not simulate per-request queues, admission control, schedulers, or routing.
  Capacity-bearing resources may be modelled fluidly (proportional share at
  any instant; advance time to the next event). KV memory may be modelled
  as a capacity with LRU eviction. No buffers, no FIFO, no priorities, no preemption.
Do not invent a new event class for mobility episodes. Episodes are JSON
  files referencing manifests; they ride alongside the ledger.
Do not score "winning policy" on a single episode. Workstream K's output
  is a regime map, not a podium.
Do not mutate a warmness map outside the K4 fluid simulator.
Do not accept "mobility episodes" as the project's new thesis until
  Workstream K's falsification gauntlet (K7) passes all three tests.
  Until then, the pivot is provisional and Workstream L is a live
  alternative outcome.
Do not score "semantic correctness" or model output quality.
Do not abbreviate request_level_no_reuse to request_level. The "no_reuse"
  qualifier is load-bearing — it documents that the baseline is strawman.
Do not write tests asserting a kv-vs-replay crossover in token count T.
  T cancels. Crossovers live in bandwidth, kv_bytes_per_token, and prefill rate.
Do not hand-edit a manifest. The manifest is derived from the trace.
Do not lean on ledger_progress scoring/split/reopen semantics for vagrant
  signals. Use ledger subtasks as graph nodes only.
```

If a task seems to require violating one of these rules, stop and escalate — don't quietly relax them.

## § 0.1 Non-goals

- Real cloud deployment, real KV-tensor migration, packet-level network simulation.
- Power/thermal modelling.
- Online scheduler with admission control.
- Semantic-quality scoring of outputs.
- New harness adapters beyond F2 until after the K7 gauntlet.

## § 0.2 Reuse contract with `coding-progress-ledger`

Vagrant imports `ledger_progress` as a library. Permitted upstream additions, in order of preference:

1. **Zero changes** — use `LedgerEvent.payload` (already `dict[str, Any]`) to carry vagrant-specific fields (`state_id`, `tokens`, `content_hash`, `site`, `mode`).
2. **Pass-through hook in `apply_event`** — landed (Workstream A2) so unknown `event_type` strings append to `ledger.events` without raising.
3. **`SubtaskCategory.STATE`** — only if state objects must show up as subtasks (probably not).

Anything bigger is a fork; do not do it. If a fourth need appears, escalate before coding.

## § 0.3 Vagrant-to-ledger mapping

| Vagrant concept    | Ledger representation                                                                  |
| ------------------ | -------------------------------------------------------------------------------------- |
| workflow           | run directory + trace file. **Not** a subtask.                                         |
| LLM call node      | `Subtask` with payload `node_type=llm_call`                                            |
| tool call node     | `Subtask` with payload `node_type=tool_call`                                           |
| subagent node      | `Subtask` with payload `node_type=subagent`                                            |
| state object       | **Not** a subtask. Carried in payload of vagrant `state_*` events.                     |
| mobility episode   | JSON file referencing per-workflow manifests + warmness + capacities. Not a ledger event. |
| subagent spawn     | `ADD_SUBTASK` (parent_id = planner). Not `SPLIT_SUBTASK` unless parent work is invalidated. |
| node start/end     | `UPDATE_STATUS` with `IN_PROGRESS` / `COMPLETE`.                                       |
| state invalidation | vagrant `state_invalidate` event. Not `INVALIDATE_SUBTASK` unless the consuming node's work also invalidates. |

`SPLIT_SUBTASK` / `REOPEN_SUBTASK` / `INVALIDATE_SUBTASK` carry **scoring semantics** in `ledger_progress`. Vagrant must not lean on them. Use ledger subtasks as **graph nodes**; do not use ledger progress scores as a vagrant signal.

## § 0.4 Vocabulary

```text
trace                = append-only JSONL of agent events (immutable input)
manifest             = derived state graph + serving group view (per-workflow replay artifact)
placement plan       = per-node site assignment (policy output)
materialization plan = per-state-object site/mode assignment (policy output)
results              = cost numbers + plots (analysis artifact)
mobility episode     = batch event: N workflows + source/destination sites + warmness + capacities
warmness map         = which sites currently hold a materialized copy of each state object
resource vector      = (network_bytes, prefill_tokens, workspace_bytes, kv_resident_bytes, wallclock_s)
reconstitution plan  = per-(workflow, state_object) reconstitution-mode assignment under capacities
```

---

## § What we built (workstreams A–H, condensed history)

**Workstreams A–E (MVP pipeline, all `done` 2026-05-05).** Trace vocabulary on `LedgerEvent` payloads with the A2 pass-through hook upstream; synthetic adapter + canonical toy fixture at `examples/traces/toy_subagent_trace.jsonl`; manifest = (nodes, state_objects, edges) by replay with bipartite source-of-truth + pairwise edge view; four closed-form cost formulas in `costs.py` (T cancels — keep the bandwidth-crossover guardrail); two MVP policies (D1 `request_level_no_reuse`, D2 `shared_state_aware`) emit plans with `reason` fields; `vagrant-bench` produces `results.csv` + `state_materialization_breakdown.csv` + `plots/duplication_factor.png`; cost-weighted duplication factor is the headline metric.

**Workstream F (real harness adapters).** F2 (SWE-agent retrospective) `done`; closes the trace-source gap, not the byte gap. F1 (OpenHands) and F3 (LangGraph/CrewAI) deferred indefinitely.

**Workstream G (optimization).** G1 brute-force oracle ≤ G1_MAX_NODES nodes; G2 greedy local search seeded from D1. **Finding:** G1 ≡ H1 on every real-trace fixture we hold (toy, g_demo, F2 SWE-agent s_07). The oracle has not found a better grouping on real inputs.

**Workstream H (extra policies and richer fixtures).**
- **H1 (`request_level_with_site_cache`)** is the competitive baseline at L1: per-node placement with materialized state reused across colocated nodes at a site. H1 numerically collapses to D2 and G1 on every linear-session real trace.
- **H2** concatenates 3 SWE-agent sessions (reusing s_07 ×3) with synthetic 1 GB workspace bytes per session and asymmetric homes; H1 < D2 by 1.6 s, sensitivity-robust.
- **H3 (`session_sticky`)** ships as an explanatory/educational policy; provably ≥ H1.
- **H4** adds `compute_repo_bytes` + `SessionSpec.workspace_path`; preserves H1<D2 with real disk bytes on the synthetic-trajectory s_07×3 fixture.
- **H5a** replaces s_07×3 with 5 distinct cached pilot trajectories; synthetic 1 GB workspaces; H1 < D2 by 3.2 s (= 2 × 1 GB cross-site at 5 Gbps).
- **H5b — the load-bearing finding.** Same 5 distinct trajectories with **real working-tree byte sums** from upstream-repo HEADs (~33 MB total). At canonical config, **D2 ≡ H1 to numerical noise (gap < 1e-9)**; sensitivity grid 0% gap survival. The H1<D2 mechanism is real (a synthetic-1-GB recovery test inside `tests/test_h5b_real_bytes.py` reproduces the H5a 3.2 s gap exactly) but byte-magnitude-sensitive — sub-threshold for SWE-bench-class instances at HEAD against a 5 Gbps single-flow link. **This negative finding triggers Workstream K.** `shared_state_aware` reframed from "deprecate-pending" to "not strictly dominated at real-repo scale."

**Workstreams I, J (deferred).** Workstream K subsumes the capacity question fluidly without queue simulation; J (live KV migration / packet-level networking) stays deferred indefinitely.

**Sensitivity tooling (`done`).** `vagrant-sensitivity` CLI + `run_sweep` helper grid-searches (kv_bytes, link_bps); `costs.py` carries a caveat block on load-bearing assumptions; `configs/sites_2site.yaml` uses a 5 Gbps single-flow inter-region link; 3 model profiles bracket the realistic 2025-2026 KV-per-token range.

---

## § Workstream K — pre-pivot audits + falsification gauntlet

**Goal.** Determine whether the mobility-episode framing earns the project's pivot, with a binary gate at K7. Phase 1 audits resolve implementation-choice critiques against the *current* implementation; Phase 2 builds minimal mobility-episode scaffolding + the three falsification tests.

### Phase 1 — Pre-pivot audits

**A1 — Workspace-payload decomposition** (`not started`, ~1 day)
`docs/A1_workspace_payload_audit.md` + `tests/test_a1_workspace_payload.py`. Decompose the workspace payload into 8 layers (`repo_tree_bytes`, `git_diff_bytes`, `touched_file_bytes`, `read_file_bytes`, `tool_output_bytes`, `test_log_bytes`, `build_artifact_bytes`, `dependency_cache_bytes`). Re-run H5b under each interpretation independently and as `(repo_tree, dependency_cache)`. Produce a sensitivity table: workspace-payload-interpretation × H1-vs-D2 gap. **Gate:** documented sensitivity table; informs gauntlet design. A row that flips the regime would mean H5b's "0% gap survival" was payload-definition-dependent.

**A2 — Home-site premise audit** (`not started`, ~0.5 day)
`docs/A2_home_site_premise.md`. Label H2/H5a/H5b explicitly under one of: distributed-origin (sessions already at distinct sites), single-source-evacuation (all workflows at one source), fan-in (subagents from different sites must merge), regional-affinity (storage/data-residency-bound). No new code; reframes existing results. **Gate:** all existing fixtures labeled; missing scenario class identified.

**A3 — D3 edge-typed grouping policy** (`not started`, ~1.5 days)
`src/vagrant_agent/policies.py` extension (`shared_state_aware_typed`) + `tests/test_a3_edge_typed_policy.py`. Replace D2's connected-component grouping with edge-type-weighted grouping over 6 edge types (`global_replicated`, `workflow_shared`, `workspace_local`, `artifact_delta`, `private_context`, `kv_prefix`). **Gate:** D3 implemented; assertion `D3.total_cost_s ≤ D2.total_cost_s + 1e-9` on every fixture; report whether D3 < H1 anywhere. ~150 LOC.

**A4 — Cost-model audit writeup** (`not started`, ~0.5 day)
`docs/A4_cost_model_audit.md`. Document (D) cost model rewards faster prefill too strongly under infinite capacity (resolved by K4); (E) additive vs `max(transfer, prefill)` pipelined (post-K). No code change. **Gate:** writeup exists; assumptions enumerated.

### Phase 2 — Mobility-episode scaffolding + gauntlet

**K0 — Calibration writeup** (`not started`, ~0.5 day)
`docs/K0_calibration.md`. Defines `episode`, `warmness_map`, `resource_vector` precisely. Includes (1) the 4-level hierarchy; (2) the mobility-episode usefulness map (6 useful + 6 not-useful scenarios); (3) the resource-vector consumption table (5 modes × 4 resources); (4) the three falsification tests stated precisely with pass/fail criteria. **Gate:** human review.

**K1 — Mobility episode schema** (`not started`, ~0.5 day)
`src/vagrant_agent/episode.py`. `MobilityEpisode(episode_id, source_sites, destination_sites, workflows, state_warmness, capacities, trigger_t_s, notes)`. `source_sites` distinguishes single-source-evacuation from distributed-origin. **Gate:** roundtrip test + adapter test (one F2 trace → 1-workflow episode). ~80 LOC.

**K2 — Warmness map** (`not started`, ~0.5 day)
`src/vagrant_agent/warmness.py`. `WarmnessMap.is_warm(state, site)`, `fraction_warm(manifest, site)`; age tracking for K4 LRU. **Gate:** unit tests at fraction 0.0/0.5/1.0. ~120 LOC.

**K3 — Resource vector** (`not started`, ~0.5 day)
`src/vagrant_agent/resources.py`. `ResourceCost(network_bytes, prefill_tokens, workspace_bytes, kv_resident_bytes, wallclock_s)`. `reconstitution_cost(state, mode, src, dst, bundle, warmness)` composes `costs.materialize_cost` for `wallclock_s`. **Gate:** crossover-parity with `costs.bandwidth_crossover_bps`; resource-conservation invariant. ~150 LOC.

**K4 — Minimal fluid simulator** (`not started`, ~1.5 days)
`src/vagrant_agent/fluid_sim.py`. Proportional-share fluid resources for network bps per link, prefill tok/s per site, workspace_hydrate bps per site. KV memory as a capacity with LRU eviction. `simulate_fluid(episode, plan, bundle, warmness, budget) → list[ActionTrace]`. Advance time to next event. **No queues, no admission, no scheduler.** **Gate:** (1) two-action-on-one-link → 2× slowdown; (2) resource-conservation invariant; (3) all-warm episode finishes in `min(prefill_tok_s)`; (4) deterministic given a seed. ~300 LOC.

**K5 — Reconstitution policies** (`not started`, ~1 day)
`src/vagrant_agent/reconstitution.py`. Six policies: `min_cost_independent`, `replay_all`, `kv_all`, `cache_reuse`, `workspace_sticky`, `mixed_min_pressure` (greedy fluid-aware oracle). Same registry shape as `policies.POLICIES`. **Gate:** each policy emits a valid plan; `mixed_min_pressure.makespan ≤ min(other 5)` within numerical noise on K6 fixtures. ~200 LOC.

**K6 — Gauntlet fixtures** (`not started`, ~1 day)
`src/vagrant_agent/adapters/herd.py` + 3 committed fixtures: `gauntlet_t1_infinite_capacity.json`, `gauntlet_t2_prefill_only.json`, `gauntlet_t3_multi_resource.json`. Each: N=100 workflows with state distributions sampled from F2/H5 traces. **Gate:** byte-deterministic regenerability; resource-budget envelope correctly carried. ~150 LOC.

**K7 — Falsification gauntlet** (`not started`, ~0.5 day)
`tests/test_k7_gauntlet.py`. Three pytest assertions:

- **T1 — Capacity-free collapse.** Run K4 on `gauntlet_t1_infinite_capacity.json` under all six K5 policies. Assert `mixed_min_pressure.makespan == cache_reuse.makespan == H1.placement_total_cost` within 1e-6. **If T1 fails**, the K4 simulator is smuggling in an effect — model says capacity matters when it shouldn't.
- **T2 — Prefill-stampede.** Run on `gauntlet_t2_prefill_only.json`. Assert `replay_all.p50_resume > kv_all.p50_resume` AND `mixed_min_pressure.p50_resume < replay_all.p50_resume - 10%`. **If T2 fails**, prefill capacity does not differentiate policies — herd idea is weak.
- **T3 — Multi-resource bottleneck.** Run on `gauntlet_t3_multi_resource.json`. Assert `mixed_min_pressure.p50_resume < min(replay_all, kv_all, cache_reuse, workspace_sticky).p50_resume - 10%`. **If T3 fails**, fixed-mode policies are competitive even under saturation — mixed planning isn't earning the abstraction.

**Gate:** all three tests pass. **K-decision-point follows.**

### K-decision-point — manual review

After K7, write `docs/K7_gauntlet_results.md` summarizing the three test outcomes with concrete numbers and CDF plots. **Decide explicitly:**

- **All three pass** → proceed to Phase 3a (K-pivot, K8+).
- **Any fail** → proceed to Phase 3b (Workstream L).

The plan must not silently slide past the gate. The writeup is the artifact supporting the decision either way.

### Phase 3a — K-pivot (conditional on gauntlet passing)

**K8.** 27-cell herd benchmark (N ∈ {1, 10, 100, 1000} × workspace_dist ∈ {SWE-small, monorepo, data-agent, RAG-doc, browser-artifact, ML-checkpoint} × warm_fraction ∈ {0.0, 0.5, 1.0}). State sizes sampled from F2/H5 trace distributions. (`not started`, deferred until gate decision.)
**K9.** Regime-map plot, time-to-useful-resume CDF, bottleneck-attribution stacked bars. (`not started`.)
**K10.** SWE-agent anchor — lift H5a/H5b into K vocabulary; confirm Regime A landing. (`not started`.)
**K11.** Phase-transition assertion: small N, `min_cost_independent` is fine; medium N, stampedes; large N, `mixed_min_pressure` wins. (`not started`.)
**K12** *(deferred)* — offline ILP oracle for small-N exact upper bound.

### Phase 3b — Workstream L: calibration paper (conditional on gauntlet failing)

**L1 — Calibration paper draft** (`not started`)
`docs/L1_calibration_paper_draft.md`. Structure: (1) thesis — for observed coding-agent traces, simple per-site cache reuse (L1) and ordinary independent placement explain most of the state-locality benefit; (2) the 4-level hierarchy as the contribution; (3) H5a → H5b drop as headline negative result; (4) Workstream K artifacts as evidence for what *would* be required to differentiate beyond L1; (5) the K0 usefulness map as positioning.

**L2 — Cleanup.** Deprecate D2 explicitly; mark `vagrant-bench` headline as L1-vs-L0; keep `vagrant-fluid-bench` as exploratory tool.

**L3 — Optional.** Note for OpenHands or LangGraph teams about A1 workspace-payload decomposition — may be more useful as a calibration tool for *other* harnesses than as a vagrant headline.

---

## Definition of done — gauntlet passed (Phase 2 → 3a)

> All three K7 tests (T1 capacity-free collapse, T2 prefill-stampede, T3 multi-resource bottleneck) pass on the committed fixtures, and the writeup at `docs/K7_gauntlet_results.md` documents the outcomes with concrete numbers. The pivot to mobility episodes is now earned; Phase 3a (K8–K11) begins.

## Definition of done — calibration paper (Phase 2 → 3b)

> Any K7 test fails. `docs/K7_gauntlet_results.md` documents which test was inconclusive and why. `docs/L1_calibration_paper_draft.md` lands as a paper-shaped writeup of the negative result; D2 is explicitly marked experimental; Workstream K stays in the repo as the gauntlet-and-tooling contribution but the project headline becomes "for observed coding-agent traces at observed scales, L1 explains most of the state-locality benefit."

Both definitions of done are real outcomes. The project does not silently coast past the gate.

---

## Open questions

- **Episode trigger explicit or implicit.** Default: explicit — episode JSON specifies `target_sites`; placement-policy is a consumer of episode setup, not the subject of K. (D2/H1/G1 stay relevant for placement; K is "what happens given the move.") Confirm or push back; consequential for K1/K4/K5.
- **"Useful resume" definition.** Default: first decoded token at destination. Per-workflow metric; episode-summary `time-to-50%-resume` configurable.
- **Site = abstract resource pool or geographic region.** Default: abstract pool — Phoenix/Seattle/Austin become labels for capacity envelopes.
- **Workspace payload for K.** Default: the union from A1's audit, not the current `compute_repo_bytes` default.
- **`tau` choice.** MVP defaults to 1 token. Real workloads may need a higher threshold; A3's edge-typed policy partially supersedes the `tau` knob.
- **Token counting at trace time.** Real harnesses may not give exact counts; estimate from text. F2 currently approximates.
- **State-object identity across reopen/invalidate.** MVP treats invalidation as a new object; revisit if real traces make this ambiguous.
