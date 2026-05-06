# TASKS — Vagrant Agent: State-Mobility Layer for Agent Workflows

This file is the working backlog for `vagrant-agent`. It is the authoritative plan; if reality diverges, update this file rather than the implementation plan.

## Project status — regime-discovery phase (post-Week-1, 2026-05-05)

> **The Week 1 result is a regime signal, not a project failure.** A post-critic K7 rerun fixed budget/planner drift, concurrent shared-state dedup, workspace hydrate units, and T3 fixture coverage. Corrected K7 now passes: T1 collapses under infinite capacity, T2 exposes prefill stampede, and T3 shows `mixed_min_pressure` beating the best fixed-mode policy by about 49% on a single-source multi-resource evacuation fixture. This earns carrying mobility episodes forward, but it still does **not** justify a universal "our policy wins" claim.
>
> The active next phase is **regime discovery**: map when per-site reuse is enough, when state locality matters, and when landing pressure requires mobility planning. The paper-level question becomes: **given an agentic workload and a mobility event, which regime is it in?**
>
> See `docs/WEEK1_REPORT.md` for the audit trail, `docs/K7_gauntlet_results.md` for the gauntlet numbers, `docs/K8_K9_regime_map_and_oracle.md` for the first regime map and oracle gap table, `docs/R4_regime_discovery_memo.md` for the current framing, `docs/L1_calibration_paper_draft.md` for the calibration-paper scaffold, and `kv-transfer-early-experiment/FINDINGS.md` for the architecture-dependent KV-vs-replay crossover result.

The current regime map hypothesis:

| Regime | What dominates | Current evidence |
| ------ | -------------- | ---------------- |
| Reuse regime | state is small or already warm | H5b: real SWE-agent bytes collapse the grouping gap |
| State-locality regime | large workspace/artifact/state transfer | H2/H5a: synthetic 1 GB workspaces produce large gaps |
| Landing-pressure regime | many workflows reconstitute at once | T2: prefill-stampede sanity check; strong baselines can also avoid replay when network is free |
| Multi-resource regime | network, prefill, workspace all finite | corrected T3: richer planner beats fixed modes on single-source evacuation |

The four-level hierarchy remains useful, but not as a podium:

| Level | Abstraction | Status today |
| ----- | ----------- | ------------ |
| L0 | No-reuse baseline (D1) | Strawman; keep only as a calibration floor |
| L1 | Strong per-site reuse + per-state mode choice | First serious baseline; often sufficient on observed traces |
| L2 | Graph grouping (D2 / D3 / G1) | Lateral abstraction; not proven better than L1 on real fixtures |
| L3 | Mobility episodes / landing pressure | Useful in corrected T3; regime-dependent and still unmapped |

**Repo grounding** (read these before starting any task):

- `vagrant_agent_repo_implementation_plan.md` — original (longer) design doc; reference, not gospel.
- `../coding-progress-ledger/ledger_progress/{core,session,serialization,queries,sidecar}.py` — vagrant rides on these.
- `AGENTS.md` (this repo and sibling) — coding rules, identical in spirit.
- `docs/WEEK1_REPORT.md` — latest audit trail and why the next step is regime discovery.
- `docs/TAKEAWAYS_FOR_REVIEW.md` — pre-K7 context for the original K/L fork.

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
Do not score "winning policy" on a single episode. The output is a
  regime map, not a podium.
Do not mutate a warmness map outside the K4 fluid simulator.
Do not treat "mixed_min_pressure wins" as the project's thesis. K7 did
  not earn that claim. The active thesis is regime discovery: identify
  when strong per-site reuse is sufficient and when richer mobility
  planning has a real ceiling.
Do not tune one fixture until a policy wins. Run sweeps, report the map,
  and keep `random_mode` in the comparison set.
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
- Broad new harness adapters before the K8/K9 regime map and oracle exist.

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

**Workstream K (mobility episodes, audits, gauntlet).** A1-A4 audits, K0 definitions, K1-K6 substrate, and K7 gauntlet are `done`. The important result is not "K wins" but "single-cell K7 is insufficient": T2 shows a landing-pressure regime, T3 shows the richer planner has not earned a broad claim, and the next step is a sweep plus oracle.

**Workstreams I, J (deferred).** Workstream K subsumes the capacity question fluidly without queue simulation; J (live KV migration / packet-level networking) stays deferred indefinitely.

**Sensitivity tooling (`done`).** `vagrant-sensitivity` CLI + `run_sweep` helper grid-searches (kv_bytes, link_bps); `costs.py` carries a caveat block on load-bearing assumptions; `configs/sites_2site.yaml` uses a 5 Gbps single-flow inter-region link; 3 model profiles bracket the realistic 2025-2026 KV-per-token range.

---

## § Workstream K — completed audits, scaffolding, and gauntlet

**Status:** `done` through corrected K7. This workstream built the measurement substrate and now provides a positive single-cell mobility-episode result. It is still the base layer for regime discovery, not a substitute for the regime map.

**A1 — Workspace-payload decomposition** (`done`)
`docs/A1_workspace_payload_audit.md` + `tests/test_a1_workspace_payload.py`. Eight layers measured: `repo_tree_bytes`, `git_diff_bytes`, `touched_file_bytes`, `read_file_bytes`, `tool_output_bytes`, `test_log_bytes`, `build_artifact_bytes`, `dependency_cache_bytes`. No measured shallow-clone layer flips H5b, but the production-relevant layers are undermeasured or zero.

**A2 — Home-site premise audit** (`done`)
`docs/A2_home_site_premise.md`. Existing multi-workflow fixtures are distributed-origin; single-source evacuation, fan-in, and regional-affinity are gaps.

**A3 — D3 edge-typed grouping policy** (`done`)
`shared_state_aware_typed` landed. D3 is not strictly better than D2; on H5b it is worse because component-level dedup interacts with global prompt materialization. Load-bearing conclusion: reconstitution must charge per-(state, site), not per component.

**A4 — Cost-model audit writeup** (`done`)
`docs/A4_cost_model_audit.md`. Documents additive vs pipelined cost, faster-prefill bias under infinite capacity, omitted decode, and raw-bytes KV.

**K0-K7 — Mobility episode substrate and gauntlet** (`done`)
K0 definitions, K1 `MobilityEpisode`, K2 `WarmnessMap`, K3 `ResourceCost`/`ResourceBudget`, K4 `simulate_fluid`, K5 seven reconstitution policies, K6 herd fixtures, and K7 gauntlet are implemented. After the critic fixes, T1/T2/T3 pass. `runs/k7_gauntlet/gauntlet_results.csv` and `docs/K7_gauntlet_results.md` are the source of truth.

**K8/K9 — First regime map and oracle** (`done`, 2026-05-06)
`src/vagrant_agent/k8_regime.py` and `src/vagrant_agent/k9_oracle.py` landed with focused tests. `scripts/run_k8_k9.py` emits `runs/k8_regime_map/` and `runs/k9_oracle/`. The first K8 map covers all requested N × state-scale × prefill × bandwidth axes using an aggregate service-time estimator for large cells; exact K4 remains available for focused cells. K8 exact-vs-aggregate calibration on 36 sampled cells shows best-policy agreement in 24/36 cells and bottleneck-label agreement in 102/216 policy rows, so aggregate heatmaps are regime hypotheses rather than final timing evidence. K9 restricted exact search now covers four 4-workflow diagnostic cells and finds oracle gaps vs strong reuse from 50.3% to 96.7%.

**V1 — Exact validation of K8 claim cells** (`done`, 2026-05-06)
`src/vagrant_agent/k8_validation.py` and `scripts/run_k8_validation.py` landed. `runs/k8_validation/` plus `docs/K8_exact_validation.md` rerun seven named claim cells through exact K4 and compare best-policy agreement, policy/cell bottleneck agreement, p50/p95 relative timing error, and a trust label. Current result: all seven selected cells are `needs_exact_k4`; aggregate K8 remains useful for discovering candidate regimes, but exact K4 is required before quoting timing or bottleneck claims.

---

## § Workstream R — regime map (active next phase)

**Goal.** Turn Vagrant from "does our policy win?" into "which regime is this workload and mobility event in?" The main artifact is a sweep-backed map over workload size, state scale, prefill pressure, and link bandwidth, with policy winner and dominant bottleneck reported per cell.

### R0 — Reframe and nomenclature cleanup

**R0.1 — Retire H1/D2 as the central story** (`in progress`)
Use "strong per-site reuse baseline vs richer mobility planning" in new writing. Keep old names in code and historical docs, but new docs should describe the strong baseline plainly: reuse any state already materialized at a site; for cold state choose the cheapest available materialization mode (replay, KV transfer, artifact copy, workspace hydrate, or warm reuse).

**B1 / R0.2 — Baseline naming audit** (`done`, 2026-05-06)
`docs/strong_site_reuse_baseline.md` explains that `cache_reuse` is the K-level representative of the strong L1 baseline and already includes per-state mode choice plus per-workflow destination preference. `strong_site_reuse` is a paper-facing alias in `reconstitution.py`; `cache_reuse` remains for historical compatibility.

### R1 — K8 regime-map sweep runner

**R1 — Grid runner** (`done`, 2026-05-06)
`src/vagrant_agent/k8_regime.py` plus `tests/test_k8_regime.py` run a deterministic grid and write `runs/k8_regime_map/regime_policy_metrics.csv` plus cell summaries.

Initial axes:

```text
N workflows:              10, 100, 1000, 10000
workspace/artifact scale: tiny, swe_bench, medium, monorepo, large_artifact
prefill capacity:         loose, moderate, tight
link bandwidth:           1, 5, 25, 100 Gbps
```

Implementation notes:
- Existing `HerdSpec` supports the episode shape; K8 maps `swe_bench` and `large_artifact` explicitly in the sweep runner.
- Exact K4 is too slow for the full 1K/10K grid. The emitted full map uses a deterministic aggregate service-time estimator and records that caveat in `runs/k8_regime_map/README.md`; `run_k8_cell(...)` remains the exact K4 path for focused validation cells.
- The CSV includes best policy, p50/p90/makespan, dominant bottleneck, mixed-vs-strong gap, and enough axis metadata to regenerate the cell.

Policies to include in every sweep cell:

```text
strong per-site reuse baseline (`cache_reuse` or alias)
replay_all
kv_all
workspace_sticky
random_mode
mixed_min_pressure
```

### R2 — Heatmaps and bottleneck attribution

**R2 — Plotting** (`done`, 2026-05-06)
`write_k8_artifacts(...)` emits:

```text
x-axis: workspace/artifact scale
y-axis: N workflows
color: best policy or dominant bottleneck
panels: loose/tight prefill × slow/fast link
```

Also emit a policy-gap table sorted by `winner_margin_vs_strong_reuse`; this prevents a colorful heatmap from hiding that the winner is only 1-3% better.

Current first-pass summary: 240 cells, 1440 policy rows, 24 heatmaps. `mixed_min_pressure` is best in 233/240 aggregate-estimated cells; `random_diversification` narrowly wins 7 medium-state cells. Dominant bottlenecks split into network 119, workspace 72, prefill 49. Calibration caveat: sampled exact K4 agrees with aggregate best-policy labels in 24/36 cells and with aggregate bottleneck labels in 102/216 policy rows.

### V1 — Exact validation of selected K8 cells

**V1 — Claim-cell validation** (`done`, 2026-05-06)
`scripts/run_k8_validation.py` emits `runs/k8_validation/claim_cell_policy_validation.csv`, `claim_cell_summary.csv`, and `docs/K8_exact_validation.md`.

Selected cells:

```text
swe_bench_reuse_scale
tiny_prefill_pressure
tiny_slow_link
medium_multi_resource
monorepo_workspace_pressure
large_artifact_slow_link
large_artifact_fast_link
```

Current result: 7/7 selected cells are `needs_exact_k4`. Aggregate/exact best-policy labels agree in several selected cells, but p50 errors and bottleneck-label drift are too large for aggregate-only timing or bottleneck claims. This is the intended V1 outcome: K8 is a discovery map; K4 validates claim cells.

### R3 — Model architecture profile axis

**R3 — Reintegrate KV study** (`not started`, ~1 day)
Translate `kv-transfer-early-experiment/FINDINGS.md` into model-profile presets for the regime sweep:

```text
heavy KV / cheap replay       (GLM-like)
compact KV / expensive replay (DeepSeek-like MLA)
hybrid                        (Qwen3-Next-like)
vanilla GQA
MLA-like
```

The claim to test: model architecture changes not just the single-request KV-vs-replay boundary, but which destination resource a migration herd stresses.

### R4 — Revised memo

**R4 — Regime-discovery memo** (`done`, 2026-05-06)
`docs/R4_regime_discovery_memo.md` records the current four-claim framing:

1. Agentic mobility is state reconstitution.
2. Per-site reuse is the first serious baseline.
3. Richer planning is regime-dependent.
4. Vagrant maps the regime.

Current answer: Path A is supported; Path B is opened but not yet earned. Workload anchors and exact validation remain required before planner-paper claims.

---

## § Workstream O — small-N oracle

**Goal.** Before tuning `mixed_min_pressure`, measure the ceiling. The oracle is diagnostic and does not need to scale.

**O1 — Offline oracle for small episodes** (`done`, 2026-05-06)
`src/vagrant_agent/k9_oracle.py` implements exact simulator-backed search for small instances:

```text
N <= small exact limit initially (current artifact uses N=4; code rejects oversized requests explicitly)
2-3 destination sites
finite network / prefill / workspace resources
same action modes as K5
```

It exhaustively enumerates workflow-level destination, prompt-mode, and workspace-mode choices. No external solver dependency was added.

Output table: `runs/k9_oracle/oracle_gap_table.csv`. Four diagnostic cells now cover tiny prefill pressure, medium multi-resource pressure, monorepo workspace pressure, and slow-link network pressure. Oracle gaps vs strong reuse range from 50.3% to 96.7%; oracle gaps vs mixed range from 0.7% in the monorepo/workspace cell to 50.0% in the medium multi-resource cell. Candidate-space caveat: this is exact only over workflow-level destination, prompt-mode, and workspace-mode choices.

Interpretation:
- If oracle barely beats strong reuse, the ceiling is low in that regime.
- If oracle beats strong reuse by 20-30%, inspect the oracle-vs-policy difference before tuning any heuristic.

**O2 — Oracle-difference explanation** (`done`, 2026-05-06)
`src/vagrant_agent/oracle_diff.py` and `scripts/run_o2.py` emit `runs/o2_oracle_diff/`. Per-cell artifacts cover destination choices, prompt modes, workspace modes, per-policy bottleneck breakdown (with an `attr` sidebar reporting the fraction of makespan with an attributed bottleneck so a sparse breakdown is not misread), and the three gaps (oracle vs mixed, oracle vs random, strong-reuse vs random). The exhaustive enumeration is shared with `k9_oracle.run_small_n_oracle` via `enumerate_oracle_plans` so the two paths cannot drift on candidate space or objective; a parity test pins the byte-identical p50.

Headline first-pass numbers (4 diagnostic cells, n=4):

| cell | oracle vs mixed | oracle vs random | strong vs random |
| ---- | --------------: | ---------------: | ---------------: |
| tiny_prefill_pressure | 35.0% | 50.2% | -13.7% |
| medium_multi_resource | 50.0% | 80.3% | -1.7% |
| monorepo_workspace_pressure | 0.7% | 33.7% | -33.6% |
| slow_link_network_pressure | 49.9% | 93.5% | -97.4% |

Negative `strong vs random` is the K7 finding restated: in slow-link / large-state cells, random_mode's diversification beats strong reuse's "everything via the cheapest cold mode at one site" choice. The `monorepo_workspace_pressure` row (oracle vs mixed = 0.7%) confirms `mixed_min_pressure` is already near-optimal when workspace bytes dominate; the other three rows show real headroom (35–50%) over `mixed`. The diff CSV identifies that the headroom comes primarily from per-workflow destination + prompt-mode choices, not workspace-mode.

---

## § Workstream S — workflow-local mobile state measurement

**Goal.** Stop using repository size as a proxy for migration payload. Measure workflow-local mobile state and classify what must move, what can be rehydrated, and what can be discarded.

**S1 — State-layer taxonomy update** (`not started`, ~0.5 day)
Extend A1's taxonomy into the production-facing split:

```text
base repo checkout
uncommitted diff
files read
files touched
tool outputs
test logs
build artifacts
dependency cache
retrieved documents
subagent transcripts
summaries / compaction outputs
```

Each layer must be classified as `globally_available`, `cheaply_rehydratable`, `must_move`, `can_be_recomputed`, or `can_be_discarded`.

**S2 — Mobile-state audit script** (`not started`, ~1.5 days)
Add a small auditor for a captured workflow directory that reports the S1 layers and emits JSON/CSV. This is the bridge to Week 3 workload anchors.

---

## § Workstream W — workload anchors

**Goal.** Anchor the synthetic regime map with non-SWE workload families chosen because they stress different state layers.

**W1 — Large-repo coding fixture** (`done`, 2026-05-06)
`src/vagrant_agent/workloads.py` introduces `WorkloadAnchor` + `W1_LARGE_REPO_CODING`. Per-workflow workspace ~1.03 GB (base repo 350 MB + dep cache 600 MB + build artifacts 80 MB + uncommitted diff 0.2 MB), with `tool_output_context` wired into per-workflow prompt-context tokens. `classify_regime` exercises strong reuse + mixed under K4 and labels the cell. The W1 hypothesis (`state_locality`) is asserted as a regression test under a 1 Gbps link cell.

**W2 — Data/RAG/artifact-heavy fixture** (`done`, 2026-05-06)
`W2_DATA_RAG_HEAVY` in `workloads.py`. Per-workflow must_move ~1.9 GB (retrieved docs + cleaned intermediates + generated plots), prompt_summaries wired into prompt-context tokens, and the two `globally_available` layers (base_data_bundle 500 MB + vector_index_shards 8 GB) are real `StateObject`s with initial warmness across every (source ∪ destination) site so a buggy policy that ignores warmness is falsifiable. Hypothesis-match regression test asserts `state_locality` under a 1 Gbps cell.

**W3 — Multi-agent fanout/fanin fixture** (`done`, 2026-05-06)
`W3_MULTI_AGENT_FANOUT` in `workloads.py`. Per-workflow shape: planner + 4 subagents + reviewer (the fanin step). Subagents share `shared_task_<wid>` (a structural test asserts cache_reuse emits one action per workflow for it, not K). Reviewer reads every private transcript + a `merge_<wid>` buffer state. `dependency_cache` is a real `globally_available` workspace state warmed at every site. Regime hypothesis is `landing_pressure` — N workflows × (K+2) llm_calls saturate prefill at the destination — and the canonical-cell hypothesis-match test pins it.

Week 3 success criterion: each anchor has a state-layer breakdown and a regime classification.

---

## Current two-week target

**Week 2 deliverables**

1. R1 K8 regime-map sweep over N × state scale × prefill cap × link bandwidth. `done`
2. R2 heatmaps of best policy and dominant bottleneck. `done`
3. O1 small-N oracle. `done`
4. Oracle-vs-baseline gap table. `done`
5. R4 revised memo: where richer mobility planning matters. `done`

**Week 3 deliverables**

1. One large-repo coding fixture.
2. One data/RAG/artifact-heavy fixture.
3. One multi-agent fanout/fanin fixture.
4. State-layer breakdown for each.
5. Placement/mobility regime classification for each.

**Definition of done for this phase**

> We can say whether T3 was one unlucky cell or representative, and whether the current heuristic is weak or the strong reuse ceiling is low. The output is a regime map and oracle gap table, not a claim that one policy universally wins.

---

## Open questions

- **Sweep scope vs runtime.** Answer: exact K4 is too slow for the full 1K/10K grid; K8 now uses aggregate estimation for the full map and exact K4 for focused cells. V1 adds named exact claim-cell validation; aggregate-only regions remain exploratory until validated.
- **Oracle formulation.** Current O1 exact enumeration over workflow-level mode/destination assignments is enough for the first ceiling check. Open follow-up: add pruning or solver-backed formulation only if larger N is required.
- **Strong baseline label.** Keep `cache_reuse` as the code name, or add `strong_site_reuse` as an alias for paper clarity?
- **Episode trigger explicit or implicit.** Default: explicit — episode JSON specifies `target_sites`; placement-policy is a consumer of episode setup, not the subject of K. (D2/H1/G1 stay relevant for placement; K is "what happens given the move.")
- **"Useful resume" definition.** Default: first decoded token at destination. Per-workflow metric; episode-summary `time-to-50%-resume` configurable.
- **Site = abstract resource pool or geographic region.** Default: abstract pool — Phoenix/Seattle/Austin become labels for capacity envelopes.
- **Workspace payload for regime sweeps.** Default: use explicit distribution labels (`swe_bench`, `medium`, `monorepo`, `large_artifact`) and keep A1/S1 layer measurements as anchors.
- **Model-profile axis size.** R3 should start with three architecture profiles, then expand only if the heatmap changes materially.
- **`tau` choice.** MVP defaults to 1 token. Real workloads may need a higher threshold; A3's edge-typed policy partially supersedes the `tau` knob.
- **Token counting at trace time.** Real harnesses may not give exact counts; estimate from text. F2 currently approximates.
- **State-object identity across reopen/invalidate.** MVP treats invalidation as a new object; revisit if real traces make this ambiguous.
