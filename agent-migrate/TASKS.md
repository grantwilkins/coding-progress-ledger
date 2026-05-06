# TASKS — Vagrant Agent: State-Mobility Layer for Agent Workflows

This file is the working backlog for `agent-migrate-agent`. It is the authoritative plan; if reality diverges, update this file rather than the implementation plan.

## Project status — regime discovery → representation-aware restart (post-Week-1, 2026-05-06)

> **The Week 1 result is a regime signal, not a project failure.** A post-critic K7 rerun fixed budget/planner drift, concurrent shared-state dedup, workspace hydrate units, and T3 fixture coverage. Corrected K7 now passes: T1 collapses under infinite capacity, T2 exposes prefill stampede, and T3 shows `mixed_min_pressure` beating the best fixed-mode policy by about 49% on a single-source multi-resource evacuation fixture. This earns carrying mobility episodes forward, but it still does **not** justify a universal "our policy wins" claim.
>
> The current phase is **regime discovery**: map when per-site reuse is enough, when state locality matters, and when landing pressure requires mobility planning. R1/R2 (sweep + heatmaps), R3 (model profile axis), V1 (claim-cell exact validation), O1/O2 (small-N oracle + diff), and W1/W2/W3 (workload anchors) are `done`.
>
> **The next phase that those tools were always pointing at is representation-aware restart.** LLM agents are restartable computations whose progress is split across model context, runtime state, tool observations, and environment side effects; the system question is *which representation of that progress to materialize at a destination* — copy, replay, hydrate, rebuild, refetch, reuse, or discard. The regime map remains the scale experiment: once valid restart packages exist, large restart episodes determine which resource becomes the bottleneck.
>
> This framing is **provisional** — it is adopted as the active language for new workstreams (C, M, E, KVA below) but does **not** yet replace regime discovery as the project's top-line claim. Promotion to top-line happens only after Workstream C and Workstream M produce evidence.
>
> See `docs/WEEK1_REPORT.md` for the audit trail, `docs/K7_gauntlet_results.md` for the gauntlet numbers, `docs/K8_K9_regime_map_and_oracle.md` for the first regime map and oracle gap table, `docs/R4_regime_discovery_memo.md` for the current framing, `docs/L1_calibration_paper_draft.md` for the calibration-paper scaffold, and `kv-transfer-early-experiment/FINDINGS.md` for the architecture-dependent KV-vs-replay crossover result.

The current regime map hypothesis:

| Regime | What dominates | Current evidence |
| ------ | -------------- | ---------------- |
| Reuse regime | state is small or already warm | H5b: real SWE-agent bytes collapse the grouping gap |
| State-locality regime | large workspace/artifact/state transfer | H2/H5a: synthetic 1 GB workspaces produce large gaps |
| Landing-pressure regime | many workflows reconstitute at once | T2: prefill-stampede sanity check; strong baselines can also avoid replay when network is free |
| Multi-resource regime | network, prefill, workspace all finite | corrected T3: richer planner beats fixed modes on single-source evacuation |

The L0/L1/L2/L3 hierarchy that earlier drafts used as a podium has been retired (R0.1). New writing should describe the strong baseline plainly (per-site reuse + per-state mode choice) rather than as a "level."

**Repo grounding** (read these before starting any task):

- `agent_migrate_agent_repo_implementation_plan.md` — original (longer) design doc; reference, not gospel.
- `../coding-progress-ledger/ledger_progress/{core,session,serialization,queries,sidecar}.py` — agent-migrate rides on these.
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
Do not lean on ledger_progress scoring/split/reopen semantics for agent-migrate
  signals. Use ledger subtasks as graph nodes only.
Do not run real agent harnesses from inside agent-migrate. Resume validation is
  STATIC: digest checks, diff applicability, transcript prefix-equality,
  harness-config parseability. No model calls. No tool execution. No
  external test runner. If a workstream needs runtime evidence, route
  through coding-data-collection upstream.
Decision (2026-05-06): do NOT relax this rule for a one-off F2
  cut-and-resume as Vagrant evidence. A single real end-to-end smoke check
  may be run only upstream, after C4 exposes the static package table, and
  only to produce an integration-risk note: which structural assumptions
  failed, which missing bytes/setup steps were discovered, and which C/M/E
  tasks must change. It must not report verifier_success, task correctness,
  a policy winner, or a headline resume-time number.
Do not use `verifier_success` (or any task-correctness signal) as a
  agent-migrate metric. The forbidden semantic-correctness rule above covers it.
  Resume packages are scored on STRUCTURAL validity and resource cost only.
Do not silently smuggle a real-harness adapter under a new workstream
  letter. F1/F3 are deferred indefinitely; that decision is not relaxed by
  Workstream C, D, or E.
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

1. **Zero changes** — use `LedgerEvent.payload` (already `dict[str, Any]`) to carry agent-migrate-specific fields (`state_id`, `tokens`, `content_hash`, `site`, `mode`).
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
| state object       | **Not** a subtask. Carried in payload of agent-migrate `state_*` events.                     |
| mobility episode   | JSON file referencing per-workflow manifests + warmness + capacities. Not a ledger event. |
| subagent spawn     | `ADD_SUBTASK` (parent_id = planner). Not `SPLIT_SUBTASK` unless parent work is invalidated. |
| node start/end     | `UPDATE_STATUS` with `IN_PROGRESS` / `COMPLETE`.                                       |
| state invalidation | agent-migrate `state_invalidate` event. Not `INVALIDATE_SUBTASK` unless the consuming node's work also invalidates. |

`SPLIT_SUBTASK` / `REOPEN_SUBTASK` / `INVALIDATE_SUBTASK` carry **scoring semantics** in `ledger_progress`. Vagrant must not lean on them. Use ledger subtasks as **graph nodes**; do not use ledger progress scores as a agent-migrate signal.

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

**Workstreams A–E (MVP pipeline, all `done` 2026-05-05).** Trace vocabulary on `LedgerEvent` payloads with the A2 pass-through hook upstream; synthetic adapter + canonical toy fixture at `examples/traces/toy_subagent_trace.jsonl`; manifest = (nodes, state_objects, edges) by replay with bipartite source-of-truth + pairwise edge view; four closed-form cost formulas in `costs.py` (T cancels — keep the bandwidth-crossover guardrail); two MVP policies (D1 `request_level_no_reuse`, D2 `shared_state_aware`) emit plans with `reason` fields; `agent-migrate-bench` produces `results.csv` + `state_materialization_breakdown.csv` + `plots/duplication_factor.png`; cost-weighted duplication factor is the headline metric.

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

**Sensitivity tooling (`done`).** `agent-migrate-sensitivity` CLI + `run_sweep` helper grid-searches (kv_bytes, link_bps); `costs.py` carries a caveat block on load-bearing assumptions; `configs/sites_2site.yaml` uses a 5 Gbps single-flow inter-region link; 3 model profiles bracket the realistic 2025-2026 KV-per-token range.

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
`src/agent_migrate_agent/k8_regime.py` and `src/agent_migrate_agent/k9_oracle.py` landed with focused tests. `scripts/run_k8_k9.py` emits `runs/k8_regime_map/` and `runs/k9_oracle/`. The first K8 map covers all requested N × state-scale × prefill × bandwidth axes using an aggregate service-time estimator for large cells; exact K4 remains available for focused cells. K8 exact-vs-aggregate calibration on 36 sampled cells shows best-policy agreement in 24/36 cells and bottleneck-label agreement in 102/216 policy rows, so aggregate heatmaps are regime hypotheses rather than final timing evidence. K9 restricted exact search now covers four 4-workflow diagnostic cells and finds oracle gaps vs strong reuse from 50.3% to 96.7%.

**V1 — Exact validation of K8 claim cells** (`done`, 2026-05-06)
`src/agent_migrate_agent/k8_validation.py` and `scripts/run_k8_validation.py` landed. `runs/k8_validation/` plus `docs/K8_exact_validation.md` rerun seven named claim cells through exact K4 and compare best-policy agreement, policy/cell bottleneck agreement, p50/p95 relative timing error, and a trust label. Current result: all seven selected cells are `needs_exact_k4`; aggregate K8 remains useful for discovering candidate regimes, but exact K4 is required before quoting timing or bottleneck claims.

---

## § Workstream R — regime map (recent phase, mostly `done`)

**Goal.** Turn Vagrant from "does our policy win?" into "which regime is this workload and mobility event in?" The main artifact is a sweep-backed map over workload size, state scale, prefill pressure, and link bandwidth, with policy winner and dominant bottleneck reported per cell.

### R0 — Reframe and nomenclature cleanup

**R0.1 — Retire H1/D2 as the central story** (`in progress`)
Use "strong per-site reuse baseline vs richer mobility planning" in new writing. Keep old names in code and historical docs, but new docs should describe the strong baseline plainly: reuse any state already materialized at a site; for cold state choose the cheapest available materialization mode (replay, KV transfer, artifact copy, workspace hydrate, or warm reuse).

**B1 / R0.2 — Baseline naming audit** (`done`, 2026-05-06)
`docs/strong_site_reuse_baseline.md` explains that `cache_reuse` is the K-level representative of the strong L1 baseline and already includes per-state mode choice plus per-workflow destination preference. `strong_site_reuse` is a paper-facing alias in `reconstitution.py`; `cache_reuse` remains for historical compatibility.

### R1 — K8 regime-map sweep runner

**R1 — Grid runner** (`done`, 2026-05-06)
`src/agent_migrate_agent/k8_regime.py` plus `tests/test_k8_regime.py` run a deterministic grid and write `runs/k8_regime_map/regime_policy_metrics.csv` plus cell summaries.

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

**R3 — Reintegrate KV study** (`done`, 2026-05-06)
`configs/model_profiles.yaml` now carries five profiles cross-checked against the corrected `kv-transfer-early-experiment/prefill-breakeven.py` (and HuggingFace config.json for each cited model):

```text
compact_kv         Kimi-K2.6-class MLA   (kv_bpt 70,272;  prefill 14,659 tok/s)
vanilla_gqa_fp16   Llama-3-70B GQA       (kv_bpt 327,680; prefill 10,221 tok/s)
frontier_v4_fp8    DeepSeek-V4-Pro CSA   (kv_bpt 9,928;   prefill 5,607 tok/s)
glm_5_mla          GLM-5 MLA + DSA       (kv_bpt 89,856;  prefill 8,244 tok/s)
qwen3_next_hybrid  Qwen3-Next-80B-A3B    (kv_bpt 24,576;  prefill 175,316 tok/s)
```

`ModelProfile` carries `single_stream_prefill_tok_s` so architecture varies on BOTH knobs, and `src/agent_migrate_agent/r3_model_sweep.py` plus `scripts/run_r3_model_sweep.py` produce the per-profile K8 sweep with a model-aware budget that scales the K8 prefill capacity by the model's per-stream rate (link bandwidth held fixed). Pilot result over 24 cells × 5 profiles: 6 cells (25%) flip `dominant_bottleneck` across architectures even though `best_policy` does not. **Critic note (2026-05-06):** the earlier "GLM-5 full MHA, kv_bpt=1,277,952" figure from the original FINDINGS table was miscoded full-MHA arithmetic on a model that is in fact MLA; the corrected numbers come from the verified configs.

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
`src/agent_migrate_agent/k9_oracle.py` implements exact simulator-backed search for small instances:

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
`src/agent_migrate_agent/oracle_diff.py` and `scripts/run_o2.py` emit `runs/o2_oracle_diff/`. Per-cell artifacts cover destination choices, prompt modes, workspace modes, per-policy bottleneck breakdown (with an `attr` sidebar reporting the fraction of makespan with an attributed bottleneck so a sparse breakdown is not misread), and the three gaps (oracle vs mixed, oracle vs random, strong-reuse vs random). The exhaustive enumeration is shared with `k9_oracle.run_small_n_oracle` via `enumerate_oracle_plans` so the two paths cannot drift on candidate space or objective; a parity test pins the byte-identical p50.

Headline first-pass numbers (4 diagnostic cells, n=4):

| cell | oracle vs mixed | oracle vs random | strong vs random |
| ---- | --------------: | ---------------: | ---------------: |
| tiny_prefill_pressure | 35.0% | 50.2% | -13.7% |
| medium_multi_resource | 50.0% | 80.3% | -1.7% |
| monorepo_workspace_pressure | 0.7% | 33.7% | -33.6% |
| slow_link_network_pressure | 49.9% | 93.5% | -97.4% |

Negative `strong vs random` is the K7 finding restated: in slow-link / large-state cells, random_mode's diversification beats strong reuse's "everything via the cheapest cold mode at one site" choice. The `monorepo_workspace_pressure` row (oracle vs mixed = 0.7%) confirms `mixed_min_pressure` is already near-optimal when workspace bytes dominate; the other three rows show real headroom (35–50%) over `mixed`. The diff CSV identifies that the headroom comes primarily from per-workflow destination + prompt-mode choices, not workspace-mode.

**O3 — Oracle decision-motif extraction** (`not started`)
For every cell where oracle beats `mixed_min_pressure` by >25%, summarize the oracle plan structure: count of workflows on replay vs KV, count of distinct destinations used, whether oracle deliberately unbalances P50 (variance of per-workflow finish times), and which resource it sacrifices to improve median. Output: `runs/o3_motifs/motif_table.csv` plus a one-paragraph natural-language pattern per cell. Explain the gap before tuning a heuristic against it.

**O4 — Quantile-aware planner prototype** (`deferred`)
A P50-aware scoring objective that runs K4 forward over a candidate partial plan and scores `np.percentile(finish_times, 50)` instead of max-pressure. P1 already pinned a regression sentinel for this gap; O4 should be picked up only after O3 makes the motif legible, and only on the four O2 diagnostic cells.

---

## § Workstream S — workflow-local mobile state measurement

**Goal.** Stop using repository size as a proxy for migration payload. Measure workflow-local mobile state and classify what must move, what can be rehydrated, and what can be discarded.

**S1 — State-layer taxonomy update** (`done`, 2026-05-06)
`src/agent_migrate_agent/state_layers.py` enumerates 11 layers — `base_repo_checkout`, `uncommitted_diff`, `files_read`, `files_touched`, `tool_outputs`, `test_logs`, `build_artifacts`, `dependency_cache`, `retrieved_documents`, `subagent_transcripts`, `summaries_compaction` — each carrying one of `globally_available`, `cheaply_rehydratable`, `must_move`, `can_be_recomputed`, `can_be_discarded`. Mobility class assignments are domain claims (e.g. `dependency_cache` is `cheaply_rehydratable`, NOT `must_move`) pinned by tests in `tests/test_state_layers.py`.

**S2 — Mobile-state audit script** (`done`, 2026-05-06)
`audit_workflow_directory(...)` walks a workflow tree, classifies each regular file via `classify_file(rel_path)` (path-component prefix matching + extension), aggregates bytes per (layer, mobility_class), and `write_audit_artifacts(...)` emits JSON/CSV. Symlinks are not followed, hardlinks counted once. `scripts/run_s2_audit.py <workflow_dir> <out_dir>` is the CLI. Tests cover conservation of total bytes, no double-counting, classifier deep-tree dispatch (e.g., `.git/objects/pack/...`), and JSON round-trip.

**S3 — State-role classification (orthogonal to S1 layers)** (`done`, 2026-05-06)
`src/agent_migrate_agent/state_layers.py` now defines `ROLE_AT_CUTS`, canonical `role_for_layer(...)`, and `materialization_for_role(...)`; `StateObject` carries optional `role_at_cut` for manifest metadata compatibility. Roles are intentionally orthogonal to mobility classes: `dependency_cache` is `cheaply_rehydratable × performance_critical`, `uncommitted_diff` is `must_move × correctness_critical`, and model/KV state is `performance_critical`. `agent_migrate_minimal` now uses S3 role mapping for included/lazy/skipped package state instead of the previous S1-only fallback, while preserving `globally_available` materialization for base repo state. `tests/test_state_roles.py` pins canonical assignments, unknown-layer hard-fails, and role→materialization boundaries.

**S4 — Role × layer byte audit** (`done`, 2026-05-06)
`AuditReport` now carries `bytes_per_role_at_cut`, `bytes_per_layer_role`, and the four C4-ready roll-ups: `bytes_total`, `bytes_must_materialize_before_resume`, `bytes_can_lazy_rehydrate`, and `bytes_can_drop`. `write_audit_artifacts(...)` emits `audit_roles.csv` and `audit_layer_roles.csv` in addition to the existing S2 JSON/CSV artifacts. Tests assert total bytes conserve across the role projection and layer×role bytes sum back to layer bytes.

---

## § Workstream C — cut-point resume validation (active next phase)

**Goal.** Make "useful resume" a property the system can actually check without running a model or a verifier. A cut point is a deterministic position in a recorded trajectory; a resume package is a manifest + bytes; validation is a deterministic structural function. This is the workstream that earns the representation-aware-restart framing.

**C1 — Cut-point definitions** (`done`, 2026-05-06)
`src/agent_migrate_agent/cut_points.py` (`CutPoint`, `find_cut_points`, `classify_phase`, `write_cut_points_csv`, `load_trace_jsonl`) plus `tests/test_cut_points.py` (21 tests) and `scripts/run_c1_cut_points.py`. A cut point is `(trace_id, workflow_id, session_id, event_index)` where `event_index` points at the `add_subtask` for the next `llm_call` of the same `session_id`, and the predicates hold:

- next event is `add_subtask` with `node_type="llm_call"`;
- a prior `llm_call` of the same `session_id` exists in the trace (no cuts across sessions);
- no subtask is open at the cut — the in-flight check (prior llm_call has emitted `update_status complete`; no mid-flight `tool_call` or other subtask);
- every prior `state_declare` carries a non-empty `content_hash`.

Phase classification (`classify_phase`) is ordinal thirds over the *session's* total llm_calls (`early_exploration | mid_edit | pre_submit`). Smoke-tests:
- `swe_agent_pilot_s_07.json` (F2): 11 llm_calls in one session → 10 cut points, all three phases.
- `examples/traces/h5a_multi_trajectory_swe.jsonl` (multi-session): 5 sessions × 2 llm_calls → 5 cut points, one per session, no inter-session cuts.

`CutPoint` also carries `prefix_tokens` (sum of declared-state tokens at the cut) and `last_state_declared` for downstream C2/C3 consumption. Hard-fails on malformed events; `load_trace_jsonl` reports `path:line` on JSON errors. Per-critic fixes from review: in-flight check uses open-subtask set only (no brittle prior-event whitelist); cuts scoped per `session_id`; `state_invalidate` between calls does not block; `update_status` clears the open set only on `status="complete"`. Spec gaps now pinned in code: multi-workflow → per-session, < 2 llm_calls in a session → no cuts.

**C2 — Resume package taxonomy** (`done`, 2026-05-06)
`src/agent_migrate_agent/resume_packages.py` (`ResumePackage`, `StateEntry`, `WorkspaceFileEntry`, plus 5 builders) and `tests/test_resume_packages.py` (28 tests). Five package types:

```text
prompt_only
transcript_plus_harness_state
transcript_plus_diff
full_workspace_snapshot
agent-migrate_minimal     (S1-fallback today; FIXME(S3): replace with S3-classified must-materialize set)
```

Each package's `state_entries` is sorted by `state_id` for deterministic byte-identical builds; `harness_config` is recursively normalized (sorted keys, deepcopy) so caller insertion order doesn't perturb output. `agent-migrate_minimal` mobility-class mapping is now explicit per critic: `must_move → included`; `globally_available → globally_available`; `cheaply_rehydratable | can_be_recomputed → lazy_rehydrate`; `can_be_discarded → SKIPPED ENTIRELY` (no manifest entry). `ResumePackage.metrics()` returns a flat CSV-ready row (package_type, trace_id, session_id, event_index, phase, transcript_prefix_hash, n_state_entries, n_workspace_files, included_bytes, lazy_rehydrate_bytes, has_diff, has_harness) for C4 ablation. Determinism is pinned by a parametrized test across all five package types.

**C3 — STATIC resume validator** (`done`, 2026-05-06)
`src/agent_migrate_agent/resume_validator.py` (`validate_package`, `ValidationResult`, `required_state_ids`, `VALIDATION_REASONS`, `CHECKS`) and `tests/test_resume_validator.py` (22 tests). Signature: `validate_package(package, events, *, base_repo_path=None) -> ValidationResult` where `ValidationResult = (valid, reasons, checks_run)`. Failures accumulate (deduped); `checks_run` lets C4 distinguish "not validated" from "validated and passed."

Checks:

```text
INCLUDED
- transcript_prefix          : sha256 over canonical events[0:cut] matches package.transcript_prefix_hash
- content_hashes             : every state_entry.content_hash matches the trace's declared hash
                               (accumulates ALL mismatches, not just the first)
- state_coverage             : every state_id read by the next llm_call (lookahead) appears in
                               package.state_entries with materialization == "included".
                               `lazy_rehydrate` does NOT satisfy a referenced read (per critic).
                               Lookahead window: cut_index+1 to next add_subtask llm_call (post-`complete`
                               reads still counted). Falls back to "all reads in window" when
                               consumer_node_id is absent.
- harness_schema             : required keys present (cwd, open_file, env)
- diff_apply (only for transcript_plus_diff):
                               git status --porcelain empty (else dirty_base_repo)
                               git apply --check accepts diff_blob at base_commit (else diff_does_not_apply)

FORBIDDEN (and pinned by `test_no_subprocess_for_non_diff_packages`)
- starting a model · running the next tool · running tests / verifier · spinning up a real harness
```

Falsification target: `test_mutated_real_diff_fails_diff_does_not_apply` builds with the recorded diff against `x.py`, mutates the hunk body so `git apply --check` rejects, and pins `diff_does_not_apply`. Sibling test (unrelated-file diff) covers the trivial drop case. `test_dirty_worktree_refused` pins the new `dirty_base_repo` reason. All five package types pass green-path validation in `test_all_five_package_types_validate_green`.

Per-critic fixes from review: dropped early-return after first content_hash_mismatch; lookahead widened past `update_status complete`; consumer_node_id-missing fallback; `dirty_base_repo` reason added; `lazy_rehydrate` no longer satisfies tool-output coverage; `required_state_ids` exposed for C4 ablation; `checks_run` field added to `ValidationResult`.

**C4 — Cut-and-resume ablation table** (`not started`)
For each cut point × package, record `valid`, `bytes_moved`, `bytes_lazy_rehydrate_target`, `extra_setup_steps_recorded` (a count, not an execution), and dominant resource if K4 is run on the package. Output: `runs/c4_ablation/ablation.csv`.

**C5 — Two-level resume metric** (`not started`)
Replace the single "useful resume" definition with three explicit fields in result CSVs and demote the first one:

```text
model_resume_s        K4 estimate of when a model COULD start (legacy "first decoded token")
environment_resume_s  K4 estimate of when the env is structurally restored
                      (workspace, harness state, tool blobs all present)
task_resume_s         max(model_resume_s, environment_resume_s); the metric
                      that should drive policy choice
```

`task_resume_s` is the new headline. `model_resume_s` stays for KV/prefill-only stories. No `verifier_s` field — § 0 forbids it.

---

## § Workstream M — representation equivalence

**Goal.** Make "multiple valid representations of the same progress" testable, not rhetorical.

**M1 — Materialization-mode registry** (`done`, 2026-05-06)
For each (state_layer, role_at_cut) combination, enumerate valid materialization modes plus a structural validator:

```text
base_repo_checkout         clone-at-commit | reuse-warm-clone   validator: commit hash
uncommitted_diff           transfer-diff | full-workspace        validator: patch applies + file digests
dependency_cache           transfer-bytes | rerun-setup-cmd     validator: lockfile hash + binary digests
kv_cache                   transfer-kv | replay-prompt          validator: model+profile+session match
retrieved_documents        copy-bytes | refetch-stable-uri      validator: stable id match
tool_outputs               copy-bytes | rerun-cmd | discard      validator: ref-graph reachability
```

`src/agent_migrate_agent/materialization_modes.py` now exposes `lookup_materialization_modes(...)`, `materialization_registry()`, and `validate_materialization_mode(...)`. The registry is keyed by `(state_layer, role_at_cut)`, not layer alone; role-sensitive cases are pinned by tests (e.g. correctness-critical `tool_outputs` cannot use `discard`, diagnostic `tool_outputs` can). Conditional representations such as `rerun-cmd`, `refetch-stable-uri`, and `rerun-setup-cmd` are explicitly marked `conditional=True`; M1 records their structural evidence requirements but does not claim static semantic equivalence. Tests also pin machine-known validator IDs, KV model/profile/session evidence, and warm-clone commit compatibility.

**M2 — Representation ablation tests** (`not started`)
For each cut point × state object, compare equivalent representations and report bytes, latency (K4), and structural validity. Falsification target: **for ≥1 anchor × profile × cell, two materialization modes from M1 produce statistically indistinguishable resume cost despite >10× byte difference.** If the falsification target never fires, M is a tautology and the registry needs richer cells.

---

## § Workstream E — real dirty-workspace evidence (sibling-routed)

**Goal.** A1 and S1 already say production-relevant byte layers are undermeasured or zero in our current fixtures. E earns the next round of byte numbers without violating the no-real-harness rule.

**E1 — Post-run workspace snapshots via `coding-data-collection`** (`not started`)
The data lives upstream, not in agent-migrate. Extend `coding-data-collection` (sibling repo — this is a permitted change per CLAUDE.md if discussed first) to capture, for each completed SWE-agent / OpenHands run already in its pipeline:
- clean repo size at base commit
- final uncommitted diff size
- dependency cache size (with lockfile hash)
- build-artifact size
- test-log size
- tool-output bytes
- touched / read file bytes

Vagrant consumes the resulting snapshot manifests through F2 ingest. **Vagrant does not run the harness.** If `coding-data-collection` does not currently retain post-run state, the upstream PR is a prerequisite — escalate before coding.

**E1.5 — Quarantined F2 cut-and-resume smoke check** (`deferred`)
Run at most one real F2 cut-and-resume through `coding-data-collection`, not Vagrant, after C4 has produced the static ablation table for the same trajectory. Purpose: falsify hidden assumptions in the static package model, not produce a benchmark result. Output: `docs/E1_5_f2_smoke_check.md` with a blocker list only:

```text
cut_id
package_type attempted
static_validator_result
missing_bytes_or_setup_steps discovered
representation assumption falsified
required TASKS.md follow-up
```

Forbidden in this artifact: verifier/task success, semantic quality, policy ranking, and headline `task_resume_s`. If the smoke check tempts us to tune a package until the one trajectory resumes, stop and return to C4/M1/E1 coverage instead.

**E2 — Dirty-workspace threshold study** (`not started`)
For each run snapshot from E1, compare four package shapes (full_workspace, base+diff, base+diff+selected_caches, base+rerun_setup) across bytes_moved and `task_resume_s` (M1 / C5). Plot when workspace bytes cross the K8 state-locality regime threshold.

**E3 — Environment reproducibility check** (`not started`)
Per workspace, tag dependencies and build artifacts as `reconstructable` / `ambiguous` / `must-copy` based on lockfile presence + registry stability + recorded setup command. Feeds S3 role assignments — promotes "domain claim" mobility classes (e.g. `dependency_cache → cheaply_rehydratable`) into evidence-backed assignments.

---

## § Workstream KVA — KV/replay phase diagram

**Goal.** Make the architecture-dependent KV-vs-replay crossover from `kv-transfer-early-experiment/FINDINGS.md` and R3 readable as one figure.

**KVA1 — Phase plot** (`not started`)
- x-axis: prefill throughput (tok/s)
- y-axis: KV bytes/token
- contour curves: fixed bandwidths (1, 5, 25, 100 Gbps) at the equal-time crossover
- points: the five model profiles from `configs/model_profiles.yaml`

Output: `runs/kva1/kv_replay_phase.{pdf,csv}`. Pure plotting from existing artifacts; no new sweep.

**KVA2 — Architecture→regime bridge.** Subsumed by W-under-R3 (`done`, 2026-05-06). Cite, do not duplicate.

---

## § Workstream W — workload anchors

**Goal.** Anchor the synthetic regime map with non-SWE workload families chosen because they stress different state layers.

**W1 — Large-repo coding fixture** (`done`, 2026-05-06)
`src/agent_migrate_agent/workloads.py` introduces `WorkloadAnchor` + `W1_LARGE_REPO_CODING`. Per-workflow workspace ~1.03 GB (base repo 350 MB + dep cache 600 MB + build artifacts 80 MB + uncommitted diff 0.2 MB), with `tool_output_context` wired into per-workflow prompt-context tokens. `classify_regime` exercises strong reuse + mixed under K4 and labels the cell. The W1 hypothesis (`state_locality`) is asserted as a regression test under a 1 Gbps link cell.

**W2 — Data/RAG/artifact-heavy fixture** (`done`, 2026-05-06)
`W2_DATA_RAG_HEAVY` in `workloads.py`. Per-workflow must_move ~1.9 GB (retrieved docs + cleaned intermediates + generated plots), prompt_summaries wired into prompt-context tokens, and the two `globally_available` layers (base_data_bundle 500 MB + vector_index_shards 8 GB) are real `StateObject`s with initial warmness across every (source ∪ destination) site so a buggy policy that ignores warmness is falsifiable. Hypothesis-match regression test asserts `state_locality` under a 1 Gbps cell.

**W3 — Multi-agent fanout/fanin fixture** (`done`, 2026-05-06)
`W3_MULTI_AGENT_FANOUT` in `workloads.py`. Per-workflow shape: planner + 4 subagents + reviewer (the fanin step). Subagents share `shared_task_<wid>` (a structural test asserts cache_reuse emits one action per workflow for it, not K). Reviewer reads every private transcript + a `merge_<wid>` buffer state. `dependency_cache` is a real `globally_available` workspace state warmed at every site. Regime hypothesis is `landing_pressure` — N workflows × (K+2) llm_calls saturate prefill at the destination — and the canonical-cell hypothesis-match test pins it.

Week 3 success criterion: each anchor has a state-layer breakdown and a regime classification.

**W under R3 — Anchor regimes are profile-dependent** (`done`, 2026-05-06, NEGATIVE FINDING)
`src/agent_migrate_agent/w_under_r3.py` cross-runs every (anchor × model × cell) and reports per-anchor regime flips relative to the `compact_kv` baseline. Across 45 rows (3 anchors × 5 profiles × 3 cells, model-aware R3 budget):

  * **W1 (large-repo coding)**: 5/12 non-baseline rows flip — `state_locality` ↔ `multi_resource` ↔ `landing_pressure` depending on profile.
  * **W2 (data RAG)**: 3/12 flip.
  * **W3 (multi-agent fanout)**: 2/12 flip — DeepSeek-V4-Pro's compressed KV moves it from `landing_pressure` to `state_locality`.

The implication: the anchor's `regime_hypothesis` field is profile-dependent, not a single label. Future writeups must qualify "W1 is state-locality" with which model architecture it was observed under. `runs/w_under_r3/` carries the artifacts; `scripts/run_w_under_r3.py` reproduces.

## § Workstream P — heuristic policy improvements

**P1 — One-step-lookahead policy** (`done`, 2026-05-06, NEGATIVE FINDING for P50)
`mixed_lookahead` in `reconstitution.py` is a real one-step-lookahead extension of `mixed_min_pressure`: at workflow `w_i`, it scores each candidate by `max(immediate_max_pressure, post_next_workflow_max_pressure)` so a candidate that individually minimizes pressure but blocks the next workflow into a saturated dst is penalized.

**Empirical finding:** on the four O2 diagnostic cells, `mixed_lookahead` does NOT close the 35-50% P50 oracle gap O2 reported. The reason is a metric mismatch: lookahead optimizes a max-pressure proxy that corresponds to MAKESPAN, while the oracle wins P50 by deliberately UNBALANCING the herd (2 fast workflows on KV, 2 slow on replay). Lookahead does improve makespan (e.g., on `tiny_prefill_pressure` makespan drops from 0.150s to 0.124s), but P50 stays flat or worsens. `tests/test_mixed_lookahead.py` pins this as a regression sentinel — a future change that closes ≥50% of the P50 gap will fail the test, forcing review.

Follow-up if needed: a P50-aware scoring objective (e.g., quantile-aware finish-time estimator running K4 forward in the lookahead, scoring `np.percentile(per_workflow_finish, 50)` over the simulated forward state). Deferred — the negative finding here is itself an answer to "is one-step-lookahead-on-pressure the right tool for P50 closure?"

---

## § Workstream D — end-to-end restart episode (simulated)

**D1 — Simulated restart episode demo** (`not started`)
After C, S3/S4, and M1 land, drive one illustrative end-to-end episode through K4 (no real harness, no verifier). Inputs:

```text
trajectories         ≥3 F2 SWE-agent traces × ≥3 cut points each (≥9 episodes total — § 0 forbids single-episode podiums)
destinations         3 destination pools (slow-link, fast-link, prefill-tight)
packages compared    prompt_only | full_workspace_snapshot | strong_site_reuse-equivalent | agent-migrate_minimal | mixed_materialization
```

Metrics: `package_structurally_valid` (C3 result, NOT verifier_success), `bytes_moved`, `model_resume_s` (C5), `environment_resume_s` (C5), `task_resume_s` (C5), p50/p90 of `task_resume_s` across episodes. D1 is **illustrative**, not a podium — it demonstrates the framework end-to-end. The regime map remains the scale claim.

---

## Current two-week target — Weeks 4-5 (representation-aware restart)

Weeks 2-3 deliverables (R1/R2/R3, O1/O2, V1, W1/W2/W3) are `done` 2026-05-06. The new two-week window prioritizes the cut-and-resume + representation-equivalence stack so the restart-representation framing is earned by evidence.

**Week 4 (highest priority)**

1. **C1** cut-point definitions over F2 traces.
2. **C2** resume-package taxonomy module.
3. **S3** state-role classification axis (orthogonal to S1 layers).
4. **M1** materialization-mode registry.
5. **C3** static resume validator (no model, no verifier, no tool execution).

**Week 5**

6. **C4** cut-and-resume ablation table.
7. **S4** role × layer byte audit.
8. **C5** two-level resume metric in result CSVs.
9. **KVA1** phase-diagram figure (pure plotting, low risk).
10. **O3** oracle decision-motif extraction.
11. **E1** post-run workspace snapshots routed via `coding-data-collection` (escalate first if upstream PR needed).

**Deferred to Week 6+**

- **M2** representation ablation tests (depends on C3 + E1).
- **D1** simulated restart episode (depends on C4 + S4 + M1).
- **E2/E3** dirty-workspace threshold + reproducibility tagging (depends on E1 landing upstream).
- **O4** quantile-aware planner prototype (depends on O3 motif legibility).

**Definition of done for this phase**

> A static resume validator exists, role × layer × bytes is measurable on at least one workflow, and the materialization-mode registry is populated with at least one falsifying-or-confirming ablation per major layer. The headline table the project should be able to produce by end of Week 5:
>
> | Resume package        | Structurally valid? | Bytes moved | task_resume_s | Notes |
> |-----------------------|--------------------:|-----------:|--------------:|-------|
> | prompt only           | weak / no on edits  | tiny        | low model, high env | static-validator output |
> | full workspace        | yes                 | huge        | low           | upper bound on safety |
> | base + diff           | yes                 | small       | medium        | depends on lockfile reproducibility |
> | agent-migrate minimal       | yes                 | small       | bounded       | S3 must-materialize set |
> | agent-migrate + caches      | yes                 | medium      | low           | adds performance_critical layers |
>
> The restart-representation framing is promoted to top-line only after this table is produced and peer-checked. Until then, regime discovery remains the project's headline claim.

---

## Open questions

Resolved (kept here briefly for searchability): "useful resume definition" → C5; "tau choice" → S3 role classification supersedes; "state-object identity across reopen/invalidate" → S1 layers + S3 roles; "model-profile axis size" → R3; "relax § 0 for one F2 end-to-end cut-and-resume" → no relaxation for Vagrant evidence, only E1.5 upstream smoke check after C4.

Live:

- **Sweep scope vs runtime.** Answer: exact K4 is too slow for the full 1K/10K grid; K8 now uses aggregate estimation for the full map and exact K4 for focused cells. V1 adds named exact claim-cell validation; aggregate-only regions remain exploratory until validated.
- **Oracle formulation.** Current O1 exact enumeration over workflow-level mode/destination assignments is enough for the first ceiling check. Open follow-up: add pruning or solver-backed formulation only if larger N is required.
- **Strong baseline label.** Keep `cache_reuse` as the code name, or add `strong_site_reuse` as an alias for paper clarity?
- **Episode trigger explicit or implicit.** Default: explicit — episode JSON specifies `target_sites`; placement-policy is a consumer of episode setup, not the subject of K. (D2/H1/G1 stay relevant for placement; K is "what happens given the move.")
- **Site = abstract resource pool or geographic region.** Default: abstract pool — Phoenix/Seattle/Austin become labels for capacity envelopes.
- **Workspace payload for regime sweeps.** Default: use explicit distribution labels (`swe_bench`, `medium`, `monorepo`, `large_artifact`) and keep A1/S1 layer measurements as anchors.
- **Token counting at trace time.** Real harnesses may not give exact counts; estimate from text. F2 currently approximates.
- **Cut-point coverage discipline.** § 0 forbids single-episode podiums. C1 should require ≥3 trajectories × ≥3 cut points each before any C4 / D1 number is reported; revisit if F2's pool is too small to support that.
- **`coding-data-collection` upstream change for E1.** Whether post-run workspace snapshots are already retained, or whether the upstream collector needs extending. Escalate before coding E1.
