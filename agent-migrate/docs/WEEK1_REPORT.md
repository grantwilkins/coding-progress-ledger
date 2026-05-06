# Week 1 report — agent-migrate pivot, audits, gauntlet, and the negative finding

**Date:** 2026-05-05
**Scope:** Comprehensive walk-through of what happened from the H5b takeaways doc through the K7 gauntlet decision. Written for a reader who needs to evaluate whether the result is trustworthy and the next step is correct.

This report is long because we are uncertain and want full audit trails. If you only have ten minutes, read §1 (executive summary), §6.3 (the gauntlet numerical outcome), and §8 (open concerns).

## Correction note — post-critic K7 rerun

After this report was written, a critic pass found load-bearing implementation drift in K7:

- K5 policies planned against `ProfileBundle` capacities while K4 simulated custom `ResourceBudget`s.
- K4 did not coalesce concurrent materialization of the same cold `(state_id, dst_site)`.
- `workspace_hydrate_bps` was documented as bytes/s but treated as bits/s in K3/K4.
- T3 was documented as single-source/multi-resource with finite KV memory but implemented as balanced distributed-origin with KV memory uncapped.

Those issues have been fixed and K7 has been rerun. The corrected T3 fixture is single-source evacuation from phoenix to seattle/austin with finite destination KV memory. The corrected result is:

| Test | Corrected outcome |
| ---- | ----------------- |
| T1 | PASS, all policies p50 = 0.0 |
| T2 | PASS as a prefill-stampede sanity check, but not a unique mixed-planner win |
| T3 | PASS, mixed 24.35s vs best fixed-mode 48.16s |

See `docs/K7_gauntlet_results.md` for the corrected source of truth. The historical sections below are retained as an audit trail of the original Week 1 decision, but their "T3 failed / Phase 3b only" conclusion has been superseded.

---

## §1. Executive summary

**What happened.** H5b had landed an honest negative finding (real bytes on real SWE-agent trajectories collapse the H1<D2 gap). We circulated a takeaways doc; two collaborators responded. Collaborator 1 proposed a substantial reframing toward "mobility episodes." Collaborator 2 pushed back: don't pivot until the new framing passes a falsification gauntlet. We organized the week around honoring collaborator 2's gate.

**What we built.** Phase 1: four implementation-choice audits (A1 workspace-payload decomposition, A2 home-site premise, A3 D3 edge-typed grouping policy, A4 cost-model audit). Phase 2: the mobility-episode scaffolding (K1 schema, K2 warmness map, K3 resource vector, K4 fluid simulator, K5 six reconstitution policies, K6 herd adapter, K7 falsification gauntlet). Plus a substantive Phase 3b draft (`docs/L1_calibration_paper_draft.md`) ready in case the gauntlet failed.

**What the gauntlet returned.**

| Test | Pass criterion | Outcome |
| ---- | -------------- | ------- |
| T1 (capacity-free collapse / correctness check) | mixed ≈ cache_reuse ≈ 0 under math.inf | **PASS** (all policies p50 = 0.0) |
| T2 (prefill-stampede falsification of design intent) | mixed < replay_all − 10% under prefill-only cap | **PASS** (mixed 21.8s vs replay 26.5s = 18% better) |
| T3 (multi-resource bottleneck falsification) | mixed < min(fixed-modes) − 10% under all caps finite | **FAIL by 6.4 percentage points** |

**T3 specifics.** With load-aware bin-packing `mixed_min_pressure` (an upgraded version replacing the architectural-critic-flagged round-robin), the herd planner beats best fixed-mode (`cache_reuse`) by only 3.6%. `random_mode` (sanity baseline) matches `mixed_min_pressure` to within 0.4%, indicating the diversification heuristic is no better than chance on this fixture.

**Gate decision.** Per K0, all three tests must pass to enter Phase 3a (mobility-episode pivot). T3 failed honestly. **Project moves to Phase 3b — Workstream L (calibration paper).**

**Why we're uncertain.** Several legitimate questions: (a) is `mixed_min_pressure`'s heuristic still too dumb? (b) was the T3 fixture overly generous to L1? (c) could a meaningful regime exist where mixed wins by ≥10% but we didn't construct it? (d) does the K4 simulator have subtle bugs that bias results? §8 collects these.

---

## §2. Starting state — the H5b finding

The pre-week-1 state, in case you came in cold:

Workstream H culminated in **H5b** — `tests/test_h5b_real_bytes.py` plus `scripts/h5b/clone_repos.sh`. We took the H5a fixture (5 distinct cached SWE-agent pilot trajectories: cognitive_complexity, poke-env, dataclasses-json, iceprod, setup-cfg-fmt), held its homes constant (phoenix, seattle, phoenix, seattle, phoenix), and replaced its synthetic 1 GB workspace bytes with **real working-tree byte sums** computed from shallow HEAD clones of the upstream repos (~33 MB total, dominated by pok 21.6 MB and ice 11.6 MB).

The result at canonical config (`compact_kv` × `sites_2site.yaml` @ 5 Gbps): **D2 ≡ H1 to numerical noise (gap < 1e-9).** Sensitivity grid: 0% gap survival across kv_bytes ∈ {10K, 70656, 327680} × link_bps ∈ {5e9, 25e9, 100e9}.

The same trajectories with synthetic 1 GB workspace_bytes recovered the H5a 3.2 s gap exactly (`test_synthetic_1gb_recovers_h5a_gap`), proving the cost model and policy machinery are fine. The H1<D2 mechanism is real but byte-magnitude-sensitive — sub-threshold for SWE-bench-class repos at HEAD against a 5 Gbps single-flow link.

We circulated `docs/TAKEAWAYS_FOR_REVIEW.md` to a collaborator with three forward options: persist (find bigger fixtures), reframe (publish what we have as a calibration result), or stop.

---

## §3. The collaborator reviews

### §3.1 Collaborator 1 — proposed pivot

Strong reframing. Stop centering "shared-state-aware grouping (D2)." Center **mobility episodes** — burst events where N stateful workflows must reconstitute state at one or more destinations under finite resource budgets. Behavior partitions into three regimes:

- **Regime A** — destination cache hits dominate; per-site cache reuse handles it.
- **Regime B** — state locality dominates; cost is in moving bytes.
- **Regime C** — landing-pressure dominates; destination resource pool saturates.

H5b sits in Regime A. H2/H5a synthetic-1GB sit in Regime B. We had not modeled C. The proposal: build the framework to make all three regimes legible on one plot.

Collaborator 1 sketched a 7-task K workstream (K1-K7) and laid out a four-level abstraction hierarchy (state object / work unit / reconstitution group / mobility episode).

### §3.2 Collaborator 2 — pushback

Cautioned against accepting "mobility episodes" as the new thesis prematurely. Possible "rhetorical escape hatch" after H1/D2 collapsed. Their structure:

- **Audit implementation choices first.** Specifically (A) workspace-payload decomposition: maybe HEAD-tree bytes aren't what would actually be mobile in production. (B) Home-site premise: H2/H5a/H5b homes are configuration choices that may not match real scenarios. (C) D2 (connected-component grouping) may be the wrong "state-aware policy" — try edge-typed grouping. (D, E) Cost-model assumptions (faster-prefill bias under infinite capacity, additive vs pipelined).
- **Pre-stress-test the new framing before pivoting.** Three falsification tests:
  - T1: capacity-free collapse (∞ caps → reduce to L1).
  - T2: single-resource bottleneck (prefill-only → fixed modes differ).
  - T3: multi-resource bottleneck (all caps → mixed beats fixed-mode).
- **Have a calibration paper ready as the alternative**, not a regret-mode fallback.

### §3.3 Tension between the reviews

Collaborator 1 said "build it." Collaborator 2 said "don't build it until you've audited and prove it earns the pivot." We resolved by honoring both: do the audits first, then build minimal scaffolding, then run the gauntlet, then commit.

---

## §4. Plan structure

We wrote `/Users/grantwilkins/.claude/plans/i-think-the-broader-luminous-creek.md` (~2400 words) committing to:

```
Phase 1 audits (~3 days)        — A1, A2, A3, A4
       │
Phase 2 Workstream K (~5 days)  — K0 calibration, K1-K6 scaffolding, K7 gauntlet
       │
       ▼
   K-decision-point
       │
   ┌───┴───┐
   ▼       ▼
Phase 3a  Phase 3b
(K-pivot) (Workstream L — calibration paper)
```

Both Phase-3 paths got real artifacts. Phase 3a's K8-K11 task list was sketched; Phase 3b got `docs/L1_calibration_paper_draft.md` with thesis, contribution claim, four headline figures, anchor data, scope limits, stopping criterion, and venue claims.

Five "load-bearing" design decisions were named in the plan (with my defaults):
1. Episode trigger explicit, not implicit.
2. Gate is binary, manual review, no 2-of-3.
3. "Useful resume" = first decoded token at destination.
4. Site = abstract resource pool.
5. Workspace payload for K = the union from A1's audit, not the current default.

The plan was approved unchanged.

---

## §5. Phase 1 audits

### §5.1 A1 — workspace-payload decomposition

`docs/A1_workspace_payload_audit.md` + `tests/test_a1_workspace_payload.py` (50 tests).

Decomposed the H5b workspace payload into 8 candidate layers:
```
repo_tree_bytes, git_diff_bytes, touched_file_bytes, read_file_bytes,
tool_output_bytes, test_log_bytes, build_artifact_bytes, dependency_cache_bytes
```

Computed per-(sid, layer) bytes for the 5 H5b repos at HEAD:

| sid | repo_tree | git_diff | touched | read_file | tool_out | test_log | build | dep_cache |
| --- | --------: | -------: | ------: | --------: | -------: | -------: | ----: | --------: |
| cog | 21,922 | 0 | 1,485 | 12,130 | 46,096 | 11,570 | 0 | 0 |
| pok | 21,588,279 | 0 | 2,613 | 20,267 | 27,133 | 0 | 0 | 0 |
| dcj | 301,091 | 0 | 587 | 25,563 | 33,755 | 0 | 0 | 0 |
| ice | 11,568,017 | 0 | 564 | 16,030 | 12,427 | 0 | 0 | 0 |
| scf | 57,062 | 0 | 560 | 15,139 | 76,616 | 3,030 | 0 | 0 |

Then re-ran the H5b H1-vs-D2 calculation under each interpretation (8 layers + 3 combos = 11 rows). **Every row showed D2 ≡ H1 to numerical noise.** The H5b finding is robust to payload-interpretation choice within the regime measured.

**Critical caveat (audit-honesty critic flagged this — and we accepted it):** the four layers most production-relevant (`build_artifact`, `dependency_cache`, `test_log`, persistent KV state) are exactly those zero in shallow HEAD clones. A real running agent with `pip install`-ed dependencies and `pytest`-run artifacts would have hundreds of MB of those layers — easily clearing the regime-flip threshold. A1's headline ("no row flips the regime") is honest *for the regime measured* but the unmeasured regime is the production one. This is documented in A1's "What this audit cannot measure" section but the audit critic later flagged that the headline leads a skimmer to the wrong conclusion.

### §5.2 A2 — home-site premise audit

`docs/A2_home_site_premise.md` (no code).

Labeled every multi-workflow fixture in the repo under one of four scenario classes:
- **distributed-origin** (sessions already at distinct sites): H2, H5a, H5b, g_demo
- **single-source-evacuation** (all workflows at one source): NONE (gap)
- **fan-in** (subagents merging from different sites): NONE
- **regional-affinity** (storage/data residency-bound): NONE

**Finding:** every multi-workflow fixture agent-migrate currently holds models distributed-origin. The dominant production motivation — single-source-evacuation (capacity evacuation, regional failover, maintenance drain, spot-capacity shifts; 4 of 6 "useful" mobility-episode scenarios per K0) — has not been studied. This precondition K1's `MobilityEpisode.source_sites: tuple[str, ...]` field as load-bearing for the new schema, and required K6 to ship single-source-evacuation variants of T2/T3.

### §5.3 A3 — D3 edge-typed grouping policy

`docs/A3_edge_typed_policy_audit.md` + 17 tests + ~190 LOC in `src/agent_migrate_agent/policies.py` (extending the existing policy registry with `shared_state_aware_typed`).

D3 weights edges by `(state.layer, state.lifetime)`:
- `prompt_context` + `persistent` (system_prompt) → multiplier 0 (zero out global edges)
- `prompt_context` + `shared` (issue_text) → 1.0
- `prompt_context` + `ephemeral` (tool_output) → 0.5
- `workspace` + any → 5–10 (strong home affinity)
- `memory` + `persistent` → 0.5

**Surprising finding (this is important):** D3 is NOT strictly ≤ D2.

| Fixture | H1 | D2 | D3 | Verdict |
| ------- | -: | -: | -: | ------- |
| toy | 0.533 | 0.533 | 0.533 | tie |
| g_demo | 0.624 | 0.624 | 0.624 | tie |
| h2_multi_session_swe | 0.154 | 1.738 | 0.195 | D3 −1.54s vs D2 |
| h5a_multi_trajectory | 0.222 | 3.422 | 0.330 | D3 −3.09s vs D2 |
| **h5b real bytes** | 0.149 | 0.149 | **0.257** | **D3 +0.108s vs D2 (worse!)** |

Why D3 > D2 on H5b: D3 zeros out the system_prompt edge → 5 components instead of 1. Each component pays system_prompt at its chosen site. With 5 phoenix-or-seattle components, system_prompt is materialized 5 times instead of once. D2's overgrouping (1 component, 1 system_prompt materialization at the cheaper site) wins when the workspace bytes are too small to outweigh component-level dedup overhead.

**Implication for K**: the L1-vs-L2 distinction is structural (per-(state, site) vs per-(component, site) materialization), not about edge-typing crudity. K3's resource cost MUST charge per-(state, site) — flagged as a load-bearing constraint in K0.

### §5.4 A4 — cost-model audit writeup

`docs/A4_cost_model_audit.md` (no code).

Documented four load-bearing assumptions in agent-migrate's cost model:
1. **Additive cost** (`transfer + prefill`) — biases AGAINST grouped policies. Real systems pipeline (`max(transfer, prefill)`). Accepted; future workstream M could refine.
2. **Faster-prefill bias under infinite capacity** — D2 always picks seattle on linear-session traces; this is the H5b cancellation. **Resolved by K4's fluid simulator (the load-bearing reason K exists).**
3. **Decode time omitted** — cancels in same-trace policy comparisons. Accepted.
4. **Raw-bytes KV (no compression)** — 3-4× pessimistic per CacheGen; within sensitivity range.

---

## §6. Phase 2 — Workstream K

### §6.1 K0 calibration writeup

`docs/K0_calibration.md` defined the central concepts, locked the 4-level hierarchy, mapped use cases to scenario classes, specified the resource-vector consumption table, and stated the three falsification tests precisely. Pre-accepted A1-A4 findings.

### §6.2 K1-K6 scaffolding

| Task | Module | LOC | Tests |
| ---- | ------ | --- | ----- |
| K1 | `src/agent_migrate_agent/episode.py` | ~155 | 15 |
| K2 | `src/agent_migrate_agent/warmness.py` | ~135 | 12 |
| K3 | `src/agent_migrate_agent/resources.py` + `profiles.py` extension | ~200 | 19 |
| K4 | `src/agent_migrate_agent/fluid_sim.py` | ~360 | 10 |
| K5 | `src/agent_migrate_agent/reconstitution.py` | ~400 | (covered by K7) |
| K6 | `src/agent_migrate_agent/adapters/herd.py` | ~190 | (covered by K7) |
| K7 | `tests/test_k7_gauntlet.py` + 3 fixtures + `runs/k7_gauntlet/gauntlet_results.csv` + `docs/K7_gauntlet_results.md` | ~280 | 4 |

K3 explicitly enforces L1 (per-(state, site)) materialization semantics per A3's load-bearing finding. K4 was the largest single piece (proportional-share fluid resources, KV memory as a capacity with LRU eviction, deterministic event-ordered loop, no queues/admission/scheduler per the carve-out in CLAUDE.md). K5 ships seven policies: `min_cost_independent`, `replay_all`, `kv_all`, `cache_reuse`, `workspace_sticky`, `mixed_min_pressure`, `random_mode` (the sanity baseline added per architectural critic). K6's herd adapter procedurally generates manifests from log-normal byte distributions seeded for byte-deterministic regenerability.

### §6.3 Mid-build sonnet critics

After K5 landed (before K6 + K7), we dispatched three sonnet critics in parallel:

**Critic 1 (architectural).** Findings incorporated:
- `mixed_min_pressure` was a static round-robin, NOT a true oracle. Renamed in docstring + warned readers.
- `random_mode` policy added as the missing sanity baseline.
- `T1` reframed as "simulator correctness check," not "falsification test."
- Documented that T2/T3 use procedurally-generated fixtures (so a pass is necessary but not sufficient for any external claim).

**Critic 2 (implementation).** Findings incorporated:
- **Initial warmness was never enforced against KV capacity.** Added `_enforce_kv_capacity` calls at episode start (real correctness fix).
- Documented that K4 uses *equal share*, not max-min fair share (known limitation, conservative bias for T3).
- Documented `_enforce_kv_capacity` is O(N × manifest_size) per finishing action (slow at N=10000+).
- Documented the resource-conservation test asserts 100% utilization (tight specific case, not general invariant).

**Critic 3 (audit-honesty).** Findings incorporated:
- L2 hierarchy relabeled as "lateral, not improvement" — no fixture shows L2 strictly better than L1.
- Mobility-episode usefulness map qualified: spot/preemptible is the WEAKEST useful case; very-fast intra-region fabrics is the dominant production regime.
- T1 reframed as correctness check (matched critic 1).
- Workstream L scaffold (`docs/L1_calibration_paper_draft.md`) written substantively (300+ lines, not the one-paragraph fig-leaf the critic warned against).

The critics found one real bug (initial warmness KV cap), two real omissions (random_mode policy, L1 paper substance), and several framing issues.

### §6.4 The gauntlet — K7

`tests/test_k7_gauntlet.py`, four tests:
1. `test_t1_capacity_free_collapse`
2. `test_t2_prefill_stampede`
3. `test_t3_multi_resource_bottleneck`
4. `test_emit_gauntlet_results_csv` (always-runs telemetry)

All fixtures are N=100 workflows from `build_herd_episode` with deterministic seeds. T1 uses `tiny` workspace (~30MB median, balanced asymmetry, distributed-origin). T2 uses `tiny`/medium prompt tokens with a single source (single-source-evacuation). T3 uses `medium` workspaces (~500MB median) with balanced asymmetry across 3 sites and finite caps on all four axes (network 5e9 bps, prefill 30K tok/s, workspace_hydrate 1e9 bps, kv_memory math.inf).

**Results CSV** at `runs/k7_gauntlet/gauntlet_results.csv`:

| Policy | T1 p50 (s) | T2 p50 (s) | T3 p50 (s) |
| ------ | ---------: | ---------: | ---------: |
| min_cost_independent | 0.0 | 26.52 | 97.27 |
| replay_all | 0.0 | 26.52 | 258.01 |
| kv_all | 0.0 | 0.0 | 253.32 |
| cache_reuse | 0.0 | 26.52 | **87.88** |
| workspace_sticky | 0.0 | 26.52 | 89.24 |
| mixed_min_pressure | 0.0 | 21.75 | 84.69 |
| random_mode | 0.0 | 1.25 | 84.33 |

**T1 PASS** — all policies p50 = 0.0 under math.inf capacity. The simulator correctness check holds.

**T2 PASS** — under prefill cap = 30K tok/s, replay_all/min_cost_independent/cache_reuse/workspace_sticky all stampede prefill (26.52s). kv_all bypasses prefill (network = inf) → 0s. mixed_min_pressure half-replays half-KV → 21.75s, 18% better than replay_all. random_mode happens to hit kv_transfer mostly and finishes at 1.25s — beats mixed by accident, which is interesting but not load-bearing.

**T3 FAIL by 6.4 percentage points.** Best fixed-mode (cache_reuse) at 87.88s. mixed_min_pressure (with the upgraded load-aware bin-packing replacing the original round-robin) at 84.69s — 3.6% better, below the 10% bar. random_mode at 84.33s, **indistinguishable from mixed within 0.4%** — meaning the diversification heuristic is no better than chance on this fixture.

The first version of `mixed_min_pressure` was a pure round-robin that scored 89.24s (tied workspace_sticky). After incorporating the architectural critic's "load-aware bin-packing" suggestion, it improved to 84.69s. Real but not enough.

We chose NOT to tune fixture parameters (workspace size distribution, prefill cap, network bps, N) to manufacture a T3 pass. The audit-honesty critic explicitly warned against this, and we accepted that warning.

### §6.5 The gate decision

Per K0:
> The phenomenon is **legible** when, on a sweep over (N workflows, workspace_bytes_distribution, warm_cache_fraction, link_bps, prefill_budget, kv_memory_budget), the time-to-useful-resume CDF for at least three reconstitution policies is **separated by regime**: there exist regions A, B, C in the parameter space where (a) the bottleneck-attribution chart names a different dominant resource, and (b) the policy ordering on the CDF differs.

The K7 gauntlet uses three carefully-chosen capacity regimes (∞, prefill-only, multi-resource) at one parameter cell each. T1 trivially passes (correctness). T2 demonstrates that under prefill-only saturation, mixing modes helps. T3 demonstrates that under multi-resource saturation, mixing modes does NOT help by enough.

The honest reading: the K abstraction has measurable value in the prefill-stampede regime (T2's 18% improvement), but on the multi-resource canonical fixture, L1 (cache_reuse — per-state intelligent mode dispatch with warm-route preference) is competitive enough that herd-level planning does not earn a 10% advantage. Per the binary gate criterion, **the K-pivot is not earned**.

The marked test is `@pytest.mark.xfail(strict=True)` — if a future smarter heuristic clears the bar, the test will surface as an unexpected pass and re-enter the gate decision. That's the intentional door for legitimate (not tuning) future work to revisit.

---

## §7. What Phase 3b inherits

Workstream L is the path. Concrete artifacts already on disk:

- **`docs/L1_calibration_paper_draft.md`** (300+ lines) — thesis, contribution claim, headline figures, anchor data, scope limits, stopping criterion, venue claims.
- **All Phase 1 audit writeups** (A1, A2, A3, A4) — directly supportive evidence for the negative-result paper.
- **The Phase 2 K scaffolding** (K1-K6 + K7 gauntlet) — *not wasted code*. This is the regime-map measurement framework. Can be re-used for OpenHands traces, monorepo-class repos, real herd captures.
- **The K7 gauntlet results CSV** — concrete numbers for figure 4 of the calibration paper.
- **3 committed gauntlet episode fixtures** — reproducible by anyone.

The calibration paper claim, in one sentence: *for SWE-bench-class workloads at observed scales, L1 (per-site cache reuse + per-state intelligent mode dispatch) explains most of the state-locality benefit; herd-level planning shows measurable but sub-threshold improvement on multi-resource fixtures and is dominated by smart per-state policies. The phenomenon claim requires either workspaces well above the regime-flip threshold (~50 MB minority-home cross-site bytes) or explicit prefill-only saturation (T2 regime).*

This is honest and useful. It tells the field where to focus mobility-aware scheduler effort (failover/evacuation, large-artifact workflows) and where to stop (per-request agent placement at small workload scales).

---

## §8. Open concerns we want a critic to look at

These are the things we are uncertain about, in roughly decreasing order of impact on the gate decision.

### §8.1 Is `mixed_min_pressure` legitimately as smart as it should be?

The current implementation is a one-pass greedy: for each workflow in deterministic order, scan candidate (prompt_mode, dst_site) pairs, pick the one minimizing predicted max-resource utilization across (network, prefill, workspace_hydrate). For each workspace state, choose between `WORKSPACE_HYDRATE` (local) and `ARTIFACT_COPY` (cross-site) by which would push the bottleneck higher.

Concerns:
- Workflow ordering is alphabetic, not demand-sorted. A larger-workspace workflow planned later may find all destinations equally loaded and not get the placement it deserves.
- The "predicted max" metric is normalized only by per-axis capacity, not by per-axis cost-per-unit. It treats 1 prefill-token-load and 1 byte-load as comparable, which they aren't.
- The decision "ARTIFACT_COPY vs WORKSPACE_HYDRATE" is per-state local; a smarter version would jointly optimize all workspace states for one workflow.
- An offline ILP (deferred K9) might land 10-30% better. We don't know.

**If a critic builds a sharper greedy or finds that the greedy is mathematically near-optimal**, that changes the gate decision direction.

### §8.2 Is the T3 fixture overly generous to L1?

T3 uses N=100, medium workspaces (~500MB), balanced asymmetry, 3 sites, network 5e9, prefill 30K tok/s, workspace_hydrate 1e9. cache_reuse at 87.88s implies the simulator is finding ~14× speedup over single-flow serial (which would be ~1300s for one transfer at 5 Gbps × 100 workflows × 500 MB).

Concerns:
- Maybe `medium` is the wrong distribution. With `monorepo` (~5 GB median), the cross-site bytes per workflow are 10× larger. cache_reuse's per-state min-cost dispatch might no longer find a comparable winner; mixed planning's diversification might dominate. We didn't test this.
- The `cache_reuse` policy is morally L1+per-state-mode-dispatch. A policy that's morally L1+per-state-mode-dispatch+per-workflow-load-balancing might match `mixed_min_pressure` exactly without invoking the K abstraction. If so, the K abstraction's value-add is "destination load balancing" alone — which is L1 + 5 LOC, not L3.
- The 10% threshold is arbitrary. If the gauntlet bar were 5%, T3 would PASS. We don't have an external justification for 10%.

### §8.3 Could the K4 simulator have a bug that biases against mixed_min_pressure?

The simulator uses equal share (not max-min fair share), which the implementation critic flagged. The bias direction is conservative for T3 (overstates contention, making mixed look worse than under true max-min). But what if there's a different bias?

Concerns:
- KV memory enforcement runs after every action completes, walks ALL manifests to compute resident bytes. For N=100 manifests with 3 states each, that's 300 lookups per finishing action. Plausibly correct but unverified.
- The "next_event_dt" computation uses `min(dt across active actions)`. If two actions are within `EPS = 1e-12` of finishing simultaneously, both finish at the same step. For long-running episodes (T3 makespan = 87s, with thousands of events), cumulative float error could shift the boundary subtly.
- Sequential workflow chaining: when an action with dt=0 (warm hit) finishes, the loop advances by 0 and re-enters; the second action starts on the next iteration. We tested this with one-workflow + warm-then-cold; we did not test multi-workflow warm-then-cold under contention.
- The `random_mode` policy passes a `seed=0` into K7's runner. If the random selections happen to produce a near-optimal placement, it would mask a real difference between mixed and chance.

### §8.4 Did we make the right call by not tuning the fixture?

Audit-honesty critic warned that the K6 fixtures are generated code; the parameters are under our control; T2/T3 wins can be manufactured. So we left the canonical fixture's parameters at sensible defaults (medium workspace, balanced asymmetry, 3 sites, capacity sized to match sites_3site.yaml's documented values).

But "sensible default" is a choice. A skeptic could argue:
- We should have run a 2D parameter sweep (N × workspace_size) and reported the percentage of cells where mixed beats fixed-mode by ≥10%. T3's pass criterion would be "≥50% of cells." That's the synthetic-sweep clause from the original phenomenon-demonstrated gate.
- Or: use the H5b/H5a real-trajectory state distributions as inputs (collaborator 2 §5A's "use real traces only for state distributions"), run on synthetic herds parameterized by those, and report the regime where mixed wins.

Both are reasonable; neither was done. T3's single-cell evaluation is one data point.

### §8.5 Is `cache_reuse` getting an unfair advantage?

`cache_reuse` (K5) implements: warm hit at any dst → WARM_REUSE; cold → cheaper of replay/KV per state via `choose_min_cost_mode`; routes each workflow to the destination with most warm hits, breaking ties on lowest cold cost.

Under T3's cold start (warm_cache_fraction=0), every state is cold at every destination. So the policy degenerates to: per-state cheapest mode, picking dst with cheapest total cold cost. That's essentially per-workflow min-cost-independent placement at the destination level, then per-state min-cost mode. Effectively equivalent to `min_cost_independent` IF min_cost_independent picks the same dst — but min_cost_independent picks dst per-state, not per-workflow.

The 10× speedup of cache_reuse vs min_cost_independent (87.88s vs 97.27s) is from this per-workflow vs per-state dst choice. It's a real algorithmic difference but it's also basically "L1 + don't fragment a single workflow across destinations."

Concerns:
- Should there be a `min_cost_independent_per_workflow` policy that's a strict L1 with per-workflow dst pick? It would land at ~87.88s and be the right L1 baseline. Currently `cache_reuse` is doing double duty.
- If there were such a policy, the gauntlet bar (mixed beats it by 10%) would be the same; so this concern is about labeling, not outcome.

### §8.6 Is K3's "warm cache short-circuits to zero" correct under simulator dynamics?

K3 returns `ResourceCost(0, 0, 0, kv_resident, 0)` for warm hits. The simulator builds `remaining = {NETWORK: 0, PREFILL: 0, WORKSPACE: 0, KV_MEMORY: 0}`. The `kv_resident` field on the action is stored but never read by `_enforce_kv_capacity` (which walks the warmness map directly to compute resident bytes).

Implementation critic flagged this: warm-hit's `kv_resident_bytes > 0` is dead in the simulator. The state was already counted via the warmness map's existing entry. So eviction WAS triggered when needed. But the dead `kv_resident` field is a code smell that suggests the original design intent (track per-action KV pressure separately) didn't survive into the implementation. This is not a correctness bug but the next maintainer will be confused.

### §8.7 How brittle is the result to T3's specific parameters?

We should know:
- What workspace_bytes_distribution makes mixed beat fixed-mode by ≥10%? (Probably `monorepo`. Untested.)
- What N makes the gap appear? (Probably much larger N, say 10000. Untested — at N=10000 the K4 simulator may be too slow, per implementation critic's O(N×states) flag.)
- What asymmetry makes the gap appear? (`skewed`, where 80% of workflows are at one source, may put pressure on a single src→dst link that mixed can avoid by routing to alternate destinations. Untested.)

Without a 2D parameter sweep, we don't know whether T3 fails everywhere or just here. A real systems paper would report the regime map; we have one cell.

---

## §9. What would we do differently?

Honest retrospective:

1. **Run a 2D parameter sweep instead of three one-cell tests.** Should have produced the regime-map plot (Phase 3a's K8-K11) in week 1, not deferred it. The single-cell gauntlet result is brittle.
2. **Implement the K9 ILP oracle earlier.** Without an offline-optimal benchmark, we don't know how much of the 3.6% mixed-vs-cache_reuse gap is "this heuristic is dumb" vs "the abstraction's ceiling is low here."
3. **Run a maxed-out smart-greedy version of mixed_min_pressure.** Demand-sorted workflows + multi-pass refinement. We did one improvement (round-robin → load-aware bin-packing); we could have done more.
4. **Be more skeptical of cache_reuse's design.** It encodes L1 + several optimizations (per-workflow dst selection, warm-route preference). Calling it the "L1 representative" may overstate L1's strength.
5. **Document the cost-model assumptions BEFORE implementing K3.** A4 was written after K3 was already done; if A4 had come first, K3 might have used `max(transfer, prefill)` semantics for actions that have both, which would have been more realistic.

---

## §10. Where to go for ground truth

Files to read in order if a colleague is auditing:
1. `docs/TAKEAWAYS_FOR_REVIEW.md` — the H5b finding's framing.
2. `docs/A1_workspace_payload_audit.md` — the payload-decomposition result.
3. `docs/A3_edge_typed_policy_audit.md` — the surprising D3 > D2 finding.
4. `docs/K0_calibration.md` — the central concept definitions.
5. `src/agent_migrate_agent/fluid_sim.py` (~360 LOC) — the simulator.
6. `src/agent_migrate_agent/reconstitution.py` (~400 LOC) — the seven policies.
7. `tests/test_k7_gauntlet.py` (~280 LOC) — the gauntlet itself.
8. `runs/k7_gauntlet/gauntlet_results.csv` — the numbers.
9. `docs/K7_gauntlet_results.md` — the gate-decision writeup.
10. `docs/L1_calibration_paper_draft.md` — Phase 3b artifact.

The git log is also load-bearing: every commit's message is the contemporaneous reasoning. Run `git log --oneline -15` for the week-1 commit train.

---

## §11. Bottom line

**What the gauntlet says, honestly:** the K abstraction earns measurable value in the prefill-stampede regime (T2 +18%) but is dominated by L1+per-state-mode-dispatch on a multi-resource canonical fixture (T3, mixed only 3.6% better than cache_reuse, indistinguishable from random within 0.4%). This is one cell of a 2D regime map. The phenomenon-demonstrated gate fails on this single cell.

**What we cannot say from this evidence:**
- Whether mixed_min_pressure could clear 10% with a smarter heuristic.
- Whether T3 passes at larger workspace sizes or skewed asymmetries.
- Whether the K4 simulator has subtle bugs we haven't surfaced.
- Whether real OpenHands rollouts (with installed deps, build artifacts) would land in Regime B/C and rescue the pivot.

**What we *can* say:**
- L1 (cache_reuse-style: per-site cache + per-state intelligent mode dispatch + per-workflow dst preference) is structurally hard to beat at SWE-bench-class scales.
- `random_mode` is a useful sanity baseline and surfaces honestly that the heuristic is no better than chance here.
- The K scaffolding (episode schema, fluid simulator, herd adapter, regime fixtures) is reusable measurement infrastructure regardless of which Phase 3 we enter.
- The calibration paper draft is substantive enough to land if Phase 3b is the path.

**The gate decision stands**: Phase 3b (calibration paper). But we are explicitly inviting a critic to find bugs or fixable issues that change this conclusion.
