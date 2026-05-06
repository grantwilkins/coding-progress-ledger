# A2 — home-site premise audit

**Status:** done, 2026-05-05
**Scope:** label every existing multi-workflow fixture under one of four scenario classes; identify which scenario classes have not been studied; precondition the K1 episode-schema design.
**Conclusion:** every multi-workflow fixture agent-migrate currently holds models the **distributed-origin scenario**. The dominant production motivation — **single-source-evacuation** — has not been studied. Workstream K must distinguish the two on input via `MobilityEpisode.source_sites`.

## What this audit is checking

H2, H5a, and H5b all assign different sessions different `workspace_home_site` values: the canonical pattern is `phoenix, seattle, phoenix, seattle, phoenix`. That assignment is a **configuration choice** in the fixture builder (`MultiSessionConfig.sessions[i].workspace_home_site`), not a measurable fact derived from the trajectories. Collaborator 2 flagged this as the biggest conceptual fragility:

> H2/H5a require different sessions to have different workspace homes. But in a single-source migration episode, all jobs may originate from the same source. If all state starts at Phoenix, then "session A home = Phoenix, session B home = Seattle" is not a natural fact; it is a configuration choice.

The H1<D2 mechanism in H2/H5a depends on at least one session's workspace home differing from the component's globally-best site. If every workflow starts at the same source, that asymmetry doesn't exist on input — and the H1<D2 mechanism cannot fire.

## Scenario taxonomy

Four classes describe how a multi-workflow placement instance gets its home asymmetry (or doesn't):

### 1. Distributed-origin scenario

Sessions / subagents are *already* running at distinct sites at the time of the placement decision. Home asymmetry is a measurable input fact.

Concrete examples:
- Regional fan-out from a planner that has already launched subagents at multiple locations.
- A second-phase migration where a prior round placed sessions at distinct sites.
- An aggregator joining results from independently-scheduled session runs.
- A multi-tenant agent platform where sessions get scheduled to whichever pool has capacity at submit time.

The H2/H5a/H5b fixtures all model this: the builder declares "session sa lives at phoenix, sb at seattle, sc at phoenix." That's the input.

### 2. Single-source-evacuation scenario

All workflows are currently at one source site. The placement decision must move them — wholly or partially — to destination(s). Home asymmetry doesn't exist on input; it would be created (or not) by the placement decision.

Concrete examples:
- Capacity evacuation: a site must reduce load and existing jobs must leave.
- Regional failover: a region degrades or fails; sessions resume elsewhere.
- Maintenance drain: a cluster is being drained; jobs move within a window.
- Spot/preemptible capacity disappears at the source.

The cleanest match to the broader mobility-episode motivation. **Vagrant has not built a fixture for this scenario.**

### 3. Fan-in scenario

Subagents launched at different sites must merge or be reviewed together. Home asymmetry is real, and the question is where the merge point should land.

Concrete examples:
- A planner fans out 5 subagents to 5 different pools; results need to come back to one site for the planner's review step.
- Regional analytics agents that need to aggregate at a single reporting destination.

This is a special case of distributed-origin, but with a fixed merge target. None of agent-migrate's current fixtures model the merge-point question explicitly.

### 4. Regional-affinity scenario

Storage / data-residency / regulatory constraints fix some workflows' homes externally. Home asymmetry is enforced by an external rule.

Concrete examples:
- GDPR data residency: an EU customer's session must stay in the EU.
- Cluster pinning: GPUs hosting persistent KV cache for a session anchor that session.
- Sticky-routing for affinity caches: load-balancer assigns sessions to pools and they stay there.

H3 (`session_sticky`) is the policy that respects this constraint. None of agent-migrate's current fixtures model the constraint as an input.

## Labeling existing fixtures

| Fixture | Trajectories | Workspace homes | Scenario class |
| ------- | ------------ | --------------- | -------------- |
| `examples/traces/h2_multi_session_swe.jsonl` | s_07 × 3 | phoenix, seattle, phoenix | distributed-origin |
| `examples/traces/h5a_multi_trajectory_swe.jsonl` | 5 distinct | phoenix, seattle, phoenix, seattle, phoenix | distributed-origin |
| `tests/test_h5b_real_bytes.py` (dynamic, real bytes) | 5 distinct | same as H5a | distributed-origin |
| `examples/traces/g_demo_trace.jsonl` | synthetic | mixed (config-driven) | distributed-origin |
| `examples/traces/toy_subagent_trace.jsonl` | synthetic | (no per-state homes) | n/a (single-component, single workflow) |

**Every multi-workflow fixture agent-migrate holds models the distributed-origin scenario.** The H5b negative finding is *that scenario, at HEAD-sized real repos*. It does not generalize to single-source-evacuation, fan-in, or regional-affinity.

## What this implies for Workstream K

1. **`MobilityEpisode.source_sites` is load-bearing**, not cosmetic. The K1 schema must distinguish `source_sites=("phoenix",)` (single-source-evacuation) from `source_sites=("phoenix", "seattle")` (distributed-origin) on input. K0/K1 already plan for this.

2. **The H5b finding's scope is narrower than the negative-finding writeup currently states.** "D2 ≡ H1 at real bytes" is true *for distributed-origin scenarios with HEAD-sized SWE-bench-class repos at the canonical config*. It is silent on single-source-evacuation, which has not been studied. The K7 gauntlet fixtures (T2 prefill-only, T3 multi-resource) should include at least one single-source-evacuation case so the gauntlet is not just "distributed-origin under capacity."

3. **The mobility-episode usefulness map (K0) must label which scenario class each use-case implies.** From collaborator 2's section 2:
   - capacity evacuation → single-source-evacuation
   - regional failover → single-source-evacuation
   - maintenance drain → single-source-evacuation
   - spot capacity shifts → single-source-evacuation
   - cross-site fanout/fanin → fan-in (or distributed-origin → fan-in)
   - large artifact workflows → distributed-origin if multi-region; single-source-evacuation if monolithic

   **Four out of six "mobility episodes are useful" scenarios are single-source-evacuation.** Corrected K7 now models this directly in T2 and T3.

4. **D2 in distributed-origin can win when D2 in single-source-evacuation cannot.** D2's mechanism requires private-state homes that pull individual nodes within a shared-state component toward different sites. In single-source-evacuation, all private state starts at the source — there's no pull. D2 collapses to "place the whole component at the cheapest destination" and H1 makes the same choice independently. This is consistent with H5b's finding *and* with the hypothesis that mobility episodes need a richer abstraction than D2 vs H1.

## What this audit does NOT change

- The H5b numerical result is unchanged: at HEAD-sized real bytes in the distributed-origin fixture, D2 ≡ H1 to numerical noise. Re-running with single-source-evacuation homes would produce a *different* result, but that's a different fixture, not a correction.
- The K7 gauntlet pass/fail criteria are unchanged. T1 (capacity-free collapse) is scenario-agnostic; corrected T2/T3 use single-source-evacuation episodes to broaden coverage.
- A1 (workspace-payload decomposition) is independent of A2 — payload decomposition affects byte magnitudes regardless of scenario class.

## Action items rolled into K1/K6

- K1 schema includes `source_sites: tuple[str, ...]` (already in plan).
- K6 fixtures: `gauntlet_t2_prefill_only.json` and corrected `gauntlet_t3_multi_resource.json` use single-source evacuation (`source_sites=("phoenix",)`). A distributed-origin T3 sibling remains useful for the regime-map sweep but is no longer the K7 gate fixture.
- K0 calibration writeup must explicitly call out which scenario class each mobility-episode use-case maps to, so the K7 gauntlet's coverage is auditable.
