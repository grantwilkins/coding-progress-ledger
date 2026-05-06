# A3 — edge-typed grouping policy (D3) audit

**Status:** done, 2026-05-05
**Scope:** implement `shared_state_aware_typed` (D3) as an edge-type-weighted variant of D2 and run it against every existing fixture; report whether edge-typing alone is sufficient to close the H1<D2 gap on multi-session fixtures or whether the L1-vs-L2 distinction is structurally about something else.
**Headline finding:** **edge-typing helps but does not close the L1 gap, AND it can make things strictly worse than D2 in some regimes.** D3 fixes overgrouping pathology (H2/H5a) by zeroing global-replicated edges, but inherits D2's component-level materialization accounting. On H5b real bytes — where D2's all-at-one-site colocation is actually correct — D3's component fragmentation forces 5× system_prompt rematerialization and lands ~108 ms WORSE than D2.

## Why this audit

Collaborator 2's section 1C flagged D2's connected-component grouping as too crude:

> D2 groups connected components by any shared state above tau. That is too crude. It can overgroup because of tiny global prompts or system/tool prefixes. The task file already knows this and exposes tau, but the underlying issue remains: connected-component grouping is not "the" Vagrant policy.

A3 implements a richer policy (D3) that classifies each shared-state edge by `(layer, lifetime)` and applies a multiplier:

| Edge type | (layer, lifetime) | Multiplier | Effect |
| --------- | ----------------- | ---------- | ------ |
| global_replicated | prompt_context + persistent | 0.0 | edge ignored (system_prompt does not force grouping) |
| workflow_shared | prompt_context + shared | 1.0 | medium — tau=1 default |
| artifact_delta | prompt_context + ephemeral | 0.5 | small/medium |
| workspace_local | workspace + any | 5–10 | strong home affinity |
| memory KV-prefix | memory + persistent | 0.5 | architecture-conditional |
| (default) | any unspecified | 1.0 | preserve D2 behavior for unknown types |

Then form components only on edges where weighted-pair-sum > tau. Place each component at its min-cost site (same as D2). Materialization accounting: same as D2 — once per (component, site).

If D3 strictly beats H1 on H5b, the H5b "negative finding" was specifically about D2's crudeness rather than the L1-vs-L2 distinction. A3's job is to test that hypothesis.

## Numerical results

`compact_kv × sites_2site.yaml @ 5 Gbps`. Costs in seconds; lower is better.

| Fixture | H1 | D2 | **D3** | G1 | D3 vs D2 | D3 vs H1 |
| ------- | -: | -: | -----: | -: | :------: | :------: |
| toy | 0.5331 | 0.5331 | 0.5331 | 0.5331 | == | == |
| g_demo | 0.6244 | 0.6244 | 0.6244 | 0.6244 | == | == |
| h2_multi_session_swe | 0.1542 | 1.7380 | **0.1948** | 0.1542 | **−1.54 s** | +0.041 s |
| h5a_multi_trajectory | 0.2220 | 3.4221 | **0.3303** | 0.2220 | **−3.09 s** | +0.108 s |
| h5b real bytes | 0.1487 | 0.1487 | **0.2569** | 0.1487 | **+0.108 s (worse!)** | +0.108 s |

Three things to note:

1. **D3 strictly beats D2 on H2 and H5a** by 1.5–3.1 s. Removing the system_prompt edge from the grouping graph dissolves the overgrouped component into per-session sub-components, each placed at its own min-cost (= workspace home) site.
2. **D3 still does not reach H1.** On H2 it's +41 ms; on H5a it's +108 ms. The remaining gap is the system_prompt and issue_text materialization paid once per component instead of once per site.
3. **D3 is strictly worse than D2 on H5b real bytes** by 108 ms. When D2's "all-at-one-site" choice is correct (real bytes are too small to outweigh the prefill-asymmetry savings, so D2 picks seattle), D3's per-session fragmentation pays system_prompt 5 times instead of once.

## What this means

**D3 is not "D2 done right."** It's a different operating point with a different failure mode. The right way to think about it:

- D2's failure mode: overgrouping. Tiny global edges force everyone into one component, paying cross-site workspace transfers when colocation is artificially required.
- D3's failure mode: fragmentation. Removing global edges splits everyone into per-session components, each of which now pays its own copy of the (correctly globally-replicable) system_prompt.

The L1 abstraction (per (state, site) dedup regardless of grouping) avoids both failure modes. **H1 has L1 semantics; D2 and D3 both have L2 semantics (per (component, site) dedup).** The L1-vs-L2 distinction is not about grouping crudity — it's about materialization accounting. Edge-typing alone cannot close the gap.

## Implications for Workstream K

1. **K3 resource vector must use L1 semantics.** When `reconstitution_cost` accounts for a state object's materialization at a site, it must charge the cost once per (state, site), not once per (component, site). Otherwise K's `mixed_min_pressure` policy will inherit D3's fragmentation pathology and look artificially worse than H1.

2. **The K7 gauntlet's T1 (capacity-free collapse) must check L1, not L2.** Under infinite capacity, the mobility-episode model should reduce to H1's per-(state, site) dedup, not D2's per-(component, site). If T1 reduces to D2 instead, the gauntlet is biased.

3. **`mobility_episode_usefulness_map` (K0) updated.** D3's existence answers a question the original map did not raise: "if D2 is too crude, would a better grouping policy fix the gap?" — answer is no, the L1-vs-L2 distinction is structural. The K0 writeup should explicitly note this so a future reader doesn't repeat the audit.

4. **D3 stays in the codebase as an explanatory baseline.** It's useful pedagogically (shows what edge-typed grouping looks like) and as a comparison policy in K7's bottleneck-attribution charts. It is not the answer.

## What this audit does NOT change

- D2's numerics on existing fixtures are unchanged.
- H1's numerics are unchanged.
- The H5a/H5b finding stands: at the configurations measured, the H1<D2 gap collapses with real bytes, and edge-typing does not rescue it.
- K0/K1/K2 plan unchanged. Only K3 has a new constraint: materialization is per (state, site), not per (component, site).

## Pinning

`tests/test_a3_edge_typed_policy.py` (17 tests):
- Registry plumbing (D3 in POLICIES; `run_policy("shared_state_aware_typed")` dispatches).
- D3 default weights have global_replicated → 0 and workspace → ≥ 5.
- D3 == D2 on linear-session fixtures (no overgrouping).
- D3 < D2 by > 1.5 s on H2 and > 3 s on H5a.
- D3 > H1 on H2/H5a/H5b (L1-vs-L2 distinction is real).
- **D3 > D2 by ~108 ms on H5b real bytes** (component fragmentation pathology).
- G1 ≤ D3 where G1 fits.
- D3 with all-ones weights numerically matches D2.
