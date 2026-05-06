# L1 — Calibration paper draft (Phase 3b artifact)

**Status:** **draft scaffold** (in case the K7 gauntlet fails). If gauntlet passes, this becomes a discarded branch; if it fails, this becomes the project's external artifact.
**Audience:** systems researchers asking "is graph-grouping or mobility-episode reasoning load-bearing for LLM-agent workload placement?"

This document exists *now* (not on-fail) because the audit-honesty critic flagged that without a serious Phase 3b artifact, the K-pivot doubles as a unidirectional commitment. This is the draft contribution Phase 3b would publish.

## Tentative thesis (the claim under negative result)

> For observed coding-agent traces at observed scales, simple per-site cache reuse (L1) explains most of the state-locality benefit; more elaborate serving-group abstractions (L2 graph-grouping) and mobility-episode planning (L3) do not produce phenomena visible at SWE-bench-class workspace sizes against single-flow inter-region links. The phenomenon claim requires either (a) workspaces well above the regime-flip threshold (~50 MB minority-home cross-site bytes at 5 Gbps), (b) destination capacity saturation on prefill/network/workspace, or (c) realistic herd-burst conditions — none of which are present in current production-cached SWE-agent traces.

This is a **calibration result**, not a "shared-state placement is wrong" result. It maps the regime where agent-migrate's policy machinery does and does not produce gaps, and it documents what would need to change to relocate to a productive regime.

## Contribution claim (what the paper would offer)

1. **A 4-level abstraction hierarchy** (L0 no-reuse / L1 site-cache / L2 graph-grouping / L3 mobility-episodes) for reasoning about agent-state placement. **Empirically anchored:** L0 vs L1 has visible gaps; L1 vs L2 is a wash on real fixtures; L1 vs L3 requires production-scale capacity pressure to differentiate.

2. **A negative result** on grouping. Across two synthetic-1-GB fixtures (H2, H5a) and one real-bytes fixture (H5b), the L2 oracle (G1 brute-force) never strictly beats L1 (H1 per-site cache reuse). The H1<L2 gap that appeared at synthetic 1-GB is a property of the byte scale, not of grouping.

3. **A regime-flip threshold derivation.** At 5 Gbps single-flow inter-region link with 1.5× prefill asymmetry (phoenix vs seattle), L1 dominates L2 when minority-home workspace bytes are below ~50 MB. The H5b SWE-bench corpus sits at ~33 MB total, sub-threshold. Production-running agent workloads (with installed dependencies, build artifacts, test logs, KV-resident state) easily exceed this — but were not measured in the SWE-bench shallow-clone setup.

4. **A workspace-payload decomposition** (the A1 audit's 8 layers): repo_tree, git_diff, touched_file, read_file, tool_output, test_log, build_artifact, dependency_cache. Calibration tool for OTHER harnesses (OpenHands, LangGraph) to measure their own regime placement.

5. **A scenario-class taxonomy** (the A2 audit's 4 classes): distributed-origin, single-source-evacuation, fan-in, regional-affinity. Distinguishes which production scenarios benefit from which abstraction.

## Headline figures

1. **Figure 1: H5a → H5b drop.** Bar chart, 4 policies (D1, H1, D2, G1) × 2 fixtures (H5a synthetic 1 GB; H5b real bytes). Shows D2 drops from 3.4 s to 0.149 s while H1 stays at 0.22 s → 0.149 s. Caption: "the L2 advantage vanishes at real scale."

2. **Figure 2: Workspace-payload decomposition sensitivity.** Heatmap of `D2 - H1` gap over (workspace_payload_layer, link_bps). All measurable layers in the H5b shallow-clone setup are sub-threshold; the row showing where the gap *would* appear (computed analytically for layered values up to 5 GB) shows the threshold at ~50 MB.

3. **Figure 3: Regime map (synthetic).** Heatmap of dominant bottleneck under cache_reuse over (workspace_bytes, N, link_bps) — **the same plot Phase 3a would produce**, but framed as "here is where mobility-episodes would matter; SWE-bench currently sits in Regime A."

4. **Figure 4: K7 gauntlet outcomes.** Three bars (T1 / T2 / T3) showing pass/fail. If T1 passes but T2 or T3 fail, that's the headline calibration result.

## Anchor data

The paper draws on:
- 5 distinct SWE-bench instance trajectories from the cached pilot-zero corpus (cognitive_complexity, poke-env, dataclasses-json, iceprod, setup-cfg-fmt).
- Real working-tree byte sums from upstream HEAD clones (~33 MB total).
- 4 model profiles (frontier_v4_fp8, compact_kv DeepSeek-V3 MLA, vanilla_gqa_fp16; plus Llama-3-405B-class as appendix).
- 2-site and 3-site capacity profiles.
- The Phase 1 audit findings (A1, A2, A3, A4).

## Honest scope limits

- **Single workload type.** SWE-bench is small focused libraries. RAG, browser agents, ML-experiment agents, data-analysis agents are not measured.
- **Static placement, not online scheduling.** Capacity-aware online schedulers (Mooncake-style, Splitwise-style) are referenced but not benchmarked.
- **Cost model omits pipelining and KV compression.** A4 documents both. Pipelining would shrink some gaps; CacheGen compression shrinks KV transfer 3-4×. Headline directions are robust, but absolute numbers are loose.
- **No production trace.** The closest-to-production fixture is H5b (real trajectories + real upstream HEAD bytes). A real running-agent rollout dir was not captured.

## Stopping criterion (when to declare the calibration result final)

The paper is ready when:
1. K7 gauntlet results documented in `docs/K7_gauntlet_results.md`.
2. The 4 figures above produced from the existing K3/K4/K6 infrastructure.
3. A "what would change the result" section enumerating the regime-flip thresholds (workspace bytes, link bps, prefill asymmetry, N).
4. A "follow-on questions for the field" section listing the unmeasured fixtures (OpenHands rollout dir, monorepo-class repos, real herd-burst).

## What this paper is NOT

- Not "graph grouping is a bad idea." It says graph grouping does not pay at the byte scale of the workloads measured.
- Not "mobility episodes are a bad idea." It says K's design intent is correct but the production regime that would exercise it was not present in the corpus.
- Not "the cost model is broken." It says additive cost biases against grouped plans, KV compression is omitted, and decode is omitted — and the headline directions survive these omissions.
- Not "agent-migrate's framework is wasted effort." It says the tooling (manifests, sensitivity sweeps, fluid simulator, regime maps) is reusable for measuring OTHER harnesses' placement regimes; the SWE-bench negative result is one anchor on a larger map.

## Venues

- HotOS / HotNets / SOSP poster: 4-page calibration with the negative result + the regime-flip threshold derivation.
- SIGCOMM measurement track: A1 + A2 audits as a measurement contribution.
- arXiv prepublication first.

This draft scaffold is itself an artifact: a paper-shaped landing point for the project's evidence in hand, regardless of K7's outcome.
