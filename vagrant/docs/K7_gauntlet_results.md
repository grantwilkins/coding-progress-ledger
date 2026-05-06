# K7 — gauntlet results & gate decision

**Date:** 2026-05-05
**Fixture parameters:** N=100 workflows; sites_3site.yaml (phoenix/seattle/austin); compact_kv model.
**Verdict:** **GAUNTLET FAILED.** T1 passes (correctness check). T2 passes (prefill stampede surfaces as designed). **T3 fails** — even with a load-aware bin-packing `mixed_min_pressure`, the herd-level abstraction beats the best fixed-mode policy (`cache_reuse`) by only ~3.6%, below the 10% bar.
**Gate decision:** Per K0 calibration writeup, the honest path is **Phase 3b — Workstream L (calibration paper)**.

## Summary of outcomes

| Test | Pass criterion | Result | Verdict |
| ---- | -------------- | ------ | ------- |
| T1 (capacity-free collapse) | mixed ≈ cache_reuse ≈ H1 within 1e-6 under math.inf capacity | All policies p50 = 0.0 | **PASS** |
| T2 (prefill stampede) | replay_all > kv_all AND mixed < replay_all − 10% | replay 26.5s, kv 0s, mixed 21.8s (18% better than replay) | **PASS** |
| T3 (multi-resource bottleneck) | mixed < min(replay, kv, cache_reuse, workspace_sticky) − 10% | best fixed: cache_reuse 87.9s; mixed 84.7s (3.6% better) | **FAIL by 6.4 pp** |

Per-policy p50 time-to-resume (seconds), T3 fixture (medium workspace ~500MB, balanced asymmetry, distributed-origin):

| Policy | p50 (s) | vs cache_reuse |
| ------ | ------: | -------------: |
| min_cost_independent | 97.27 | +10.7% (worse) |
| replay_all | 258.01 | **+193.6% (stampedes prefill)** |
| kv_all | 253.32 | **+188.3% (stampedes network)** |
| cache_reuse | **87.88** | baseline |
| workspace_sticky | 89.24 | +1.5% |
| mixed_min_pressure (load-aware) | 84.69 | **−3.6%** |
| random_mode | 84.33 | −4.0% |

**Striking finding:** `random_mode` (sanity baseline) lands within 0.4% of the load-aware `mixed_min_pressure`. The K abstraction's heuristic is no better than chance on this fixture.

## What we learned

1. **L1 (per-site cache reuse + per-state intelligent mode dispatch) is hard to beat** at the configurations measured. `cache_reuse` is essentially L1 done well; it picks per-state min cost mode AND routes each workflow to the destination with most warm hits. On the T3 fixture (cold-start, 3 destinations, medium workspaces, finite caps), this is already near-optimal.

2. **Replay-all and KV-all DO stampede** under their respective bottleneck resources (3× the cache_reuse cost). The naive worst-case policies are confirmed to be naive. T2 demonstrates this cleanly.

3. **Round-robin diversification is no better than chance.** The original `mixed_min_pressure` (now superseded) round-robined modes and destinations without considering load. It tied `random_mode` exactly. The load-aware bin-packing replacement is marginally better but still within noise of chance.

4. **A 10% advantage requires either a smarter heuristic OR a different fixture regime.** The current bin-packing greedy is a one-pass myopic decision; an offline ILP (deferred K9) might find a better assignment. Or: a fixture with extreme prefill cap (e.g., 1K tok/s instead of 30K) would force a stampede that mixed actually exploits — but that's tuning the fixture to manufacture a pass, exactly what the audit-honesty critic warned against.

5. **T2 passing is meaningful** — at single-resource saturation (prefill-only), the herd-level planner DOES help by 18%. The K abstraction has *some* measurable value; it's just not 10% on the multi-resource canonical fixture.

## Gate decision: Phase 3b — Workstream L

Per `TASKS.md` Definition of done — calibration paper:

> Any K7 test fails. ... Workstream L (calibration paper) is the project's external artifact.

**The gauntlet failed by an honest 6.4 percentage points on T3.** No fixture-tuning will make this pass without manufacturing the gap. The honest landing point is the calibration paper.

### What Phase 3b inherits

- `docs/L1_calibration_paper_draft.md` — the substantial scaffold landed in K6 part 2.
- All Phase 1 audit findings (A1, A2, A3, A4) — directly relevant.
- The Workstream K artifacts:
  - `MobilityEpisode` schema (K1)
  - `WarmnessMap` (K2)
  - `ResourceCost` + `ResourceBudget` (K3)
  - `simulate_fluid` (K4) — the regime-map tooling lives on
  - 7 reconstitution policies (K5)
  - `build_herd_episode` adapter (K6)
  - 3 committed gauntlet fixtures (`examples/episodes/gauntlet_*.json`)
  - This results file (K7)

These are **not wasted code.** They're the regime-map measurement framework that the calibration paper would publish. The negative finding is the headline; the framework is the contribution.

### What Phase 3b does NOT do

- Does not declare `mobility episodes are wrong`. They are the right abstraction for evacuation/failover/herd scenarios — those just aren't visible in the workloads measured.
- Does not abandon Workstream K's scaffolding. The herd benchmark + fluid simulator are reusable for future fixtures (OpenHands rollouts, monorepo-class repos).
- Does not deprecate the existing MVP pipeline (D1, H1, D2, D3, G1). Those continue to ship.

## Workstream K status — revised

- **K1–K7:** done.
- **K8 (SWE-agent anchor):** not started; **superseded by Phase 3b**, which references the H5b finding directly.
- **K9 (offline ILP oracle):** deferred indefinitely. Would only matter if a follow-on fixture shows that the greedy bin-packing is significantly suboptimal.
- **K-pivot (Phase 3a):** **NOT entered.** Gate failed; pivot is not earned.

## Next concrete step

Begin Phase 3b: flesh out `docs/L1_calibration_paper_draft.md` with the four headline figures defined there (H5a→H5b drop, payload sensitivity heatmap, regime map, K7 gauntlet outcomes). Reuse this writeup's per-policy numbers in figure 4.

The MVP pipeline + sensitivity sweep + audit findings + this gauntlet result together form a coherent calibration contribution: **for SWE-bench-class workloads at observed scales, L1 (per-site cache reuse) explains most of the state-locality benefit; herd-level planning earns at most ~3.6% on multi-resource fixtures and is dominated by a smart per-state policy. The phenomenon claim requires either workloads above the regime-flip threshold (~50 MB minority-home cross-site bytes) OR explicit prefill-only saturation (T2 regime), neither of which is present in the SWE-bench shallow-clone setup.**

This is honest. It's also genuinely useful — it tells the field where to focus mobility-aware scheduler effort (failover/evacuation) and where to stop (per-request agent placement at small workload scales).
