# Vagrant — state of the project as of 2026-05-05

This is a candid status doc to share with a collaborator. We think we may be drifting off the original path and want a second opinion before deciding what to do next. **Read this cold; don't read TASKS.md first.** TASKS.md is the per-workstream backlog; this doc is the why-we're-uneasy.

## The thesis we set out to test

> An agent workflow with **shared state across nodes** has a different optimal placement than the same workflow treated as **N independent requests**, and the difference is **large enough to matter**.

Concretely: a "shared-state-aware" policy (D2) should beat a "per-request" policy (H1, the competitive baseline) on cost, on at least one real harness trace, by enough that the gap is robust to plausible variation in cost-model constants.

This is **not** a system. It is a derivation pipeline (trace → manifest → placement plan → cost estimate) plus an experiment that asks whether the gap is real.

## Where we are

The MVP pipeline (workstreams A–E) is green: toy trace replays, manifest derives, two policies emit plans, `vagrant-bench` produces a CSV + plot end-to-end. The pipeline can *express* the phenomenon. That part is fine.

The phenomenon-demonstrated gate is **not met.** That's the part this doc is about.

Recent ladder of findings (each fixture is more "real" than the last):

| Fixture                                | Trajectory  | Workspace bytes | H1 vs D2 result               |
| -------------------------------------- | ----------- | --------------- | ----------------------------- |
| Toy trace (single planner + 3 children)| synthetic   | synthetic       | H1 ≡ D2 (single component)    |
| F2 SWE-agent s_07 (single session)     | real        | (no workspace)  | H1 ≡ D2 (linear-session)      |
| H2 multi-session                       | s_07 × 3    | synthetic 1 GB  | **H1 < D2 by 1.6 s**          |
| H4 real workspace bytes                | s_07 × 3    | real (10 MB+)   | **H1 < D2** (direction holds) |
| H5a multi-trajectory                   | 5 distinct  | synthetic 1 GB  | **H1 < D2 by 3.2 s**          |
| H5b real bytes on H5a                  | 5 distinct  | real (~33 MB)   | **H1 ≡ D2** (gap collapses)   |

## The positive findings (real)

- **The mechanism is mathematically sound and visible in the model.** Where workspace bytes are large enough relative to the inter-site link bandwidth, D2's forced colocation pays a cross-site `8 * B / link_bps` artifact_copy that H1 avoids. The H5a 3.2 s gap is exactly `2 × 8 × 1 GB / 5 Gbps`. We can derive it on paper.
- **The pipeline can detect and quantify the gap.** `vagrant-bench` + `vagrant-sensitivity` produce auditable CSVs and plots. 278 tests, all green.
- **The mechanism survives sensitivity sweeps when the synthetic input is large.** H2 and H5a both pass the bracketing grid (kv_bytes ∈ {10K, 70K, 327K} × link_bps ∈ {5e9, 25e9, 100e9}) at 100% gap_robust with sign consistency.
- **G1 (brute-force oracle) and G2 (local search) confirm H1 is at-least-near-optimal everywhere we've checked.** D2 is never strictly better than the oracle.

## The honest negative finding: H5b

H5b is the cleanest test we've run. It takes the H5a fixture (5 distinct real SWE-agent trajectories, asymmetric homes) and replaces the **only** synthetic input — the 1 GB workspace bytes — with real working-tree byte sums computed from shallow clones of the five upstream repos at HEAD.

| sid | repo                          | bytes      | home    |
| --- | ----------------------------- | ---------- | ------- |
| cog | Melevir/cognitive_complexity  | 22 KB      | phoenix |
| pok | hsahovic/poke-env             | 21.6 MB    | seattle |
| dcj | lidatong/dataclasses-json     | 301 KB     | phoenix |
| ice | WIPACrepo/iceprod             | 11.6 MB    | seattle |
| scf | asottile/setup-cfg-fmt        | 57 KB      | phoenix |

Result at the canonical config (compact_kv × sites_2site.yaml @ 5 Gbps single-flow):

```
H1 = 0.148675 s
D2 = 0.148675 s
gap = 0.0 (numerical noise: ~1e-17)
```

Sensitivity grid: **0% gap survival** across all 9 grid points. Sign-consistent at 0.

A guarded mechanism-recovery test in the same suite scales the same trajectories' workspace_bytes back up to 1 GB synthetic and recovers the H5a 3.2 s gap exactly. So the cost model is correct; the gap is byte-magnitude-sensitive, and at HEAD-sized real repos for these instances, it's sub-threshold.

Why does it cancel exactly? D2 is free to colocate at the *faster-prefill* site (seattle, 1.5× phoenix). The prompt-context replay savings from the faster site exactly offset the cross-site workspace transfer cost at this byte scale. Solving `8 × B / 5 Gbps = prompt_context_savings` gives a regime-flip threshold near ~50 MB of cross-site workspace bytes. Real HEAD-sized repos in this corpus total 33 MB across the seattle minority.

## Why we think we may be drifting

Every "H1 < D2" win in the table above has a synthetic load-bearing input. As soon as we make every input real, the gap disappears. Specifically:

1. **H2 reused s_07 × 3 (synthetic trajectory diversity)** and used synthetic 1 GB workspaces.
2. **H4 reused s_07 × 3** but with real bytes — kept the gap, but the trajectory was still synthetic.
3. **H5a used real trajectories** but kept synthetic 1 GB workspaces — kept the gap.
4. **H5b made both real** — gap vanished.

The first time both axes are real, the gap is gone. That's a strong signal. Three possibilities, in roughly the order we find them plausible:

(a) **The phenomenon is real but only at scales we haven't realistically captured.** Monorepo-scale repos (Linux kernel: ~5 GB; Chromium: ~20 GB; ML model checkpoints: 100 GB+) would put us back in the gap-survives regime. Our HEAD-sized clones happen to be tiny. Pilot-zero SWE-bench instances trend toward small focused libraries; real production agent workloads operate on much bigger trees.

(b) **The phenomenon is an artifact of asymmetries we set up rather than measured.** "Workspace home_site" is a config knob we choose. In real life, where would workspace state's home actually live, and why would two sessions' workspaces differ in home? In SWE-bench-style, every session is independent — the home asymmetry across sessions is by construction, not observation. Maybe we're just measuring our own setup choices.

(c) **The cost model is missing something that biases us toward optimism.** The model omits KV compression (CacheGen 3-4×), inter-state pipeline overlap (real systems do `max(transfer, prefill)` not `transfer + prefill`), decode time. Including any of those would generally *shrink* gaps, not grow them — so this probably hurts the gap, not helps. But the prefill-asymmetry cancellation in H5b suggests the model may be more sensitive to site capability differences than we initially thought.

We do *not* think the model is broken. The mechanism is derivable on paper and the recovery test confirms it. The question is whether the regime where the gap shows up is the regime real workloads actually inhabit.

## Specific things we want the collaborator to push back on

Listed in roughly decreasing order of how much it would change our plan:

1. **Is the home-asymmetry-across-sessions premise actually realistic?** When a multi-session agent workload runs in production, are sessions' workspaces really anchored at *different* sites? Or do they all start at one home and the "home" concept doesn't really apply to ephemeral session state? If sessions don't have meaningful home asymmetry, the H1<D2 mechanism may not even be relevant to the workload we care about.

2. **What's a realistic workspace byte distribution for production agent workloads?** SWE-bench gives us 10-30 MB. OpenHands rollouts probably similar. But real-world agent harnesses might operate on much larger contexts (multi-repo monorepos, ML training assets, vendored deps). If P50 production workspace bytes is >100 MB, the H5b finding doesn't kill the thesis. If P50 is <50 MB, it does.

3. **Is `shared_state_aware` even the right policy to be benchmarking?** D2 colocates an entire shared-state component at one site. That's a coarse choice. Maybe the real research question is "given a graph of state-object-to-node consumption, is there a placement that beats per-session?" — i.e., G1's brute-force oracle, not D2. We've shown G1 ≡ H1 ≡ D2 on all current real-trace fixtures. If we never find a fixture where G1 strictly beats H1, the headline isn't "shared-state-aware wins"; it's "shared state doesn't change optimal placement at the regimes we can observe."

4. **Is the 5 Gbps single-flow link assumption load-bearing?** AWS single-flow inter-region maxes out around there, but RDMA-class fabrics within a region run 100s of Gbps. At 100 Gbps, the workspace-transfer term shrinks 20× and the gap is even smaller. Our justification is single-flow inter-region (paper-class network), but a critic could argue that's pessimistic for the bandwidth regime, optimistic for the latency regime, etc. Is this the right reference point?

5. **Should we pivot the headline?** Three options:
   - **Persist**: hunt down a real-trace fixture with monorepo-scale workspaces. Probably means adapting OpenHands traces from real-world repos (not SWE-bench), or a `git checkout` of each instance's `base_commit` to see if older/different commits had bigger trees. Cost: weeks; outcome uncertain.
   - **Reframe**: change the headline from "shared-state-aware beats per-request" to "**site asymmetry plus large shared state changes the optimal placement** — here is a framework to reason about it, here is the regime where it matters." We then publish vagrant as a *tool* that lets people check whether their workload is in the gap regime. The negative H5b finding becomes a feature ("we measured, here's where the gap actually shows up").
   - **Stop**: accept that the original thesis doesn't replicate at real workload scale, write up the finding as a calibration paper ("we expected X, here's why we don't see X"), move on.

6. **Are there workloads we should be looking at that we haven't considered?** Our mental model has been "SWE-agent on SWE-bench instances." But agentic systems also include: agent-driven data analytics (huge inputs), agent-driven RAG (large index state), browser-use agents (DOM + screenshot state), multi-document tool-using agents (PDF/CSV state). Each has a different state-byte profile. Is any of those a closer match to where the gap would be visible?

## What we'd do next, by default — pending pushback

If we get no input, the default plan is:

1. **Try to falsify (a) cheaply.** Pick 2-3 instances whose upstream repos are known to be large (e.g., `numpy`, `pandas`, `scikit-learn` — 100 MB+ working trees) and run H5b' on them. If the gap reappears at >100 MB scale, the byte-magnitude-sensitivity story is intact. If it doesn't, we've falsified (a) too.

2. **Independently, write up a "what does the gap depend on" sweep.** Treat workspace bytes, prefill asymmetry, link bandwidth, and prompt-context size as independent axes; map the regime where H1<D2 actually exists. This becomes either the headline of a calibration paper, or the appendix of a phenomenon paper, depending on whether (1) succeeds.

3. **Hold off on F1 (OpenHands adapter), G3, I, J.** None of those move the gate. F1 might surface different state structure but won't fix a byte-magnitude regime issue.

We'd want the collaborator's input on (1) and (3) especially before sinking another week into either.

## Where to look

- `TASKS.md` for the per-workstream backlog and detailed findings (especially H2, H4, H5a, H5b sections).
- `examples/traces/h5a_multi_trajectory_swe.jsonl` — the canonical multi-trajectory fixture.
- `tests/test_h5a_multi_trajectory.py` — H5a numerics + structural invariants.
- `tests/test_h5b_real_bytes.py` — H5b honest negative finding + mechanism-preservation recovery test.
- `scripts/h5b/clone_repos.sh` — to reproduce H5b's workspaces from scratch.
- `src/vagrant_agent/sensitivity.py` — the bracketing-grid sweep used by the gate.
- `src/vagrant_agent/costs.py` (docstring caveat block) — the load-bearing assumptions in the cost model.

If you're in a hurry: read this doc, skim the H5b section of `TASKS.md`, look at the docstring at the top of `tests/test_h5b_real_bytes.py`. That's the case for "we may be drifting" in three short reads.
