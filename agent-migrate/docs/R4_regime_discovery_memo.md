# R4 — regime-discovery memo

**Date:** 2026-05-06  
**Status:** current project framing after corrected K7, K8 validation, and K9 oracle broadening.

## Thesis

Vagrant should be framed as a regime-discovery framework for agentic mobility,
not as a single-policy victory story. The aggregate regime map is an
exploratory map, not a claims engine.

The current question is:

> Given an agentic workload and a mobility event, which state-reconstitution
> regime is it in?

## Claim 1: agentic mobility is state reconstitution

When a workflow moves, the system is making state available somewhere else:
prompt/KV state, workspace and artifact state, local tool outputs, dependency
caches, retrieved documents, transcripts, and summaries.

This is the durable abstraction from Week 1. Routing requests is not enough;
the unit of analysis is materializing state under finite destination resources.

## Claim 2: per-site reuse is the first serious baseline

H5b remains load-bearing: at observed SWE-agent-scale working-tree bytes, strong
per-site materialization reuse collapses the naive grouping gap. A baseline that
does not reuse already-materialized state at a site is a strawman.

The central comparison should be:

```text
strong per-site reuse + materialization choice
vs
richer mobility planning under finite landing resources
```

not D2 vs H1 as a generic policy contest.

## Claim 3: richer planning is regime-dependent

Corrected K7 reopened the mobility-episode path. After fixing budget/planner
drift, shared-state materialization coalescing, workspace hydrate units, and the
T3 fixture, K7 now shows:

```text
T1: capacity-free collapse passes
T2: prefill-stampede sanity check passes
T3: mixed_min_pressure beats best fixed-mode by about 49% on one
    single-source multi-resource evacuation cell
```

That earns carrying L3 mobility episodes forward. It does not prove that
`mixed_min_pressure` is generally best.

K8 then produced a full synthetic aggregate map over:

```text
N workflows × workspace/artifact scale × prefill capacity × link bandwidth
```

The aggregate map is directionally useful: small-state cells tend toward
prefill/network pressure, while medium and larger-state cells shift toward
workspace/network pressure.

But the K8 exact-vs-aggregate calibration is a caution:

```text
36 sampled exact K4 cells
best-policy agreement:       24 / 36 cells
bottleneck-label agreement:  102 / 216 policy rows
median relative p50 error:   48.9%
max relative p50 error:      789.2%
```

V1 then reran seven named claim cells through exact K4. All seven currently
receive `needs_exact_k4`: aggregate and exact often agree on a winner, but
timing error and bottleneck-label drift are too large for aggregate-only
claims.

The methodological rule is therefore:

> K8 is useful for finding candidate regimes. It is not yet calibrated enough
> to support quantitative claims about timing, bottlenecks, or policy dominance
> without exact K4 validation.

So the aggregate heatmaps should be treated as regime hypotheses, not final
timing evidence. Exact K4 or real workload anchors must validate any cell that
becomes a paper claim.

## Claim 4: Vagrant maps the regime

K9 broadened the small-N oracle to four restricted exact cells. The oracle
enumerates workflow-level destination, prompt-mode, and workspace-mode choices,
then evaluates each plan with K4. It does not yet search per-state destination
choices or action order.

The current diagnostic result:

| Scenario | Oracle gap vs strong reuse | Oracle gap vs mixed |
| -------- | -------------------------: | ------------------: |
| tiny prefill pressure | 56.2% | 35.0% |
| medium multi-resource | 80.6% | 50.0% |
| monorepo workspace pressure | 50.3% | 0.7% |
| slow-link network pressure | 96.7% | 49.9% |

This says the ceiling above strong reuse is real in these synthetic small cells.
It also says the current heuristic is not uniformly weak: in the workspace-heavy
cell it is nearly oracle-level, while prefill, multi-resource, and slow-link
cells still leave meaningful gaps.

The next use of K9 should be explanatory, not heuristic tuning. For each cell
where oracle beats mixed, inspect:

```text
destination choices
prompt modes
workspace modes
resource bottleneck trajectory
whether oracle exploits candidate-space choices mixed_min_pressure ignores
```

## Current path

The current paper path is a calibration and regime-discovery paper. A planner
paper becomes plausible only if exact validated cells and workload anchors show
that mixed planning improves recovery time beyond strong reuse and random
diversification in realistic mobility episodes.

Path A, calibration/measurement paper:

> For observed SWE-agent-scale workloads, strong per-site reuse captures most
> mobility benefit. Richer planning becomes relevant in identifiable large-state
> or landing-pressure regimes, and Vagrant maps that boundary.

Path B, mobility-planner paper, would require more:

```text
exact K4 validation on claim cells
real or realistic herd traces
oracle gaps that survive broader candidate spaces
a heuristic that beats random and strong baselines outside synthetic cells
```

The next work should therefore be validation and anchoring, not arbitrary
heuristic tuning:

```text
exact K4 validation for any cells used as claims
oracle-vs-mixed explanation on the four K9 cells
large-repo coding fixture with state-layer breakdown
artifact/data/RAG-heavy fixture with state-layer breakdown
multi-agent fanout/fanin fixture with state-layer breakdown
mixed-vs-random and oracle-vs-random reporting in every experiment
```

## Bottom line

Vagrant's strongest current claim is:

> Agentic mobility is a state-reconstitution problem with multiple regimes. For
> small agent workflows, strong per-site reuse is the first-order mechanism. As
> state scale grows or many workflows land together, the bottleneck shifts
> among prefill, network, workspace hydration, and KV memory. Vagrant
> identifies these regime boundaries, but any quantitative claim must be
> validated with exact simulation or workload anchors rather than aggregate
> heatmaps alone.
