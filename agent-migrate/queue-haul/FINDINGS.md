# Queue-Haul — Findings

Power-first job dispatch: hit a grid power-shed target `S*` by moving the *right* jobs off a
source pool at least disruption. Static one-shot snapshot, one source pool with either one
destination pool or a controlled multi-destination validation, absolute watts via parameter
sweep. The math is in `formulation.md`, the parameters in `assumptions.md`. Every figure is reproducible — run the matching `plot_*.py`, which writes
`outputs/<name>.{pdf,png}` and prints a one-paragraph console report. All center numbers below
are from those reports (BF16, seed 42 unless a band is shown).

---

## 1. The pipeline (T1–T4)

Four modules turn workload + hardware assumptions into a shed plan:

```
power.py  →  the scalar prices the dispatch consumes (no node curve is ever evaluated)
instance.py →  a sampled session population (load ℓ and KV footprint m per job)
impact.py  →  per-job power freed ΔP_j and per-job move cost c_j (replay vs KV-transfer)
dispatch.py →  LP / MILP that picks which jobs move and how, + greedy / random baselines
```

### T1 — prices (`power.py`, `outputs/power_validation.png`)

The node never appears as a curve in the solver; it appears as four prices. At center:

- amortized price **`p̄ = P_busy/ρ* = 10.5 kW`** per node-unit of load (the autoscaler-drain
  upside),
- guaranteed plateau slope **`s_plat = p̄/30 = 350 W`** per node-unit (realized even if no node
  drains) — the **bracket ratio is 30×**,
- memory price **`μ = P_idle/S_node = 208 W/held session` (BF16)**, `74 W (FP8)`.

The left panel confirms the prices are consistent with the (plot-only) ramp–plateau curve: the
curve passes `(ρ*, P_busy)`, so the origin secant has slope `p̄` and the plateau has slope
`s_plat`. The right panel is the regime test `N = max(L/ρ*, S_held/S_node)`: at `L=8`, memory
binds past **154 held sessions (BF16)** / **432 (FP8)** — FP8's bigger KV capacity pushes the
crossover out by the cap ratio.

### T2 — population (`instance.py`, `outputs/instance_validation.png`)

A drawn population of **591 sessions** at center (active 179 / idle 141 / cold 271), `E[T] ≈ 64k`
tokens, mean active load `ℓ ≈ 0.053`. **Left panel:** active jobs spread along `ℓ > 0` and sit
in HBM; idle/cold jobs pin to `ℓ ≈ 0` yet still carry their KV footprint `m` — the two axes the
dispatch trades off. **Right panel (the key structural fact):** because the population is sized
to the pool (`n_jobs = occupancy·N·s_node`), the memory side `S_held/s_node = occupancy·N` is **constant (≈38)**,
independent of precision and context. Sweeping context short→long, the *measured* load term
`L/ρ*` falls through that constant line — the load→memory crossover is an **output** (BF16 ≈ 12k
tokens, FP8 ≈ 35k), not something packed into the setup.

### T3 — per-job impact (`impact.py`, `outputs/impact_validation.png`)

- **A — replay cost is not constant-rate.** `ρ_dest(T)` is a function (flat ≈63k tok/s below
  `T*≈29k`, decaying `~1/T` above), so rebuild cost is near-flat for short jobs then steepens,
  diverging from the constant-F line a fixed prefill rate would predict.
- **B — current future-impact reporting is single-price.** The code reports `p̄·ℓ` for every
  class. Raw `f,g` are stored, but token-energy work power is not calibrated yet, so the
  figure intentionally shows all classes on the single-price diagonal.
- **C — the memory score spreads widely** around `μ=208 W` (CoV **1.35**, tail-heavy context).
  The annotated load↔memory rank Spearman is measured from the synthetic draw; it is not assumed.

### T4 — solver (`dispatch.py`)

LP over `y∈[0,1]` minimizing total downtime `Σ y·c` s.t. `Σ y·ΔP ≥ S*`, source egress,
destination rebuild by the deadline, and destination headroom. **Two solves, not a branch:** on
infeasibility it re-solves to max-shed and reports the shortfall. `bind_dp` commits the
*guaranteed* floor — `dp_memory` when memory binds, else `dp_guaranteed`; never the optimistic
`dp_expected`. Validated end-to-end in T5.

---

## 2. Experiments (T5–T8)

### T5 — random vs greedy vs LP (`outputs/dispatch_validation.png`)

Two policies under the same movement budgets, isolated by active, cache-resident session
class. The y-axis is **disruption intensity**: aggregate movement downtime divided by the
requested certified shed (`s/kW`). This is the interpretable version of the LP objective:
how many session-disruption seconds we spend per kW delivered.

Ordinary chat, long chat/code, and reasoning chat collapse to the same sorted plan, so
greedy and LP have the same disruption intensity. Agentic tool loops are the useful case:
both policies meet the same certified power target, but LP cuts disruption intensity by
up to **33.5%** at **17.4 kW** (`73.0 → 48.5 s/kW`) by choosing the lower-disruption action
mix under the shared movement budgets.

### Companion — source-size sweep (`outputs/dispatch_scale.png`)

The 4-node agentic fixture is small, so this companion scales the **source** from **4 to
128 nodes** and measures the best LP cut in disruption intensity over a common feasible target
sweep. It shows two different stories:

- **Fixed destination (`48` destination nodes, `W=16`)**: the event ceiling saturates at
  **61.4 kW** and the LP cut washes out by **32 source nodes**. This is not evidence that
  coordination is useless; it says this fixed destination is the bottleneck.
- **Scaled destination (same ratio as the 4-node fixture: `dest_nodes=12N`, `W=4N`)**:
  the ceiling grows to **468.1 kW** at **128 source nodes**, and the LP still cuts disruption
  intensity by about **9.6-11.0%** for **32-128 source nodes**.

### T6 — certify low, report high (`outputs/certify_report_validation.png`)

Read two prices off each plan: the guaranteed floor vs the expected upside (once removed load
lets nodes shut off). Budgets slack to isolate the price story.

- **Compute-bound pool:** the certified floor is a small slice of the current single-price
  future proxy. The gap is exactly the **30× bracket**. Certifying against the floor never
  over-promises.
- **Memory-bound pool:** the certificate is the memory floor. The load-only future proxy is not
  the certificate here: it is **0.0–0.9×** the memory floor on this fixture, so the 30× load
  bracket **does not transfer**.
- **Containment holds:** every `S*` certified feasible under `s_plat` is met under `p̄` (PASS).

### T7 — sensitivity sweeps (`outputs/sensitivity_sweeps.png`)

Sweep `ρ*`, MFU, and the bracket ratio on a load-bound pool (where all three actually enter the
score). **Two-part result: selection is robust, absolute shed is sensitive.**

- **Which jobs to move barely changes** — ordering agreement with baseline **≥ 0.998** across
  every sweep, by both orderings (power-freed and downtime-per-watt).
- **How much you can cut moves smoothly** — the largest guaranteed reduction (center ≈ **4 kW**)
  falls **39%** across the utilization range, **9%** across MFU, **71%** across the price ratio.

### Companion — deadline sweep by class (`outputs/deadline_sweep.png`)

The deadline changes the coordination story mainly for agentic tool loops. Ordinary chat and
long chat/code reach their full shed ceilings quickly because replay is cheap enough to move
the whole active population: **15.3 kW by ~10 s** and **15.3 kW by ~13 s**, with no meaningful
greedy/LP gap. Reasoning chat reaches **14.8 kW by ~13 s** with only a tiny gap
(LP-greedy **0.2 kW**, MILP-greedy **0.1 kW**). Agentic tool loops ramp more slowly:
LP reaches **19.8 kW by ~60 s**, with a mid-deadline max LP-greedy gap of **1.7 kW**
at **24.2 s** and a deployable MILP-greedy gap of **1.2 kW**. Below the ~5 s migration
startup floor, no move completes and the reduction is zero.

### T8 — load vs memory regime (`outputs/regime_boundary.png`)

The structural payoff. The regime reduces to one inequality on total load:

```
memory-bound  ⟺  L < occupancy·N·ρ*    (because S_held/s_node = occupancy·N is constant)
R = (S_held/s_node)/(L/ρ*) = occupancy·N/(L/ρ*),   crossover at R = 1.
```

Cross `R=1` two independent ways — **(a)** raise idle/cold fraction (× two γ curves) at a fixed
~13k context, **(b)** lengthen context `E[T]` 5k→40k at center state-mix — and check they agree.

| series | R range | brackets R=1 | Jaccard @ R≈1 | feasible frac |
|---|---|---|---|---|
| (a) γ=0.5 | [0.58, 6.68] | yes | 0.00 | 0.60 |
| (a) γ=1.0 | [0.44, 5.46] | yes | 0.00 | 0.45 |
| (b) context | [0.50, 2.71] | yes | 0.00 | 0.66 |

**Spearman(dp_expected, dp_memory) = +0.091 ± 0.025** on this synthetic draw.

1. **Both walks bracket and agree at R=1** — the boundary is a property of `N`, reachable by
   idling jobs or by growing KV; `γ` only shifts *where along the active-fraction knob* you hit
   `R=1` (it delays the switch via population size, `n_jobs ∝ 1+γ`), not the `R=1` location.
2. **The flip sheds a near-disjoint job set** (Jaccard ≈ 0 everywhere): the load-ranked and
   memory-ranked dispatch plans share almost no jobs.
3. **The load and memory rankings differ enough to matter** — `dp_expected` and `dp_memory`
   are only weakly related in this draw. The regime flag genuinely chooses *which jobs move*,
   not just how the same jobs are priced.
4. **Caveat:** under realistic budgets ~33–53% of points are budget-infeasible (max-shed), and
   the LP minimizes downtime, so the realized selection also reflects move cost — in the memory
   regime the cheapest evictions are many small-KV jobs, so the shed-vs-ranking correlation panel
   shows a step at `R=1` rather than a clean ℓ→memory crossover. The headline is the low Jaccard
   near the boundary; the correlation panel is the honest, confounded one.

---

## 3. Tests (T9 — `tests/`)

| # | claim | test |
|---|---|---|
| 1 | ranking invariant under `p̄` scaling | `test_power::test_ranking_invariant_under_p_bar_scaling` |
| 2 | regime switch at the `N=max` crossover | `test_power::test_regime_crossover` |
| 3 | greedy = LP away from boundaries | `test_dispatch::test_greedy_equals_lp_off_boundary` |
| 4 | every solver output satisfies all constraints | `test_dispatch::test_every_constraint_satisfied` |
| 5 | no cold job carries load | `test_instance::test_cold_idle_carry_no_load_but_keep_kv` |
| 6 | BF16↔FP8 shifts S_node & threshold, load regime unchanged | `test_power::test_precision_shifts_memory_threshold_not_load_regime` |
| 7 | long sampled turns cap effective turn rate instead of exceeding one-session occupation | `test_instance::test_long_turns_cap_effective_turn_rate` |

T8's own results are tested in `tests/test_regime.py`: the load and memory rankings are only
weakly related in the synthetic draw, the regime flip sheds a near-disjoint set (Jaccard < 0.3),
and both walks bracket `R=1` with the measured regime flipping exactly at the `N=max` crossover.

**Correction (claim #6):** an earlier note said FP8 shifts `S_node` "~2×", but the real cap ratio is
`365/130 ≈ 2.81×`; the test asserts the true value.

---

## 4. Headline takeaways

- **Selection is robust; absolute shed is sensitive.** Which jobs to move is invariant to `p̄`
  scaling, `ρ*`, MFU, and the bracket ratio (T1 invariant, T7 ≥ 0.998 agreement); only the
  feasibility margin moves with them.
- **Two regimes, two value stories.** Load-bound: the autoscaler-drain upside is large
  (`30× bracket` in the current single-price proxy) but only the `s_plat` floor is guaranteed
  (T6 left). Memory-bound: the saving *is* the freed memory; the load bracket does not transfer
  (T6 right).
- **Coordination shows up only at tight shared-resource boundaries.** In the class-isolated
  ceiling plot, greedy now matches LP for all four classes. In the deadline companion, agentic
  loops retain a smaller mid-deadline gap: **1.2 kW** for MILP-greedy and **1.7 kW** for LP-greedy.
- **The deadline binds only for big-KV moves** (T8/memory regime); cheap short-context moves are
  capacity-bound, not time-bound (deadline companion).
- **Crossing `R=1` reorders the dispatch onto a different job set** in this synthetic draw,
  making the regime flag a real decision, not a relabeling.
