# Queue-Haul — Findings

Power-first job dispatch: hit a grid power-shed target `S*` by moving the *right* jobs off a
source pool at least disruption. Static one-shot snapshot, one source pool → one destination
pool, absolute watts via parameter sweep. The math is in `formulation.md`, the parameters in
`assumptions.md`. Every figure is reproducible — run the matching `plot_*.py`, which writes
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
- two-price split `p̄_pre = 3.5 kW`, `p̄_dec = 17.5 kW` (decode is 5× prefill per busy-second,
  closing to `(p̄_pre+p̄_dec)/2 = p̄`),
- memory price **`μ = P_idle/S_node = 208 W/held session` (BF16)**, `74 W (FP8)`.

The left panel confirms the prices are consistent with the (plot-only) ramp–plateau curve: the
curve passes `(ρ*, P_busy)`, so the origin secant has slope `p̄` and the plateau has slope
`s_plat`. The right panel is the regime test `N = max(L/ρ*, S_held/S_node)`: at `L=8`, memory
binds past **154 held sessions (BF16)** / **432 (FP8)** — FP8's bigger KV capacity pushes the
crossover out by the cap ratio.

### T2 — population (`instance.py`, `outputs/instance_validation.png`)

A drawn population of **591 sessions** at center (active 178 / idle 145 / cold 268), `E[T] ≈ 66k`
tokens, mean active load `ℓ ≈ 0.045`. **Left panel:** active jobs spread along `ℓ > 0` and sit
in HBM; idle/cold jobs pin to `ℓ ≈ 0` yet still carry their KV footprint `m` — the two axes the
dispatch trades off. **Right panel (the key structural fact):** because the population is sized
to the pool (`n_jobs = α·N·s_node`), the memory side `S_held/s_node = α·N` is **constant (≈38)**,
independent of precision and context. Sweeping context short→long, the *measured* load term
`L/ρ*` falls through that constant line — the load→memory crossover is an **output** (BF16 ≈ 12k
tokens, FP8 ≈ 35k), not something packed into the setup.

### T3 — per-job impact (`impact.py`, `outputs/impact_validation.png`)

- **A — replay cost is not constant-rate.** `ρ_dest(T)` is a function (flat ≈63k tok/s below
  `T*≈29k`, decaying `~1/T` above), so rebuild cost is near-flat for short jobs then steepens,
  diverging from the constant-F line a fixed prefill rate would predict.
- **B — two-price has opposite-sign skew by class.** Prefill-skewed agentic jobs fall *below*
  the single-price diagonal (mean gap **−144 W**, a prefill discount); decode-skewed chat and
  reasoning rise *above* it (chat **+22 W**); only a phase-balanced job lands on the line.
- **C — the memory score spreads widely** around `μ=208 W` (CoV **1.35**, tail-heavy context).
  The annotated **load↔memory rank Spearman ≈ +0.05** is the seed of T8: the two rankings are
  essentially uncorrelated.

### T4 — solver (`dispatch.py`)

LP over `y∈[0,1]` minimizing total downtime `Σ y·c` s.t. `Σ y·ΔP ≥ S*`, source egress,
destination rebuild by the deadline, and destination headroom. **Two solves, not a branch:** on
infeasibility it re-solves to max-shed and reports the shortfall. `bind_dp` commits the
*guaranteed* floor — `dp_memory` when memory binds, else `dp_guaranteed`; never the optimistic
`dp_expected`. Validated end-to-end in T5.

---

## 2. Experiments (T5–T8)

### T5 — random vs greedy vs LP (`outputs/dispatch_validation.png`)

Three policies under the same movement budgets. **Ceilings separate as random < greedy < LP:
17.8 / 32.8 / 63.5 kW** (memory-bound pool, 296 jobs). Selection (greedy over random) then global
repacking transfers→replays (LP over greedy, the **~2× coordination gap**) each fit more shed
under the same links. On the fair per-watt view, **greedy lies exactly on the LP/MILP cost
frontier** (optimal where it can operate, just stops sooner) while random pays far more downtime
per watt (at `S*=17.4 kW`: random cost 756 s / shed 10.1 kW / 12% feasible, vs greedy = LP = 502 s).

### T6 — certify low, report high (`outputs/certify_report_validation.png`)

Read two prices off each plan: the guaranteed floor vs the expected upside (once removed load
lets nodes shut off). Budgets slack to isolate the price story.

- **Compute-bound pool:** the certified floor is a small slice of what you deliver. Single-price
  gap is exactly the **30× bracket**; the two-price expected runs to **≈45× guaranteed** because
  the shed jobs are decode-skewed. Certifying against the floor never over-promises.
- **Memory-bound pool (mirror image):** the shed jobs are already idle, so the node-shut-off
  bonus collapses — the 30× bracket **does not transfer**, **expected ≈ guaranteed (1.3×, range
  0.6–3.1×)**. The freed memory is itself the realized saving.
- **Containment holds:** every `S*` certified feasible under `s_plat` is met under `p̄` (PASS).

### T7 — sensitivity sweeps (`outputs/sensitivity_sweeps.png`)

Sweep `ρ*`, MFU, and the bracket ratio on a load-bound pool (where all three actually enter the
score). **Two-part result: selection is robust, absolute shed is sensitive.**

- **Which jobs to move barely changes** — ordering agreement with baseline **≥ 0.998** across
  every sweep, by both orderings (power-freed and downtime-per-watt).
- **How much you can cut moves smoothly** — the largest guaranteed reduction (center ≈ **6 kW**)
  falls **39%** across the utilization range, **9%** across MFU, **71%** across the price ratio.

### Companion — power vs deadline (`outputs/deadline_sweep.png`)

A tighter deadline caps the achievable reduction **only when the data is large to move**.
Short-context jobs finish moving almost instantly, so the limit is destination capacity (no
deadline in it) — flat at **3.58 kW** beyond ~6 s. Long-context jobs throttle on transfer time,
so the curve ramps to **16.61 kW**, plateauing by a ~140 s deadline. Below the ~5 s migration
startup floor (connection ramp, batch form, pipeline fill), no move completes and the reduction
is zero.

### T8 — load vs memory regime (`outputs/regime_boundary.png`)

The structural payoff. The regime reduces to one inequality on total load:

```
memory-bound  ⟺  L < α·N·ρ*        (because S_held/s_node = α·N is constant)
R = (S_held/s_node)/(L/ρ*) = α·N/(L/ρ*),   crossover at R = 1.
```

Cross `R=1` two independent ways — **(a)** raise idle/cold fraction (× two γ curves) at a fixed
~13k context, **(b)** lengthen context `E[T]` 5k→40k at center state-mix — and check they agree.

| series | R range | brackets R=1 | Jaccard @ R≈1 | feasible frac |
|---|---|---|---|---|
| (a) γ=0.5 | [0.48, 6.31] | yes | 0.00 | 0.47 |
| (a) γ=1.0 | [0.34, 4.17] | yes | 0.03 | 0.33 |
| (b) context | [0.38, 2.39] | yes | 0.01 | 0.53 |

**Spearman(dp_expected, dp_memory) = +0.011 ± 0.013.**

1. **Both walks bracket and agree at R=1** — the boundary is a property of `N`, reachable by
   idling jobs or by growing KV; `γ` only shifts *where along the active-fraction knob* you hit
   `R=1` (it delays the switch via population size, `n_jobs ∝ 1+γ`), not the `R=1` location.
2. **The flip sheds a near-disjoint job set** (Jaccard ≈ 0 everywhere): the load-ranked and
   memory-ranked dispatch plans share almost no jobs.
3. **Because the two rankings are independent** (`Spearman ≈ 0`) — context `T` is drawn
   independently of the rate/token distributions that drive load. The regime flag genuinely
   chooses *which jobs move*, not just how the same jobs are priced.
4. **Caveat:** under realistic budgets ~33–53% of points are budget-infeasible (max-shed), and
   the LP minimizes downtime, so the realized selection also reflects move-cost — in the memory
   regime the cheapest evictions are many small-KV jobs, so the shed-vs-ranking correlation panel
   shows a step at `R=1` rather than a clean ℓ→memory crossover. The headline (Jaccard ≈ 0,
   Spearman ≈ 0) is robust; the correlation panel is the honest, confounded one.

---

## 3. Tests (T9 — `tests/`, 38 passing)

| # | claim | test |
|---|---|---|
| 1 | ranking invariant under `p̄` scaling | `test_power::test_ranking_invariant_under_p_bar_scaling` |
| 2 | regime switch at the `N=max` crossover | `test_power::test_regime_crossover` |
| 3 | greedy = LP away from boundaries | `test_dispatch::test_greedy_equals_lp_off_boundary` |
| 4 | every solver output satisfies all constraints | `test_dispatch::test_every_constraint_satisfied` |
| 5 | no cold job carries load | `test_instance::test_cold_idle_carry_no_load_but_keep_kv` |
| 6 | BF16↔FP8 shifts S_node & threshold, load regime unchanged | `test_power::test_precision_shifts_memory_threshold_not_load_regime` |

T8's own results are tested in `tests/test_regime.py`: rankings uncorrelated at center
(Spearman ≈ 0), the regime flip sheds a near-disjoint set (Jaccard < 0.3), and both walks bracket
`R=1` with the measured regime flipping exactly at the `N=max` crossover.

**Correction (claim #6):** the TODO says FP8 shifts `S_node` "~2×", but the real cap ratio is
`365/130 ≈ 2.81×`; the test asserts the true value.

---

## 4. Headline takeaways

- **Selection is robust; absolute shed is sensitive.** Which jobs to move is invariant to `p̄`
  scaling, `ρ*`, MFU, and the bracket ratio (T1 invariant, T7 ≥ 0.998 agreement); only the
  feasibility margin moves with them.
- **Two regimes, two value stories.** Load-bound: the autoscaler-drain upside is large
  (`30× bracket × decode skew ≈ 45×`) but only the `s_plat` floor is guaranteed (T6 left).
  Memory-bound: the saving *is* the freed memory; expected ≈ guaranteed, the bracket does not
  transfer (T6 right).
- **Coordination is worth ~2×** the shed ceiling over a decentralized greedy (T5).
- **The deadline binds only for big-KV moves** (T8/memory regime); cheap short-context moves are
  capacity-bound, not time-bound (deadline companion).
- **Crossing `R=1` reorders the dispatch onto a disjoint job set**, because the load and memory
  rankings are statistically independent (T8) — making the regime flag a real decision, not a
  relabeling.
