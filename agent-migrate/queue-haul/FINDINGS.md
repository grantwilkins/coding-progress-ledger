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
node_knee.py →  optional source-placement exploration for expected ramp–plateau node shed
```

### T1 — prices (`power.py`, `outputs/power_validation.png`)

The node never appears as a curve in the solver; it appears as four prices. At center:

- amortized price **`p̄ = P_busy/ρ* = 10.5 kW`** per node-unit of load (the autoscaler-drain
  upside),
- guaranteed plateau slope **`s_plat = p̄/30 = 350 W`** per node-unit (realized even if no node
  drains) — the **bracket ratio is 30×**,
- memory pressure diagnostic **`μ = P_idle/S_node = 208 W/held session` (BF16)**, `74 W (FP8)`;
  this is no longer a dispatch certificate.

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
- **B — future-impact reporting separates base load from token work.** The code reports
  `P_idle/ρ*·ℓ + c1·f + c2·g`, with `p̄·ℓ` retained only as the single-price comparison.
- **C — the memory score spreads widely** around `μ=208 W` (CoV **1.35**, tail-heavy context).
  The annotated load↔memory rank Spearman is measured from the synthetic draw; it is not assumed.

### T4 — solver (`dispatch.py`)

LP over `y∈[0,1]` minimizing total downtime `Σ y·c` s.t. `Σ y·ΔP ≥ S*`, source egress,
destination rebuild by the deadline, and destination headroom. **Two solves, not a branch:** on
infeasibility it re-solves to max-shed and reports the shortfall. `bind_dp` commits the
active-work certificate — `s_plat·ell + c_prefill·f + c_decode·g` in every regime. `dp_memory`
is kept as a memory-pressure diagnostic and destination/source-capacity signal, not as certified
grid watts. Validated end-to-end in T5.

---

## 2. Experiments (T5–T8)

### T5 — random vs integer greedy vs LP (`outputs/dispatch_validation.png`)

Two policies under the same movement budgets, isolated by active, cache-resident session
class. The y-axis is **disruption intensity**: aggregate movement downtime divided by the
requested certified shed (`s/kW`). This is the interpretable version of the LP objective:
how many session-disruption seconds we spend per kW delivered.

The baseline is now true integer first-fit: it never splits the marginal job and can overshoot
`S*` by one job. The validation script prints, for each panel, active resource rows, row duals,
fractional LP variables, max job granularity (`max ΔP_j/S*`), max row granularity
(`max_j a_{rj}/b_r`), and Spearman correlations between `ΔP` and the cost/resource columns.
At the current class-isolated maxima the reported LP cuts are mostly granularity gaps:
active rows are `none` and `frac_vars=1`.

### Companion — source-size sweep (`outputs/dispatch_scale.png`)

The 4-node agentic fixture is small, so this companion scales the **source** from **4 to
128 nodes** and measures the best LP cut in disruption intensity over a common feasible target
sweep. It shows two different stories:

- **Fixed destination (`48` destination nodes, `W=16`)**: the event ceiling saturates and the
  LP cut washes out as the source grows. This is not evidence that coordination is useless; it
  says this fixed destination is the bottleneck.
- **Scaled destination (same ratio as the 4-node fixture: `dest_nodes=12N`, `W=4N`)**:
  the ceiling grows with the source, and the LP still cuts disruption intensity once the common
  feasible target is high enough for resource choices to matter.

### Companion — DeepSeek-V4-Flash proxy (`outputs/dispatch_validation_deepseek_v4_flash.png`, `outputs/dispatch_expected_deepseek_v4_flash.png`)

DeepSeek-V4-Pro does not fit the current single-node model abstraction: 1.6T total parameters
need model-parallel weight placement before the source/destination node accounting is meaningful.
The rerun therefore uses **DeepSeek-V4-Flash** as the single-node-compatible V4 proxy:
**284B total / 13B active parameters**, **1M context**, compressed-attention KV, and a conservative
FP8-sized weight footprint.

Under the same **4 held-memory-node** event, the smaller KV footprint makes the held population much
larger (agentic: **2001 sessions**). If every agentic session is active, that population is **33.0
serving-node equivalents**, so this is not a 4-serving-node compute slice. It flips to **load-bound**.
The LP still improves the normalized objective, but the exact magnitude depends on the active-work
certificate and should be read from the regenerated script output.

The certified grid floor is the active-work certificate, not the future-node proxy. The additive
future field should be treated as a reported proxy, not the dispatch certificate.

### T6 — certify active work, report high (`outputs/certify_report_validation.png`)

Read two prices off each plan: the active-work certificate vs the future node-drain upside.
Budgets slack to isolate the price story.

- **Compute-bound pool:** the certificate is active serving work on the source; the old **30×
  bracket** remains a single-price reference, not the token-energy estimate.
- **Memory-bound pool:** held KV is a capacity constraint and future node-drain opportunity, not a
  certificate. Idle/cold-only memory relief now contributes **0 certified watts**.
- **Containment holds:** every feasible `S*` under the active-work certificate is met under the
  future estimate (PASS).

### T7 — sensitivity sweeps (`outputs/sensitivity_sweeps.png`)

Sweep `ρ*`, MFU, and the bracket ratio on a load-bound pool (where all three actually enter the
score). **Two-part result: selection is robust, absolute shed is sensitive.**

- **Which jobs to move barely changes** — ordering agreement with baseline **≥ 0.998** across
  every sweep, by both orderings (power-freed and downtime-per-watt).
- **How much you can cut moves smoothly** — after switching to the active-work certificate, the
  center short-context load-bound fixture reports a much larger certified ceiling and weaker
  sensitivity to utilization/MFU/price-ratio than the old plateau-only floor. Read the current
  values from `plot_sensitivity_sweeps.py`.

### Companion — deadline sweep by class (`outputs/deadline_sweep.png`)

The deadline changes the coordination story mainly when movement resources bind. Below the ~5 s
migration startup floor, no move completes and the reduction is zero. Current kW ceilings should
be read from the regenerated script output because the certificate is now active-work power, not
memory occupancy.

### T8 — load vs memory regime diagnostic (`outputs/regime_boundary.png`)

The structural payoff. The regime reduces to one inequality on total load:

```
memory-bound  ⟺  L < occupancy·N·ρ*    (because S_held/s_node = occupancy·N is constant)
R = (S_held/s_node)/(L/ρ*) = occupancy·N/(L/ρ*),   crossover at R = 1.
```

Cross `R=1` two independent ways — **(a)** raise idle/cold fraction (× two γ curves) at a fixed
~13k context, **(b)** lengthen context `E[T]` 5k→40k at center state-mix — and check they agree.

| series | R range | brackets R=1 | corr_cert @ R≈1 | corr_mem @ R≈1 |
|---|---|---|---|---|
| (a) γ=0.5 | [0.58, 6.68] | yes | +0.41 | -0.49 |
| (a) γ=1.0 | [0.44, 5.46] | yes | +0.41 | -0.49 |
| (b) context | [0.50, 2.71] | yes | +0.42 | -0.48 |

**Spearman(certificate, dp_memory) = +0.022 ± 0.013** on this synthetic draw.

1. **Both walks bracket and agree at R=1** — the boundary is a property of `N`, reachable by
   idling jobs or by growing KV; `γ` only shifts *where along the active-fraction knob* you hit
   `R=1` (it delays the switch via population size, `n_jobs ∝ 1+γ`), not the `R=1` location.
2. **The certified dispatch remains active-work driven across the boundary.** Near `R≈1`, the
   shed set correlates positively with the active-work certificate and negatively with memory
   pressure.
3. **The load and memory rankings differ enough to be dangerous as a certificate** — the active
   certificate and `dp_memory` are only weakly related in this draw. The regime flag is therefore
   diagnostic and should not choose *which jobs move*.

---

## 3. Tests (T9 — `tests/`)

| # | claim | test |
|---|---|---|
| 1 | ranking invariant under `p̄` scaling | `test_power::test_ranking_invariant_under_p_bar_scaling` |
| 2 | regime switch at the `N=max` crossover | `test_power::test_regime_crossover` |
| 3 | integer greedy is whole-job and LP lower-bounds it | `test_dispatch::test_integer_greedy_overshoots_off_boundary_and_lp_lower_bounds` |
| 4 | every solver output satisfies all constraints | `test_dispatch::test_every_constraint_satisfied` |
| 5 | no cold job carries load | `test_instance::test_cold_idle_carry_no_load_but_keep_kv` |
| 6 | BF16↔FP8 shifts S_node & threshold, load regime unchanged | `test_power::test_precision_shifts_memory_threshold_not_load_regime` |
| 7 | long sampled turns cap effective turn rate instead of exceeding one-session occupation | `test_instance::test_long_turns_cap_effective_turn_rate` |

T8's own results are tested in `tests/test_regime.py`: the load and memory rankings are only
weakly related in the synthetic draw, the regime flag no longer changes certified dispatch, and
both walks bracket `R=1` with the measured regime flipping exactly at the `N=max` crossover.

**Correction (claim #6):** an earlier note said FP8 shifts `S_node` "~2×", but the real cap ratio is
`365/130 ≈ 2.81×`; the test asserts the true value.

`tests/test_node_knee.py` covers the separate node-knee exploration path: explicit source-node
placement, convex removed-load value under the ramp–plateau curve, active-knee concentration cases,
node-drain greedy, and the tiny exact oracle. This path reports modeled `node_expected_w`, not a hard
grid guarantee.

`plot_node_knee_deadline_sweep.py` writes `outputs/node_knee_deadline_sweep.{pdf,png}`. On the
current active-agentic fixture, the old additive LP never reaches the modeled node-expected target.
Active-knee LP reaches it at the shortest tested post-startup deadline (`10 s`) with the lowest
disruption intensity (`24.5 s/kW`). Random jobs are much worse (`16 s`, `59.2 s/kW`), while random
nodes behave close to node-drain greedy (`12 s`, `40.3 s/kW`) because they still concentrate removal
on one source node.

`plot_node_knee_scale_workload_sweep.py` writes
`outputs/node_knee_scale_workload_sweep.{csv,pdf,png}`. It sweeps `1/2/4` source nodes, all four
active cached workload classes, deadlines `10/30/120 s`, and target fractions `25/45/65%` of full
modeled node-expected removable power. Current median disruption intensities show the same pattern:
random jobs are expensive (`27.6/28.6/39.6/45.6 s/kW` by class), random nodes track node-drain, and
the additive LP misses every agentic node-expected target.

---

## 4. Headline takeaways

- **Selection is robust; absolute shed is sensitive.** Which jobs to move is invariant to `p̄`
  scaling, `ρ*`, MFU, and the bracket ratio (T1 invariant, T7 ≥ 0.998 agreement); only the
  feasibility margin moves with them.
- **Memory is a constraint, not watts.** Load-bound and memory-bound pools now certify the same
  active-work column. Memory pressure still limits admission/held capacity and is reported as a
  future node-drain diagnostic, but it no longer creates smooth per-session grid watts.
- **Coordination shows up only at tight shared-resource boundaries.** The class-isolated
  dispatch plot now reports active rows and granularity diagnostics so a fractional LP win is not
  mistaken for a resource-coupling win.
- **The deadline binds only for big-KV moves** (T8/memory regime); cheap short-context moves are
  capacity-bound, not time-bound (deadline companion).
- **Crossing `R=1` reorders the dispatch onto a different job set** in this synthetic draw,
  making the regime flag a real decision, not a relabeling.
