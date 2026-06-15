# Queue-Haul — Findings

Results from the dispatch experiments (T5–T9). The math is in `formulation.md`, the
parameters in `assumptions.md`, and each result is reproducible by running the matching
`plot_*.py` script (figures land in `outputs/`).

---

## The regime structure (the spine of T8)

The dispatch runs in one of two regimes, picked by `N = max(L/ρ*, S_held/S_node)`:

- **load-bound** (`L/ρ*` binds): compute is the constraint; jobs are valued by load `ℓ`
  (`dp_guaranteed = s_plat·ℓ`).
- **memory-bound** (`S_held/S_node` binds): held KV sessions are the constraint; the node
  sits at idle, and jobs are valued by context length `T` (`dp_memory = μ·T/E[T]`).

**Key simplification (derived, not assumed).** Because `generate()` sizes the population to
the pool — `n_jobs = α·N·s_node` — the memory side `S_held/s_node = α·N` is **constant**
(≈38.4 at center, `n_nodes=32`). `γ`, `cap`, and `E[T]` all cancel out of it. So the regime
reduces to a single inequality on total load:

```
memory-bound  ⟺  L < α·N·ρ*   (≈ 30.7 W-units at center)
```

Everything in T8 is a walk of `L` across that constant threshold. We report the dimensionless
**regime ratio**

```
R = (S_held/s_node) / (L/ρ*) = α·N / (L/ρ*),     crossover at R = 1.
```

`R < 1` load-bound, `R > 1` memory-bound. `R` is invariant to `n_nodes` (both `L` and the
threshold scale with it).

---

## T8 — load vs memory regime (`plot_regime_boundary.py`, `outputs/regime_boundary.png`)

We cross `R = 1` two independent ways and check they tell the same story:

- **(a) idle/cold × γ** — at a fixed short context (~13k), raise the idle/cold fraction
  (drops `L`), drawn as two γ curves {0.5, 1.0}.
- **(b) context short→long** — at center state_mix, sweep `E[T]` from ~5k (load) to ~40k
  (memory), keeping `pool.mean_context_tokens` in sync.

Each point solves the dispatch twice — forcing the load ranking and forcing the memory
ranking — under realistic link/headroom budgets, averaged over 8 seeds at `n_nodes=32` (BF16).

### What we found

| series | R range | brackets R=1 | Jaccard @ R≈1 | feasible frac |
|---|---|---|---|---|
| (a) γ=0.5 | [0.48, 6.31] | yes | 0.00 | 0.47 |
| (a) γ=1.0 | [0.34, 4.17] | yes | 0.03 | 0.33 |
| (b) context | [0.38, 2.39] | yes | 0.01 | 0.53 |

**Spearman(dp_expected, dp_memory) = +0.011 ± 0.013** (over all sweep points).

1. **Both walks bracket R=1 and agree on the boundary.** Whether you reach the crossover by
   idling jobs (a) or lengthening context (b), the regime flips at the same place, `R = 1`.
   `γ` only shifts *where along the active-fraction knob* you hit `R = 1` — not the `R = 1`
   location — which is why the γ=0.5 and γ=1.0 curves overlay when plotted against `R`. This
   is "γ delays the switch," operating through population size (`n_jobs ∝ 1+γ → higher L`).

2. **The regime flip reorders the dispatch onto a near-disjoint job set** (top panel). The
   Jaccard overlap between the load-ranked and memory-ranked shed sets is ~0 across the entire
   sweep — the two regimes shed almost completely different jobs.

3. **The two rankings are statistically independent** (bottom panel): `Spearman(dp_expected,
   dp_memory) ≈ 0`. This is the *cause* of finding (2): the generator draws context length `T`
   independently of the rate/token distributions (Δ, Y) that drive load, so the load ranking
   and the memory ranking share no information. The regime flag genuinely chooses *which jobs
   move*, not just how the same jobs are priced — a substantive result, not a relabeling.

4. **Realized selection also reflects move-cost, not ranking alone** (middle panel). The LP
   minimizes downtime, so the shed set's correlation with each ranking shifts at `R = 1` but
   does not form a clean ℓ→memory handoff; in the memory regime the cost-cheapest evictions
   are many small-KV jobs, so `corr(shed, dp_memory)` can go negative. This is honest behavior
   of a cost-minimizing dispatch under realistic budgets, not an artifact.

### Caveats

- **Realistic budgets** (Grant's choice) mean ~33–53% of sweep points are budget-infeasible
  (the solver returns a max-shed plan). The Jaccard and ranking-independence results are robust
  to this; the middle panel's correlations partly reflect what fits the links. Swapping to
  slack budgets (`SLACK_E/SLACK_M`) isolates ranking from contention if a cleaner read is
  wanted.
- The middle panel is the only confounded one; the top (Jaccard ≈ 0) and bottom (Spearman ≈ 0)
  panels carry the headline.

---

## T9 — tests (`tests/`, 38 passing)

The six headline claims each have a passing test. Five already existed; T8 added two.

| # | claim | test |
|---|---|---|
| 1 | ranking invariant under `p̄` scaling | `test_power::test_ranking_invariant_under_p_bar_scaling` |
| 2 | regime switch at the `N=max` crossover | `test_power::test_regime_crossover` |
| 3 | greedy = LP away from boundaries | `test_dispatch::test_greedy_equals_lp_off_boundary` |
| 4 | every solver output satisfies all constraints | `test_dispatch::test_every_constraint_satisfied` |
| 5 | no cold job carries load | `test_instance::test_cold_idle_carry_no_load_but_keep_kv` |
| 6 | BF16↔FP8 shifts S_node & threshold, load regime unchanged | `test_power::test_precision_shifts_memory_threshold_not_load_regime` |

T8's own results are tested in `tests/test_regime.py`: rankings uncorrelated at center
(Spearman ≈ 0), the regime flip sheds a near-disjoint set (Jaccard < 0.3), and both walks
bracket `R = 1` with the measured regime flipping exactly at the `N=max` crossover.

**Correction worth noting (claim #6):** the TODO says FP8 shifts `S_node` "~2×", but the real
KV-capacity ratio is `CAP_FP8/CAP_BF16 = 365/130 ≈ 2.81×`. The test asserts the true 2.8×.

---

## Prior experiments (recap)

- **T5** (`plot_dispatch_validation.py`) — random < greedy < LP by shed ceiling; greedy lies on
  the LP cost frontier where feasible; the greedy↔LP gap is the value of central coordination.
- **T6** (`plot_certify_report_validation.py`) — every `S*` certified feasible under guaranteed
  prices (`s_plat`) is met under amortized prices (`p̄`); the guaranteed/expected gap tracks the
  bracket ratio (~30× dense center).
- **T7** (`plot_sensitivity_sweeps.py`) — job ranking flat across `ρ*`, MFU, and bracket-ratio
  sweeps; feasibility margin moves monotonically. Selection robust, absolute shed sensitive.
