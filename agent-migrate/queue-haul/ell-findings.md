# ℓ and power-pricing — findings

Why the job-load (ℓ) / power model was audited, what the evidence says, and the one change it
forces. Companion to `formulation.md`. Source of truth for the measurements: `~/powertrace-sim/`
(measured A100/H100 vLLM traces) and [Wilkins et al. 2026, arXiv:2603.18383](https://arxiv.org/abs/2603.18383).

## TL;DR

1. **Power-as-roofline-in-utilization is correct** (ramp from `P_idle` to `P_busy`, knee, then flat) — and now *measured*, not assumed.
2. **The two-price direction was right**: decode is the power-dense phase, by the *average-power* argument (prefill = higher instantaneous power but brief; decode = lower but lasts the whole generation → more integrated energy).
3. **The parameterization was wrong**: the expected shed used a made-up split `p̄_dec = 5·p̄_pre` derived from the amortized price `p̄`, not from measured energies. **Fix: price the expected shed by the measured per-token energies, `ΔP_exp = c₁·f + c₂·g`.**

## The question

`ℓ_j = f_j/ρ(T_j) + g_j/G` feeds two power prices. Two things needed checking against reality:
- **(A)** Is node power really a ramp-then-plateau *roofline in utilization*?
- **(B)** Is the two-price split (`p̄_dec ≈ 5·p̄_pre`, decode frees ~5× a prefill unit) right in *direction* and *magnitude*?

## How it was found

**Literature.** POLCA (arXiv:2308.12908): inference uses ≤80% of provisioned power → confirms `P_busy = 0.8·TDP`. TokenPowerBench (arXiv:2512.03024): energy-per-token plateaus beyond batch 256 → saturating. Multiple phase studies (arXiv:2511.05597, 2605.11999) and **our own paper** (arXiv:2603.18383 §2.1): **prefill 80–90% TDP, decode 40–60% TDP** — prefill is the higher *instantaneous*-power phase.

**powertrace-sim (our measured traces).** A subagent mined the repo. It fits three nested power models to ~80k five-second windows over 25 node types:
- two-price linear `P = c₀ + c₁·f + c₂·g` — `scripts/eval/two_price_fit.py:224`
- saturating (Michaelis–Menten) `P = P₀ + dP·w/(1+w)`, `w = a·f + b·g` — `scripts/eval/saturating_fit.py:54`
- operator ramp-plateau `P = P_idle + (P_busy−P_idle)·min(ℓ/k, 1)` — `scripts/eval/operator_table.py:64`

Key tables: `results/two_price_fit/{summary,saturating_summary,operator_table}.csv`, `results/two_price_fit/RESULTS.md`, `feature-test/results/attribution_*.csv`, `results/rps_sweep*/rps_vs_avg_power.csv`.

## What the data confirms

**(A) Roofline — yes.** RESULTS.md: *"power is nearly a step (idle → ~peak by ℓ ≈ 0.1, then flat)."*
- Saturating fit **R² 0.91–0.99**; a linear fit collapses to **0.25** on dense 70B+ → power is concave/saturating.
- RPS sweeps flat from ~2 to 64 req/s; idle/busy ≈ **3.2–4.4×**.
- **The bracket ratio and knee fall out of the fits** (`saturating_summary.csv`):

  | node | `amort_over_plat` (= `p̄/s_plat`) | `ell_power_knee` |
  |---|---:|---:|
  | llama-3-405b H100 TP8 | **58.5** | 0.09 |
  | llama-3-70b A100 TP8 | **30.1** | 0.15 |
  | llama-3-70b H100 TP8 | **17.2** | 0.29 |
  | gpt-oss-120b A100 (MoE) | **4.1–5.0** | gradual (≳0.3) |

  This *is* `assumptions.md`'s swept bracket [17, 58] and "~3–5× MoE", and the ℓ≈0.10 knee — empirical, not assumed. **Caveat: ℓ≈0.10 is a dense-70B+ number; MoE (our Qwen3-235B target) saturates later.**

**(B) Two-price direction — decode is power-dense.** Measured three independent ways:
- per-token energy `c₂/c₁` ≈ **9.4×** median (5.2–25), `two_price_fit.py`
- per-busy-second `p_dec/p_pre` ≈ **8×** median (3.6–17.7), `saturating_fit.py:149`
- feature-zeroing energy attribution: decode **33–77%** of dynamic power vs prefill **1–4%**, `feature-test/results/attribution_*.csv`

The reconciliation with "prefill 80–90% TDP": prefill draws more *instantaneously* but is **brief** (compute-bound, thousands tok/s); decode draws less but runs the **whole generation** (memory-bound, ~hundreds–thousands tok/s). **Integrated per session, decode wins** — `c₂ ≫ c₁`.

## What was wrong (and what wasn't)

**Wrong — the magnitude/calibration of the split.** `power.py:58-64` sets `p_pre = 2p̄/(1+r)`, `p_dec = r·p_pre` with `r = 5`, i.e. the two prices are a *made-up split of the amortized price `p̄`*, not the measured energies. The consistent value implied by the data, `r = (c₂/c₁)·(G/ρ) ≈ 2–5` and **context-dependent**, is not a flat 5.

**Not wrong — the direction, and the `1/ρ(T)` prefill shape.** The old prefill term `p_pre·(f/ρ(T))` carries a `1/ρ(T)` factor that is actually *physical* (prefill energy/token grows with context as attention grows). The problem was never the shape; it was that both prices were ungrounded.

**The fix (chosen):** price the expected shed directly by the measured per-token energies — the model powertrace-sim already validates at R² 0.83–0.99:

$$\Delta P_j^{\text{exp}} = c_1\, f_j + c_2\, g_j$$

`ℓ` keeps `ρ(T)` and `G` for **capacity** (load/held headroom); the **power** price moves to `c₁, c₂`. The single amortized `p̄` is the phase-*average*; `(c₁, c₂)` is the phase-*resolved* split — they differ because `c₂/c₁ ≠ ρ/G`.

## The change — fields & code

Spec (`formulation.md`) is done (commit `a86f051`). Code still lags:

| file | now | change |
|---|---|---|
| `power.py:45,58-64` | `phase_ratio=5`, props `p_pre`, `p_dec` (from `p̄`) | replace with fields **`c1_j_per_prefill_tok`**, **`c2_j_per_decode_tok`** (measured J/tok); drop `phase_ratio`, `p_pre`, `p_dec` |
| `instance.py:47-49,84-86` | stores `ell_pre=f/ρ(T)`, `ell_dec=g/G` | also expose raw **`f`** (`rate·δ·active`) and **`g`** (`rate·Y·active`) so the price has `f, g` directly |
| `impact.py:53` | `dp_expected = p_pre·ell_pre + p_dec·ell_dec` | **`dp_expected = pool.c1·pop.f + pool.c2·pop.g`** |
| `tests` (T9) | two-price-vs-single-price phase-skew test | re-anchor to `c₁·f + c₂·g`; assert `c₂/c₁` sets the tilt |

Unchanged: `dp_guaranteed = s_plat·ℓ`, `dp_memory = μ·T/E[T]`, all capacity constraints, the regime test.

## Open decision — `c₁, c₂` for the target node

Qwen3-235B-A22B (MoE) is **not in the measured set**. Nearest analogs (`summary.csv`):

| node | `c₁` (J/prefill-tok) | `c₂` (J/decode-tok) | `c₂/c₁` |
|---|---:|---:|---:|
| gpt-oss-120b A100 TP8 (MoE) | 0.062 | 0.87 | 14.1 |
| llama-3-70b H100 TP8 (dense) | 0.148 | 1.76 | 11.9 |
| llama-3-405b H100 TP8 (dense) | 0.290 | 1.52 | 5.3 |

Options: (a) default to the **gpt-oss-120B MoE** analog as a labeled, sweepable placeholder; (b) pin to a chosen dense node; (c) hold the code until `c₁, c₂` are fit on a real Qwen3 trace. The *ranking* depends only on `c₂/c₁` (≈ 5–14); absolute shed scales with `c₁`.
