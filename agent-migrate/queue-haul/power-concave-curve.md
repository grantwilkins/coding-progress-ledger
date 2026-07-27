# The concave power curve

This note fixes the load scalar \(\ell\) and the power law \(P(\ell)\) that the
rest of the formulation consumes. It is the companion to `formulation.md`: that
document uses \(\ell_s=f_s/F+g_s/G\) and the marginal source-power gain
\(w_s=P(L)-P(L-\ell_s)\) as givens; here we define both from measured session
statistics and state the concavity that licenses them. Sessions are indexed by
\(j\) below, matching the paper draft; in `formulation.md` the same object is
\(s\) and the aggregate load is \(L\equiv\ell\).

We only model the accelerators serving the model (NVIDIA GPUs). CPU, memory, and
network draw are either constant loads or have small dynamic range next to the
GPUs, so "power" means GPU power throughout. GPU power oscillates strongly at the
subsecond scale during inference, but the grid only cares about the average over
an \(X\)-second window; we average over 5 s. Appendix A argues the result is
insensitive to that choice.

## What a job is

All LLM inference decomposes into three states, general enough for chat and
agentic interfaces and specific enough to correlate with observed GPU power:

- **prefill** — the prompt is tokenized and turned into KV pairs appended to the
  KV cache;
- **decode** — output tokens are produced sequentially;
- **pause** — a gap with no work for this session: a user reading or typing, or
  in an agentic loop a tool call, a shell command, or a stalled exchange.

Each state has a different average power, a different compute/memory balance, and
a different throughput. A load scalar must combine all three so that a single
number captures a session's contribution to a node's average power.

## From session statistics to a load scalar

Let a session \(j\) draw its per-turn prompt and output token counts from random
variables \(p_j\) and \(d_j\) (a *turn* is one input to the model). We take
\(p_j\perp d_j\) for convenience — not required, but it keeps the two states
separable. A workload class — from historical arrivals, or from one session when
its trace is available — also has expected turn rates \(\tau_j^{p},\tau_j^{d}\)
(turns per second) for prefill and decode; for a chat turn these coincide, and
splitting them lets agentic traffic issue prefill and decode work at different
cadences. Expected per-phase token rates over the horizon are then

\[
f_j=\tau_j^{p}\,\mathbb{E}[p_j],\qquad
g_j=\tau_j^{d}\,\mathbb{E}[d_j],
\]

the expected prefill and decode tokens per second. The pause state needs no
symbol: it is the residual time in which \(f_j\) and \(g_j\) contribute nothing,
and it enters automatically because the rates are averaged over the whole
horizon, pauses included.

## Pinning the capacities

To compare sessions we need throughput, which is context- and load-dependent:
with running contexts \(T_j^{p},T_j^{d}\), prefill and decode throughput are
\(F(T_j^{p})\) and \(G(T_j^{d})\). Appendix X shows the chosen value is only a
scaling term — the sole requirement is that it be **pinned across sessions** on a
node type. We pin it to a reference high-percentile sustained rate: \(F\) and
\(G\) are the fastest prefill and decode token rates the node holds over a 5 s
window (we use the p99.5 windowed rate). The per-session load is then the
unitless

\[
\boxed{\;\ell_j=\frac{f_j}{F}+\frac{g_j}{G}\;}
\]

Each term is the fraction of a phase's reference capacity the session's dwell
time in that state consumes, so \(\ell_j\) folds the two active regimes into one
dimensionless occupancy. For a node serving many sessions the loads add:

\[
\ell=\sum_j \ell_j .
\]

Because \(\ell\) is dimensionless and built only from a session's expected token
statistics, a single GPU has no fixed upper bound on it: \(\ell\) cannot by
itself say whether a node is saturated. It is **directional** — \(\ell=0\) is
idle, and larger \(\ell\) is more load.

Measurement caveat. On this data \(F\approx 1.6\text{–}1.9\text{k}\) prefill
tok/s and \(G\approx 1.2\text{–}1.7\text{k}\) decode tok/s (`operator_table.csv`).
Both are **workload-limited lower bounds** on capacity: every sweep used the same
ShareGPT mix at \(\le 4\) qps, so \(F,G\) are the fastest rates *observed*, not the
hardware ceiling. Within a node type this is harmless — the normalization cancels
in any ranking — and conservative in absolute watts; only cross-node-type
comparisons of \(\ell\) inherit the bias. A capacity-targeted sweep would pin them
exactly.

## The concave power law

Fitting 5 s-averaged node power against this load gives, per node type, a
saturating (Michaelis–Menten / Hill-1) law

\[
P(f,g)=P_0+\Delta P\,\frac{w}{1+w},\qquad w=a f+b g,\qquad
P_0,\Delta P,a,b\ge 0 .
\]

\(P_0\) is the idle floor, \(P_0+\Delta P\) the saturation ceiling, and
\((\Delta P\,a,\ \Delta P\,b)\) the low-load marginal power per prefill and decode
token. To read the law against \(\ell\), pin the mean phase mix so that along that
ray \(w=\kappa\,\ell\) with the node constant \(\kappa=a\,\bar f+b\,\bar g\)
(evaluated at \(\ell=1\)). Then

\[
\boxed{\;P(\ell)=P_0+\Delta P\,\frac{\kappa\,\ell}{1+\kappa\,\ell}\;}
\]

which is strictly increasing and **concave**:

\[
\frac{dP}{d\ell}=\frac{\Delta P\,\kappa}{(1+\kappa\ell)^2}>0,\qquad
\frac{d^2P}{d\ell^2}=\frac{-2\,\Delta P\,\kappa^2}{(1+\kappa\ell)^3}<0 .
\]

The fit reaches \(R^2\) 0.91–0.99 across all 25 measured node types (7 models —
Llama-3 8B/70B/405B, DeepSeek-R1-Distill 8B/70B, GPT-OSS 20B/120B — over A100 and
H100 with tensor parallelism 1–8, ~80k windows). A purely linear fit falls to
\(R^2\approx 0.24\) on the dense 70B/405B models, where power is nearly a step —
idle to near-peak by \(\ell\approx 0.1\), then flat — and rises to \(\approx 0.89\)
on the MoE/8B models, where the ramp is gradual. The mechanism is direct: more
sessions holding their separate states means more constant occupation of decode
and prefill, hence higher average power, until the node has no more headroom to
fill.

The fit is on \(w=af+bg\), a linear form on the two-vector \((f,g)\); the scalar
\(\ell=f/F+g/G\) is the pinned dispatch coordinate. Along a fixed phase mix the
two are proportional, so \(P(\ell)\) above is well-defined and concave — but it is
a projection, not a claim that power depends only on \(\ell\) off that ray.

## Marginal power and load shedding

The concave law is what makes the load term useful for curtailment. From a curve
\(P(\ell)\) the marginal power freed by removing session \(j\) is

\[
w_j=P(\ell)-P(\ell-\ell_j)\ \ge\ 0 ,
\]

and because \(P\) is nondecreasing and concave, summing these full-load marginals
is a **conservative lower bound** on the actual reduction from moving a set
\(M\):

\[
\sum_{j\in M}\big[P(\ell)-P(\ell-\ell_j)\big]
\ \le\
P(\ell)-P\!\Big(\ell-\sum_{j\in M}\ell_j\Big).
\]

Each term is evaluated at the top of the curve, where the slope is smallest; as
sessions actually leave, the operating point slides down into the steeper region
and each further removal frees *more* power than its at-full-load marginal. The
sum therefore under-promises. This is exactly the gain `formulation.md` credits
per move (\(w_s=P(L)-P(L-\ell_s)\)) and why summing candidate gains is safe before
the exact model is re-evaluated after integer selection.

Two operating points bracket the price of load. Power flattens **early** (the
power knee, \(\ell\approx 0.03\text{–}0.15\) for dense 70B+, \(0.3\text{–}0.6\)
for 8B/MoE, where \(P\) reaches \(P_0+0.8\,\Delta P\), i.e. \(\kappa\ell=4\)),
while inter-token latency departs **late** (the latency knee \(\rho^\ast\), the
highest \(\ell\) within 25% of the best median ITL — measured at
\(\ell\approx 0.5\text{–}1.0\) where the sweep reached it, else \(\rho^\ast=0.8\)
by convention). Near a setpoint \(\rho^\ast\) two prices coexist:

\[
s_{\text{plat}}=\left.\frac{dP}{d\ell}\right|_{\rho^\ast}
=\frac{\Delta P\,\kappa}{(1+\kappa\rho^\ast)^2}
\qquad(\text{fixed-node marginal, on the plateau}),
\]
\[
\bar p=\frac{P(\rho^\ast)}{\rho^\ast}
\qquad(\text{autoscaler-amortized price per node-unit}).
\]

They differ by \(\approx 3\text{–}58\times\) (\(3\text{–}5\times\) for MoE,
\(17\text{–}44\times\) for dense 70B, \(58\times\) for the 405B). Ranking uses the
additive amortized side (\(\Delta P_j=\bar p\,\ell_j\)); a certified guarantee
uses the conservative plateau side, because on a genuinely fixed node power is a
step and is not additive.

## Why two prices before one

Decode and prefill tokens are not interchangeable: a two-price linear fit
\(P=c_0+c_1 f+c_2 g\) gives \(c_1/c_2\approx 0.04\text{–}0.19\), i.e. a decode
token costs roughly \(5\text{–}25\times\) (typically \(\sim 7\text{–}10\times\))
the energy of a prefill token, stable across every configuration. Keeping the two
prices is what makes \(w=af+bg\) — and hence \(\ell\) — phase-aware. But on
ShareGPT-like traffic the two phases are \(\approx 0.8\)-correlated and decode
dominates the variance, so collapsing to the single scalar \(\ell\) costs only
\(\Delta R^2\) 0.02–0.07. The single-price \(\ell\) is therefore licensed here; a
workload where prefill and decode decouple (heavy-context agents, long tool
outputs) would need the two-vector \(w\) instead.

## Symbol table

| symbol | meaning |
|---|---|
| \(p_j,\ d_j\) | per-turn prompt / output token counts of session \(j\) (random variables) |
| \(\tau_j^{p},\ \tau_j^{d}\) | expected prefill / decode turn rate [turns/s] |
| \(f_j=\tau_j^{p}\mathbb{E}[p_j]\) | expected prefill token rate [tok/s] |
| \(g_j=\tau_j^{d}\mathbb{E}[d_j]\) | expected decode token rate [tok/s] |
| \(F,\ G\) | pinned reference prefill / decode throughput [tok/s] (p99.5 sustained) |
| \(\ell_j=f_j/F+g_j/G\) | session load (dimensionless occupancy) |
| \(\ell=\sum_j\ell_j\) | node load (\(L\) in `formulation.md`) |
| \(P_0,\ \Delta P\) | idle floor and idle-to-ceiling swing [W] |
| \(w=af+bg,\ \kappa\) | saturation argument; \(w=\kappa\ell\) along the mean mix |
| \(\rho^\ast\) | latency-knee setpoint (highest \(\ell\) within 25% of best ITL) |
| \(s_{\text{plat}},\ \bar p\) | fixed-node vs. amortized price of load [W per node-unit] |

Sources: `archive/research_artifacts/results/two_price_fit/` — `summary.csv`
(linear two-price fit), `saturating_summary.csv` (the concave law and its knees),
`operator_table.csv` (the dispatch-facing ramp model). Fit code:
`scripts/legacy/{two_price_fit,saturating_fit,operator_table}.py`.

## Deltas from the current draft

Three places where the pasted draft disagrees with the code/data — corrected
above:

1. **\(\ell_j=f_j/F+g_j/F\) → \(f_j/F+g_j/G\).** Decode must be normalized by the
   decode capacity \(G\), not \(F\). Both authoritative sources
   (`saturating_fit.py`, `formulation.md`) use \(g/G\); the draft's \(g/F\) is a
   typo.
2. **"concave curve fit of varying degree" → a specific Michaelis–Menten law**
   \(P=P_0+\Delta P\,w/(1+w)\). Naming the functional form makes the marginal,
   the knee (\(\kappa\ell=4\)), and the two prices exact rather than qualitative.
3. **Coverage.** The result files attest 7 models × {A100, H100} × TP 1–8 = 25
   node types on a single ShareGPT mix at \(\le 4\) qps — 2 GPU generations and
   varied tensor parallelism, as the draft says. The draft's "10 arrival rates"
   and "5 token distributions" are **not** attested by these files (the sweep is
   ShareGPT-only, capped at 4 qps); confirm those against the raw sweep config
   before printing them, or drop them.
