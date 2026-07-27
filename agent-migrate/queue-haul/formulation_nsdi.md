# Queue-Haul: the NSDI formulation

This document is the self-contained mathematical statement of the multi-session
migration problem, written to be lifted directly into the paper. It continues
the two sections already drafted — §\ref{sec:marginal-power}, which prices what
a session's departure is worth at the source, and §\ref{sec:single-session},
which prices what one session's arrival costs at a destination — and closes the
loop by coupling many sessions across shared limits.

It covers two settings:

- **§A. Requirement frontier.** One source power domain jointly chooses replay
  and KV actions against a pinned destination class and one logical WAN route;
  no concrete destination inventory is required.
- **§B. Optional concrete comparison.** A supplied destination may compare its
  residual vector against the frontier and add replica packing or multiple
  sites later.

Both are written as one linear program and solved greedily. The point of the
document is that this is not a heuristic compromise: the LP has a structure —
a session-partition matrix plus a small fixed number of dense resource rows —
that bounds how much integrality can cost, and makes a one-shot approximation
of the LP duals a principled scoring rule rather than a guess. §C states and
proves those structural results. §D records every constant, its provenance, its
error, and its domain of validity, so that a reviewer can check each coefficient
against a measurement. §E states what the formulation does not model.

Provenance convention used throughout: **measured** (an instrument produced it),
**fitted** (a model was fit to measurements, with a held-out error), **assumed**
(an operator input or a sensitivity variable with no measurement behind it), and
**target** (semantics the formulation specifies but the current implementation
does not yet enforce). Any result that depends on an *assumed* quantity is
labelled `sensitivity` and never reported as an admission guarantee.

---

## 0. Notation, and the bridge to the earlier sections

The earlier sections fix most of the symbols. This document adds only what is
needed for the multi-session problem, and reuses the existing symbols exactly.

| Symbol | Meaning | First defined |
|---|---|---|
| \(j\in\mathcal J\) | session | §\ref{sec:marginal-power} |
| \(i\in\mathcal I\) | source serving instance (one model replica, \(\ge1\) GPUs) | §\ref{sec:marginal-power} |
| \(f_j,g_j\) | expected prefill and decode token rates (tok/s) | §\ref{sec:marginal-power} |
| \(F,G\) | reference prefill/decode throughputs for the node type (tok/s) | §\ref{sec:marginal-power} |
| \(\ell_j=f_j/F+g_j/G\) | dimensionless session load | §\ref{sec:marginal-power} |
| \(L_i=\sum_{j\in\mathcal S_i}\ell_j\) | node load | §\ref{sec:marginal-power} |
| \(P_i^{\mathrm{on}},P_i^{\mathrm{idle}},P_i^{\mathrm{off}},\Delta P_i\) | node power curve and its dynamic range | §\ref{sec:marginal-power} |
| \(\delta_{ij}(L_i)\) | source-side marginal power of removing \(j\) from \(i\) | §\ref{sec:marginal-power} |
| \(m\) | model | §\ref{sec:single-session} |
| \(k\in\mathcal K\) | destination replica | §\ref{sec:single-session} |
| \(a\in\{R,K\}\) | migration method: context replay or KV transfer | §\ref{sec:single-session} |
| \(T_j,C_j,K_{jk}\) | context length, serialized context bytes, missing KV bytes | §\ref{sec:single-session} |
| \(B,\tau^{\mathrm{RTT}},\rho_{km}(T),\mu_{km},\eta_m\) | effective WAN rate, one fixed route RTT, prefill rate, KV-ingest rate, KV bytes/token | §\ref{sec:single-session} |
| \(t^{R}_{ijk},t^{K}_{ijk}\) | isolated migration times | §\ref{sec:single-session} |
| \(H\) | hold period at the destination | §\ref{sec:single-session} |

New in this document:

| Symbol | Meaning |
|---|---|
| \(n\in\mathcal N\) | destination **site** (a facility; contains replicas) |
| \(q\) | **pinned replica type** — model revision, tokenizer, KV ABI and dtype, hardware, precision, parallel layout, engine and scheduler configuration |
| \(\mathcal P_{n,q}\subseteq\mathcal K\) | **pool**: the replicas of type \(q\) at site \(n\) |
| \(c=(j,a)\) | requirement-frontier **candidate** — the atomic decision |
| \(x_c\in\{0,1\}\), \(y_j=\sum_a x_{(j,a)}\) | selection variables |
| \(w_c\) | source power gain of \(c\) (W) |
| \(t_c\) | migration duration of \(c\) (s); \(t_{(j,a,k)}=t^{a}_{i(j)jk}\) |
| \(b_c\) | bytes candidate \(c\) puts on the logical WAN route |
| \(v_c=(v^{p}_c,v^{d}_c)\) | destination service work of \(c\) (service-seconds per wall-second, prefill and decode coordinates) |
| \(\kappa_c\) | destination live-KV demand of \(c\), in blocks |
| \(\Theta\) | requested source power reduction (W) |
| \(H^{\mathrm{mig}},H^{\mathrm{res}}\) | migration horizon and residency horizon (s); \(H^{\mathrm{res}}\) is the hold period \(H\) of §\ref{sec:single-session} |
| \(r\in\mathcal R\), \(u_{r,c}\) | resource row, and \(c\)'s normalized consumption of it |
| \(\lambda_j\ge0,\;y_r\ge0\) | LP duals of the session rows and resource rows |
| \(\hat y_r\) | the price estimate greedy uses in place of \(y_r\) |

Two collisions are resolved deliberately. \(\Delta P_i\) remains the *node's*
fitted dynamic range, so the requested shed is written \(\Theta\). \(\mathcal K_q\)
denotes allocatable KV **capacity in blocks** and is distinct from \(K_{jk}\),
the KV **bytes** a migration must move.

### 0.1 The fitted power curve, exactly

§\ref{sec:marginal-power} writes \(P^{\mathrm{on}}_i(L)=P^{\mathrm{idle}}_i+\Delta P_i\,L/(1+L)\).
The measured fit is a Michaelis–Menten law in the raw rates,

\[
P(f,g)=P_0+\Delta P\,\frac{\alpha f+\beta g}{1+\alpha f+\beta g},
\]

projected onto the mean workload ray as \(P(\ell)=P_0+\Delta P\,\kappa\ell/(1+\kappa\ell)\)
with \(\kappa=\alpha\bar f+\beta\bar g\). The paper's \(L\) is therefore
\(\ell\) rescaled by \(\kappa\): the two forms agree under
\(L=\kappa\ell\), equivalently under reference throughputs \(F/\kappa,G/\kappa\).
For the reference type (GPT-OSS-20B, A100 80 GB, BF16, TP=1) the fitted values
are \(P_0=67.1204\) W, \(P_0+\Delta P=424.446\) W, \(\kappa=1.94858\),
\(F=1448.32\) tok/s, \(G=1260.38\) tok/s, at a 5 s averaging window
(\(R^2=0.9901\), 3,725 windows, 31 runs). The profile truncates the curve at
\(\ell_{\max}=0.531358\), where \(P=248.890\) W; the asymptote is never reached
inside the supported domain. **This must be stated in the paper**, because the
node's usable dynamic range is \(248.890-67.120=181.77\) W, not \(\Delta P\).

The law is not specific to this type: it holds at \(R^2\in[0.911,0.990]\) across
25 measured node types (7 model families \(\times\) \{A100, H100\} \(\times\)
TP 1–8, 82,046 five-second windows, 680 runs). A two-price linear decomposition
on the same data gives decode tokens a per-token energy cost
\(5.25\)–\(25.24\times\) that of prefill tokens, stable across every
configuration. Both fits are currently **in-sample**; §D.5 records this as an
open validity gap.

---

## §A. Many sessions, one destination class

### A.1 What the event is

At \(t=0\) a source power domain is told to reduce its measured GPU power from
\(P^{0}_{\mathrm{src}}\) to at most \(P_{\mathrm{lim}}\) by deadline \(D\). The
required reduction is \(\Theta=P^{0}_{\mathrm{src}}-P_{\mathrm{lim}}\). Two
horizons follow, and they must not be conflated:

\[
H^{\mathrm{mig}}=D-C-W,\qquad H^{\mathrm{res}}\ \ge\ H^{\mathrm{mig}},
\]

where \(C\) is controller delay and \(W\) is the trailing window over which the
grid measures power. Every migration must *commit* inside \(H^{\mathrm{mig}}\);
every landed session occupies destination memory and service capacity for
\(H^{\mathrm{res}}\), the hold period \(H\) of §\ref{sec:single-session}. Sizing
KV residency at \(H^{\mathrm{mig}}\) — as a short deadline invites — understates
the destination's obligation by the ratio of the two horizons.

Planning is nonanticipative. It never reads sampled future request times or
sizes; it materializes expected state conservatively as

\[
\widehat T_j(h)=\left\lceil T_j+\gamma_j h\right\rceil,
\qquad
\widehat C_j(h)=\left\lceil C_j\,\frac{\widehat T_j(h)}{T_j}\right\rceil,
\]

with \(\gamma_j\) the expected context growth rate (tok/s). Sampled requests are
consumed only by execution and evaluation.

### A.2 The candidate is the atomic decision

A requirement-frontier **candidate** is \(c=(j,a)\): move session \(j\) by
replay or KV transfer on the pinned destination class. Both actions enter one
candidate table, so a plan may use both methods across sessions while selecting
at most one action per session. It exists only if the eligibility predicate
\(E_c\) holds.
Eligibility is a conjunction of exact matches, not a score:

- replay requires equal model revision, tokenizer, and durable-log execution
  contract;
- KV transfer additionally requires an exact KV ABI, layout, and dtype match;
- the replica must be warm and healthy, and its type must have a measured
  profile covering this session's context length, workload direction, and the
  route's bandwidth;
- the pool must not have disabled the method.

Destination hardware need **not** equal source hardware. Heterogeneous replay is
eligible whenever the destination type carries its own measured profile.
Eligibility is a boolean predicate over pinned identities; it is the one part of
the formulation that involves no measured constant at all.

Each candidate carries five quantities, and nothing else enters the optimization:

\[
c\ \longmapsto\ \bigl(w_c,\; t_c,\; b_c,\; v_c,\; \kappa_c\bigr).
\]

**Source power gain** \(w_c=\delta_{i(j)j}(L_i)\), the marginal of
§\ref{sec:marginal-power}, evaluated at the node's *initial* load. When \(j\) is
the last session on its node and the node can reach a lower power state before
\(D\), \(w_c\) additionally carries the transition credit
\(P^{\mathrm{idle}}_i-P^{\mathrm{off}}_i\). \(w_c\) depends on \(j\) only, not on
\(a\) — the relief is entirely a source-side quantity.

**Duration** \(t_c=t^{a}_{i(j)jk}\), exactly the isolated migration time of
§\ref{sec:single-session}, evaluated at \(\widehat T_j(H^{\mathrm{mig}})\):

\[
t_{(j,R,k)}
=\frac{\widehat C_j}{B}+\tau^{\mathrm{RTT}}
+\alpha_{R,q}\!\left[\frac{\widehat T_j}{\rho_{km}(\widehat T_j)}+\tau^{\mathrm{cmp}}_{q}(1+\chi_j)\right]
+\tau^{\mathrm{sw}}_q,
\]

\[
t_{(j,K,k)}
=\max\!\left\{\frac{K_{jk}}{B},\ \frac{K_{jk}}{\mu_{km}}\right\}
+\tau^{\mathrm{RTT}}
+\tau^{\mathrm{res}}_{q}+c^{\mathrm{catch}}_{q}(\widehat T_j)
+\tau^{\mathrm{sw}}_q,
\]

where \(\chi_j=\mathbb 1[\gamma_j>0]\) counts one catch-up round,
\(\alpha_{R,q}\) is the replay compute/completion calibration, and
\(\tau^{\mathrm{res}}_q\) is the fitted KV residual. Both fits are held out to a
different context *and* a different bandwidth and never underpredict (§D.3).
The \(\max\) in the KV expression is the ingest floor of
§\ref{sec:single-session}; commit `ef435092` restored it in the pool
implementation. `route_rtt_s` contributes exactly one fixed P50 RTT per action,
with no RTT/2 conversion and no TCP model.

**Bytes on the wire.** \(b_c=\widehat C_j\) for replay and
\(b_c=K_{jk}=\lfloor\widehat T_j/L^{\mathrm{tx}}_q\rfloor\,\eta_m L^{\mathrm{tx}}_q\)
for KV transfer (only *sealed* blocks are copied; an unsealed tail is
reconstructed, never shipped).

**Destination service work.** The two-dimensional coordinate
\(v_c=\bigl(f_j/F_q(\widehat T_j),\ g_j/G_q(\widehat T_j)\bigr)\).
This is deliberately *not* collapsed to a scalar the way \(\ell_j\) is at the
source, because prefill and decode consume different destination resources and
the two coordinates move independently across hardware. The target semantics
integrate over the horizon,

\[
v^{p}_{c}=\frac1{H^{\mathrm{res}}}\!\!\sum_{\text{req}\in H^{\mathrm{res}}}\!\!
\tau^{P}_q\bigl(T^{\mathrm{full}},T^{\mathrm{hit}},T^{\mathrm{miss}}\bigr),
\qquad
v^{d}_{c}=\frac1{H^{\mathrm{res}}}\int_0^{H^{\mathrm{res}}}\!\frac{g_j(t)}{G_q(T_j(t))}\,dt,
\]

and the implemented \(f_j/F_q(\widehat T_j)\) is an append-token/cold-rate
normalization coordinate, not a conservative physical bound on cached prefill:
an uncached suffix still attends to the cached prefix, so hit tokens cannot
simply be subtracted from a cold-prefill curve. Prefix reuse does not reduce
generation-phase work at all.

**Live KV.** \(\kappa_c=\bigl\lceil \widehat T_j(H^{\mathrm{res}})/L_q\bigr\rceil\)
blocks, where \(L_q\) is the engine's paging block size. Rounding is per
session, independently — v1 gives **no** cross-session prefix-sharing credit.
This overstates physical memory when unrelated sessions share exact prefixes;
the consequence is uncredited headroom and possible false-negative admission,
never a false positive. Note that \(L_q\) (the vLLM page, 16 tokens) and
\(L^{\mathrm{tx}}_q\) (the LMCache transfer block, 256 tokens) are different
quantities and are separately measured.

### A.3 The five consumable resources

For a pinned serving class inside its measured envelope, exactly five
quantities decide whether an already-warm pool can land a set of sessions.
Compatibility, warmness, and hardware identity are eligibility predicates, not
consumables; GPU count, FLOPs, occupancy, HBM bandwidth, and scheduler policy
are absorbed into the measured rates and are not portable capacity constants.

| # | Row \(r\) | One row per | Consumption \(u_{r,c}\) (unnormalized) | Capacity | Horizon |
|---|---|---|---|---|---|
| 1 | source stream | source instance \(i\) | \(t_c\,\mathbb 1[i(c)=i]\) | \(S_i\,H^{\mathrm{mig}}\) | \(H^{\mathrm{mig}}\) |
| 2 | logical WAN route | event | \(b_c\) | \(B\,H^{\mathrm{mig}}\) | \(H^{\mathrm{mig}}\) |
| 3 | service facet | pool \(\mathcal P\), facet \(\iota\) | \(N_\iota\!\cdot\! v_c\,\mathbb 1[k(c)\in\mathcal P]\) | \(\lvert\mathcal P\rvert h^{\mathrm{mode}}_\iota-N_\iota\!\cdot\!\!\sum_{k\in\mathcal P} b_k\) | \(H^{\mathrm{res}}\) |
| 4 | live KV | pool \(\mathcal P\) | \(\kappa_c\,\mathbb 1[k(c)\in\mathcal P]\) | \(\lvert\mathcal P\rvert\mathcal K_q-\sum_{k\in\mathcal P}\kappa^0_k\) | \(H^{\mathrm{res}}\) |
| 5 | migration slot | pool \(\mathcal P\) | \(t_c\,\mathbb 1[k(c)\in\mathcal P]\) | \(\lvert\mathcal P\rvert H^{\mathrm{mig}}\) | \(H^{\mathrm{mig}}\) |

Rows 1 and 5 are *time* budgets: a source instance can sustain \(S_i\)
concurrent outbound streams, and a destination replica can absorb one migration
at a time. Source streams are indivisible, so selected durations are packed
into \(S_i\) bins of size \(H^{\mathrm{mig}}\) per source instance. Row 2 is a
*byte* budget: a fluid relaxation of the transfer
schedule over one logical WAN route. Its central rate is 5 Gbps; sensitivity
uses 1/5/10 Gbps and P50 RTT classes 10/60/90/150/240 ms. Rows 3 and 4 are
*occupancy* budgets that persist after the event.

The service facet deserves a precise statement. For type \(q\), the admissible
region is a polyhedron in the two-dimensional work coordinate,

\[
\mathcal C^{\mathrm{mode}}_{q}=\bigl\{v\ge0:\ N_q v\le h^{\mathrm{mode}}_q\bigr\},
\qquad
h^{\mathrm{norm}}_q\le h^{\mathrm{emg}}_q\le h^{\mathrm{stab}}_q,
\]

with common nonnegative normals so the three policy envelopes are nested.
`normal` and `emergency` are independently chosen operator policies attempted in
that order; `stable` is not an admission mode but the outer hard-safety ceiling
that the execution validator checks independently on each concrete replica. The
smallest useful model has a single facet, \(f/F+g/G\le h\); more facets are
admitted **only** when valid held-out mixed-load data reject the single facet.

The single facet is a *conservative inner approximation*, and saying so
disarms the obvious objection. \(\{v: v^p/1+v^d/1\le1\}\) is the simplex
inscribed in the box \(\{v^p\le1\}\cap\{v^d\le1\}\): it under-admits and never
over-admits relative to independent per-phase limits. The geometry, not a
fitting argument, is why one facet is the right default.

Ignoring indivisibility, a homogeneous pool's service-only residual is the
Minkowski sum \(\bigoplus_{k}\mathcal R_k\) of per-replica residuals
\(\mathcal R_k=\{u\ge0:b_k+u\in\mathcal C^{\mathrm{mode}}_{q}\}\). Equal GPU
counts therefore need not imply equal available capacity, and aggregate
feasibility never implies replica feasibility (§A.8).

Every row is then divided by its residual capacity so that all constraints read
\(\le1\):

\[
u_{r,c}\ :=\ \frac{\text{consumption of }r\text{ by }c}{\text{residual capacity of }r},
\qquad
\text{giving}\qquad Ux\le\mathbf1 .
\]

A row whose residual capacity is non-positive makes the pool unavailable — a
baseline already outside the requested envelope is not a small violation to be
optimized against.

### A.4 The objective and reported frontier

For each source-power target, the solver jointly selects replay and KV
candidates and reports the unnormalized destination requirement vector:

\[
\left(\sum_c v_cx_c,\ \sum_c\kappa_cx_c,\ \sum_ct_cx_c,\
\sum_cb_cx_c,\ \operatorname{makespan}(x)\right)
\quad\text{versus}\quad \sum_cw_cx_c.
\]

The main sweep uses targets at 10/25/50/75/90/100% of maximum sheddable power,
source streams 1/2/4/8, bandwidth 1/5/10 Gbps, and route P50 RTT
10/60/90/150/240 ms. Method counts are reported so the mixed replay/KV choice is
visible. Destination power caps and pricing are outside this formulation.

The deployed objective is lexicographic: meet the conservative power target,
then minimize migration work. If the target is unreachable, maximize valid shed
and then minimize work at that maximum. The result is reported as
`target_unmet` with an explicit watt shortfall. It is never described as
successful curtailment.

### A.5 The integer program

Collecting §A.2–§A.4, expand each candidate over its source's stream bins
\(h\in\{1,\ldots,S_{i(j)}\}\). The requirement problem is

\[
\begin{aligned}
\text{(REQ)}\qquad
\min_{x}\quad & \sum_{c,h}t_c x_{c,h}\\
\text{s.t.}\quad
& \sum_{c,h} w_c x_{c,h}\ \ge\ \Theta
&&\text{(power target, covering)}\\
& \sum_{a,h}x_{(j,a),h}\ \le\ 1 && \forall j\in\mathcal J
&&\text{(one move per session)}\\
& \sum_{c:i(c)=i}t_cx_{c,h}\le H^{\mathrm{mig}}
&&\forall i,h &&\text{(indivisible source-stream bins)}\\
& \sum_{c,h}b_cx_{c,h}\le BH^{\mathrm{mig}}
&& &&\text{(logical-WAN fluid bound)}\\
& x_{c,h}\le E_c,\qquad x_{c,h}\in\{0,1\}.
\end{aligned}
\]

Destination service work, KV blocks, and replay/KV migration-slot seconds are
summed outputs, not constraints backed by invented capacity. The solver first
maximizes conservative modeled gain, then minimizes duration subject to
achieving \(\min(\Theta,\Theta^{\max})\). A supplied concrete destination may
later add normalized residual rows \(Ux\le\mathbf1\).

In matrix form, with \(A\) the session-incidence matrix,

\[
Ax\le\mathbf 1,\qquad Ux\le\mathbf 1,\qquad x\in\{0,1\}^{\lvert\mathcal C\rvert},
\]

where \(U\) contains the exact source-stream-bin rows and the WAN fluid row for
the frontier, plus supplied residual rows only for concrete admission. \(A\) is
a **partition matrix**: every column belongs to exactly one row. This single
fact is what makes the problem tractable, and §C exploits it.

Three properties of the formulation are worth naming explicitly.

**The power target is a lower bound, and deliberately so.** By concavity of
\(P^{\mathrm{on}}\), summing initial marginals underestimates the relief a set
of departures actually produces (Lemma C.1). The LP therefore optimizes a
conservative surrogate; exact source power is recomputed through
\(P^{\mathrm{on}}\) after integer selection, and the plan is accepted only if
the exact value clears the limit.

**Source sessions and methods are chosen jointly.** Replay and KV candidates
compete in one solve; the formulation never chooses one method globally.

**The WAN row is necessary, not sufficient.** Source stream bins are exact, but
the WAN byte row does not prove a concurrent schedule. Its makespan is a lower
bound until event execution validates it.

### A.6 Robust semantics

Let \(\omega\in\Omega\) jointly index a demand forecast and an empirical profile
case (\(F_{q,\omega},G_{q,\omega},N_{q,\omega},h_{q,\omega}\), migration
coefficients). The correct statement of feasibility is an
existence claim over a *single* assignment and a *single* nonanticipative
policy \(\pi\) that must work in every declared case:

\[
\exists\,x,\pi:\quad
\begin{cases}
\sum_{a,k}x_{(j,a,k)}=y_j,\ y_j\in\{0,1\} & \forall j,\\[2pt]
x_c\le E_c=\min_\omega E_{c,\omega},\\[2pt]
\sum_c w_c x_c\ \ge\ \Theta, & w_c=\min_\omega w_{c,\omega},\\[2pt]
b_{k,\omega}+\sum_{c:k(c)=k}v_{c,\omega}x_c\in\mathcal C^{\mathrm{mode}}_{q(k),\omega} & \forall k,\omega,\\[2pt]
\kappa^0_{k,\omega}+\sum_{c:k(c)=k}\kappa_{c,\omega}x_c\le\mathcal K_{q(k)} & \forall k,\omega,\\[2pt]
\operatorname{makespan}\bigl(\operatorname{Schedule}(\pi,x,\omega)\bigr)\le H^{\mathrm{mig}} & \forall\omega.
\end{cases}
\]

\(\pi\) is history-dependent: it maps observed completions to actions but cannot
read future case information. Eligibility must hold in *every* case and the
power gain is a simultaneous lower bound *across* cases — both are
\(\min_\omega\), not expectations. The current implementation admits one central
case, so these quantifiers are target semantics; §D.5 records what it would take
to populate \(\Omega\) (the profile already declares `faster` and `slower`
cases, and ten seeds).

### A.7 What the LP does not certify

Operational admission requires more than the five rows:

| Situation | Binding state | Consequence |
|---|---|---|
| incompatible or cold | pinned identity, warmness | no candidate exists |
| service-contested | baseline + landed work reaches the envelope | admit less, or label `sensitivity` |
| KV-contested | block-rounded histories reach allocatable HBM | reject, or choose another replica |
| packing-contested | pool aggregate fits, indivisible sessions do not | repair the assignment, or reject |
| route-contested | migrations share effective WAN bandwidth | schedule later or miss the deadline |
| endpoint-contested | replay compute, ingest, source stream, or migration slot serializes | schedule explicitly, check makespan |
| budget-contested | overlapping service and migration exceed either independent budget | admit less or schedule explicitly |
| stale or unreserved | state changed after planning | reacquire an atomic lease, or reject |

These are intersections, not alternative destination types: a site can be
service-, KV-, and route-contested at once, and the planner reports the largest
modelled pressure. `feasible` in the operational sense additionally requires an
accepted evidence status, warm/healthy attestation of the full pinned runtime,
a fresh baseline and residual-route snapshot held by a lease through commit. Until
those gates exist, a placement that the optimizer returns is reported as
`sensitivity/possible`, not as a safe plan.

Service and migration may overlap when both independent budgets fit. The paired
foreground measurements remain descriptive and do not add an interference
constraint.

### A.8 From pool aggregate to concrete replica

The LP is solved over pool-aggregated columns \((j,a,\mathcal P)\); sessions are
then packed onto concrete replicas inside their chosen pool. Aggregate
feasibility does not imply replica feasibility, because sessions are
indivisible: the aggregate row is a fractional-bin-packing relaxation of a
vector bin-packing problem.

Packing is deterministic best-fit on the worst normalized pressure,

\[
\operatorname{press}(k,c)=\max\!\left\{
\max_\iota\frac{N_\iota\!\cdot\!(b_k+v_c)}{h^{\mathrm{mode}}_\iota},\
\frac{\kappa^0_k+\kappa_c}{\mathcal K_q},\
\frac{\theta_k+t_c}{H^{\mathrm{mig}}}\right\},
\]

items ordered by decreasing standalone pressure, each placed on the replica
minimizing resulting pressure subject to \(\operatorname{press}\le1\). On
failure the correct repair is a **cut**,

\[
\sum_{c\in\mathcal C_\ell}x_c\ \le\ \lvert\mathcal C_\ell\rvert-1,
\]

added to the selection problem, which is then re-solved. (The current
implementation instead drops the single worst resource-per-watt candidate from
the incumbent set and re-packs — cheaper, but it can reject a set that another
replica assignment could have placed. An exact DFS assignment oracle exists and
is used in tests as a ground truth for whether a packing failure was genuine.)

Note that \(\operatorname{press}\) takes a **max**, not a sum, across service,
KV, and migration. This is intentional: service and migration may overlap, but
each independent budget must remain at or below one.

Finally, the aggregate byte row is not a schedule. The target execution model
reserves per-time capacity on the logical WAN route, every source stream, every
destination ingest engine, and every replica migration slot, and the
discrete-event simulator checks it: transfers share the route by max-min fair
progressive filling, KV flows additionally traverse a virtual per-destination
ingest link at \(\mu_{km}\), and a source slot is held from admission all the way
through quiesce, catch-up, and switch — not merely for the network phase. Any
temporary staging allocation would have to become its own measured row; there
is none today. The byte-row makespan is therefore a lower bound until event
execution validates the concurrent route schedule.

---

## §B. Optional concrete destinations

### B.1 The claim: more columns, not new variables

This section is a later admission comparison, not an input required by the
frontier. Adding destination sites does not change the mathematical form. A route is a
fixed edge list attached to a candidate, so *choosing where a session goes and
how it gets there is choosing a column*, and the incidence constraint
\(Ax\le\mathbf1\) already enforces that a session takes at most one. The
multi-site program is

\[
\text{(IP-}\mathcal N\text{)}\qquad
\min_x \sum_c t_c x_c
\quad\text{s.t.}\quad
\sum_c w_cx_c\ge\Theta,\quad
Ax\le\mathbf1,\quad
U^{\mathcal N}x\le\mathbf1,\quad
x\in\{0,1\}^{\lvert\mathcal C\rvert},
\]

with

\[
\mathcal C=\bigl\{(j,a,k): j\in\mathcal J,\ a\in\{R,K\},\ k\in\textstyle\bigcup_n\mathcal K_n,\ E_{(j,a,k)}\bigr\}.
\]

No flow variables and no path variables are required. This holds as long as a
single session's state travels as one unit along one path. It stops holding in
exactly two cases, and both are worth stating as scope: if a session's bytes may
be *split* across multiple paths, or if a *single* destination offers several
alternative routes whose selection interacts with other sessions' routing, then
path choice must become its own variable class and the program becomes a
multi-commodity flow with integer commodities. Neither is needed for "pick one
of \(N\) sites", and neither is in scope.

### B.2 The row inventory

With \(N\) sites and \(Q\) types:

| Row family | Count, one site | Count, \(N\) sites | Comment |
|---|---|---|---|
| source stream | \(\lvert\mathcal I\rvert\) | \(\lvert\mathcal I\rvert\) | **unchanged** — a source-side limit |
| logical route | 1 | \(N\) | one effective route per site; shared physical edges are a later refinement |
| service facet | \(\Phi\) | \(\sum_{n,q}\Phi_{n,q}\) | one per pool per facet |
| live KV | 1 | \(\lvert\{(n,q)\}\rvert\) | one per pool |
| migration slot | 1 | \(\lvert\{(n,q)\}\rvert\) | one per pool |

Columns grow by a factor of \(\sum_n\lvert\{q:\text{compatible}\}\rvert\); rows
grow additively. The matrix stays extremely sparse: a candidate touches exactly
one source row, \(\lvert\mathrm{path}(c)\rvert\) edge rows, and three or four
rows of its own pool — a column has \(O(1)\) nonzeros regardless of \(N\).

### B.3 The coupling that only exists when \(N>1\)

V1 has one logical route per destination, each with an effective bandwidth and
fixed P50 RTT. It does not require a source-fabric/egress/WAN/ingress/fabric
decomposition. Shared physical bottlenecks may be introduced later as explicit
rows when a concrete topology supplies independently meaningful capacities.

**Adding sites cannot relieve a source-side bound.** The source-stream row has
capacity \(S_iH^{\mathrm{mig}}\) and does not depend on \(N\) at all. In the
reference configuration \(S_i=1\), and 178 of 179 source replicas saturate that
row; no number of destinations changes it. This is the single most important
structural statement the multi-site section can make: *the constraint that binds
evacuation is on the side you are leaving, not the side you are going to.*

**Site diversity buys destination capacity and route alternatives, but no
source streams.** Whether spreading over \(N\) sites helps is decided by the
binding set reported by concrete admission.

**The fluid byte row and the duration model diverge under sharing.** Each
candidate's \(t_c\) is computed as if it owned the bottleneck rate
\(\min_{e\in\mathrm{path}(c)}B_e\). When the shared egress row is tight, that is
optimistic, and the LP's makespan estimate drifts from the simulator's max-min
schedule. With one route the two agree vacuously; with \(N\) sites they do not.
The fix inside the LP form is to charge each candidate the *residual* rate after
background reservations, \(\tau_{\mathrm{route}}(b,\mathrm{path})\ge b/\min_{e}B^{\mathrm{alloc}}_e\),
and to let the discrete-event validator carry the exact concurrent schedule.
The honest framing is: the byte row is a necessary pruning relaxation, and
deadline feasibility is certified by simulation, not by the LP.

### B.4 Heterogeneous destinations

Multiple sites make heterogeneous hardware unavoidable, and this is where a
single service facet is most exposed. Changing destination hardware is exactly
the operation that decouples prefill and decode — the regime in which one facet
\(f/F+g/G\le h\) is least defensible.

The formulation absorbs heterogeneity without structural change. Each type \(q\)
carries its own \(F_q(\cdot),G_q(\cdot),\mathcal K_q,\mu_q,L_q,N_q,h_q\), action
power, and migration coefficients; a candidate's entries in \(U\) are
(measured consumption)\(\,\div\,\)(measured capacity) and nothing else changes —
not the rows, not the columns, not the sparsity pattern, not the objective, not
the solver, not the packer, not the validator. Profiling a new type is one
offline job: KV capacity by closed form, a 5-point prefill sweep, a 5-point
decode sweep, and a 16-point mixed grid — 4–6 GPU-hours. Quantifying that cost
is itself a contribution, since the comparable systems publish no number.

Two guards are needed at \(N>1\) that are invisible at \(N=1\).

1. A type whose measured workload-direction range excludes a session must cause
   that *pool* to be skipped, not the whole plan to abort. Compatibility filtering
   must be candidate-local.
2. Pool-level aggregates (KV capacity, service bound) must never be summed
   across types. They are only meaningful within a pool.

The invariance test, run rather than asserted, supports the claim that only
numbers move. Scaling KV capacity by \(10\times\) and \(100\times\) changes
nothing — not one session, not one watt, in any workload — because KV was never
the binding row (0.34 and 0.77 at reference). Scaling prefill throughput by
\(100\times\) buys 11.5 % on the service-bound workload and 0 % on the
transition-bound ones, and the reason is exact: in the destination service
coordinate \(v=(v^p,v^d)\), interactive coding's aggregate demand splits 3.60
prefill against 21.20 decode, so prefill is 15 % of the load and even infinite
prefill throughput cannot remove more than 15 %. **Decode, not
prefill, holds the destination** — and decode throughput is the
slowest-improving constant in the table (HBM-bandwidth-bound, ~1.64\(\times\) per
GPU generation) while KV capacity and prefill throughput improve fastest.
Two caveats belong in the caption: the sweep scales sink constants only, holding
source packing fixed by design; and \(100\times\) prefill exceeds the roofline
by \(15\times\), so it is a limit probe establishing insensitivity *along that
ray*, not a forecast.

### B.5 Site-differentiated comparison

Concrete sites differ through supplied residual service, KV, migration, and
route budgets. Comparing those vectors against the requirement frontier does
not change the candidate or solver structure. Destination power prices and caps
are outside this formulation.

The multi-source generalization is equally cheap: with several source domains
under independent curtailment orders, replace the single covering row by one per
domain,

\[
\sum_{c:\,i(c)\in\mathcal I_d}w_cx_c\ \ge\ \Theta_d\qquad\forall d,
\]

which adds \(\lvert\mathcal D\rvert-1\) covering rows and leaves everything else
alone. The bounds of §C degrade gracefully in the total row count.

### B.6 What genuinely breaks the formulation

Being able to enumerate the complete list is a stronger claim than "the model is
general". Exactly three things break it.

1. **A new kind of consumable appears** — staging memory, facility power, a
   licence, an ingest engine distinct from the migration slot. The fix is to add
   a row; nothing else changes.
2. **A resource stops being additive.** Cross-session prefix sharing is the live
   example: KV would become a block-union rather than a sum, which concrete
   packing cannot express. This is the one extension that would require a
   non-linear packing stage.
3. **A resource stops being linear in the candidate** — interference that grows
   faster than the sum of the parts. The response is another facet, and the rule
   is to add one only when held-out mixed-load data reject the single facet.

Changing a hardware constant is not on the list, and that is the robustness
claim.

---

## §C. Structure: why a clean LP is solved greedily

This section is the technical core. It states why the natural relaxation is
tight enough that a greedy rule with approximate duals is not a compromise —
and, equally important, identifies the one modelling choice that decides
whether any approximation guarantee exists at all.

**The organizing result.** Write the problem in two orientations:

\[
\text{(MAX)}\ \max\Bigl\{\textstyle\sum_c w_cx_c:\ Ax\le\mathbf1,\ Ux\le\mathbf1\Bigr\}
\qquad
\text{(COV)}\ \min\Bigl\{\textstyle\sum_c t_cx_c:\ \textstyle\sum_c w_cx_c\ge\Theta,\ Ax\le\mathbf1,\ Ux\le\mathbf1\Bigr\}.
\]

\(Ax\le\mathbf1\) is the independent-set polytope of a **partition matroid**, so
(MAX) is *budgeted matroid independent set*: its feasible set is downward
closed, and Grandoni and Zenklusen give a deterministic PTAS for it at any fixed
number of budgets, satisfying every budget strictly. (COV) is not downward
closed — the covering row forbids dropping elements — which puts it in the
*budgeted matroid basis* regime, where Grandoni and Zenklusen show deciding
feasibility is NP-complete at two or more budgets. For (COV) the situation is
worse still: a direct subset-sum reduction (§C.4) makes feasibility NP-complete
already at \(R=1\).

The practical instruction is therefore sharp, and it is a formulation choice
rather than an algorithmic one: **state the problem as maximize shed subject to
packing, and treat the operator's target \(\Theta\) as an acceptance test on the
result, not as a constraint given to the solver.** The deployed lexicographic
order already does this in its fallback path; §C.4 argues it should be the
primary path.

Throughout, write the LP relaxation of (MAX),

\[
\text{(LP)}\qquad
\max_x\ \sum_c w_cx_c
\quad\text{s.t.}\quad
Ax\le\mathbf1,\quad Ux\le\mathbf1,\quad 0\le x\le\mathbf1,
\]

with \(A\) a partition matrix over sessions, \(U\in\mathbb R_{\ge0}^{R\times\lvert\mathcal C\rvert}\),
and \(R=\lvert\mathcal R\rvert\) the total number of resource rows. The
deployed order — minimize work subject to a covering power target — is treated
in §C.4.

### C.1 The linear power surrogate is conservative

**Lemma C.1.** *Let \(P\) be nondecreasing and concave on \([0,L]\) and define
\(\varphi(t)=P(L)-P(L-t)\). Then \(\varphi\) is nondecreasing, convex, and
\(\varphi(0)=0\); consequently \(\varphi\) is superadditive and*

\[
\Delta P_i(M)=P(L)-P\Bigl(L-\sum_{j\in M}\ell_j\Bigr)\ \ge\ \sum_{j\in M}\delta_{ij}(L).
\]

*Proof.* \(t\mapsto P(L-t)\) is concave (concave function composed with a
decreasing affine map), so \(\varphi=P(L)-P(L-\cdot)\) is convex; it is
nondecreasing because \(P\) is; \(\varphi(0)=0\) by construction. A convex
function vanishing at the origin is superadditive: for \(s,t\ge0\),
\(\varphi(s)=\varphi\bigl(\tfrac{s}{s+t}(s+t)+\tfrac{t}{s+t}\cdot0\bigr)\le\tfrac{s}{s+t}\varphi(s+t)\)
and symmetrically for \(t\), and adding gives
\(\varphi(s)+\varphi(t)\le\varphi(s+t)\). Induction extends this to \(\lvert M\rvert\)
terms. \(\square\)

For the fitted Michaelis–Menten form,
\(\varphi(t)=\Delta P\,t/\bigl[(1+L)(1+L-t)\bigr]\), whose second derivative is
strictly positive on \([0,L)\). The surrogate's pessimism is bounded: the ratio
\(\varphi\bigl(\sum_j\ell_j\bigr)\big/\sum_j\varphi(\ell_j)\) is at most
\(1+L\), attained in the limit of many infinitesimal sessions draining the node
completely. So the LP can understate a full drain by up to a factor \(1+L\) —
which is precisely the case the operator cares most about, and an argument for
adding an explicit node-emptying term to the objective rather than relying on
the summed marginals to find it.

**Consequence.** The LP's objective \(\sum_c w_cx_c\) is a *lower bound* on the
power a plan actually relieves. Meeting \(\Theta\) in the LP therefore implies
meeting it in the exact model, and the recomputation after integer selection can
only improve the reported shed. This is why a linear objective over a concave
physical law is safe here — the direction of the error is fixed by the geometry,
not by luck. It also means the sum is *pessimistic* in exactly the regime that
matters: as a node empties, later removals travel the steep part of the curve,
so a plan that drains a node is worth strictly more than the LP credits it.

### C.2 The relaxation has almost no fractionality

**Theorem C.2 (fractionality).** *Let \(x^\star\) be a basic feasible solution of
\(\{Ax\le\mathbf1,\ Ux\le\mathbf1,\ 0\le x\le\mathbf1\}\) with \(A\) a partition
matrix and \(U\) having \(R\) rows, and let \(\mathcal F=\{c:0<x^\star_c<1\}\),
\(\mathcal S_{\mathcal F}\) the set of sessions owning a column of
\(\mathcal F\). Then*

\[
\lvert\mathcal S_{\mathcal F}\rvert\le R,
\qquad
\lvert\mathcal F\rvert\le 2R,
\qquad
\sum_{c\in\mathcal F}x^\star_c\ \le\ R .
\]

This is not new, and the paper should cite it rather than claim it. It is
Grandoni and Zenklusen's theorem that any point on a dimension-\(d\) face of a
matroid polytope has at most \(2d\) fractional components with fractional mass
at most \(d\), specialized to a partition matroid with \(R\) budget rows; the
\(m\)-machine case appears earlier inside the proof of Lenstra, Shmoys and
Tardos's Theorem 1, who credit the idea to Dantzig. The specialization is short
enough to give in full:

*Proof.* Write \(\mathcal S^{t}\) for the sessions in \(\mathcal S_{\mathcal F}\)
whose \(A\)-row is tight, \(\mathcal S^{s}\) for those whose \(A\)-row is slack,
and \(R_{\mathcal F}\le R\) for the number of tight resource rows meeting
\(\mathcal F\).

*(i) Full column rank.* If the tight rows restricted to \(\mathcal F\) had rank
\(<\lvert\mathcal F\rvert\), pick \(d\neq0\) in their kernel supported on
\(\mathcal F\); then \(x^\star\pm\varepsilon d\) is feasible for small
\(\varepsilon\) — tight rows stay exact, slack rows stay slack, and the
\(\mathcal F\)-coordinates are strictly interior to their box — contradicting
vertexhood. Hence
\(\lvert\mathcal F\rvert\le\lvert\mathcal S^{t}\rvert+R_{\mathcal F}\).

*(ii) A tight session carries at least two.* If \(j\in\mathcal S^{t}\) owned a
single fractional column, its other columns are integral and sum to some integer
\(k\ge0\), forcing \(1-k\in(0,1)\), which no integer satisfies.

*(iii) The slack case.* A session in \(\mathcal S^{s}\) may own exactly one
fractional column, so counting columns gives
\(\lvert\mathcal F\rvert\ge2\lvert\mathcal S^{t}\rvert+\lvert\mathcal S^{s}\rvert\).
With (i),
\(2\lvert\mathcal S^{t}\rvert+\lvert\mathcal S^{s}\rvert\le\lvert\mathcal S^{t}\rvert+R_{\mathcal F}\),
hence \(\lvert\mathcal S_{\mathcal F}\rvert=\lvert\mathcal S^{t}\rvert+\lvert\mathcal S^{s}\rvert\le R_{\mathcal F}\le R\).

*(iv)* \(\lvert\mathcal F\rvert\le\lvert\mathcal S^{t}\rvert+R_{\mathcal F}\le2R_{\mathcal F}\le2R\).

*(v)* Each session's total mass is at most 1, so
\(\sum_{c\in\mathcal F}x^\star_c\le\lvert\mathcal S_{\mathcal F}\rvert\le R\). \(\square\)

Step (iii) is not decoration. Omitting it proves only
\(\lvert\mathcal S^{t}\rvert\le R\), which does not bound
\(\lvert\mathcal S_{\mathcal F}\rvert\), and the missing configuration is real:
with \(R=1\), two single-candidate sessions and \(u=(0.8,0.8)\), the point
\(x=(1,0.25)\) is a vertex whose second session has a slack \(A\)-row and one
fractional column.

**Hypotheses.** *Required:* \(A\) is a partition incidence matrix with unit
coefficients, integral right-hand side, integral box, and \(x^\star\) a vertex
(\(Ax=\mathbf1\) works identically). *Not required:* \(U\ge0\), \(u_{r,c}\le1\),
or any cap on candidates per session. Each requirement was checked by breaking
it: columns in more than one session row violate \(\lvert\mathcal F\rvert\le2R\)
in 124/3000 random vertices, non-unit \(A\) coefficients in 402/3000, and
\(Ax\le1.5\) in 568/2000 — while signed \(U\in[-0.5,0.5]\) gives 0/2000,
confirming that nonnegativity is genuinely unnecessary here. All three bounds
hold with zero violations across 4,000–6,000 random vertices, and
\(\lvert\mathcal F\rvert=2R\) is attained for \(R=1,\dots,5\), so the constants
are tight.

**Corollary C.3 (rounding loss).** *Let \(x^\star\) be a basic optimum of (LP)
and let \(\bar x\) round every fractional coordinate down to 0. Then \(\bar x\)
is integral and feasible, and*

\[
\sum_c w_c\bar x_c\ \ge\ \mathrm{LP}^\star-\!\!\sum_{c\in\mathcal F}\!w_cx^\star_c
\ \ge\ \mathrm{LP}^\star-R\max_c w_c
\ \ge\ \mathrm{OPT}_{\mathrm{IP}}-R\max_c w_c.
\]

The middle step uses the *fractional-mass* bound \(\sum_{\mathcal F}x^\star_c\le R\),
not the count bound, which is why the loss is \(R\max_cw_c\) and not
\(2R\max_cw_c\). Feasibility is immediate: every row of \(A\) and \(U\) is a
\(\le\) constraint with nonnegative coefficients, so reducing any \(x_c\) cannot
violate one.

This is the whole story of why the LP is worth so little here: **the integrality
gap is bounded by a handful of sessions, independent of how many sessions there
are.** With \(R=5\) and \(10^4\) sessions each worth \(O(1)\) W out of
\(O(10^4)\) W, the bound is a fraction of a percent — and the measured gaps are
0.15 % (coding: LP 7,215.45 W vs greedy 7,204.36 W) and 0.27 % (agentic:
9,004.43 vs 8,980.01), with the LP at 95.9 % of a trivial upper bound. The
theory predicts the magnitude before the experiment measures it.

**Corollary C.4 (an \((R{+}1)\)-approximation, and a PTAS).** *Under the
additional hypotheses \(w\ge0\), \(U\ge0\), and \(u_{r,c}\le1\), let \(x^\star\)
be a basic LP optimum, \(\hat x=\lfloor x^\star\rfloor\), and
\(B=\max_c w_c\). Then*

\[
\max\bigl\{\textstyle\sum_c w_c\hat x_c,\ B\bigr\}\ \ge\ \frac{1}{R+1}\,\mathrm{LP}^\star\ \ge\ \frac{1}{R+1}\,\mathrm{OPT}.
\]

*Proof.* \(\hat x\) is feasible by Corollary C.3. A session holding an integral
1 saturates its \(A\)-row and therefore owns no fractional column, so for each
\(j\in\mathcal S_{\mathcal F}\) all of \(j\)'s mass is fractional and sums to at
most 1. Hence
\(\mathrm{LP}^\star-\sum_cw_c\hat x_c=\sum_{j\in\mathcal S_{\mathcal F}}\sum_{c\in j\cap\mathcal F}w_cx^\star_c
\le\sum_{j\in\mathcal S_{\mathcal F}}\max_{c\in j\cap\mathcal F}w_c\le\lvert\mathcal S_{\mathcal F}\rvert B\le RB\).
If \(\sum_cw_c\hat x_c<\mathrm{LP}^\star/(R+1)\) then \(RB>\tfrac{R}{R+1}\mathrm{LP}^\star\),
so \(B>\mathrm{LP}^\star/(R+1)\); either way the max attains the bound. \(\square\)

The factor is exactly \(R+1\), and it is the *session* bound
\(\lvert\mathcal S_{\mathcal F}\rvert\le R\) that buys it — using the weaker
column bound \(\lvert\mathcal F\rvert\le2R\) would give \(2R+1\). The condition
\(u_{r,c}\le1\) is load-bearing: it is what makes a single candidate feasible on
its own, so that \(B\) is attainable. We found no canonical citation for this
form, so the paper should carry the proof inline.

Guessing the \(R/\varepsilon\) heaviest elements first drives the same round-down
loss below \(\varepsilon\,\mathrm{OPT}\), which is Grandoni and Zenklusen's
deterministic PTAS for budgeted matroid independent set at fixed \(R\), with
every budget satisfied strictly.

For \(R=1\) the problem is a knapsack and admits an FPTAS [IK75, Law79]. The
usual statement is that no FPTAS exists for \(R\ge2\); we could not verify that
claim at sentence level in either of the sources it is normally attributed to,
so the paper should either check it or state it as folklore. Without session
rows, the fixed-\(R\) PTAS is Frieze and Clarke's [FC84].

The submodular \((1-1/e)\) machinery does **not** apply, and it is the most
tempting wrong move available: the true set function
\(M\mapsto\Delta P_i(M)\) is *supermodular* by Lemma C.1, which is the opposite
of what those results require. The generalized-assignment results do not apply
either — they need one capacity row per machine, whereas a candidate here
touches one source row, \(\lvert\mathrm{path}(c)\rvert\) edge rows, and its
pool's service, KV, and migration rows, with the source and edge rows *shared
across pools*.

**Corollary C.5 (scaling to many sites).** In (IP-\(\mathcal N\)) the bound
becomes \(R_{\mathcal N}\max_cw_c\) with
\(R_{\mathcal N}=\lvert\mathcal I\rvert+\lvert\mathcal E\rvert+\sum_{n,q}(\Phi_{n,q}+2)+N\).
The gap grows *linearly in the topology*, not in the fleet — a hundred sites
with a hundred edges costs a few hundred sessions of slack out of a million.
The PTAS survives too, since \(R_{\mathcal N}\) is fixed by the deployment, not
by the workload.

**One implementation caveat that must be in the paper.** Theorem C.2 is a
statement about *basic* solutions. The current solve uses an interior-point
conic solver with no crossover, so what comes back is an analytic-center-like
interior point at which essentially every coordinate is fractional, and the
bound is not realized — which is precisely why the deployed rounding is a
sort-and-fill over all columns rather than a repair of \(\le2R\) of them.
Switching to a simplex solve (or enabling crossover) makes the bound
constructive and turns Corollary C.3 into an executable guarantee. This is a
one-line change with a provable payoff and it should be made before the
submission.

### C.3 The dual, and what greedy is approximating

The dual of (LP), with \(\lambda_j\ge0\) on session rows and \(y_r\ge0\) on
resource rows, is

\[
\min_{\lambda,y\ge0}\ \sum_j\lambda_j+\sum_r y_r
\quad\text{s.t.}\quad
\lambda_{j(c)}+\sum_r u_{r,c}y_r\ \ge\ w_c\quad\forall c .
\]

Define the **reduced cost** of a candidate at prices \(y\),

\[
\bar w_c(y)=w_c-\sum_r u_{r,c}y_r .
\]

At optimality, \(\lambda_j=\max\{0,\max_{c:j(c)=j}\bar w_c(y^\star)\}\), and
complementary slackness says \(x^\star_c>0\) only when
\(\bar w_c(y^\star)=\lambda_{j(c)}\). Read operationally: **given the right
prices, the problem separates by session.** Each session independently picks the
candidate with the best reduced cost and takes it if that value is positive.
There is no coupling left. All the difficulty is in the \(R\) numbers \(y^\star\).

This is the structural reason a greedy is the natural algorithm rather than a
fallback. The LP does not need to be solved to be *used*; it needs \(R\) prices,
and \(R\) is five.

**The exactly-solvable case, and its exact limit.** For a *pure* one-row
packing LP — no session rows — sorting by \(w_c/u_{1,c}\) and filling until the
row saturates is optimal, with one fractional variable at the boundary and
optimal dual price \(y^\star=w_{c^\dagger}/u_{1,c^\dagger}\) at the critical
candidate. That is Dantzig's fractional-knapsack result and it is the reason
ratio ordering is the right primitive.

It does **not** survive the addition of the session rows, and the paper must not
claim it does. Once each session may contribute at most one of several
candidates, ratio ordering is strictly suboptimal even at \(R=1\). The minimal
instance: one row of capacity 1, session 0 owning
\(c_1=(w{=}10,u{=}0.5)\) and \(c_2=(w{=}1,u{=}0.01)\), session 1 owning
\(c_3=(w{=}9,u{=}0.5)\). Ratio ordering puts \(c_2\) first at ratio 100, which
consumes session 0's slot; \(c_3\) then fits and \(c_1\) is blocked, for a total
of 10 against the optimum 19 attained by \(\{c_1,c_3\}\). Letting
\(w_{c_2},u_{c_2}\to0\) with \(w_{c_2}/u_{c_2}\to\infty\) makes the gap
unbounded. A cheap decoy candidate that is excellent per unit of resource but
worthless in absolute terms can consume the session slot that the optimum needed
for a heavy candidate. What remains true is the
weaker and still useful statement that the reference runs bind on essentially
one row family at a time — source streams at 1.00, or migration slots at 1.00
with route at 0.99 — so the instances are *close* to the regime where ratio
ordering is near-optimal, which is consistent with the measured 0.15–0.27 %
gaps. Closeness, not optimality, is the claim.

**The price heuristic.** The implementation estimates prices in one shot. Let
\(c^\star(j)\) be the cheapest legal candidate for session \(j\) by total
normalized consumption, let

\[
\sigma_r=\sum_{j\in\mathcal J}u_{r,c^\star(j)}
\]

be the demand row \(r\) would see if every session took its cheapest option
(capacities are already normalized to 1), and set

\[
\hat y_r=\max\{1,\sigma_r\}.
\]

\(\sigma_r\) is an oversubscription ratio: a row with \(\sigma_r\le1\) is not
contested and is priced at the floor; a row with \(\sigma_r=3\) is three times
oversubscribed and is charged three times as much. Counting one candidate per
session — rather than all of them — is what keeps mutually exclusive replay/KV
and multi-site alternatives from inflating every price simultaneously. The score
is the bang-per-buck ratio

\[
\operatorname{score}(c)=\frac{w_c}{\sum_r\hat y_r u_{r,c}},
\]

candidates are sorted once, and the ordering is swept taking every candidate
that fits the remaining capacity and whose session is unclaimed. The loop is
\(O(\lvert\mathcal C\rvert\log\lvert\mathcal C\rvert)\) with an \(O(1)\)-nonzero
feasibility check per column.

Four honest statements about this heuristic.

- **Ratio ordering is a breakpoint ordering, and that is why it is the right
  primitive.** Scale the price vector by \(\theta\ge0\). The reduced cost
  \(w_c-\theta\sum_r\hat y_ru_{r,c}\) hits zero at
  \(\theta_c=w_c/\sum_r\hat y_ru_{r,c}\), which is exactly the score. Sorting by
  the score therefore sorts candidates by *how expensive prices would have to
  get before this candidate stops being worth taking* — the correct order for a
  sweep that consumes capacity and thereby raises the shadow price. Ratio and
  reduced-cost orderings coincide only when either the consumption
  \(\sum_r\hat y_ru_{r,c}\) or the gain \(w_c\) is constant across candidates;
  otherwise they genuinely differ, and the difference is a source of greedy's
  residual gap. The ratio is well-formed here because \(u_{r,c}\) is normalized
  by capacity and hence dimensionless, so the denominator adds commensurable
  quantities — an unnormalized version would be adding seconds to bytes.
- **It is a one-shot version of a price-update scheme.** Iteratively reweighting
  \(y_r\) upward in proportion to observed congestion is the standard way to
  solve fractional packing LPs approximately; the implementation takes the first
  such step and stops. Prices are never recomputed inside the loop, so a row
  that becomes contested *because of* the choices greedy makes is never
  repriced. The functional shape is that of the standard packing-LP price
  schemes, but **no guarantee from that literature transfers**, because those
  bounds require the iteration.
- **The floor at one is not free.** Setting \(\hat y_r=\max\{1,\sigma_r\}\)
  charges an uncontested row the same unit price as a marginally contested one,
  and by complementary slackness the correct price on a slack row is *zero*. The
  floor cannot express zero, so it inverts orderings. Minimal instance: \(R=2\),
  two single-candidate sessions \(A=(w{=}1,u{=}(0.1,1.0))\) and
  \(B=(w{=}1,u{=}(0.5,0))\). Then \(\sigma=(0.6,1.0)\) and
  \(\hat y=(1,1)\), scoring \(A\) at \(0.909\) and \(B\) at \(2.0\), so the
  heuristic prefers \(B\). If row 2 is slack at the LP optimum, \(y^\star_2=0\)
  and the true scores are \(10/y_1\) against \(2/y_1\) — \(A\) is better by
  \(5\times\), and the order is exactly reversed.
- **The deployed greedy has no approximation guarantee, and its worst case is
  unbounded.** This must be stated plainly. The unboundedness is proved by the
  decoy family above. On randomized search over 20,000 instances the worst
  greedy-to-LP ratio observed was **0.108** — below \(1/(R{+}1)=1/3\), which
  confirms directly that the guarantee of Corollary C.4 attaches to *LP
  round-down plus best-single*, not to this heuristic. What is provable about
  greedy is only the bracket: its output is feasible and integral, and
  \(\mathrm{LP}^\star\ge\mathrm{OPT}_{\mathrm{IP}}\) bounds it from above.

**Two changes make the greedy defensible, and both are small.**

*Add the session slot to the denominator.* The score's denominator prices the
resource rows but omits the session row entirely, whose dual is \(\lambda_j\).
Restoring it as a unit price,

\[
\operatorname{score}(c)=\frac{w_c}{1+\sum_r\hat y_ru_{r,c}},
\]

makes the score "gain per unit of one session slot plus priced resources", and
incidentally removes a latent singularity: the current form divides by a
\(10^{-12}\) guard, so a candidate with an all-zero resource column would score
as effectively infinite and consume its session's slot ahead of everything. (On
the pool path that column cannot arise today — every candidate has positive
duration and hence positive source and migration entries, and cold sessions are
filtered out before candidates are built — but the guard is one refactor away
from mattering.) Measured effect: the decoy instance goes from \(0.526\times\)
to \(1.000\times\) LP, the search worst case from \(0.108\times\) to
\(0.907\times\), and the worst over 20,000 random instances from \(0.108\) to
\(0.283\). It is still not a guarantee.

*Take the better of greedy and the best single candidate.* This is what converts
the pipeline into a genuine \((R{+}1)\)-approximation via Corollary C.4. Two
lines, no measurement, no new data.

### C.4 Covering targets, and where every guarantee dies

The deployed orientation is (COV): minimize work subject to a covering power
row. This is not a cosmetic difference from (MAX). It leaves the downward-closed
regime, and with it every approximation result of §C.2.

**Feasibility itself is NP-complete, already at \(R=1\).** Reduce from
subset-sum \((a_1,\dots,a_n;T)\): one session per item with a single candidate,
\(w_c=a_c\), \(t_c=0\), \(u_{1,c}=a_c/T\), and \(\Theta=T\). Then \(x\) is
feasible iff \(\sum_ca_cx_c\ge T\) *and* \(\sum_ca_cx_c\le T\), i.e. iff the
selected items sum to exactly \(T\); membership in NP is immediate. Already
\(a=(3,5,7,11)\), \(T=13\) is LP-feasible at \(x=(0,0,0.286,1)\) with **no
integral solution at all**. This sits alongside Grandoni and Zenklusen's
hardness for budgeted matroid *basis* at two or more budgets, reached by a
different route and agreeing. So there is no approximation algorithm for (COV)
in general — not a PTAS, not a constant factor — and no LP rounding can be
guaranteed to return a feasible integral point. Every guarantee must relax
either \(\Theta\) or the capacities.

**Rounding does not work in either direction.** Rounding down preserves the
packing rows but can break the covering row; the shortfall is bounded by
Corollary C.3 but not eliminated. Rounding up is more interesting: define
\(\tilde x\) by keeping every integral 1 and, for each fractional session
\(j\in\mathcal S_{\mathcal F}\), setting to 1 the single candidate
\(c_j=\arg\max_{c\in j\cap\mathcal F}w_c\). Then \(\tilde x\) is integral,
satisfies \(A\tilde x\le\mathbf1\), and **meets the power target exactly, with
no relaxation** — since
\(\sum_{c\in j\cap\mathcal F}w_cx^\star_c\le w_{c_j}\sum_{c\in j\cap\mathcal F}x^\star_c\le w_{c_j}\)
— at a cost of at most \(\mathrm{LP}^\star_{\mathrm{cost}}+(R{+}1)t_{\max}\) and
a capacity violation of at most

\[
\sum_c u_{r,c}\tilde x_c\ \le\ 1+(R{+}1)\max_c u_{r,c}\qquad\forall r .
\]

That is the honest bicriteria statement, and its usefulness depends entirely on
a small-demand hypothesis: if \(\max_cu_{r,c}\le\varepsilon\) over the fractional
candidates the violation is a mild \(1+(R{+}1)\varepsilon\), but with no such
hypothesis it degrades to \(R+2\) and is near-vacuous. In our setting a capacity
violation is exactly what one may not ship — the rows are physical HBM, physical
link time, and a physical deadline — so the round-up branch is a theoretical
completeness result, not a deployable one.

**The remedy is to reorient, not to round harder.** Solve (MAX). Its optimum
\(\Theta^{\max}\) upper-bounds what any integral plan can shed. Then:

1. If \(\Theta>\Theta^{\max}\), the target is unreachable. Report `target_unmet`
   with the explicit shortfall \(\Theta-\Theta^{\max}\). Never describe it as
   successful curtailment.
2. If \(\Theta\le\Theta^{\max}-R\max_cw_c\), the rounded-down solution already
   clears the target by Corollary C.3 — no repair, and the guarantee holds.
3. In the narrow band between, a bounded fill of at most \(R\) sessions is
   needed and may fail. That failure is reportable, and it is the only regime
   where the covering hardness actually bites.
4. Only then minimize work, as a second lexicographic stage at fixed shed.

This is what the deployed fallback path already does when the covering solve is
infeasible. The recommendation is to make it the *primary* path: the operator's
\(\Theta\) is an acceptance test applied to a solved trade-off curve, not a
constraint handed to the solver. That single reformulation is what moves the
problem from "no approximation exists" to "a PTAS exists," and it costs nothing
operationally, because a planner that reports the achievable frontier is
strictly more useful to an operator than one that reports infeasible.

For completeness: treating the covering row as one of the \(R\) rows keeps
Theorem C.2's counting argument valid — the proof never used the sign of the
non-session constraints — so the *fractionality* bound survives at \(R{+}1\)
even though the approximation guarantee does not.

### C.5 Where greedy actually loses, and why

The measured gaps are 0.15 % and 0.27 % on the two transition-bound workloads
and large — 2,753 W vs 1,964 W — on the one service-bound workload. The
asymmetry is explained by the price heuristic, not by the LP:

- On a **transition-bound** instance one row dominates, the instance is close to
  the \(R=1\) case, and ratio ordering is near-optimal (§C.3).
- On a **service-bound** instance the service row saturates at 1.0000 while
  other rows have slack, so the single-shot price \(\hat y\) misprices the
  contested row relative to the free ones, and greedy stops early. Repricing
  once after the first saturation would close most of this.
- At very large scale the LP's *rounding* becomes the liability rather than its
  optimum: at \(10^6\) sessions the deployed pipeline overshoots the target by
  \(1.89\times\), making 949,031 moves where 524,241 suffice. That is a defect
  in the sort-and-fill (it does not stop at the target during the fill pass),
  not in the relaxation.

The paper should report the **LP triple** — fractional bound, rounded, packed —
plus an exact integer optimum on 100–500-session instances, with the watt gap at
each step. That table, not a claim of optimality, is the defensible statement.

---

## §D. The constants, their provenance, and their domains

### D.1 Measured

| Quantity | Value | Error | Provenance |
|---|---|---|---|
| Power vs load | \(P_0=67.1204\) W, \(P_0{+}\Delta P=424.446\) W, \(\kappa=1.94858\); usable range \(67.12\to248.89\) W over \(\ell\in[0,0.531358]\) | 5 % | 5 s windows, \(R^2=0.9901\), 3,725 windows |
| Cross-type generality | \(R^2\in[0.911,0.990]\) across 25 node types, 82,046 windows, 680 runs | — | in-sample (see D.5) |
| Decode:prefill energy | \(5.25\)–\(25.24\times\) per token | — | linear two-price fit, all 25 types |
| \(F,G\) | 1448.32, 1260.38 tok/s (p99.5 of positive-rate windows at 5 s) | window-sensitive: \(F\) moves \(2.51\times\), \(G\) \(1.41\times\) over 1–30 s | measured |
| KV capacity | 963,152 tokens/replica = 44.09 GiB | exact readback | vLLM 0.22.0, 0.75 memory utilization |
| KV bytes/token | 49,152 B = 48 KiB | closed form \(2\times24\times8\times64\times2\) | back-solves the measured capacity to 0.0065 % |
| KV ingest | 620.78 MB/s = 12,630 tok/s | — | CPU-mediated LMCache path |
| Prefill \(F_q(T)\) | 4,655–7,634 tok/s over 256–31,562 tok | 25 % | 10.7–17.6 % MFU vs a 43,333 tok/s roofline |
| Decode \(G_q(T)\) | 3,774 \(\to\) 77.9 tok/s over 256 \(\to\) 31,562 tok (48.5\(\times\) collapse) | 25 % | \(1/G=a+bT\) holds to 4 % below 16K, 19 % over at 24.5K, 138 % at 31.5K |
| Tail replay rate | 919.4 tok/s | 30 % | measured |
| Action power | replay 189.84 W dst / 2.06 W src; KV 11.84 W dst / 3.18 W src | — | measured at concurrency 1 only; **totals at that concurrency, not per session** |
| Foreground cost | replay \(+1.084\) s TTFT, \(+3.45\) ms/tok TPOT; KV \(+4.7\) ms, \(+0.42\) ms/tok | \(n=1\), \(n=5\) | not a percentile bound |
| Sleep | saves 0.0158 W | 100 % | sleep is useless; only power-off recovers the floor |

### D.2 Fitted, held out

| Model | Form | Held-out error | Property |
|---|---|---|---|
| Replay time | \(C_j/B+0.58666\,[\,T/\rho(T)+1.3404(1+\chi)\,]+\tau^{\mathrm{sw}}\) | **9.6 %** median at 24K | never underpredicts |
| KV time | \(K_{jk}/B+1.13382+\text{catch-up}\) | **7.8 %** median at 24K | never underpredicts |

Both were fit on six 16K rows and evaluated at a different context *and* a
different bandwidth. They are the strongest artifacts in the evidence base and
should anchor the paper's fidelity argument. A categorical idle/busy split was
tested and **rejected** — it underpredicted two held-out rows.

### D.3 Assumed, and load-bearing

| Assumption | Value | Consequence if wrong |
|---|---|---|
| Source migration streams \(S_i\) | **1/2/4/8 sensitivity** | report fixed-plan invariants and reoptimized resource changes; our own campaign measured 591 MB/s at concurrency 2 and 1.206 GB/s at concurrency 4 against a 111 MB/s serialized ceiling |
| Service bound \(h\) | 0.096953, with `normal = emergency = stable` | a probe that passed, used as an upper bound; there is **no failure anywhere in the dataset**, and a *higher* passing point at 0.114063 is ignored. The mode machinery is inert |
| Service and migration overlap | independent budgets, `max` not `sum` | allowed when both budgets fit |
| WAN | one logical route: 5 Gbps central, 1/5/10 Gbps; P50 RTT 10/60/90/150/240 ms | explicit sensitivity; one RTT is added per action |
| Request rate | 1/180 req/s/session (0.31–0.86 req/s/replica) | inside Llumnix's published 0.42–1.9 req/s/instance range |
| Migration concurrency | 1 per destination replica | untested |

### D.4 Hard-fail domains

Outside these, the code raises rather than extrapolating — with the exceptions
noted, which are defects:

| Quantity | Valid range | On violation |
|---|---|---|
| load \(\ell\) | \([0,\ 0.531358]\) | `ValueError` |
| power curve shape | nondecreasing **and concave** | `ValueError` |
| prefill / decode context | \([256,\ 31562]\) tok | `ValueError` |
| replay context | \([3473,\ 31562]\) tok | `ValueError` |
| concurrency | \(\{1\}\) for rate curves and action power | `ValueError` |
| loaded-migration context | \([16384,\ 24576]\) tok | `ValueError` |
| loaded-migration bandwidth | \([625,\ 1250]\) MB/s (5–10 Gbps) | `ValueError` |
| destination context rate | table endpoints | **silently returns \(\min\) rate** when `extrapolate=True` — a defect |
| migration components | same 16K–24K, 5–10 Gbps | **soft**: surfaces as `evidence_status=unsupported_extrapolation` |

The measured migration domain ends at 24,576 tokens while the simulator runs to
31,562, and **100 % of interactive-coding moves are extrapolated**
(`in_domain_fraction` = 0.000 / 0.339 / 0.347 across the three workloads).
Closing this is a named experiment, not a caveat to be buried.

### D.5 Evidence status, stated plainly

The service envelope is **not** established. Of 9,181 recorded requests: 47 runs
are measurement-invalid (50 requests returned HTTP 200 with zero prompt and
output usage despite a forced token); 6 deterministic forced-token signatures in
the range 200110–200952 identify a harness defect, since no successful request
uses a forced ID \(\ge200000\); 60 of 66 complete-work runs are append-hot
because the runner prewarms only the historical prefix while vLLM's automatic
prefix cache is never reset, so later cells reuse nominally-future append blocks;
and 1 run is an excluded prefix under-hit. The physical execution is the
contamination unit — one append-hot request excludes the whole run.

What survives is **five private-prefix-consistent executions**, and because the
archive did not retain returned token identity, finish reason, or stream
completion, even those cannot satisfy the completion contract. They are
descriptive sensitivity anchors, not admissible service points. Four of the five
sit at radius 0.096953 (the number wired into the planner as
`normal = emergency = stable`); the fifth, interactive coding in emergency mode,
passed at 0.114063. Coding has exactly one such execution.

Consequently: **there is no admissible service point, no service boundary, and
no private-prefix-consistent failure anywhere in the dataset.** A bound with no
failure in the dataset is not a bound. Any placement depending on it — which is
all of them — is `sensitivity/possible`.

The private-prefix consistency test, at 16-token blocks, is

\[
\mathrm{cached}_j\ \le\ \left\lfloor\frac{\mathrm{prompt}_j-\mathrm{append}_j}{L_q}\right\rfloor L_q .
\]

Two further validity gaps: the power law is fit and scored on the same windows,
with no held-out split and no MAPE — the single most likely reason a reviewer
rejects a paper selling "grounded in our own measurements". And the analytic
predictor *is* the simulator, differing only by a logging flag, so
"our model matches our simulator" has nothing behind it. Both are fixable
without a GPU.

---

## §E. Scope, and what is deliberately not modelled

**We model the landing, not the stay.** All five resource rows are either
transition-time rows (streams, edges, migration slots) or instantaneous
occupancy rows (service, KV at \(t=0\)). None is a *sustained* constraint. A
real curtailment event runs about two hours; the residency horizon is 180 s and
sessions are frozen snapshots that do not grow after they land. Letting them
grow at their traced rate, destination KV occupancy reaches 139.6/107.6/110.4 %
at 15 minutes and 534.2/179.0/206.3 % at two hours — **the sink fills 217–289
seconds after the migration completes.** The evacuation succeeds for about four
minutes.

This is a missing dimension, not a tuning problem, and the paper must pick one
of three responses and say so: model destination session churn (arrivals and
departures), model progressive eviction or tier-offload during the stay, or
model return migration when the event ends. Stating plainly that we solve the
*landing* problem and that sustained occupancy is out of scope is defensible —
but only if said.

**Also out of scope**, and each for a stated reason:

- *Cold model loading and reallocation.* A separate placement problem; v1
  assumes warm weights, whose memory is already excluded from measured
  \(\mathcal K_q\).
- *Continuous-batching latency prediction.* The service facet is a fluid
  admission model. It does not predict per-request TTFT, and the paper should
  stop implying it does. The claim to make is: *given a destination that reports
  its residual capacity, this evacuation either fits or it does not, and here is
  exactly the residual it requires and exactly what the transition costs in
  watts, bytes, and time at both sites.* The headline figure is **required
  destination residual vs. watts shed**, with the reader's own bound as a
  horizontal line — a plot invariant to the 0.096953 argument.
- *Cross-session prefix sharing.* Deliberately uncredited; produces
  false negatives, never false positives.
- *Failure, lease expiry, and rollback.* The simulator is authoritative only for
  deterministic schedules under fixed healthy resources. Execution must
  revalidate the lease, retain source ownership until commit, and abort the
  destination attempt on any pre-commit failure.
- *Tiered memory as a substitute for HBM.* DRAM or SSD may stage a transfer but
  an active landed session must fit the live HBM KV row, unless a lazy-retrieval
  mode acquires its own measured latency and service envelope.

**Feasibility, in the scope that is modelled**, requires all four of: exact
modelled source power after integer selection at or below the limit; the
trailing-window source power at \(D\) at or below the limit; every selected
migration committed by \(D\); and any requested sleep or off transition finished
by \(D\). Experiment acceptance additionally requires every request observed by
\(D\) to start by \(D\) — a routing-readiness check, not an end-to-end latency
claim.

---

## §F. Mapping to paper sections

| Paper section | Source here | Headline claim |
|---|---|---|
| §\ref{sec:marginal-power} (drafted) | §0.1, Lemma C.1 | concave power ⇒ marginals are load-dependent and summing them is conservative |
| §\ref{sec:single-session} (drafted) | §A.2 | replay vs KV is a bandwidth-vs-prefill crossover, with an ingest floor |
| §5 Many sessions, one destination class | §A | required residual resources vs. watts shed with joint replay/KV selection |
| §6 Concrete destination comparison | §B | site choice is an optional column refinement; the binding constraint may remain on the source side |
| §7 Solving it | §C | partition matroid + \(R\) budgets ⇒ \(\le2R\) fractional and \(\le R\) fractional mass ⇒ the LP is worth at most \(R\max_cw_c\) more than rounding it down, and max-shed admits a PTAS while the covering form admits nothing |
| §8 Evaluation | §D | required destination residual vs. watts shed; the invariance sweep; the LP triple |
| §9 Limitations | §E | we model the landing, not the stay |

Figures the formulation directly supports:

1. **Required destination residual vs. watts shed**, per workload, with the
   frontier already computed (coding meets every cut to 90 % while preserving
   7,746 of 10,000 sessions; interactive coding saturates at a 20 % cut and
   never moves again because the sink-service row pins at 1.0000).
2. **The binding-set map**: which of the five row families is at \(\ge0.95\), as
   a function of \(S_i\), KV capacity, and prefill throughput — the invariance
   sweep of §B.4, which is also the answer to "what if the hardware improves".
3. **The mixed replay/KV method frontier** across bandwidth, RTT, and source-stream sensitivity.
4. **The LP triple** — fractional / rounded / packed — against an exact integer
   optimum at 100–500 sessions (§C.5).
5. **Ramp rate in MW/min**, which is the output format regulators are actively
   writing into the requirement, and which this formulation computes directly.

---

## §G. References for the structural results

Verified against Crossref/OpenAlex/dblp/arXiv, and for the two load-bearing
entries against the paper PDFs. Safe to print as written.

- **[GZ10]** Fabrizio Grandoni and Rico Zenklusen. "Approximation Schemes for
  Multi-Budgeted Independence Systems." *ESA 2010*, LNCS 6346, pp. 536–548.
  DOI 10.1007/978-3-642-15775-2_46. Full version arXiv:1002.2147, retitled
  "Optimization with More than One Budget."
  *Theorem 3* is the fractionality bound of Theorem C.2; *Corollary 2* is the
  PTAS; *Theorem 1* is the basis-side hardness.
- **[LST90]** Jan Karel Lenstra, David B. Shmoys, Éva Tardos. "Approximation
  Algorithms for Scheduling Unrelated Parallel Machines." *Mathematical
  Programming* 46(1–3):259–271, 1990. DOI 10.1007/BF01585745.
  The \(m\)-machine form of the fractionality bound appears **inside the proof
  of Theorem 1, p. 263**, not as a numbered lemma — cite it that way.
- **[FC84]** A. M. Frieze and M. R. B. Clarke. "Approximation algorithms for the
  m-dimensional 0–1 knapsack problem: Worst-case and probabilistic analyses."
  *EJOR* 15(1):100–109, 1984. DOI 10.1016/0377-2217(84)90053-5.
- **[IK75]** Oscar H. Ibarra and Chul E. Kim. "Fast Approximation Algorithms for
  the Knapsack and Sum of Subset Problems." *JACM* 22(4):463–468, 1975.
  DOI 10.1145/321906.321909.
- **[Law79]** Eugene L. Lawler. "Fast Approximation Algorithms for Knapsack
  Problems." *Mathematics of Operations Research* 4(4):339–356, 1979.
  DOI 10.1287/moor.4.4.339.
- **[GK07]** Naveen Garg and Jochen Könemann. "Faster and Simpler Algorithms for
  Multicommodity Flow and Other Fractional **Packing** Problems." *SIAM J.
  Comput.* 37(2):630–652, 2007. DOI 10.1137/S0097539704446232. Conference
  version FOCS 1998, pp. 300–309. (The title has no "and covering".)
- **[ST93]** David B. Shmoys and Éva Tardos. "An Approximation Algorithm for the
  Generalized Assignment Problem." *Mathematical Programming* 62(1–3):461–474,
  1993. DOI 10.1007/BF01585178. Cited only to say it does *not* apply.
- **[FGMS11]** Lisa Fleischer, Michel X. Goemans, Vahab S. Mirrokni, Maxim
  Sviridenko. "Tight Approximation Algorithms for Maximum **Separable**
  Assignment Problems." *Mathematics of Operations Research* 36(3):416–431,
  2011. DOI 10.1287/moor.1110.0499. The SODA 2006 version (pp. 611–620) says
  "General", not "Separable" — the paper was renamed.
- **[NWF78]** G. L. Nemhauser, L. A. Wolsey, M. L. Fisher. "An analysis of
  approximations for maximizing submodular set functions—I." *Mathematical
  Programming* 14(1):265–294, 1978. DOI 10.1007/BF01588971.
- **[CCPV11]** Gruia Călinescu, Chandra Chekuri, Martin Pál, Jan Vondrák.
  "Maximizing a Monotone Submodular Function Subject to a Matroid Constraint."
  *SIAM J. Comput.* 40(6):1740–1766, 2011. DOI 10.1137/080733991. Cite this,
  **not** IPCO 2007 — continuous greedy arrived via Vondrák, STOC 2008.

**Do not print without checking first.** Each of these was reached for during
the analysis and could not be confirmed:

- *Dantzig 1963* (*Linear Programming and Extensions*, Princeton). Only [LST90]'s
  credit line is confirmed; the primary entry was never checked. Attribute via
  [LST90] or verify.
- *Edmonds 1971*, "Matroids and the greedy algorithm," *Mathematical
  Programming* 1:127–136 — unverified; it was introduced during synthesis, not
  during the verification pass.
- *Kellerer, Pferschy, Pisinger*, **Knapsack Problems**, Springer 2004: Ch. 9
  "Multidimensional Knapsack Problems," pp. 235–283, with approximation in §9.4,
  is confirmed as the right chapter, but the specific strongly-NP-hard /
  no-FPTAS sentence is not confirmed at sentence level. *Korte & Schrader 1981*,
  the usual source for the no-FPTAS half, is also unverified.
- *Chekuri, Vondrák, Zenklusen* SODA 2011 — full text unobtainable; do not cite.
  Their FOCS 2010 paper (pp. 575–584, arXiv:0909.4348) **is** verified but gives
  \((1-1/e-\varepsilon)\) for submodular objectives and a randomized PRAS, not a
  deterministic PTAS, for the linear budgeted case. **Never cite CVZ for the
  PTAS — that is [GZ10].**
- The \((R{+}1)\)-approximation of Corollary C.4 — no canonical citation found.
  Print the proof, cite nothing. (Weak negative: the literature sweep was cut
  short before covering surveys.)
