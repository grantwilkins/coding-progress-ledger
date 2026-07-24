# Queue-Haul formulation

Queue-Haul chooses whole LLM sessions to move out of a source power domain
before a deadline. It models source power, migration work, destination
admission, concrete replica packing, and exact migration execution. Source
power is constrained; destination power is reported.

A model profile supplies measured power and service curves, replay and KV
timing, KV block layout, action power, concurrency limits, transition times,
resident-KV capacity, and uncertainty cases.

## Sessions

A session is the unit of selection, migration, routing, and KV residency. A
session \(s\) contains:

- its current serving instance and active or cold state;
- current context \(T_s\), durable-log bytes \(B_s\), and movable flag;
- expected prefill and decode token rates \(f_s,g_s\);
- optional expected context growth \(\gamma_s\) in tokens/s;
- optional sampled requests, each with an arrival gap, prompt tokens, and
  output tokens; and
- for a cold session, its probability of waking during the planning horizon.

An active session consumes expected service load and resident KV. In the
legacy scalar model its normalized load is

\[
\ell_s=\frac{f_s}{F}+\frac{g_s}{G}.
\]

A pool-aware destination instead retains the two-dimensional work vector

\[
d_{s,q}=
\left(\frac{f_s}{F_q(T_s)},\frac{g_s}{G_q(T_s)}\right)
\]

for destination type \(q\). A cold session has \(f_s=g_s=0\), consumes no live
KV until it wakes, and can use only replay on request. An active session can
use replay or KV transfer. Selection considers only movable sessions whose
serving instance is entirely inside the local source-power scope.
The workload-direction domain is checked using
\(d_{s,q}^{prefill}/(d_{s,q}^{prefill}+d_{s,q}^{decode})\), not the raw input
token fraction.

Planning never reads sampled future request times or sizes. For a planning
horizon \(h\), it conservatively materializes expected active-session state as

\[
\widehat T_s(h)=\left\lceil T_s+\gamma_s h\right\rceil,\qquad
\widehat B_s(h)=\left\lceil B_s\frac{\widehat T_s(h)}{T_s}\right\rceil.
\]

Expected service rates remain fixed during a plan. Sampled requests are used
only by execution and evaluation.

## Time horizons and power

Let \(D\) be the power deadline, \(C\) controller delay, and \(W\) the measured
trailing power window. Migration admission uses

\[
H_m=D-C-W.
\]

A pool-aware architecture may set a separate residency horizon \(H_r\);
otherwise \(H_r\) is the simulated end time after controller delay. Separating
the horizons prevents short migration deadlines from understating long-lived
destination KV.

For initial source load \(L\), moving session \(s\) has conservative marginal
source-power gain

\[
w_s=P(L)-P(L-\ell_s).
\]

Because the measured power curve is nondecreasing and concave, summing these
initial marginal gains is a conservative lower bound on the reduction from
moving several sessions. The exact power model is reevaluated after integer
selection. Source power is credited only when a migration commits, not when
its background copy finishes.

The profile also supplies total action power at each measured concurrency for
replay, KV transfer, catch-up, and node transitions. The simulator updates that
total when concurrency changes; it does not multiply a measured concurrent
total by the number of sessions.

## Candidates and resources

The legacy planner has replay and KV-transfer candidates for each active
session. With a `DestinationArchitecture`, a candidate is

\[
c=(s,a,p),
\]

where \(a\) is replay or KV transfer and \(p\) is a compatible destination
pool. Replay requires matching model, tokenizer, and durable-log contract. KV
transfer additionally requires a matching KV ABI.

Every candidate records source-power gain, migration work and duration, exact
route bytes, destination service work, and resident KV tokens. A candidate
outside the measured context, bandwidth, workload-direction, or loaded-profile
range hard-fails or is excluded.

The common sparse relaxation is

\[
Ax\le\mathbf1,\qquad Ux\le\mathbf1,\qquad 0\le x\le1,
\]

where \(A\) permits at most one candidate per session and each row of \(U\) is
normalized by its capacity. The resource rows are:

| Resource | Capacity |
|---|---|
| source-instance migration | source streams \(\times H_m\) |
| exact route link | link bytes/s \(\times H_m\) |
| pool service facet | replicas \(\times\) policy bound minus baseline work |
| pool live KV | replica KV capacity minus baseline KV at \(H_r\) |
| pool migration occupancy | replicas \(\times H_m\) |

The legacy adapter represents destination service with aggregate replay time,
KV-service time, compute load, and resident-KV rows. Its flexible destination
links form one balanced link pool. The architecture path instead preserves
exact pools, routes, per-replica baselines, and service facets.

Replay uses expected durable-log bytes, replay time, measured replay completion,
and route-switch time. KV transfer uses setup, complete sealed KV blocks, the
slower of network transfer and destination ingestion, measured initial
completion, final missing blocks, partial-tail reconstruction, synchronization,
and route-switch time. Source migration occupancy lasts until commit.

## Destination admission

Each destination type \(q\) has measured context-conditioned prefill and decode
rates, KV capacity \(K_q\), and common nonnegative service-facet normals
\(N_q\). Its nested policy envelopes satisfy

\[
h_q^{normal}\le h_q^{emergency}\le h_q^{stable}.
\]

For pool \(p\), baseline service work \(b_p\), and selected candidates \(y_c\),
admission in mode \(m\) requires

\[
N_q\left(b_p+\sum_{c\in p}d_cy_c\right)
\le |p|h_q^m.
\]

Live-state admission separately requires

\[
K_p^0+\sum_{c\in p}\widehat T_c(H_r)y_c\le |p|K_q.
\]

Normal admission is attempted first; emergency is tried only if normal cannot
meet the source-power target. Stable is not an admission mode. It is the outer
hard limit independently checked on each concrete replica during execution.
A baseline already outside the requested admission envelope makes that pool
unavailable.

Baseline work and KV may come from explicit per-replica fields or destination
background sessions, but never both.

A migration model preserves exact transport lower bounds and calibrates only
runtime-dependent work. Its component semantics are:

\[
\tau_{s,R,q}=
\frac{B_s^{log}}{b_{route}}+
\alpha_{R,q}\tau_{s,R}^{compute,old}(T_s)+c_{R,q},
\]

for replay, and

\[
\tau_{s,K,q}=
\max\left(
\frac{B_s^{sealed}}{b_{route}},
\frac{B_s^{sealed}}{C_{ingest,q}(u)}
\right)+c_{K,q}(T_s,u),
\]

for KV transfer, plus separately measured catch-up and route-switch terms when
the residual does not already contain them. Migration and destination
ingestion may overlap when the measured primitive supports it, so their times
are not blindly added. A load-dependent residual is permitted only within a
domain measured over the migration interval; requested load is never
substituted for achieved load.

The compact v7 data identifies only exploratory low-work coefficients. It
supports a replay compute/completion calibration and additive KV route plus
residual timing, but not a load-dependent term. The current scalar
`LoadedCoefficients` scales the complete duration and therefore cannot encode
this component model without violating the route-time floor. No v7 destination
profile is accepted.

## Greedy

Greedy uses the same candidate and normalized resource matrices as the LP. It
first finds one initially cheapest legal candidate per session and sums those
columns to estimate demand \(d_r\) for each resource. Counting one candidate
per session avoids double-counting mutually exclusive replay, KV, or pool
alternatives. The approximate resource price is

\[
\pi_r=\max(1,d_r).
\]

Candidate \(c\) receives score

\[
\operatorname{score}(c)=
\frac{w_c}{\sum_r \pi_r u_{r,c}}.
\]

Sessions are sorted once by their best candidate score. In that order, greedy
takes the highest-scoring candidate that preserves every remaining capacity
and packing cut. The legacy path stops on exact modeled power; the pool-aware
path stops on accumulated conservative gain. Both reevaluate exact source power
after integer selection. This approximates LP dual pricing while remaining
approximately \(O(N\log N)\).

## LP

The default LP minimizes migration work while meeting requested conservative
power reduction \(\Delta P\):

\[
\begin{aligned}
\operatorname{minimize}\quad &\sum_c m_cx_c\\
\operatorname{subject\ to}\quad
&\sum_c w_cx_c\ge\Delta P,\\
&Ax\le\mathbf1,\quad Ux\le\mathbf1,\quad 0\le x\le1.
\end{aligned}
\]

If the target is infeasible, a separate solve maximizes conservative power
shed. The pool-aware path then minimizes work at that maximum. The result is
valid best effort with `failure_reason="target_unmet"` and an explicit watt
shortfall, not successful curtailment. The legacy `lp_peak_first` and
`lp_work_first` variants retain alternative objective orders for comparison.

Fractional results are rounded deterministically to whole sessions while
preserving every resource row. The legacy path then balances selected sessions
across destination instances. The pool-aware path retains the chosen pool and
packs each session onto a concrete replica by worst service, KV, and migration
pressure.

Aggregate pool feasibility does not imply replica feasibility. If deterministic
packing fails, the failed candidate set becomes a cut and selection is solved
again:

\[
\sum_{c\in C_k}x_c\le |C_k|-1.
\]

Small cases have an exact replica-assignment oracle. Plans report packing
repair count and time.

## How sessions are updated during execution

The simulator maintains mutable execution state separately from the immutable
input session:

1. **Requests.** A request waits if its current serving instance is busy or its
   session is paused. At request completion, prompt plus output tokens are
   added to the session context and to resident KV on the currently serving
   instance. Durable-log accounting grows in proportion to context. Request
   events affect migration state and timing but do not yet change the
   expected-rate power model.
2. **Snapshot preparation.** A move snapshots current context when its source
   stream starts. Replay transfers and reconstructs the corresponding durable
   log. KV transfer sends only complete sealed blocks; an unsealed tail is
   never copied as a block.
3. **Updates during copying.** Replay catches up log growth after quiescence.
   After the initial KV snapshot is ready, each completed request exposes newly
   sealed blocks as an append transfer. Those appends use the same destination
   KV queue and shared links as other copies.
4. **Quiescence and catch-up.** At the planned quiescence time, the source stops
   admitting work for the session and waits for any active request. It then
   waits for outstanding appends, transfers missing sealed blocks, reconstructs
   the final partial tail by replay, and performs measured synchronization.
   Background KV pacing does not cap this paused final catch-up.
5. **Commit.** After route-switch time, commit atomically changes session
   routing and power ownership. Current resident KV is removed from the source
   and added to the selected destination. Pending requests can then start
   there. No source-power reduction is credited before this point.
6. **Cold sessions.** Replay-on-request switches the route without creating
   live KV. The first later request triggers durable-log replay, installs the
   reconstructed context at the destination, and then begins service.

Source-instance stream limits serialize or overlap moves as configured.
Transfers share every common network link with work-conserving max-min rates.
KV copies additionally queue per destination and share measured destination
ingestion capacity. If the requested final source state is sleep or off, a node
transitions only after every dependent session has committed.

## Validation and scope

The event simulator is authoritative because aggregate resource feasibility
does not guarantee a valid schedule through serial stages and shared queues. A
plan is feasible only when:

1. exact modeled source power after integer selection is at or below the limit;
2. the trailing-window modeled source power at \(D\) is at or below the limit;
3. every selected migration commits by \(D\); and
4. any requested sleep or off transition finishes by \(D\).

Experiment acceptance additionally requires every request observed by \(D\) to
start by \(D\). This checks routing readiness, not end-to-end request latency.

The legacy scalar LP and greedy support active sessions, the central profile
case, one aggregate destination pool, and an awake final state. Passing a
`DestinationArchitecture` enables multiple pools and routes, normal/emergency
admission, concrete replica packing, and stable-envelope validation; its
current planning scope remains active sessions, the central case, and an awake
final state. Random planning retains cold-session and replay-on-request
coverage.

Unsupported context, workload direction, destination load, bandwidth,
compatibility, topology, concurrency, or profile cases hard-fail. Continuous
destination scheduling, request-level dynamic power, replanning, model loading,
and predictive latency remain out of scope. Unbracketed or non-monotone
service evidence and loaded probes without migration-interval overlap also
hard-fail during profile reduction. Measurement requirements and open evidence
are tracked in `DATA_TO_COLLECT.md`.
