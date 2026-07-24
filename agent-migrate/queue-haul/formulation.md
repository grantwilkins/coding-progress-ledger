# Queue-Haul formulation

Queue-Haul chooses whole LLM sessions to move out of a source power domain
before a deadline. It models source power, migration work, destination
admission, concrete replica packing, and exact migration execution. Source
power is constrained; destination power is reported.

A trustworthy destination profile must be conditional on a pinned serving
class: model revision, weight and KV dtypes, engine and scheduler
configuration, accelerator layout, and tensor/pipeline parallelism. It supplies
measured power and service curves, replay and KV timing, KV block layout,
action power, concurrency limits, transition times, allocatable KV capacity,
and uncertainty cases. The current destination schema does not encode that
complete tuple or multiple uncertainty cases.

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

For cache-aware admission, let
\(p_{s,q,r,\omega}\) be expected prefill accelerator-seconds per wall-second
from a measured cache-conditioned work function
\(\tau_q^P(T^{full},T^{hit},T^{miss})\). Guaranteed reuse may change this work,
but subtracting hit tokens from a cold-prefill throughput is not generally
valid: an uncached suffix still attends to the cached prefix. Prefix reuse does
not reduce generation-phase work for the same full context and output. If the
cache-conditioned profile or protected block identity is unknown, use the
full-prefill approximation \(p_{s,q,r,\omega}=f_s/F_q(T_s)\); this is the
current implementation. An observed cache-hit rate alone is not a residency
guarantee. Decode \(G_q\) is keyed by live sequence length. Using one \(T_s\)
for both cold prefill and decode is the current measured simplification, not a
portable identity.

Over service-demand horizon \(H_s\), the target coordinates are

\[
p_{s,q,r,\omega}=
\frac{1}{H_s}\sum_{j\in requests(H_s)}
\tau_{q,\omega}^P(T_j^{full},T_j^{hit},T_j^{miss}),\qquad
\delta_{s,q,\omega}=
\frac{1}{H_s}\int_0^{H_s}
\frac{g_{s,\omega}(t)}{G_{q,\omega}(T_s(t))}\,dt.
\]

The current implementation approximates these with
\((f_s/F_q(T_s),g_s/G_q(T_s))\). A physical claim with material context growth
must integrate a supported trajectory or use the worst supported rate over the
projected range.

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
transfer additionally requires a matching KV ABI. The method's
workload-affinity rule must also pass.

This is the v1 aggregate candidate. The general candidate is \(c=(s,a,r)\)
with derived \(p(r)\) and source-method-replica path \(path(s,a,r)\).

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
| route-link fluid bound | allocatable link bytes/s \(\times H_m\) |
| pool service facet | replicas \(\times\) policy bound minus baseline work |
| additive pool live KV | replica KV capacity minus baseline KV at \(H_r\) |
| pool migration occupancy | replicas \(\times H_m\) |

The legacy adapter represents destination service with aggregate replay time,
KV-service time, compute load, and resident-KV rows. Its flexible destination
links form one balanced link pool. The architecture path instead preserves
exact pools, routes, per-replica baselines, and service facets. Physical
prefix sharing is nonadditive and is not represented in \(U\); without an
exact block-union packing check, the relaxation should use block-rounded
additive demand. V1 still uses unrounded token equivalents.

Replay uses expected durable-log bytes, replay time, measured replay completion,
and route-switch time. KV transfer uses setup, complete sealed KV blocks, the
slower of network transfer and destination ingestion, measured initial
completion, final missing blocks, partial-tail reconstruction, synchronization,
and route-switch time. Source migration occupancy lasts until commit.

## Destination admission

The logical decision is destination-aware. Let \(z_{s,a,r}\) select source
session \(s\), migration method \(a\), and concrete warm replica \(r\) of type
\(q(r)\), and let \(\pi\) be a fixed nonanticipative migration ordering and
resource-allocation policy. The set is admissible only if there exist \(z,\pi\)
satisfying eligibility, steady placement, and transition constraints:

\[
\operatorname{DestinationOK}(z,\pi)=
\operatorname{Compatible}(z)\land
\operatorname{ServicePack}(z)\land
\operatorname{KVUnion}(z)\land
\operatorname{MigrationSchedule}(z,\pi)\land
\operatorname{ImpactOK}(z).
\]

It is joined to source selection by

\[
\sum_{a,r}z_{s,a,r}=y_s,\quad y_s\in\{0,1\},\qquad
z_{s,a,r}\le E_{s,a,r},\qquad
P_{src}\left(S\setminus
\{s:y_s=1\}\right)\le P_{limit}.
\]

The optimizer may use conservative marginal power gains to choose candidates,
but it reevaluates this exact source condition after integer selection.
Robust semantics require the same \(z,\pi\) to satisfy service, KV, migration,
and impact constraints for every \(\omega\in\Omega\); \(\pi\) may react only to
observed completions, not future case information. The current implementation
has only one central case, so this multi-case quantifier is proposed.
Compatibility fixes the model and tokenizer, durable-log and KV contracts,
hardware, precision, parallel layout, engine configuration, warmness, and
every measured context, workload, bandwidth, and concurrency domain. The
current implementation creates pool candidates \(c=(s,a,p)\), then performs
concrete replica packing. It checks every returned assignment, but the greedy
packer plus cuts is not a complete search for every feasible assignment. Its
current compatibility fingerprint does not yet encode the complete pinned
hardware/runtime tuple or uncertainty cases. Destination hardware need not
equal source hardware: heterogeneous replay is eligible when \(q\) has its own
measured profile and the model/tokenizer/log contract matches; KV additionally
requires its exact transfer ABI/layout contract.

Each destination type \(q\) has measured context-conditioned prefill and decode
rates, allocatable KV capacity \(K_q\), and nonnegative service-facet normals.
An uncertainty case \(\omega\) jointly selects demand plus empirical
\(F_{q,\omega}\), \(G_{q,\omega}\), \(N_{q,\omega}\), and
\(h_{q,\omega}^m\), defining

\[
\mathcal C_{q,\omega}^m=
\{d\ge0:N_{q,\omega}d\le h_{q,\omega}^m\},\qquad
h_{q,\omega}^{normal}\le h_{q,\omega}^{emergency}
\le h_{q,\omega}^{stable}.
\]

The current schema has one central case and common normals; multiple cases are
target semantics.

For replica \(r\), baseline service work \(b_{r,\omega}\), and every declared
uncertainty case \(\omega\), service admission over demand horizon \(H_s\) in
mode \(m\) requires

\[
N_{q(r),\omega}\left(
b_{r,\omega}+
\sum_{s,a}
\left(
\ p_{s,q(r),r,\omega},
\delta_{s,q(r),\omega}
\right)z_{s,a,r}
\right)
\le h_{q(r),\omega}^m.
\]

The rates and context keys are horizon forecasts or conservative bounds over
\(H_s\), not an indefinite steady-state promise.

The empirical set described by these inequalities is the affinity blob of one
pinned replica, not a theoretical FLOP limit. Its baseline-conditioned
residual is
\(\mathcal R_r=\{u\ge0:b_r+u\in\mathcal C_{q(r)}^m\}\).
Ignoring indivisibility, a homogeneous pool has the service-only fluid
relaxation \(\bigoplus_r\mathcal R_r\); actual admission must partition whole
session vectors among replicas and also satisfy KV, migration, route, and
impact constraints. Equal GPU counts therefore need not imply equal available
capacity.

An evidence-robust label additionally requires every case to remain inside
demonstrated conditional inner support. Feasibility that depends on a synthetic
headroom value, interpolation between measured affinity rays, or an assumed
facet shape is sensitivity/possible, not evidence-robust.

The current pool relaxation sums baseline work \(b_p\) and requires

\[
N_q\left(b_p+\sum_{c\in p}d_cy_c\right)
\le |p|h_q^m.
\]

Let \(\mathcal B_r^0\) be physical KV blocks already protected on replica
\(r\), and \(\mathcal B_{s,r,\omega}(H_r)\) the blocks required by an admitted
session through the residency horizon. A shared block must have the exact
pinned-engine cache key and full-block granularity. Exact live-state admission
is

\[
\left|
\mathcal B_r^0\cup
\bigcup_{s,a:z_{s,a,r}=1}\mathcal B_{s,r,\omega}(H_r)
\right|\le K_{q(r)}^{blocks}
\quad\forall r,\omega.
\]

Shared prefix blocks count once; private tails and projected generation remain
per session and are rounded to engine blocks. \(K_q^{blocks}\) is the
allocatable KV budget after model weights, activations, graph captures, and
runtime workspace for the pinned configuration. Every block used to reduce
\(p_{s,q,r,\omega}\) must belong to a separate
\(\mathcal R_{s,r,\omega}^{hit}\) set guaranteed available before that
request's prefill. It may contain baseline protected blocks or blocks installed
by migration only when the schedule enforces install before use. Membership in
the session's future required-block set is not enough. Without identities, the
necessary aggregate additive pruning relaxation is

\[
B_p^0+\sum_{c\in p}
\left\lceil\frac{\widehat T_c(H_r)}{L_q^{block}}\right\rceil y_c
\le |p|K_q^{blocks}.
\]

The current schema exposes token-equivalent capacity and sums unrounded context
tokens. It receives no sharing credit, but still needs block rounding or
one-private-block tail headroom before it is a physical-memory guarantee.

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
\tau_{route}(B_s^{log},path(s,R,r))+
\alpha_{R,q}\tau_{s,R}^{compute+completion,old}(T_s)+\tau_q^{switch},
\]

for replay, and

\[
\tau_{s,K,q}=
\max\left(
\tau_{route}(B_s^{sealed},path(s,K,r)),
\frac{B_s^{sealed}}{C_{ingest,q}}
\right)+c_{K,q}(T_s),
\]

for KV transfer, plus separately measured catch-up and route-switch terms when
the residual does not already contain them. Migration and destination
ingestion may overlap when the measured primitive supports it, so their times
are not blindly added. A load-dependent residual is permitted only within a
domain measured over the migration interval; requested load is never
substituted for achieved load.

The schedule assigns residual route bandwidth after background reservations,
and route time is never below bytes divided by the allocated bottleneck rate.
Geographic routes require measured fixed/per-round latency; the v7 shaped local
route does not identify it. `MigrationSchedule` means one nonanticipative
schedule whose makespan is at most \(H_m\) in every declared case and that
reserves each source stream, route edge, destination ingest/copy engine,
replica migration slot, and measured temporary staging allocation over time.
Aggregate byte and occupancy rows are necessary pruning relaxations, not that
schedule proof.

The recovered v7 data identifies coefficients only for its recorded
concurrency-one request schedules in the measured 16K/10-Gbps and
24K/5-Gbps cells; the fact that at most one request overlaps is not by itself a
workload bound. It supports a replay compute/completion calibration and
additive KV route plus residual timing, but not a load-dependent term. The
current scalar
`LoadedCoefficients` scales the complete duration and therefore cannot encode
this component model without violating the route-time floor. No v7 destination
profile is accepted.

Foreground impact is a provisional method policy rather than a service row.
The paired observations rank KV ahead of replay for latency-sensitive busy
pools, but one arriving request does not establish a tail-SLO bound. A robust
result currently requires idle/drained foreground or an explicit impact bound
inside the exact measured overlap domain. This dynamic predicate is proposed
semantics and is not encoded by the current architecture types.

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
and predictive latency remain out of scope. Measurement-invalid or unbracketed
service evidence and loaded probes without recorded migration-interval state
also hard-fail during profile reduction. Measurement requirements and open
evidence are tracked in `DATA_TO_COLLECT.md`.
