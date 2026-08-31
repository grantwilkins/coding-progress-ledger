# Queue-Haul formulation

> This document is the **implementation spec**: it tracks what the code does,
> what it approximates, and where target semantics differ from current
> behaviour. The **paper-facing** statement — the clean LP for the one-site and
> many-site settings, the structural results that justify solving it greedily,
> and the full constant/provenance ledger — is `formulation_nsdi.md`. Keep the
> two in sync: this file is authoritative for implementation status, that file
> is authoritative for the mathematical statement and its citations.

Queue-Haul chooses whole LLM sessions to move out of a source power domain
before a deadline. Its primary result is required destination residual resources
versus source watts shed; this needs a pinned destination class but no concrete
destination inventory. Concrete admission and replica packing are optional
later comparisons. Destination power caps and pricing are outside the model.

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
\(p_{s,q,r,\omega}\) be expected pinned-replica service-seconds per wall-second
from a measured cache-conditioned work function
\(\tau_q^P(T^{full},T^{hit},T^{miss})\). Guaranteed reuse may change this work,
but subtracting hit tokens from a cold-prefill throughput is not generally
valid: an uncached suffix still attends to the cached prefix. Prefix reuse does
not reduce generation-phase work for the same full context and output. If the
cache-conditioned profile or protected block identity is unknown, the current
implementation uses \(p_{s,q,r,\omega}=f_s/F_q(T_s)\). This is an
append-token/cold-rate normalization coordinate, not a conservative physical
cached-prefill bound. It transfers only within the empirically observed cache,
affinity, context, and concurrency domain. An observed cache-hit rate alone is
not a residency guarantee. Decode \(G_q\) is keyed by live sequence length.
Using one \(T_s\) for both cold prefill and decode is the current measured
simplification, not a portable identity.

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

For the v7 private-prefix contract with block size \(L_q^{block}\), request
\(j\) is cache-valid only when

\[
cached_j\le
\left\lfloor\frac{prompt_j-append_j}{L_q^{block}}\right\rfloor
L_q^{block}.
\]

The physical execution is the contamination unit. Any request exceeding this
bound reuses the nominal new append and excludes the whole execution from
service-capacity fitting. A forensic geometry audit finds five
private-prefix-consistent executions and 60 append-hot complete-work
executions. Because the archive lacks stream-completion evidence, the five are
descriptive sensitivity anchors rather than admissible service points. One
colder under-hit is excluded after a likely silent prewarm failure. The
persistent campaign process means these are not statistically independent
replications.

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

The implemented `ExpectedPower.marginal` is richer than this expression: it
works per GPU slot, and when \(s\) is the last remaining dependent of a node it
also credits that node's transition to the requested final state. So \(w_s\) can
carry a whole-node sleep or shutdown term.

The profile also supplies total action power at each measured concurrency for
replay, KV transfer, catch-up, and node transitions. The simulator updates that
total when concurrency changes; it does not multiply a measured concurrent
total by the number of sessions.

## Candidates and resources

The requirement frontier has replay and KV-transfer candidates for each active
session:

\[
c=(s,a),\qquad a\in\{\text{replay},\text{KV}\}.
\]

The solver selects both actions jointly, so a plan may contain both methods
while the session-incidence row permits at most one action per session. Replay
requires matching model, tokenizer, and durable-log contract. KV transfer
additionally requires a matching KV ABI. The method's workload-affinity rule
must also pass.

Optional concrete admission refines this to \(c=(s,a,r)\) for a supplied
replica inventory.

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
| logical-WAN fluid bound | effective bytes/s \(\times H_m\) |
| pool service facet | replicas \(\times\) policy bound minus baseline work |
| additive pool live KV | replica KV capacity minus baseline KV at \(H_r\) |
| pool migration occupancy | replicas \(\times H_m\) |

Migrations may start concurrently and share every link on their route. The WAN
byte row is a fluid relaxation; the event executor validates the concurrent
route schedule.

Service and migration may overlap and are independently budgeted; both rows
must fit. The pool path has no destination ingest or replay-concurrency row: the legacy
`max_destination_replays` and `max_destination_kv_streams` rows were replaced by
one method-agnostic migration-occupancy row of capacity \(\lvert p\rvert H_m\).

The legacy adapter represents destination service with aggregate replay time,
KV-service time, compute load, and resident-KV rows. Its flexible destination
links form one balanced link pool. The architecture path instead preserves
exact pools, routes, per-replica baselines, and service facets. Physical
prefix sharing is deliberately uncredited in v1; the target relaxation uses
block-rounded additive demand.

Block rounding is implemented by the requirement-frontier and pool-aware paths.
The pool candidate charges \(\lceil\widehat T_s(H_r)/L_q\rceil\) blocks against a
capacity of \(\lvert p\rvert\lfloor K_q/L_q\rfloor\), with per-replica and
per-background-session baselines rounded independently; only the legacy scalar
adapter still sums unrounded \(\widehat T_s(H_m)\) tokens against
`kv_capacity_tokens`. Note
also that the paging block \(L_q\) (16 tokens) and the KV **transfer** block
(256 tokens, `block_bytes` 12,582,912) are different measured quantities.

Replay uses expected durable-log bytes, replay time, measured replay completion,
and route-switch time. KV transfer uses setup, complete sealed KV blocks, the
slower of network transfer and destination ingestion, measured initial
completion, final missing blocks, partial-tail reconstruction, synchronization,
and route-switch time. Source migration occupancy lasts until commit.

## Destination admission

Concrete admission is optional. Let \(z_{s,a,r}\) select source
session \(s\), migration method \(a\), and concrete warm replica \(r\) of type
\(q(r)\), and let \(\pi\) be a fixed nonanticipative migration ordering and
resource-allocation policy. The set is admissible only if there exist \(z,\pi\)
satisfying eligibility, steady placement, and transition constraints:

\[
\operatorname{DestinationOK}(z,\pi)=
\operatorname{Compatible}(z)\land
\operatorname{ServicePack}(z)\land
\operatorname{KVPrivate}(z)\land
\operatorname{MigrationSchedule}(z,\pi).
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
and impact constraints for every \(\omega\in\Omega\). Here \(\pi\) is a
history-dependent policy selected ex ante; it maps observed completions to
actions but cannot read future case information. Define
\(E_{s,a,r}=\min_\omega E_{s,a,r,\omega}\), and use a source-power lower bound
valid simultaneously across declared source cases. The current implementation
has only one central case, so these robust quantifiers are proposed.
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

The minimal profile-conditioned specialization uses one scalar service-work
facet rather than a latency model. For a pinned hardware/runtime, incumbent
profile, added-cohort family, service horizon, and TTFT/TPOT policy, define

\[
W_0+\Delta W=
b_f+b_d+\sum_s\left(w_{s,f}+w_{s,d}\right)x_s
\le W_{safe}.
\]

This is exactly the existing facet `N=((1,1),)`; no new solver row or variable
is required. `W_safe` is the largest tested point whose complete repeat
envelope satisfies both P90 latency targets, exact completion, drain, and the
declared queue-stability rule. Exceeding it means that admission is not
certified, not that the workload is physically impossible. It predicts
neither TTFT nor TPOT and is not GPU utilization. The work coordinates and
bound are inseparable from their measured prefill/decode rates and pinned
serving stack; a bound must be rescaled or remeasured before use with another
normalization.

For a fixed request class, `ProfileRateLimit` performs the opt-in conversion
from a measured total-rate boundary `Lambda_safe` to this existing row. It
computes per-request work `e=(P/F_q(T),O/G_q(T))`, baseline `lambda_0 e`, and
bound `Lambda_safe (e_f+e_g)`. It hard-fails another destination type, context,
or prefill/decode ray. It does not alter destination schemas, planner rows, or
historical profiles; callers must explicitly consume the returned conversion.

For the completed 4K A100 service class, `W=0.50` is a tested safe floor under
the campaign rates `F=16758.928` and `G=3597.591` tokens/s. The incumbent uses
approximately `W_0=0.25`. Three mixes across three fresh restart blocks passed
at `W=0.50`. At `W=0.70`, every held-out mix remained comfortably inside both
latency SLOs, but each passed the every-repeat stability contract in only two
of three blocks. The evidence therefore brackets, but does not identify, a
higher stable bound: `0.50 <= W_safe < 0.70` under this contract. `W=0.50`
must not be installed as a production cap if it makes the intended workload
infeasible; one targeted `W=0.60` confirmation is required to raise it. The
artifact remains non-promoting, and no historical destination profile or
planner default is changed.

An evidence-robust label requires an accepted envelope and every case inside
its support. The five v7 private-prefix-consistent executions provide
descriptive sensitivity anchors only. The colder under-hit is not promoted
through an untested cache-monotonicity assumption. Feasibility that depends on
a legacy anchor, synthetic headroom value, interpolation between affinity rays,
or an assumed facet shape is sensitivity/possible.

The current pool relaxation sums baseline work \(b_p\) and requires

\[
N_q\left(b_p+\sum_{c\in p}d_cx_c\right)
\le |p|h_q^m.
\]

Target v1 gives no cross-session sharing credit. Exact private-state admission is

\[
B_{r,\omega}^0+
\sum_{s,a}
\left\lceil
\frac{\widehat T_{s,\omega}(H_r)}{L_{q(r)}^{block}}
\right\rceil z_{s,a,r}
\le K_{q(r)}^{blocks}
\quad\forall r,\omega.
\]

\(B_{r,\omega}^0\) and \(K_q^{blocks}\) are physical block counts after model
weights, activations, graph captures, and runtime workspace. Each session's
private history and projected growth are block-rounded independently.
Prefix-compute credit applies only to that session's history installed and
protected until the credited request begins. V7 observes request-time reuse but
does not prove such retention under arbitrary pressure. Without a runtime
reservation, prospective prefix-compute credit is sensitivity-only. The
necessary aggregate pool pruning relaxation is

\[
B_p^0+\sum_{c\in p}
\left\lceil\frac{\widehat T_c(H_r)}{L_q^{block}}\right\rceil x_c
\le |p|K_q^{blocks}.
\]

The legacy schema exposes token-equivalent capacity and sums unrounded context
tokens; it receives no sharing credit and still needs block rounding or one
block of tail headroom per resident session before it is a physical-memory
guarantee. The pool path implements the block-rounded form above, with the one
residual gap that its capacity is \(\lfloor K_q/L_q\rfloor\) — a floor division
of a measured token capacity — rather than a natively measured block count.

A later shared-KV extension may replace the private sum with an exact union of
protected pinned-engine block keys. Evictable idle cache entries receive no
admission credit.

Normal admission is attempted first; emergency is tried only if normal cannot
meet the source-power target. Stable is not an admission mode. It is the outer
hard limit independently checked on each concrete replica during execution.
A baseline already outside the requested admission envelope makes that pool
unavailable.

Baseline work and KV may come from explicit per-replica fields or destination
background sessions, but never both.

A migration model preserves exact transport lower bounds and calibrates only
runtime-dependent work. Log and sealed-KV bytes are evaluated at the snapshot
selected by \(\pi\); append and catch-up bytes remain separate phases. Its
component semantics are:

\[
\tau_{s,R,q}=
\tau_{route}(B_s^{log},path(s,R,r))+
\alpha_{R,q}\tau_{s,R}^{compute+completion,old}(T_s)+\tau_q^{switch}
+\mathtt{route\_rtt\_s},
\]

for replay, and

\[
\tau_{s,K,q}=
\max\left(
\tau_{route}(B_s^{sealed},path(s,K,r)),
\frac{B_s^{sealed}}{C_{ingest,q}}
\right)+c_{K,q}(T_s)+\mathtt{route\_rtt\_s},
\]

for KV transfer, plus separately measured catch-up and route-switch terms when
the residual does not already contain them. Migration and destination
ingestion may overlap when the measured primitive supports it, so their times
are not blindly added. A load-dependent residual is permitted only within a
domain measured over the migration interval; requested load is never
substituted for achieved load.

The fitted v7 KV residual is `observed - route_floor`; it can be used in the
explicit `max(route, ingest)` form only where route time is the measured
bottleneck. Otherwise it must be redefined or refit.

The pool KV path now implements
`max(route, bytes/ingest_rate) + residual + catch_up` (commit `ef435092`).
Separately, the pool replay branch silently
substitutes the minimum tabulated rate when context falls outside the measured
replay curve, where the legacy path raises; that contradicts the hard-fail rule
stated above and should raise.

V1 has one logical WAN route. Its bandwidth term is bytes divided by effective
bandwidth; `route_rtt_s` is then added once after the transfer/ingest pipeline,
exactly one fixed P50 RTT per migration action with no RTT/2 conversion. The
central bandwidth is 5 Gbps; sensitivity
uses 1/5/10 Gbps and P50 RTTs of 10/60/90/150/240 ms. No TCP or multi-edge
topology model is required. `MigrationSchedule` means one nonanticipative
schedule whose makespan is at most \(H_m\) in every declared case and that
reserves each logical route, destination ingest/copy engine, and replica
migration slot. Any temporary staging allocation must become an
explicit measured row; current code has none. Aggregate byte and occupancy rows
are necessary pruning relaxations, not that schedule proof.

The recovered v7 data identifies coefficients only for its recorded
concurrency-one request schedules in the measured 16K/10-Gbps and
24K/5-Gbps cells; the fact that at most one request overlaps is not by itself a
workload bound. It supports a replay compute/completion calibration and
additive KV route plus residual timing, but not a load-dependent term. The
current scalar
`LoadedCoefficients` scales the complete duration and therefore cannot encode
this component model without violating the route-time floor. No v7 destination
profile is accepted.

Service and migration may overlap when both independent budgets fit. The paired
foreground observations remain descriptive and are not an extra interference
constraint.

Operational admission additionally requires an accepted evidence status, a
full pinned runtime and warm/healthy attestation, a fresh live-state snapshot
held by a lease through commit, per-session KV block rounding, a supported
service horizon. The current architecture implementation does not enforce all
of these gates, so its successful placements remain descriptive or
`sensitivity/possible`.

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

The legacy path sorts sessions once by their best candidate score and tries that
session's methods in score order; the pool path sorts candidates globally, so a
session's alternative pools are scattered through the order rather than tried
consecutively. In either order, greedy takes the highest-scoring candidate that
preserves every remaining capacity. Greedy has no cut awareness — cuts exist
only in the packing repair loop after selection. The legacy path stops on exact
modeled power; the pool-aware path stops on accumulated conservative gain. Both
reevaluate exact source power after integer selection. Prices and scores are
computed once, before the loop, and are never recomputed as rows saturate. If
the stable-order pass misses the target, footprint-first and gain-first scans
reuse those prices, followed by a session-gain scan that chooses each option by
the lowest resulting peak utilization. The highest-gain miss is retained, while
a successful stable-order result is never replaced. No greedy path invokes an
LP or MILP.

This approximates LP dual pricing while remaining approximately
\(O(N\log N)\). `formulation_nsdi.md` §C.3 derives what it is approximating and
states its failure modes, chiefly that flooring the price at one inverts the
ordering on uncontested rows. The `1e-12` divisor guard would let a candidate
with an all-zero resource column score as infinite; that column cannot arise
today, because every candidate has a positive duration and therefore a positive
source and migration entry, and cold sessions are filtered out of the pool
table before candidates are built.

## LP

The default LP minimizes predicted action duration while meeting requested conservative
power reduction \(\Delta P\):

\[
\begin{aligned}
\operatorname{minimize}\quad &\sum_c t_cx_c\\
\operatorname{subject\ to}\quad
&\sum_c w_cx_c\ge\Delta P,\\
&Ax\le\mathbf1,\quad Ux\le\mathbf1,\quad 0\le x\le1.
\end{aligned}
\]

Endpoint replica-seconds remain in \(U\) as a physical capacity coefficient;
they are not reused as the objective when route and endpoint stages overlap.
If the target is infeasible, a separate solve maximizes conservative power
shed. The pool-aware path then minimizes duration at that maximum. The result is
valid best effort with `failure_reason="target_unmet"` and an explicit watt
shortfall, not successful curtailment. The `lp_peak_first` and `lp_work_first`
variants retain alternative objective orders on the legacy path only; the pool
path dispatches every `lp*` solver name to the same objective, so all three
produce identical plans there.

The LP is solved by an interior-point conic solver without crossover, so the
returned point is not a vertex and essentially every coordinate is fractional.
The \(\le 2R\) fractionality bound of `formulation_nsdi.md` §C.2 applies to
basic solutions and is therefore not realized today; this is why rounding is a
sweep over all columns rather than a repair of a handful.

Fractional results are rounded deterministically to whole sessions while
preserving every resource row. The legacy path then balances selected sessions
across destination instances. The pool-aware path retains the chosen pool and
packs each session onto a concrete replica by worst service, KV, and migration
pressure.

Aggregate pool feasibility does not imply replica feasibility. The target repair
is a cut on the failed candidate set, after which selection is solved again:

\[
\sum_{c\in C_k}x_c\le |C_k|-1.
\]

The implementation does not add this cut or re-solve the LP. It orders each
pool's selected actions from best to worst total normalized resource per watt,
keeps each feasible placement fixed, and rejects an action when no replica can
fit it. The result is maximal only under that frozen greedy order; reassignment
could admit a different set. Small cases have an exact replica-assignment oracle
for comparison. Plans report rejected-action count and complete repair-pass
time as packing repair count and time. Later deadline repair only removes
actions from this valid assignment; it never repacks the retained subset.

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

The event simulator is authoritative only for deterministic schedules under
fixed healthy modeled resources; aggregate resource feasibility alone does not
guarantee a valid schedule through serial stages and shared queues. It has no
failure, lease-expiry, or rollback model. Execution must revalidate the lease,
retain source ownership until commit, and abort the destination attempt on a
pre-commit failure. Within that scope, a plan is feasible only when:

1. exact modeled source power after integer selection is at or below the limit;
2. the trailing-window modeled source power at \(D\) is at or below the limit;
3. every selected migration commits by \(D\); and
4. any requested sleep or off transition finishes by \(D\).

Experiment acceptance additionally requires every request observed by \(D\) to
start by \(D\). This checks routing readiness, not end-to-end request latency.

The requirement frontier evaluates source targets at
10/25/50/75/90/100% of maximum shed.
The legacy scalar LP and greedy support active sessions, the central profile
case, one aggregate destination pool, and an awake final state. Passing a
`DestinationArchitecture` enables multiple pools and routes, normal/emergency
admission, concrete replica packing, and stable-envelope validation; its
current planning scope remains active sessions, the central case, and an awake
final state. Random planning retains cold-session and replay-on-request
coverage.

Malformed type-level context, workload, compatibility, topology, concurrency,
or profile inputs hard-fail. Candidate-local unsupported method/path cases are
omitted by the current prototype and are not yet reported structurally.
Continuous destination scheduling, request-level dynamic power, replanning,
model loading, and predictive latency remain out of scope.
Measurement-invalid or unbracketed service evidence and loaded probes without
recorded migration-interval state hard-fail during profile reduction.
Measurement requirements and open evidence are tracked in
`DATA_TO_COLLECT.md`.
