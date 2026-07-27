# Queue-Haul NSDI formulation

Queue-Haul moves stateful inference sessions away from a source power domain
before a deadline. It chooses which sessions to move, whether to replay context
or transfer KV state, and which compatible destination pool accepts each
session.

The primary system output is an executable migration schedule:

> Given measured handoff primitives and advertised compatible destination-pool
> budgets, report how much source accelerator power can be shed by a deadline
> and the executable migration schedule that achieves it.

The schedule names the whole sessions to move, replay or KV transfer for each
session, destination pool, migration start and finish, commit, first-token
completion, source shutdown where applicable, and resource use, slack, debt,
recovery, achieved shed, and unmet shed. The requirement frontier summarizes
these schedules across requested shed targets; it does not replace them.

The source model is hardware-specific because the objective is watts. A
destination is an operator contract expressed in measured resource units.
Queue-Haul does not infer how a provider creates headroom.

Every input and result is labelled **measured**, **fitted**, **assumed**, or
**simulated**. An assumed input is a sensitivity, never an admission guarantee.

## 1. Event and handoff

At time zero, a source receives an accelerator-power limit
\(P_{\mathrm{lim}}\), deadline \(D\), and trailing measurement window \(W\).
Controller delay is \(C\), so migrations have

\[
H = D-C-W
\]

seconds to switch routing. The source must remain below the limit throughout
the final \(W\)-second window.

Queue-Haul performs a request-boundary live handoff:

1. prepare state while the source still owns the session;
2. quiesce at a request boundary;
3. copy or reconstruct the final delta;
4. switch routing; and
5. receive the first post-switch token from the destination.

This is not arbitrary mid-token migration. Landing succeeds at the first
post-switch token. The destination pool accepts the session's declared ongoing
service and KV demand after that point. Long-term fleet management and return
migration are outside the claim.

If every session leaves an accelerator, Queue-Haul may shut it down. An off
accelerator contributes 0 W and must reach that state before the final power
window. V1 does not plan around this bonus: it credits only the conservative
active-load power reduction, then applies any shutdown gain after selection.

## 2. Source power

Session \(j\) on source instance \(i\) has expected prefill and decode rates
\(f_j\) and \(g_j\). For measured reference rates \(F_i\) and \(G_i\),

\[
\ell_j=f_j/F_i+g_j/G_i,\qquad
L_i=\sum_{j\in\mathcal S_i}\ell_j.
\]

The current measured curve is

\[
P_i(L)=P_{0,i}+\Delta P_i\frac{\kappa_iL}{1+\kappa_iL}
\]

inside its measured domain. For a selected set \(M_i\), exact predicted shed is

\[
w_i(M_i)=P_i(L_i)-P_i\left(L_i-\sum_{j\in M_i}\ell_j\right).
\]

The planner uses a conservative additive lower bound \(w_j\) for selection and
reevaluates the nonlinear curve after selecting whole sessions. Final held-out
group removals must satisfy

\[
P_i(L_i)-P_i(L_i-\ell(M_i))\ge \sum_{j\in M_i}w_j.
\]

Any violation is a failed power model, not noise to suppress.

## 3. Destination pool contract

A destination pool is

\[
p=(\text{site},\text{compatible serving type}).
\]

The type pins model revision, tokenizer, durable-log contract, KV ABI and
dtype, hardware, parallel layout, engine, scheduler configuration, and warm
health state. Replay requires the model, tokenizer, and durable-log contract.
KV transfer additionally requires the exact KV contract. Compatibility is a
Boolean eligibility check, not a price or capacity.

The public candidate is

\[
c=(j,a,p),\qquad a\in\{R,K\}.
\]

The destination manager, not Queue-Haul, places an accepted session on a
replica. Any internal fragmentation has already been removed from the pool's
advertised budget.

Each pool advertises:

- normal serving capacity;
- stable serving capacity;
- an event admission limit for ongoing prefill and decode work;
- a temporary service-debt budget in replica-seconds;
- replay reconstruction and KV-ingest capacity;
- usable live-KV capacity in physical blocks;
- route bandwidth and queued-byte capacity; and
- the evidence status and validity range for every value.

Normal capacity grounds the site's usual latency policy. Stable capacity is the
largest measured rate with non-growing work and no failures. The operator may
advertise event flex above normal capacity, but never above stable capacity.
Queue-Haul reports work and queues; it does not predict TTFT or TPOT from the
flex percentage.

For sensitivity \(m\in\{0,.05,.10,.20\}\), event capacity is

\[
C^{\mathrm{event}}_{p,r}
=\min(C^{\mathrm{normal}}_{p,r}
     +mC^{\mathrm{stable}}_{p,r},
     C^{\mathrm{stable}}_{p,r}).
\]

Five percent therefore means five percentage points of measured stable pool
capacity. It never means five percent more sessions.

## 4. Candidate demand

Candidate \(c=(j,a,p)\) carries:

\[
(w_j,\ t_c,\ b_c,\ s_c,\ m_c,\ k_c).
\]

- \(w_j\): conservative source accelerator watts removed.
- \(t_c\): measured or conservatively fitted handoff duration.
- \(b_c\): route bytes.
- \(s_c\): ongoing prefill/decode service work.
- \(m_c\): temporary reconstruction or ingest work.
- \(k_c\): block-rounded live-KV demand.

Replay sends compact durable context and consumes destination reconstruction
work. KV transfer sends sealed KV blocks and consumes route and ingest work.
An unsealed tail is reconstructed during catch-up.

For effective route rate \(B_p\), fixed route RTT \(\tau_p\), destination
prefill rate \(\rho_p(T)\), KV-ingest rate \(\mu_p\), and fitted residuals:

\[
t^R_c =
\frac{C_j}{B_p}+\tau_p+
\alpha_p\frac{T_j}{\rho_p(T_j)}
+r^R_p+\tau^{\mathrm{catch}}_j+\tau^{\mathrm{switch}}_p,
\]

\[
t^K_c =
\max\left(\frac{K_j}{B_p},\frac{K_j}{\mu_p}\right)+\tau_p+
r^K_p+\tau^{\mathrm{catch}}_j+\tau^{\mathrm{switch}}_p.
\]

The max preserves measured overlap between transfer and ingest. Exact route
time is never hidden inside a fitted slowdown.

Ongoing work is expressed in replica-equivalent service:

\[
s_c=\left(f_j/F_p(T_j),\ g_j/G_p(T_j)\right).
\]

One replica-second is the work one pinned replica performs in one second.
Integrated pools use their measured rule for sharing prefill and decode.
Disaggregated pools expose separate prefill and decode budgets.

Live KV is rounded for each session:

\[
k_c=\left\lceil T_j/L_p\right\rceil
\]

blocks. V1 gives no cross-session prefix-sharing credit.

## 5. Steady admission, transition debt, and recovery

Let \(b_{p,r}\) be existing pool work, \(C^{\mathrm{event}}_{p,r}\) the
operator's ongoing event limit, and \(C^{\mathrm{stable}}_{p,r}\) stable
capacity. Selected sessions must satisfy

\[
b_{p,r}+\sum_c s_{c,r}x_c
\le C^{\mathrm{event}}_{p,r}
\le C^{\mathrm{stable}}_{p,r}.
\]

Migration may temporarily queue serving work. Over the migration window, the
planner uses this aggregate work bound:

\[
Q_{p,r}=
\max\left(
0,\ H\left(b_{p,r}+\sum_c s_{c,r}x_c\right)
  +\sum_c m_{c,r}x_c
  -HC^{\mathrm{stable}}_{p,r}
\right).
\]

For advertised debt budget \(E_{p,r}\),

\[
Q_{p,r}\le E_{p,r}.
\]

Debt sensitivity uses the same 0/5/10/20% scale:

\[
E_{p,r}=e\,H\,C^{\mathrm{stable}}_{p,r}.
\]

Required recovery time is a result:

\[
R_{p,r}=
\begin{cases}
Q_{p,r}/
\left(C^{\mathrm{stable}}_{p,r}
-b_{p,r}-\sum_cs_{c,r}x_c\right),&
\text{positive spare capacity},\\
+\infty,&\text{positive debt and no spare capacity},\\
0,&Q_{p,r}=0.
\end{cases}
\]

This is not a time-scheduled queue bound. Work that arrives late can create a
larger queue even when the aggregate fits. The event simulator independently
schedules shared routes, reconstruction endpoints, requests, commits, and
source power. It also drives a fluid pool-service queue from realized replay
start/finish and commit times. An executed point is invalid if that queue
exceeds the advertised debt budget or cannot recover.

Replay prefill contributes measured serving-transition work. KV ingest uses a
separate measured ingest slot. Queue-Haul does not charge KV ingest to prefill
or decode service unless interference is measured.

## 6. Other resource constraints

For every session,

\[
\sum_{a,p}x_{jap}\le1,\qquad x_{jap}\in\{0,1\}.
\]

Only eligible candidates exist.

For each pool,

\[
\sum_c k_cx_c\le K_p
\]

for advertised usable live-KV blocks.

For each logical route,

\[
\sum_c b_cx_c\le B_pH.
\]

This byte constraint is a fluid lower bound. The event simulator validates the
actual shared-route schedule, transferred bytes, and completion time.

Source preparation streams and pool transition capacity are time budgets:

\[
\sum_{c:i(c)=i}t_cx_c\le S_iH,
\qquad
\sum_{c:p(c)=p}t_cx_c\le M_pH.
\]

Indivisible actions are packed into stream bins before acceptance.

## 7. Objective, schedule, and frontier

For requested shed \(\Theta\), solve lexicographically:

1. meet \(\sum_c w_cx_c\ge\Theta\);
2. minimize migration work and resource debt.

If the target is unreachable, maximize valid shed and report `target_unmet`
with the watt shortfall. It is never called successful curtailment.

For each selected migration, the planner emits session, source, action, pool,
start, transfer or reconstruction finish, quiesce, commit, first-token
completion, bytes, transition work, ongoing work, KV blocks, and conservative
source watts credited. The execution validator rejects a plan whose concrete
schedule violates an advertised resource or misses the deadline.

The requirement frontier summarizes the validated schedules and reports, for
every target:

- achieved and unmet accelerator watts;
- selected sessions and replay/KV mix;
- route bytes and minimum route rate;
- reconstruction and ingest work;
- ongoing prefill/decode headroom;
- live-KV blocks;
- service debt and required recovery;
- source-stream occupancy;
- binding resource set;
- predicted and realized makespan; and
- sessions, context tokens, and KV bytes still exposed.

The main targets are 10/25/50/75/90/100% of maximum modeled shed.

## 8. Fixed and multi-pool contracts

Adding a destination adds candidate columns and pool/route rows. It does not
change the decision variable.

The fixed-contract experiment uses one canonical compatible integrated
destination pool while requested shed rises. Its workload, packing, deadline,
route, service, debt, KV, reconstruction, and ingest budgets do not change.
This isolates joint whole-session selection and replay/KV coordination.

The multi-pool experiment then opens that contract and varies pool count,
composition, compatibility, and headroom. Two capacity regimes answer different
questions:

1. hold total destination resources fixed and split them over 1/2/4/8 pools;
2. give each of 1/2/4/8 pools the same event budget.

The first isolates fragmentation, compatibility, and route diversity. The
second measures the value of additional headroom and the point where a shared
source constraint stops further benefit. Resource diversity and compatibility
diversity are varied in separate controlled experiments.

Binding resources are a set, not an exclusive cause. Every result reports all
resources with zero normalized residual slack; simultaneous route, transition,
service, debt, KV, source-stream, and deadline constraints remain visible.

## 9. Evidence hierarchy and scope

Every input and result is classified as:

- **Measured:** source power curves, two-A100 replay/KV correctness, exact KV
  bytes and blocks, and measured capacities.
- **Fitted:** conservative migration-time and throughput models used only in
  their stated validity domains.
- **Simulated:** large-scale coordination, scheduling, queue, and power outcomes
  driven by measured, fitted, and assumed inputs.
- **Assumed / sensitivity:** destination service flex, debt budgets, unmeasured
  route bandwidth and RTT, multi-pool inventories, disaggregated pools, and
  unmeasured hardware points.

The two-A100 testbed validates implementation details abstracted by simulation:
background preparation, overlap, catch-up, request-boundary quiesce, route
switch, first post-switch token, timing jitter, and realized source power
change. The simulator evaluates coordination and scale. It is not direct
evidence of production destination behavior.

Current evidence supports:

- measured GPT-OSS-20B/A100 source power curves, still requiring held-out group
  removal validation;
- end-to-end replay and compatible KV reconstruction on two A100s;
- conservative replay and KV duration fits in their stated context/bandwidth
  ranges;
- exact KV block accounting and measured KV capacity; and
- descriptive loaded-migration interference observations.

The archived destination campaign does not support an accepted service
boundary. Its valid-looking points have no measured failing boundary, and most
other cells have invalid completion or cache state. Until the targeted rerun
brackets passing and failing points, service flex and debt remain
`assumed/sensitivity`.

Out of scope:

- facility power and full-site power claims;
- arbitrary mid-token migration;
- return migration and wake-up;
- predicting unrelated destination arrivals or provider fleet policy;
- cold model placement;
- cross-session KV sharing;
- TCP behavior; and
- long-term destination equilibrium.

## 10. Canonical scenario and sensitivities

One versioned canonical scenario supplies the fixed-contract results. It records
the workload; source packing, hardware, and model; one compatible integrated
destination pool; deadline; route bandwidth and RTT; event service flex; debt
budget; KV, reconstruction, and ingest capacity; and random seed. Each value
has units, evidence status, provenance, validity range, and the evidence needed
to replace it. Existing central defaults are canonical if they provide this
record; otherwise use one documented mid-range point.

The standard targets are 10/25/50/75/90/100% of maximum modeled shed. Separate
scenario sweeps use 30/60/120/300-second deadlines, 1/5/10-Gbps routes, and
0/5/10/20% service flex and debt. Workload, source packing, deadline, bandwidth,
and seed sweeps are scenarios, not statistical error bars.

## 11. Evaluation A: mechanism validation

**Question A.** Can the measured handoff primitives and source power model
predict an actual request-boundary migration?

The main result is a compact two-A100 end-to-end timeline for replay and KV
transfer:

- x-axis: time;
- intervals: source serving, bulk preparation, route transfer, replay
  reconstruction or KV ingest, catch-up, quiesce, and route switch;
- markers: first post-switch token and source shutdown where applicable; and
- overlay: source power plus predicted and realized timing.

Report prediction error and safety margin. A small companion scatter or table
may compare predicted and measured migration time and power shed across the
measured corpus. The same timeline answers whether the analytical plan remains
executable by comparing predicted and realized makespan, debt, recovery, and
power shed; do not repeat its Gantt chart elsewhere.

A three-A100 cross-region demonstration is out of the plan unless it reveals a
qualitatively new implementation constraint not covered by this validation.

## 12. Evaluation B: many sessions under one fixed contract

Keep the canonical destination contract fixed as requested shed rises.

**Question B1.** What shared resources are consumed, and what becomes binding?

Plot normalized residual slack

\[
\operatorname{slack}_r =
\frac{\operatorname{capacity}_r-\operatorname{use}_r}
     {\operatorname{capacity}_r}
\]

against requested shed in watts and/or percent of maximum modeled shed at the
standard targets. Zero is binding; negative values are visible failures. Use
one readable multi-line panel or small multiples for source preparation
streams, route bytes or time, replay reconstruction, KV ingest, ongoing prefill,
ongoing decode, service debt, live-KV blocks, and deadline or realized
makespan. Report the complete binding-resource set. This figure explains how
the fixed contract is spent.

**Question B2.** Does joint replay/KV planning produce more executable shed than
simpler policies?

Plot requested shed on the x-axis and executable achieved shed on the y-axis,
with a \(y=x\) reference and unmet watts below target. Main-paper series are
Queue-Haul, all replay, all KV, the best simple greedy baseline, and an exact
integer or LP reference where tractable. Put the full baseline set in secondary
material. A companion panel shows selected replay/KV mix. Annotate the first
infeasible target and complete binding-resource set for each policy.

**Question B3.** Does the analytical plan remain executable after concrete
scheduling? Use Evaluation A's timeline and the predicted-versus-realized
makespan, debt, recovery, and power-shed columns rather than another Gantt.

## 13. Evaluation C: many sessions across many pools

**Question C1.** How does maximum executable shed change with 1, 2, 4, and 8
compatible pools?

Use two separate panels with pool count on the x-axis and maximum executable
source accelerator-power shed in watts and percent of maximum modeled shed on
the y-axis:

1. **Fixed total resources:** split constant total destination resources across
   the pools to isolate fragmentation, route diversity, and compatibility.
2. **Fixed resources per pool:** give every pool the same budget so total
   capacity grows, exposing added headroom until a shared source constraint
   binds.

Use a few representative workloads in the main paper and the full sweep in a
robustness matrix or appendix. Attribute gains to total capacity,
fragmentation, route diversity, or compatibility diversity rather than to pool
count alone.

**Question C2.** Which destination improvements increase achievable shed, and
where do their gains stop?

Use identical-axis small multiples with advertised headroom or capacity
multiplier on the x-axis and maximum executable shed on the y-axis, one panel
each for route bandwidth or queued bytes, replay reconstruction, KV ingest,
ongoing prefill, ongoing decode, event debt, and live-KV blocks. Use 1/2/4/8
pool lines only when readable; otherwise use one representative count and move
the full matrix to the appendix. Each curve must show the knee at which another
resource joins the binding set. Do not put unlike physical units on one axis.

**Question C3.** How does the executable schedule change with the contract?

Select three or four points from C1-C2, such as route-, reconstruction-,
service-, and KV-memory/ingest-constrained points. Plot time on the x-axis and
destination pools on the y-axis, with one rectangle per migration, width equal
to scheduled duration, replay/KV fill or hatch, commit and first-token markers,
shared route and transition-resource occupancy, and final achieved shed. These
schedule-morphing examples explain the capacity-curve shapes.

**Question C4.** How do workload shape and destination diversity change the
required contract?

Run resource diversity and compatibility diversity separately. Resource
diversity varies route, reconstruction, ingest, service, or KV budgets while
holding compatibility fixed. Compatibility diversity varies eligible
action/pool choices while holding total physical resources fixed. For coding,
interactive coding, agentic, and ShareGPT-like conversation workloads, report
maximum executable shed and the complete binding-resource set in a compact
binary or normalized-slack matrix. Never assign one exclusive failure cause
when constraints bind together.

## 14. Evaluation D: planner quality and scale

**Question D1.** How close is the control-path planner to exact and relaxed
references? Compare exact integer, LP bound, rounded/packed plan, Queue-Haul
greedy, and focused baselines on executable shed, resource debt, optimality
gap, planning time, and memory.

**Question D2.** Can Queue-Haul plan for 10K, 100K, and 1M sessions within an
operationally useful budget? Plot planning time and memory against session
count for ten seeds. Every point retains provenance and an execution-validator
result.

## 15. Result-table contract

Figures consume tidy result tables, never ad hoc simulator objects. Every
result row contains:

- experiment and scenario ID, seed, and workload;
- source hardware, model, packing, deadline, and measurement window;
- requested, achieved, and unmet watts;
- selected sessions; replay/KV counts and bytes; and pool assignment;
- route, reconstruction, ingest, service, debt, recovery, and KV use;
- normalized slack for every resource and the complete binding-resource set;
- predicted and realized makespan and source shutdown time;
- sessions, context tokens, and KV bytes still exposed; and
- per-input measured/fitted/assumed/simulated status, validity range, and
  provenance.

The schedule table contains one row per selected migration: session ID, source,
action, pool, start, transfer/reconstruction finish, quiesce, commit,
first-token completion, bytes, transition work, ongoing work, KV blocks, and
conservative source watts credited.

## 16. Main-paper figure budget and claim boundary

The main-paper sequence is:

1. source power-model validation;
2. replay/KV single-session crossover and measured breakdown;
3. two-A100 end-to-end execution timeline;
4. fixed-contract resource slack versus requested shed;
5. Queue-Haul versus coordinated-planning baselines;
6. maximum shed versus 1/2/4/8 pools under fixed-total and fixed-per-pool
   resources;
7. destination bottleneck-improvement small multiples;
8. representative schedule morphing;
9. planner quality and scale; and
10. compact workload/hardware robustness matrix.

Full deadline, route, workload, packing, seed, and hardware sweeps belong in
secondary material.

Queue-Haul may claim:

> Given measured handoff primitives and advertised compatible-pool budgets,
> Queue-Haul computes and validates the source accelerator-power shed achievable
> by a deadline.

It does not claim facility or grid power from accelerator measurements,
arbitrary mid-token migration, hidden provider capacity or unrelated arrivals,
safe service headroom before the service-boundary gate passes, long-term
destination equilibrium, production admission certificates without live
leasing and revalidation, or measured hardware generality for assumed
profiles. Infeasible targets report unmet watts and exposed work, never
successful curtailment.
