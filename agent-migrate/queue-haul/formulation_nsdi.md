# Queue-Haul NSDI formulation

Queue-Haul moves stateful inference sessions away from a source power domain
before a deadline. It chooses which sessions to move, whether to replay context
or transfer KV state, and which compatible destination pool accepts each
session.

The primary planner output is an executable fixed plan:

> Given measured handoff primitives and advertised compatible destination-pool
> budgets, report how much source accelerator power can be shed by a deadline
> and the executable migration schedule that achieves it.

The plan names the whole sessions to move, replay or KV transfer for each,
destination pool and replica, route, and order. The execution report adds
start, reconstruction or ingest completion, commit, first-token observations,
source transition, resource use, debt, recovery, achieved shed, and unmet
shed. The requirement frontier summarizes plans across requested targets.

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

seconds to switch routing. Average source power over the final \(W\)-second
window must be below the limit.

Queue-Haul performs a request-boundary live handoff:

1. prepare state while the source still owns the session;
2. quiesce at a request boundary;
3. copy or reconstruct the final delta;
4. switch routing; and
5. verify a post-switch destination response in hardware evaluation.

This is not arbitrary mid-token migration. Power ownership changes at commit;
the hardware result separately verifies the first post-switch token. The
destination pool accepts declared ongoing service and KV demand after commit.
Long-term fleet management and return migration are outside the claim.

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

A destination pool identifies a compatible serving type, replicas, route, and
allowed methods. The V1 compatibility fingerprint contains model, tokenizer,
durable-log contract, and KV ABI identifiers. Replay requires the first three;
KV transfer additionally requires the KV identifier. Hardware/runtime identity
and warm-health attestation remain provenance or production extensions rather
than explicit V1 compatibility fields.

The public candidate is

\[
c=(j,a,p),\qquad a\in\{R,K\}.
\]

Pool choice is public. Queue-Haul deterministically packs accepted sessions
onto concrete replicas to validate service, KV, and migration occupancy.

Each pool advertises:

- normal serving capacity;
- stable serving capacity;
- an event admission limit for ongoing prefill and decode work;
- a temporary service-debt budget in replica-seconds;
- replica inventory, from which the planner derives migration capacity;
- usable live-KV capacity in physical blocks;
- route identity and path, with link rates supplied by the scenario; and
- evidence status and provenance.

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
(w_j,\ t_c,\ b_c,\ s_c,\ u_c,\ k_c).
\]

- \(w_j\): conservative source accelerator watts removed.
- \(t_c\): measured or conservatively fitted handoff duration.
- \(b_c\): route bytes.
- \(s_c\): ongoing prefill/decode service work.
- \(u_c\): temporary serving-transition work.
- \(k_c\): block-rounded live-KV demand.

Replay sends compact durable context and contributes serving-transition work.
KV transfer sends sealed KV blocks. Reconstruction, transfer, and ingest are
included in the fitted duration \(t_c\). The planners also charge aggregate
replay service and KV-transfer/ingest work to distinct method-specific rows; an
unsealed tail is reconstructed during catch-up.

For effective route rate \(B_p\), destination prefill rate \(\rho_p(T)\),
KV-ingest rate \(\mu_p\), and fitted residuals:

\[
t^R_c =
\frac{C_j}{B_p}+
\alpha_p\frac{T_j}{\rho_p(T_j)}
+r^R_p+\tau^{\mathrm{catch}}_j+\tau^{\mathrm{switch}}_p,
\]

\[
t^K_c =
\max\left(\frac{K_j}{B_p},\frac{K_j}{\mu_p}\right)+
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
  +\sum_c u_{c,r}x_c
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
schedules shared routes, replay and KV endpoints, requests, commits, and source
power. It also drives a fluid pool-service queue from realized replay
start/finish and commit times. An executed point is invalid if that queue
exceeds the advertised debt budget or cannot recover.

Replay prefill contributes measured serving-transition work. KV transfer and
ingest contribute to the distinct KV migration-work row; neither is charged as
ongoing destination service.

## 6. Other resource constraints

For every session, the executable plan chooses at most one candidate. If
\(y_c\) denotes that whole-session choice,

\[
\sum_{c:j(c)=j}y_c\le1,\qquad y_c\in\{0,1\}.
\]

The optimizer described below does not solve this integer program directly. It
first replaces \(y_c\) by a fractional LP variable \(x_c\in[0,1]\), then rounds
the LP solution to a resource-feasible set of whole sessions. This distinction
is important: the LP objective is solved optimally for the relaxation, but the
emitted migration plan is not claimed to be the optimal integer solution.

Only eligible candidates exist.

For each pool,

\[
\sum_c k_cx_c\le K_p
\]

for advertised usable live-KV blocks.

For each physical link \(\lambda\),

\[
\sum_{c:\lambda\in\operatorname{path}(c)} b_cx_c\le B_\lambda H.
\]

These byte constraints are fluid relaxations. The event simulator validates
the actual shared-link schedule, transferred bytes, and completion time.

For a non-fluid pool with \(N_p\) replicas, replay and KV transfer have separate
aggregate occupancy budgets:

\[
\sum_{c:p(c)=p,\ a(c)=a}t_cx_c\le N_pH,
\qquad a\in\{R,K\}.
\]

For a pool with the implemented fluid migration contract, the replay row is
instead measured serial replay work bounded by \(N_pH\) times the advertised
replay speedup, and the KV row is bytes bounded by \(N_pH\) times advertised
ingest bytes/s. After selection, indivisible actions are packed onto replicas.
For non-fluid pools, replay and KV occupancies are accumulated separately on
each replica, and the larger must fit within \(H\); service and live-KV bounds
must also hold independently on every replica.

## 7. LP, fallback, and implemented greedies

The implementation assembles all shared constraints into a normalized sparse
matrix. Let \(A_{rc}\) be candidate \(c\)'s demand on resource \(r\), divided by
that resource's capacity, and let \(J_{jc}=1\) when candidate \(c\) moves session
\(j\). Thus the common LP feasible region is

\[
\mathcal X=\left\{x:\ 0\le x_c\le1,\quad
                 \sum_cJ_{jc}x_c\le1\ \forall j,\quad
                 \sum_cA_{rc}x_c\le1\ \forall r\right\}.
\]

In the hardware campaign and canonical simulator, the Queue-Haul policy uses
the implemented `lp_work_first` objective. For requested shed \(\Theta\), its
first phase is

\[
s^*=\min_{x\in\mathcal X,\ s\ge0}s
\quad\text{s.t.}\quad
\sum_c w_cx_c+s\ge\Theta.
\tag{LP-1}
\]

When the target is attainable, \(s^*=0\). When it is not, minimizing shortfall
is exactly the maximum-evacuation fallback under the same deadline and resource
constraints:

\[
\Theta-s^*=\max_{x\in\mathcal X}\sum_cw_cx_c.
\]

This equality is for the fractional relaxation. Deterministic rounding and the
execution checks can produce less evacuation; the code does not claim a
maximum executable integer set.

The implementation fixes \(s\) to \(s^*\), within solver tolerance, and then
minimizes total modeled migration work,

\[
q^*=\min_{x\in\mathcal X}\sum_c q_cx_c
\quad\text{s.t.}\quad
\sum_cw_cx_c+s^*\ge\Theta,
\tag{LP-2}
\]

where \(q_c=t_c\) for the fixed aggregate planner and is the measured migration
work field for a pool with a fluid migration service. Finally it fixes that
work, again within tolerance, and minimizes the largest normalized resource
pressure \(\phi\), subject to \(Ax\le\phi\mathbf1\) and \(\phi\le1\). These are
three lexicographic phases, not a weighted sum.

The pool-aware `lp` and `lp_highs` reference backends express the same target-
then-work policy by first solving LP-2 with \(s=0\). If that problem is
infeasible, they maximize fractional shed, then minimize work at the resulting
maximum. The column-generation backends implement the equivalent two-phase
shortfall-then-work form. These backend variants are simulator/planner-quality
tools; the hardware campaign names `lp_work_first`.

The fractional solution is converted to an executable plan by deterministic
rounding. Candidates are considered using their LP mass and migration work;
only a candidate whose session is still unchosen and whose addition keeps every
normalized resource row at most one is accepted. A second pass considers
remaining candidates by watts per migration work. Rounding may therefore fail
to attain a fractionally attainable target. The planner recomputes the exact
nonlinear source power for the rounded whole-session set and reports the
remaining watt shortfall. A result with positive shortfall is `target_unmet`,
never successful curtailment. Concrete placement and eager-parallel simulation
are subsequent acceptance checks and may remove further moves.

### Static greedy

The implemented `greedy` uses the same eligible candidates and normalized
resource columns as the LP. For each session it first identifies the candidate
with the smallest unpriced normalized demand, \(\sum_rA_{rc}\). It then computes
one duplicate-safe scarcity price per resource,

\[
\pi_r=\max\left\{1,
\sum_j A_{r,c_j^{\min}}\right\},
\]

and gives every candidate the fixed score

\[
\operatorname{score}(c)=
\frac{w_c}{\sum_r\pi_rA_{rc}}.
\]

The pool-aware greedy scans candidates once in descending score, accepting a
candidate only if its session is unchosen and all resource rows still fit. The
fixed aggregate planner used by the existing hardware path ranks sessions by
their best candidate score and, for each session, tries replay and KV transfer
in score order. It stops after exact recomputation reaches the power limit or no
more candidates fit. Neither implementation recomputes scarcity prices after
an acceptance. Ties are resolved by stable integer/session order.

### Lagrangian greedy

`greedy_lagrangian` is implemented only in the pool-aware planner. The hardware
campaign invokes it through the implemented one-pool, idle dedicated-sink
adapter; this does not make multi-pool or loaded-destination behavior hardware
evidence. The checksum-pinned July 30 hardware bundle predates this policy, so
the repository currently contains no completed hardware evidence for the
Lagrangian extension; its present evidence is simulation.

It starts from the static greedy prices. Given resource prices \(\pi\) and a
power multiplier \(\eta\), it chooses each session's cheapest action using

\[
d_c=\frac{q_c}{H}+\sum_r\pi_rA_{rc}.
\]

For each source instance, sessions are sorted by \(d_c/\ell_j\). The algorithm
evaluates every prefix of that order with the exact fitted source power curve,
not the additive \(w_j\)'s, and retains the prefix minimizing

\[
\sum_{c\in Q}d_c-
\eta\,\frac{w_i(Q)}{\max\{\Theta,1\}}.
\]

It repeats the prefix construction for one deterministic alternative-action
ordering. Around each selected prefix it retains its neighboring lengths, the
singleton and full prefixes, and evenly spaced prefix lengths. Below 75% of all
removable awake power the implementation performs one price pass and uses eight
prefix buckets; at or above 75% it performs four passes and uses 64. After each
pass it applies projected subgradient updates

\[
\pi_r\leftarrow\max\{0,\pi_r+\alpha(u_r-\bar u_r)\},
\qquad
\eta\leftarrow\max\left\{0,
\eta+\alpha\frac{\Theta-w(Q)}{\max\{\Theta,1\}}\right\},
\]

with \(\alpha=0.5/\sqrt{k}\), actual normalized use \(u_r\), and limit
\(\bar u_r\). Route limits reserve the largest candidate's isolated non-route
tail because the byte-row relaxation alone does not represent that time.

The recovery stage combines retained source-local prefixes by target-capped
exact watts gained per additional migration work, incrementally rejecting any
combination that exceeds an aggregate row. It then packs the selected actions
on concrete replicas once. If the target is unreachable or that packing fails,
it reruns recovery with packing checked on every accepted change. The final
sparse resource sum, replica assignment, exact source power, and eager-parallel
deadline prediction are all checked before the plan is accepted.

Both greedies are heuristics. Neither supplies an LP bound or a global integer-
optimality claim, and the Lagrangian pass count and retained-prefix buckets are
fixed implementation budgets rather than convergence guarantees.

## 8. Schedule and frontier

Selected moves are ordered by migration work per conservative watt. At
controller completion, the executor starts every selected move eagerly.
Independent paths proceed in parallel; overlapping paths share link capacity,
and replay and KV endpoint queues use the selected order to break simultaneous
arrivals. The order is therefore a priority relation, not serial execution.

For each selected migration, the planner emits session, action, pool, replica,
route, order, and optional pacing metadata. The plan reports aggregate
conservative and exact source power. The execution report supplies phase timing,
commit, observed request timing, bytes, and realized power. The validator
rejects a plan whose concrete schedule violates an advertised resource or
misses the deadline.

The requirement frontier summarizes the validated schedules and reports, for
every target:

- achieved and unmet accelerator watts;
- selected sessions and replay/KV mix;
- route bytes and minimum route rate;
- migration and serving-transition work;
- ongoing prefill/decode headroom;
- live-KV blocks;
- service debt and required recovery;
- binding resource set;
- predicted and realized makespan; and
- sessions, context tokens, and KV bytes still exposed.

The main targets are 10/25/50/75/90/100% of maximum modeled shed.

## 9. Fixed and multi-pool contracts

Adding a destination adds candidate columns and pool/route rows. It does not
change the decision variable.

The fixed-contract experiment uses one canonical compatible integrated
destination pool while requested shed rises. Its workload, packing, deadline,
route, service, debt, KV, and migration budgets do not change.
This isolates joint whole-session selection and replay/KV coordination.

The multi-pool experiment then opens that contract and varies pool count,
composition, compatibility, and headroom. Two capacity regimes answer different
questions:

1. hold total destination resources fixed and split them over 1/2/4/8 pools;
2. give each of 1/2/4/8 pools the same event budget.

The first isolates fragmentation, compatibility, and route diversity. The
second measures the value of additional headroom and the point where a shared
route constraint stops further benefit. Resource diversity and compatibility
diversity are varied in separate controlled experiments.

Binding resources are a set, not an exclusive cause. Every result reports all
resources with zero normalized residual slack; simultaneous route, transition,
service, debt, KV, and deadline constraints remain visible.

## 10. Evidence hierarchy and scope

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
change. Hardware owns measured execution and timing of the frozen plan.
Software owns optimizer intent, modeled admission, and fleet-scale sensitivity.
A disagreement is a validation failure. Simulation is not direct evidence of
production destination behavior.

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

## 11. Canonical scenario and sensitivities

One versioned canonical scenario supplies the fixed-contract results. It records
the workload; source packing, hardware, and model; one compatible integrated
destination pool; deadline; route bandwidth and RTT; event service flex; debt
budget; KV and migration capacity; and random seed. Each value
has units, evidence status, provenance, validity range, and the evidence needed
to replace it. Existing central defaults are canonical if they provide this
record; otherwise use one documented mid-range point.

The standard targets are 10/25/50/75/90/100% of maximum modeled shed. Separate
scenario sweeps use 30/60/120/300-second deadlines, 1/5/10-Gbps routes, and
0/5/10/20% service flex and debt. Workload, source packing, deadline, bandwidth,
and seed sweeps are scenarios, not statistical error bars.

## 12. Evaluation A: mechanism validation

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

## 13. Evaluation B: many sessions under one fixed contract

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
one readable multi-line panel or small multiples for route bytes or time,
aggregate migration time, ongoing prefill, ongoing decode, service debt,
live-KV blocks, and deadline or realized
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

## 14. Evaluation C: many sessions across many pools

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
each for route bandwidth, aggregate migration capacity, ongoing prefill,
ongoing decode, event debt, and live-KV blocks. Use 1/2/4/8
pool lines only when readable; otherwise use one representative count and move
the full matrix to the appendix. Each curve must show the knee at which another
resource joins the binding set. Do not put unlike physical units on one axis.

**Question C3.** How does the executable schedule change with the contract?

Select three or four points from C1-C2, such as route-, migration-, service-,
and KV-memory-constrained points. Plot time on the x-axis and
destination pools on the y-axis, with one rectangle per migration, width equal
to scheduled duration, replay/KV fill or hatch, commit and first-token markers,
shared route and transition-resource occupancy, and final achieved shed. These
schedule-morphing examples explain the capacity-curve shapes.

**Question C4.** How do workload shape and destination diversity change the
required contract?

Run resource diversity and compatibility diversity separately. Resource
diversity varies route, migration, service, or KV budgets while
holding compatibility fixed. Compatibility diversity varies eligible
action/pool choices while holding total physical resources fixed. For coding,
interactive coding, agentic, and ShareGPT-like conversation workloads, report
maximum executable shed and the complete binding-resource set in a compact
binary or normalized-slack matrix. Never assign one exclusive failure cause
when constraints bind together.

## 15. Evaluation D: planner quality and scale

**Question D1.** How close is the control-path planner to exact and relaxed
references? Compare an exact integer oracle where tractable, the fractional
target-first LP surrogate and its rounded/packed plan, Queue-Haul greedy, and
focused baselines on executable shed, resource debt, planning time, and memory.
Do not report the target-first LP as an upper bound on exact nonlinear shed; a
separately defined chord relaxation is required for that claim.

**Question D2.** Can Queue-Haul plan for 10K, 100K, and 1M sessions within an
operationally useful budget? Plot planning time and memory against session
count for ten seeds. Every point retains provenance and an execution-validator
result. The current single-seed 1M sensitivity is not a positive operational
answer: although Lagrangian greedy is faster than static greedy, it takes about
12 minutes, uses 5.02 GB peak RSS, and leaves four seconds of migration-deadline
slack. Replica packing remains the dominant measured bottleneck.

## 16. Result-table contract

Figures consume tidy result tables, never ad hoc simulator objects. Every
result row contains:

- experiment and scenario ID, seed, and workload;
- source hardware, model, packing, deadline, and measurement window;
- requested, achieved, and unmet watts;
- selected sessions; replay/KV counts and bytes; and pool assignment;
- route, migration, service, debt, recovery, and KV use;
- normalized slack for every resource and the complete binding-resource set;
- predicted and realized makespan and source shutdown time;
- sessions, context tokens, and KV bytes still exposed; and
- per-input measured/fitted/assumed/simulated status, validity range, and
  provenance.

The schedule table contains one row per selected migration: session ID, source,
action, pool, start, transfer/reconstruction finish, quiesce, commit,
first-token completion, bytes, transition work, ongoing work, KV blocks, and
conservative source watts credited.

## 17. Main-paper figure budget and claim boundary

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
