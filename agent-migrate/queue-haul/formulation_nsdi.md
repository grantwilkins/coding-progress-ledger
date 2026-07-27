# Queue-Haul NSDI formulation

Queue-Haul moves stateful inference sessions away from a source power domain
before a deadline. It chooses which sessions to move, whether to replay context
or transfer KV state, and which compatible destination pool accepts each
session.

The main result is a requirement frontier:

> For each requested source accelerator-power reduction, report the destination
> and route resources required to produce an executable handoff plan.

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

## 7. Objective and frontier

For requested shed \(\Theta\), solve lexicographically:

1. meet \(\sum_c w_cx_c\ge\Theta\);
2. minimize migration work and resource debt.

If the target is unreachable, maximize valid shed and report `target_unmet`
with the watt shortfall. It is never called successful curtailment.

The requirement frontier reports, for every target:

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

The main targets are 10/25/50/75/90/100% of maximum modeled shed. Main
deadlines are 30/60/120/300 seconds. Main route rates are 1/5/10 Gbps.

## 8. One and many destinations

Adding a destination adds candidate columns and pool/route rows. It does not
change the decision variable.

Two multi-pool experiments answer different questions:

1. hold total destination resources fixed and split them over 1/2/4/8 pools;
2. give each of 1/2/4/8 pools the same event budget.

The first isolates fragmentation, compatibility, and route diversity. The
second measures the value of additional headroom and the point where a shared
source constraint stops further benefit. Resource diversity and compatibility
diversity are varied in separate controlled experiments.

## 9. Evidence and scope

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
passes, service headroom remains `sensitivity`.

Out of scope:

- facility power and full-site power claims;
- arbitrary mid-token migration;
- return migration and wake-up;
- predicting unrelated destination arrivals or provider fleet policy;
- cold model placement;
- cross-session KV sharing;
- TCP behavior; and
- long-term destination equilibrium.

## 10. Evaluation questions

The evaluation follows the design memo:

1. Does the source model predict group-removal power?
2. Are replay and KV distinct measured reconstruction actions?
3. Does joint planning choose sensible actions under contention?
4. How much power can one destination pool absorb?
5. Does the plan execute by the deadline?
6. Is the greedy planner close to exact and relaxed references?
7. What do additional destination pools buy?
8. Does the abstraction survive hardware changes?
9. Do results survive workload changes?
