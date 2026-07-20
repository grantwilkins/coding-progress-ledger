# Queue-Haul formulation

Each session has a source instance, context, sampled requests, durable log, and
expected prefill and decode rates. A model profile supplies the measured power
curve, service rates, replay rates, KV block layout, action times, concurrency
limits, resident KV-token capacity, and uncertainty cases. Active sessions are
placed only when both compute load and resident KV fit; cold sessions consume
neither until reactivation.

The planner chooses whole sessions and one of three actions: replay, KV
transfer, or replay on request. Random, greedy, and LP selection are
separate policies. Destination placement is a balanced pass.
Destination placement enforces the same compute and resident-KV limits as the
source placement.
Only local source power is constrained; destination power is reported.

## Greedy

For active sessions and an awake final state, the greedy policy uses the same
normalized resource rows and usable window \(H\) as the LP. Each action's score
is its largest normalized resource use weighted by that resource's total demand.
Sessions are sorted once by their best action score. In that order, the policy
chooses the lowest-scoring replay or KV-transfer action that fits every remaining
source, network, destination service, compute, and resident-KV capacity. The
event simulator still decides whether the aggregate reservations form an exact
schedule.

## LP planner

The first LP scope is active sessions, replay and KV transfer, one destination
pool, the central profile case, and an awake final source state. Other cases
hard-fail.

For session \(j\), \(x_j^R,x_j^K\in[0,1]\) select replay or KV transfer and
\(z_j\in[0,1]\) leaves the session at the source:

\[
x_j^R+x_j^K+z_j=1.
\]

The usable migration window reserves the measured trailing power window:

\[
H=D-\text{controller delay}-\text{power window}.
\]

Methods whose unloaded duration exceeds \(H\) are disabled. Every remaining
resource \(r\) has a normalized linear load:

\[
\sum_{j,a} u_{j,a,r}x_j^a\leq 1.
\]

The resources are source-instance move time, each fixed network link, the
balanced destination-link pool, destination replay time, destination KV
service time, destination compute load, and resident KV tokens. Replay uses
measured durable-log bytes and replay time. KV transfer uses complete sealed
blocks, the slower of path transfer and destination loading, and final
reconstruction of the partial tail followed by measured synchronization.
Source-instance time covers the complete unloaded move because a source move
slot remains occupied until commit.

For initial source load \(L_s\), the linear power coefficient for session \(j\)
is its measured one-session reduction:

\[
g_j=P(L_s)-P(L_s-\ell_j).
\]

Since \(P\) is nondecreasing and concave, \(\sum_j g_jx_j\) is a conservative
power-reduction estimate: removing several sessions from one instance saves at
least the sum of their initial marginal reductions. With requested reduction
\(\Delta P\), the restored Queue-Haul program is:

\[
\begin{aligned}
\operatorname{minimize}\quad
&\sum_{j,a}c_j^a x_j^a\\
\operatorname{subject\ to}\quad
&\sum_j g_j(x_j^R+x_j^K)\geq\Delta P.
\end{aligned}
\]

The exact concave constraint \(P_{\rm final}\leq P_{\rm limit}\) is not an LP.
The planner therefore solves the safe linear bound, then evaluates the rounded
plan with the exact power curve. If the target-constrained program is
infeasible, a separate solve maximizes the conservative power reduction under
the same resource limits. `lp_peak_first` minimizes shortfall, peak resource
use, then work; `lp_work_first` swaps the last two objectives. These comparison
solvers are not the default formulation.

The fractional result is rounded deterministically. A whole-session assignment
is accepted only when it preserves every LP resource limit. Remaining power
shortfall is filled in LP-fraction order, then by power reduction per unloaded
second, without violating those limits. Destination placement remains a
separate balanced pass that enforces per-instance compute and KV residency.

The simulator follows background preparation, source quiescing, catch-up,
route switch, commit, and optional sleep or shutdown. KV copies wait in a FIFO
per destination before transferring; admitted transfers share every named
link. Source power changes only at commit. A plan is feasible only when
central-case execution meets both the move deadline and the trailing power
window.

A run is accepted when:

1. the trailing-window modeled source power is at or below the limit;
2. every planned move commits by the deadline; and
3. every request observed by the deadline starts by the deadline.

The third condition checks routing readiness, not end-to-end request latency.
Request events affect context and timing but not dynamic power yet; output
columns therefore say `modeled_*_power`.

The LP conservatively budgets migration resources through
\(D-\text{power window}\), but acceptance separates the two deadlines: modeled
trailing-window power must meet its limit at \(D\), and every migration must
commit by \(D\). `lp_power_shortfall_w` reports exact remaining source-power
shortfall after rounding. `lp_peak_pressure` reports the largest rounded
normalized LP resource load. The discrete-event result remains authoritative
because aggregate LP capacity does not guarantee a schedule with serial stages
and shared queues.

Unsupported context, load, concurrency, topology, or profile cases hard-fail.
Open measurement work is listed in `DATA_TO_COLLECT.md`.
