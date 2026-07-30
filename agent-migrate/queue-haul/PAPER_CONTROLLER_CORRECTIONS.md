# Controller section corrections

This file maps the pasted `Controlling Many Migrations` section to the current
pool-aware planner and eager event simulator. The replacement text below is
ready to paste into the paper.

## Line-by-line correction ledger

| Original text anchor | Correction |
|---|---|
| “enough source power is removed by the deadline” | Define the usable migration horizon as \(H=D-C-W\); commit and the trailing power window must fit before \(D\). |
| “five resource limits” | Replace with the variable implemented rows: every path link, every pool service facet, optional debt facets, pool KV blocks, and pool migration occupancy. |
| “omit source egress” | Delete. Every link present in a candidate path is charged. |
| “route and reconstruction stages have separate queues and are charged separately” | Separate queues are executor mechanics; V1 admission uses link bytes and one conservative pool-migration row. |
| \(x_{jk}^{R},x_{jk}^{K},y_{jk}\) | Use one candidate index \(c=(j,a,p)\) and \(x_c\); this directly represents multiple pools and compatible actions. |
| \(\gamma_i\) chord price | Replace with conservative initial marginal \(w_j=P_i(L_i)-P_i(L_i-\ell_j)\). |
| \(z_i\) and idle-to-off objective bonus | Delete from controller selection. Shutdown is realized only after all dependent commits and the transition. |
| “maximize” the displayed LP | Replace with target-first minimization of migration work; maximize conservative shed only when the target is infeasible. |
| “\(\mathrm{OPT}_{LP}\) upper-bounds” exact shed | Delete for the controller LP. A separately named chord oracle may supply that loose upper bound. |
| \(v_q(j)\), \(b_r^q\), and \(c_q(j,a,k)\) | Delete from the default greedy, whose values and scarcity prices are computed once. |
| “scores the remaining moves again” | Replace with one global static candidate ordering. |
| \(\mathcal F_q\) includes current queue simulation | Delete. Greedy checks aggregate rows; packing and the event simulator validate the completed plan. |
| “Crossing power knees” as Queue-Haul default | Relabel as opt-in experimental `greedy_bundle`; it is implemented but not the default policy. |
| “priority list” implies serial service | State that order controls launch and simultaneous queue ties; selected moves otherwise launch eagerly and overlap. |
| “which move receives [route] service first” | Replace with work-conserving shared-link contention; endpoint queues use FIFO order for simultaneous arrivals. |
| “added queueing-delay bound” | Replace with service-debt budget and required-recovery validation; V1 does not predict TTFT/TPOT bounds. |
| “source power is counted only after destination is ready” | Tighten to “source power falls at atomic request-boundary commit.” |
| “If a move runs late, the source retains the session” | State that the source retains ownership until commit; a post-deadline commit receives no deadline shed credit, although simulation may continue. |
| “fixed plan” | Retain. There is no online replanning in V1. |

## Designing a Controller

Replace the first three paragraphs, from “The previous section gives” through
“charged separately in the table,” with:

> The previous section gives each compatible session action a conservative
> source-power value and a destination demand vector. With many sessions,
> actions contend for links and destination pools, so Queue-Haul selects them
> jointly.
>
> Let \(H=D-C-W\) be the migration horizon after controller delay \(C\) and the
> trailing power window \(W\). The aggregate planner has one byte row for every
> physical link \(\lambda\), one ongoing-service row for every advertised pool
> facet, optional service-debt rows, one live-KV-block row per pool, and one
> migration-time row per pool. A candidate has zero demand on rows it does not
> use. In particular,
> \[
> \sum_{c:\lambda\in\operatorname{path}(c)}b_cx_c\le B_\lambda H,
> \qquad
> \sum_{c:p(c)=p}t_cx_c\le N_pH,
> \]
> where \(N_p\) is the number of replicas in pool \(p\). The current planner has
> no separate source-stream, reconstruction, or KV-ingest row. Those phases
> remain explicit in the duration model and event simulator.
>
> These aggregate rows are selection relaxations. Queue-Haul subsequently
> packs whole actions onto replicas, where each replica must satisfy its
> service, live-KV, and migration-time limits. The event simulator then checks
> shared-link timing, endpoint queues, commits, source power, service debt, and
> recovery.

Delete the special case that omits source egress when
\(B^{\mathrm{out}}\geq\sum_kB_k\). Every link present in a candidate path is
already charged. Delete the claim that route and ingest are separate planner
rows.

## LP reference program

Replace the complete LP discussion, from “Let \(x_{jk}^{R}\)” through “gives a
lower bound,” with:

> Let \(\mathcal C\) contain the eligible candidates
> \(c=(j,a,p)\), where \(a\in\{R,K\}\). A candidate exists only when its
> compatibility checks pass and its isolated duration is at most \(H\). Let
> \(x_c\in[0,1]\), let \(A_{rc}=d_{cr}/b_r\) be its normalized demand on
> aggregate resource row \(r\), and define the conservative source value
> \[
> w_j=P_{i(j)}(L_{i(j)})-
> P_{i(j)}(L_{i(j)}-\ell_j).
> \]
> Concavity gives, for every integral selected set \(M_i\),
> \[
> P_i(L_i)-P_i\!\left(L_i-\sum_{j\in M_i}\ell_j\right)
> \geq\sum_{j\in M_i}w_j.
> \]
>
> For requested shed \(\Theta\), the reference relaxation is
> \[
> \begin{aligned}
> \underset{x\geq0}{\operatorname{minimize}}\quad&
> \sum_{c\in\mathcal C}t_cx_c\\
> \text{subject to}\quad&
> \sum_{c:j(c)=j}x_c\leq1 &&\forall j,\\
> &\sum_c A_{rc}x_c\leq1 &&\forall r,\\
> &\sum_c w_{j(c)}x_c\geq\Theta,\\
> &x_c\leq1 &&\forall c.
> \end{aligned}
> \]
> If the target is infeasible, Queue-Haul first maximizes the conservative
> additive shed and then minimizes work at that value. It rounds candidates in
> LP-value order, retains only whole actions that fit, and repairs any replica
> packing failure by dropping the least efficient conflicting action. Finally,
> it reevaluates the exact nonlinear source-power curve and validates the event
> schedule.
>
> This LP is a target-first selection reference, not an upper bound on exact
> nonlinear shed. A chord-priced LP may be evaluated separately as a loose
> upper-bound oracle, but it is not the current controller.

Delete \(\gamma_i\), \(z_i\), the off-state term, the “five resource limits”
claim, and the claim that the displayed LP is Queue-Haul’s upper bound.
Shutdown gain is deliberately absent from selection and is credited only when
execution actually drains the source before the deadline.

## Greedy Selection and Ordering Sessions

Replace the text and equations from “Solving and rounding” through “scores the
remaining moves again” with:

> The control-path policy avoids solving the relaxation. It first computes the
> normalized cost \(a_c=\sum_r A_{rc}\) and identifies the cheapest candidate
> \(c_j^\star\) for each session. It then assigns each resource the one-time
> scarcity price
> \[
> \pi_r=\max\left\{1,\sum_j A_{r c_j^\star}\right\}
> \]
> and scores every candidate by
> \[
> q(c)=
> \frac{w_{j(c)}}{\sum_r\pi_rA_{rc}}.
> \]
> Queue-Haul sorts candidates once by decreasing \(q(c)\), admits a candidate
> when its session is still unselected and every aggregate row fits, and stops
> when the conservative target is met or no candidate remains. It then performs
> replica packing and exact nonlinear power reevaluation.

Delete the dynamic quantities \(L_i^q\), \(b_r^q\), \(v_q\), \(c_q\), and
\(\mathcal F_q\). The current greedy does not update power values or resource
prices after each choice and does not append candidates to the fluid simulator
during selection.

Replace “Crossing power knees” with a short implementation-status paragraph:
prefixes of length two and three plus the full feasible instance drain are
available as the opt-in `greedy_bundle` policy. They dynamically use current
exact drain gain and remaining resource slack, but are not the default
Queue-Haul controller. Report them as an experimental policy until they are
validated on representative fleet traces.

Replace the final priority-list paragraph with:

> After selection and packing, Queue-Haul orders moves by increasing migration
> work per conservative watt:
> \[
> \sigma=\operatorname{sort}_{c\in M}\left(t_c/w_{j(c)}\right).
> \]
> This order supplies deterministic priority for simultaneous endpoint
> arrivals; it does not serialize dispatch.

## Dispatch and Realized Power

Replace the complete final block with:

> \(\noindent\textbf{Dispatch and realized power.}\)
> When planning completes at time \(C\), the executor starts every selected
> move eagerly. Independent routes proceed in parallel. Flows on overlapping
> paths share link capacity, while replay and KV work enter their endpoint
> queues with \(\sigma\) breaking simultaneous-arrival ties. The aggregate
> link-byte and pool-time rows prune impossible selections; the deterministic
> event simulator supplies the authoritative completion, queue, and debt check
> for the completed plan.
>
> Source power falls only when a request-boundary handoff commits and the
> source copy is no longer needed. If all dependent sessions leave an
> instance, shutdown gain is counted only when that state transition completes
> before the deadline. Selection uses the awake-state conservative marginal,
> so it never relies on this bonus.
>
> The controller executes a fixed plan without online replanning. A move that
> commits after the deadline receives no shed credit at the deadline, and the
> plan is reported infeasible; the simulator may continue the move to record
> its eventual outcome. The source retains ownership until commit.

Delete the claims that Queue-Haul appends each candidate to the fluid model
during greedy selection, checks a per-candidate added-delay bound, or retains a
late session indefinitely. The current validator checks the completed plan,
reports service debt and recovery rather than a predicted TTFT/TPOT bound, and
does not abort a move at \(D\).
