Assuming the drift is fixed, the clean story is:

**Queue-haul is a convex snapshot allocator for stateful reconstruction under resource pressure.**
It should choose *where* each retained session resumes and *how* its state is reconstructed, while respecting shed targets, destination bottlenecks, deadline envelopes, and measured service curves.

The ideal convex version should not try to directly model every event in the queue. It should solve a **convex relaxation** that produces a resource-aware allocation, then validate that allocation with the discrete queue simulator and real-serving measurements.

## 1. The ideal convex formulation

### Decision variables

Let:

* (g \in G): workload class or session group.
* (k \in K): destination site.
* (a \in A): reconstruction action.
* (r \in R): resource type.

Actions should include at least:

[
A = {\text{replay}, \text{local-prefix-reuse}, \text{cross-instance-cache-reuse}, \text{state-materialize}}
]

plus a stay/no-move column.

Use continuous variables:

[
x_{gka} \ge 0
]

where (x_{gka}) is the number, or fractional number, of class-(g) sessions assigned to destination (k) using action (a).

Also use:

[
s_g \ge 0
]

for sessions that stay at the source.

This matches the current CVXPY structure conceptually: a nonnegative allocation matrix (y), movement variables (x), row-sum preservation, and a retained-prefill/shed target. The current solver already has the basic shape of this relaxation: nonnegative allocation, per-class demand conservation, retained-prefill target, resource utilization constraints, and log-barrier resource penalties. 

### Demand conservation

Every class must either move or stay:

[
s_g + \sum_{k,a} x_{gka} = d_g
]

This is important because the formulation is not “migrate everything.” It is “move enough work to satisfy a shed target while preserving service.” The previous drift audit correctly warned that partial shed and evacuation need to be named separately. 

### Resource costs

For each class, destination, action, and resource, precompute:

[
b^r_{gka}
]

Examples:

* Network bytes.
* Prefill tokens.
* State-ingest bytes or state-load work.
* Decode-admission pressure.
* HBM/cache residency pressure.
* Optional CPU/NVMe load for cache materialization.

Then total destination load is affine:

[
L_{kr}(x) = \ell_{kr} + \sum_{g,a} b^r_{gka}x_{gka}
]

where (\ell_{kr}) is background load.

Capacity constraints:

[
L_{kr}(x) \le (1-\epsilon)C_{kr}
]

The key change from the current simplified version is that **state transfer must have its own receiving-side pressure**, not just network pressure. The research notes already frame the real problem as network plus receiving-side recomputation/materialization pressure, not just bytes in flight. 

### Shed target

Use a linear target:

[
\sum_g v_g\sum_{k,a}x_{gka} \ge S
]

where (v_g) can be:

* retained prefill seconds,
* source HBM state removed,
* source GPU power reduction contribution,
* source decode capacity freed,
* or a weighted combination.

For early experiments, use retained-prefill seconds because it is already present in the code and easy to reason about. The retained-state frontier code already sweeps drain windows and retained-prefill fractions, which is the right experimental shape. 

For power-facing experiments, replace or augment (v_g) with estimated source power relief over a shed interval.

### Objective

A clean objective is:

[
\min_x
\sum_{g,k,a} q_{gka}x_{gka}
+
\sum_{k,r} w_{kr}\phi\left(\frac{L_{kr}(x)}{C_{kr}}\right)
+
\text{deadline penalties}
]

where:

[
\phi(u) = -\log(1-u)
]

for (0 \le u < 1).

This is convex because (L_{kr}(x)) is affine and (-\log(1-u)) is convex on (u<1). Your current objective already uses this basic risk-plus-barrier structure: linear reconstruction risk plus log barriers on network and prefill utilization. 

The important improvement is to generalize the barrier over all real resource buckets:

[
r \in {\text{network}, \text{prefill}, \text{state-ingest}, \text{decode-admission}, \text{cache-memory}}
]

not just network and prefill.

### Deadline envelope

The dangerous part is deadlines. A pure LP deadline cap is not enough if jobs must pass through serial stages, like network then prefill. The audit already found the failure mode: separate cumulative caps can pass while the serial queue still misses deadlines. 

For the ideal convex version, I would use **conservative deadline envelopes** rather than pretending the convex program fully simulates the queue.

For each deadline threshold (D), destination (k), and resource (r):

[
\sum_{g: D_g \le D}\sum_a b^r_{gka}x_{gka}
\le
\alpha_{kr}D \cdot \mu_{kr}
]

where:

* (\mu_{kr}) is the service rate for resource (r) at destination (k),
* (\alpha_{kr}\in(0,1]) is a fixed stage budget,
* and (\sum_r \alpha_{kr} \le 1) for serial stages.

For example, replay might need:

[
\alpha_{\text{net}}D
]

for network completion and:

[
\alpha_{\text{prefill}}D
]

for prefill completion.

This is conservative, but it is convex and easier to defend. Then the queue simulator validates the rounded allocation.

You can also use soft deadline slacks:

[
z_{kDr} \ge
\sum_{g: D_g \le D}\sum_a b^r_{gka}x_{gka}
------------------------------------------

\alpha_{kr}D\mu_{kr}
]

[
z_{kDr} \ge 0
]

and penalize:

[
\sum_{k,D,r} \gamma_1 z_{kDr} + \gamma_2 z_{kDr}^2
]

That keeps the program convex.

### Shared-prefix and resident-state locality

For an ideal implementation, fixed hit rates are not enough. The interesting version should include shared-prefix structure.

Let (p) index reusable prefix/state blocks. Add a continuous relaxation variable:

[
z_{pk} \in [0,1]
]

meaning “prefix/state block (p) is materialized at destination (k)” in the relaxed sense.

Then for class (g) that depends on prefix (p(g)), constrain:

[
x_{gka} \le d_g z_{p(g)k}
]

and charge once-per-prefix materialization cost:

[
\sum_{p,k} c_{pk}z_{pk}
]

This is still convex if (z) is continuous. It is a relaxation of the true discrete cache placement problem, but that is fine. The rounded/MPC implementation can materialize whole prefixes.

This directly fixes the earlier formulation gap: the drift audit noted that the old workload state only had (T,d,\text{deadline},h_{\text{ctx}},h_{\text{kv}}), with no prefix class, block set, or once-per-shared-prefix cost. 

### Receding-horizon control

Do not try to make the whole dynamic system one giant convex program. That will either become ugly or false.

Use this instead:

1. Freeze current queues, resident states, background load, and service curves.
2. Solve the convex snapshot problem.
3. Execute a small slice of the allocation.
4. Update queues and resident state.
5. Re-solve.

That gives you a clean convex program at every decision point while still handling dynamic locality and queue pressure.

## 2. What this convex formulation should prove

The ideal implementation should prove four things:

1. **The optimizer changes reconstruction actions for the right reason.**
2. **The optimizer uses bottleneck shadow prices, not hand-coded heuristics.**
3. **The convex relaxation is close enough to executable rounded policies.**
4. **The policy converts state movement into safe shed capacity or power flexibility.**

The experiments should be built around those claims.

## 3. Best experiments for the convex version

### Experiment 1: Convex phase diagram with shadow prices

**Hypothesis:**
The convex optimizer recovers the replay/state crossover in the single-request, uncongested case, then bends that boundary under congestion because network, prefill, and state-ingest resources acquire different shadow prices.

In the simple case, replay wins when:

[
\frac{\beta T}{\lambda} + \frac{T}{\rho}
<
\frac{\eta T}{\lambda}
]

so:

[
\lambda^* = \rho(\eta-\beta)
]

This crossover is already central in the project notes. 

**Ideal experiment:**

Sweep:

* network bandwidth,
* prefill rate,
* state-ingest rate,
* cache hit rate,
* background load,
* retained-prefill target.

For each point, plot:

* replay fraction,
* state-materialization fraction,
* cache-reuse fraction,
* max network utilization,
* max prefill utilization,
* max state-ingest utilization,
* dual price of each resource,
* objective value,
* queue-simulated deadline miss rate.

**Success signal:**
In the low-load case, the decision boundary matches the analytical crossover. Under congestion, the boundary shifts according to dual prices.

The strongest figure would show something like:

[
\text{effective replay cost}
============================

q_{\text{replay}}
+
\pi_{\text{net}}b^{\text{net}}*{\text{replay}}
+
\pi*{\text{prefill}}b^{\text{prefill}}_{\text{replay}}
]

versus:

[
\text{effective state cost}
===========================

q_{\text{state}}
+
\pi_{\text{net}}b^{\text{net}}*{\text{state}}
+
\pi*{\text{ingest}}b^{\text{ingest}}_{\text{state}}
]

where (\pi) are dual prices. That is much better than “we placed jobs.”

### Experiment 2: Resource-price explanation experiment

**Hypothesis:**
The convex solution is interpretable: when a destination bottleneck tightens, the corresponding dual price rises, and the optimizer routes around it.

**Ideal experiment:**

Construct a three-destination setup:

* Destination A: high network, low prefill.
* Destination B: low network, high prefill.
* Destination C: medium everything, but high background decode load.

Run the same workload under increasing shed target.

Report:

* chosen destination/action mix,
* resource dual prices,
* marginal cost per moved retained-prefill second,
* p95/p99 resume delay after queue simulation.

**Success signal:**
As prefill becomes scarce, replay actions decline. As network becomes scarce, state-transfer/cache-materialization actions decline. As decode admission becomes scarce, even cheap reconstruction paths avoid that destination.

This is the cleanest way to show the convex formulation is doing real bottleneck reasoning.

### Experiment 3: Safe shed frontier

**Hypothesis:**
The convex formulation can find the maximum safe shed frontier better than replay-only, state-only, least-loaded, and online-greedy policies.

**Ideal experiment:**

For each drain window:

[
H \in {10s, 20s, 50s, 100s, 300s, 1000s, 1800s}
]

binary search the largest shed target (S) such that the rounded allocation passes:

* retained target met,
* network pressure below 1,
* prefill pressure below 1,
* state-ingest pressure below 1,
* p95 delay/deadline below threshold,
* deadline miss rate below threshold,
* drain completes within the window.

The current retained-frontier code already has the right structure: it sweeps drain windows, policies, release policies, workload seeds, and retained-prefill fractions, and reports frontier quantities like max safe retained-prefill fraction, deadline miss rate, pressure, action mix, and drain completion. 

**Success signal:**
The convex policy should not win everywhere. It should win in the mixed regimes where neither all-replay nor all-state is right.

That is the compelling result: **queue-haul expands the safe operating frontier in the regimes where reconstruction choices matter.**

### Experiment 4: Adversarial deadline heterogeneity

**Hypothesis:**
A convex deadline-aware allocator beats load-aware policies when average capacity is sufficient but the wrong classes are sent to the wrong resources.

**Ideal setup:**

Two destinations:

* A: fast network, weak prefill.
* B: slower network, strong prefill.

Two classes:

* urgent short-context sessions,
* long-context slack sessions.

A naive least-loaded policy will look fine by aggregate utilization but will create deadline misses by sending urgent replay work into the weak prefill pool.

**Compare:**

* convex queue-haul,
* replay-only,
* state-only,
* least-loaded,
* network-greedy,
* crossover-greedy,
* exact integer oracle for small cases.

The existing integer-optimality code already compares CVXPY-rounded policies, repaired rounded policies, greedy policies, replay-only, state-only, and exact integer objective/queue optima on small cases. 

**Success signal:**
The convex policy should match or approach the integer oracle on small cases, then scale to large cases where exact enumeration is impossible.

### Experiment 5: Convex relaxation gap and rounding stress

**Hypothesis:**
The convex relaxation becomes practically executable when classes are small enough, but it can fail near deadline/resource cliffs if classes are too coarse.

**Ideal experiment:**

Vary class granularity:

* 10 classes,
* 25 classes,
* 50 classes,
* 100 classes,
* per-request classes on small workloads.

For each granularity, compare:

* fractional convex objective,
* rounded objective,
* repaired rounded objective,
* exact integer objective for small cases,
* queue-simulated miss rate.

Metrics:

* objective gap,
* deadline miss gap,
* target shortfall,
* over-capacity events,
* runtime,
* largest per-class resource chunk relative to deadline capacity.

**Success signal:**
Show a clean convergence story: as class chunks get smaller, rounded solutions approach the convex relaxation.

This matters because the drift audit identified fractional routing as a real relaxation gap: CVXPY routes continuous class fractions, while execution rounds later. 

### Experiment 6: Shared-prefix herding

**Hypothesis:**
When sessions share reusable prefixes or state blocks, the convex relaxation should intentionally cluster related sessions at the same destination if the reuse benefit beats congestion cost.

**Ideal setup:**

Generate prefix-overlap graphs:

* no overlap,
* pairs,
* star,
* chain,
* dense cluster,
* mixed real-like clusters.

Compare:

* convex queue-haul without prefix variables,
* convex queue-haul with prefix variables,
* least-loaded,
* random spread,
* replay-only,
* all-cache/state.

Metrics:

* prefix materializations,
* cache hit rate,
* bytes moved,
* prefill tokens recomputed,
* p95/p99 resume latency,
* deadline misses,
* destination entropy,
* safe shed frontier.

**Success signal:**
With shared prefixes, the prefix-aware convex policy should herd related sessions. With no overlap, it should stop herding. That contrast is the evidence.

This is one of the best “this is not generic scheduling” experiments.

### Experiment 7: Robust convex queue-haul

**Hypothesis:**
A robust convex version sacrifices some shed capacity but stays safe under coefficient error.

**Ideal formulation:**

Instead of exact coefficients (b^r_{gka}), use uncertainty sets:

[
b^r_{gka} \in [\hat b^r_{gka}(1-\delta), \hat b^r_{gka}(1+\delta)]
]

Then solve with inflated demands:

[
\tilde b^r_{gka} = \hat b^r_{gka}(1+\delta_r)
]

or with separate uncertainty budgets by resource.

**Experiment:**

Solve under estimated coefficients, simulate under perturbed true coefficients.

Perturb:

* prefill rate,
* network throughput,
* state-ingest rate,
* cache hit probability,
* background load,
* deadline slack.

Metrics:

* safe probability,
* frontier loss,
* p95/p99 delay,
* miss rate,
* objective regret.

**Success signal:**
The robust policy should have a smaller nominal frontier but a much higher pass rate under measurement error.

This is important because the project’s own notes say the simulator should consume measured service curves rather than invent behavior. 

### Experiment 8: Dual-price capacity planning

**Hypothesis:**
The convex program’s dual variables can predict which added resource would most improve shed capacity.

**Ideal experiment:**

At a fixed workload and shed target, solve the convex problem and record dual prices for:

* network,
* prefill,
* state-ingest,
* decode admission,
* cache memory.

Then add a small amount of one resource:

* +10% network,
* +10% prefill,
* +10% state-ingest,
* +10% decode admission,
* +10% cache capacity.

Re-solve and compare actual objective/frontier improvement to the dual prediction.

**Success signal:**
The largest dual price should identify the best resource upgrade.

This is a very strong systems result because it turns the optimizer into an operator tool: it says whether the shed is blocked by WAN, prefill, KV loading, decode, or memory.

### Experiment 9: Slack-aware receding-horizon routing

**Hypothesis:**
Agent/tool-call slack lets the controller hide reconstruction work and increase safe shed without increasing user-visible resume latency.

**Ideal setup:**

Use dynamic sessions with:

* user think time,
* tool-call pauses,
* retrieval pauses,
* code-execution pauses,
* multi-agent wait states.

At each control epoch, solve the convex snapshot with release times and slack windows. Execute a small slice, update queues and resident state, then re-solve.

Metrics:

* visible TTFT inflation,
* hidden reconstruction work,
* deadline miss rate,
* p95/p99 resume latency,
* source power shed,
* queue buildup.

The research notes already point to profiling slack in agents and multi-agent settings as part of validating the action space. 

**Success signal:**
Slack-aware queue-haul should move more work at the same visible latency than a router that assumes every reconstruction is immediately user-blocking.

### Experiment 10: Real serving primitive calibration

**Hypothesis:**
The convex formulation is only credible if its action coefficients come from measured serving paths.

**Ideal experiment:**

Measure service curves for:

1. Replay.
2. Local prefix reuse.
3. Cross-instance cache reuse.
4. State materialization.

The notes already define these four paths as capability validation, and explicitly say active in-flight decode migration is out of scope. 

For each action, measure:

* network bytes,
* prefill tokens,
* state-load time,
* TTFT,
* p95/p99 latency,
* cache hit/miss behavior,
* output equivalence or acceptable stochastic equivalence,
* GPU memory pressure,
* CPU/NVMe pressure if applicable.

**Success signal:**
The convex coefficients are not hand-waved. They come from measured service curves, and the simulator uses those curves.

## 4. What I would implement first

The clean implementation order should be:

1. **General resource tensor**

[
B[g,k,a,r]
]

with resources:

[
{\text{network}, \text{prefill}, \text{state-ingest}, \text{decode}, \text{cache-memory}}
]

2. **General action set**

[
A = {\text{replay}, \text{local-reuse}, \text{cross-cache-reuse}, \text{state-materialize}}
]

3. **Convex target solver**

Minimize linear risk plus resource barriers subject to row conservation, shed target, action feasibility, and capacity constraints.

4. **Deadline-envelope solver**

Add hard or soft conservative deadline envelopes.

5. **Prefix-relaxation solver**

Add continuous prefix materialization variables (z_{pk}).

6. **Rounding and repair**

Round continuous allocations, then repair to satisfy target and resource constraints.

7. **Queue simulator validation**

Use the queue simulator as the execution validator, not as the convex proof.

8. **Receding-horizon controller**

Re-solve periodically with updated queues, resident states, and background load.

## 5. The strongest paper narrative

The ideal paper claim should be:

> Queue-haul gives a convex, interpretable relaxation for stateful LLM session relocation. It converts a shed target into reconstruction choices across replay, cache reuse, and state materialization, using measured service curves and resource shadow prices. The convex solution is validated by queue simulation and rounded execution, and it exposes when the true bottleneck is network, prefill, state ingest, decode admission, or cache memory.

That is much stronger than “we placed jobs in time.”

The best figures would be:

1. **Replay/state/cache phase diagram with dual prices.**
2. **Safe shed frontier versus drain window.**
3. **Bandwidth/prefill/state-ingest sensitivity curves.**
4. **Adversarial deadline heterogeneity case.**
5. **Convex relaxation gap versus class granularity.**
6. **Prefix-aware herding experiment.**
7. **Robustness under service-curve error.**
8. **Dual-price capacity planning validation.**

If you only have room for three: do the **phase diagram**, **safe shed frontier**, and **relaxation-gap/oracle validation**. Those prove the math, the systems value, and the honesty of the convex relaxation.


Here is the sharper experimental story I would use:

**Queue-haul should not mainly demonstrate “we placed jobs before deadlines.”** That is table stakes. It should demonstrate that the formulation correctly decides **how to reconstruct state** under competing bottlenecks: send context and replay, move KV/state, exploit locality, avoid prefill storms, and convert that into safe power/load shedding. Your own notes frame the problem exactly this way: KV is large, context is smaller, replay saves network but creates prefill pressure, and the hard case is many stateful jobs competing for network and destination-side reconstruction capacity. 

Before using these as report-facing claims, fix the blocking coefficient issues. The drift audit says replay prefill currently uses raw-context locality instead of KV residency, STATE is modeled as network-only, and the LP crossover is unit-inconsistent because per-link network is mixed with aggregate prefill capacity. Those bugs directly affect any experiment about replay/state choice. 

## Best hypotheses and experiments

### 1. **Action phase diagram: replay vs state transfer is not a rule of thumb**

**Hypothesis:** Queue-haul recovers the simple replay/state crossover in the uncongested single-request limit, but under multi-request congestion the boundary shifts because replay consumes destination prefill and state transfer consumes network/state-ingest capacity.

**Experiment:** Build a controlled sweep over network bandwidth `λ`, prefill rate `ρ`, KV size `η`, context size `β`, background prefill load, and retained-state target. Start with one class and one destination to verify the analytic crossover:

[
\lambda^* = \rho(\eta - \beta)
]

Then scale to many classes and destinations with heterogeneous deadlines and background loads. Plot action mix: replay fraction, state-transfer fraction, destination entropy, max network utilization, max prefill utilization, p95 resume delay, and deadline miss rate.

**Baselines:** all-replay, all-state, crossover greedy, least-loaded, online queue greedy, random.

**Success signal:** In the no-load case, the optimizer flips near the analytic crossover. Under load, the phase boundary moves in the expected direction: replay becomes less attractive when prefill is congested, and state transfer becomes less attractive when network/state-ingest is congested.

**Why it is strong:** This proves the formulation is making the core economic tradeoff, not just finding a feasible assignment. It also connects cleanly to adjacent systems work: DistServe and SplitWise both motivate separating prefill/decode resources because the phases have distinct bottlenecks and SLO effects. ([arXiv][1]) ([Microsoft][2])

**Status:** High value, but only valid after fixing the coefficient/unit drift.

---

### 2. **Prefill-storm avoidance under burst shedding**

**Hypothesis:** A locality-aware mixed policy can shed more source work than all-replay without creating a prefill storm at the destination.

**Experiment:** At `t = 0`, force a burst drain from one source into three destinations. Sweep the retained-prefill target from low to high and compare policies under fixed drain windows: 10s, 20s, 50s, 100s, 300s, 1000s. Report the **maximum safe retained-prefill fraction** under strict safety: target met, absolute deadline miss rate ≤ 1%, p95 delay/deadline ≤ 1, network pressure ≤ 1, prefill pressure ≤ 1, and drain completion within the window. The current retained-frontier script already uses this kind of strict SAFE definition. 

**Baselines:** all-replay, all-state, least-loaded, online queue, deadline-aware CVXPY, and exact oracle on small cases.

**Success signal:** Queue-haul should support a larger safe frontier than all-replay when prefill is tight, and larger than all-state when network is tight. It should not win everywhere; the interesting result is showing **where** and **why** it wins.

**Why it is strong:** It turns the paper’s “prefill storm” motivation into a measurable frontier: how much source work can be safely drained before tail latency collapses.

**Important caveat:** Current retained-frontier code uses burst-at-zero in that script, while core queue evaluation can also use paced releases. The report must label burst vs paced arrivals explicitly. 

---

### 3. **Bandwidth value curve: more network can change the reconstruction primitive**

**Hypothesis:** Increasing inter-site bandwidth does not only reduce delay; it changes the optimal reconstruction action mix. Low bandwidth should favor replay/context movement, high bandwidth should favor KV/state movement, and middle regimes should use both.

**Experiment:** Sweep network bandwidth scale, e.g. `0.25×` to `2×`, while holding prefill capacity and workload fixed. For each bandwidth, find the largest safe retained-prefill fraction and record action mix, network pressure, prefill pressure, actual evacuated state TB, p95 delay, and deadline misses. Your existing `run_network_bandwidth_tradeoff.py` is already close to this shape: it sweeps network scale and retained-prefill fraction, then records largest safe fraction, evacuated state, request migration fraction, queue depths, deadline metrics, and action fractions. 

**Success signal:** The curve should show a real knee: below some bandwidth, state transfer is too expensive; above it, replay becomes wasteful because it burns prefill. The best policy should adapt across that knee.

**Why it is strong:** This gives a clean “operator knob” result. It answers: *What is an extra 25/100/400 Gbps worth for stateful migration?*

**Status:** Strong after coefficient fixes and after adding state-ingest capacity if you want to claim true state-transfer realism.

---

### 4. **Deadline-aware routing beats load-aware routing in adversarial heterogeneity**

**Hypothesis:** Least-loaded routing fails when average capacity is fine but deadline classes and resource bottlenecks are mismatched. Queue-haul should route by deadline, action cost, and resource pressure jointly.

**Experiment:** Construct a two-destination adversarial case:

Destination A: fast network, weak prefill.
Destination B: slower network, strong prefill.

Workload has two classes:

Urgent short-context jobs.
Long-context slack jobs.

The trap is that a naive least-loaded policy sends urgent replay jobs to a destination that looks underloaded but creates a serial prefill bottleneck. Queue-haul should assign urgent jobs to the path with better end-to-end completion, not just lower utilization.

**Metrics:** p95/p99 resume delay, absolute deadline miss rate, action/destination heatmap, objective gap to exact integer queue optimum.

**Success signal:** Queue-haul should reduce misses versus least-loaded and online greedy in this constructed case. On small versions, compare to exact integer oracle; your integer-optimality scripts already enumerate target-feasible allocations and compare policies against best objective and best queue allocation. 

**Why it is strong:** It demonstrates the formulation sees the *deadline matrix*, not just “capacity remaining.”

**Caveat:** The audit notes current online baselines are fixed-batch EDF passes, not real event-time routers with new arrivals and dynamic background traffic. Do not call them production online baselines unless upgraded. 

---

### 5. **Shared-prefix herding: routing can create future locality**

**Hypothesis:** If sessions share prefixes or reusable state, routing overlapping sessions to the same destination can be better than spreading them evenly, because the first reconstruction creates resident state that later requests can reuse.

**Experiment:** Add explicit prefix groups: `shared_prefix_tokens`, `private_suffix_tokens`, prefix/block IDs, and destination resident sets. Generate cohorts with overlap graphs: star, chain, dense cluster, and no-overlap control. Compare:

random spread,
least-loaded spread,
static queue-haul without resident updates,
dynamic queue-haul with resident updates.

After each routed request, update:

[
R_k \leftarrow R_k \cup B_j
]

and recompute future replay/state costs.

**Metrics:** bytes moved, prefill tokens recomputed, cache hit rate, p95 resume latency, deadline misses, destination entropy, and total safe shed fraction.

**Success signal:** Dynamic queue-haul should intentionally “herd” related sessions when the future reuse benefit exceeds congestion cost. In no-overlap controls, it should stop herding.

**Why it is strong:** This is probably the most uniquely queue-haul experiment. It shows routing decisions have externalities.

**Status:** Requires model extension. The drift audit says shared-prefix amortization and resident-state dynamics are currently absent: workload state only has `T`, `d`, `deadline_s`, `h_ctx`, and `h_kv`; there is no prefix class, block set, or once-per-shared-prefix cost. 

---

### 6. **Slack harvesting for agents and tool calls**

**Hypothesis:** Agent/tool-call pauses create hidden time windows where state can be replayed, loaded, or materialized before the next user-visible turn. A slack-aware router should reduce visible TTFT inflation compared with a router that treats every migration as immediately blocking.

**Experiment:** Use synthetic or trace-derived agent sessions with pauses from tool calls, code execution, retrieval, browser actions, and user think time. At shed time, jobs have different slack windows. Let the router decide whether to:

do immediate replay,
ship/cache state now,
prewarm during slack,
defer reconstruction,
or leave the job at source.

**Metrics:** user-visible resume latency, TTFT inflation, hidden reconstruction work completed under slack, deadline misses, prefill queue depth, state-ingest queue depth, and source power reduction.

**Success signal:** Slack-aware routing should move more retained work at the same visible p95/p99 latency, especially for agentic workloads with long tool waits.

**Why it is strong:** This moves the work beyond generic LLM serving into the stateful/agentic setting your journal actually cares about. Your notes already flag profiling slack in different regimes and multi-agent/tool-call pauses as an important experimental direction. 

**Status:** Requires dynamic arrivals and slack fields. Current simulator has static fleet dynamics and fixed background load fractions, so this cannot be fully claimed with the existing formulation alone. 

---

### 7. **Capability validation: prove the abstract actions correspond to real serving paths**

**Hypothesis:** The formulation’s action space is only credible if replay, local prefix reuse, cross-instance KV reuse, and state materialization have measurable, stable service curves on the serving stack.

**Experiment:** Single-instance and two-instance testbed:

Replay: send conversation context to a new instance and compare output equivalence and TTFT.
Local prefix reuse: route repeated turns to the same vLLM instance and measure prefix-cache hit behavior.
Cross-instance reuse: use LMCache or similar to make KV available across compatible instances.
State materialization: measure the cost to load cached state into the destination serving path.

**Metrics:** output equivalence rate, TTFT, replay prefill time, KV load time, bytes moved, cache hit/miss rate, CPU/GPU memory pressure, and variance across context lengths.

**Success signal:** Each abstract action gets a measured service curve with confidence intervals. The simulator should consume those curves rather than hand-picked constants.

**Why it is strong:** It prevents the paper from being “just an optimizer.” Your own notes already name these four paths as capability validation experiments and say active in-flight decode migration is out of scope. 

External systems make this direction credible: PagedAttention/vLLM demonstrates KV-cache management and sharing within/across requests, LMCache exposes KV caches across engines for offload and reuse, and Mooncake frames KV cache as a first-class scheduling resource in disaggregated serving. ([arXiv][3]) ([arXiv][4]) ([arXiv][5])

---

### 8. **Relaxation gap and rounding stress test**

**Hypothesis:** The convex relaxation is reliable when classes are fine-grained, but near deadline/resource cliffs, rounding can create misses even when the fractional solution looks feasible.

**Experiment:** Generate small exact-oracle cases and larger rounded cases. Sweep:

number of classes,
requests per class,
class size skew,
deadline tightness,
retained-prefill target,
network/prefill slack.

Compare fractional CVXPY, rounded CVXPY, repaired rounded CVXPY, exact integer objective optimum, and exact integer queue optimum.

**Metrics:** objective gap, queue gap, deadline miss gap, target shortfall, over-capacity events, and runtime.

**Success signal:** Show where the relaxation is excellent and where it fails. For report-facing claims, define the safe operating region: e.g. max class work relative to deadline capacity.

**Why it is strong:** This is not a flashy “we win” experiment, but it is scientifically honest and will make the work much more credible.

**Status:** Very important because the drift audit explicitly says fractional routing has a real relaxation gap and large classes near deadline caps can violate schedule constraints after rounding. 

---

### 9. **Robustness to coefficient error**

**Hypothesis:** Queue-haul remains safe under moderate measurement error in `η`, `ρ`, `β`, bandwidth, and background load if it uses headroom; without headroom it overfits the service model.

**Experiment:** Solve using estimated coefficients, then simulate using perturbed “true” coefficients. Perturb one dimension at a time and jointly:

KV bytes/token ±10–50%,
prefill rate ±10–50%,
network bandwidth ±10–50%,
background load spikes,
cache hit-rate error.

Run with and without safety headroom.

**Metrics:** safe probability, deadline miss rate, p95/p99 delay inflation, objective regret, and amount of shed work preserved.

**Success signal:** There should be a visible tradeoff: more headroom reduces maximum shed fraction but increases safety under model error.

**Why it is strong:** Real deployments will not know perfect service curves. This experiment makes the formulation operational rather than brittle.

---

### 10. **Model-architecture sensitivity**

**Hypothesis:** The right migration policy changes across model architectures because KV size and prefill speed vary. A single “always replay” or “always move KV” rule is wrong.

**Experiment:** Run the same workload over a model grid: small KV/fast prefill, large KV/slow prefill, large KV/fast prefill, etc. Use real model catalog values where available, but also include synthetic extremes to stress the formulation. Plot safe shed frontier and action mix by model and bandwidth.

**Metrics:** replay/state fraction, max safe retained-prefill fraction, bytes moved, prefill tokens recomputed, deadline misses, and frontier area under curve.

**Success signal:** Different models should produce different optimal policies. For example, a huge-KV model should avoid state transfer under weak network; a slow-prefill model should avoid replay under prefill pressure.

**Why it is strong:** It shows queue-haul is not just a scheduling heuristic. It is parameterized by actual model architecture.

**Status:** Good after fixing the per-GPU vs aggregate prefill denominator bug.

---

## Experiments I would prioritize first

I would do these in this order:

1. **Coefficient-corrected phase diagram**
   Proves the core math is right.

2. **Adversarial deadline/resource heterogeneity**
   Proves the optimizer is doing something smarter than least-loaded routing.

3. **Burst retained-state frontier**
   Produces the clearest systems result: maximum safe drain/shed frontier.

4. **Capability validation on real serving paths**
   Proves replay/state/cache actions are not fictional.

5. **Shared-prefix herding after extending the model**
   This is the most novel experiment, but current code cannot honestly support it yet.

## Claims to avoid until fixed

Do not claim “state transfer is best” or “replay is best” from the current code. STATE is network-only today, and replay prefill locality is wrong. 

Do not call the default retained-prefill target “evacuation.” The audit says default retained shed is partial and the stay column allows unmoved work. 

Do not claim deadline LP feasibility is sufficient for execution. The audit gives a simple serial network-then-prefill counterexample where cumulative LP caps pass but queue completion misses. 

Do not claim online generality from the current online baselines. They are fixed-batch EDF passes, not event-time routers with changing arrivals, background traffic, and resident-state updates. 

## The cleanest paper narrative

The best story is:

**Queue-haul turns stateful LLM migration into a bottleneck-aware reconstruction problem.** It does not merely place jobs. It chooses between replaying context, moving state, exploiting cache locality, and using slack, under network, prefill, state-ingest, and deadline pressure. The strongest experiments are the ones that force those choices to change and then show the policy changes for the right reason.

[1]: https://arxiv.org/abs/2401.09670?utm_source=chatgpt.com "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
[2]: https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/?utm_source=chatgpt.com "Splitwise: Efficient generative LLM inference using phase splitting - Microsoft Research"
[3]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[4]: https://arxiv.org/abs/2510.09665?utm_source=chatgpt.com "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"
[5]: https://arxiv.org/abs/2407.00079?utm_source=chatgpt.com "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
