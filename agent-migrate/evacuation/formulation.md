# Staged Convex Evacuation Program for Stateful LLM Jobs

## Problem Setting

A source site must be cleared of all active LLM inference jobs within a fixed deadline. Each job carries stateful context (a KV cache and a compact token-level representation) that must be reconstructed at a destination site by one of two actions: **replay** (send compact context, recompute KV via prefill) or **state transfer** (send the KV cache directly, ingest at destination). The optimizer computes a fractional evacuation allocation over the deadline window: how many jobs move, where they move, by which action, which resources bind, and which jobs remain infeasible. Destination instances are warm for specific models, so prefill and state-ingest capacity are partitioned by model. The program is solved as a sequence of convex problems, each fixing the previous optimum.

---

## 1. Sets and Indices

| Symbol | Description |
|--------|-------------|
| $m \in \mathcal{M}$ | Models. Each model defines a distinct serving runtime (weights, tokenizer, attention architecture). |
| $q \in \mathcal{Q}$ | Workload classes. Each class is a (model, token-length bucket) pair. $\mathcal{Q}_m \subset \mathcal{Q}$ denotes the classes belonging to model $m$, and $m(q)$ the model of class $q$. |
| $\ell \in \mathcal{L}$ | Destination sites, excluding the evacuating source. |
| $a \in \mathcal{A} = \{R, S\}$ | Actions: replay ($R$) or state transfer ($S$). |

---

## 2. Parameters

### 2.1 Per-class parameters

Each workload class $q \in \mathcal{Q}_m$ is characterized by:

| Symbol | Units | Description |
|--------|-------|-------------|
| $n_q$ | jobs | Number of class-$q$ jobs to evacuate from the source. |
| $T_q$ | tokens | Effective token length per job. This already reflects shared-prefix policy: private suffix length if the shared prefix is resident at destination, full context length if not. Shared-prefix handling is preprocessing, not a decision inside the optimizer. |
| $\beta_q$ | B/tok | Context representation size per token (compact token-level encoding, typically 4 B/tok for uint32 token IDs). |
| $\eta_q$ | B/tok | KV cache size per token (model- and architecture-dependent; ranges from ~10 KiB/tok for MLA to ~192 KiB/tok for full MHA). |
| $\rho_q$ | tok/s | Single warm-instance prefill throughput for class $q$, log-interpolated from benchmark-derived per-model prefill curves at context length $T_q$. If destinations have different hardware, generalize to $\rho_{q\ell}$; the convex structure is unchanged. |

Note: $\beta_q$ and $\eta_q$ are constant within a model ($\beta_q = \beta_m$, $\eta_q = \eta_m$ for $q \in \mathcal{Q}_m$). Only $T_q$ and $\rho_q$ vary across classes within the same model.

The derived **prefill time** per replay job is:

$$\tau_q^{\text{pfill}} = \frac{T_q}{\rho_q} \quad \text{(GPU-seconds per job)}$$

### 2.2 Per-destination-model parameters

| Symbol | Units | Description |
|--------|-------|-------------|
| $W_{\ell m}$ | instances | Available warm GPU-instances for model $m$ at destination $\ell$, already net of background load. These serve both replay (prefill) and state-transfer (ingest) for model $m$. |
| $\mu_{\ell m}^{\text{ing}}$ | B/s | Per-instance state-ingest rate for model $m$ at destination $\ell$. For a PCIe host-staged path: $\mu_{\ell m}^{\text{ing}} = \text{TP}_{\ell m} \cdot \text{BW}_{\text{PCIe}}$, where $\text{TP}_{\ell m}$ is the tensor-parallel degree and $\text{BW}_{\text{PCIe}}$ is the per-GPU host-to-device bandwidth (~64 GB/s for H100 PCIe Gen5). |

Convention: $W_{\ell m}$ is the single source of capacity for both prefill and ingest. Background load is reserved before optimization by setting $W_{\ell m}$ to the effective available count. Do not apply additional utilization reductions to derived capacities.

### 2.3 Per-destination network parameter

| Symbol | Units | Description |
|--------|-------|-------------|
| $\Lambda_\ell$ | B/s | Effective network ingress bandwidth into destination $\ell$ (or effective source-to-$\ell$ path capacity), net of background. |

### 2.4 Global parameters

| Symbol | Units | Description |
|--------|-------|-------------|
| $D$ | seconds | Evacuation deadline (fixed input, not a variable). |
| $d^{\text{miss}}$ | seconds | Reconstruction-cost penalty assigned to unmoved jobs. Declared operational convention (e.g., $D$ or the next recovery window $2D$), not a tuning weight. |

---

## 3. Deadline-Window Capacities

**Network (per destination, shared across models):**

$$C_\ell^{\text{net}} = \Lambda_\ell \cdot D \quad \text{(bytes)}$$

**Prefill (per destination and model):**

$$C_{\ell m}^{\text{pfill}} = W_{\ell m} \cdot D \quad \text{(GPU-seconds)}$$

**State-ingest (per destination and model):**

$$C_{\ell m}^{\text{ing}} = W_{\ell m} \cdot \mu_{\ell m}^{\text{ing}} \cdot D \quad \text{(bytes)}$$

All capacities are derived from $W_{\ell m}$, $\mu_{\ell m}^{\text{ing}}$, $\Lambda_\ell$, and $D$. No additional utilization fractions are applied.

---

## 4. Resource Coefficients

The per-job resource footprint of each action:

| Coefficient | Replay ($a = R$) | State transfer ($a = S$) |
|-------------|:-:|:-:|
| $b_{q}^{\text{net},a}$ | $\beta_q T_q$ | $\eta_q T_q$ |
| $b_{q}^{\text{pfill},a}$ | $T_q / \rho_q$ | $0$ |
| $b_{q}^{\text{ing},a}$ | $0$ | $\eta_q T_q$ |

Replay sends compact context ($\beta_q T_q$ bytes) over the network, then consumes $T_q / \rho_q$ GPU-seconds of prefill on a warm instance of model $m(q)$. State transfer sends the full KV cache ($\eta_q T_q$ bytes) over the network and ingests the same volume at a warm instance of model $m(q)$.

---

## 5. Decision Variables

$$x_{q\ell}^R \geq 0, \quad x_{q\ell}^S \geq 0 \qquad \forall\, q \in \mathcal{Q},\; \ell \in \mathcal{L}$$

Fractional job counts: $x_{q\ell}^R$ is the number of class-$q$ jobs replayed at destination $\ell$; $x_{q\ell}^S$ is the number state-transferred.

$$z_q \geq 0 \qquad \forall\, q \in \mathcal{Q}$$

Unmoved jobs: $z_q$ is the number of class-$q$ jobs not reconstructed by the deadline. This is not a normal action; it is an infeasibility certificate.

---

## 6. Compatibility Constraints

If destination $\ell$ cannot reconstruct class $q$ via action $a$ (no warm instances, model incompatibility, missing runtime, tokenizer/version mismatch, or policy restriction), impose:

$$x_{q\ell}^a = 0$$

At minimum:

$$W_{\ell,\, m(q)} = 0 \implies x_{q\ell}^R = x_{q\ell}^S = 0$$

This avoids divide-by-zero in pressure terms and makes model compatibility explicit.

---

## 7. Aggregate Loads

**Network (per destination, summing over all classes):**

$$L_\ell^{\text{net}}(x) = \sum_{q \in \mathcal{Q}} \left[\beta_q T_q \, x_{q\ell}^R + \eta_q T_q \, x_{q\ell}^S\right]$$

**Prefill (per destination and model, summing over classes of that model):**

$$L_{\ell m}^{\text{pfill}}(x) = \sum_{q \in \mathcal{Q}_m} \frac{T_q}{\rho_q} \, x_{q\ell}^R$$

**State-ingest (per destination and model, summing over classes of that model):**

$$L_{\ell m}^{\text{ing}}(x) = \sum_{q \in \mathcal{Q}_m} \eta_q T_q \, x_{q\ell}^S$$

All are affine in $x$.

---

## 8. Unloaded Per-Job Reconstruction Times

For replay:

$$c_{q\ell}^R = \frac{\beta_q T_q}{\Lambda_\ell} + \frac{T_q}{\rho_q}$$

For state transfer:

$$c_{q\ell}^S = \frac{\eta_q T_q}{\Lambda_\ell} + \frac{\eta_q T_q}{\mu_{\ell,\, m(q)}^{\text{ing}}}$$

The state-transfer cost uses the **per-instance** ingest rate $\mu_{\ell m}^{\text{ing}}$, not the aggregate rate $W_{\ell m} \cdot \mu_{\ell m}^{\text{ing}}$. A single job ingests onto one warm instance. The aggregate $W_{\ell m}$ belongs in the capacity constraint (Section 9), not the per-job cost.

These are surrogates for single-job reconstruction time under no contention, not completion-time guarantees under congestion.

---

## 9. Base Constraints

**Conservation (per class):**

$$\sum_{\ell \in \mathcal{L}} \left(x_{q\ell}^R + x_{q\ell}^S\right) + z_q = n_q \qquad \forall\, q \in \mathcal{Q}$$

**Network capacity (per destination):**

$$L_\ell^{\text{net}}(x) \leq C_\ell^{\text{net}} \qquad \forall\, \ell \in \mathcal{L}$$

**Prefill capacity (per destination and model):**

$$L_{\ell m}^{\text{pfill}}(x) \leq C_{\ell m}^{\text{pfill}} \qquad \forall\, \ell \in \mathcal{L},\; m \in \mathcal{M}$$

**State-ingest capacity (per destination and model):**

$$L_{\ell m}^{\text{ing}}(x) \leq C_{\ell m}^{\text{ing}} \qquad \forall\, \ell \in \mathcal{L},\; m \in \mathcal{M}$$

**Source egress (optional):** If the source has a shared outbound pipe with capacity $C_0^{\text{egress}}$:

$$\sum_{\ell \in \mathcal{L}} L_\ell^{\text{net}}(x) \leq C_0^{\text{egress}}$$

If omitted, state explicitly that $\Lambda_\ell$ already accounts for source-side limits.

**Bounds:**

$$0 \leq x_{q\ell}^a \leq n_q, \qquad 0 \leq z_q \leq n_q$$

---

## 10. Normalized Resource Pressure

The pressure index set is:

$$\mathcal{I} = \{(\ell, \text{net}) : \ell \in \mathcal{L}\} \cup \{(\ell, m, \text{pfill}) : \ell \in \mathcal{L},\, m \in \mathcal{M},\, C_{\ell m}^{\text{pfill}} > 0\} \cup \{(\ell, m, \text{ing}) : \ell \in \mathcal{L},\, m \in \mathcal{M},\, C_{\ell m}^{\text{ing}} > 0\}$$

For each index with positive capacity:

$$p_\ell^{\text{net}}(x) = \frac{L_\ell^{\text{net}}(x)}{C_\ell^{\text{net}}}, \qquad p_{\ell m}^{\text{pfill}}(x) = \frac{L_{\ell m}^{\text{pfill}}(x)}{C_{\ell m}^{\text{pfill}}}, \qquad p_{\ell m}^{\text{ing}}(x) = \frac{L_{\ell m}^{\text{ing}}(x)}{C_{\ell m}^{\text{ing}}}$$

Zero-capacity pairs are excluded from $\mathcal{I}$; the corresponding $x$ variables are zeroed by compatibility constraints. With $|\mathcal{L}|$ destinations and $|\mathcal{M}|$ models, $|\mathcal{I}| \leq |\mathcal{L}| + 2|\mathcal{L}||\mathcal{M}|$.

---

## 11. Per-Class Average Reconstruction Cost

$$r_q(x, z) = \frac{1}{n_q} \left[\sum_{\ell \in \mathcal{L}} \left(c_{q\ell}^R \, x_{q\ell}^R + c_{q\ell}^S \, x_{q\ell}^S\right) + d^{\text{miss}} \, z_q\right]$$

The average unloaded reconstruction cost for class $q$, including the declared penalty for unmoved jobs. This is used in the fairness stages (Stages 3–4) because it discriminates between assignments even when all are individually deadline-feasible.

**Alternative for deadline-violation stages:** If individual assignments can exceed $D$ (i.e., $c_{q\ell}^a > D$ for some pairs), define per-assignment deadline violation $\delta_{q\ell}^a = \max(c_{q\ell}^a - D, 0)$ and use $h_q(x,z) = (1/n_q)[\sum_\ell (\delta_{q\ell}^R x_{q\ell}^R + \delta_{q\ell}^S x_{q\ell}^S) + d^{\text{miss}} z_q]$ in place of $r_q$. In our experiments, $\delta_{q\ell}^a = 0$ for all assignments, so $h_q$ collapses to the unmoved penalty alone and $r_q$ is strictly more informative.

---

## 12. Staged Objective

The program is solved as a sequence of convex problems. Each stage introduces one new objective and fixes the optimal value of all previous stages as constraints. The stages are not combined into a weighted scalar.

---

### Stage 1: Maximize evacuated jobs (LP)

$$\min_{x, z} \quad Z(x, z) = \sum_{q \in \mathcal{Q}} z_q$$

subject to the base constraints (Section 9) and compatibility constraints (Section 6).

**Output:** $Z^\star$. The quantity $N - Z^\star$ is the maximum number of jobs that can be evacuated within deadline $D$ under the aggregate resource model, where $N = \sum_q n_q$. If $Z^\star = 0$, full evacuation is feasible in the convex relaxation.

---

### Stage 2: Minimize peak normalized pressure (LP)

$$\min_{x, z, \phi} \quad \phi$$

subject to the base constraints, the Stage 1 optimality constraint

$$\sum_{q} z_q = Z^\star,$$

and the pressure-ceiling constraints

$$p_i(x) \leq \phi \qquad \forall\, i \in \mathcal{I}.$$

**Output:** $\phi^\star$. Among all plans that move the maximum number of jobs, $\phi^\star$ is the lowest achievable worst-case resource utilization.

---

### Stage 2b (optional): Minimize congestion potential (QP)

$$\min_{x, z} \quad \Psi(x) = \frac{1}{2} \sum_{i \in \mathcal{I}} \left(p_i(x)\right)^2$$

subject to the base constraints, $\sum_q z_q = Z^\star$, and $p_i(x) \leq \phi^\star$ for all $i \in \mathcal{I}$.

**Output:** $\Psi^\star$. A convex quadratic tie-breaker among equally good evacuation and peak-pressure plans. This is a congestion potential, not an exact sum of queuing delays. The QP should not replace Stages 1 or 2.

---

### Stage 3: Minimize worst-class average reconstruction cost (LP)

$$\min_{x, z, H} \quad H$$

subject to the base constraints, $\sum_q z_q = Z^\star$, $p_i(x) \leq \phi^\star$ for all $i \in \mathcal{I}$, and optionally $\Psi(x) \leq \Psi^\star$, plus:

$$r_q(x, z) \leq H \qquad \forall\, q \in \mathcal{Q}.$$

**Output:** $H^\star$. Among all plans with the same evacuation count and peak pressure, $H^\star$ is the lowest achievable worst-class average reconstruction cost.

---

### Stage 4: Minimize variation across classes (LP)

$$\min_{x, z, V} \quad V$$

subject to all previous-stage constraints, $r_q(x,z) \leq H^\star$ for all $q$, and

$$r_q(x, z) - \bar{r}(x, z) \leq V \qquad \forall\, q \in \mathcal{Q},$$

$$\bar{r}(x, z) - r_q(x, z) \leq V \qquad \forall\, q \in \mathcal{Q},$$

where $\bar{r}(x, z) = \frac{1}{N} \sum_{q} n_q \, r_q(x, z)$.

**Output:** $V^\star$.

---

## 13. Complete Lexicographic Summary

The core formulation:

$$\operatorname{lexmin}_{x, z} \quad \left(\sum_q z_q, \;\; \phi, \;\; H, \;\; V\right)$$

With the optional QP tie-breaker:

$$\operatorname{lexmin}_{x, z} \quad \left(\sum_q z_q, \;\; \phi, \;\; \Psi(x), \;\; H, \;\; V\right)$$

Stages 1, 2, 3, and 4 are LPs. Stage 2b is a QP. The full pipeline is LP → LP → (optional QP) → LP → LP.

---

## 14. Diagnostic Overload Problem

If $Z^\star > 0$, a separate diagnostic identifies which resources break under forced full evacuation.

**Force** $z_q = 0$ for all $q$. Introduce normalized overload slacks $s_i \geq 0$ for each $i \in \mathcal{I}$ and a minimax bound $\sigma \geq 0$. Replace hard capacity constraints with:

$$p_i(x) \leq 1 + s_i, \qquad s_i \leq \sigma \qquad \forall\, i \in \mathcal{I}.$$

**Phase 1:** Solve $\min \sigma$. Output: $\sigma^\star$.

**Phase 2:** Fix $\sigma = \sigma^\star$. Solve $\min \sum_i s_i$. Output: per-index overload slacks.

Interpretation: $s_\ell^{\text{net}} > 0$ means destination $\ell$'s network is insufficient. $s_{\ell m}^{\text{pfill}} > 0$ means model $m$'s prefill pool at $\ell$ is insufficient. $s_{\ell m}^{\text{ing}} > 0$ means model $m$'s ingest capacity at $\ell$ is insufficient.

Both phases are LPs.

---

## 15. Endogenous Crossover (Stage 2b KKT)

At the QP optimum, stationarity gives the condition under which replay is preferred over state transfer for class $q$ (with $m = m(q)$) at destination $\ell$:

$$\frac{T_q}{\rho_q} \cdot \frac{L_{\ell m}^{\text{pfill}}}{(C_{\ell m}^{\text{pfill}})^2} < (\eta_q - \beta_q) T_q \cdot \frac{L_\ell^{\text{net}}}{(C_\ell^{\text{net}})^2} + \eta_q T_q \cdot \frac{L_{\ell m}^{\text{ing}}}{(C_{\ell m}^{\text{ing}})^2}$$

The crossover depends on model architecture (through $\eta_q$ and $\rho_q$), model-specific pool utilization (through $L_{\ell m}^{\text{pfill}}$ and $L_{\ell m}^{\text{ing}}$), and shared network load (through $L_\ell^{\text{net}}$). It is endogenous to the optimization.

---

## 16. Dual Interpretation (Stage 2)

Dualizing the pressure-ceiling constraints with multipliers $\pi_i \geq 0$ separates the problem by class. Each class $q$ (with $m = m(q)$) routes to the cheapest compatible $(\ell, a)$ pair at prices:

$$\text{eff. cost}_{q\ell}^R(\pi) = \pi_\ell^{\text{net}} \frac{\beta_q T_q}{C_\ell^{\text{net}}} + \pi_{\ell m}^{\text{pfill}} \frac{T_q / \rho_q}{C_{\ell m}^{\text{pfill}}}$$

$$\text{eff. cost}_{q\ell}^S(\pi) = \pi_\ell^{\text{net}} \frac{\eta_q T_q}{C_\ell^{\text{net}}} + \pi_{\ell m}^{\text{ing}} \frac{\eta_q T_q}{C_{\ell m}^{\text{ing}}}$$

For mirror descent, normalize by class: $y_{q\ell}^a = x_{q\ell}^a / n_q$, $y_q^0 = z_q / n_q$. Each class lies on a simplex $\sum_{\ell,a} y_{q\ell}^a + y_q^0 = 1$, making entropic mirror descent natural.

---

## 17. Post-Optimization Pipeline

### 17.1 Rounding

The fractional solution is converted to integer job counts. For each class $q$, all categories participate in rounding:

$$\{x_{q\ell}^R\}_\ell, \quad \{x_{q\ell}^S\}_\ell, \quad z_q$$

1. Take floors of all categories.
2. Compute remaining jobs from the conservation deficit.
3. Assign remaining units by largest fractional remainder, rejecting assignments that would violate capacity.

Including $z_q$ in the rounding is essential: without it, rounding can evacuate more jobs than $Z^\star$ allows and violate capacity constraints.

### 17.2 Queue simulator

The rounded integer solution is evaluated in a discrete-event simulator with serial stages:

- Replay path: network transfer → prefill (on a warm instance of model $m(q)$).
- State-transfer path: network transfer → state ingest (on a warm instance of model $m(q)$).

Report: $p_{50}$, $p_{90}$, $p_{99}$ completion times; deadline miss rate; queue depth over time; resource utilization over time; gap between convex-predicted pressure and simulated completion behavior.

---

## 18. Stated Assumptions

1. **Fixed batch.** All jobs to evacuate are known at the start of the deadline window.
2. **Fixed deadline.** $D$ is supplied as input; the optimizer does not choose the evacuation horizon.
3. **No source stay.** The source site is excluded from $\mathcal{L}$.
4. **Grouped workload.** Jobs are grouped by model and effective token-length bucket.
5. **Two reconstruction actions.** Replay and state transfer are the only actions in the convex core.
6. **Warm model instances are given.** $W_{\ell m}$ is fixed input. The optimizer does not decide where to place model weights.
7. **Prefill is model-specific.** Replay consumes $T_q / \rho_q$ GPU-seconds, not raw tokens.
8. **Prefill and ingest are partitioned by model.** A warm instance for model $m$ cannot serve another model.
9. **Network is destination-level.** $\Lambda_\ell$ is effective ingress bandwidth, shared across all models.
10. **Background load is already reserved.** $W_{\ell m}$ and $\Lambda_\ell$ are available capacities for the evacuation window.
11. **Shared-prefix handling is preprocessing.** The optimizer receives effective token lengths. It does not update resident cache state.
12. **Compatibility is explicit.** State transfer requires compatible runtime, tokenizer, architecture, and model version. Incompatible pairs are masked out.
13. **State-ingest is modeled, not assumed slack.** The constraint is always present. Whether it binds is an experimental outcome, not an assumption.
14. **Fractional relaxation.** The convex solution is fractional. Integer assignments are recovered by rounding, which includes $z_q$ as a category.
15. **Unloaded reconstruction times are surrogates.** $c_{q\ell}^a$ uses the per-instance ingest rate $\mu_{\ell m}^{\text{ing}}$, not the aggregate rate. These are not queue completion-time guarantees.
16. **Prefill rates are benchmark-derived scenario parameters.** The $\rho_q$ values are interpolated from measured or estimated per-model prefill curves. They should not be treated as universal constants.
17. **Schedule validity is external.** The queue simulator evaluates serial stages, batching, queue order, and decode-side effects excluded from the convex model.
