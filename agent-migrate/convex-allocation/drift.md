Queue-haul drift verification
Generated: 2026-05-26

Status legend
=============

VERIFIED     The statement is directly supported by the uploaded code/journal.
PARTIAL      The core mechanism is supported, but part of the statement depends on missing files or an interpretation.
UNVERIFIED   The uploaded files are insufficient to verify the claim.
STALE        The uploaded code contradicts the old claim; the current ledger correctly treats it as stale/resolved.
ADDED        Additional drift found during this audit.

Executive verdict
=================

Most substantive drift statements in the uploaded ledger are true. The highest-priority blockers are:

1. Replay prefill locality is inconsistent with the intended KV-locality model.
2. State transfer is modeled as network-only, with no state-ingest/decode/admission pressure.
3. R0/q mixes per-link network latency with aggregate fleet prefill rate.
4. The simulator/LP/reporting semantics differ across deadline metrics, arrival pacing, SAFE definitions, and partial-vs-full evacuation.
5. Shared-prefix amortization and resident-state dynamics are not implemented.

Several ledger items depend on files that were not uploaded. I marked those as partial or unverified instead of assuming they are true.

Verified original drift ledger
==============================

Blocking model drift
--------------------

B1. Replay prefill skips on raw-context locality.
Status: PARTIAL / VERIFIED mechanism.

Evidence:
- src/coefficients.py:45 defines one_minus_ctx = 1.0 - problem.h_ctx.
- src/coefficients.py:46 defines one_minus_kv = 1.0 - problem.h_kv.
- src/coefficients.py:52 sets b_net[:,:,REPLAY] = beta * T * one_minus_ctx.
- src/coefficients.py:53 sets b_prefill[:,:,REPLAY] = T * one_minus_ctx.
- src/coefficients.py:54 sets b_net[:,:,STATE] = eta * T * one_minus_kv.

Verdict:
- The code definitely uses raw-context locality h_ctx for replay prefill work.
- Under the journal's KV-locality/reuse semantics, replay prefill should depend on missing KV/state, i.e. h_kv, not raw-context availability.
- The specific statement that tests/test_objective.py encodes the wrong behavior is UNVERIFIED because that test file was not uploaded.

Why it matters:
- If h_ctx is high but h_kv is low, the current code incorrectly makes replay cheap in both network and prefill.
- That changes action choice, objective values, queue pressure, and any reported crossover behavior.

Fix direction:
- If replay can reuse resident KV blocks, change b_prefill[:,:,REPLAY] to T * (1 - h_kv), then update tests.
- If replay is intended to always recompute full context regardless of resident KV, document that h_kv does not affect replay and rename h_ctx/h_kv semantics accordingly.

B2. STATE is network-only.
Status: VERIFIED.

Evidence:
- src/coefficients.py:54 sets b_net[:,:,STATE] = eta * T * one_minus_kv.
- src/coefficients.py leaves b_prefill[:,:,STATE] at zero.
- src/coefficients.py:57 sets R0[:,:,STATE] = b_net[:,:,STATE] / problem.lambda_Bps.
- src/problem.py:11-27 has capacity fields for network and prefill only: lambda_Bps, rho_prefill, C_net, C_prefill, ell_net, ell_prefill. There is no C_state, mu_state, decode admission, ingest queue, or HBM-admission capacity.
- src/queueing.py:206 initializes completion after network, and src/queueing.py:210-219 only sends REPLAY jobs to prefill. STATE jobs complete after network.

Verdict:
- The ledger is correct. The journal's third pressure, state-ingest/decode, is absent.

Why it matters:
- State transfer can look much safer than it is because the code charges only network bytes, not state materialization, KV admission, decode startup, or memory pressure.

Fix direction:
- Add a third resource bucket for state ingest/materialization/decode admission, with coefficients b_state or b_ingest, capacity C_state, background ell_state, and queue simulation service.

B3. R0 mixes single-link network with aggregate-fleet prefill.
Status: VERIFIED.

Evidence:
- src/problem.py:172 computes lambda_Bps = lambda_gbps * 1e9 / 8.0, a per-destination link rate.
- src/problem.py:173 computes rho_prefill = model.prefill_tok_s * gpu_count, an aggregate destination prefill throughput.
- src/coefficients.py:56 divides one request's replay prefill work by problem.rho_prefill.
- src/coefficients.py:58 sets q = R0.
- src/objective.py:15 sums coeffs.q * x.
- src/baselines.py:68 ranks greedy moves using coeffs.q.
- src/problem.py:156-160 patches transition-coupled default gpu_count to [1,1,1] for generated workloads, but src/problem.py:135-139 uses 72 GPUs for default generated workloads outside that regime.

Numerical check from uploaded constants:
- GLM-5: beta=4 B/tok, eta=89,900 B/tok, prefill_tok_s=8,300 tok/s.
- T=100,000, lambda=25 Gbps = 3.125e9 B/s, gpu_count=72.
- Current aggregate replay R0 = beta*T/lambda + T/(rho*72) = 0.1675 s.
- State R0 = eta*T/lambda = 2.8768 s.
- Per-GPU replay R0 = beta*T/lambda + T/rho = 12.0483 s.

Verdict:
- The ledger's numeric example is correct to rounding.
- This is load-bearing because q is used by the objective and greedy baselines.

Fix direction:
- Decide whether rho_prefill is per-request service rate or aggregate capacity. If aggregate, do not use it as a single-request latency denominator without an explicit queueing approximation. Use per-worker service time in R0/q and aggregate capacity in C_prefill, or replace the single-server model with a multi-server queue.

B4. The LP does not generally recover lambda* = rho(eta-beta).
Status: PARTIAL / VERIFIED mechanism.

Evidence:
- The journal gives the single-request crossover lambda* = rho(eta - beta).
- src/catalog.py:14-16 computes that direct formula for ModelParams.crossover_gbps.
- src/problem.py:173 multiplies prefill_tok_s by gpu_count in make_problem().
- Therefore default generated LP runs can shift the effective crossover by gpu_count unless the regime patches gpu_count or the test manually supplies per-GPU rho.

Unverified part:
- The “H3 helper” and isolated test were not uploaded, so I cannot verify those references.

Verdict:
- The unit inconsistency mechanism is real.
- Claims that default LP runs recover the single-request crossover are unsafe unless the run pins rho to the intended per-request service rate.

Formulation drift
-----------------

F1. Shared-prefix amortization is absent.
Status: VERIFIED.

Evidence:
- The journal models shared state C_s and private suffix state C_{j,p|s}, with once-per-prefix amortization.
- src/workload.py:8-14 stores only T, d, deadline_s, h_ctx, h_kv.
- src/workload.py:152-178 aggregates by averaged T/deadline/locality.
- src/workload.py:213-226 summarizes/merges by mean T, mean deadline, mean h_ctx, mean h_kv.
- src/coefficients.py charges replay and state costs per moved class member using full class T.

Verdict:
- The code cannot represent shared prefix length T_s, private suffix length T_p, prefix class, block set, block hashes, or once-per-shared-prefix cost.

F2. Resident-state dynamics are static.
Status: VERIFIED.

Evidence:
- src/workload.py samples h_ctx and h_kv once during workload generation.
- src/coefficients.py consumes h_ctx and h_kv as fixed coefficients.
- No data structure updates resident block sets after routing a request.

Verdict:
- The journal equation R_k <- R_k union B_j is not implemented.
- Routing one request never makes later overlapping requests cheaper.

F3. The action space is collapsed.
Status: VERIFIED.

Evidence:
- src/coefficients.py:9-11 defines ACTIONS = ("replay", "state").
- The allocation matrix has those move columns plus a stay column.
- The journal's capability validation lists replay, local prefix reuse, cross-instance cache reuse, and state materialization as distinct paths. Active in-flight decode migration is explicitly out of scope in the journal.

Verdict:
- Local prefix reuse, cross-instance reuse, and state materialization are collapsed into exogenous h_ctx/h_kv and two actions.
- This is acceptable only if the report scopes the code as a reduced model, not a full implementation of the journal action space.

F4. “Online” baselines are fixed-batch passes.
Status: VERIFIED.

Evidence:
- src/baselines.py:89-137 implements _solve_online by creating an EDF-ordered request list from a fixed demand vector and walking it once.
- It updates local loads, but it does not receive new stochastic arrivals, update resident-state sets, or include dynamic background queues.

Verdict:
- Calling these baselines “online” is misleading unless explicitly scoped as static-order greedy baselines.

Convex relaxation drift
-----------------------

C1. Deadline LP feasibility is necessary, not sufficient.
Status: VERIFIED.

Evidence:
- src/cvxpy_solver.py:96-118 constrains cumulative network and prefill work separately by destination and deadline threshold.
- src/queueing.py:195-219 simulates serial network service followed by serial prefill service for replay jobs.

Counterexample:
- Two replay jobs have deadline D.
- Each job needs network service D/2 and prefill service D/2.
- Cumulative LP caps see total network D and total prefill D, so both caps can pass.
- Serial network-then-prefill scheduling completes job 1 at D and job 2 at 1.5D, so the second misses.

Verdict:
- The LP constraints are useful filters but not schedule certificates.

C2. Post-solve row renormalization can erase solver margin.
Status: VERIFIED as a risk.

Evidence:
- src/cvxpy_solver.py:66-67, 130-131, and 220-221 clamp y.value nonnegative and renormalize each row to sum to d after solve.
- The modeled capacity constraints use u <= 1 - eps.
- src/metrics.py:168-177 asserts feasibility only by checking utilization < 1, not by preserving u <= 1 - eps.

Verdict:
- The returned allocation can lose the solver's strict eps certificate even if it remains inside the barrier domain.
- I did not find a concrete uploaded test case where this fails; the structural risk is real.

Fix direction:
- Project onto the full feasible set, validate against 1-eps, or avoid post-solve renormalization except for numerically tiny row-sum errors with a full constraint recheck.

C3. Soft-deadline penalties weaken as constraint count grows.
Status: VERIFIED.

Evidence:
- src/cvxpy_solver.py:199-204 divides the overrun penalty by n_overrun = 2 * K * |deadlines|.

Verdict:
- Adding destinations or deadline strata lowers the marginal weight of each violation unless the weights are retuned.

C4. Slater/interior feasibility is not checked.
Status: VERIFIED.

Evidence:
- src/problem.py:106-179 constructs capacities, loads, deadlines, and retained targets but does not prove or check strict interior feasibility for the paper's relaxation.
- src/mirror_descent.py:29-81 separately tries to build a capacity-interior start and a greedy target-feasible point for its own method.

Verdict:
- Generated cases may be feasible in practice, but the code does not establish a general Slater condition.

C5. Fractional routing is a real relaxation gap.
Status: VERIFIED qualitatively; PARTIAL on the stated asymptotic bound.

Evidence:
- src/cvxpy_solver.py uses continuous cp.Variable allocations.
- src/queueing.py:72-96 rounds allocations for execution.
- src/queueing.py:367-430 uses exact dynamic-programming rounding only below a state-count cap and a greedy fallback above it.

Verdict:
- Fractional feasibility does not guarantee rounded queue feasibility.
- The ledger's Omega(max_g b_g/(rate*D)) statement is a reasonable analytical warning, but it is not directly proved by the uploaded code.

C6. Mirror descent should be scoped narrowly.
Status: VERIFIED.

Evidence:
- src/mirror_descent.py implements exponentiated-gradient style row updates with a scalar alpha bisection for the retained-prefill constraint.
- It uses the plain penalized objective/gradient, not the deadline-aware or soft-deadline formulations in src/cvxpy_solver.py.

Verdict:
- It is not a first-order solver for the deadline-aware or soft-deadline models.

C7. SolverResult.objective is current, but easy to misuse.
Status: PARTIAL / mostly VERIFIED.

Evidence:
- src/cvxpy_solver.py returns objective(problem, coeffs, y_value) for solve_cvxpy, solve_deadline_aware_cvxpy, and solve_soft_deadline_cvxpy.
- The native soft-deadline solver objective is stored separately as diagnostics["soft_deadline_problem_value"].

Correction to ledger wording:
- The old drift note that solver objectives differ across CVXPY solvers is stale.
- However, “native solver objectives live only in diagnostics” is fully true only for the soft-deadline native objective in the uploaded code. Plain solve_cvxpy does not store native prob.value separately, and deadline-aware stores retained_prefill_moved_s/objective_s diagnostics, not a generic native solver objective.

Queueing and experiment drift
-----------------------------

Q1. Default retained shed is partial, not evacuation.
Status: VERIFIED.

Evidence:
- src/problem.py:109 defaults retained_prefill_fraction to 0.4.
- Allocation includes a stay column.
- src/problem.py:178 computes retained_prefill_target_s as a fraction of total retained prefill seconds.

Verdict:
- Unless a run sets fraction=1.0, “evacuation” is inaccurate. It is a partial retained-prefill shed target.

Q2. Arrival pacing is mixed.
Status: VERIFIED, and slightly stronger than ledger wording.

Evidence:
- src/queueing.py defaults drain_window_s=1800.0 for core queue evaluators.
- src/queueing.py:503-507 and 559-607 release all jobs at zero when drain_window_s=0, otherwise uniformly by rank over the drain window.
- experiments/run_retained_state_frontier.py:66 sets QUEUE_RELEASE_SPAN_S = 0.0.
- experiments/run_retained_state_frontier.py:251-258 calls queue_metrics with drain_window_s=QUEUE_RELEASE_SPAN_S, i.e. burst-at-zero queueing.
- experiments/run_retained_state_frontier.py:352-359 then reports retained_prefill_removal_rate using the outer drain_window_s.

Verdict:
- The retained frontier is not simply “using a drain window”; it uses burst-at-zero reconstruction arrivals and separately tests whether completion fits inside the drain window. That semantic distinction should be explicit in report figures.

Q3. SAFE definitions still diverge.
Status: PARTIAL.

Evidence verified:
- experiments/run_retained_state_frontier.py:362-370 uses strict target, absolute deadline, pressure, and drain-completion checks.
- experiments/run_network_bandwidth_tradeoff.py:122-130 checks target, release-relative deadline metrics, pressure, and drain completion.

Unverified parts:
- experiments/plot_queue_centered.py, experiments/run_queue_failure_diagnostics.py, and experiments/run_report_experiments.py were not uploaded.

Verdict:
- Divergence between retained frontier and network bandwidth scripts is verified.
- The rest of the ledger's SAFE-definition comparison is unverified due missing files.

Q4. Frontier search assumes near-monotonicity.
Status: VERIFIED.

Evidence:
- experiments/run_retained_state_frontier.py:203-210 binary-searches retained_prefill_fraction.
- experiments/run_retained_state_frontier.py:211-215 samples local offsets and one stress fraction.

Verdict:
- This can miss nonlocal safe islands caused by rounding, EDF grouping, or solver action changes.

Q5. Baseline comparisons are not deadline-matrix fair.
Status: VERIFIED.

Evidence:
- src/baselines.py:117 ranks least-loaded by utilization/load proxy.
- src/baselines.py:100-118 makes the online queue greedy see EDF order and local backlog choices, not the full destination/deadline cumulative load matrix.
- src/cvxpy_solver.py:96-118 gives deadline-aware CVXPY cumulative deadline constraints for every destination and deadline threshold.

Verdict:
- Do not claim broad deadline superiority without a deadline-aware online baseline or equal information comparison.

Q6. Seed coverage is improved but still thin.
Status: PARTIAL / VERIFIED core.

Evidence:
- src/evaluation.py:11-18 defaults to seed=7, jobs=10,000, classes=48.
- src/evaluation.py:38-52 CLI defaults match those point-estimate settings.
- experiments/run_retained_state_frontier.py:65 sets WORKLOAD_SEEDS = range(16).
- experiments/run_retained_state_frontier.py:544-550 uses those seeds only through the CLI helper for generated workloads.
- experiments/run_retained_state_frontier.py:144-160 programmatic default run_retained_state_frontier() uses only workload_config.seed unless a seed list is passed.

Verdict:
- The core warning is true: report-facing defaults and many sweeps can still be point estimates.
- The statement about plots showing means/std could not be verified without the plotting code.

Q7. Fleet dynamics are static.
Status: VERIFIED.

Evidence:
- src/problem.py:126-132 samples one fixed workload.
- src/problem.py:141-177 sets fixed background load fractions.
- src/queueing.py:447-500 builds queue records only from the chosen moved allocation.

Verdict:
- No new background arrivals, cross-shed coupling, destination power dynamics, or power-target feedback are modeled.

Documentation and artifact drift
--------------------------------

D1. README “manifest” wording overstates the implementation.
Status: PARTIAL / specific README claim UNVERIFIED.

Evidence verified:
- The journal describes manifest-like fields: session/model/runtime metadata, shared prefix tokens, private suffix tokens, state bytes, deadlines, source, and optional hashes.
- The uploaded code has no manifest fields, block hashes, compatibility IDs, prefix classes, or T_s/T_p state.

Unverified part:
- README.md and experiments/plot_queue_centered.py were not uploaded, so the exact README wording and plot labels cannot be verified.

Verdict:
- The general manifest drift is real. The specific README/plot claim remains unverified.

D2. Generated artifacts are diagnostic, not evidence by themselves.
Status: UNVERIFIED as an artifact claim, but good practice.

Evidence:
- outputs/sweep/generated_* artifacts were not uploaded.

Verdict:
- I cannot verify whether committed CSVs contain unsafe, failed, or unresolved claim rows.
- The recommendation is still correct: regenerate with exact workload label, seed policy, corrected model, and explicit SAFE definition before report use.

Stale prior notes
-----------------

S1. “SAFE can mean overloaded” is stale for the current retained frontier.
Status: STALE / verified resolved for uploaded retained frontier.

Evidence:
- experiments/run_retained_state_frontier.py:362-370 checks target, absolute p95, absolute miss rate, network pressure, prefill pressure, and drain completion.

Caveat:
- This does not prove every other script uses the same SAFE definition.

S2. “Deadline-aware CVXPY can return success below target” is stale.
Status: STALE / verified resolved for uploaded deadline-aware solver.

Evidence:
- src/cvxpy_solver.py:88-90 constrains retained_prefill >= target.
- src/cvxpy_solver.py:252-254 asserts the returned allocation meets target.

S3. “Invalid queue actions silently pass” is stale.
Status: STALE / verified resolved for evaluate_static_queue path.

Evidence:
- src/queueing.py:572-574 rejects unknown queue actions in _paced_records.
- Allocation-derived actions come from the ACTIONS tuple.

S4. “ProblemData is frozen but stores mutable arrays” is stale.
Status: STALE / verified resolved.

Evidence:
- src/problem.py:82-86 copies array fields and marks them read-only.

Additional drift found in this audit
====================================

A1. Retained-prefill and state-byte metrics conflate action semantics.
Status: ADDED / VERIFIED.

Evidence:
- src/metrics.py:38-40 retained_prefill_moved_s counts tau for every moved request, regardless of replay vs state.
- src/metrics.py:47-53 resident_state_bytes and resident_state_moved_bytes count full eta*T state bytes for all moved requests.
- experiments/run_retained_state_frontier.py:327-346 reports actual_evacuated_state_tb from resident_state_moved_bytes.

Drift:
- A replayed request sends context bytes, not KV bytes. A state-transfer request sends KV/state bytes. The current “actual evacuated state” metric is really equivalent state footprint of moved sessions, not actual network payload.

Fix direction:
- Report separate metrics: context bytes sent, KV/state bytes sent, state bytes admitted/materialized, equivalent source state evacuated, and moved prefill seconds.

A2. State-transfer retained-prefill fraction is conceptually odd.
Status: ADDED / VERIFIED.

Evidence:
- src/metrics.py computes retained-prefill action mix using problem.tau for both replay and state moves.
- src/queueing.py reports replay_work and state_work using tau for both actions.

Drift:
- “state_transfer_retained_prefill_fraction” means fraction of the source retained-prefill target satisfied by moving sessions via state action. It does not mean state transfer consumed prefill service. This label can confuse readers.

Fix direction:
- Rename to moved_source_work_fraction_by_action or shed_target_fraction_by_action.

A3. Programmatic retained-frontier default does not use the advertised 16 generated seeds.
Status: ADDED / VERIFIED.

Evidence:
- experiments/run_retained_state_frontier.py:65 defines WORKLOAD_SEEDS = range(16).
- experiments/run_retained_state_frontier.py:144 sets workload_seeds = tuple(workload_seeds or (workload_config.seed,)).
- experiments/run_retained_state_frontier.py:544-550 uses range(16) only via the CLI helper.

Drift:
- Calling run_retained_state_frontier() directly uses one seed, not 16. Only the CLI path uses the multi-seed default.

Fix direction:
- Make the function default match the CLI, or require explicit workload_seeds and log the seed set in outputs.

A4. Uploaded experiment bundle is not self-contained.
Status: ADDED / VERIFIED for uploaded files.

Evidence:
- experiments/run_network_bandwidth_tradeoff.py imports experiments.plot_queue_centered._max_waiting_depth_points, but plot_queue_centered.py was not uploaded.
- experiments/run_integer_optimality_cases.py imports experiments.run_queue_failure_diagnostics.repair_rounded_allocation, but run_queue_failure_diagnostics.py was not uploaded.

Drift:
- Some uploaded scripts cannot run from the uploaded bundle alone.

Fix direction:
- Add missing modules, remove optional imports, or guard them with local fallback functions.

A5. Integer queue execution has undocumented integer preconditions.
Status: ADDED / VERIFIED.

Evidence:
- src/queueing.py:72-96 calls _integer_array on problem.d and problem.T.
- Generated workloads round T to multiples of 256 and use integer demand counts, but arbitrary ProblemData can use floats.

Drift:
- The simulator is not a generic continuous-class executor. It requires integer request counts and integer token lengths.

Fix direction:
- Document this precondition, or support noninteger T by scaling/rounding with explicit error control.

A6. Aggregated deadlines can hide urgent tails.
Status: ADDED / VERIFIED.

Evidence:
- src/workload.py:213-226 averages deadline_s inside each aggregate bucket.
- All class members are later treated as having that mean deadline.

Drift:
- Mean deadlines can make a class appear less urgent than its urgent members. This directly affects deadline-aware constraints and queue miss metrics.

Fix direction:
- Aggregate by deadline quantiles or store per-class deadline distribution, at least min/p50/p95 or EDF bands.

A7. The optimizer has no source-side bottleneck or source egress model.
Status: ADDED / VERIFIED.

Evidence:
- ProblemData has destination capacities and per-destination lambda_Bps, but no shared source egress capacity, source network queue, or WAN fabric coupling.

Drift:
- The model can overload a hypothetical source egress link or shared fabric without noticing, especially when many destinations are used at once.

Fix direction:
- Add source egress and optional shared network-resource constraints.

A8. Power flexibility is not actually modeled.
Status: ADDED / VERIFIED.

Evidence:
- ProblemData has no MW, GPU power, source power target, destination power headroom, or power trace fields.
- The retained target is prefill-seconds moved, not watts or joules.

Drift:
- The code can support reconstruction/load-movement experiments, but not direct claims that changing sites produces a measured power reduction.

Fix direction:
- Add a calibrated source/destination power model or measured power traces before claiming a power-shed frontier.

A9. Background traffic lacks queue/deadline interaction.
Status: ADDED / VERIFIED.

Evidence:
- src/problem.py uses fixed ell_net and ell_prefill fractions.
- src/queueing.py builds queue records only for the selected moved allocation.

Drift:
- Background traffic only reduces capacity; it never contributes burstiness, queue waiting, deadline misses, cache effects, or preemption conflicts.

Fix direction:
- Either label background as static capacity reservation or add sampled background arrival processes.

A10. State compatibility is hidden inside scalar h_kv.
Status: ADDED / VERIFIED.

Evidence:
- src/workload.py stores h_kv as a numeric matrix only.
- There are no manifest fields, block IDs, prefix hashes, model/template IDs, or compatibility constraints.

Drift:
- The router can send state to a destination with high h_kv without proving the specific blocks are compatible or sufficient.

Fix direction:
- Represent block sets/prefix IDs and compatibility constraints explicitly, at least in the testbed manifest.

A11. Soft-deadline solver returns a different objective than it optimizes.
Status: ADDED / VERIFIED.

Evidence:
- src/cvxpy_solver.py:145-239 optimizes risk + barriers + soft deadline penalties.
- It returns SolverResult.objective = objective(problem, coeffs, y_value), which excludes the soft-deadline penalty.
- The full native solved value is placed in diagnostics["soft_deadline_problem_value"].

Drift:
- Comparing SolverResult.objective for the soft-deadline solver to another policy does not compare the actual optimization objective that selected the allocation.

Fix direction:
- Return both base_objective and optimized_objective explicitly.

A12. Deadline-aware CVXPY maximizes moved retained prefill, not latency, after feasibility.
Status: ADDED / VERIFIED.

Evidence:
- src/cvxpy_solver.py:121 builds cp.Problem(cp.Maximize(retained_prefill), constraints).
- If retained_prefill_cap is set to the target, it finds a feasible allocation at the cap rather than minimizing latency/risk among feasible allocations.

Drift:
- The allocation may be arbitrary among feasible target-meeting solutions except for solver numerical preferences. It is not a latency-minimizing deadline-aware policy unless followed by a second-stage objective.

Fix direction:
- Use lexicographic optimization: first maximize retained target or enforce target, then minimize base objective / queue proxy / deadline slack.

A13. Objective risk is mean/base-time-like, not tail-latency calibrated.
Status: ADDED / VERIFIED.

Evidence:
- src/objective.py:15 uses sum(q*x), with q=R0.
- Tail metrics p95/p99/deadline miss are only evaluated in queueing.py after allocation, or approximated by deadline constraints/penalties in specific solvers.

Drift:
- A low objective does not necessarily imply low p95/p99 reconstruction delay, especially after rounding and two-stage queueing.

Fix direction:
- Treat the base objective as a routing cost proxy, and validate/report tail metrics separately.

A14. The model assumes independent destination links and no shared WAN contention.
Status: ADDED / VERIFIED.

Evidence:
- lambda_Bps is a vector indexed by destination.
- Capacity constraints sum by destination only.

Drift:
- If links share a source uplink, metro fabric, or backbone bottleneck, the model can overestimate feasible migration volume.

Fix direction:
- Add shared resource constraints for source egress and network groups.

A15. available_rates() assumes all resource capacities imply one common window.
Status: ADDED / VERIFIED.

Evidence:
- src/metrics.py:12-17 concatenates C_net/lambda_Bps and C_prefill/rho_prefill and raises if they are not all close to the same window.

Drift:
- Custom problems with resource-specific windows or measurement horizons will fail. This is fine if intentional, but it is an unstated modeling restriction.

Fix direction:
- Document common-window assumption or generalize rate reconstruction per resource.

A16. Generated workload aggregation can smear locality across incompatible sessions.
Status: ADDED / VERIFIED.

Evidence:
- src/workload.py:152-178 merges buckets using distance over log T, log deadline, h_ctx, and h_kv.
- There is no session identity, prefix identity, model compatibility, or block-set identity.

Drift:
- A class with averaged h_kv can represent sessions that do not share actual resident blocks. This makes locality probabilistic and can overstate reusable state if interpreted deterministically.

Fix direction:
- Use stochastic interpretation explicitly, or aggregate only by compatible prefix/block classes.

A17. Network bandwidth tradeoff uses transition-coupled gpu_count patch, so it does not exercise the default 72-GPU aggregate issue.
Status: ADDED / VERIFIED.

Evidence:
- experiments/run_network_bandwidth_tradeoff.py:52 builds make_problem(..., regime="transition-coupled", retained_prefill_fraction=1.0).
- src/problem.py:156-160 sets transition-coupled generated gpu_count to [1,1,1] when gpu_count is default.

Drift:
- This sweep avoids the aggregate-prefill bug by regime-specific patching. Other generated regimes still use gpu_count=72.

Fix direction:
- State which sweeps are protected by the transition-coupled patch and which are not.

A18. Full evacuation and source drain are not enforced by the stay column unless retained_prefill_fraction=1.0 and target metric matches all source work.
Status: ADDED / VERIFIED.

Evidence:
- Allocation row sums allow stay column y[:,-1].
- The retained target is a scalar lower bound, not a per-source zero-residual constraint.

Drift:
- A policy can meet the retained-prefill target while leaving strategically important or long-running sessions unmoved.

Fix direction:
- Add optional constraints for no-stay, class-specific movement requirements, or source power/resource evacuation constraints.

Recommended remediation order
=============================

1. Fix replay prefill locality or explicitly redefine replay semantics.
2. Add state-ingest/decode/admission pressure as a separate resource and queue stage.
3. Separate per-request latency service rates from aggregate capacity rates.
4. Rename/report action-specific bytes and moved-source-work metrics correctly.
5. Add shared-prefix/private-suffix state and block/prefix compatibility if claims require it.
6. Make deadline safety semantics consistent: absolute vs release-relative, burst vs paced, drain-completion checks, and pressure checks.
7. Add a deadline-aware online baseline or weaken deadline-superiority claims.
8. Regenerate all report figures with explicit seeds, median/IQR, corrected model, and explicit SAFE definition.
9. Add source/network shared bottlenecks and background arrival processes if fleet claims require them.
10. Add power model or measured power traces before claiming a power-shed frontier.
