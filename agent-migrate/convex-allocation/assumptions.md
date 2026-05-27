Queue-haul audit assumptions
Generated: 2026-05-26

Scope assumptions
=================

A0. I treated the uploaded research journal as the intended formulation and the uploaded Python files as the current implementation. The drift ledger was treated as a set of claims to verify, not as ground truth.

A1. I audited only the files provided in this conversation. Several files referenced by the ledger were not uploaded, so claims that depend on them are marked unverified or partial in drift.txt. Missing referenced files include:
- tests/test_objective.py
- README.md
- experiments/plot_queue_centered.py
- experiments/run_queue_failure_diagnostics.py
- experiments/run_report_experiments.py
- committed outputs/sweep/generated_* CSVs or plots

A2. File names in the upload had suffixes such as coefficients(5).py and run_retained_state_frontier(4).py. In drift.txt I use the canonical paths implied by imports and by the ledger: src/*.py and experiments/*.py.

A3. I did not assume the drift ledger was complete. I checked for additional drift against the journal goals: stateful LLM session reconstruction, KV-cache locality, shared-prefix reuse, state materialization, queueing pressure, deadline/tail metrics, and power-shed simulation.

A4. External lookup was used only to sanity-check general systems context, not to validate the project's exact coefficients or experimental numbers. The external systems context checked was: DistServe for prefill/decode disaggregation, Splitwise for phase splitting, vLLM prefix caching for block-level KV reuse, and LMCache for KV reuse/offload across serving instances. These sources support the plausibility of the intended primitives but do not prove this implementation is correct.

Modeling assumptions
====================

A5. h_ctx[g,k] is interpreted as raw-context locality for class g at destination k. It can reduce replay network transfer because less raw context must be transmitted.

A6. h_kv[g,k] is interpreted as computed KV/state locality for class g at destination k. If replay can reuse resident KV blocks or cached prefixes, replay prefill work should depend on the missing KV/state fraction, not the missing raw-context fraction.

A7. The current workload generator enforces h_kv <= h_ctx. This is a plausible sampling convention but does not by itself imply that raw-context locality and KV locality can be substituted in cost equations.

A8. The key assumption behind the ledger's first blocking issue is: when some KV for a request is already resident at the destination, replay should recompute only the missing KV work. Under this assumption b_prefill[REPLAY] should be proportional to T * (1 - h_kv). If the intended semantics are instead that replay always recomputes from raw context and never reuses resident KV, then h_kv should not affect replay prefill, but the journal/code semantics need to say that explicitly.

A9. A compatible state transfer is not complete at network receipt. It also needs state ingest/materialization and decode admission. The journal explicitly distinguishes network pressure, prefill pressure, and state-ingest/decode pressure; the code currently has only network and prefill buckets.

A10. KV/state compatibility is assumed to require at least model identity, tokenizer/template identity, runtime/cache layout compatibility, positional encoding compatibility, and a way to identify blocks or prefixes. The current code has no compatibility dimension, block IDs, prefix IDs, or manifest fields.

A11. Shared-prefix reuse should be modeled separately from private suffix state. If N jobs share a prefix of length T_s and have private suffixes T_p, a once-per-prefix transfer/recompute cost is materially different from charging each job full T. The code's class-level aggregate T cannot represent that distinction.

A12. Resident-state dynamics are assumed to matter. If request j makes block set B_j resident at destination k, then later requests with overlapping blocks should be cheaper. The current code treats h_ctx and h_kv as fixed exogenous coefficients, so this dynamic is absent.

Units and service-rate assumptions
==================================

A13. lambda_Bps is interpreted as per-destination network throughput in bytes per second. In make_problem() it is derived from lambda_gbps * 1e9 / 8.

A14. rho_prefill is ambiguous in the implementation. make_problem() sets rho_prefill = model.prefill_tok_s * gpu_count, which is an aggregate fleet throughput. But compute_coefficients() divides a single request's replay prefill work by rho_prefill when computing R0/q, which treats it like a per-request service rate. This is only valid if a single prefill job can use all GPUs as a single aggregate service pool, or if the model intentionally approximates an M/G/k queue by a single server with aggregate rate. That assumption is not stated in the journal.

A15. C_net and C_prefill are capacity budgets over a window_s, not instantaneous service rates. metrics.available_rates() reconstructs a common window H from C_net/lambda_Bps and C_prefill/rho_prefill.

A16. beta is treated as bytes per token of transmitted raw/context representation. The code assumes beta=4 for the catalog models. This excludes protocol, serialization, compression, and retransmission overhead unless those are already folded into beta.

A17. eta is treated as bytes per token of KV/state payload. The code assumes a single eta per model and does not model layer-by-layer, precision, quantization, backend layout, or partial-block effects.

A18. The model catalog values are treated as local experiment constants, not independently verified facts. The audit verifies how they are used, not whether the catalog numbers are empirically correct.

Deadline and queueing assumptions
=================================

A19. deadline_s is interpreted as per-request resume/reconstruction slack. The code uses both release-relative metrics (delay > deadline_s) and absolute metrics (completion_time > deadline_s). Any report must say which one is being used.

A20. LP cumulative deadline constraints are necessary feasibility screens, not schedule certificates for the two-stage network-then-prefill queue. A cumulative network cap plus a cumulative prefill cap can both be satisfied while serial composition still misses deadlines.

A21. The queue simulator models one serial network server per destination and one serial prefill server per destination, with EDF scheduling. Any multi-GPU parallelism is absorbed into the scalar service rate rho_prefill rather than modeled as multiple servers.

A22. State transfers in the simulator use network service only. They do not consume prefill, ingest, decode, HBM-admission, or cache-materialization service.

A23. Arrival pacing is a major experimental condition. Core queue evaluation can release requests uniformly over drain_window_s, but retained-state frontier currently evaluates reconstruction queues with burst-at-zero release. Those are different stress models.

A24. Background traffic is static capacity consumption. It is not modeled as a stream of queued background jobs with deadlines, burstiness, cache locality, or preemption.

Experiment and reporting assumptions
====================================

A25. retained_prefill_target_s is a source-shed target measured in equivalent prefill seconds. It is not automatically full evacuation. With the default retained_prefill_fraction=0.4, the stay column can leave substantial source work unmoved.

A26. “Actual evacuated state” metrics in the current code are best read as equivalent resident-state bytes of moved sessions. They are not actual network bytes sent, because replay sends context bytes while state transfer sends KV/state bytes.

A27. “Request migration fraction” is a fraction of request count moved, not a fraction of tokens, state bytes, prefill seconds, or power.

A28. Power-shed claims require a power model or power traces. The uploaded optimizer/simulator can estimate reconstruction load and deadlines, but it does not directly model MW, GPU power traces, source-site power reduction, destination power caps, or electrical headroom.

A29. Integer queue execution assumes integer-valued request counts and integer-valued context lengths. The generated workload usually satisfies this after rounding/aggregation, but arbitrary ProblemData inputs may not.

A30. Rounding is part of execution, not a cosmetic postprocess. Fractional CVXPY allocations can satisfy capacity/deadline constraints while rounded allocations violate queue deadlines or capacity pressure.

A31. The uploaded source bundle is not self-contained for all experiments. Some experiment scripts import modules that were not uploaded. Conclusions about those scripts are therefore limited to static inspection of the available code.

Risk-priority assumptions
=========================

A32. Blocking model drift should be fixed before trusting report-level numerical claims. The biggest load-bearing issues are replay prefill locality, missing state-ingest/decode pressure, and aggregate-vs-per-request prefill-rate semantics.

A33. If the paper/report claims “online routing,” “state materialization,” “shared-prefix amortization,” “safe evacuation,” or “power flexibility,” those terms need to be scoped carefully unless the implementation is expanded.

A34. Existing CSVs and plots, if present elsewhere, should be treated as diagnostics until regenerated with the corrected model, explicit seed policy, explicit arrival pacing, and explicit SAFE definition.

A35. A good remediation target is not just passing current tests. Current tests may encode the wrong behavior, especially around replay prefill locality and state-transfer pressure.
