# Queue-Haul Next Steps - 2026-07-06

The static story is done enough: active-knee MILP selects the right source-node removals and the DES separates selected, egress-realized, and rebuild-realized relief. The next phase is experimental. Target venue: NSDI.

## Locked Decisions

Settled 2026-07-05; do not relitigate without new information.

| Axis | Decision |
|---|---|
| Hardware | One multi-GPU node. |
| Models | Two-tier. Largest model that fits whole-node (Qwen3-235B FP8 if it fits) for single-instance service curves. Small MoE (Qwen3-30B-A3B) for the multi-instance testbed, with eta/rho/G refit for it. |
| Stack | vLLM + LMCache. |
| Paper bar | Full testbed demo required: 1 source + 3 destination instances on partitioned GPUs. |
| Router | Online router is required and is the new contribution: per-arrival decisions from observable queue pressure, slack, and cache locality. |
| Shed semantics | Both regimes in one sweep: partial shed (node-knee selection) is the mainline; full evacuation is the target=100% endpoint. |
| MILP/DES role | Static active-knee MILP = clairvoyant oracle baseline. DES = policy gym where online policies are developed before touching GPUs. |
| Power evidence | Per-GPU DCGM/nvidia-smi traces grouped by instance, plus per-phase power calibration. Node-knee claims stay model-side but calibrated. |
| Workload | Synthetic session driver implementing the four assumptions.md classes. |

## Track 0 - Gating Sim Fixes - Done

Implemented 2026-07-06. These no longer block Track 2.

Landed:

1. `W` semantics: rebuild capacity now uses the destination spare pool, not a dedicated reconstruction pool.
2. `simulate(mode=...)` and planner deadline filters hard fail on unknown modes.
3. Target basis is standardized to full node-expected removable power.
4. Planner-side rebuild cushion `kappa < 1` has a sensitivity plot.
5. No-wait egress + rebuild lower-bound deadline filters apply to LP/MILP and whole-job baselines.
6. Stage windows before startup ramps clamp to zero, not negative capacity.
7. Movement bandwidth and utilization parameters hard fail outside their physical domains.
8. Active-knee LP/MILP exhaustively enumerates small source-node active regions and hard fails above the exhaustive cap.

Remaining modeling caveat: rebuild and post-rebuild serving still share the same spare pool only through aggregate headroom rows and the DES admission check; detailed time-varying background serving contention is Track 3/4 work.

## Track 1 - Single-Instance Capability Validation (big model)

Establishes the action space and measures the service curves. Split it into a curves-first Stage 1a and an LMCache capability Stage 1b so GPU runs can start without building a second profiling stack.

**Stage 1a - Curves-only vLLM profiling.** Implemented as `queue-haul/stage1_curves.py`, a thin runbook wrapper around `powertrace-sim`'s existing vLLM probe stack. It emits decode, prefill, and mixed-grid probe commands, auto-prunes default prefill lengths to the served context window, and leaves raw bundles under `queue-haul/runs/stage1/` when executed.

Measured first:

- `rho_dest(T)`: prefill tok/s vs context, with default context lengths pruned by `--max-model-len`.
- Decode ceiling `G`: decode tok/s vs concurrency from `decode_staircase`.
- TTFT vs context for replay/prefill-style requests.
- Per-phase power traces from the delegated `powertrace-sim` bundles.

Stage 1a deliverables:

- Powertrace-style raw bundles for decode, prefill, and mixed-grid probes.
- A short artifact index: runbook path, bundle paths, model, TP, context window, and extra vLLM args.
- No constants patch to `power.py`/`impact.py` until measured artifacts are reviewed.

**Stage 1b - LMCache capability and ingest.** Keep this separate from Stage 1a. Main measurements use one whole-node vLLM+LMCache instance; cross-instance KV reuse is a compatibility probe run serially or on smaller partitions if the big model cannot run twice on the node.

Action paths to validate:

1. **Replay fidelity.** Resend full context, recompute KV via prefill, and verify greedy next-token equivalence under fixed model/tokenizer/template/sampling config.
2. **Local prefix reuse.** Repeated turns to the same instance; measure prefix-cache hits, misses, evictions, and TTFT delta.
3. **Cross-instance KV reuse.** KV produced by one instance, consumed by a second compatible instance via LMCache shared backend; verify greedy next-token equivalence and hard failures for model, tokenizer, template, precision, KV-layout, missing-block, and corrupt-cache mismatches.
4. **State materialization.** Cost to load cached state into the serving path: bytes/s ingest, time-to-decode-admission, at several KV sizes.

Stage 1b adds:

- `mu_in`: state ingest B/s, host-staged.
- `alpha_in`: prefill slowdown while KV ingest is active.
- TTFT for KV-load paths.
- Startup ramps `tau_pre`, `tau_in`.
- A short pass/fail table for replay, local reuse, cross-instance reuse, and expected incompatibility failures.

## Track 2 - Online Router v0

New component. Interface first, policy second.

**Session manifest** (handed up at movement time):

```text
session_id, model_id/tokenizer_id/template_id
T, shared_prefix_tokens Ts, private_suffix_tokens Tp
estimated_context_bytes = beta*T, estimated_kv_bytes = eta*T
deadline_or_slack, source_node
optional prefix/block hashes
```

**Policy v0:** at each source session's next-invocation time, score each destination from observable queue depth (prefill and ingest), session slack, and cache locality (resident prefix blocks); pick destination and replay-vs-KV jointly, i.e. the min over the two cost estimates from the formulation. Selection for partial shed uses live node-marginal power density on the source side.

**Development loop:** build and tune entirely in the DES policy gym (post Track 0 fixes), scored against the static active-knee MILP oracle solved with full information at t=0. The gap to oracle is itself a paper figure.

## Track 3 - Shed Testbed (small model)

The paper's centerpiece demo.

Setup:

- 4 vLLM+LMCache instances on partitioned GPUs: 1 source, 3 destinations. Destinations carry background traffic.
- Inter-site links emulated with network namespaces + tc shaping; per-link bandwidths swept to cover replay-dominant, KV-dominant, and mixed regimes.
- Synthetic session driver generates live multi-turn stateful load in the four classes (chat, long chat/code, reasoning, agentic tool loop) with assumptions.md turn-rate/Delta/Y/T parameters.
- At t=0 the source receives a shed command; sweep the target from partial up to full evacuation.

Policy arms: random, network-greedy, least-loaded, all-replay, all-KV, online router (Track 2), static-MILP oracle plan executor.

Metrics: deadline misses, p95/p99 resume latency, TTFT inflation, destination queue buildup, network bytes, cache hit rate, per-GPU power traces (source drop, destination rise).

## Track 4 - Fleet Simulator Re-Parameterization

Feed Track 1 measured curves and Track 3 observed queueing behavior back into the DES/fleet simulator. Deliverable: the safe power-shed frontier per policy - maximum source power reduction achievable before p95/p99 resume or deadline-miss constraints break.

## Dependencies

```text
Track 0 gates Track 2 (gym must be trustworthy before tuning policies in it)
Track 1 gates Track 4 (frontier needs measured curves)
Track 2 gates Track 3's main arm (other arms can run first)
Tracks 0 and 1 start in parallel
```

## Non-Goals

- In-flight decode migration; semantics remain future-invocation reconstruction.
- Receding-horizon / re-solve control until static execution-aware accounting is stable.
- Production traces; the synthetic driver is the controlled workload.
- Multi-model or heterogeneous-hardware pools.
