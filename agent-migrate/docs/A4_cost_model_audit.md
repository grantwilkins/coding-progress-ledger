# A4 — cost-model audit

**Status:** done, 2026-05-05
**Scope:** document the load-bearing assumptions in agent-migrate's cost model that collaborator 2 flagged in sections 1D and 1E. No code change here; some assumptions are resolved by Workstream K's fluid simulator (K4), others are documented and accepted as the cost of staying tractable.
**Headline finding:** the cost model has at least four load-bearing simplifications. Two (additivity, decode omission) are documented and acceptable; one (faster-prefill bias under infinite capacity) is exactly what K4's fluid simulator addresses; one (KV compression) would shrink some K results but does not flip any conclusion drawn so far.

## Summary table

| Assumption | Today | Effect on existing results | Resolution |
| ---------- | ----- | -------------------------- | ---------- |
| 1. Additive cost (`transfer + prefill`) | always sums | overstates cost when transfer and prefill could pipeline (`max(transfer, prefill)`); biases AGAINST D2/D3 grouping | document; not in scope for K |
| 2. Faster-prefill bias under infinite capacity | colocation at faster site is unconstrained | D2 always picks seattle on linear-session traces; this is the H5b cancellation | resolved by K4's fluid prefill capacity |
| 3. Decode time omitted | not modelled | cancels in policy differences on the same trace; safe | document; never resolved |
| 4. Raw-bytes KV (no compression) | `kv_transfer_s = 8 * T * kv_bytes_per_token / link_bps` charges raw KV | overstates KV transfer cost ~3-4× per CacheGen | document; sensitivity sweep covers via `kv_bytes_per_token` axis |

## 1. Additive vs pipelined cost

`materialize_cost(state, mode, src, dst, bundle)` returns a scalar. When a placement plan moves K state objects, the policy code sums all K costs. Real systems can pipeline transfer and compute: while bytes are still arriving, the destination GPU may be processing earlier-arrived bytes for prefill. The right mental model is `max(network_time, prefill_time)`, possibly piecewise. Vagrant's additivity overstates total time.

**Effect on existing results.** Anything with both KV transfer AND context replay paid as part of the same plan is over-counted. This biases AGAINST policies that move many states at once (D2/D3 colocating components), because additive costs grow linearly with state count whereas pipelined costs saturate at `max(...)`.

**Resolution.** Out of scope for K. Workstream K's fluid simulator (K4) does NOT pipeline by design; it tracks per-resource utilization but each action's `wallclock_s` is still computed additively at the formula level. A future workstream M (post-pivot) could replace `wallclock_s` with a piecewise model where `prefill_s` and `transfer_s` overlap proportionally to instantaneous progress. M is not on the immediate path.

**Fail-loud safeguard.** `costs.py` docstring already carries a caveat block (committed in workstream H4); A4 does not change it but cross-references it from K0.

## 2. Faster-prefill bias under infinite capacity

`shared_state_aware` (D2) picks the site minimizing the sum of materialization costs across the component. Under the canonical `sites_2site.yaml`, seattle has 1.5× phoenix prefill (45k vs 30k tok/s). For a component dominated by prompt-context replay, seattle wins by ~33% on prefill — and there is no capacity constraint preventing D2 from sending every component there.

This is **exactly the cancellation observed in H5b**:

```
prefill_savings_at_seattle = 11 prompt_states × 7000 tok / (30000 - 45000⁻¹) ≈ 0.078 s
extra_workspace_transfer_at_seattle = 8 × 380 KB / 5e9 ≈ 0.0006 s
prefill_savings >> workspace_transfer  →  D2 picks seattle, D2 ≡ H1
```

**Effect on existing results.** Any policy that can freely pick a site is biased toward seattle on linear-session and small-byte fixtures. This is realistic IF capacity is infinite (which is what the model says today). It is NOT realistic at production scale — collaborator 2 flagged this as a key motivation for mobility-episodes.

**Resolution.** **K4 fluid simulator.** With `prefill_tok_s_per_site` as a finite capacity, sending N concurrent reconstitutions to seattle saturates the prefill resource and slows everyone down. The "always pick seattle" strategy stops being free. This is the load-bearing reason K4 exists.

**Test pin.** A1 audit's no-row-flips finding (H5b is robust to payload-interpretation choice within distributed-origin) is partly a consequence of this assumption. K7's T1 (capacity-free collapse) explicitly verifies that under infinite capacity, K's policies match L1 — i.e., that the bias persists when intended.

## 3. Decode time omitted

The cost model does not include token-by-token decode at the destination. A real workflow's wall-clock time after reconstitution is dominated by decode, not by prefill or transfer. We omit it because:

- Decode rate is typically ~10-200 tok/s per stream at production scale; 1000-token continuation = 5-100 s of decode.
- Two policies placing the *same* K nodes pay the *same* decode time per workflow.
- Decode therefore CANCELS in any same-trace policy comparison.

**Effect on existing results.** None on relative orderings between policies. Absolute wall-clock predictions are loose; agent-migrate has never claimed otherwise.

**Resolution.** Never. The MVP and K both keep "useful resume" = first-token-decoded as the metric (per K0 calibration). Long-tail decode is a different question.

**Caveat for K.** `mixed_min_pressure` may *want* to consider decode admission as a fifth bottleneck dimension. We've explicitly chosen not to model decode admission in the K MVP (per the hard-rules block). If K7's T3 fails because of decode-admission saturation, that's a meaningful finding — not a model bug.

## 4. Raw-bytes KV (no compression)

`kv_transfer_s = 8 * T * kv_bytes_per_token / link_bps` charges raw KV bytes over the wire. CacheGen (SIGCOMM '24, arXiv 2310.07240) reports 3.5–4.3× lossless compression for KV cache. Production systems with KV transfer would almost certainly use it.

**Effect on existing results.** `kv_transfer_s` is overstated by ~3-4×. The bandwidth crossover B* = `8 * kv_bytes_per_token * dst_prefill_tok_s` shifts left by the same factor, meaning kv_transfer becomes attractive at a lower link bandwidth than the model says.

**Resolution.** The sensitivity sweep already covers this implicitly: the `kv_bytes_per_token` axis spans 10K–327K, which is a 32× range. A 3-4× compression factor is well inside that range. A claim that survives the sensitivity sweep is robust to KV compression *as a constant factor*. A regime-flip-from-compression test would require an asymmetric compression model (some states compress, others don't) — not in scope for K.

## What is NOT in this audit

- **The decode admission question.** Section 3 above documents why we omit it; if K7's T3 fails because of decode saturation, that's evidence to revisit.
- **Per-stream prefill scaling at long context.** Real systems' prefill rate drops at >32K tokens because attention is O(N²). The toy and pilot trajectories live in the linear regime. Any K fixture with >32K-token contexts would need a piecewise prefill model.
- **Heterogeneous-hardware site profiles.** Phoenix and seattle differ only in `prefill_tok_s`. Real fleets have varied GPU types with different KV-per-token byte costs (FP16 vs FP8 vs BF16). The 3-model bracket in `model_profiles.yaml` covers the per-model variation; per-site model variation is not modelled.

## Action items rolled into K0/K3/K7

- K0 calibration writeup: cross-reference this audit and explicitly accept the four assumptions on the project's behalf.
- K3 resource vector: do NOT re-derive `wallclock_s` from a max(transfer, prefill) model — keep additivity for parity. Note in K3's docstring that `network_bytes` and `prefill_tokens` are tracked separately so a future M-workstream pipelining model can compute `max(...)` from those.
- K7 T2 (prefill-stampede): MUST exhibit a stampede under finite prefill capacity and replay_all. If it does not, either the K4 simulator is wrong or the prefill-bias assumption from §2 was actually realistic-without-capacity (which would mean K is not justified).

This audit's net is small in code — zero — but it pins the cost-model contract for K and prevents future drift.
