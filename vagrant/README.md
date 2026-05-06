# vagrant-agent

State-mobility layer between agent harnesses and serving/runtime backends. Given an agent workflow trace and a placement-change event, decides what state moves together, what splits, and how each state object is materialized at the destination.

This is **not** a new agent harness or a new serving engine. It is a derivation pipeline:

```text
trace.jsonl  ──build_manifest──▶  manifest.json  ──run_policy──▶  placement + materialization plans  ──cost──▶  duplication-factor plot
```

## Status

MVP pipeline complete; SWE-agent F2 adapter (real-trace) and G1/G2 optimizers landed. **Workstream H1 (`request_level_with_site_cache`, the fair baseline) revealed that the previously-claimed 2× / 7.47× duplication-factor gaps between `request_level_no_reuse` (D1) and `shared_state_aware` (D2) are entirely explained by per-site cache reuse (H1's bookkeeping), not by shared-state-aware grouping.** On every linear-session fixture (toy, g_demo at `tau=1`, SWE-agent F2 `s_07`), H1, D2, and G1 collapse to identical numbers.

**Workstream H2** (`examples/traces/h2_multi_session_swe.jsonl`) constructs the smallest multi-session fixture that puts D2 in its natural habitat and finds **H1 strictly beats D2 by ~11×** (0.154 s vs 1.738 s), with the gap surviving 9/9 of the kv_bytes × link_bps sensitivity grid. The mechanism: when shared state links sessions into one D2 component while private workspaces have asymmetric home_sites, D2's grouping forces a cross-site workspace transfer that H1's per-session placement avoids. The trajectory text is real (s_07 reused 3×); the per-session workspace bytes/home_sites are *synthetic*.

**Workstream H3** ships `session_sticky` (places every node sharing a `session_id` at one site, with H1-style cache-reuse bookkeeping). On the H2 fixture it numerically matches H1; on a constructed intra-session-disagreement fixture it loses to both H1 and D2. Provably `session_sticky >= H1` by construction (same objective under tighter constraint). Useful as an explanatory baseline / future constraint surface (GPU pinning, data residency); not competitive on the current cost model.

**Workstream H4** ships `vagrant_agent.workspace.compute_repo_bytes` and a `SessionSpec.workspace_path` hook so multi-session traces can pull workspace bytes from a real filesystem snapshot. The H1<D2 mechanism survives end-to-end when bytes come from disk (`tests/test_h4_workspace_bytes.py`). What still doesn't survive: the *trajectory* in H2 is s_07 reused 3×. Closing that gap (a real multi-instance SWE-bench corpus with rollout dirs preserved) is the remaining step before "phenomenon demonstrated on real harness traces" — tracked as deferred H5.

The original thesis ("agent workflow with shared state has a different optimal placement than treated as N independent requests") **decomposes cleanly into two effects**: per-site cache reuse (H1's bookkeeping) and shared-state-aware grouping (D2's component-level placement). H2 shows the grouping effect is *real* but goes the *wrong way* — D2's component-level placement is strictly worse than H1's per-node placement when private states have asymmetric homes. Until H4 surfaces real workspace bytes, the project's claim is "we have a framework that distinguishes the two effects, and on the only fixture where they diverge, H1 wins."

Vagrant Agent decomposes stateful workflow mobility into two effects: (1) per-site materialization reuse, and (2) graph-level grouping constraints. The project asks when the second effect exists in real agent traces after accounting for the first.

## Clone → plot in under 10 minutes

```bash
git clone <this-repo>
cd vagrant
uv sync                                    # installs editable deps incl. ../coding-progress-ledger
uv run pytest -q                           # 179 tests, ~1s
uv run vagrant-bench \
    --trace examples/traces/toy_subagent_trace.jsonl \
    --out runs/mvp_demo
```

That writes:

```text
runs/mvp_demo/
  manifest.json
  results.csv
  state_materialization_breakdown.csv      # the audit source-of-truth
  summary.json
  request_level_no_reuse/
    placement_plan.json
    materialization_plan.json
  shared_state_aware/
    placement_plan.json
    materialization_plan.json
  plots/duplication_factor.png             # the headline plot
```

Expected console output (toy trace, default 5 Gbps single-flow inter-region link):

```text
request_level_no_reuse:        total_cost_s=1.0861, dup_factor=2.0375
request_level_with_site_cache: total_cost_s=0.5331, dup_factor=1.0000
shared_state_aware:            total_cost_s=0.5331, dup_factor=1.0000
```

The cost-weighted duplication factor is **`Σ(cost_s · materialization_count) / Σ(cost_s)`** — i.e., the cost the policy actually paid divided by the lower-bound cost of one materialization per `(state, site)`. **The 2× gap belongs to per-site cache reuse (H1's bookkeeping), not to shared-state-aware grouping**: H1 and D2 produce numerically identical totals on this fixture and on every other linear-session fixture in the repo. Reading the audit CSV is the definitive way to reconcile the metric with the per-row decisions.

### Sensitivity sweep (defending the gap against constants)

The headline ratio depends on three load-bearing constants — `kv_bytes_per_token`, `dst_prefill_tok_s`, `link_bps` — each spanning >1 order of magnitude across plausible 2025–2026 deployments. Run:

```bash
uv run vagrant-sensitivity \
    --trace examples/traces/toy_subagent_trace.jsonl \
    --out runs/sensitivity_demo
```

That sweeps `kv_bytes ∈ {10K, 70K, 320K}` (V4-class compact / V3 MLA / Llama-3-70B GQA) × `link_bps ∈ {5, 25, 100, 400} Gbps` (AWS single-flow inter-region / aggregate / 100GbE / RDMA-class) and writes `sensitivity.csv` with a `gap_robust` column. On the toy trace the documented finding is **0% survival** for D2 vs H1 (linear-session collapses) and **100% survival** for H1 vs D1 (cache reuse always wins). The H2 multi-session fixture sustains **100% survival for H1 vs D2** (gap is bytes-layer, scales with `1/link_bps`, never inverts).

## What the toy trace exercises

A planner LLM call with three subagents A, B, C. All four read a small shared system prefix (200 tokens) and a larger shared repo context (8000 tokens). A and C share a 4 MB workspace artifact (C writes, both read). B has a 12 000-token private context. See `docs/toy_trace.md` for the structural diagram and the `tau` analysis (the threshold below which any sharing groups nodes into the same component).

The toy is **adversarial by design** — it is constructed so a no-reuse baseline must duplicate. Showing the gap proves the framework can represent the phenomenon; it does not prove the phenomenon exists in real workloads.

## Reuse contract

Vagrant rides on the `coding-progress-ledger` framework: it imports `LedgerEvent`, `apply_event`, `replay`, `LedgerSession`, and the JSONL serializer. Vagrant-specific event types (`state_declare`, `state_read`, `state_write`, `state_invalidate`, `placement_decision`, `materialization_plan`, `migration_start`, `migration_end`) ride on `LedgerEvent.event_type` as plain strings via the upstream pass-through hook. See `CLAUDE.md` for the rule set and the four-test invariant set that protects the hook.

## Read first

- `CLAUDE.md` — full rules and reuse contract.
- `AGENTS.md` — coding rules (succinct, hard-fail, test, commit).
- `TASKS.md` — workstreams, gates, completed and deferred work.
- `docs/manifest_schema.md` — Serving Group Manifest schema.
- `docs/toy_trace.md` — the toy scenario and `tau` sensitivity.

## Sibling repos

- `../coding-progress-ledger/` — the ledger framework; vagrant imports from this.
- `../coding-estimator/`, `../coding-data-collection/` — downstream/upstream of the ledger.
