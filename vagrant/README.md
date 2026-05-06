# vagrant-agent

State-mobility layer between agent harnesses and serving/runtime backends. Given an agent workflow trace and a placement-change event, decides what state moves together, what splits, and how each state object is materialized at the destination.

This is **not** a new agent harness or a new serving engine. It is a derivation pipeline:

```text
trace.jsonl  ──build_manifest──▶  manifest.json  ──run_policy──▶  placement + materialization plans  ──cost──▶  duplication-factor plot
```

## Status

MVP pipeline complete; SWE-agent F2 adapter (real-trace) and G1/G2 optimizers landed. **Workstream H1 (`request_level_with_site_cache`, the fair baseline) revealed that the previously-claimed 2× / 7.47× duplication-factor gaps between `request_level_no_reuse` (D1) and `shared_state_aware` (D2) are entirely explained by per-site cache reuse (H1's bookkeeping), not by shared-state-aware grouping.** On every linear-session fixture (toy, g_demo at `tau=1`, SWE-agent F2 `s_07`), H1, D2, and G1 collapse to identical numbers.

The original thesis ("agent workflow with shared state has a different optimal placement than treated as N independent requests") is **on life support, not refuted**: a constructed multi-private-state-with-different-homes fixture proves H1 ≠ D2 fixtures exist, but no real-trace fixture in the repo currently exercises the multi-component path. The next task (TASKS.md § Workstream H2) is a multi-session SWE-agent concat fixture that puts D2 in its natural habitat and decides whether the project pivots to a "site-cache-reuse-is-everything" finding or sustains the original grouping thesis.

## Clone → plot in under 10 minutes

```bash
git clone <this-repo>
cd vagrant
uv sync                                    # installs editable deps incl. ../coding-progress-ledger
uv run pytest -q                           # 122 tests, ~1s
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

Expected console output (toy trace):

```text
request_level_no_reuse:        total_cost_s=1.0759, dup_factor=2.0379
request_level_with_site_cache: total_cost_s=0.5279, dup_factor=1.0000
shared_state_aware:            total_cost_s=0.5279, dup_factor=1.0000
```

The cost-weighted duplication factor is **`Σ(cost_s · materialization_count) / Σ(cost_s)`** — i.e., the cost the policy actually paid divided by the lower-bound cost of one materialization per `(state, site)`. **The 2× gap belongs to per-site cache reuse (H1's bookkeeping), not to shared-state-aware grouping**: H1 and D2 produce numerically identical totals on this fixture and on every other linear-session fixture in the repo. Reading the audit CSV is the definitive way to reconcile the metric with the per-row decisions.

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
