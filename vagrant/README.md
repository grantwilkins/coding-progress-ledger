# vagrant-agent

State-mobility layer between agent harnesses and serving/runtime backends. Given an agent workflow trace and a placement-change event, decides what state moves together, what splits, and how each state object is materialized at the destination.

This is **not** a new agent harness or a new serving engine. It is a derivation pipeline:

```text
trace.jsonl  ──build_manifest──▶  manifest.json  ──run_policy──▶  placement + materialization plans  ──cost──▶  duplication-factor plot
```

## Status

MVP pipeline complete. The pipeline can express and measure shared-state duplication on a synthetic trace; this is **not** evidence that the duplication occurs at meaningful magnitude in real agent workloads. The baseline (`request_level_no_reuse`) is a deliberate strawman that materializes shared state once per consumer; the phenomenon is only claimed demonstrated when the gap reproduces against a competitive baseline (`request_level_with_site_cache`, Workstream H) on a real harness trace (Workstream F). See `TASKS.md` § 0 for framing rules.

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

Expected console output:

```text
request_level_no_reuse: total_cost_s=1.0759, dup_factor=2.0379
shared_state_aware:     total_cost_s=0.5279, dup_factor=1.0000
```

The cost-weighted duplication factor is **`Σ(cost_s · materialization_count) / Σ(cost_s)`** — i.e., the cost the policy actually paid divided by the lower-bound cost of one materialization per `(state, site)`. `shared_state_aware` materializes each state exactly once per component-site, so its factor is 1.0 by construction; `request_level_no_reuse` materializes shared state once per consumer, so its factor exceeds 1.0 in proportion to how much sharing the trace contains. Reading the audit CSV is the definitive way to reconcile the metric with the per-row decisions.

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
