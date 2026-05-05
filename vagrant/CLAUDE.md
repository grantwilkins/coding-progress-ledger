# CLAUDE.md — vagrant-agent

Working rules for Claude (and any other agent) operating in this repo. Read this before editing anything.

## What this repo is

`vagrant-agent` is a **state-mobility layer** between agent harnesses and serving/runtime backends. Given an agent workflow trace and a placement-change event, it decides what state moves together, what splits, and how each state object is materialized at the destination.

It is **not** a new agent harness, a new serving engine, or a scheduler. It is a derivation pipeline: trace → manifest → placement plan → cost estimate.

The single MVP claim it must demonstrate: shared state across nodes changes optimal placement vs. request-level routing, by enough to matter, on at least one real trace.

## Authoritative documents (read in this order)

1. `TASKS.md` — the working backlog. Workstreams, gates, status. **If reality diverges from this file, update this file.**
2. `vagrant_agent_repo_implementation_plan.md` — the long-form design doc. Reference, not gospel; `TASKS.md` overrides where they conflict.
3. `AGENTS.md` — coding rules (succinct, hard-fail, test, commit).
4. `../coding-progress-ledger/AGENTS.md` and `../coding-progress-ledger/TASKS.md` — sibling-repo conventions; vagrant follows the same shape.

## Sibling repos in this codebase

This repo lives under `/Users/grantwilkins/houdini/`. The siblings:

- `coding-progress-ledger/` — the append-only ledger framework. **Vagrant imports from this; do not fork it.** See § Reuse contract below.
- `coding-estimator/` — downstream consumer of ledger artifacts; predicts on-time finish from progress curves.
- `coding-data-collection/` — upstream trace-collection scripts.

Coding rules across all four repos are intentionally identical. If you see a rule here that contradicts a sibling, the sibling wins for code in that sibling, and you should flag the divergence.

## Reuse contract with `coding-progress-ledger`

Vagrant rides on `ledger_progress`. Specifically:

- `LedgerEvent` and its `payload: dict[str, Any]` carry vagrant-specific fields. Do not invent a new event class.
- `apply_event` / `replay` are the replay engine. Do not write a second one.
- `LedgerSession` is the write API. `vagrant_agent.session` thinly extends it.
- `serialization.py` handles JSONL roundtrip. Do not write a second one.
- `queries.py` provides leaf-work queries. Reuse `active_incomplete_leaves` etc.

**Permitted upstream changes**, in order of preference:

1. **Zero changes** — use the existing `payload` dict.
2. **A pass-through hook in `apply_event`** so unknown event_type strings append to the events list without raising. ~10 lines. This is the only change vagrant *requires* upstream.
3. **A new `SubtaskCategory.STATE`** if state objects must show up as subtasks. Probably unnecessary; default to payloads.

Anything bigger than the above is a fork. Do not do it. Escalate first.

## Hard rules (do-not-list)

```text
Do not fork ledger_progress. Import it.
Do not invent a new event class. Use LedgerEvent + payload.
Do not write a second JSONL serializer or replay engine.
Do not add an ILP solver in the MVP.
Do not simulate queues, capacity, or multi-region networks in the MVP.
Do not add real harness adapters before the MVP plot exists.
Do not score "semantic correctness" or model output quality.
Do not claim performance numbers from a single toy trace.
Do not hand-edit a manifest. Manifests are derived; fix the trace or the replay.
Do not mutate derived state in place. Emit events, replay, recompute.
Do not skip tests or commits. See AGENTS.md.
```

If a task seems to require violating one of these, stop and escalate.

## Distinguishing artifacts (vocabulary)

Use these terms exactly:

```text
trace                = append-only JSONL of agent events (immutable input)
manifest             = derived state graph + serving group view (replay artifact)
placement plan       = per-node site assignment (policy output)
materialization plan = per-state-object site/mode assignment (policy output)
results              = cost numbers + plots (analysis artifact)
```

A "manifest" is never edited. "Trace" is never mutated. "Plans" are policy outputs; if a plan is wrong, the policy is wrong, not the manifest.

## When to use plan mode vs. just edit

This repo is small and the rules are tight. Default to editing directly for code changes inside an existing workstream. Enter plan mode only when the change crosses workstream boundaries, touches the reuse contract, or modifies `TASKS.md` structure (not just task status).

## When to ask the user

Ask before:

- Modifying `coding-progress-ledger` (any change, even trivial).
- Adding a new module to `vagrant_agent/` not listed in the proposed repo layout.
- Adding a new policy beyond `request_level` and `shared_state_aware` during MVP.
- Adding a real harness adapter.
- Adding any dependency beyond `numpy`, `matplotlib`, `pyyaml`, `pytest`, and `ledger_progress`.

Otherwise proceed.

## Test/commit cadence

Per `AGENTS.md`: run `uv run pytest` after every change, commit each task as its own commit, and update `TASKS.md` to reflect completed work and surfaced follow-ups before finishing.

## What "done" means at each gate

`TASKS.md` has the per-workstream gates. The MVP gate (§ Workstream E) is the one to optimize for: a single command produces a duplication-factor plot showing the two policies differ on the toy trace, and a `pytest` assertion encodes the gap. Everything before that is scaffolding.
