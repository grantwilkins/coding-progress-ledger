# TASKS — Vagrant Agent: State-Mobility Layer for Agent Workflows

This file is the working backlog for `vagrant-agent`. It is the authoritative plan; if reality diverges, update this file rather than the implementation plan.

The thesis, in one sentence:

> An agent workflow with **shared state across nodes** has a different optimal placement than the same workflow treated as **N independent requests**, and the difference is large enough to matter.

**The MVP is a pipeline demo, not evidence for the thesis.** The MVP succeeds when the pipeline can *express and measure* shared-state duplication on one synthetic trace, end-to-end. The toy trace (§ A6) is **adversarial by design** — it is constructed so that a no-reuse baseline must duplicate shared state. Showing the gap on this trace proves the framework can represent the phenomenon; it does not prove the phenomenon exists in the wild.

The phenomenon is **only claimed demonstrated** when the same gap reproduces on at least one real harness trace (Workstream F) or on a synthetic sweep with a non-trivial co-location tradeoff (e.g., grouping shared nodes forces expensive private-state materialization). Until then, all claims are scoped to "the framework can express this."

We are **not** building: a new agent harness, a new serving engine, an ILP solver, a multi-policy benchmark suite, or a real-cloud migration system. See § 0.1 for the explicit non-goals.

Repo grounding (read these before starting any task):

- `vagrant_agent_repo_implementation_plan.md` — the original (longer) design doc. Reference, not gospel. Where this TASKS.md disagrees, this file wins.
- `../coding-progress-ledger/ledger_progress/core.py` — `LedgerEvent`, `apply_event`, `replay`. Vagrant rides on these.
- `../coding-progress-ledger/ledger_progress/session.py` — `LedgerSession` write API.
- `../coding-progress-ledger/ledger_progress/serialization.py` — JSONL roundtrip.
- `../coding-progress-ledger/ledger_progress/queries.py` — `active_incomplete_leaves` and friends.
- `../coding-progress-ledger/ledger_progress/sidecar.py` — live instrumentation pattern.
- `../coding-progress-ledger/AGENTS.md` and this repo's `AGENTS.md` — coding rules, identical in spirit.

**Status markers** on each task: `not started` · `in progress` · `blocked` · `done` · `deferred`. These are plain text — they are not ledger events.

---

## § 0. Project rules for all agents

```text
Do not fork ledger_progress. Import it.
Do not invent a new event class. Ride on LedgerEvent.
Do not add an ILP solver in the MVP.
Do not simulate queues, capacity, or multi-region networks in the MVP.
Do not add real harness adapters before the synthetic pipeline produces the headline plot.
Do not score "semantic correctness" or model quality.
Do not claim "the phenomenon is demonstrated" from the toy trace. The toy is adversarial by design and proves expressiveness, not the phenomenon.
Do not abbreviate request_level_no_reuse to request_level. The "no_reuse" qualifier is load-bearing — it documents that the baseline is strawman.
Do not write tests asserting a kv-vs-replay crossover in token count T. T cancels. Crossovers live in bandwidth, kv_bytes_per_token, and prefill rate.
Do not hand-edit a manifest. The manifest is derived from the trace; if it is wrong, fix the trace or the replay.
Do not lean on ledger_progress scoring/split/reopen semantics for vagrant signals. Use ledger subtasks as graph nodes only.
```

If a task seems to require violating one of these rules, stop and escalate — don't quietly relax them.

Every output must distinguish:

```text
trace            = append-only JSONL of agent events (immutable input)
manifest         = derived state graph + serving group view (replay artifact)
placement plan   = per-node site assignment (policy output)
materialization plan = per-state-object site/mode assignment (policy output)
results          = cost numbers + plots (analysis artifact)
```

## § 0.1 Non-goals (explicit)

The following are **out of scope for this repo** until the MVP plot exists:

- Real cloud deployment, real KV-tensor migration, packet-level network simulation.
- Power/thermal modelling.
- Online scheduler with admission control.
- Semantic-quality scoring of outputs.
- Real harness adapters (OpenHands / SWE-agent / LangGraph / CrewAI).
- More than 2 sites, more than 1 model profile, more than 2 policies.

These are real research questions; they do not unlock the MVP claim.

## § 0.2 Reuse contract with `coding-progress-ledger`

Vagrant imports `ledger_progress` as a library. Permitted upstream additions, in order of preference:

1. **Zero changes** — use `LedgerEvent.payload` (already `dict[str, Any]`) to carry vagrant-specific fields (`state_id`, `tokens`, `content_hash`, `site`, `mode`).
2. **Pass-through hook in `apply_event`** — ~10 lines so unknown `event_type` strings append to the events list without raising. Lets vagrant emit `state_read` / `state_write` / `placement_decision` events on the same ledger.
3. **`SubtaskCategory.STATE`** — only if state objects must show up as subtasks (probably not; prefer payloads).

Anything bigger than the above is a fork; do not do it. If a fourth need appears, escalate before coding.

## § 0.3 Vagrant-to-ledger mapping

This is the strict mapping from vagrant concepts onto `ledger_progress`. Lock it in before A1; downstream code assumes it.

| Vagrant concept    | Ledger representation                                                                  |
| ------------------ | -------------------------------------------------------------------------------------- |
| workflow           | run directory + trace file. **Not** a subtask.                                         |
| LLM call node      | `Subtask` with payload `node_type=llm_call`                                            |
| tool call node     | `Subtask` with payload `node_type=tool_call`                                           |
| subagent node      | `Subtask` with payload `node_type=subagent`                                            |
| state object       | **Not** a subtask. Carried in payload of vagrant `state_*` events.                     |
| subagent spawn     | `ADD_SUBTASK` (parent_id = planner). **Not** `SPLIT_SUBTASK` unless parent work is invalidated. |
| node start/end     | `UPDATE_STATUS` with `IN_PROGRESS` / `COMPLETE`. (Dedicated `node_start` / `node_end` events may be added post-MVP if lifecycle distinct from progress is needed.) |
| state invalidation | vagrant `state_invalidate` event. **Not** `INVALIDATE_SUBTASK` unless the consuming node's work also invalidates. |

**Discipline.** `SPLIT_SUBTASK` / `REOPEN_SUBTASK` / `INVALIDATE_SUBTASK` carry **scoring semantics** in `ledger_progress`. Vagrant must not lean on those semantics. Use ledger subtasks as **graph nodes**; do not use ledger progress scores as a vagrant signal in the MVP.

---

## § Workstream A — Trace (MVP)

**Goal.** Vagrant emits and reads agent traces using `ledger_progress`'s event log. No new file format, no new replay engine.

**A1 — Event vocabulary** (`done`, 2026-05-05)
Define `vagrant_agent/events.py`: a small module of string constants for the 8 vagrant-specific event types.

```text
state_declare
state_read
state_write
state_invalidate
placement_decision
materialization_plan
migration_start
migration_end
```

Each carries a payload schema documented in the module docstring. Workflow lifecycle, node lifecycle, subagent spawn/return, and invalidation reuse `ledger_progress` enums (`ADD_SUBTASK`, `UPDATE_STATUS`, `SPLIT_SUBTASK`, `REOPEN_SUBTASK`, `INVALIDATE_SUBTASK`).

**A2 — Pass-through hook in ledger_progress** (`done`, 2026-05-05, upstream — 18 invariant tests in coding-progress-ledger/tests/test_extension_events.py)
Land a ~10-line change in `ledger_progress/core.py:apply_event` so unknown event_type values append to `ledger.events` without mutating subtasks. Coordinate with the ledger maintainer; this is the only upstream change required for the MVP.

A2's definition of done is the four-test invariant set, all in `coding-progress-ledger/tests/`:

```text
1. unknown vagrant event is preserved in replayed ledger.events
2. unknown vagrant event does not mutate ledger.subtasks
3. unknown vagrant event survives JSONL roundtrip (write -> read -> equal)
4. ledger scoring (scoring.score) is unchanged by the presence of unknown events
```

The hook must not silently drop unknown events; it must append. Tests for queries (`active_incomplete_leaves` etc.) must explicitly skip unknown-event types.

**Scope clarification (post-A1).** The MVP plan said "~10 lines in `apply_event`". On reading the upstream code, two places block unknown event types:

1. `ledger_progress/core.py:apply_event` (line 102) — raises on unknown handler.
2. `ledger_progress/core.py:LedgerEvent.__post_init__` (line 69) — coerces `event_type = EventType(self.event_type)`, raising on unknown strings.
3. `ledger_progress/serialization.py:event_from_dict` (line 21) — does the same coercion on JSONL load.

Smallest viable upstream design: change `LedgerEvent.event_type` to permit either `EventType` or a non-empty `str`; teach `apply_event` and `serialization` to pass through string values that are not `EventType` members. ~25 lines + tests, not 10. Same invariants as before.

**A3 — Synthetic adapter** (`done`, 2026-05-05)
`vagrant_agent/adapters/synthetic.py` writes a JSONL trace for the canonical scenario (§ A6). Parameters: `num_subagents`, `shared_context_tokens`, `private_context_tokens` (per child), `workspace_bytes`, `system_prefix_tokens`, `seed`. Output is a `ledger_progress`-compatible JSONL.

The generator is the source of truth. The committed JSONL (§ A6) is regeneratable byte-for-byte from a fixed seed/config. Tests assert this.

**A4 — Trace session helpers** (`deferred until A2 lands`)
Thin wrapper `vagrant_agent/session.py` extending `LedgerSession` with `record_state_declare/read/write/invalidate` methods. No replay logic, no scoring — just convenience.

**A5 — Prompt-segment hashing** (`done`, 2026-05-05)
Helper that hashes text segments (or accepts pre-computed hashes when text is unavailable, e.g., in adapters that only see token counts). Used by A3 to produce `content_hash` payloads.

For synthetic traces, hashes should be deterministic, symbolic, and inspectable — e.g., `hash_shared_repo_v1`, `hash_workspace_AC_v1`, `hash_private_B_v1`, `hash_system_prefix_v1`. The hashing helper still exists for real adapters; synthetic just bypasses it with stable strings.

**A6 — Canonical toy trace** (`done`, 2026-05-05 — file committed, regeneratable, and round-trips through ledger_progress.replay)
Commit `examples/traces/toy_subagent_trace.jsonl`: parent planner uses shared repo context; spawns 3 subagents A/B/C; A and C share workspace; B has large private context only.

**Add one extra state object: a small shared system/tool prefix consumed by all four nodes** (`system_prefix_tokens` ≈ 200). Purpose: with `tau=1` (any sharing), this trivial prefix glues *everything* together and the policy collapses to a single component. Surfacing this immediately motivates `tau` as a real knob, not a default. Document the expected behavior at `tau=1` vs. `tau=1000` in `docs/toy_trace.md`.

This is the single trace the MVP plot is computed against. **Adversarial by design**: it is constructed so a no-reuse baseline must duplicate. Showing the gap proves expressiveness, not the phenomenon.

**Gate (A done).** `examples/traces/toy_subagent_trace.jsonl` exists, replays under `ledger_progress.replay()` without error, and a `pytest` roundtrip test passes. ✅ **Workstream A gate met** as of 2026-05-05. A4 (`TraceSession` wrapper) deferred — pure ergonomics, not on the MVP critical path.

---

## § Workstream B — Manifest (MVP)

**Goal.** From a trace, derive `(nodes, state_objects, edges)` — the Serving Group Manifest.

**B1 — Replay-derived state index** (`done`, 2026-05-05)
`vagrant_agent/manifest.py`: replay a trace, walk events, accumulate state objects, and record producers/consumers per node. Reuses `ledger_progress.replay`; does not duplicate it.

**State identity rule.** Every state object carries both `state_id` and `content_hash`.

```text
state_id     = explicit identifier from the producer event (synthetic adapters supply this)
content_hash = hash of segment text or token sequence (real adapters supply this)
primary key  = state_id if present, else content_hash
```

Two reads of `"OK"` from different tools have the same `content_hash` but different `state_id` — they are not the same state object. This rule is necessary for real adapters; synthetic uses stable `state_id`s.

**B2 — Edge construction** (`done`, 2026-05-05)
The **canonical representation** is bipartite: `node --uses--> state_object`, stored as the consumer/producer lists on each `StateObject`. Pairwise `StateEdge(node_a, node_b, state_id, tokens)` is a **derived view** for policy convenience.

For MVP, emit pairwise edges for every pair of consumers of the same state object, weighted by token count. Keep the bipartite source-of-truth so the derived view can be regenerated.

**B3 — Manifest serialization** (`done`, 2026-05-05)
JSON output with `nodes`, `state_objects`, `edges`. Schema documented in `docs/manifest_schema.md`. Round-trip test.

**B4 — `vagrant-manifest build` CLI** (`done`, 2026-05-05)
`vagrant-manifest build --trace <jsonl> --out <json>`. Single command, no flags beyond input/output for MVP.

**Gate (B done).** ✅ Met as of 2026-05-05. `vagrant-manifest build` on the toy trace produces a manifest where:

- Shared repo context: 1 `StateObject`, 4 consumers (parent + 3 subagents).
- Workspace state: 2 consumers (A, C).
- Private contexts: 1 consumer each.
- `tests/test_manifest.py:test_workstream_b_gate_consumer_counts` asserts all six counts in one block.

**Critic-review fixes folded into B (2026-05-05).**

- Reconciliation pass after event loop so trace event ordering doesn't leave `WorkNode.required_state` / `produced_state` inconsistent with the bipartite source.
- Validation pass: every node referenced as a producer/consumer must have an `add_subtask` event.
- Hardened state-id semantics: any duplicate `state_declare` for the same `state_id` hard-fails (was: silently tolerated when hash matched).
- Removed redundant `dict.fromkeys` consumer dedup; deduplication is enforced upstream by `_read`.
- CLI dead `return 2` removed; `parent_id` read from `Subtask` not payload.

---

## § Workstream C — Cost (MVP)

**Goal.** Closed-form materialization cost. No queues, no capacity.

**C1 — Model and site profile loaders** (`done`, 2026-05-05)
`vagrant_agent/profiles.py` reads `configs/model_profiles.yaml` and `configs/sites_2site.yaml`. MVP ships **one model** (e.g., `compact_kv`) and **two sites** (`phoenix`, `seattle`) with a single inter-site link.

**C2 — Cost formulas** (`done`, 2026-05-05)
`vagrant_agent/costs.py` implementing exactly four formulas from the plan, lines 479–484:

```text
kv_transfer_s    = 8 * T * kv_bytes_per_token / link_bps
context_replay_s = T / dest_prefill_tok_s
text_transfer_s  = 8 * text_bytes / link_bps
artifact_copy_s  = 8 * artifact_bytes / link_bps
```

**Important.** Token count `T` cancels in the kv-vs-replay comparison; **there is no crossover in `T`**. The crossover is in `link_bps`, `kv_bytes_per_token`, or `dest_prefill_tok_s`. Do not write tests that imply otherwise.

**C3 — Cost crossover unit tests** (`done`, 2026-05-05)
At least one test per formula. The crossover test must be in **bandwidth**, not token count:

```text
B* = 8 * kv_bytes_per_token * dest_prefill_tok_s
below B*: context_replay < kv_transfer (replay is cheaper on a slow link)
above B*: kv_transfer < context_replay (transfer wins on a fast link)
```

Also test: holding `link_bps` fixed, varying `kv_bytes_per_token` and `dest_prefill_tok_s` produces the same crossover identity.

**C4 — Materialization mode chooser** (`done`, 2026-05-05 — first-in-tuple tie-break documented)

```python
choose_min_cost_mode(state, src_site, dst_site, allowed_modes) -> (mode, cost_s)
```

Returns the cheapest feasible mode and its cost. Used by both policies in D, and by the audit CSV in E4. Even if MVP policies fix one mode per state-type, this helper is needed immediately.

**Gate (C done).** Given a state object and (src, dst, allowed_modes), `choose_min_cost_mode` returns a deterministic `(mode, cost_s)`. The four formulas have unit tests including a bandwidth crossover.

---

## § Workstream D — Policy (MVP)

**Goal.** Two policies. One closed-form rule each.

**D1 — `request_level_no_reuse` baseline** (`done`, 2026-05-05)
Each node placed independently at the site with lowest sum-of-cost across its required state objects. State materialization is paid **per node**, even if multiple nodes are colocated. This is a deliberately strawman baseline — its job is to make the duplication accounting visible, not to model a competitive system.

**Naming discipline.** Use `request_level_no_reuse` everywhere; do not abbreviate to `request_level`. A future, more competitive baseline (`request_level_with_site_cache` — same per-node placement, but materialized state is reused across colocated nodes at a site) is a deferred policy in Workstream H.

**D2 — `shared_state_aware` policy** (`done`, 2026-05-05)
Build the node-state-share graph (edge if two nodes share `> tau` tokens of state). For each connected component, place the whole component at the site with lowest sum-of-cost. Shared state is materialized **once per component-site**, not once per node.

`tau` is a single config knob measured in tokens on shared-state edges. Default = 1 (any sharing groups). Expose `--tau` on `vagrant-bench` and `vagrant-plan` even in the MVP; the toy trace's small system prefix (§ A6) is intentionally chosen to motivate the knob.

**D3 — Plan emission** (`done`, 2026-05-05)
Both policies emit two artifacts.

`placement_plan.json`:

```json
{"node_id": "n1", "site": "phoenix", "cost_s": 4.2, "reason": "min_cost"}
```

`materialization_plan.json`:

```json
{
  "state_id": "repo_context",
  "content_hash": "hash_shared_repo_v1",
  "site": "seattle",
  "mode": "context_replay",
  "cost_s": 12.3,
  "consumers": ["n1", "n2", "n3", "n4"],
  "reason": "min_cost"
}
```

The `reason` field is human-readable provenance for paper figures and debugging. It records *why* the policy made this choice (e.g., `"min_cost"`, `"forced_by_colocation"`, `"only_feasible_mode"`).

**Deferred policies (NOT MVP).** `request_level_with_site_cache`, `session_sticky`, `prefix_group`, `subagent_group`, `workspace_group`, agglomerative-merge adaptive heuristic, ILP oracle. Each has its own future workstream; do not implement now.

**Gate (D done).** On the toy manifest, both policies emit valid plans with non-empty `reason` fields. The `request_level_no_reuse` plan duplicates shared state; the `shared_state_aware` plan does not. A test asserts the cost-weighted duplication factor differs.

---

## § Workstream E — Bench + plot (MVP)

**Goal.** One end-to-end command, one plot, one audit CSV.

**E1 — `vagrant-bench` CLI** (`done`, 2026-05-05)
`vagrant-bench --trace <jsonl> --policies request_level_no_reuse,shared_state_aware --tau <int> --out <dir>`. Runs both, writes `results.csv`, `plots/duplication_factor.png`, and the per-state breakdown from E4.

**E2 — Metrics** (`done`, 2026-05-05)
`vagrant_agent/metrics.py` computing exactly:

- `shared_state_duplication_factor` (cost-weighted, see below)
- `repeated_prefix_fraction`
- `state_layer_breakdown`

Skip recovery-time CDFs and queue timeseries until queues exist.

The headline metric is **cost-weighted**, not token-weighted, because a 500K-token state and a 1K-token state should not count equally:

```text
shared_state_duplication_factor =
    sum over s of materialization_count_s(policy) * cost_s
  / sum over s of ideal_materialization_count_s    * cost_s
```

where `ideal_materialization_count_s` is the lower bound (one materialization per site that has at least one consumer). A token-weighted variant may also be reported as `shared_state_duplication_factor_tokens`, but the cost-weighted version is the one in the headline plot.

**E3 — Plot** (`done`, 2026-05-05)
`plots/duplication_factor.png`: two bars, one per policy, on the toy trace. Matplotlib, no styling beyond defaults.

**E4 — State materialization breakdown CSV** (`done`, 2026-05-05 — incl. num_consumers + ideal_materialization_count columns; headline metric reproducible from CSV alone)
`state_materialization_breakdown.csv` — one row per `(policy, state_id, site)`. Columns:

```text
policy
state_id
content_hash
state_type
site
mode
tokens
bytes
cost_s
num_consumers
materialization_count
reason
```

This CSV is the auditable source for every plot. If the plot looks weird, the CSV explains why. Required for MVP.

**Gate (E done — MVP PIPELINE COMPLETE).**

```text
$ vagrant-trace summarize examples/traces/toy_subagent_trace.jsonl
$ vagrant-manifest build --trace examples/traces/toy_subagent_trace.jsonl --out examples/manifests/toy.json
$ vagrant-bench \
    --trace examples/traces/toy_subagent_trace.jsonl \
    --policies request_level_no_reuse,shared_state_aware \
    --out runs/mvp_demo
```

…produces:

```text
runs/mvp_demo/results.csv
runs/mvp_demo/state_materialization_breakdown.csv
runs/mvp_demo/plots/duplication_factor.png
```

…showing `shared_state_aware < request_level_no_reuse` on cost-weighted duplication factor. `pytest` is green. **The pipeline is demonstrated on an adversarial toy trace; the phenomenon is not yet claimed.** Continue to F to attempt a real-trace replication, or stop and reassess.

---

## § Workstream F — Real harness adapters (deferred)

Only after E is green.

**F1.** OpenHands adapter — translate session log → vagrant trace. **(deferred — no OpenHands trajectories cached; F2 satisfies the gate.)**
**F2.** SWE-agent adapter — reuse the `coding-progress-ledger` SWE-agent retrospective pipeline. **(`done`, 2026-05-05.)**
**F3.** LangGraph / CrewAI — only if F1 or F2 surfaces something the synthetic adapter missed. **(deferred.)**

**Gate (F done).** At least one real harness produces a trace that replays into a non-trivial manifest (≥ 5 nodes, ≥ 2 shared state objects). ✅ **Met as of 2026-05-05** via F2 on `tests/fixtures/swe_agent_pilot_s_07.json`: 11 nodes, 10 state objects, 7 non-trivial shared (excl. `system_prompt` + `issue_text`). Accumulation model: tool output at turn N read by every subsequent ai turn.

Pre-flight + code critic findings folded in: accumulation model (replaces "next-turn-only" reads), first-non-system-turn validation, dropped `update_status in_progress` ceremony (cuts ~33% events), cached `state_id → (hash, tokens)` to remove O(N²) scans, adversarial-trajectory tests (empty / ai-before-user / missing system_prompt / no-repeats), and the `non_trivial_shared_state_count` diagnostic with `exclude=` test.

Phenomenon claim still pending H1 (`request_level_with_site_cache`) on real traces.

## § Workstream G — Optimization (deferred)

Only after F1 or F2.

**G1.** Offline brute-force oracle on small instances (≤ 16 nodes). **(`done`, 2026-05-05 — pure-Python enumeration over K^N placements; no solver dep. Hard-fails above G1_MAX_NODES.)**
**G2.** Online heuristic: greedy single-node local search seeded from D1's per-node placement. **(`done`, 2026-05-05 — replaces the agglomerative-clustering framing per pre-flight critic; cleaner termination proof.)**
**G3.** Multi-objective weighting. **(deferred — MVP cost model has one dimension (seconds); λ-weighting requires ≥2 dimensions, which arrive with Workstream I capacity / J semantic.)**

**Per-state `home_site` enabler.** `StateObject.home_site: str | None` was added (default None → falls back to `bundle.home_site`) so per-state asymmetry can exist. Synthetic + multi_component adapters write it; SWE-agent adapter leaves None. The cost model is unchanged: same-site = `context_replay` (T/prefill); different-site = the existing four formulas.

**Gate (G).** G1 ≤ D2 on every instance (optimizer is at-least-as-good); G1 strictly beats D2 at fragmenting tau via the `g_demo_trace` fixture (saves 12.7% by avoiding per-component bookkeeping duplication). G1 ≡ G2 on all current fixtures (no local-optima trap). All three properties asserted in tests.

## § Workstream H — Extra policies (deferred)

The MVP baseline (`request_level_no_reuse`) is intentionally a strawman. Before any external claim, replace or supplement it with a competitive baseline.

**H1 — `request_level_with_site_cache`** (`done`, 2026-05-05)
Same per-node placement as `request_level_no_reuse`, but materialized state is **reused across colocated nodes at the same site**. ~6 LOC delegating to existing `_plan_from_placement` + `_place_per_node_min_cost` helpers. Pre-flight + code+findings Opus critics. **Finding**: H1 numerically collapses to D2 and G1 on every existing fixture (linear-session structure puts every node at the same site); a constructed multi-private-state-different-homes fixture proves H1 ≠ D2 *can* happen but no real-trace fixture currently exercises it.

**H2 — Multi-session SWE-style fixture (synthetic workspaces)** (`done`, 2026-05-05)
Concatenates 3 F2-style SWE-agent sessions (reusing the cached `swe_agent_pilot_s_07.json` traj three times, truncated to 2 ai turns each) into one trace with a shared `system_prompt` state and **synthetic** per-session `workspace_<sid>` states whose `home_site` and `bytes` are assigned by the fixture builder. Adapter: `src/vagrant_agent/adapters/swe_agent_multi.py`. Canonical fixture: `examples/traces/h2_multi_session_swe.jsonl` (3 sessions, workspace_homes `[phoenix, seattle, phoenix]`, 1 GB each, 6 nodes total — fits G1's `K^N <= G1_MAX_ENUMERATIONS` cap). Pre-flight + code+findings Opus critics.

**Numerical findings on the canonical fixture** (`compact_kv` × `sites_2site.yaml`):

| Policy | Total cost (s) | Notes |
| ------ | -------------- | ----- |
| D1 (`request_level_no_reuse`)       | 0.3883 | strawman; system_prompt paid per-consumer |
| H1 (`request_level_with_site_cache`) | 0.1542 | per-node placement; sa,sc → phoenix, sb → seattle |
| D2 (`shared_state_aware`, τ=1)       | 1.7380 | one component; all → phoenix; pays workspace_sb cross-site |
| G1 (`g1_brute_force`)               | 0.1542 | oracle ≡ H1 (64 enumerations) |
| G2 (`g2_local_search`)              | 0.1542 | seeded from D1; finds the same floor |

**H1 strictly beats D2 by ~1.58 s ≈ 11×** on this fixture. Mechanism: D2 forces all 6 nodes to phoenix (single component linked by `system_prompt`); workspace_sb has `home_site=seattle`, so D2 pays `8 × 1 GB / 5 Gbps = 1.6 s` artifact_copy. H1's per-node placement keeps sb at seattle and the workspace stays local.

**Sensitivity**: gap survives at **9/9 = 100%** of the `kv_bytes ∈ {10K, 70656, 327680} × link_bps ∈ {5e9, 25e9, 100e9}` grid (`tests/test_h2_multi_session.py:test_h1_d2_gap_survives_full_sensitivity_grid`). The gap is bytes-layer (`8*B/bps`), independent of `kv_bytes_per_token`, so it scales with `1/link_bps` but never inverts.

**Honest framing — what H2 is and is not.** The trajectory text is real (s_07), but the load-bearing state for the gap is the synthetic per-session workspace that no F2 adapter currently surfaces. H2 is therefore "**mechanism demonstrated on synthetic-but-structurally-realistic fixture**", not a real-trace phenomenon claim. It satisfies the *synthetic-sweep* clause of the phenomenon-demonstrated gate (multi-private-state-with-different-homes, sensitivity-robust at 100%); it does **not** satisfy the *real-trace* clause, because the real workspace bytes that drive the gap come from the fixture builder, not from the trajectory. A future H4-or-F3 task — a real adapter that surfaces filesystem/workspace bytes from a harness trace — would be required to graduate from "mechanism demonstrated" to "phenomenon demonstrated on a real harness trace."

**H3 — `session_sticky` policy (experimental / educational)** (`done`, 2026-05-05)
Each node sharing a `session_id` is placed at one site (per-(state, site) cache reuse, like H1). Implementation: `policies.run_session_sticky` (~50 LOC). `WorkNode` gained `session_id` propagated from `add_subtask` payload; F2 stamps `instance_id`, the H2 multi-session adapter stamps `spec.session_id`. Mixed presence (some nodes have `session_id`, some don't) hard-fails. Placement reasons in audit CSVs carry the session id (`session_sticky:sa`, etc.).

**Finding (negative result, locked into tests).** session_sticky is **provably ≥ H1 by construction**: it solves the same per-state-cost minimization at one site under a strictly tighter constraint. On H2 it coincides with H1 numerically (each session's nodes already share a per-node best-site = its workspace home). On a constructed fixture where two intra-session nodes have private workspaces with different homes and no shared state, session_sticky pays ~1.6 s artifact_copy that H1 (and D2) avoid. The invariant `session_sticky.total_cost_s() >= H1.total_cost_s() - eps` is asserted across toy / H2 / F2-derived fixtures.

session_sticky's only useful regime is "I have an external reason to keep a session co-located that the cost model doesn't capture" — e.g., GPU memory pinning, regulatory data residency, sticky-routing for affinity caches. Vagrant's MVP cost model doesn't carry those terms, so session_sticky ships as an **explanatory baseline**, not a competitive policy. Marked `experimental/educational` accordingly.

**H4 — real workspace-bytes helper** (`done`, 2026-05-05)
`src/vagrant_agent/workspace.py` exports `compute_repo_bytes(path, exclude_patterns=(".git",)) -> int`. Walks a directory tree with `os.walk(followlinks=False)`, sums `os.lstat(...).st_size` (so symlinks contribute their own inode size, not their target's; symmetric with the directory walk). Default exclusion `.git` reflects the cost-model semantic that a remote materialization would `git clone` from origin rather than copy local git internals across a serving link.

`SessionSpec.workspace_path` (in `swe_agent_multi.py`) now optionally accepts a directory path; when set, the workspace state's `bytes` field is computed from disk via the helper. Setting both `workspace_path` and a non-default `workspace_bytes` hard-fails (silent overrides are footguns).

**Finding.** H4 closes the **synthetic-bytes integration gap** of H2: a 10 MB direction test (`test_real_workspace_bytes_preserve_h1_lt_d2_direction`) and a gated 1 GB pinned-numerical test (`VAGRANT_SLOW_TESTS=1`) prove the H1<D2 mechanism survives end-to-end when the workspace state is sourced from a real filesystem. The **synthetic-trajectory gap** remains for H2 (closed by H5a below); the **real-bytes-on-real-trajectories gap** remains open (H5b).

**H5a — multi-trajectory SWE-agent fixture (real trajectories, synthetic bytes)** (`done`, 2026-05-05)
Closes H2's trajectory-reuse gap by replacing `swe_agent_pilot_s_07.json × 3` with **5 distinct cached pilot-zero trajectories** (Melevir/cognitive_complexity, hsahovic/poke-env, lidatong/dataclasses-json, WIPACrepo/iceprod, asottile/setup-cfg-fmt) — distinct repos, mix of pass/fail outcomes. Reuses the existing `swe_agent_multi.py` adapter unchanged; the canonical fixture is `examples/traces/h5a_multi_trajectory_swe.jsonl`. Workspace bytes remain synthetic (1 GB per session, set by the fixture builder); workspace homes are asymmetric (`phoenix, seattle, phoenix, seattle, phoenix`) so 2 of 5 workspaces live at the minority site.

**Numerical findings** (`compact_kv` × `sites_2site.yaml`):

| Policy | Total cost (s) | vs H2 |
| ------ | -------------- | ----- |
| D1 (`request_level_no_reuse`)        | 0.6545 | larger (5 sessions of per-consumer materialization) |
| H1 (`request_level_with_site_cache`) | 0.2220 | per-session placement; 6 nodes phoenix, 4 nodes seattle |
| D2 (`shared_state_aware`, τ=1)       | 3.4221 | colocates phoenix; pays 2× cross-site workspace = 3.2 s |
| G1 (`g1_brute_force`)                | 0.2220 | oracle ≡ H1 (1024 enumerations) |
| G2 (`g2_local_search`)               | 0.2220 | finds the same floor |

**H1 strictly beats D2 by 3.2 s ≈ 15×** — exactly 2× the H2 gap, because 2 (vs 1) workspaces live at the minority site. Sensitivity grid passes 100% sign-consistent (same bytes-layer mechanism as H2). 21 tests in `tests/test_h5a_multi_trajectory.py` cover structural invariants (5 distinct issue_text content_hashes — proving real-trajectory variation, not s_07 × 5), placement asymmetry, mechanism (homes-all-equal collapses H1 == D2), enumeration cap, and byte-deterministic regenerability.

**Honest framing — what H5a is and is not.** Trajectory text is now real (5 distinct SWE-bench instances), so the H1<D2 mechanism survives **trajectory variation**, not just s_07 replay. The load-bearing workspace bytes are still synthetic (1 GB integers from the fixture builder), so H5a graduates the H2 finding from "one-trajectory-replayed-N-times" to "N-distinct-trajectories", but **does not** close the real-bytes gap. To claim "phenomenon demonstrated on real harness traces" we need H5b.

**H5b — real workspace bytes on the H5a trajectories** (`done`, 2026-05-05 — **honest negative finding**)
Shallow-cloned the 5 H5a upstream repos at HEAD (`scripts/h5b/clone_repos.sh`, ~50 MB total network) and re-ran the H5a fixture with `SessionSpec.workspace_path` set per session in place of synthetic 1 GB `workspace_bytes`. Homes held identical to H5a (`phoenix, seattle, phoenix, seattle, phoenix`) so that the only variable changed is the byte source. Trace is generated dynamically per-environment, not committed (real bytes drift with HEAD).

**Working-tree byte sizes at HEAD (excludes `.git`):**

| sid | repo                          | bytes (snapshot) |
| --- | ----------------------------- | ---------------- |
| cog | Melevir/cognitive_complexity  | 21,922           |
| pok | hsahovic/poke-env             | 21,588,279       |
| dcj | lidatong/dataclasses-json     | 301,091          |
| ice | WIPACrepo/iceprod             | 11,568,017       |
| scf | asottile/setup-cfg-fmt        | 57,062           |

**Numerical findings** (`compact_kv` × `sites_2site.yaml` @ 5 Gbps):

| Policy | Total cost (s) | Note |
| ------ | -------------- | ---- |
| H1 (`request_level_with_site_cache`) | 0.148675 | per-session placement (6 phx, 4 sea) |
| D2 (`shared_state_aware`, τ=1)        | 0.148675 | colocates **at seattle** (faster prefill) |

**`D2 ≡ H1` to numerical noise (gap < 1e-9).** The H5a/H2 H1<D2 finding **does NOT survive at real byte magnitudes for these instances at HEAD**. Sensitivity grid: **0% gap survival** across the bracketing `kv_bytes ∈ {10K, 70656, 327680} × link_bps ∈ {5e9, 25e9, 100e9}` grid (sign-consistent at 0).

**Why.** `shared_state_aware` is free to colocate the whole component at the *faster* site (seattle, 1.5× phoenix prefill). At synthetic 1 GB the workspace cross-site cost dominates and forcing colocation pays 2 × 1.6 s (the H5a result). At real HEAD-sized repos (10s of MB), the prompt-context replay savings from picking the faster site exactly cancel the cross-site workspace-transfer cost. The H1<D2 mechanism is real but byte-magnitude sensitive — sub-threshold for these particular instances at HEAD against this 5 Gbps link.

**Mechanism is preserved (locked into tests).** `test_synthetic_1gb_recovers_h5a_gap` runs the same trajectories with synthetic 1 GB workspace_bytes and recovers H1 < D2 by 3.2 s exactly — the H5a result. Proves the gap is real, just byte-scale-sensitive. 13 tests in `tests/test_h5b_real_bytes.py`, env-var-gated on `VAGRANT_H5B_WORKSPACES` (default `/tmp/h5b_workspaces`); auto-skip when clones absent.

**Caveat — HEAD vs `base_commit`.** The cached pilot trajectory JSON does not surface each instance's SWE-bench pre-fix `base_commit`, and we don't load the SWE-bench dataset metadata locally. HEAD-of-`main` is defensible as "real bytes from the same upstream repo at a real commit"; the H1<D2 mechanism is bytes-layer, so byte regime matters more than exact commit. A higher-fidelity H5b' would `git checkout` each instance's `base_commit` before the byte sum, but the qualitative finding (gap collapses at real-repo scale) does not depend on exact-commit fidelity.

**Phenomenon-demonstrated gate status: NOT MET.** The gate (TASKS.md ~487-495) requires the H1<D2 gap to survive on at least one real-trace fixture *with* sensitivity-grid robustness. H5b is the strongest real-trace fixture vagrant currently has (real trajectories AND real bytes), and it produces a 0% survival rate. Paths forward: (a) larger upstream repos (monorepos at >100 MB), (b) a slower link (~1 Gbps), (c) a less prefill-asymmetric site config, or (d) accept the gap is byte-magnitude-sensitive and reframe what the headline claims. Each is a follow-on workstream, not a vagrant MVP item.

**`shared_state_aware` status — revised.** Was: "experimental; provably worse than H1 on H2/H4/H5a, where 1 GB synthetic workspaces force the gap." Now: D2 is **not strictly dominated at real-repo scale on this fixture** — D2 ≡ H1 to within numerical noise on H5b at HEAD, courtesy of an exact cancellation between cross-site workspace transfer cost and seattle-side prefill savings. The "deprecate after H5b" plan from H4's writeup is **withdrawn**: parity at real bytes is not the strawman behavior the deprecation was based on. Whether D2 is *strictly better* than H1 on any real-trace fixture is still open — H5b shows parity, not advantage.

## § Workstream I — Capacity, queues, multi-site (deferred)

Network capacity, prefill GPU-seconds budget, decode admission, ≥ 4 sites, link congestion, recovery-time CDFs. This is the second paper.

## § Workstream J — Live migration / KV (deferred indefinitely)

Per the plan's own non-goals (lines 861–867). Out of scope for this repo.

---

## Definition of done — MVP pipeline (not phenomenon)

```text
1. examples/traces/toy_subagent_trace.jsonl exists, replays, and is regeneratable from a fixed seed.
2. vagrant-manifest build produces a manifest with the expected sharing structure (state_id + content_hash both present).
3. costs.py has unit tests for all four formulas, including a bandwidth (not token-length) crossover.
4. choose_min_cost_mode is implemented and tested.
5. request_level_no_reuse and shared_state_aware policies both emit plans (with reason fields) on the toy manifest.
6. vagrant-bench produces results.csv, state_materialization_breakdown.csv, and plots/duplication_factor.png.
7. The two bars differ on cost-weighted duplication factor, and a pytest assertion encodes that gap.
8. README.md walks a reader from clone -> plot in <10 minutes.
9. README and any docs explicitly frame this as "framework can express the phenomenon," not "phenomenon demonstrated."
```

## Definition of done — phenomenon demonstrated

> **Gate revised 2026-05-05 (post-H1).** H1 (`request_level_with_site_cache`) collapsed numerically to D2 (`shared_state_aware`) and G1 (`g1_brute_force`) on every existing fixture (toy, g_demo at tau=1, SWE-agent F2 s_07). The previously-claimed 2× / 7.47× gaps between D1 and D2 are entirely explained by **per-site cache reuse** (H1's bookkeeping), not by shared-state-aware grouping. The original gate language conflated those two effects.

> **Gate revised 2026-05-05 (post-modeling-assumption audit).** The gap must be **sensitivity-robust** across the realistic 2025–2026 cost-model design space, not just a point estimate at fixed constants. The single constants `kv_bytes_per_token`, `dst_prefill_tok_s`, and `link_bps` each span >1 order of magnitude across plausible production deployments (V4-class compact KV ≈ 10K bytes/tok, DeepSeek-V3 MLA ≈ 70K, vanilla GQA FP16 ≈ 320K; H100 single-stream prefill 15K–60K tok/s; AWS single-flow inter-region ≈ 5 Gbps vs. RDMA-class ≈ 400+ Gbps). A claim that survives only at the 25 Gbps × 70K-bytes/tok point but flips elsewhere is not a phenomenon claim.

The cost-weighted duplication-factor gap **between `shared_state_aware` (D2) and `request_level_with_site_cache` (H1)** must reproduce on **at least one of**:

- a real harness trace exercising **multiple sessions** OR multi-private-state-with-different-homes structure (i.e., a manifest where some nodes' per-node best-site differs from their component's best-site), or
- a synthetic sweep where private-state homes pull individual nodes within a shared-state component toward different sites, so D2's grouping forces a more expensive private-state materialization that H1 avoids by splitting,

**AND** the gap must survive the `vagrant-sensitivity` sweep across the bracketing grid `kv_bytes ∈ {10000, 70656, 327680}` × `link_bps ∈ {5e9, 25e9, 100e9}` (i.e., positive `gap_robust` at ≥ 50% of grid points, with sign consistency — not flips). A gap that exists only at one corner of the grid is not a phenomenon.

The `request_level_no_reuse` (D1) baseline does **NOT** satisfy this gate; it is a strawman whose gap is closed by per-site caching alone.

**Documented finding (not a regression).** On linear-session traces (toy, g_demo, F2 SWE-agent s_07), `H1 ≡ D2 ≡ G1` numerically to 1e-9 at every sensitivity grid point (`tests/test_sensitivity.py:test_gap_survival_rate_on_default_toy_at_realistic_link` asserts 0% survival on the toy, and D1 < H1 strict inequality at 100% of grid points). This is a real property of those fixtures, not a bug in either policy. The constructed-divergence test (`tests/test_h1_policy.py:test_h1_diverges_from_d2_on_constructed_fixture`) proves H1 ≠ D2 fixtures *exist*, and the H2 multi-session fixture (`examples/traces/h2_multi_session_swe.jsonl`) and the H5a multi-trajectory fixture (`examples/traces/h5a_multi_trajectory_swe.jsonl`) both sustain a 100% sensitivity-robust H1<D2 gap — but **only at synthetic 1 GB workspace_bytes**. **H5b — real bytes on the same H5a trajectories at HEAD-sized upstream repos (~33 MB total) — closes that corner with a 0% survival rate**: the synthetic 1 GB scale was load-bearing for the gap, and at HEAD-sized real repos the prompt-context replay savings from picking the faster site exactly cancel the cross-site workspace-transfer cost. The phenomenon-demonstrated gate is therefore not met at the real-trace + real-bytes corner; what's required to close it is a fixture with workspace bytes well above the ~50 MB regime-flip threshold (monorepo-scale repos, slower link, or less prefill-asymmetric site config) — see the H5b writeup in Workstream H for the full breakdown.

**Modeling-assumption caveats.** The cost model omits (a) KV compression à la CacheGen (3–4× factor on `kv_transfer_s`), (b) inter-state pipeline overlap (real systems compute `max(transfer, prefill)`, vagrant sums), (c) decode time (cancels in policy differences). See `costs.py` docstring for full caveat block. Any phenomenon claim must be robust to these omissions, or must explicitly justify why the omitted term doesn't flip the headline.

## Definition of done — research result

A real harness trace, ≥ 2 model profiles, ≥ 4 sites, capacity-aware costs (Workstream I), and at least one competitive baseline (Workstream H). Out of scope until F is green.

---

## Minimal first batch (MVP path, in order)

1. A1 — `vagrant_agent/events.py` (8 string constants + payload docstrings).
2. A2 — ledger_progress pass-through hook (upstream) + 4-test invariant set.
3. A3 + A5 — deterministic synthetic generator + segment-hashing helper.
4. A6 — commit toy trace JSONL; document tau=1 vs tau=1000 behavior.
5. B1 + B2 + B3 + B4 — manifest with `state_id`+`content_hash`, bipartite source + pairwise edge view, JSON schema, CLI.
6. Tests for expected consumer counts on the toy manifest.
7. C1 + C2 — profiles and four cost formulas.
8. C3 — bandwidth crossover test (not token-length).
9. C4 — `choose_min_cost_mode`.
10. D1 — `request_level_no_reuse` (named explicitly, not abbreviated).
11. D2 — `shared_state_aware` with `--tau` exposed.
12. D3 — plan emission with `reason` fields.
13. E4 — `state_materialization_breakdown.csv` (audit before plotting).
14. E2 — cost-weighted `shared_state_duplication_factor`.
15. E3 + E1 — plot and `vagrant-bench` CLI wiring.
16. README walkthrough; framing discipline ("expresses the phenomenon").

Stop here. Decide whether to continue to F.

Do not let the agent touch adapters, queues, ILP, real harnesses, or extra policies until step 16 is green.

---

## Open questions

These are real uncertainties; resolve as you go, do not block the MVP on them:

- **Token counting at trace time.** When the harness is real, do we get exact input-token counts per LLM call, or do we estimate from text? Synthetic adapter assumes exact; the SWE-agent retrospective pipeline does not always have it.
- **Content hashing granularity.** Hash whole segments, or hash chunks (e.g., 512-token windows)? Whole-segment is simpler and probably enough for the MVP. Chunked sharing is a Workstream H question.
- **`tau` choice.** MVP defaults to 1 token (any sharing). Real workloads may need a higher threshold to avoid trivial sharing dominating. Add a sensitivity study only after F is green.
- **State-object identity across reopen/invalidate.** When a state is invalidated and a fresh version is written, is it a new state object or the same one with a new content_hash? MVP treats it as a new object; revisit if real traces make this ambiguous.
