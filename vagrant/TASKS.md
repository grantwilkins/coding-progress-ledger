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

**H2 — Multi-session SWE-agent fixture** (next task — recommended by H1 findings critic)
Concat 2-3 F2-style SWE-agent trajectories with a shared system_prompt state and disjoint per-trajectory workspaces (`home_site` set per trajectory). Smallest realistic fixture that puts D2 in its natural habitat (multiple components with private states pulling in different directions). Decides whether the project pivots to a "site-cache-reuse-is-everything" finding or sustains the original grouping thesis.

**H3 — Other deferred policies.** `session_sticky`, `prefix_group`, `subagent_group`, `workspace_group`. Add only if H2 surfaces a need.

**`shared_state_aware` status.** Marked **experimental** (NOT deprecated). On linear-session traces it is provably equivalent to H1; at fragmenting tau it is strictly worse than H1. Keep it pending H2 — if multi-session real traces also collapse, deprecate then.

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

The cost-weighted duplication-factor gap **between `shared_state_aware` (D2) and `request_level_with_site_cache` (H1)** must reproduce on **at least one of**:

- a real harness trace exercising **multiple sessions** OR multi-private-state-with-different-homes structure (i.e., a manifest where some nodes' per-node best-site differs from their component's best-site), or
- a synthetic sweep where private-state homes pull individual nodes within a shared-state component toward different sites, so D2's grouping forces a more expensive private-state materialization that H1 avoids by splitting.

The `request_level_no_reuse` (D1) baseline does **NOT** satisfy this gate; it is a strawman whose gap is closed by per-site caching alone.

**Documented finding (not a regression).** On linear-session traces (toy, g_demo, F2 SWE-agent s_07), `H1 ≡ D2 ≡ G1` numerically to 1e-9. This is a real property of those fixtures, not a bug in either policy. The constructed-divergence test (`tests/test_h1_policy.py:test_h1_diverges_from_d2_on_constructed_fixture`) proves H1 ≠ D2 fixtures *exist*, but no real-trace fixture in the repo currently exercises that case.

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
