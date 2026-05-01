# N1 — Live instrumentation decision and wire-format protocol

This satisfies `TASKS.md` § Workstream N, task **N1**. It is the load-bearing design doc for everything after — N2 (sidecar implementation), N3 (first live run), N4 (parity report), and downstream agent-framework integrations beyond SWE-agent.

**One-line:** *Hybrid sidecar with a stable wire-format protocol; agents emit JSONL step records (low-friction default) or explicit ledger ops (high-fidelity opt-in); the same sidecar consumes both.*

## 1. The decision

| Option | Verdict | One-line reason |
|---|---|---|
| In-agent only (patch each framework's source to call `LedgerSession`) | **rejected** | every new framework = bespoke integration; tightly couples ledger to one agent's internals |
| Sidecar only (parse stdout heuristically, no agent cooperation) | **rejected as sole mode** | ceiling on fidelity; can't capture agent intent at boundaries (e.g. "I'm now investigating X") |
| **Hybrid: stable wire format, sidecar consumes both raw step records and explicit ledger ops** | **chosen** | works for any framework that can `print(json.dumps({...}))`; agents that integrate deeply get full fidelity via one extra optional field |

The hybrid is the only option that makes the framework downstream-usable for **any** agent framework — Claude Code, LangGraph, OpenAI Assistants, custom RL, SWE-agent — without forking each one.

## 2. The wire-format protocol (v1.0)

The protocol is **the** interface. Once shipped, agents compile against it; the sidecar's internal design can change but the wire format cannot break without a `schema_version` bump.

### 2.1 Event shape

One JSONL line per agent step:

```json
{
  "schema_version": "1.0",
  "run_id": "swe_agent_pilot_live_001",
  "step": 17,
  "timestamp": "2026-04-30T12:34:56.123456+00:00",

  "agent_step": {
    "thought": "I should replace the request URL.",
    "action": "edit",
    "command": "edit 274:274 ...",
    "files_touched": ["iceprod/core/functions.py"],
    "observation": "tool ack: line 274 updated",
    "exit_status": null,
    "tool_name": "edit"
  },

  "ledger_ops": [
    {"op": "add",      "id": "S2", "category": "PRODUCT",
     "description": "Replace getip.php request"},
    {"op": "complete", "id": "S2",
     "evidence": ["step 17: edit 274:274 acknowledged"]}
  ]
}
```

### 2.2 Field semantics

**Top-level (required):**
- `schema_version` — the wire-format version this event targets. Sidecar refuses unknown major versions.
- `run_id` — identifies the live run. Multiple agents may emit to one sidecar; `run_id` is the routing key.
- `step` — monotonic per `run_id`. Gaps are allowed (skipped steps); the sidecar maps `step` to `LedgerEvent.step`.
- `timestamp` — ISO-8601 UTC. The sidecar always uses this if present (overrides its own clock).

**`agent_step` (optional, sidecar-mode input):**

The raw observation. If present and `ledger_ops` is empty, the sidecar runs `infer_events(agent_step)` over `(thought, action, command, observation)` to derive ledger ops. The inferrer's heuristics are framework-specific (a per-framework module, e.g. `ledger_progress.adapters.swe_agent`).

Fields:
- `thought`, `action`, `command`, `observation` — mirror the existing `docs/SWE_AGENT_TRACE_SCHEMA.md` shape so retrospective imports and live emissions share one format.
- `files_touched` — list of repo-relative paths. Used by the inferrer to classify edits as PRODUCT vs ENVIRONMENT.
- `tool_name` — discriminator for the inferrer's vocabulary→category map.
- `exit_status` — `null` until the run terminates; final value records harness-vs-agent submit provenance (closes K2's submit-provenance gap).

**`ledger_ops` (optional, in-agent-mode input):**

Explicit ledger ops the sidecar applies verbatim via `LedgerSession`. If non-empty, `agent_step` is recorded for context but the inferrer is **not** consulted — the agent's own model wins. Each op is one of:

```text
{"op": "add",        "id": "S1", "category": "...", "description": "...",
                     "weight": 1.0, "parent_id": null}
{"op": "start",      "id": "S1", "evidence": ["..."]}
{"op": "complete",   "id": "S1", "evidence": ["..."]}
{"op": "block",      "id": "S1", "reason": "...", "evidence": ["..."]}
{"op": "reopen",     "id": "S1", "reason": "..."}
{"op": "invalidate", "id": "S1", "reason": "..."}
{"op": "split",      "id": "S1", "reason": "...",
                     "children": [{"id": "S1.1", "description": "...", "category": "..."}]}
{"op": "add_evidence","id": "S1", "evidence": ["..."]}
```

The op vocabulary mirrors `LedgerSession`'s public API exactly. No new verbs; the protocol is just a wire-format projection of the existing API.

### 2.3 Replay-safety

Two strict invariants:

1. **Idempotence:** re-feeding the same JSONL stream to a fresh sidecar produces a byte-identical `ledger.jsonl`. (The protocol contains no relative state; every event is self-contained given `run_id` + `step`.)
2. **Timestamp authority:** if the agent emits a `timestamp`, the sidecar uses it. The sidecar never overrides agent-stamped wall-clock (so traces replayed offline are time-faithful).

Both invariants are testable; N2's test plan must include a replay-equality test and a timestamp-authority test.

### 2.4 Versioning

`schema_version` follows semver-like rules:

```text
1.0 → 1.1: additive (new optional fields). Old sidecars ignore unknown fields.
1.x → 2.0: breaking. Sidecar refuses 2.x events with version mismatch error.
```

The protocol commits to 1.0 stability for at least one year after N3 ships.

## 3. Sidecar surface (informative; N2 implements)

```bash
# stdin mode (one JSONL event per line)
agent | python -m ledger_progress.sidecar \
    --run-dir runs/<live_run_id> \
    --adapter swe_agent

# file-tail mode (agent writes to a file; sidecar tails it)
python -m ledger_progress.sidecar \
    --run-dir runs/<live_run_id> \
    --input-file /var/log/agent_events.jsonl \
    --adapter swe_agent
```

Behaviour:

- Maintains one `LedgerSession` per `run_id`.
- On each input line: validates schema, applies `ledger_ops` (if any) or runs `adapter.infer_events(agent_step)`, then writes the resulting `LedgerEvent`s to `runs/<run_dir>/ledger.jsonl` (append-only).
- After every batch, also re-derives `progress.csv` and `summary_by_category.json` (so `ledger-run watch` (U1) can show live progress without reimplementing).
- Refuses unknown adapters; the adapter set is per-framework and lives under `ledger_progress/adapters/`.
- Latency target: < 100ms per event (well within the agent step granularity).

## 4. Adapters (framework-specific inferrers)

Each adapter is a Python module exposing `infer_events(agent_step: dict) -> list[dict]`. The same vocabulary→category map already documented in the retrospective addendum is the inferrer's seed:

- `ledger_progress/adapters/swe_agent.py` — first adapter; lifts the existing `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md` § 1 vocabulary-to-category map directly. N2 ships this.
- `ledger_progress/adapters/generic.py` — fallback that emits one `add+complete` PRODUCT subtask per `tool_name` invocation. Lossy but always works. N2 ships this too.
- `ledger_progress/adapters/<framework>.py` — added per framework as adoption demands (Claude Code, LangGraph, OpenAI Assistants, etc.). Each is ~50–150 lines.

This is what makes the tool downstream-usable: the wire format is universal; the adapter set grows as frameworks adopt it.

## 5. Why this works for the four mission verbs

| Mission verb | How this design serves it |
|---|---|
| **automated** | Agents emit JSONL during execution; the sidecar maintains a live `LedgerSession` without manual annotation. Today's 25 LLM-authored retrospective ledgers become the parity benchmark, not the product. |
| **check and query** | `LedgerSession` is in memory while the sidecar runs. Workstream U (`ledger-run watch / query`) reads the live `ledger.jsonl` and answers progress questions in real time, using the query API shipped 2026-04-30. |
| **long range** | `LedgerEvent.timestamp` is wall-clock; agents that run for hours/days produce a time-indexed event stream. Workstream V's wall-clock features (`elapsed_seconds`, `seconds_since_progress_increase`) become meaningful. Workstream T's `LedgerSet` aggregates many `run_id`s into one project view. |
| **progress** (10 mission features) | The observation channel already exposes all 10 features (commit `5bdcab6`); the sidecar's output `ledger.jsonl` flows through the same `build_ledger_observation_dataset.py` pipeline as retrospective ledgers. No second channel; one path for both. |

## 6. What this design explicitly does NOT do

- **No agent-side intelligence.** The sidecar runs heuristic inferrers; it does not call an LLM, does not interpret natural language, does not synthesize subtasks the agent didn't surface. The "do not infer completion from progress" rule (§ 0) extends to "do not infer subtasks from agent affect."
- **No agent-side gating.** The sidecar never blocks the agent. Worst case (sidecar crashes) the agent keeps running; the JSONL stream becomes the post-hoc input to a fresh sidecar invocation.
- **No prediction.** The sidecar emits ledger events. Predicting "will this finish on time" is the estimator's job (Workstream Q / V2), consuming the channel features the sidecar produces.
- **No HTTP server in v1.** That's Workstream U3, optional, only built if U1+U2 prove the demand.

## 7. Acceptance criteria (for this decision doc)

```text
chosen branch documented (hybrid sidecar)
wire-format protocol v1.0 specified with field-by-field semantics
replay-safety invariants stated and testable
adapter pattern defined; first two adapters named (swe_agent, generic)
mission-verb mapping explicit (§ 5)
explicit non-goals stated (§ 6)
```

## 8. Acceptance criteria (for N2, the implementation)

The sidecar implementation must pass:

```text
ledger_progress/sidecar.py exists and exposes a __main__ entry point
adapters: ledger_progress/adapters/swe_agent.py and .../generic.py
tests: tests/test_sidecar.py covers
    - synthetic 5-step JSONL → 5-event ledger.jsonl, all timestamped
    - schema_version mismatch raises
    - replay-equality: re-feeding the same JSONL produces byte-identical ledger.jsonl
    - timestamp-authority: agent-stamped timestamps survive
    - ledger_ops mode: explicit ops bypass the inferrer
    - agent_step mode: inferrer produces sane events for swe_agent vocabulary
    - run-dir invariants: ledger-run check-run passes after the run
sidecar latency: < 100ms per event under synthetic load
no agent code changes required for the swe_agent adapter
```

## 9. Open questions (deferred to N3+)

These are not blockers for N2:

- Should the sidecar emit a `manifest.json` summary at the end of a run? (Likely yes; N3 demand-test.)
- How does the sidecar handle multi-process agents (multiple writers to one `ledger.jsonl`)? v1 is single-writer; concurrency is a future extension.
- When an adapter's heuristic disagrees with explicit `ledger_ops` mid-stream, who wins? Spec says `ledger_ops` always wins; this is the documented hybrid contract.
- Should `infer_events` be deterministic? Yes — locked in by the replay-equality test in N2's acceptance.

## 10. Pointers

- Mission paragraph: `runs/swe_agent_pilot/GO_NO_GO_MEMO.md` (M1) and `CRITIC_AUDIT.md` § 1.
- Wire format informed by: `docs/SWE_AGENT_TRACE_SCHEMA.md` (the retrospective shape).
- Op vocabulary mirrors: `ledger_progress/session.py:LedgerSession` (the public API).
- Adapters follow: `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md` § 1 (vocabulary→category map).
- Consumers: `ledger_progress/queries.py` (the query API) and `scripts/build_ledger_observation_dataset.py` (the channel) — both already mission-aligned post-`5bdcab6`.
- Next step: N2, `ledger_progress/sidecar.py`. Decision doc complete.
