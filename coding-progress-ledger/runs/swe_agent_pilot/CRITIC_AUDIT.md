# Critic audit — H4–K vs. the user's stated mission

This synthesizes four parallel critic reports and the surgical fixes
taken in response. The user's stated mission is:

> "an automated way to check and query progress for long range
> agentic tasks with a ledger design."

Each underlined verb maps to a specific code surface; each surface
was audited.

## 1. The four critics, one verdict each

| Critic | Verdict |
|---|---|
| **Mission-fit** | "0 of 4 (H4, I, J, K) advanced the mission; all 4 consolidated retrospective annotation infrastructure. The pilot infrastructure is backwards: you should be instrumenting a live agent and answering live progress queries; instead you have spent A–M building a high-fidelity retrospective annotation lab whose output is CSV files." |
| **Observation-channel correctness** | "5 of 10 mission features are first-class. Feature 3 half-exposed. Features 7, 8, 9, 10 (validation attempts, evidence strength, stalled intervals, newly discovered scope) have no direct columns. K1 evidence audit is orphaned from the channel." |
| **Long-range / scale** | "The framework is scoped to short single-task SWE-bench-shaped traces. Two load-bearing limiters: (a) longest tested trace is f_02 at 509 steps inside one bug-fix; (b) zero timestamp anywhere in core/serialization/importer/dataset. LedgerSet (Workstream T) has zero implementing code." |
| **Automation / live** | "~15% of the mission delivered. All 25 ledgers were authored by LLMs reading dead traces (`Claude (... AI-driven first pass)`, `Opus subagent (... cold pass)`). No `observe()` hook, no sidecar, no callback, no live query API. Workstream N is `not started, out of pilot scope`." |

## 2. Mission-verb → gap → action map

| Mission verb | Concrete gap (critic) | Surgical fix taken | Strategic gap remaining |
|---|---|---|---|
| **automated** | No agent-side emission; no `observe()` / sidecar / callback. All ledgers are post-hoc. | (none — needs a real design pass) | **Workstream N must be promoted from "out of pilot scope" to next.** |
| **check and query** | Only `score()` and `active_*_leaves()` were first-class. | Added `current_step`, `active_blocked_leaves`, `reopens_since`, `newly_discovered_since`, `last_validation_event`, `stalled_for` to `ledger_progress/queries.py`. Eight tests at `tests/test_live_query_api.py`. | A live query *server* (HTTP / CLI watch / monitor) does not exist. |
| **long range** | Zero timestamps anywhere. `LedgerEvent` has step-only. `LedgerSet` (T) is paper. | Added optional `timestamp: str \| None` to `LedgerEvent`, round-tripped through serialization, auto-stamped by `LedgerSession` (override via `clock=lambda: None`). Six tests at `tests/test_timestamp_field.py`. | LedgerSet (Workstream T) still zero implementing code; multi-issue projects unrepresentable. |
| **progress** (10 features) | Features 7, 8, 9, 10 missing from observation channel. K1 evidence audit orphaned. | Added 17 new columns to the channel: per-category progress (`product_progress` / `validation_progress` / `investigation_progress`), per-step event windows (`step_added_subtasks`, `step_*_completes`, etc.), evidence strength (`step_strong_completions` / `cum_*` — K1 classifier now wired in), stalled intervals (`steps_since_progress_increase` / `_completion` / `_subtask_added`). Eight tests at `tests/test_channel_mission_features.py`. | None — all 10 mission paragraph features now first-class. |

## 3. Score after the surgical fixes

- Tests: **283 passing** (was 261 before the audit; +22 from new invariants).
- Mission-paragraph features (1–10): **10/10 first-class as columns** (was 5/10).
- Live query functions: **6 added** to `queries.py` (was 2).
- Timestamp support: **present** end-to-end (was absent).
- SWE-agent observation dataset: rebuilt; integrity passes; **191/191 native rows**, zero warnings.

What this **does not** fix:
- Automation pillar: still ~15% delivered. No agent emits events live; no monitor queries live state.
- Long-range pillar: timestamps are *present in the schema* but every existing ledger was emitted by a clock-`None` driver, so no real wall-clock data exists yet. New ledgers from running agents would carry timestamps.
- Multi-task aggregation: LedgerSet still unimplemented.

## 4. Strategic recommendations (mission-fit critic, lightly amended)

**Kill / defer indefinitely**
1. **Workstream R** (external write-up) — premature; locks in the retrospective framing as the product.
2. **Workstream P** (cross-source generalization, P1–P3) — adds annotation surface on a hypothesis (progress shape) that has no live consumer.
3. **Workstream O** (100-trace retrospective scale-out) — the M1 memo's smoke test runs at chance by design; trace 21+ is debt.

**Promote to immediate priority**
1. **Workstream N — Live SWE-agent instrumentation.** The single workstream that makes "automated" true. The defer-because-it's-expensive argument is exactly backwards: those weeks of engineering are the product. Sub-tasks N1 (sidecar vs in-agent) and N2 (one minimal hook) are the next move.
2. **Live query CLI / monitor.** The query *functions* exist after this commit; the CLI surface (`ledger-run watch`, `ledger-run query --status blocked --since-step N`, optional `ledger-run serve`) does not. Small build, large mission impact.
3. **Workstream T1 — LedgerSet protocol doc.** Unblock the only roadmap item that addresses multi-task scope.

**The one-sentence call.** The pilot infrastructure (A–M) was not wasted — it produced a clean schema and 20 high-fidelity retrospective annotations with strong inter-annotator agreement — but everything *forward* should be live-instrumentation-shaped, not annotation-shaped.

## 5. Next-step proposal (concrete)

If the user agrees with the strategic recommendation, the next move is one workstream (estimated 1–2 weeks):

```text
Workstream N (live instrumentation, sidecar branch)
  N1.  decide sidecar vs in-agent (sidecar likely wins for portability)
  N2.  build a minimal stdin-JSONL → ledger.jsonl sidecar:
         python -m ledger_progress.sidecar --run-dir X
         consumes one line of structured agent step per stdin event
         emits LedgerSession events using existing add/complete/start/block API
  N3.  hook one SWE-agent run on a known-success and known-failure instance
  N4.  parity report: live ledger vs retrospective ledger same instance
  N5.  add `ledger-run watch <run_dir>` so the CSV / progress.csv updates
       on each new event (re-derive incrementally)
```

The query API (this commit) is what `watch` and any future monitor consume. The timestamp field (this commit) is what makes deadline modeling possible once a live agent is emitting events.

## 6. Pointers

- Mission-fit critic report: in conversation log.
- Observation-channel critic report: in conversation log.
- Long-range / scale critic report: in conversation log.
- Automation critic report: in conversation log.
- New code: `ledger_progress/queries.py` (live query API); `ledger_progress/core.py` + `ledger_progress/serialization.py` + `ledger_progress/session.py` (timestamps); `scripts/build_ledger_observation_dataset.py` (17 new mission-feature columns).
- New tests: `tests/test_timestamp_field.py`, `tests/test_live_query_api.py`, `tests/test_channel_mission_features.py`.
- Existing M1 memo (defers N): `runs/swe_agent_pilot/GO_NO_GO_MEMO.md` § 9 — recommend revising in light of this audit.
