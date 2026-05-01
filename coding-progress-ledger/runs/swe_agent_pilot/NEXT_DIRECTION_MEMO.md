# M2 (revised) — Next direction memo (post-CRITIC_AUDIT)

This supersedes the original M2 framing (*"define next sample size for retrospective scale-out"*). It records the pivot forced by the four-critic audit on 2026-04-30 and states the gating criterion for the next phase.

**One-line:** *Move from retrospective annotation of dead traces to live instrumentation that emits a queryable ledger for any agent framework.*

## 1. What the pilot proved (and didn't)

The 20-trace retrospective pilot (A–M) **proved** the schema, the protocol, and the observation channel survive contact with real agent traces:

- Schema: zero core changes needed (I2).
- Protocol: 5/5 quadrant inter-annotator agreement (H4).
- Channel: 191/191 native rows, all 10 mission features now first-class columns (post-audit fix `5bdcab6`).
- Failure-mode discrimination: progress signal genuinely decoupled from outcome (F4).

The pilot **did not** prove anything about the mission's load-bearing verbs:

- *Automated:* every one of the 25 SWE-agent ledgers was authored by an LLM reading a dead trace post-hoc. Zero live emissions.
- *Check and query:* the query API shipped (commit `5bdcab6`) but has no CLI / monitor / server surface.
- *Long range:* the longest trace is 509 steps inside one bug-fix; multi-issue, multi-day, multi-week scope is unrepresentable today (no `LedgerSet`, until 2026-04-30 no timestamps).

## 2. Why retrospective scale-out is not the next step

The original M1 memo recommended *"scale retrospective to 100 traces, gated on H4."* H4 passed, but the four-critic audit found that:

- The smoke test on retrospective data runs at chance by design (M1 § G2.1). 80 more retrospective ledgers buy statistical power for a measurement the framework explicitly says is meaningless on retrospective input.
- The retrospective channel cannot close the K2 evidence gaps (hidden-work-gap visibility, agent-vs-harness submit provenance, pre-fix baseline runs). Only live instrumentation can.
- Annotating 80 more dead traces locks in the retrospective framing as the product. The mission says "automated"; that's not what 80 more annotators produces.

Cost-of-being-wrong (per M1 § 11) for the original recommendation: 20–30 hours of re-annotation if the protocol revision changes a leaf. Cost-of-being-wrong for the live pivot: roughly the same, but produces the actual product.

## 3. The pivot

**New direction:** build live instrumentation that any agent framework can target, with a stable wire-format protocol and a sidecar that maintains a queryable ledger as the agent runs.

Concrete deliverables (workstreams, in priority order):

| Workstream | Deliverable | Mission verb |
|---|---|---|
| **N1** | Decision doc + wire-format protocol (`docs/LIVE_INSTRUMENTATION_DECISION.md`) | (foundation) |
| **N2** | `ledger_progress/sidecar.py` — consumes JSONL events, maintains a live `LedgerSession`, writes `ledger.jsonl` with timestamps | automated |
| **N3** | One live SWE-agent run (success + failure) with the sidecar | automated |
| **N4** | Live-vs-retrospective parity report on the same instance | (validation) |
| **U1** | `ledger-run watch <run_dir>` CLI — tails ledger.jsonl, prints live progress | check and query |
| **U2** | `ledger-run query <run_dir> --status blocked / --stalled-for ge N / --reopens-since N` | check and query |
| **T1** | `docs/LEDGER_SET_PROTOCOL.md` — multi-issue scope unblocker | long range |
| **V1** | Wall-clock columns on the observation channel (`elapsed_seconds`, etc.) | long range |

Workstreams T1, K3 (cheap classifier win), and N1 can run in parallel.

## 4. Gating criterion for live N=20

**Do not** extend live instrumentation to N=20 (Workstream N5) unless N4's parity report demonstrates:

```text
1. Same final coding-progress on at least 2 SWE-bench instances within 0.05
   when comparing live ledger vs the existing retrospective ledger for the
   same instance.
2. K2 gaps closed: live captures at least one signal that retrospective
   could not reconstruct (e.g. real submit-vs-harness-termination provenance,
   real pre-fix baseline test result, in-process repro evidence).
3. Sidecar runs without modifying the agent's source code (portability test).
4. Total wall-clock latency of the sidecar < 100ms per agent step
   (so it doesn't change agent behaviour).
```

Cost of being wrong about the pivot:

| If… | Cost |
|---|---|
| Live parity fails at N4 | ~1 week of sidecar work; protocol stays useful for the *next* agent framework attempted. |
| Sidecar's heuristic event inferrer is too lossy | Falls back to in-agent mode for that framework; integrate `ledger_ops` directly. The hybrid design is the safety net. |
| The wire format spec needs revision | Bump `schema_version` to 1.1; old agents keep working. |
| The pivot is wrong and retrospective should have continued | Worst case ~2 weeks lost; pilot artifacts are unchanged and re-usable as parity benchmarks. |

The pivot is cheap to be wrong about because the retrospective channel stays as the parity benchmark for any live work. Nothing already shipped is invalidated.

## 5. Definition of "next phase done"

The next phase ships when:

```text
N1 — wire-format protocol + decision doc committed
N2 — sidecar implementation, tests green
N3 — at least one live SWE-agent run produces a real-time ledger.jsonl with timestamps
N4 — live-vs-retrospective parity report meets § 4 gating criteria
U1 — `ledger-run watch` works on the live ledger
T1 — LedgerSet protocol doc committed (parallel)
```

After that, M3 will be the next direction memo. Workstream R (paper write-up) becomes meaningful the moment N4 ships, not before.

## 6. Pointers

- Critic audit: `runs/swe_agent_pilot/CRITIC_AUDIT.md`
- Original M1 memo (now superseded): `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`
- Concretized N1–N5 + U + V + T promotion: `TASKS.md` § "Forward priorities (post-CRITIC_AUDIT)"
- New surface code shipped 2026-04-30: `ledger_progress/queries.py` (live query API), `ledger_progress/core.py:LedgerEvent.timestamp`, `scripts/build_ledger_observation_dataset.py` (17 new mission-feature columns).
