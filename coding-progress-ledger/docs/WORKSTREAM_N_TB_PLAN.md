# Workstream N_TB — Live ledger on Terminal-Bench-style tasks via subagent runners

Status: **plan, not started** (drafted 2026-05-03).

## 1. Why this workstream exists

Workstream N closed against SWE-agent (N1–N6 ✓). But every "live" run we
have is still upstream-mediated: the sidecar consumes a *normalized* trace
that originated as a recorded SWE-agent trajectory. Wall-clock timestamps
on the N6 batch are **synthetic** (`live_instrumentation.json::timestamp_source
== "synthetic"`) because the upstream traces lack per-step timestamps.

This workstream removes that mediation. A Claude Code subagent runs in an
isolated worktree against a Terminal-Bench-style task spec, emits ledger
events as it actually codes, and the live sidecar ingests the stream in real
time. The result is the first batch of ledgers in this repo whose timestamps
are *physically real* and whose discovered-work sequence reflects an agent's
own choices, not a replay of someone else's.

This is also the cleanest cross-source test left in the project. Hermes
(`H_PARITY`) proved the framework reads a foreign retrospective trace shape;
TB-via-subagent proves it reads a foreign **live** trace shape, with the
agent harness, the task source, and the wall clock all changed at once.

### What this is NOT

- Not a new benchmark for Claude. We are not measuring agent skill; we are
  measuring whether the ledger channel survives contact with a non-SWE-agent
  live source.
- Not a controller. The ledger remains an observation channel — never an
  input to the subagent's decisions.
- Not a Docker port of Terminal-Bench. We adopt TB's *task shape* (instruction
  + verifier + isolated env), but rehost as `uv`-runnable repos in worktrees
  so the subagent can run end-to-end without container plumbing. A future
  workstream may re-add Docker if it becomes load-bearing.

## 2. Mission gate this advances

From `TASKS.md`'s "mission delivered" definition:

> A live agent has emitted at least one `ledger.jsonl` with timestamps
> (Workstream N — N3 acceptance)

N3 is satisfied technically, but the timestamps are synthetic. NTB1–NTB10
satisfy it on first principles: *real agent, real task, real clock.*

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  scripts/run_tb_subagent.py  (driver, runs once per task)          │
│  ─────────────────────────────────────────────────                 │
│  1. Reads tasks/<id>/task.yaml + verifier.sh                       │
│  2. Creates a worktree at runs/tb_live/<id>/repo/                  │
│  3. Spawns subagent via Agent(subagent_type="general-purpose",     │
│       isolation="worktree", prompt=<task + ledger protocol>)       │
│  4. Subagent uses LedgerSession in-band (writes ledger.jsonl       │
│       directly under runs/tb_live/<id>/)                           │
│  5. Driver runs verifier.sh against the worktree → test_output.txt │
│  6. Driver runs `ledger-run check-run` to validate the ledger      │
│  7. Driver runs all four downstream pipelines on the run dir       │
└────────────────────────────────────────────────────────────────────┘
```

### 3.1 In-band emission (primary path)

The subagent imports `ledger_progress` and uses `LedgerSession` per
`docs/AGENT_USAGE.md`. The session's auto-clock supplies real wall-clock
ISO-8601 timestamps. At end-of-run, the subagent calls
`session.export_jsonl("runs/tb_live/<id>/ledger.jsonl")`.

Pros: zero new adapter code, mirrors the protocol shipped to humans,
honest: the agent's own decomposition is what gets recorded.

Cons: relies on the subagent following the protocol. Mitigation: the prompt
includes a worked example, the driver fails the run if `ledger.jsonl`
doesn't exist, and `check-run` rejects malformed ledgers.

### 3.2 Out-of-band emission (stretch path, NTB-stretch only)

A new adapter `ledger_progress/adapters/claude_code_agent.py` consumes a
JSONL stream of Agent tool calls (Read / Edit / Bash / Write) and infers
ledger ops mechanically:

| Tool call          | Inferred op                                                |
|--------------------|------------------------------------------------------------|
| `Bash(pytest …)`   | `start` (if not started) + `complete`/`block` per exit code on the active VALIDATION leaf |
| `Edit` / `Write`   | `add_evidence` on the active PRODUCT leaf                  |
| `Read`             | `add_evidence` on the active INVESTIGATION leaf            |
| `Bash(grep \| rg)` | `add_evidence` on the active INVESTIGATION leaf            |

The adapter is intentionally weak — it cannot infer SPLIT or REOPEN
without semantic parsing. We compare it against the in-band ledger to
quantify how much of the channel is mechanically observable from raw
tool calls (extends W1's event observability matrix to a non-SWE source).

Defer NTB-stretch unless NTB1–NTB10 ship cleanly.

### 3.3 Run-directory layout

```
runs/tb_live/<task_id>/
├── task.md               # cleaned instruction (ported from TB task.yaml)
├── verifier.sh           # the deterministic test (exit 0 = pass)
├── repo/                 # worktree the subagent edits (gitignored)
├── ledger.jsonl          # in-band emission, real timestamps
├── progress.csv          # exported by LedgerSession.export_curve_csv
├── final_diff.patch      # git diff of repo/ at end of run
├── test_output.txt       # captured verifier output
├── transcript.md         # subagent's user-visible turns (driver-captured)
├── live_instrumentation.json  # {timestamp_source: "wallclock", clock_drift_ms, ...}
├── summary_by_category.json   # produced by rescore_run
└── notes.md              # any human after-action notes
```

Directories under `runs/tb_live/<task_id>/repo/` are gitignored just as
the existing `runs/task_*/repo/` are.

## 4. Phases (NTB1 – NTB10)

### NTB1. Task spec format + verifier contract
Status: not started.

Define the on-disk shape every TB-ported task must follow. This is the
single source of truth NTB2–NTB4 build against.

Outputs:
- `docs/TB_LIVE_TASK_FORMAT.md` — the spec.
- `tasks/tb_live/_template/task.md` + `verifier.sh` + `solution_reference/`
  — a worked template.

Acceptance:
- Spec names the four required files (`task.md`, `verifier.sh`,
  `solution_reference/`, `expected_categories.json`).
- Verifier contract: shell-exit 0 ⇔ task complete; non-zero ⇔ incomplete;
  stdout/stderr captured to `test_output.txt`.
- `solution_reference/` is hidden from the subagent (driver excludes it
  from the worktree); used only by the driver to (a) verify the verifier
  itself passes on a known-good solution before the subagent ever runs
  and (b) seed the retrospective re-annotation in NTB8.

### NTB2. Port 12 tasks (the "TB-12")
Status: not started. **The 12 tasks are spec'd in §5 below.**

For each task: write `task.md`, `verifier.sh`, populate `solution_reference/`,
and confirm `verifier.sh` against `solution_reference/` exits 0. Pre-tag
each task with `expected_categories.json` (annotator's hypothesis of how
the leaves should distribute across PRODUCT/VALIDATION/INVESTIGATION) so
NTB8 has a target.

Outputs:
- `tasks/tb_live/<task_id>/` × 12 (see table in §5).
- `tests/test_tb_live_verifiers.py` — runs every `verifier.sh` against
  its `solution_reference/` and asserts exit 0 (so the bench cannot rot).

Acceptance:
- All 12 verifiers pass on their own reference solution.
- Each task's `task.md` is ≥ 200 words and ≤ 1500 words (long enough to
  decompose meaningfully, short enough that the subagent isn't reading
  for an hour).
- Categorical balance over the 12: ≥ 4 PRODUCT-heavy, ≥ 3 VALIDATION-heavy,
  ≥ 2 INVESTIGATION-heavy, ≥ 2 mixed bug-fix.

### NTB3. Subagent driver
Status: not started.

Build `scripts/run_tb_subagent.py`. The driver does **not** itself code
— it orchestrates the subagent, captures artifacts, and validates the
resulting run dir.

Key responsibilities:
1. Build the worktree skeleton (copy `task.md` + `repo/` seed files;
   exclude `solution_reference/` and `verifier.sh` from the agent's
   working tree).
2. Construct the subagent prompt (see §6).
3. Invoke the subagent (see §6).
4. After subagent returns, run `verifier.sh` against the worktree; capture
   `test_output.txt` and `final_diff.patch`.
5. Validate: `ledger.jsonl` must exist, `check-run` must pass, every event
   must carry an ISO-8601 timestamp with `timestamp_span_seconds >= 60`.
6. Run all four downstream pipelines (`build_ledger_observation_dataset`,
   `build_estimator_checkpoints`, `build_q_labels`,
   `label_observation_shapes`) on the run dir.
7. Write `live_instrumentation.json` with `timestamp_source: "wallclock"`,
   `subagent_model`, `subagent_runtime_seconds`, `verifier_exit_code`.

Outputs:
- `scripts/run_tb_subagent.py`
- `tests/test_run_tb_subagent.py` — uses a stub subagent (deterministic
  ledger writer) to exercise the driver without spending real model calls.

Acceptance:
- `uv run python scripts/run_tb_subagent.py --task tb_live/<id> --runs-dir
  runs/tb_live/` produces a run dir that passes all post-conditions in (5)
  and (6) above.
- Driver fails-fast (non-zero exit) if any post-condition is violated;
  no half-written run dirs remain.

### NTB4. Pilot run on the easiest task
Status: not started.

Pick the one TB-12 task estimated at the lowest subagent runtime and
the cleanest verifier (candidate: `markdown-to-html-cli`). Run the driver
end-to-end with a real subagent. Iterate on the prompt until:
- `ledger.jsonl` has ≥ 5 leaves and ≥ 2 categories.
- `coding_progress` reaches 1.0 iff `verifier.sh` exits 0 (channel-vs-outcome
  decoupling: it's fine if the agent is wrong, but the channel must agree
  with itself).
- Subagent ran for ≥ 5 wall-clock minutes.

Outputs:
- `runs/tb_live/markdown-to-html-cli/` — the pilot run dir.
- `runs/tb_live/NTB4_PILOT_NOTES.md` — what worked, what the prompt had
  to say to elicit honest decomposition.

Acceptance:
- Pilot run dir passes all NTB3 driver post-conditions.
- Notes name at least one ledger pattern (e.g. SPLIT, REOPEN, BLOCKED)
  that the subagent emitted *organically* — not because the prompt asked
  for it specifically.

### NTB5. Run the remaining 11 tasks
Status: not started.

Run the driver across the rest of the TB-12, parallelizable in batches of
~3 (subagent runs are CPU-light but each takes 30 min – 2 hr; 3-way
parallelism keeps a single workstation comfortable).

Outputs:
- `runs/tb_live/<task_id>/` × 12 total.
- `runs/tb_live/NTB5_BATCH_SUMMARY.md` — per-task: subagent runtime,
  verifier pass/fail, leaf count, final coding_progress, shape labels.

Acceptance:
- All 12 run dirs pass the NTB3 post-conditions.
- Median subagent runtime ≥ 20 minutes; max ≤ 4 hours (anything longer
  signals the task isn't tractable for this harness — note as a finding).
- ≥ 4 of 12 tasks failed verification (we want some failures to test the
  channel; if 12/12 pass we picked tasks that were too easy).
- ≥ 1 task ended with verifier-pass but `coding_progress < 1.0` (validates
  the decoupling thesis on a fresh source).

### NTB6. Live-vs-retrospective parity report (the core scientific test)
Status: not started.

For each of the 12 runs, retrospectively annotate the subagent's transcript
(`transcript.md`) without looking at the live ledger. Compare:
- Leaf count delta (live − retro)
- Category-multiset Jaccard
- Shape-label agreement (`stuck_loop`, `high_progress_failure`,
  `validation_exposes_new_work`, etc.)
- Final `coding_progress` delta

This is the same parity exercise N4 did against SWE-agent, now on a
source where the live channel is genuinely first-party.

Outputs:
- `annotations/tb_live_retro/<task_id>.json` × 12 (retrospective ledgers).
- `runs/tb_live/NTB6_PARITY_REPORT.md`.

Acceptance:
- Median Jaccard on category multiset ≥ 0.6.
- Shape-label agreement on `stuck_loop` and `high_progress_failure`
  matches or exceeds N4's frontier-policy bar.
- Disagreements are *named* — not papered over. Each material divergence
  gets a row in the report explaining what the live channel saw that the
  retro annotator missed (or vice versa).

### NTB7. Q1 transfer evaluation
Status: not started.

Re-run `scripts/build_q_labels.py` over `runs/tb_live/`. Report per-target
positive rates and compare against:
- SWE-agent live (`runs/swe_agent_live_wallclock/`)
- Hermes retrospective (`runs/hermes_pilot_h5_v2/`)

Goal: confirm the Q targets are exercised on a third source, identify
which are TB-specific (likely `validation_exposes_new_work` if our task
verifiers are well-designed) and which are universal.

Outputs:
- `datasets/tb_live_q_labels.csv`
- `runs/tb_live/NTB7_Q_TRANSFER.md`

Acceptance:
- ≥ 3 of 5 Q1 targets exercised on the TB-12 batch.
- Cross-source comparison table ships in the report.

### NTB8. Sidecar parity check (the channel equivalence test)
Status: not started.

Run the existing `ledger_progress.sidecar` against a wire-format
re-emission of each TB run (NTB-stretch path). Confirm that the sidecar
produces a `ledger.jsonl` byte-equal to (or replay-equivalent to) the
in-band ledger after normalizing timestamps.

This proves that the sidecar wire format is sufficient to carry the
in-band channel, even when the original source is a Claude Code subagent.

Outputs:
- `scripts/replay_tb_to_sidecar.py`
- `runs/tb_live_sidecar_replay/<task_id>/` × 12
- `runs/tb_live/NTB8_SIDECAR_PARITY.md`

Acceptance:
- Replay-equality on all 12 (after timestamp normalization), or every
  divergence is named and traced to a documented adapter limitation.

### NTB9. Retrospective re-categorization across sources
Status: not started (uses the LedgerSet machinery from Workstream T).

Wrap the 12 TB live ledgers as a 12-member `LedgerSet`. Aggregate
weight-weighted-mean coding progress. Compare against the equivalent
rollups for SWE-agent live (20-member) and Hermes retrospective
(30-member). The point is to demonstrate that the set-level aggregation
holds across heterogeneous sources.

Outputs:
- `runs/tb_live/tb_live_rollup_set.jsonl`
- `runs/tb_live/NTB9_SET_COMPARISON.md`

Acceptance:
- Three set-level scores reported (TB-12, SWE-agent-20, Hermes-30) with
  per-member weight = 1.0.
- Report names which source has the widest distribution and which is
  tightest.

### NTB10. Workstream report + TASKS.md update
Status: not started.

Write the workstream-closing report. Mirror the HP6 report's tone:
TL;DR table, what changed, what to know, caveats, reproducer block.

Outputs:
- `runs/tb_live/NTB_REPORT.md`
- `TASKS.md` updated: NTB1–NTB10 marked `done`, mission-gate paragraph
  updated to reflect first non-synthetic wallclock batch.

Acceptance:
- Report includes the four-pipeline `uv run` command block (HP6-style
  reproducer).
- Mission-delivered checklist ticks the "wallclock-real" sub-bullet
  (currently implied but not visibly tracked).

## 5. The TB-12

Twelve tasks chosen to be (a) tractable for a Claude Code subagent in a
worktree without Docker, (b) substantial enough to take ≥ 20 minutes of
real work, (c) varied across category mix, and (d) likely to surface
distinct ledger patterns. TB IDs cite the closest analog in
`laude-institute/terminal-bench` `tasks/`; we port the *shape* of those
tasks rather than vendor the binary fixtures.

| # | task_id | TB analog | Category mix | Verifier mechanism | Est. runtime | Why this task |
|---|---------|-----------|---|---|---|---|
| 1 | `markdown-to-html-cli` | `build-tex-cli`-shape | PRODUCT-heavy | pytest over a fixture corpus of `.md` + expected `.html` | 25–40 min | Cleanest pilot. Many subtasks: tokenizer, block parser, inline parser, CLI flags, edge-case escaping. Forces SPLIT. |
| 2 | `csv-streaming-dedup` | `process-large-csv` | PRODUCT + VALIDATION | shell verifier: `python tool.py < big.csv | wc -l` against expected count, plus `/usr/bin/time -v` memory ceiling check (must stay < 256 MB on a 500 MB input) | 45–75 min | Memory ceiling forces real engineering, not just correctness. Likely BLOCKED leaf when first attempt OOMs. |
| 3 | `lru-cache-threadsafe` | `concurrent-counters` | PRODUCT + VALIDATION | pytest with `pytest-stress` running 32 producer threads + 4 evictor threads for 30s; assert no lost-update | 30–60 min | Concurrency bugs surface as REOPEN events when the first "complete" is shown wrong by the stress test. |
| 4 | `tar-extract-with-traversal-guard` | `archive-utils` | PRODUCT-heavy + INVESTIGATION | pytest fixtures: 3 valid tars, 4 malicious tars (path traversal, symlink escape, hardlink escape, absolute path), one zip-bomb-ish small tar | 40–70 min | Security thinking forces INVESTIGATION leaves before PRODUCT. |
| 5 | `recover-corrupted-sqlite` | `recover-db` | INVESTIGATION-heavy | shell verifier: tool extracts ≥ N rows from a hand-truncated `.sqlite` file; row checksums must match a manifest | 60–120 min | Forces hypothesis-test loops against opaque binary state — natural BLOCKED + REOPEN territory. |
| 6 | `fix-broken-pyproject-build` | `fix-build-system` | PRODUCT + VALIDATION | shell: `pip install -e . && python -c "import broken_pkg; broken_pkg.entry()"` exits 0 | 25–45 min | Multi-layered bug fix (pyproject syntax + entry point + dependency pin). Each layer is a leaf. |
| 7 | `decouple-state-from-controller` | `refactor-without-regressions` | PRODUCT-heavy + VALIDATION | pytest: all 14 existing tests pass + 5 new tests for the extracted pure-function pipeline | 50–90 min | Refactor-flavored, no functional change. Tests the channel's ability to track progress on work that doesn't change the public API. |
| 8 | `sliding-window-rate-limiter` | `rate-limit-service` | PRODUCT + VALIDATION | pytest: token-bucket with 10 tenants × 100 req/s; deterministic time injection; assert per-tenant fairness within ±5% over 10s window | 35–55 min | Easy to write a wrong-but-plausible solution, so VALIDATION leaves dominate. |
| 9 | `xss-filter-bypass-then-fix` | `break-filter-js-from-html` | INVESTIGATION → PRODUCT → VALIDATION | two-phase verifier: (a) `bypass.txt` must defeat the original filter; (b) patched filter must reject `bypass.txt` AND keep the original 12 benign-input tests green | 60–90 min | The natural decomposition is sequential and visible: investigate → exploit → patch → re-validate. Should produce a clean monotonic ledger. |
| 10 | `graph-tarjan-scc` | `graph-algorithms` | PRODUCT + VALIDATION | pytest: 8 hand-built graphs (DAG, single-cycle, two-SCCs, fully-connected, disconnected, self-loops, single-node, empty) | 30–50 min | Algorithmic; the verifier catches off-by-one and sign-error bugs cleanly. |
| 11 | `directory-watcher-log-rotator` | `log-rotation` | PRODUCT + VALIDATION | shell: spawn tool watching `./logs`; driver writes 50 MB across 1000 files with `dd` and `cp`; assert rotation files exist with expected suffix scheme + total bytes preserved | 40–70 min | Wall-clock-sensitive: tool must respond to filesystem events within bounded latency. Excellent timestamp test for V1's `seconds_since_progress_increase`. |
| 12 | `b-tree-on-disk` | `build-index` | PRODUCT-heavy | pytest: insert 50k random keys, range-scan returns sorted, restart-and-reopen preserves all keys, page size = 4 KB enforced | 90–180 min | The most ambitious task. Forces multiple SPLITs (insert / split / search / range / persist). Likely some IN_PROGRESS leaves at end if the agent runs long. |

### Why exactly 12 and not 15

Three reasons:
1. Twelve gives a 4-way category split (PRODUCT / VALIDATION / INVESTIGATION /
   mixed) at 3 each — a clean denominator.
2. Twelve is the largest batch the NTB6 retrospective re-annotation can
   absorb in one human sitting (Hermes pilot-zero burned ~21 min/trace;
   12 × 25 min ≈ 5 hours of focused annotation, doable in a day).
3. Twelve subagent runs at median 45 min ≈ 9 wall-clock hours of agent
   time; with 3-way parallelism that compresses to ~3 hours, which is a
   single-session NTB5 batch.

If NTB5 finishes faster than expected and the channel looks robust, three
stretch tasks are pre-staged in §5.1.

### 5.1 Stretch tasks (NTB-stretch only, not in the TB-12)

| stretch_id | one-line | why stretch |
|---|---|---|
| `compile-cython-extension` | resolve dep conflicts + build a tiny C extension via `setup.py` | needs a working compiler toolchain in the worktree; NTB1 spec must call out platform deps |
| `tcp-echo-server-graceful-shutdown` | TCP server with SIGTERM-drains-connections semantics | needs port allocation discipline; nice but adds infra burden |
| `json-schema-validator-subset` | implement `type`/`required`/`enum`/`pattern` from JSON Schema draft-07 | great validation density but overlaps thematically with #10 |

## 6. Subagent prompt template

```
You are working on a Terminal-Bench-style coding task in an isolated git
worktree. Your only success criterion is: when you finish, the script
./verifier.sh (which you cannot read or modify) must exit 0.

Read task.md for the spec. Read README.md if it exists. Edit code under
./repo. Run pytest, scripts, or shell commands as needed.

Track your progress with the ledger. Import ledger_progress and use a
LedgerSession (see docs/AGENT_USAGE.md, included below). Add subtasks as
you discover them — not as a plan up front. Mark complete only with
concrete evidence (test output, diff, command output). Use:

- start / complete for normal flow
- block when waiting on an external condition
- split when one vague subtask becomes several checkable ones
- reopen when something you completed turns out wrong
- invalidate when an approach should remain in history but stop counting

Categorize each leaf as PRODUCT (code that ships), VALIDATION (tests,
asserts, manual checks), or INVESTIGATION (reading, searching, tracing).
Most tasks have all three.

When done, export the ledger:

    session.export_jsonl("../ledger.jsonl")
    session.export_curve_csv("../progress.csv")

Then exit. The driver will run the verifier and the downstream pipelines.

<<INSERTED: docs/AGENT_USAGE.md verbatim>>
<<INSERTED: task.md verbatim>>
```

The prompt is deliberately *not* prescriptive about decomposition. NTB4
will iterate on it; the goal is to elicit honest channel emission, not
to script the agent.

## 7. Acceptance gates (rolled up)

| Gate | Bound |
|---|---|
| All 12 verifiers pass on their own reference solution | strict |
| All 12 subagent runs produce a ledger.jsonl that passes `check-run` | strict |
| Every event carries an ISO-8601 timestamp; per-run `timestamp_span_seconds >= 600` | strict |
| Median subagent runtime ≥ 20 min, max ≤ 4 hr | soft (notes findings if violated) |
| ≥ 4 of 12 tasks fail verification | soft (channel-vs-outcome) |
| ≥ 1 task: verifier passes but `coding_progress < 1.0` | soft (decoupling) |
| NTB6 median category-multiset Jaccard ≥ 0.6 | strict |
| ≥ 3 of 5 Q1 targets exercised | soft |
| Sidecar replay-equivalent on all 12 | strict (or each divergence named) |

## 8. Risk register

1. **Subagent ignores the ledger protocol.** Mitigation: NTB4 pilot
   iteration; driver fails fast if `ledger.jsonl` is empty or malformed;
   include AGENT_USAGE.md verbatim in the prompt.
2. **Subagent declares "done" without running the verifier.** That's
   acceptable — the driver runs the verifier itself. The interesting
   case is the *gap* between agent self-assessment and verifier outcome,
   which is exactly what `high_progress_failure` measures.
3. **Worktree state pollution between runs.** Mitigation: Agent's
   `isolation: "worktree"` already handles this; driver tears down the
   worktree on success and preserves it on failure for inspection.
4. **TB-12 tasks turn out trivial.** Mitigation: NTB5 acceptance requires
   ≥ 4 failures; if we get 12/12 pass on the first batch, replace the
   easiest tasks with stretch tasks from §5.1.
5. **Wall-clock runtime explodes.** Cap each subagent run at 4 hours via
   the driver; tasks that hit the cap are recorded as IN_PROGRESS and
   feed into the channel as legitimate "agent didn't finish" data.
6. **Cross-source comparison contaminated by harness bias.** Acknowledged.
   The TB-12 batch is a *different* harness, *different* task source,
   *different* clock — distributions won't match Hermes or SWE-agent.
   The point is not distributional parity but *channel survival*.

## 9. Out of scope (explicit)

- Docker-based TB tasks. Re-add later if a task category requires it.
- Training, RL, or any controller use of the ledger. Pure observation.
- Comparing Claude vs other models. We use one subagent model end-to-end.
- TB leaderboard reproduction. We're testing the *ledger*, not the agent.
- LedgerSet aggregation rule changes. T1's weight-weighted mean stands.

## 10. Reproducer (target shape)

When NTB10 closes, the reproducer block at the bottom of `NTB_REPORT.md`
should look like:

```bash
# 1. Verify the bench: every TB-12 task's verifier passes on its reference solution.
uv run pytest tests/test_tb_live_verifiers.py

# 2. Run all 12 subagent jobs (parallelism = 3 by default).
for task in tasks/tb_live/*/; do
  uv run python scripts/run_tb_subagent.py \
    --task "$task" \
    --runs-dir runs/tb_live/ \
    --max-parallel 3
done

# 3. Four parity pipelines (unchanged from the rest of the project).
uv run python scripts/build_ledger_observation_dataset.py --runs-dir runs/tb_live/ ...
uv run python scripts/label_observation_shapes.py runs/tb_live/ ...
uv run python scripts/build_estimator_checkpoints.py --runs-dir runs/tb_live/ ...
uv run python scripts/build_q_labels.py --runs-dir runs/tb_live/ ...

# 4. Live-vs-retro parity report.
uv run python scripts/build_tb_live_parity_report.py \
  --live-runs runs/tb_live/ \
  --retro-annotations annotations/tb_live_retro/ \
  --out runs/tb_live/NTB6_PARITY_REPORT.md

# 5. Set-level rollup across sources.
uv run python scripts/build_tb_live_rollup_set.py
```

## 11. File index (to be created)

```
docs/
  TB_LIVE_TASK_FORMAT.md           # NTB1
  WORKSTREAM_N_TB_PLAN.md          # this file
tasks/tb_live/
  _template/                        # NTB1
  markdown-to-html-cli/             # NTB2  (pilot — NTB4)
  csv-streaming-dedup/              # NTB2
  lru-cache-threadsafe/             # NTB2
  tar-extract-with-traversal-guard/ # NTB2
  recover-corrupted-sqlite/         # NTB2
  fix-broken-pyproject-build/       # NTB2
  decouple-state-from-controller/   # NTB2
  sliding-window-rate-limiter/      # NTB2
  xss-filter-bypass-then-fix/       # NTB2
  graph-tarjan-scc/                 # NTB2
  directory-watcher-log-rotator/    # NTB2
  b-tree-on-disk/                   # NTB2
scripts/
  run_tb_subagent.py                # NTB3
  replay_tb_to_sidecar.py           # NTB8 (stretch)
  build_tb_live_parity_report.py    # NTB6
  build_tb_live_rollup_set.py       # NTB9
ledger_progress/adapters/
  claude_code_agent.py              # NTB-stretch (3.2)
runs/tb_live/
  <task_id>/                        # NTB4–NTB5
  NTB4_PILOT_NOTES.md
  NTB5_BATCH_SUMMARY.md
  NTB6_PARITY_REPORT.md
  NTB7_Q_TRANSFER.md
  NTB8_SIDECAR_PARITY.md
  NTB9_SET_COMPARISON.md
  NTB_REPORT.md
annotations/tb_live_retro/
  <task_id>.json                    # NTB6
tests/
  test_tb_live_verifiers.py         # NTB2
  test_run_tb_subagent.py           # NTB3
```
