# Inter-annotator agreement report (H2)

This satisfies `TASKS.md` § Workstream H, task **H2**. It analyses
the diff between two independent annotation passes over 5 SWE-agent
pilots and identifies where the ledger is **stable** (annotators
agree on shape) vs where it is **subjective** (judgment-call gaps
the protocol does not resolve).

## 1. Setup

- **v1 (original annotator):** Claude (E1 pass), specs at
  `annotations/swe_agent_pilot/`. The 20-pilot dataset.
- **v2 (cold pass, second annotator):** Claude Opus subagent
  (Workstream H), specs at `annotations/swe_agent_pilot_v2/`. Read
  protocol + addendum + per-pilot trajectory only; explicitly NOT
  shown v1's annotations or `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`.
- **Selection:** 5 of 20 pilots covering all four success/progress
  quadrants and the load-bearing trace shapes:
  - `s_01` (clean success), `s_03` (success with reopen),
  - `f_01` (failure, submit-without-validation),
  - `f_06` (high-progress failure, hidden-work gap),
  - `f_03` (113-step stuck-loop failure).
- **Tool:** `scripts/compare_annotations.py` produced
  `datasets/h_inter_annotator_report.md` (raw metrics) and
  `datasets/h_inter_annotator_summary.json`.

**Caveat on the inter-annotator value:** v2 is another LLM reading
of the same protocol. Biases between v1 and v2 are likely correlated
(same model family). This pass tests *protocol clarity* — would an
independent reader of the docs reach similar shapes? — not the
stronger question of *human-AI annotator agreement*. A human pass on
the same 5 would close that gap.

## 2. Headline numbers

| Metric                           | Value                       |
|----------------------------------|-----------------------------|
| Pairs compared                   | 5 / 5                       |
| Mean coding-progress delta (v2 − v1) | **+0.10** (v2 trends slightly higher) |
| Mean *absolute* coding-progress delta | **0.10**              |
| Mean leaf count delta (v2 − v1)  | **+1.0** (v2 splits finer)  |
| Mean category-vector L1 distance | 1.80                        |
| Verdict distribution             | high: 1, moderate: 2, low: 2 |

Same success/progress quadrant on all 5 pilots: **5 / 5 agreement**.
The only quadrant boundary that came close to flipping was `f_01`
(0.67 vs 1.00); both readings are still in the failure-with-some-progress
region, just at different points. **No pilot crossed a quadrant.**

## 3. Where the ledger is stable

These are claims both annotators agreed on, with the same evidence
and the same shape:

1. **The four canonical progress shapes hold.**
   - `s_01` ends at 1.00 / 1.00 with full pipeline (both).
   - `f_03` ends with INVESTIGATION blocked at the stuck loop, no
     PRODUCT (both).
   - `f_06` ends at 1.00 with the repro hidden-work gap explicitly
     flagged in run_notes (both).
   - `s_03` ends at 1.00 with multi-stage PRODUCT work (both).
2. **Stuck-loop detection.** Both annotators applied general § 6
   to `f_03` and produced a BLOCKED leaf. The exact step
   differs (v1 step 22 vs v2 step 30) — see § 4 for the wording
   gap — but the **fact** of stuck-loop is shared.
3. **No annotator used `final_success` as evidence.** Both pass
   this on all 5 pilots (`whether_final_success_used_only_at_end:
   true` in every quality block).
4. **Hidden-work gaps surface in `run_notes` § 6, not as
   discovered subtasks.** Both annotators flagged `f_06`'s repro
   as suspect without retro-fitting a discovered subtask.
5. **No false reopens.** v1 used REOPEN once (s_03); v2 used
   REOPEN zero times. Neither side used REOPEN where the trace
   didn't surface a contradiction.

## 4. Where the ledger is subjective

These are real, recurring judgment-call gaps. Each maps to a
proposed protocol revision in H3.

### 4.1 Validation-as-implicit-discovered-work (f_01, **largest delta: 0.33**)

**v1 model:** 4 leaves including a `VALIDATION` leaf left at
`not_started` because the agent submitted without running tests.
Final progress 0.67. Rationale: the bug-fix issue type implies
validation; an agent skipping it produces a process anomaly the
ledger should surface.

**v2 model:** 3 leaves with no `VALIDATION` leaf. Final progress
1.00. Rationale: the protocol says "annotate only visible trace
evidence"; validation never surfaced in the trace, so it is
hidden work, recorded in run_notes § 6 as a gap.

**Both readings are defensible under the current protocol.** The
ambiguity is the protocol's, not the annotators'. Importantly,
this is the same gap I flagged in `feedback_validation_implicit_for_bug_fix.md`
during E1 self-review; it is now confirmed empirically by an
independent annotator.

This is "different ledger, **partly different conclusions**" — both
agree the agent skipped validation, but disagree on whether to
encode that as a discovered-but-not-started leaf or as a hidden-work
note. The downstream observation channel records 0.67 vs 1.00.

### 4.2 Granularity of leaves (every pilot, +1 leaf delta on average)

**Pattern:** v2 produces more leaves than v1 across the board
(`s_01`: 7 vs 6; `s_03`: 8 vs 6; `f_06`: 7 vs 5; `f_03`: 3 vs 2;
`f_01`: 3 vs 4 — the inverse, driven by the f_01 question above).
The granularity differences are roughly:

- v1 collapses "build repro + observe output" into one
  INVESTIGATION leaf; v2 splits it into INVESTIGATION (build) +
  VALIDATION (observe).
- v1 collapses multi-stage edits across files into one PRODUCT
  leaf with REOPEN; v2 splits each stage as its own complete leaf.

**Effect on outcomes:** identical final progress on `s_01`, `s_03`,
`f_06`. Modest delta on `f_03` (0.50 → 0.67) because v2's extra
VALIDATION leaf increases the "complete" numerator.

This is "different ledger, **same conclusions**" — both annotators
finished at the same quadrant; v2 just had a finer mesh. The
protocol does not specify granularity; both readings are
legitimate.

### 4.3 ENVIRONMENT category boundary (s_03, f_06)

**v2 used ENVIRONMENT** for agent-internal setup work:
- `s_03`: making `SimpleHeuristicsPlayer` importable from
  `poke_env.player.__init__` (v2 calls this "blocking product
  work" → ENVIRONMENT).
- `f_06`: agent-internal scaffolding work the trace surfaces
  before the product edit.

**v1 used PRODUCT or INVESTIGATION** for the same activity, on the
basis that updating `__init__.py` exports is part of the fix
(behavioral change to reachable code).

**Both readings are inside the protocol's wording.** § 5
ENVIRONMENT says "setup work that blocks product work without being
part of it" — this leaves room for two interpretations of agent-
internal package wiring.

This is "different ledger, **same conclusions**" — both readings
yield the same final progress on the affected pilots.

### 4.4 Stuck-loop "third iteration begins" wording (f_03)

v1: step 22. v2: step 30. Both follow general § 6, but disagree on
whether to count assistant turns or tool responses, and whether the
third iteration "begins" at the start of the third repetition or
at its conclusion.

**Both readings produce the BLOCKED status on the right leaf.**
Step number is the only thing that differs. Downstream metrics
that depend on the exact step (e.g. progress curve point counts)
inherit a few-step difference; metrics that depend on "is this
leaf blocked" are unaffected.

This is "different ledger, **same conclusions**" — but the wording
gap is real and wastes annotator time. H3's revision tightens it.

### 4.5 REOPEN vs separate leaves (s_03)

v1 used a single PRODUCT leaf for the `__init__.py` work with a
REOPEN at step 22 when the first edit was insufficient. v2 modeled
the same trace as two separate complete leaves (each successful
edit is its own leaf).

**Both produce final progress 1.00.** The shape of the progress
curve differs (v1 has a visible dip; v2 has a smoother climb), but
the framework's claim — non-monotonicity is allowed and preserved
when present — holds in v1 and is silent in v2. Neither is wrong;
the protocol allows both readings.

This is "different ledger, **same conclusions**" — but the
distinction matters for downstream consumers that look at REOPEN
counts as a process signal.

## 5. Verdict per pilot

| pilot | verdict | "different ledger, same conclusions"? | comment |
|-------|---------|----------------------------------------|---------|
| `s_01` | high | yes | tiny granularity delta (1 PRODUCT leaf) |
| `f_06` | moderate | yes | v2 splits scaffolding into ENVIRONMENT + extra VAL |
| `f_03` | moderate | mostly | granularity + stuck-loop wording gap |
| `s_03` | low | yes | granularity + ENVIRONMENT vs PRODUCT + REOPEN-vs-split |
| `f_01` | low | **NO** — different conclusions | implicit-validation gap; final progress 0.67 vs 1.00 |

**Bottom line:** 4 of 5 pilots show "different ledger, same
conclusions" — the framework's discriminating shape claims hold.
**1 of 5 (f_01)** shows different conclusions, and that one is
exactly the methodology gap I'd flagged for H during E1
self-review. The protocol needs to pick a side on
implicit-validation; until it does, two reasonable annotators will
diverge here.

## 6. Implications

1. **The framework's load-bearing claims survive the second
   annotator.** All five pilots stay in the same success/progress
   quadrant. The high-progress failure (`f_06`) and the
   non-monotonic success (`s_03`) are recognized as such by both.
   Inter-annotator agreement at the *quadrant* level is 5/5.
2. **Per-pilot progress is annotator-dependent at the ~0.1
   resolution.** Mean absolute progress delta is 0.10. If any
   downstream analysis treats per-pilot progress as a
   high-precision number, that's overstating annotator
   reproducibility.
3. **The single biggest reproducibility win** is to nail down the
   implicit-validation rule. Doing so eliminates the only
   "different conclusions" disagreement of the five.
4. **Per-pilot leaf count and category vector are NOT
   reproducible at the unit level.** v2 systematically splits
   finer. Any analysis that depends on exact leaf counts (e.g.
   "average pilots have N leaves") should report a per-annotator
   range rather than a single point.

## 7. What this DOES NOT measure

- Real human-AI inter-annotator gap. Both v1 and v2 are LLM
  passes; their biases are likely correlated. A human re-pass on
  the same 5 would meaningfully expand the agreement signal.
- Workflow with a fresh human reading. The cold-pass instructions
  to v2 explicitly forbid reading v1's notes; a real human
  annotator might consult them as a starting point and produce
  yet another shape.
- Agreement under protocol revisions. H3's proposed revisions are
  expected to *increase* agreement (especially on f_01 and f_03
  step). Until applied and re-tested, the change is unmeasured.
