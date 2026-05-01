## M1. Go / no-go memo — SWE-agent retrospective pilot N=20

This memo satisfies `TASKS.md` § Workstream M, task **M1**. It is a
synthesis over the artifacts produced by Workstreams A–H, not a new
analysis. Every claim cites the file that produced it.

---

### 1. Did the ledger schema represent real SWE-agent traces?

**Yes — with three additive protocol refinements; zero core-schema
changes.**

- 20 / 20 pilots have a replayable `ledger.jsonl` that passes
  `ledger-run check-run` (E2; `runs/swe_agent_pilot/swe_agent_pilot_*/`).
- 18 / 20 pilots fit the post-refinement protocol cleanly. The 2
  flagged pilots (`f_02` 509 steps, `f_07` 183 steps) drove three real
  protocol refinements, all additive:
  1. § 6 stuck-loop generalized to **any** cycle length (cycle-of-1,
     cycle-of-2 oscillation). — `f_07`.
  2. § 6(b) tool-response-loop variant: identical responses across
     varied queries. — `f_02`.
  3. Addendum pitfalls #6 (harness-forced termination ≠ agent submit)
     and #7 (`final_diff.patch` is a state diff, not an action diff).
- **No edits** to `ledger_progress/core.py` were needed. No new
  `Status`, no new `EventType`, no new `SubtaskCategory`.

Source: `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 4.

### 2. Was retrospective annotation feasible?

**Yes. ~21 min median per pilot.**

- Total annotation time: **492 min (~8.2 h)** across 20 pilots.
- Median per pilot: **21 min** (range 12–55, mean 24.6).
- The 55-min outlier (`f_07`, 183 steps) drove protocol refinement #1
  above; without it the median would be lower.
- All 20 pilots produced `ledger.jsonl`, `progress.csv`,
  `progress_by_category.csv`, `summary_by_category.json`,
  `annotation_quality.json`, and an extended `run_notes.md`.

Source: `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 1, § 2.

### 3. Were evidence gaps tolerable?

**Yes — and they are doing useful work.**

- 19 evidence-gap citations across 20 pilots fall into three repeating
  patterns: submit-without-validation, hidden-work gaps the agent
  surfaced but did not act on, and mid-edit harness-forced termination.
- Each pattern produces a distinguishable ledger shape (0.67 cluster,
  1.00-with-flagged-hidden-work, 0.50–0.71 failure tail). The framework
  is **detecting** these gaps, not being defeated by them.
- `whether_final_success_used_only_at_end = true` for all 20 pilots and
  for both annotators on the H subset (5/5). The upstream label was
  never used as ledger evidence during a walk.

Source: `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 5;
`runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md` § 3.

**Caveat (open):** Workstream K (evidence-strength audit) is *not
started*. We have a count of evidence gaps, not a quantitative
strong-vs-weak classification on the SWE-agent corpus.

### 4. Did failed traces provide useful diversity?

**Yes — dramatically. This is the load-bearing finding.**

- SWE-agent populates **all 4** of the success × progress quadrants;
  the toy/live corpus populates **2 of 4**.
- Failures span coding-progress 0.50–1.00 and are produced by *distinct
  mechanisms* (stuck loops, no in-trace validation, hidden-work gaps,
  harness force-quit), each leaving a different ledger shape.
- BLOCKED status is exercised only by real traces (7 leaves across 6
  SWE-agent pilots; 0 in toy/live).
- INVESTIGATION drop-source is exercised only by real traces (14 step
  rows in SWE-agent; 0 in toy/live).

Source: `datasets/observation_distribution_comparison.md` § 1, § 2,
§ 3.1–3.3, § 5.

### 5. Did high-progress failures appear?

**Yes — and one (`f_06`) is the canonical shape the protocol
predicted.**

- `f_06` (`googleapis__python-spanner-317`): coding-progress **1.00**,
  upstream `final_success=False`. The agent's `reproduce.py` returned
  *"Script completed successfully, no errors"* — i.e. the repro never
  triggered the bug — and the agent moved on. All discovered work
  reached `complete` with in-trace evidence; failure sits entirely in
  undiscovered hidden work.
- `f_09` is also flagged failure-high-progress by the audit
  (coding-progress 0.83 with VALIDATION reopen).
- Both annotators in H independently identified `f_06` as the
  hidden-work-gap shape, in `run_notes` § 6, without retro-fitting a
  discovered subtask.

Source: `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 3;
`datasets/swe_agent_pilot_observations_step_audit.md` (Quadrants);
`runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md` § 3.

### 6. Did native category annotation work?

**Annotators: yes. Builder reporting: only partly.**

- Every D1 / D4 / E1 spec assigns a category explicitly on every
  `add_subtask` and `split_subtask` event. No annotator relied on the
  default.
- The dataset builder's `category_resolution_mode` field, however,
  reports **mixed: 181 rows / native: 10 rows** on the SWE-agent step
  table. One run (`s_03`) carries a "large native/resolved divergence"
  warning.
- Workstreams **J1 / J2 (native-category audit + enforcement) are
  *not started*.** Until they are, the divergence between
  annotation-level "native" and builder-level "mixed" is not
  fully diagnosed.

Source: `datasets/swe_agent_pilot_observations_step_audit.md`
(Category Resolution); `TASKS.md` § Workstream J.

### 7. Did smoke prediction plumbing still work?

**Yes — and it correctly behaves at chance.**

- Three model variants (`progress_only`, `ledger_basic`,
  `elapsed_only`) ran leave-one-run-out on the 20-pilot step table
  with no leakage and no future-event use. All exclusions documented.
- AUROC: 0.37 / 0.38 / 0.18. Near chance is the **predicted**
  behaviour: progress is decoupled from outcome by design (per
  `feedback_progress_vs_outcome_decoupling.md`); a fair predictor on
  progress alone should not work.
- Pipeline surfaced a real builder bug: `resolve_final_success` was
  inferring labels from `test_output.txt` text and misclassifying 3
  upstream successes. **Fixed** in commit 7df39ba; the 10/10 split is
  now correct.

Source: `datasets/swe_agent_pilot_completion_smoke_report.md` § G2.1,
§ G2.4, § G2.5; `datasets/observation_distribution_comparison.md`
§ 3.6.

### 8. Should we scale to 100 traces?

**Conditionally yes. One gate first.**

The case for scaling is strong: schema survives, annotation is
feasible at 21 min/pilot (100 traces ≈ 35–45 h), all four quadrants
populated, observation pipeline clean (zero integrity failures on the
N=20 step table), smoke plumbing works.

The blocker is reproducibility. H2 reports inter-annotator
**quadrant agreement 5/5** but **conclusion agreement 4/5** — the
`f_01` disagreement (0.67 vs 1.00) is the single methodology gap. H3
proposed three protocol revisions and added one acknowledgement; the
HIGH-severity revision (bug-fix tasks always carry implicit validation
work) directly targets the `f_01` gap, but **has not been re-tested by
re-annotation under the revised protocol.**

Scaling to 100 without that re-test risks 100 ledgers carrying the
same unresolved implicit-validation ambiguity, multiplying the
problem H surfaced.

Source: `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md` § 4.1, § 5;
`docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md`.

### 9. Should we instead instrument live SWE-agent runs?

**Not yet — but it is the right second move.**

Live instrumentation (Workstream N) would close two gaps that
retrospective annotation provably **cannot** close:

1. The `f_06`-style hidden-work-gap is detectable retrospectively only
   because we *know* the upstream label is `False`. A live trace with
   an instrumented repro-result observation would surface the gap as
   it happened.
2. The harness-forced-termination cluster (six pilots) loses agent
   intent: was the last edit endorsed? A live `submit` event removes
   the ambiguity.

But N's engineering surface (sidecar vs in-agent, parity
report, integration with one SWE-agent invocation) is real, and the
retrospective channel still has unconsumed signal — the K1 evidence
audit, the J1/J2 native-category work, the full-corpus shape
validation under the revised protocol. Burning the live-instrumentation
budget before the retrospective channel is fully used would be
premature.

Source: `TASKS.md` § Workstream N (sketch); § Workstream K (sketch).

---

### 10. Recommendation

**Scale the retrospective study to 100 traces, gated on a 5-pilot
re-annotation under the H3-revised protocol that closes the `f_01`
conclusion gap. Defer live instrumentation (N) to the next
go/no-go.**

Concretely, the gate is one short workstream before scaling:

```
H4 (gate, ~3 hours):
  re-annotate the 5 H pilots (s_01, s_03, f_01, f_06, f_03) under
  the H3-revised protocol (Revision 1 HIGH applied)
  pass: f_01 produces the same conclusion in both v3 readings
  fail: re-open H3, do not scale
```

If the gate passes, scale to 100 with the revised protocol. If it
fails, the right next step is *protocol revision*, not *retrospective
scale*, not *live instrumentation*.

Reason this is the right single recommendation and not the
alternatives:

- **Not "live instrumentation" first:** live closes real evidence gaps
  but costs weeks of engineering (sidecar / agent patch, parity
  report). Retrospective is 35–45 h of annotation for a 5× corpus.
  Use the cheap signal first; spend the live budget on the next
  decision.
- **Not "revise schema":** none of the pilot evidence calls for a
  core-schema change. Three additive protocol refinements landed
  during the pilot; no fourth is asked for.
- **Not "pause":** the framework's load-bearing claim — progress is
  decoupled from outcome — has empirical support from N=20 and
  inter-annotator agreement at the quadrant level. Pausing now would
  be under-using completed work.

### 11. Cost of being wrong about the recommendation

| If the recommendation turns out wrong because… | What that costs |
|---|---|
| The H4 gate passes but the corpus at N=100 still shows reproducibility issues we missed at N=5 | ~35–45 h of annotation needs partial redo. Up to ~70 % of each ledger is salvageable as raw event records; redo is targeted at the implicit-validation leaf. Estimated rework: **8–12 h.** |
| Live instrumentation would have surfaced a hidden-work gap that retrospective cannot, and a paper / decision gets blocked on it | Delay of one go/no-go cycle (~2 weeks). N can be picked up immediately after retro-100; no work is invalidated. |
| The H4 gate fails and we should have paused and revised the protocol harder before any further annotation | ~3 h of H4 annotation, plus deferred clarity. Cheap. The gate exists *because* this is the cheapest place to discover this. |
| We scale to 100 *without* the H4 gate (the recommendation, ignored) | All 100 ledgers carry the unresolved implicit-validation ambiguity. Estimated full re-annotation cost if H3 revision changes a leaf: **20–30 h.** This is the cost we are explicitly avoiding. |

The H4 gate is what makes the recommendation cheap to be wrong about:
**~3 hours buys an empirical answer to the only methodology gap H
surfaced.** Without the gate, the cost of being wrong scales with the
N=100 annotation budget, not with N=5.

---

### 12. Pointers

- Pilot summary: `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md`
- Distribution comparison: `datasets/observation_distribution_comparison.md`
- Smoke report: `datasets/swe_agent_pilot_completion_smoke_report.md`
- Inter-annotator report: `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`
- Protocol revisions: `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md`
- Step-table audit: `datasets/swe_agent_pilot_observations_step_audit.md`
- Open workstreams referenced: J (native categories), K (evidence audit), N (live instrumentation), Q (predictive modeling).
