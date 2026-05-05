# Reviewer briefing — coding-estimator v0 (revision 2)

_Last updated 2026-05-05 (commit 548c42a). Standalone document for an
independent reviewer who has not seen this project before. Read top to
bottom; the TL;DR is load-bearing._

> **Revision note.** The previous version of this briefing (commit
> 96d6558) reported the v0 verdict as **FAIL**, called G5 "the
> cheapest experiment" without numbers, and listed Hermes-vs-upstream
> as "unknown — diagnose first." All three of those have since been
> resolved. The "What changed since revision 1" section below is the
> diff; the rest of the briefing is rewritten to reflect the current
> state.

## TL;DR (60 seconds)

The pipeline is in a measurement-honest state. The v0 verdict is
**INDETERMINATE**, blocked solely by data gaps:

- `tb_live` is 12 successes, 0 failures (P1.b, P1.d cannot be tested).
- `hermes_pilot_h5_v2` is upstream-unannotated, all 30 runs (P1.c
  cannot be tested as the plan defines it).

After fixing those two data defects, every other gate condition
**already passes** — including the structured D5 behavioral leakage
audit (P1.g) and the submit-without-validation polarity check (P1.h).

The recentered scientific finding is now the headline:

> **Prefix-only ledger features predict near-future progress dynamics.
> They do not yet improve terminal success prediction over elapsed
> time at this N.**

Quantitatively, on `swe_agent_pilot` (20 runs, 499 labeled
checkpoints, LORO):

|                            | Brier | AUROC | Δ vs G2 (Brier) |
|----------------------------|------:|------:|----------------:|
| `y_future_progress_drop_h5` G2  | 0.142 | 0.626 |       —     |
| `y_future_progress_drop_h5` G4  | 0.039 | 0.977 |  **−0.102** |
| `y_future_progress_drop_h5` G5  | 0.078 | 0.897 |    −0.064   |
| `y_success_eventual` G2         | 0.283 | 0.281 |       —     |
| `y_success_eventual` G4         | 0.291 | 0.410 |    **+0.009** |
| `y_success_eventual` G5         | 0.272 | 0.385 |    −0.010   |

Read this as: the observation channel measures **work-frontier
dynamics** very well, and **completion outcomes** not at all. That is
a publishable boundary; the project is doing what a good measurement
system should.

`not_safe_for_control = true` is enforced and recorded on the model
card. The headline output is `reports/V0_FINDINGS.md` (publishable
narrative) plus `reports/NOT_READY_FOR_SCHEDULING.md` (prioritized
unblocks).

The reviewer's job, in priority order: (1) decide whether the v0
finding (dynamics yes, success no) is worth publishing as-is or
whether it should be held for a larger-N replication; (2) commission
the upstream Hermes annotation pass that unblocks P1.c; (3) commission
the `tb_live_v2` collection that unblocks P1.b and P1.d.

---

## What's changed since revision 1

Concrete delta against the previous reviewer briefing.

### Verdict

| | revision 1 | revision 2 |
|---|---|---|
| Overall | FAIL | INDETERMINATE |
| P1.c | FAIL (over-stated on swe-alone) | INDETERMINATE (honest — hermes labels missing) |
| P1.g | INDETERMINATE (no D5 artifact) | PASS (D5 audit shipped, clean) |
| P1.h | required: no | required: yes (now blocks if every winner is on `y_submit_without_validation`) |
| `not_safe_for_control` | true | true (unchanged) |

### Data-gap diagnoses

- **Hermes labels.** Revision 1 listed three candidate causes
  ("unannotated upstream / missing field / local wiring"). Revision 2
  has the diagnosis: **all 30 runs have
  `source_metadata.final_success: null` and
  `annotation_mode: not_annotated`** — confirmed upstream gap.
  See `reports/HERMES_LABEL_DIAGNOSIS.md` for the cross-check and
  three remediation paths.

### New work shipped this round

| What | Where |
|---|---|
| G5 ledger-dynamics features | `coding_estimator/checkpoints/dynamics.py` |
| G5 baseline spec | `coding_estimator/baselines/ledger_dynamics.py` |
| G5 evaluation driver | `scripts/run_g5_eval.py` → `reports/g5/` |
| Structured D5 audit | `coding_estimator/leakage/d5_audit.py` |
| D5 driver | `scripts/run_d5_audit.py` → `reports/d5_audit.{md,json}` |
| Human-baseline scaffolding | `coding_estimator/eval/human_baseline.py`, `scripts/run_human_baseline.py` |
| TB-12 midpoint prompts | `reports/human_baseline/prompts/` (6 files) |
| V0 publishable memo | `reports/V0_FINDINGS.md` |
| Hermes diagnosis | `reports/HERMES_LABEL_DIAGNOSIS.md` |
| Recentered framing | `TASKS.md` § 0.0β + sign-off + gate report headers |

### What was *answered* since revision 1

- **"Is G5 the cheapest experiment to do next?"** — Built. G5 helps
  on the dynamics targets (Δ vs G2 = −0.064 on `y_future_progress_drop_h5`)
  but does **not** rescue success prediction (Δ vs G2 = −0.010 on
  `y_success_eventual`, well within noise). Combining G4+G5 sits at
  G4. Conclusion: dynamics feature additions alone do not unlock
  the success target at this N. The next move is data, not features.

- **"Is the Hermes gap upstream or local?"** — Upstream (confirmed).
  Local label-build code correctly raises `UnresolvableLabelError`;
  no wiring fix in this repo will change anything.

- **"Is P1.g actually testable?"** — Yes, now. The D5 audit
  artifact has a real schema (`schema_version: 1.1.0`, structured
  `findings`, methodology callout). P1.g rejects bare
  `{clean: true}` and ships clean on the v0 dataset.

### What's still open

- **TB-12 cohort outcome diversity.** Still 12/12 successes. P1.b
  and P1.d remain blocked. No engineering will fix this — needs
  fresh agent runs.
- **Hermes annotation.** Still upstream-pending. P1.c remains
  blocked.
- **Whether to publish v0 findings as-is or wait.** Reviewer's call.
- **Whether to flip `canonical_for_v0=False` for hermes** (path 2 in
  `reports/HERMES_LABEL_DIAGNOSIS.md`) until annotation lands.
- **Tests pass count.** 622 → **644** (22 new for G5/D5/human-baseline).

---

## What this estimator is supposed to do

A coding agent run produces a `ledger.jsonl` — a stream of events
recording what the agent has tried, completed, blocked on, validated,
reopened, etc. From the prefix of that stream up to time `t`, the
estimator outputs calibrated probabilities for:

- `y_success_eventual` — does this run finish successfully?
- `y_future_progress_drop_h5` — will progress regress in (t, t+5]?
- `y_validation_new_work_h5` — will validation expose new work in (t, t+5]?
- `y_submit_without_validation` — does the run terminate without
  validation? (run-constant — non-trivial AUROC at non-terminal t is
  a *data property*, not skill.)

It is explicitly **not** a scheduler, a controller, or an action
selector. It is a *belief layer*. The downstream consumer is deferred
to a future repo and is gated on this one passing P1.

The recentered v0 framing makes the first two targets the **primary
headline** and demotes the success target to a **secondary / negative
result**. See `reports/V0_FINDINGS.md` for the full argument.

The plan that defines all of this: `TASKS.md` (now 2120 lines after
the § 0.0β banner). Workstream labels A–R correspond to project
phases; A–P are in scope, Q/R are deferred.

---

## Architecture

### Code layout

```
coding-estimator/
├── coding_estimator/
│   ├── checkpoints/
│   │   ├── features/registry.py     # Feature columns, groups, missingness semantics
│   │   ├── build.py                 # Local re-implementation of the upstream builder
│   │   ├── replay.py                # Prefix-replay over ledger.jsonl
│   │   ├── fills.py                 # Per-source canonical fill application
│   │   └── dynamics.py              # NEW: G5 features as post-processing layer
│   ├── labels/
│   │   ├── registry.py              # V0_TARGETS — declared targets and their semantics
│   │   ├── terminal.py              # y_success_eventual, y_finish_step, ...
│   │   ├── dynamics.py              # y_future_progress_drop_h5, etc.
│   │   ├── shapes.py                # Run-shape labels (slicing, NOT predictions)
│   │   └── balance.py               # Label class-balance audit
│   ├── splits/
│   │   ├── builder.py               # Build per-source split JSONs from manifests
│   │   └── protocol.py              # Fold/Split, loro(), ltfo(), holdout()
│   ├── baselines/
│   │   ├── constant.py              # G1 — constant base-rate
│   │   ├── time_only.py             # G2 — elapsed_steps (+ wall-clock when tb_live)
│   │   ├── ledger_basic.py          # G4 — closure/frontier/instability/discovery
│   │   └── ledger_dynamics.py       # NEW: G5 — diagnostic dynamics group
│   ├── models/
│   │   ├── empirical_bin.py         # I0 — calibration-only model
│   │   ├── logreg.py                # I1 — sklearn LogisticRegression on G4 features
│   │   ├── readiness.py             # Training-readiness preflight
│   │   └── cards.py                 # Model card schema validator + builder + writer (N1/N2)
│   ├── eval/
│   │   ├── harness.py               # evaluate_cell, predict_cell — per-cell CV evaluation
│   │   ├── bootstrap.py             # Run-level Brier bootstrap CIs
│   │   ├── metrics.py               # AUROC (with tie-rank), Brier, ECE, log-loss
│   │   ├── slices.py                # Phase / shape slicing
│   │   ├── tb_live.py               # K1 — tb_live-only LORO eval
│   │   ├── tb_qualitative.py        # K3 — TB-12 rollup
│   │   ├── transfer.py              # L3 — retro→live with feature-group ablation
│   │   ├── failure_modes.py         # O1, O5, O7
│   │   ├── go_no_go.py              # P1.a–h + verdict aggregator
│   │   ├── sign_off.py              # P2 sign-off package + P3 ready/not-ready routing
│   │   └── human_baseline.py        # NEW: midpoint-prefix prompts + comparison
│   ├── calibration/
│   │   ├── metrics.py               # J1 — brier, ECE, reliability_table
│   │   ├── recalibrate.py           # J4 — Platt / isotonic / source-isotonic
│   │   └── report.py                # J2/J3/J5 — markdown reliability + slice + headline
│   ├── leakage/
│   │   ├── guard.py                 # Forbidden-column audit (exact / prefix / suffix)
│   │   ├── run_constancy.py         # Run-constant feature × target pair audit
│   │   ├── audit.py                 # Pre-existing CHECKPOINT_CONSTRUCTION_AUDIT renderer
│   │   └── d5_audit.py              # NEW: structured JSON D5 producer for P1.g
│   ├── ingest/                      # Source registry + path resolution
│   ├── reports/                     # Jinja eval-report renderer
│   └── profile/                     # Data profiling (D5, F, etc.)
├── datasets/                        # Parquet checkpoint and label tables
├── models/                          # Saved estimator bundles (gitignored)
├── reports/                         # Generated artifacts (see "Reading order")
├── schemas/                         # JSON schemas for validation
├── docs/                            # MODEL_CARD_TEMPLATE, VERSIONING, ESTIMATOR_*
├── scripts/                         # Driver scripts (one per workstream phase)
└── tests/                           # 644 pytest tests
```

### The data flow (unchanged)

```
upstream coding-progress-ledger repo
    │   produces ledger.jsonl + run_manifest.json + summary_by_category.json
    │   per run, per source
    ▼
ingest/sources.py + paths.py        (locates run directories)
    │
    ▼
checkpoints/build.py                (replay → per-checkpoint feature row)
    │
    ▼
datasets/checkpoints_<source>.parquet  (and an _all variant)
labels/build.py + V0_TARGETS         (compute labels per checkpoint per target)
    │
    ▼
datasets/labels_<source>.parquet     (long form — one row per run, ckpt, target)
    │
    ▼
splits/builder.py                    (LORO / LTFO / LOSO splits as JSON)
    │
    ▼
checkpoints/dynamics.attach_g5_features (NEW: post-processing layer)
    │
    ▼
eval/harness.py + baselines/ + models/
    │
    ▼
leakage/d5_audit.run_d5_audit       (NEW: structural + behavioral checks)
eval/go_no_go.evaluate_gate         (consumes D5 + harness outputs)
eval/sign_off.build_sign_off        (gates → model card → P3 routing)
    │
    ▼
reports/  +  models/<id>/  (artifacts)
```

The upstream ledger source remains at
`/Users/grantwilkins/houdini/coding-progress-ledger/`. That repo
defines what a checkpoint *is*, what events count, how progress is
measured. This repo treats that as a frozen API.

### Splits (unchanged)

Four split schemes, all explicit in `splits/protocol.py`:

- **holdout** — random run-disjoint train/test. Default for headline.
- **loro** — leave-one-run-out, per source. Stress-test for run
  generalization within a source.
- **ltfo** — leave-task-family-out. Stress-test for task generalization
  within a source.
- **loso** — leave-one-source-out. Stress-test for source-to-source
  transfer. Used for retro→tb_live.

All splits are run-disjoint. The harness rejects two test folds
sharing a run (`evaluate_cell` raises).

---

## Data sources & current state

Three canonical sources today (`coding_estimator/ingest/sources.py`):

| source              | runs | checkpoints | labels (non-masked y_success)              | y_success rate |
|---------------------|-----:|------------:|--------------------------------------------|---------------:|
| `swe_agent_pilot`   |   20 |         599 | 599 (full coverage, 10 succ / 10 fail)     |          0.50  |
| `hermes_pilot_h5_v2`|   30 |         896 | **0** — all 30 runs upstream-unannotated   |       n/a      |
| `tb_live`           |   12 |          83 | 83 (full coverage, 12 succ / **0 fail**)   |          1.00  |

This table is the load-bearing fact for everything that follows.

### Diagnoses recorded since revision 1

1. **`tb_live`** — single-class y_success. Cohort selection problem;
   not a sample-size problem. P1.b, P1.d, the tb_live half of O7 all
   structurally un-evaluable until failures are added.

2. **`hermes_pilot_h5_v2`** — confirmed upstream-unannotated.
   Diagnosis (one-page): `reports/HERMES_LABEL_DIAGNOSIS.md`. Three
   remediation paths laid out:
   - Path 1: re-run the upstream LLM-annotation pipeline. ~50-run
     P1.c becomes testable. Recommended.
   - Path 2: flip `canonical_for_v0=False` and re-frame P1.c as
     "swe alone" with the § C1 caveat banner. Compromise.
   - Path 3: drop P1.c entirely. Requires explicit policy change.

3. **`swe_agent_pilot`** is the only canonical source with both
   classes and complete labels. It carries the v0 measurement.

### Source-version fields

Every checkpoint row carries `source_protocol_version`. The card
emitter records the set of versions seen per source. If the upstream
ledger format bumps version, the estimator must rebuild from scratch
— no cross-version blending is permitted (see `docs/VERSIONING.md`).

---

## What's built (workstream-by-workstream)

`TASKS.md` is the source of truth for status. Quick map (revision 2):

| Workstream | What | Status |
|---|---|---|
| A | source registry, ingest, replay scaffold | shipped |
| B | label registry + computation hooks | shipped |
| C | canonical-source decisions, manifests | shipped |
| D | feature builder + leakage audits (D5) | shipped, **D5 structured artifact ✓** |
| E | label computation for V0_TARGETS | shipped |
| F | data-readiness profiling + go-no-go on data | shipped |
| G | baseline ladder | shipped, **G5 (dynamics) added as diagnostic** |
| H | LTFO + slice eval + jinja eval-report | shipped |
| I | I0 empirical-bin + I1 logistic regression | shipped |
| J | calibration metrics, recalibration, slice/headline reports | shipped |
| K | tb_live-only checkpoint eval (K1) + qualitative rollup (K3) | shipped (K2 deferred) |
| L | retro→live transfer with feature-group ablation (L3 only) | shipped (L1/L2/L4 deferred/blocked) |
| M | online inference / streaming | DEFERRED until P passes |
| N | model card schema + bundle + versioning | shipped |
| O | failure-mode tests (O1, O5, O7 only) | shipped |
| P | go/no-go gate + sign-off + readiness | shipped — verdict **INDETERMINATE** |
| Q | semantic features + sequence models | DEFERRED until P passes |
| R | scheduler consumer | EXPLICITLY OUT OF SCOPE |
| **N/A** | **Human-baseline scaffolding (recommended)** | **shipped, awaiting human pass** |
| **N/A** | **V0 findings memo + Hermes diagnosis** | **shipped** |

Tests: 644 pytest tests, all passing. Most workstreams have a
research-test-creator-style file in `tests/` that targets specific
plausible wrong implementations rather than coverage filler.

---

## The current verdict & why it sits where it does

`reports/ESTIMATOR_GO_NO_GO.md` is the canonical output. The verdict
is **INDETERMINATE**, blocked by three required conditions:

| id   | required | outcome       | one-line reason                                                                 |
|------|:--------:|---------------|---------------------------------------------------------------------------------|
| P1.a |   yes    | ✅ pass       | G4 wins or ties G2 on 6 of 8 (target, source) cells.                            |
| P1.b |   yes    | ⚠ indeter.   | tb_live `y_success_eventual` is single-class (12/12); ECE_3bin can't be tested. |
| P1.c |   yes    | ⚠ indeter.   | hermes labels missing; `swe ∪ hermes` not testable as plan defines it.          |
| P1.d |   yes    | ⚠ indeter.   | tb_live `y_success_eventual` single-class; LOSO Brier can't be compared.        |
| P1.e |   yes    | ✅ pass       | No forbidden columns (exact/prefix/suffix all checked).                         |
| P1.f |   yes    | ✅ pass       | Zero run-constant (feature, target) pairs across G4 training folds.             |
| P1.g |   yes    | **✅ pass**   | **D5 audit clean (62 runs / 1578 checkpoints; 0 findings).**                    |
| P1.h |   yes    | ✅ pass       | Winning cells span multiple targets; SWV-only caveat does not apply.            |

The single difference between revision 1 and revision 2 of the gate:
P1.g was indeterminate (no D5 artifact). Now it's pass (D5 ships,
clean). The other state changes (P1.c indeterminate vs the prior
fail, P1.h required) reflect bug fixes from the post-implementation
critic round, not data movement.

### How the verdict is computed

`coding_estimator/eval/go_no_go.py::_decide_verdict`:

```python
required = [c for c in conditions if c.required]
if not required:
    return "indeterminate"
if any(c.outcome == "fail" for c in required):
    return "fail"
if all(c.outcome == "pass" for c in required):
    return "pass"
return "indeterminate"
```

This is a *strict* aggregator: any single indeterminate among the
required-set blocks pass. The cost is that v0 stays indeterminate
until tb_live cohort diversity, hermes labels all land. The benefit
is that the gate cannot pass on incomplete evidence.

### What the verdict means for `not_safe_for_control`

`coding_estimator/eval/sign_off.py::_decide_not_safe_for_control`
trips the `not_safe_for_control` flag if **any** of the following:

- gate.verdict != "pass"
- any required GateCondition.outcome != "pass"
- O1.outcome != "pass"
- O5.outcome != "pass"
- any per-source O7 result.outcome == "fail"

For v0 today, items 1, 2, and 5 trip. The flag is `true`, embedded in
`models/ledger_basic_v0.1/model_card.json` (which validates against
`schemas/model_card_schema.json`). Per `docs/VERSIONING.md`, flipping
the flag from `true` to `false` requires a `<major>.<minor>` version
bump *and* the gate plus every failure-mode test passing with margin.

---

## The recentered v0 finding

Revision 1 buried this as "the hard finding (O7)" — the negative
result on terminal success. Revision 2 promotes it to the **headline**:

### Primary v0 claim

> **Prefix-only ledger features predict near-future progress dynamics.**

Evidence (`reports/g5/g5_eval.md`, `swe_agent_pilot` LORO,
`y_future_progress_drop_h5`):

| model           | Brier | AUROC | Δ vs G2 (Brier) |
|-----------------|------:|------:|----------------:|
| G2 time_only    | 0.142 | 0.626 |       —         |
| G4 ledger_basic | 0.039 | 0.977 |     **−0.102**  |
| G5 dynamics     | 0.078 | 0.897 |       −0.064    |
| G4 + G5         | 0.042 | 0.973 |       −0.100    |

A ten-point Brier improvement over the elapsed-time baseline at AUROC
0.977 is large for v0. Corroborates on `tb_live` (10 runs, 23 labeled
checkpoints): G4 still beats G2 by 0.037 Brier.

`y_validation_new_work_h5` shows the same direction at smaller
absolute Briers (very rare positives on swe_agent_pilot; richer base
rate on tb_live) — ledger features beat or match elapsed time on
every cell.

**Two qualifications on the G5 result:**
1. G5 alone clears G2 by ~6 Brier points but does **not** stack
   additively with G4 (G4+G5 ≈ G4). G5 carries an independent but
   *overlapping* share of the same dynamics signal, not complementary
   information.
2. On `tb_live`, G5 alone is slightly worse than G4 alone (0.124 vs
   0.096). At very small N, the dynamics features need more data to
   compete with the broader G4 set.

### Secondary / negative claim

> **Prefix-only ledger features do not improve terminal success
> prediction over elapsed time at this N.**

Evidence (`swe_agent_pilot` LORO, `y_success_eventual`):

| model           | Brier | AUROC | Δ vs G2 (Brier) |
|-----------------|------:|------:|----------------:|
| G2 time_only    | 0.283 | 0.281 |       —         |
| G4 ledger_basic | 0.291 | 0.410 |       +0.009    |
| G5 dynamics     | 0.272 | 0.385 |       −0.010    |
| G4 + G5         | 0.292 | 0.411 |       +0.010    |

G4 is *worse* than G2 by 0.009. G5 alone gets −0.010 (within noise).
Combining (G4+G5) sits back at G4. The strongest scientific gate (O7
in `coding_estimator/eval/failure_modes.py`) demands a +0.02 Brier
lift; none of the ledger configurations reaches that on the
largest retrospective source.

`tb_live` is uninformative for this target: 12/12 successes
(single-class y).

### Interpretation

The estimator's job description in the project mission is "a belief
layer over live coding-progress ledgers" that "consumes prefix-only
ledger features and outputs calibrated probabilities over successful
completion by future horizons, remaining time, and near-future
progress dynamics."

The v0 measurement says: **the dynamics half of that mission is
achievable today on retrospective data, by a wide margin on the
largest source. The completion half is dominated by a one-feature
elapsed-time baseline at this N, even after adding the dynamics
feature group.**

A ledger watching a coding agent measures **what the agent has
visibly done so far**. That signal is local: it predicts what the
agent will do next better than it predicts whether the run will
ultimately succeed. Terminal success is downstream and confounded by
hidden requirements (test harness specifics, failure-mode coverage,
unannotated retrospective traces). The current data plus the current
feature set are not yet enough to bridge that gap.

This is the publishable boundary. The observation channel is doing
what it should — it sees process shape — and the v0 gate's strict
+0.02 threshold has correctly told us we cannot promise more.

---

## Open problems & decision points (revision 2)

These are places where another human's judgment matters because the
data does not decide. Some have moved since revision 1.

### Decision 1 (refined) — Is the success-prediction negative result terminal?

Revision 1 framed this as "is G5 the cheapest experiment?" — implying
the answer might rescue success. **G5 is built; it does not rescue
success.** The decision now refines:

- **Option 1A — More feature engineering can save it.** Possible
  candidates: longer-horizon dynamics (slope_25, recency over 50
  steps), interaction features between progress and validation, or
  semantic features (Workstream Q). Cost: weeks. Risk: muddies the
  ledger-native claim.
- **Option 1B — More data can save it.** Hermes labels + a 50-run
  retrospective pool might reveal a small G4 lift on success that's
  currently within noise. Cost: weeks-to-months waiting on upstream
  annotation. Lower risk.
- **Option 1C — The negative result is the v0 finding.** Publish:
  "ledger features predict process dynamics but not completion at
  this N; here is the boundary." Cost: low. Risk: gives up on the
  flagship target.

The user's feedback explicitly recommended 1B + 1C
(`do not push toward scheduling`). The reviewer should confirm or
override.

### Decision 2 (still open) — Pipeline-broken or pipeline-starved?

`tb_live` has 12 successes and 0 failures. For the gate to ever
become testable, this must change. Two paths:

- **Pipeline-broken**: the TB-12 cohort was selected to be tractable.
  Fix: run the agent on harder TB tasks; collect ≥ 5 failures.
  ~1–2 weeks.
- **Pipeline-starved**: the cohort *is* the available distribution.
  Fix: enlarge the cohort to ≥ 30 runs regardless of outcome ratio.
  ~1–3 months.

The reviewer should know which is which before recommending. The
project does not have visibility into the upstream cohort selection.

### Decision 3 (still open) — Threshold sanity at N=20

The plan justifies +0.02 as "the strongest scientific gate." At N=20
runs with run-level bootstrap CIs, the noise floor on Brier is
roughly ±0.05. So a threshold tighter than the noise floor is
operationally ambiguous: G4 could be genuinely identical to G2 in
expectation and still fail O7 by sign.

A reviewer might argue: at N=20, gate O7 at +0.05 (the same threshold
P1.b/P1.d use) and accept that as the v0 finding. With +0.05, swe
delta of -0.009 is still a fail (negative, not just small) — so this
loosening doesn't rescue v0. But it makes the math match the noise.

### Decision 4 (still open) — Should we ship the indeterminate verdict?

`reports/NOT_READY_FOR_SCHEDULING.md` is on disk. The model card
records `not_safe_for_control = true`. The bundle at
`models/ledger_basic_v0.1/` is reproducible from scratch.

Argument for shipping as-is: the artifact is honest. Anyone who
reads the sign-off learns exactly what cleared and what didn't. The
recentered V0_FINDINGS memo gives the publishable shape.

Argument against: "indeterminate" is a weaker headline than "G4 beats
G2 by 0.102 Brier on `y_future_progress_drop_h5` with run-level
bootstrap, at AUROC 0.977". The dynamics finding might warrant its
own publication independent of the gate.

### Decision 5 (NEW) — Should `hermes_pilot_h5_v2` be flipped to non-canonical?

Path 2 in `reports/HERMES_LABEL_DIAGNOSIS.md` recommends flipping
`canonical_for_v0=False` until annotation lands, so consumers of the
source registry don't quietly count hermes as in-scope. This is a
contentful change that affects:

- the gate report (P1.c reframed as "swe alone with caveats")
- TASKS.md § 0.0β banner
- model card source_versions

If the upstream annotation pass is < 4 weeks away, hold the flag.
If > 4 weeks, consider flipping.

---

## Recommended next steps (my opinion; reviewer's to override)

In strict priority order. Steps 1–3 are data work; step 4 is a small
human study; only after those are in is more model work justified.

1. **Annotate the 30 hermes_pilot_h5_v2 runs upstream.** The runs
   exist; the ledgers exist; the checkpoints frame already has 896
   hermes rows. The gap is a single upstream annotation pass. After
   it lands, run:
   ```bash
   uv run python -c "
   from coding_estimator.labels.build import write_combined_labels
   from pathlib import Path
   write_combined_labels(Path('datasets'))
   "
   ```
   No code changes in this repo. P1.c becomes testable on ~50 runs
   immediately.

2. **Collect `tb_live_v2` with outcome diversity.** ≥ 30 runs / ≥ 10
   failures / real wall-clock / same sidecar/ledger protocol. Do NOT
   tune the agent to make tasks succeed. Without failures, P1.b /
   P1.d / O7-on-tb_live remain structurally un-evaluable.

3. **Run the human-baseline experiment.** The scaffolding is on
   disk:
   ```bash
   uv run python scripts/run_human_baseline.py prepare \
     --checkpoints datasets/checkpoints_all.parquet \
     --out-dir reports/human_baseline --n-samples 6
   ```
   produces 6 midpoint-prefix prompts at
   `reports/human_baseline/prompts/`. One human reads them, fills in
   `reports/human_baseline/human_predictions.csv`. Then:
   ```bash
   uv run python scripts/run_human_baseline.py compare \
     --checkpoints datasets/checkpoints_all.parquet \
     --labels datasets/labels_all.parquet \
     --out-dir reports/human_baseline
   ```
   produces `reports/human_baseline/comparison.{md,csv}`. The
   comparison answers whether the ledger is *readable* as a belief
   signal. If G4 matches the human on dynamics, the channel carries
   the signal; if G4 trails the human, the model is weak.

4. **Decide whether to publish the v0 dynamics finding now or wait
   for steps 1–3 to land.** The dynamics result is robust on
   `swe_agent_pilot` and corroborates on `tb_live`. Publishing it
   independent of the success-target negative result is defensible;
   the V0_FINDINGS memo is the draft.

5. **Re-run the full pipeline after steps 1–3.** Every artifact
   regenerates from `scripts/`:
   ```
   scripts/run_baselines.py        # G ladder
   scripts/run_model_ladder.py     # I ladder
   scripts/run_calibration.py      # J reports
   scripts/run_g5_eval.py          # G5 dynamics evaluation
   scripts/run_tb_live_eval.py     # K1 + K3
   scripts/run_retro_to_live.py    # L3
   scripts/run_failure_modes.py    # O
   scripts/run_d5_audit.py         # D5 audit JSON
   scripts/run_go_no_go.py         # P1
   scripts/run_sign_off.py         # P2 + P3
   scripts/run_human_baseline.py   # human study
   ```
   None takes more than ~30 seconds on the current dataset.

### What we do NOT recommend (per user feedback)

- **Loosening the +0.02 O7 threshold** to get a pass.
- **Adding semantic / text features (Workstream Q)** before fixing
  data defects.
- **Building any controller, scheduler, or online-inference surface**
  (Workstreams M, R remain explicitly out of scope until P passes).
- **Chasing more feature engineering on `y_success_eventual`** without
  a hypothesis specific enough to predict G5 wouldn't reach (since
  we now know G5 doesn't move it).

---

## Things that surprised me / non-obvious gotchas

Eight items from revision 1 still apply; three are new this round.

- **`y_submit_without_validation` is run-constant.** P1.h was added
  to block a v0 that passes only via this target. P1.h is now
  required (revision 2 fix). Currently passes on real data because
  winners span multiple targets.

- **`fraction_timeout_consumed` only populates on `tb_live`.** It is
  not available on retrospective sources. O7's "use only the timeout
  feature" idea translates as "use elapsed_steps" on retro sources;
  G2 (elapsed_steps) is the v0 stand-in.

- **The Platt recalibrator was using sklearn's default L2
  regularization until commit 433ccd1.** Standard Platt is the
  unregularized MLE on logit(p). The fix is at
  `coding_estimator/calibration/recalibrate.py:62` (`C=1e10`).

- **The kfold isotonic recalibrator in P1.b previously fell back to
  in-sample fit when n_runs < 2.** That was a silent footgun. Now
  raises `InsufficientRunsForRecalibrationError` and P1.b returns
  indeterminate.

- **The `models/` directory is gitignored.** Bundles are reproducible
  from `scripts/run_sign_off.py` against the commit SHA recorded on
  the card.

- **`save_model_bundle` validates the model card JSON before
  writing.** Tested at
  `tests/test_model_card.py::test_write_card_validates_before_writing`.

- **The leakage audit covers `exact`, `prefix`, AND `suffix`.**
  Tested at `tests/test_go_no_go.py::test_p1e_catches_*`.

- **The upstream `coding-progress-ledger` repo is the source of
  truth.** This repo is downstream. If a feature column changes name
  or semantics there, this repo's feature registry must bump and
  every existing bundle becomes invalid (`docs/VERSIONING.md`).

### New gotchas this round

- **D5 shuffle dispatches on `V0_TARGETS[t].run_constant_flag`.** A
  prior version of the shuffle test collapsed every run to a single
  first-row label even for non-run-constant targets — that was
  methodologically wrong and produced an artifact finding on
  tb_live/`y_validation_new_work_h5`. Fixed in commit 548c42a:
  run-constant → run-level shuffle, non-run-constant → row-level
  shuffle. Plus `SHUFFLE_MIN_RUNS=8` and `SHUFFLE_MIN_CHECKPOINTS=30`
  floors so seed-variance can't manufacture findings at small N.

- **`G5_FEATURES` are derived columns, not built columns.** They
  exist only after `attach_g5_features()` is called; the parquet on
  disk does not carry them. Any code that consumes a checkpoints
  frame and wants G5 must call the helper. Tested for prefix-only
  invariance under truncation (`test_dynamics_g5.py`).

- **`HUMAN_BASELINE_TARGETS` includes `y_success_eventual`** even
  though tb_live's success target is single-class. That makes the
  human's success-probability number scientifically uninformative
  on the current cohort (no failure exemplars to anchor against),
  but the prompt structure is preserved for `tb_live_v2` when
  failures are added. Reviewer may want the human to focus on the
  dynamics target only.

---

## Key files to read first (revision 2 reading order)

1. `reports/V0_FINDINGS.md` — **start here**. The publishable
   narrative; primary claim, secondary negative result, what the
   pipeline guarantees, what the data answers, what it doesn't.
2. `reports/REVIEWER_BRIEFING.md` — this document.
3. `reports/HERMES_LABEL_DIAGNOSIS.md` — why P1.c is indeterminate
   and how to fix it.
4. `reports/ESTIMATOR_GO_NO_GO.md` — full P1.a–h gate evidence.
5. `reports/sign_off_ledger_basic_v0.1.md` — consumer-facing v0
   sign-off.
6. `reports/g5/g5_eval.md` — per-target G2 / G4 / G5 / G4+G5
   comparison.
7. `reports/d5_audit.md` and `reports/d5_audit.json` — structured
   D5 behavioral leakage audit.
8. `reports/NOT_READY_FOR_SCHEDULING.md` — prioritized BLOCKING /
   DATA / AUDIT actions.
9. `TASKS.md` (start at § 0.0β banner) — the plan.
10. `coding_estimator/eval/go_no_go.py` — the gate definition; each
    P1.* is its own function.
11. `coding_estimator/eval/failure_modes.py` — O1, O5, O7
    definitions.
12. `coding_estimator/checkpoints/dynamics.py` — the new G5 layer.
13. `coding_estimator/leakage/d5_audit.py` — the new D5 producer.
14. `models/ledger_basic_v0.1/model_card.md` — the public face of
    the v0 estimator.

---

## What I would tell a reviewer in two sentences

The estimator pipeline now produces an honest indeterminate verdict
because the data gaps blocking it are documented and remediable
upstream, and the recentered v0 finding (ledger features predict
process dynamics by a wide margin on the largest available source,
but not terminal success at this N) is a real, publishable boundary.
The decision the reviewer faces is whether to publish the dynamics
result now, hold for the Hermes annotation + tb_live_v2 collection
to land, or — since v0's flagship success target is dominated by
elapsed time and dynamics features did not change that — refine the
project's framing toward what the observation channel is actually
measuring.
