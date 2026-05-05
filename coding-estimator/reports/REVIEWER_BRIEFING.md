# Reviewer briefing — coding-estimator v0

_Generated 2026-05-05. For an independent reviewer who has not seen this
project before. Read top to bottom; the TL;DR is load-bearing._

## TL;DR (60 seconds)

The v0 estimator (a logistic regression on prefix-only ledger features,
predicting four belief-state targets per checkpoint) **does not clear
its no-regression gate**. The gate is honestly INDETERMINATE — not
FAIL — at current data, because four of eight required conditions
cannot be evaluated:

- two of them depend on `tb_live` having outcome diversity, but the
  current TB-12 cohort is **12 successes, 0 failures**;
- one depends on `hermes_pilot_h5_v2` labels that have not been built
  into the labels parquet;
- one depends on a `D5` behavioral leakage audit artifact that has
  not been produced.

The single most important finding is from **O7 (timeout-bias)**: on
the largest source we have (`swe_agent_pilot`, 20 runs, 599
checkpoints), G4 (ledger features) Brier = 0.291 and G2 (elapsed
time only) Brier = 0.283. **G4 is 0.009 worse than G2.** Per the
plan's strongest scientific gate, the v0 ledger features add nothing
beyond elapsed time on the largest available retrospective source.
This is not a bug. This is what the data says.

`not_safe_for_control = true` is enforced and recorded on the model
card. The headline output is `reports/NOT_READY_FOR_SCHEDULING.md`.

The reviewer's job, in priority order: (1) decide whether O7's finding
is terminal for the v0 *feature design* (vs. fixable by adding the
deferred dynamics group G5), (2) decide whether the data gaps blocking
P1.b/c/d/g are upstream-pipeline problems or this-repo problems, (3)
decide whether to ship anyway with the indeterminate verdict and
loosened thresholds.

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
selector. It is a *belief layer*. The downstream consumer is
deferred to a future repo and is gated on this one passing P1.

The plan that defines all of this: `TASKS.md` (2045 lines). Workstream
labels A–R correspond to project phases; A–P are in scope, Q/R are
deferred.

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
│   │   └── fills.py                 # Per-source canonical fill application
│   ├── labels/
│   │   ├── registry.py              # V0_TARGETS — declared targets and their semantics
│   │   ├── terminal.py              # y_success_eventual, y_finish_step, y_timeout, ...
│   │   ├── dynamics.py              # y_future_progress_drop_h5, etc.
│   │   ├── shapes.py                # Run-shape labels (slicing, NOT predictions)
│   │   └── balance.py               # Label class-balance audit
│   ├── splits/
│   │   ├── builder.py               # Build per-source split JSONs from manifests
│   │   └── protocol.py              # Fold/Split, loro(), ltfo(), holdout()
│   ├── baselines/
│   │   ├── constant.py              # G1 — constant base-rate
│   │   ├── time_only.py             # G2 — elapsed_steps (+ wall-clock when tb_live)
│   │   └── ledger_basic.py          # G4 — closure/frontier/instability/discovery
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
│   │   └── sign_off.py              # P2 sign-off package + P3 ready/not-ready routing
│   ├── calibration/
│   │   ├── metrics.py               # J1 — brier, ECE, reliability_table
│   │   ├── recalibrate.py           # J4 — Platt / isotonic / source-isotonic
│   │   └── report.py                # J2/J3/J5 — markdown reliability + slice + headline
│   ├── leakage/
│   │   ├── guard.py                 # Forbidden-column audit (exact / prefix / suffix)
│   │   └── run_constancy.py         # Run-constant feature × target pair audit
│   ├── ingest/                      # Source registry + path resolution
│   ├── reports/                     # Jinja eval-report renderer
│   └── profile/                     # Data profiling (D5, F, etc.)
├── datasets/                        # Parquet checkpoint and label tables
├── models/                          # Saved estimator bundles (gitignored)
├── reports/                         # Generated artifacts
├── schemas/                         # JSON schemas for validation
├── docs/                            # MODEL_CARD_TEMPLATE, VERSIONING, ESTIMATOR_*
├── scripts/                         # Driver scripts (one per workstream phase)
└── tests/                           # 622 pytest tests
```

### The data flow

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
eval/harness.py + baselines/ + models/
    │
    ▼
reports/  +  models/<id>/  (artifacts)
```

The upstream ledger source is at
`/Users/grantwilkins/houdini/coding-progress-ledger/`. That repo
defines what a checkpoint *is*, what events count, how progress is
measured. This repo treats that as a frozen API.

### Splits

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
| `hermes_pilot_h5_v2`|   30 |         896 | **0** (labels not in `labels_all.parquet`) |       n/a      |
| `tb_live`           |   12 |          83 | 83 (full coverage, 12 succ / **0 fail**)   |          1.00  |

This table is the load-bearing fact for everything that follows.

### Implications

1. **`tb_live` has zero failures.** Any condition that requires both
   classes of `y_success_eventual` on `tb_live` is structurally
   un-evaluable: P1.b, P1.d, the tb_live half of O7. This is not
   a sample-size problem; it is a *cohort selection* problem. The
   TB-12 cohort was apparently chosen with the model "we want hard
   ones the agent can plausibly do" rather than "we want a
   representative mix of outcomes."

2. **`hermes_pilot_h5_v2` labels are missing from
   `datasets/labels_all.parquet`.** The checkpoints frame has 30
   hermes runs and 896 hermes checkpoints, but the label parquet
   only contains rows for `swe_agent_pilot` and `tb_live`. P1.c —
   the *one* condition the plan explicitly asks to be a CI-exclusion
   result — was supposed to evaluate on `swe ∪ hermes (~50 runs,
   400+ checkpoints)`. Without hermes labels, that combined pool
   doesn't exist; the swe-alone result is not the test the plan
   designed.

   Open question for the reviewer: is this a real gap upstream, or
   is it just that the local label-build script was never run for
   hermes? Look at:
   - `coding_estimator/labels/build.py`
   - `coding_estimator/models/readiness.py:53` (warns
     "hermes_pilot_h5_v2: 0 of 30 runs produced labels (unresolvable=30,
     malformed=0). Source registry caveat?")
   - upstream repo's annotation status

   If hermes labels need fresh upstream re-annotation, P1.c stays
   indeterminate for weeks. If it's a build-script bug, this is a
   day's work.

3. **`swe_agent_pilot` is the only source with both classes and a
   complete label table.** That makes it the de-facto headline
   source. O7 says G4 is *worse* than G2 there. That finding is
   the load-bearing scientific result of the project so far.

### Source-version fields

Every checkpoint row carries `source_protocol_version`. The card
emitter records the set of versions seen per source. If the upstream
ledger format bumps version, the estimator must rebuild from scratch
— no cross-version blending is permitted (see `docs/VERSIONING.md`).

---

## What's built (workstream-by-workstream)

`TASKS.md` is the source of truth for status. Quick map:

| Workstream | What | Status |
|---|---|---|
| A | source registry, ingest, replay scaffold | shipped |
| B | label registry + computation hooks | shipped |
| C | canonical-source decisions, manifests | shipped |
| D | feature builder + leakage audits (D5) | shipped (D5 partial — see P1.g below) |
| E | label computation for V0_TARGETS | shipped |
| F | data-readiness profiling + go-no-go on data | shipped |
| G | baseline ladder (G1, G2, G4 only; G3/G5/G6 deferred) | shipped |
| H | LTFO + slice eval + jinja eval-report | shipped |
| I | I0 empirical-bin + I1 logistic regression (I2-I5 deferred) | shipped |
| J | calibration metrics, recalibration, slice/headline reports | shipped |
| K | tb_live-only checkpoint eval (K1) + qualitative rollup (K3) | shipped (K2 deferred) |
| L | retro→live transfer with feature-group ablation (L3 only) | shipped (L1/L2/L4 deferred/blocked) |
| M | online inference / streaming | DEFERRED until P passes |
| N | model card schema + bundle + versioning | shipped |
| O | failure-mode tests (O1, O5, O7 only; O2/O3/O4/O6 deferred) | shipped |
| P | go/no-go gate + sign-off + readiness | shipped — verdict INDETERMINATE |
| Q | semantic features + sequence models | DEFERRED until P passes |
| R | scheduler consumer | EXPLICITLY OUT OF SCOPE — different repo |

Tests: 622 pytest tests, all passing. Most workstreams have a
research-test-creator-style file in `tests/` that targets specific
plausible wrong implementations rather than coverage filler.

---

## The current verdict & why it sits where it does

`reports/ESTIMATOR_GO_NO_GO.md` is the canonical output. The
verdict is **INDETERMINATE**, blocked by four required conditions:
P1.b, P1.c, P1.d, P1.g.

Per-condition state:

| id   | required | outcome       | one-line reason                                                                 |
|------|:--------:|---------------|---------------------------------------------------------------------------------|
| P1.a |   yes    | ✅ pass       | G4 wins or ties G2 on 6 of 8 (target, source) cells.                            |
| P1.b |   yes    | ⚠ indeter.   | tb_live `y_success_eventual` is single-class (12/12); ECE_3bin can't be tested. |
| P1.c |   yes    | ⚠ indeter.   | hermes labels missing; `swe ∪ hermes` not testable as plan defines it.          |
| P1.d |   yes    | ⚠ indeter.   | tb_live `y_success_eventual` single-class; LOSO Brier can't be compared.        |
| P1.e |   yes    | ✅ pass       | No forbidden columns (exact/prefix/suffix all checked).                         |
| P1.f |   yes    | ✅ pass       | Zero run-constant (feature, target) pairs across G4 training folds.             |
| P1.g |   yes    | ⚠ indeter.   | D5 audit artifact not provided; bare `{clean: true}` rejected.                  |
| P1.h |   yes    | ✅ pass       | Winning cells span multiple targets; SWV-only caveat does not apply.            |

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
required required-set blocks pass. That is a deliberate choice. The
alternative — "indeterminate counts as not-blocked" — would let the
gate pass on incomplete data. The cost is that the v0 verdict is
permanent until tb_live cohort diversity, hermes labels, and D5 audit
all land.

### What the verdict means for `not_safe_for_control`

`coding_estimator/eval/sign_off.py::_decide_not_safe_for_control`
trips the `not_safe_for_control` flag if **any** of the following:

- gate.verdict != "pass"
- any required GateCondition.outcome != "pass"
- O1.outcome != "pass"
- O5.outcome != "pass"
- any per-source O7 result.outcome == "fail"

For v0 today, all five of these trip. The flag is `true`, embedded in
`models/ledger_basic_v0.1/model_card.json` (which validates against
`schemas/model_card_schema.json`). Per `docs/VERSIONING.md`, flipping
the flag from `true` to `false` requires a `<major>.<minor>` version
bump *and* the gate plus every failure-mode test passing with margin.

---

## The hard finding: O7 fails on swe_agent_pilot

This deserves its own section because the plan calls O7 "the
strongest scientific gate."

```
O7 — timeout-bias test:
  Per source under LORO: compare G2 (time-only) against G4 (ledger-basic)
  on y_success_eventual. PASS iff Brier_G2 - Brier_G4 >= 0.02.
  FAIL iff < 0.02. INDETERMINATE iff y is single-class on the source.
```

Result on `swe_agent_pilot` (20 runs, 599 checkpoints, 10/10 outcome
split):

```
Brier_G2 = 0.283
Brier_G4 = 0.291
Brier_G2 - Brier_G4 = -0.009  →  FAIL
```

Reading: G4 (closure + frontier + instability + discovery feature
groups) is *worse* than G2 (just `elapsed_steps`) at predicting
`y_success_eventual` under LORO. The negative delta is small, but it
is on the *largest* canonical source we have, and the gate threshold
is +0.02 — so even being a tie would fail.

### What this could mean

Three explanations to weigh, not mutually exclusive:

1. **The G4 feature set is the wrong set for `y_success_eventual` at
   this N.** Closure / frontier / instability / discovery describe
   the *current state* of work; they don't describe *trajectory*.
   The deferred G5 group (ledger-dynamics: progress acceleration,
   stuck-loop detection, validation freshness) is the natural addition.
   If G5 lifts G4 above G2, the gate becomes meaningful.

2. **At N=20 runs, logistic regression is overfitting the LORO folds
   on G4's ~16 features but not on G2's 1 feature.** G2's lower
   capacity is a happy regularizer. This would resolve at larger N
   without changing features.

3. **`y_success_eventual` is a fundamentally hard target from
   prefix-only features at this checkpoint cadence.** Final outcome
   depends on terminal validation, which is by definition outside
   the prefix's information set. The `coding_progress` feature is
   informative *because* the agent's running progress is monotone
   in the absence of regressions; once it's at 0.9, success is
   plausible regardless of what the rest of the ledger looks like.
   In that regime, `elapsed_steps` (which correlates with progress
   when progress is monotone) carries most of the signal.

The reviewer's call here matters. Option 1 says "build G5, retest, this
is fixable." Option 2 says "wait for more data." Option 3 says "the
v0 framing is wrong; pick targets that aren't dominated by elapsed
time." All three have evidence; none is decisive.

### Specific reading recommendations

- `reports/baseline_h/baseline_results.md` — full G1/G2/G4 numbers
  across LORO/LTFO/LOSO. Shows that G4 *does* beat G2 on
  `y_future_progress_drop_h5` on swe_agent_pilot (Brier 0.039 vs
  0.142, AUROC 0.97 vs 0.92). The G4-loses pattern is specific to
  `y_success_eventual`. That asymmetry is informative.
- `reports/retro_to_live_transfer.md` — the L3 ablation. Removing
  feature groups one at a time. On `y_success_eventual`,
  `g4_minus_frontier` actually *improves* over `g4_full` (Brier 0.136
  vs 0.145). That suggests one group may be hurting — worth probing.
- `reports/calibration/calibration_v0.md` — every (model, source,
  target) cell flagged `not_safe_for_control` post-isotonic
  recalibration. Most cells are flagged. Calibration is a separate
  problem from the discrimination problem O7 catches.

---

## Open problems & decision points

These are places where another human's judgment matters because the
data does not decide.

### Decision 1 — Is G5 next, or is the v0 framing wrong?

The cheapest experiment: implement G5 (ledger-dynamics features —
progress slope, stuck-loop detection, validation freshness — there's
a partial spec in `coding_estimator/labels/dynamics.py` for the
*label* side; the feature side is sketched in `TASKS.md` § Workstream
G but not implemented). Re-run O7. If G5 lifts G4 above G2 by ≥ 0.02
on swe_agent_pilot, the v0 framing is salvageable.

If G5 doesn't lift it, the strongest scientific result of this project
becomes "prefix-only ledger features at N=20 do not improve
`y_success_eventual` prediction over elapsed time." That is a *valid*
finding to publish, but it changes what v0 is for.

### Decision 2 — Is the data pipeline broken or starved?

`tb_live` has 12 successes and 0 failures. For the gate to ever
become testable, this must change. Two paths:

- **Pipeline-broken**: the TB-12 cohort was selected to be tractable.
  Fix: run the agent on harder TB tasks; collect ≥ 5 failures.
- **Pipeline-starved**: the cohort *is* the available distribution.
  Fix: enlarge the cohort to hit at least 50 runs, regardless of
  outcome ratio, and accept whatever outcome distribution emerges.

The first is a 1–2 week experiment. The second is a 1–3 month data
collection. The reviewer should know which is which before
recommending.

`hermes_pilot_h5_v2` has 30 runs of checkpoints but 0 labels. The
relevant scripts are `coding_estimator/labels/build.py` and
`coding_estimator/models/readiness.py`. The latter logs:

```
hermes_pilot_h5_v2: 0 of 30 runs produced labels
(unresolvable=30, malformed=0). Source registry caveat?
```

That message is the smoking gun. `unresolvable=30` says every run's
label-build path returned "I cannot compute this." Find why. Probable
candidates: missing `summary_by_category.json` (the label builder
for `y_success_eventual` on retrospective sources is annotation-driven,
not ledger-driven), or the `final_success_source` field absent on
hermes manifests. **This is the most leveraged fix in the project**:
unblocking it gives P1.c a real test on ~50 runs.

### Decision 3 — Is the gate's threshold of +0.02 right at N=20?

The plan justifies +0.02 as "the strongest scientific gate." But
at N=20 runs with run-level bootstrap CIs, the noise floor on Brier
is roughly ±0.05 (from the existing baseline reports). So a threshold
*tighter than* the noise floor is operationally ambiguous: G4 could
be genuinely identical to G2 in expectation and still fail O7 by
sign.

A reviewer might argue: at N=20, gate O7 at +0.05 (the same threshold
P1.b/P1.d use) and accept that as the v0 finding. With +0.05, swe
delta of -0.009 is still a fail (because it's *negative*, not just
small) — so this loosening doesn't rescue v0. But it's worth doing
because it makes the math match the noise.

### Decision 4 — Should we ship the indeterminate verdict?

`reports/NOT_READY_FOR_SCHEDULING.md` is on disk. The model card
records `not_safe_for_control = true`. The bundle at
`models/ledger_basic_v0.1/` is reproducible from scratch.

Argument for shipping as-is: the artifact is honest. Anyone who
reads the sign-off learns exactly what cleared and what didn't.
That is exactly what a sign-off package is for.

Argument against: "indeterminate" is not a strong enough finding to
publish. The bigger story is O7's negative result, which deserves
its own write-up rather than being buried as one row in a
multi-condition gate.

---

## Recommended next steps (my opinion; reviewer's to override)

In order of leverage:

1. **Diagnose the hermes label gap.** 1–2 days of work. Either fixes
   P1.c entirely (best case: gate becomes testable) or surfaces a
   real upstream blocker the project has been silently working around.
   Start at `coding_estimator/labels/build.py` and the upstream
   `runs/<run_id>/summary_by_category.json` files.

2. **Implement G5 dynamics features and rerun O7.** 3–5 days. The
   single experiment that disambiguates Decision 1. If G5 doesn't
   change O7's outcome on swe_agent_pilot, the v0 framing needs
   serious reconsideration.

3. **Audit the TB-12 cohort.** 1 day. Look at how the 12 tasks were
   chosen and whether the all-success outcome is selection-driven
   or distribution-driven. This determines whether tb_live is
   fixable in weeks or months.

4. **Ship the D5 audit artifact in the new structured format.** 2
   days. The new schema requires `schema_version`, `n_runs_audited`,
   `n_checkpoints_audited`, `findings`, `clean`. The D5 logic
   already exists in `coding_estimator/profile/leakage.py`; just
   needs to write the JSON artifact in the new shape. Unblocks P1.g
   without changing what's audited.

5. **Loosen the O7 gate to +0.05** (matching P1.b/d) and document
   the rationale. Doesn't change v0's verdict, but makes the
   threshold defensible at N=20.

6. **Only after 1–4: rerun the full pipeline, regenerate every
   report, decide.** The pipeline is end-to-end automated:

   ```bash
   uv run python scripts/run_baselines.py     ...   # G ladder
   uv run python scripts/run_model_ladder.py   ...   # I ladder
   uv run python scripts/run_calibration.py    ...   # J reports
   uv run python scripts/run_tb_live_eval.py   ...   # K1+K3
   uv run python scripts/run_retro_to_live.py  ...   # L3
   uv run python scripts/run_failure_modes.py  ...   # O
   uv run python scripts/run_go_no_go.py       ...   # P1
   uv run python scripts/run_sign_off.py       ...   # P2+P3
   ```

   None of these takes more than ~30 seconds on the current dataset.

---

## Things that surprised me / non-obvious gotchas

- **`y_submit_without_validation` is run-constant.** A logistic
  regression that predicts it gets a deceptively high AUROC at
  non-terminal checkpoints because the label is the same for every
  row of a run. P1.h was specifically added to block a v0 that
  passes only via this target. Currently passes (winners span
  multiple targets), but a future feature change could re-trigger
  this.

- **`fraction_timeout_consumed` only populates on `tb_live`.** It
  is not available on retrospective sources. This is why O7's
  "use only the timeout feature" formulation in the original plan
  doesn't translate cleanly: there is no timeout feature on
  swe_agent_pilot. The implementation uses G2 (elapsed_steps) as
  the v0 stand-in.

- **The Platt recalibrator was using sklearn's default L2
  regularization until very recently.** Standard Platt is the
  unregularized MLE on logit(p). The fix is at
  `coding_estimator/calibration/recalibrate.py:62` (`C=1e10`).
  This shifted the recalibrated cells in the J reports
  materially after it landed.

- **The kfold isotonic recalibrator in P1.b *previously* fell back
  to in-sample fit when n_runs < 2.** That was a silent footgun;
  fitting on test data trivially gives ECE ≈ 0. The fix raises
  `InsufficientRunsForRecalibrationError` and P1.b returns
  indeterminate. Same surface area in
  `coding_estimator/calibration/report.py::kfold_recalibrated_predictions`
  — that one falls back to single-fold fit-and-apply when
  unique_runs < 2 (descriptive context only, not a gate).

- **The `models/` directory is gitignored** (except `.gitkeep`).
  Bundles are reproducible from `scripts/run_sign_off.py` against
  the commit SHA recorded on the card. Don't expect to find the
  bundle in CI.

- **`save_model_bundle` validates the model card JSON before
  writing.** An invalid card never hits disk. Tested at
  `tests/test_model_card.py::test_write_card_validates_before_writing`.

- **The leakage audit covers `exact`, `prefix`, and `suffix` lists**
  in `schemas/forbidden_columns.json`. Tested for all three
  in `tests/test_go_no_go.py::test_p1e_catches_*`. A naive impl
  that only checks `exact` would silently pass — that wrong
  implementation existed at one point.

- **The `coding-progress-ledger` repo is the upstream.** This
  repo is downstream. If a feature column changes name or
  semantics there, this repo's feature registry must bump and
  every existing bundle becomes invalid. `docs/VERSIONING.md`
  formalizes the rule.

---

## Key files to read first (in this order)

1. `TASKS.md` — the plan. Long, but every workstream's status is
   recorded inline.
2. `reports/ESTIMATOR_GO_NO_GO.md` — the verdict.
3. `reports/sign_off_ledger_basic_v0.1.md` — the consumer-facing
   sign-off. If you only read one report, read this.
4. `reports/NOT_READY_FOR_SCHEDULING.md` — the prioritized
   recommendations.
5. `coding_estimator/eval/go_no_go.py` — the gate definition.
   Each P1.* is its own function.
6. `coding_estimator/eval/failure_modes.py` — O1, O5, O7
   definitions.
7. `coding_estimator/labels/registry.py` — `V0_TARGETS`. What
   the estimator predicts.
8. `coding_estimator/checkpoints/features/registry.py` — what
   features exist, on which sources, with what missingness
   semantics.
9. `models/ledger_basic_v0.1/model_card.md` — the public face
   of the v0 estimator.

---

## What I would tell a reviewer in two sentences

The estimator pipeline is end-to-end working and the v0 verdict is
honestly INDETERMINATE — the gates that should fail don't have
enough data to fail, and the gate that does have data (O7 on
swe_agent_pilot) fails. The decision to make is whether the
ledger-feature framing is salvageable by adding the deferred
dynamics group G5, or whether the v0 design needs deeper rethinking
before any more engineering.
