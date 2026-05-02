# Hermes parity scale-up report (HP5)

Framing: HP5 scales N from 5 (HP4) to **30** balanced across categories
and configs. Annotation is **deterministic heuristic**, not human
judgment. Numbers below reflect the heuristic; comparisons against HP4
and SWE-agent are between heuristic-Hermes and human-annotated
counterparts and must be read through that lens.

## TL;DR

- N = **30** pilots, all four downstream pipelines run **unchanged**.
- `ledger-run check-run` PASSES on all 30 pilots.
- G1 (SPLIT in production) — **resolved**: 23 multi-tool-call gpt
  turns across HP5 (vs 0 in HP4), 77 SPLIT-derived assistant steps.
- G2 (degenerate sampling pool) — **resolved**: pool grew 6 → **3,223**.
- G3 (no upstream `final_success`) — **still N/A by design**; Q6 and
  outcome-gated shape tags remain N/A.
- Heuristic vs HP4 human annotation on the 5 overlap pilots:
  category overlap ≥ 50% on every pilot (test-enforced); leaf counts
  diverge mildly (heuristic over-splits VALIDATION when re-runs share
  intent but differ in command).

## Heuristic-annotation caveat (READ FIRST)

`scripts/auto_annotate_hermes.py` deterministically maps tool calls to
SubtaskCategory using the table in
`docs/HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md` § 1, then groups
**consecutive same-category** assistant steps into one leaf. Completion
is non-error tool response on the leaf's last step; BLOCKED is 3+
identical observation bodies (Pitfall H3) **or** an explicit error
key. This is *not* human annotation. Specifically the heuristic:

- Cannot infer intent from context — a `bash` shell call with `pytest`
  is VALIDATION, but a debugger script that *imports* pytest is also
  classified VALIDATION. Human annotators would distinguish.
- Splits a single intentional VALIDATION subtask across multiple
  leaves whenever the agent re-runs with different flags after a
  category change in between (overcounts leaves).
- Misses ENVIRONMENT subtasks that human annotators infer from a
  failed import (e.g., HP4 pilot_03 had ENVIRONMENT for missing
  `pytest`; the heuristic classified the same step as BLOCKED
  VALIDATION because the response had `exit_code: 0` but stdout said
  "No module named pytest" — no error key, but human knew).
- Does NOT cite `<think>` content (Pitfall H2 enforced by test).

## Sample composition

Source: HuggingFace `lambda/hermes-agent-reasoning-traces`, configs
`kimi` (1,500 rows scanned) and `glm-5.1` (3,000 rows scanned).
Combined inventory: 4,500 rows. Filter: category ∈
{Terminal & Coding, Repository Tasks, File Operations}, config ∈
{kimi, glm-5.1}, trajectory_length ≥ 6.

| (category, config)              | pilots |
|---------------------------------|-------:|
| (Terminal & Coding, kimi)       | 6 |
| (Terminal & Coding, glm-5.1)    | 6 |
| (Repository Tasks, kimi)        | 6 |
| (Repository Tasks, glm-5.1)     | 6 |
| (File Operations, glm-5.1)      | 6 |
| (File Operations, kimi)         | 0 (none in pool — kimi has 0 rows of File Operations after filtering) |

Trajectory length: min 13 / median 22 / max 54.

Reproducibility pin: `external_data/hermes/manifests/hermes_pilot_h5_sample.csv`.
Inventory CSVs: `hermes_inventory_kimi_3k.csv` (1,500 rows; REST
returned HTTP 429 above 1,500), `hermes_inventory_glm_3k.csv` (3,000).

## Acceptance gate

| Gate | Result |
|------|--------|
| `scripts/build_ledger_observation_dataset.py` runs unchanged | PASS |
| `scripts/build_estimator_checkpoints.py` runs unchanged | PASS |
| `scripts/build_q_labels.py` runs unchanged | PASS |
| `scripts/label_observation_shapes.py` runs unchanged | PASS |
| No edits to `ledger_progress/core.py` | PASS |
| No edits to `LedgerEvent` / `Status` / `SubtaskCategory` | PASS |
| `ledger-run check-run` passes on all 30 pilots | PASS |
| Heuristic vs human overlap ≥ 50% on HP4 pilots | PASS (test) |

## Outputs

| File | Rows |
|------|-----:|
| `datasets/hermes_pilot_h5_observations_event.csv` | 370 |
| `datasets/hermes_pilot_h5_observations_step.csv`  | 370 |
| `datasets/hermes_pilot_h5_estimator_checkpoints.csv` | 370 |
| `datasets/hermes_pilot_h5_q_labels.csv`           | 370 |
| `datasets/hermes_pilot_h5_shape_labels.csv`       | 30 |

## Distributional comparison

### Coding-progress (heuristic)

| corpus | n | min | median | mean | max |
|--------|--:|----:|-------:|-----:|----:|
| HP4 (Hermes, human, N=5)         | 5  | 1.000 | 1.000 | 1.000 | 1.000 |
| **HP5 (Hermes, heuristic, N=30)** | 30 | 0.500 | 0.800 | 0.792 | 1.000 |
| SWE-agent pilot (human, N=18)    | 18 | 0.600 | 1.000 | 0.951 | 1.000 |

HP4's all-1.000 distribution is a small-N artifact (5 carefully chosen
Terminal & Coding traces, all clean). HP5 spreads progress across
[0.50, 1.00] because the heuristic blocks-on-first-error rather than
reasoning about whether the agent recovered. This is a *heuristic
artifact*, not a population claim about Hermes agents.

Coding-progress buckets across HP5 (n=30):

```
[1.00]        9
(0.75, 1.00)  7
(0.50, 0.75] 10
(0.25, 0.50]  4
[0.00, 0.25]  0
```

### SubtaskCategory distribution (per-leaf)

| category      | HP4 (N=28) | HP5 (N=170) |
|---------------|-----------:|------------:|
| product       | 12 (43%)   | 46 (27%) |
| validation    | 6 (21%)    | 58 (34%) |
| investigation | 3 (11%)    | 48 (28%) |
| artifact      | 5 (18%)    | 9  (5%)  |
| environment   | 2 (7%)     | 9  (5%)  |

The heuristic over-weights VALIDATION (default classification for
unknown `bash`/`shell` invocations whose command does not match any
INVESTIGATION/PRODUCT/ENVIRONMENT regex). It under-weights
ARTIFACT because Hermes tasks rarely emit explicit `submit_answer`
calls; many close on `skill_manage` (which we classified as ARTIFACT
via the terminal-tool rule).

### Status distribution (per-leaf)

| status   | HP4   | HP5      |
|----------|------:|---------:|
| complete | 26    | 122 (72%) |
| blocked  | 2     | 48  (28%) |

The heuristic's "BLOCKED on first error" rule is more aggressive than
human annotators, who frequently mark a leaf complete after the agent
recovers from an error within the same intent block.

### Shape-tag prevalence (HP5, n=30)

| tag                          | true | notes |
|------------------------------|-----:|-------|
| `stuck_loop`                 | 22   | fires on any BLOCKED leaf — heuristic over-flags |
| `no_validation_frontier`     | 6    | pilots with zero VALIDATION leaves |
| `submit_without_validation`  | 0    | requires both ARTIFACT-complete + no VALIDATION-complete |
| `validation_induced_reopen`  | 0    | requires REOPEN_SUBTASK; heuristic emits none |
| `clean_success`              | 0    | requires `final_success=True`; Hermes has none |
| `high_progress_failure`      | 0    | requires `final_success`; N/A |
| `low_progress_success`       | 0    | requires `final_success`; N/A |
| `hidden_work_gap`            | 0    | evidence-phrase trigger; never matched |
| `nonmonotone_recovery`       | 0    | requires reopen + success; N/A |

`stuck_loop` over-firing is the same W2 legacy-name issue called out in
HP4 (the tag fires on any BLOCKED leaf, regardless of whether the
agent looped). HP5's heuristic makes this far more visible because
it issues BLOCKED on transient errors that human annotators would
fold into a recovering subtask.

### Q1 channel-native targets (positives at horizon=5)

| target                                    | positives | total |
|-------------------------------------------|----------:|------:|
| `future_progress_drop`                    | 0 | 370 |
| `product_reopened_after_completion`       | 0 | 370 |
| `validation_exposes_new_work`             | 0 | 370 |
| `stuck_loop_next_window`                  | 0 | 370 |
| `submit_without_validation_state`         | 0 | 370 |

All zero. This is consistent with the heuristic emitting:

- no REOPEN_SUBTASK events (so reopens / progress drops cannot fire),
- BLOCKED at the same checkpoint where loop-detection would mask the
  next-window label (HP4 saw the same suppression for pilot_03/05),
- no ARTIFACT-complete + missing VALIDATION-complete coincidences.

This says **the heuristic does not synthesize the kind of
mid-trajectory state changes that Q1's channel-native targets key on**.
A human annotator could plausibly insert REOPEN events; the heuristic
does not.

### Leaves per pilot (HP5)

```
distribution: 1 1 1 1 1 2 2 4 4 4 4 5 5 5 5 5 5 6 6 7 7 7 8 8 8 9 11 11 12 15
min=1 median=5 max=15 mean=5.7
```

Six pilots produced single-leaf annotations. Manual spot-check shows
these are short traces dominated by one tool category (e.g., five
`patch` calls in a row → one PRODUCT leaf), or multi-step traces where
every assistant step was the same tool family.

## G1 status — SPLIT in production: RESOLVED

HP4: 0 multi-tool-call gpt turns across 5 pilots.

HP5: **23 multi-tool-call gpt turns** across 30 pilots (4,500 rows of
inventory scanned), producing **77 SPLIT-derived assistant steps**.
The Pitfall H1 SPLIT path is now exercised in production at non-trivial
volume. The 22 synthetic regression tests in
`tests/test_normalize_hermes_trace_semantics.py` are no longer the only
SPLIT coverage.

## G2 status — sampling pool: RESOLVED

| metric              | HP4 | HP5 |
|---------------------|----:|----:|
| inventory rows scanned | 1,000 | 4,500 |
| rows after filters  | 6 | 3,223 |
| rows used as pilots | 5 | 30 |
| pool oversample factor | 1.2× | 107× |

Sampling balanced across (category × config) buckets. Five of six
target buckets filled fully (6 each); File Operations × kimi has zero
rows in the kimi inventory after filtering.

## G3 status — upstream success: STILL N/A

Hermes ships no `final_success` field. All outcome-gated shape tags
(`clean_success`, `high_progress_failure`, `low_progress_success`,
`nonmonotone_recovery`) report `false` for every pilot. Q6 (outcome
prediction) is structurally undefined. This is the channel-vs-outcome
decoupling thesis bearing weight, not a bug.

## Heuristic vs HP4 human annotation — overlap pilots

The auto-annotator was re-run on `runs/hermes_pilot/hermes_pilot_{01..05}/`
and category-overlap was computed against the human specs in
`annotations/hermes_pilot/`. Test
`tests/test_auto_annotate_hermes.py::test_overlap_with_human_annotation`
enforces ≥ 50% category-multiset overlap per pilot.

Per-pilot disagreements observed:

| pilot | human leaves | heuristic leaves | notable diff |
|-------|-------------:|-----------------:|--------------|
| pilot_01 | 5 | 5 | heuristic split VALIDATION 1×→2× (terminal-flag matrix) |
| pilot_02 | 4 | 7 | heuristic added INVESTIGATION + extra PRODUCT/VALIDATION leaves; over-segmented |
| pilot_03 | 8 | 11 | heuristic missed ENVIRONMENT (missing `pytest`); split VALIDATION 2×→4× |
| pilot_04 | 5 | 5 | heuristic split ARTIFACT 1×→2× (skill_manage retry) |
| pilot_05 | 6 | 6 | heuristic missed ENVIRONMENT, added VALIDATION (off by one) |

**Where the heuristic disagreed with human annotations** (failure
modes flagged for the next iteration):

1. **Missed ENVIRONMENT category from indirect signals.** Human
   annotators mark a step ENVIRONMENT when an agent attempts an action
   whose failure implies an environment problem (e.g., "No module
   named pytest" on stdout with `exit_code: 0`). The heuristic relies
   only on the tool name and the JSON `error` key.
2. **Over-splits VALIDATION across re-runs.** The heuristic groups by
   tool category, so a sequence
   `validation → investigation → validation` produces three leaves;
   humans collapse this into one ongoing VALIDATION subtask when the
   investigation was a debugging detour.
3. **Treats `skill_manage` as ARTIFACT.** Per Pitfall H4,
   `skill_manage` is the Hermes terminal tool of choice. Humans mark
   the *successful* `skill_manage` complete and the failed one as
   internal retry, both under one ARTIFACT leaf. The heuristic emits
   two adjacent ARTIFACT leaves on retry.

## Honest claims

What HP5 supports:

- The framework runs unchanged on **30** Hermes pilots across **5**
  (category × config) buckets — distributional parity is now
  *defensible* in shape, not just feasibility.
- The SPLIT-rule path (Pitfall H1) is exercised in production at
  N=23 multi-call turns, not zero.
- The heuristic auto-annotator is deterministic (test-enforced), cites
  no `<think>` content (test-enforced), maintains step-monotone
  ledgers (test-enforced), and overlaps with human annotation at
  ≥ 50% on every HP4 pilot.

What HP5 does NOT support:

- Any claim that the *agent population* shows X% stuck-loop /
  Y% submit-without-validation. The numbers are heuristic-driven.
  `stuck_loop = 22/30` is a heuristic artifact (fires on first
  BLOCKED), not an agent-behavior claim.
- Outcome prediction (Q6) — Hermes has no labels.
- Any cross-source comparison of process-shape rates between HP5 and
  SWE-agent that does not flag the human-vs-heuristic asymmetry.
- Quality of category assignment on individual leaves at the level
  human annotation provides; spot checks find ≥ 1 mismatch per
  overlap pilot.

## Failures and surprises

- **HF datasets-server rate-limited at 1,500 kimi rows** (HTTP 429
  on the second 50-row page after a fresh 429); the kimi inventory
  was dialed back from 3,000 → 1,500 per the spec's fallback.
- **kimi has zero `File Operations` rows after the min-length filter**
  in the first 1,500. The (File Operations × kimi) bucket is empty;
  HP5 still hits N=30 because the round-robin sampler topped up other
  buckets evenly.
- **All Q1 targets fire 0 positives** (370 checkpoint-rows). This is
  not a pipeline failure — it correctly reflects that the heuristic
  emits no REOPEN events and no submit-without-validation states. A
  follow-up should either inject REOPEN events when humans would, or
  acknowledge that Q1 needs human annotations.
- **`stuck_loop` fires on 22/30 pilots.** Same W2-naming issue
  flagged in HP4: the tag fires on any BLOCKED leaf. Heuristic
  blocks-on-first-error makes this far more visible. Tag name needs
  retiring or re-spec'ing in a future workstream.

## Reproducibility

```bash
# Step 1 — wider inventory (rate-limit aware: kimi capped at 1,500).
uv run python scripts/hermes_inventory.py --config kimi --max-rows 1500 \
  --out-csv external_data/hermes/manifests/hermes_inventory_kimi_3k.csv \
  --cache-dir external_data/hermes/pilot_cache --progress
uv run python scripts/hermes_inventory.py --config glm-5.1 --max-rows 3000 \
  --out-csv external_data/hermes/manifests/hermes_inventory_glm_3k.csv \
  --cache-dir external_data/hermes/pilot_cache --progress

# Step 2 — balanced sampler (N=30 across 3 categories x 2 configs).
uv run python scripts/sample_hermes_pilot_v2.py \
  --inventory-csv external_data/hermes/manifests/hermes_inventory_kimi_3k.csv \
  --inventory-csv external_data/hermes/manifests/hermes_inventory_glm_3k.csv \
  --out-csv external_data/hermes/manifests/hermes_pilot_h5_sample.csv \
  --n-pilots 30

# Step 3 — import pre-annotation artifacts.
uv run python scripts/import_hermes_trace.py \
  --sample-csv external_data/hermes/manifests/hermes_pilot_h5_sample.csv \
  --runs-dir runs/hermes_pilot_h5 \
  --raw-cache-dir external_data/hermes/pilot_cache

# Step 4 — heuristic auto-annotation.
uv run python scripts/auto_annotate_hermes.py --runs-dir runs/hermes_pilot_h5

# Step 5 — four parity pipelines (unchanged from HP4).
uv run python scripts/build_ledger_observation_dataset.py \
  --runs-dir runs/hermes_pilot_h5 \
  --output-csv datasets/hermes_pilot_h5_observations_event.csv \
  --output-event-csv datasets/hermes_pilot_h5_observations_event.csv \
  --output-step-csv datasets/hermes_pilot_h5_observations_step.csv \
  --summary-md datasets/hermes_pilot_h5_observations_summary.md

uv run python scripts/label_observation_shapes.py runs/hermes_pilot_h5 \
  --csv datasets/hermes_pilot_h5_shape_labels.csv \
  --report datasets/hermes_pilot_h5_shape_report.md

uv run python scripts/build_estimator_checkpoints.py \
  --runs-dir runs/hermes_pilot_h5 \
  --step-csv datasets/hermes_pilot_h5_observations_step.csv \
  --shape-labels datasets/hermes_pilot_h5_shape_labels.csv \
  --out-csv datasets/hermes_pilot_h5_estimator_checkpoints.csv \
  --out-summary datasets/hermes_pilot_h5_estimator_checkpoints_summary.md

uv run python scripts/build_q_labels.py \
  --runs-dir runs/hermes_pilot_h5 \
  --checkpoint-csv datasets/hermes_pilot_h5_estimator_checkpoints.csv \
  --out-csv datasets/hermes_pilot_h5_q_labels.csv
```

## Files

| Artifact | Path |
|----------|------|
| Sample manifest          | `external_data/hermes/manifests/hermes_pilot_h5_sample.csv` |
| Run dirs (gitignored)    | `runs/hermes_pilot_h5/hermes_pilot_h5_{001..030}/` |
| Auto-annotator           | `scripts/auto_annotate_hermes.py` |
| Sampler v2               | `scripts/sample_hermes_pilot_v2.py` |
| Tests                    | `tests/test_auto_annotate_hermes.py` (12 tests, parameterized to 34) |
| This report              | `runs/hermes_pilot_h5/HERMES_H5_REPORT.md` |
