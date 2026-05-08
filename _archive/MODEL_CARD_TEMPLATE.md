# Model card template

Every saved estimator under `models/<estimator_id>/` ships a
`model_card.md` that follows this layout. The schema in
`schemas/model_card_schema.json` is the machine-readable contract;
this document is the reader-facing layout.

A consumer (downstream scheduler, audit tool, sign-off reviewer)
should be able to answer the following questions from the card alone:

1. What does this estimator predict, and on what units?
2. What was it trained on, when, against which commit?
3. What are its calibration statistics on the headline holdout?
4. Is it safe to drive control actions?
5. Has it been adversarially probed (Workstream O)? With what result?
6. Does it pass the v0 go/no-go gate (Workstream P)?

If any of those answers is missing, the card is invalid.

---

## `<estimator_id>`

## Intended use

- Bullet list of supported use cases.
- At least one bullet required.

## Non-use cases

- Explicit list of contexts where this estimator must not be consumed.
- Always include "control / scheduling / modulation" unless P1 + O1
  + O5 + O7 all pass with margin and `not_safe_for_control = false`.

## Not safe for control

- `true` or `false`. Default `true`.

## Training data

- canonical sources: comma-separated list of source ids.
- inputs: paths to the checkpoint and label parquet files used.
- n_runs / n_checkpoints: actual size of the training frame.
- commit_sha: short git sha of the build commit.

## Source versions

- one bullet per source: `<source_id>: <source_protocol_version>`.

## Features

- groups: list of feature groups consumed.
- (optional) ablations performed.

## Target definitions

- one bullet per target with horizon units, horizon value,
  run-constant flag.

## Split protocol

- headline metrics: scheme, seed.
- diagnostics: list of additional schemes evaluated.

## Calibration status

- one bullet per target with Brier, ECE, AUROC, n_checkpoints, and
  the calibration method (`raw`, `platt`, `isotonic`,
  `source_isotonic`, or `constant`).

## Failure-mode results (Workstream O)

- O1 (progress-overconfidence): outcome + metric.
- O5 (source-leakage): outcome + metric.
- O7 (timeout-bias): outcome per source + Brier deltas.

## Go/no-go gate (Workstream P)

- verdict: `pass` / `fail` / `indeterminate`.
- report_path: pointer to `reports/ESTIMATOR_GO_NO_GO.md`.

## Known limits

- prose bullets describing each known limit, ideally with the
  pointer to the test or report that surfaced it.

---

## Conformance

A model card is valid iff `schemas/model_card_schema.json` validates
its JSON sidecar (`models/<estimator_id>/model_card.json`). The
markdown is the human-readable face; the JSON is the source of truth.
