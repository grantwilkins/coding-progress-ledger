# Hermes parity report (HP4)

Framing per HP2-critic: this is a **feasibility report**, not a
distributional parity claim. N=5 with no upstream success label is
not enough to claim "the channel is stable across sources." It is
enough to claim "the framework runs unchanged on a non-SWE source."

## Acceptance gate (TASKS.md § H_PARITY HP4)

| Gate | Result |
|------|--------|
| `scripts/build_ledger_observation_dataset.py` runs unchanged | ✓ |
| `scripts/build_estimator_checkpoints.py` runs unchanged | ✓ |
| `scripts/build_q_labels.py` runs unchanged (no `final_success` needed) | ✓ |
| `scripts/label_observation_shapes.py` runs unchanged | ✓ |
| No edits to `ledger_progress/core.py` | ✓ |
| No edits to `LedgerEvent` / `Status` / `SubtaskCategory` enums | ✓ |
| `ledger-run check-run` passes for all 5 pilots | ✓ |
| All 5 SubtaskCategory values exercised | ✓ (PRODUCT, VALIDATION, INVESTIGATION, ENVIRONMENT, ARTIFACT) |

**Verdict: HP4 gate PASSES. The framework is source-agnostic in
practice, not just in design.**

## Outputs (existing scripts, zero source edits)

| File | Rows | Notes |
|------|-----:|-------|
| `datasets/hermes_pilot_observations_event.csv` | 61 | per-event observation table |
| `datasets/hermes_pilot_observations_step.csv` | 59 | per-step observation table |
| `datasets/hermes_pilot_shape_labels.csv` | 5 | one row per pilot |
| `datasets/hermes_pilot_estimator_checkpoints.csv` | 59 | W3 checkpoint table |
| `datasets/hermes_pilot_q_labels.csv` | 59 | Q1–Q4 channel-native labels |

## Per-pilot summary

| pilot | category | leaves complete | coding_progress | shape_tags |
|-------|----------|-----------------|----------------:|------------|
| hermes_pilot_01 | Terminal & Coding | 4/4 | 1.000 | (none) |
| hermes_pilot_02 | Terminal & Coding | 3/3 | 1.000 | (none) |
| hermes_pilot_03 | Terminal & Coding | 6/6 | 1.000 | stuck_loop |
| hermes_pilot_04 | Terminal & Coding | 4/4 | 1.000 | (none) |
| hermes_pilot_05 | Terminal & Coding | 4/4 | 1.000 | stuck_loop |

`stuck_loop` fires on the two pilots whose annotations include a
BLOCKED leaf with reason text containing `loop`/`stuck`. The other
three pilots have no observation-channel anomaly visible to W2
under the current rule set.

## Q1 channel-native targets (positive counts at horizon=5)

| target | positives | total |
|--------|----------:|------:|
| `future_progress_drop` | 18 | 59 |
| `product_reopened_after_completion` | 0 | 59 |
| `validation_exposes_new_work` | 1 | 59 |
| `stuck_loop_next_window` | 0 | 59 |
| `submit_without_validation_state` | 0 | 59 |

`product_reopened_after_completion = 0` is consistent with the
annotations: no pilot has a REOPEN_SUBTASK event. `stuck_loop_next_window
= 0` because the BLOCKED-with-loop event in pilot_03 / pilot_05 fires
**at** the checkpoint where `repeated_observation_loop_flag` becomes
true; the W3 mask correctly suppresses the label there. This is the
expected behavior, not a bug.

## What stayed stable across SWE-agent → Hermes

- **Replay engine** unchanged (zero edits to `ledger_progress/core.py`).
- **Subtask category enum** sufficient — no need for new categories.
- **Q1–Q5 channel-native targets** all computed correctly without
  `final_success`. The `submit_without_validation_state` label is
  derived purely from internal ledger state and works on Hermes.
- **W3 estimator checkpoint table** — every feature group populates;
  legacy-no-timestamp path used (Hermes has no wall-clock data).

## What is sparse on Hermes (and why)

- **`label_*` columns in W3** — `label_final_success` is empty (no
  upstream label). `label_success_by_horizon` is `False` for every
  row (because `success is None`). This is correct behavior, not a
  break: the columns are *labels*, models drop them at the schema
  layer per Q5.
- **Shape labels gated on success** — tags like `clean_success`,
  `low_progress_success`, `high_progress_failure` require a known
  outcome to compute. They all read `false` on Hermes pilots. Only
  the success-agnostic tags (`stuck_loop`, `submit_without_validation`,
  etc.) populate. This is by design.

## Acknowledged gaps (pre-flagged by HP2 critic)

### G1. SPLIT rule is unverified in production

Zero of 1,000 cached `kimi` rows had a multi-`<tool_call>` `gpt`
turn. The 5 pilots therefore do not exercise Pitfall H1
(`HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md` § 4). The SPLIT path is
covered by **22 synthetic regression tests** in
`tests/test_normalize_hermes_trace_semantics.py`; the production
path is exercised at SPLIT count = 1 only.

Mitigation if HP scales: include `glm-5.1` in the next sampling
wave (the HP1-saved sample row from `glm-5.1` showed similar
single-call structure, but a wider scan is warranted) **or** sample
from `Agent Tools` / `Multi-Tool` categories where multi-call turns
are more common.

### G2. Sampling pool degenerate

Only 6 of 1,000 inventory rows passed inclusion criteria I1–I5
(`Terminal & Coding` × `kimi` × len ≥ 6 × has-tool_call × unique
id). 5 of 6 became pilots; the sampler's `--seed` parameter was
plumbed but unused. Re-run with `--max-rows 5000` next time, or
relax I1 to include `Repository Tasks` and `File Operations`.

### G3. No upstream success label

This is a feature, not a bug — it forces the channel-vs-outcome
decoupling thesis to bear weight. **Q6 (final-success prediction)
remains N/A on Hermes by definition.** The mission-locked memory
note (channel decoupled from outcome) is preserved.

## Honest claims (per HP2 critic § "phrase HP4 as feasibility")

- ✓ The pipeline runs to completion on Hermes traces without code
  changes.
- ✓ All five `SubtaskCategory` values were exercised across the 5
  pilots.
- ✓ No protocol-breaking pitfalls surfaced beyond H1–H5.
- ✗ N=5 is **not** sufficient to claim "channel features are stable
  across sources." Stability is a distributional claim; this is a
  feasibility claim.

## Reproducibility

Pilot rows are pinned: 5 raw rows committed under
`external_data/hermes/pilot_cache/kimi/{115,128,260,315,501}.json`
(per `.gitignore` exception). Re-run from a clean clone:

```bash
# 1. (no re-streaming required — raw rows are pinned)
uv run python scripts/import_hermes_trace.py \
  --sample-csv external_data/hermes/manifests/hermes_pilot_sample.csv \
  --runs-dir runs/hermes_pilot \
  --raw-cache-dir external_data/hermes/pilot_cache

# 2. annotate from spec
uv run python scripts/annotate_pilots_from_spec.py \
  --spec-dir annotations/hermes_pilot \
  --runs-dir runs/hermes_pilot

# 3. run the four parity pipelines (this report's outputs)
uv run python scripts/build_ledger_observation_dataset.py --runs-dir runs/hermes_pilot ...
uv run python scripts/label_observation_shapes.py runs/hermes_pilot ...
uv run python scripts/build_estimator_checkpoints.py --runs-dir runs/hermes_pilot ...
uv run python scripts/build_q_labels.py --runs-dir runs/hermes_pilot ...
```

## Files

| Artifact | Path |
|----------|------|
| Annotation specs | `annotations/hermes_pilot/hermes_pilot_{01..05}.{json,notes.md}` |
| Run dirs (gitignored) | `runs/hermes_pilot/hermes_pilot_{01..05}/` |
| Parity outputs | `datasets/hermes_pilot_*.csv` + `*.md` |
| This report | `runs/hermes_pilot/HERMES_PARITY_REPORT.md` |
| Tests | `tests/test_normalize_hermes_trace.py`, `test_normalize_hermes_trace_semantics.py`, `test_hermes_inventory.py`, `test_sample_hermes_pilot.py`, `test_import_hermes_trace.py`, `test_hermes_annotations.py` (118 tests total) |
