# Hermes annotation precheck (T1)

> **Note (2026-05-05, post-shipping).** After this precheck shipped, the
> project decided to **defer all of Workstream T**. Hermes HF is not a
> labeled outcome source by design — its published schema does not
> include `final_success`, `verifier_pass`, `eval_log`, or any benchmark
> outcome field. The precheck below is still factually correct (the
> upstream metadata fields are unset and the local loader is not
> broken), but the *implied next step* — annotate `final_success`
> upstream — is no longer the right move. See `TASKS.md § Workstream T`
> banner for the full reframing. Hermes remains a valid
> **process-dynamics** source.

_Generated 2026-05-05. Closes T1 in TASKS.md (Workstream T)._

## Question

Before doing any upstream annotation work, confirm whether
`hermes_pilot_h5_v2` is unannotated **upstream** or whether the local
label loader in this repo is failing to resolve labels that already
exist. T1 is the precheck that decides:

- A. upstream annotation missing → proceed to T2;
- B. local label loader broken → fix loader before T2;
- C. source should be removed from canonical v0 until annotation lands.

## Method

Walk every directory under
`../coding-progress-ledger/runs/hermes_pilot_h5_v2/` and count, per run:

- `ledger.jsonl` present (`prefix-only replayer prerequisite`);
- `source_metadata.json::final_success` not null (canonical success label);
- `source_metadata.json::final_success_source` not `"missing"`;
- `source_metadata.json::annotation_mode` not `"not_annotated"`.

Then run the local loader (`coding_estimator.labels.build.build_source_labels`)
against the source and report the resulting frame shape and per-run stats.

Both checks are run via `uv run python` against the upstream commit
pinned in `datasets/manifests/upstream_commit.json`.

## Result

```text
n_runs                                 = 30
n_ledger_jsonl                         = 30
n_final_success_set                    = 0
n_annotation_mode_not_in_not_annotated = 0
final_success           values: {None: 30}
annotation_mode         values: {'not_annotated': 30}
final_success_source    values: {'missing': 30}
```

Local loader output:

```text
build_source_labels('hermes_pilot_h5_v2') -> df.shape=(0, 0)
SourceLabelStats(
  source_id='hermes_pilot_h5_v2',
  n_runs_total=30,
  n_runs_labeled=0,
  n_runs_unresolvable=30,
  n_runs_malformed=0,
)
```

Key observations:

- **All 30 ledgers exist.** Replay prerequisites are intact.
- **All 30 runs are unannotated upstream.** `final_success` is `null`
  for every run; `final_success_source` is `"missing"` for every run;
  `annotation_mode` is `"not_annotated"` for every run.
- **Local loader is correct.** The loader correctly classifies all 30
  runs as `unresolvable` (not `malformed`), which is the right behavior
  given upstream returns `null`. There is no local wiring bug.
- **Cause #2 (`final_success_source` missing) is a specific instance of
  cause #1 here.** The field exists but is set to `"missing"` because
  no annotation pass has run.

This re-confirms `reports/HERMES_LABEL_DIAGNOSIS.md` (2026-05-05) at
the current upstream commit. The HP6 heuristic auto-annotator
(`auto_annotate_hermes`) has run on these traces — it produced the
ledger leaves and `annotation_quality.json` — but it is by design a
*structural* annotator and does not assign `final_success`. Hermes
traces ship no upstream eval logs (`test_output.txt` is a placeholder),
so `final_success` cannot be machine-derived from the trace alone.

## Verdict

**Path A — upstream annotation missing.** Proceed to T2.

The local loader is correct. The ledgers are intact. The only gap is
upstream `source_metadata.final_success` (and the corresponding
`final_success_source` and `annotation_mode` fields). T2 must
materialize these labels in the upstream `coding-progress-ledger`
repository.

Path B (loader broken) is ruled out: `n_runs_malformed=0` and
`unresolvable` is the correct classification for `final_success=null`.

Path C (drop `hermes_pilot_h5_v2` from canonical_for_v0) is a
contentful policy change and is not recommended until T2 is attempted
and either lands or is determined to be more than 2–4 weeks away
(see `reports/HERMES_LABEL_DIAGNOSIS.md` § Recommendation).

## Inputs verified

```text
../coding-progress-ledger/runs/hermes_pilot_h5_v2/hermes_pilot_h5_*/ledger.jsonl
../coding-progress-ledger/runs/hermes_pilot_h5_v2/hermes_pilot_h5_*/source_metadata.json
../coding-progress-ledger/runs/hermes_pilot_h5_v2/hermes_pilot_h5_*/annotation_quality.json
../coding-progress-ledger/annotations/hermes_pilot/  (annotation pattern reference)
```

`reports/HERMES_LABEL_DIAGNOSIS.md` is unchanged and remains the
companion document; this precheck just re-runs the counts at the
T1 acceptance gate so the v1 plan can advance to T2 without ambiguity.

## Cross-references

- `reports/HERMES_LABEL_DIAGNOSIS.md` — original diagnosis (2026-05-05).
- `coding_estimator/ingest/sources.py` — source registry caveat for `hermes_pilot_h5_v2`.
- `coding_estimator/labels/build.py` — `UnresolvableLabelError` path.
- `TASKS.md § Workstream T` — T1 acceptance gate; T2 acceptance criteria.
