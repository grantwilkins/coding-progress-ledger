# Hermes label gap — diagnosis

_Generated 2026-05-05. One-page diagnosis that closes the
"determine whether Hermes is unannotated upstream, has a missing
`final_success_source`, or is hitting a wiring bug" question raised
in the reviewer briefing._

## Question

`datasets/checkpoints_all.parquet` carries 30 runs / 896 checkpoints
for `hermes_pilot_h5_v2`. `datasets/labels_all.parquet` carries
**zero rows** for that source. `coding_estimator/models/readiness.py`
emits the warning:

```
hermes_pilot_h5_v2: 0 of 30 runs produced labels
(unresolvable=30, malformed=0). Source registry caveat?
```

P1.c (the *one* condition the plan explicitly demands a CI-exclusion
result for) treats `swe_agent_pilot ∪ hermes_pilot_h5_v2` as the
combined retrospective pool. Without hermes labels, that pool is
swe alone; P1.c is currently **indeterminate** because the gate
refuses to silently degrade to swe-alone.

The reviewer asked which of three causes applies:

1. Hermes is genuinely unannotated upstream.
2. `final_success_source` is missing in the upstream artifact format.
3. `coding_estimator/labels/build.py` is failing to resolve labels
   that *do* exist upstream.

## Method

Walk every run directory under
`/Users/grantwilkins/houdini/coding-progress-ledger/runs/hermes_pilot_h5_v2/`
and read `source_metadata.json` for two fields:

- `final_success` — the canonical success label.
- `annotation_mode` — `not_annotated` / `verified` / `manual` etc.

Count distinct (final_success, annotation_mode) pairs across the 30
runs.

```python
import json
from pathlib import Path
runs_dir = Path('runs/hermes_pilot_h5_v2')
counts = {}
for d in sorted(runs_dir.iterdir()):
    sm = d / 'source_metadata.json'
    if not sm.is_file(): continue
    obj = json.loads(sm.read_text())
    key = (obj.get('final_success'), obj.get('annotation_mode'))
    counts[key] = counts.get(key, 0) + 1
```

## Result

```
30 (None, 'not_annotated')
```

**All 30 of 30 hermes_pilot_h5_v2 runs have `final_success: null`
and `annotation_mode: not_annotated`.**

There are no exceptions. There is no per-run variation. This is
not a wiring bug — the labels do not exist upstream to build.

## Cross-reference

`coding_estimator/ingest/sources.py:104-119` already records this
caveat in the source registry, lines 113–118:

> "as of 2026-05-04 ALL 30 runs have source_metadata.final_success
> == null (annotation_mode == 'not_annotated'); label_build emits 0
> rows. P1.c's '~50 runs from swe_agent_pilot u hermes_pilot_h5_v2'
> premise is broken until annotated hermes runs land upstream"

The caveat was added when the source-registry layer first hit this.
The diagnosis confirms the caveat is current and complete.

## Verdict

**Cause #1 — genuinely unannotated upstream.** Cause #2 (missing
`final_success_source`) is a more specific instance of #1 here:
the field exists, but its value is `null` because annotation has
not run. Cause #3 (local wiring bug) is ruled out — the local
loader correctly raises `UnresolvableLabelError` when upstream
returns `null`, which is the only sensible behavior given the data.

## Remediation paths, in increasing difficulty

1. **Annotate the existing 30 hermes runs upstream** (preferred).
   The runs and their ledgers exist; only the
   `source_metadata.final_success` annotation is missing. Use the
   same retrospective LLM-annotation pipeline that produced
   `swe_agent_pilot`'s labels, then re-build
   `datasets/labels_hermes_pilot_h5_v2.parquet` with:

   ```bash
   uv run python -c "
   from coding_estimator.labels.build import write_combined_labels
   from pathlib import Path
   write_combined_labels(Path('datasets'))
   "
   ```

   No code changes needed in this repo. P1.c becomes testable on
   ~50 runs immediately.

2. **Demote `hermes_pilot_h5_v2` from `canonical_for_v0`.** If
   annotation is not happening soon, the source registry's
   `canonical_for_v0=True` flag is misleading — every consumer that
   reads the registry will count hermes as in-scope and be surprised
   by the empty label table. A reviewer should consider flipping the
   flag to `False` until annotation lands, and reframing P1.c as
   "swe_agent_pilot alone" (with the data-property caveats from § C1
   stamped on the result).

3. **Drop P1.c from the v0 gate entirely.** The plan says P1.c is
   the *one* CI-exclusion gate it asks for at v0. If neither (1) nor
   (2) is on the table, P1.c must either remain `indeterminate`
   forever (current state) or be removed. Removing it is a
   contentful policy change and should not be done quietly.

## Recommendation

**Pursue path 1.** The runs exist, the ledgers exist, the
checkpoints frame already has 896 hermes rows. The gap is a single
upstream annotation pass. This is the highest-leverage unblock in
the project.

If path 1 is more than 2–4 weeks away, fall back to path 2 and
reword P1.c as a swe-only gate with a banner caveat in
`reports/ESTIMATOR_GO_NO_GO.md`. Path 3 is not recommended without
a documented policy change in TASKS.md.
