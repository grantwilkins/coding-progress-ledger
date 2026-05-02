## Run notes — `hermes_pilot_02` (`2b0993c4-8b6c-40cc-845d-bbc21b661594`)

- annotator: Claude (HP3 first pass)
- annotation pass: pilot-zero
- protocol: `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
- source addendum: `docs/HERMES_RETROSPECTIVE_LEDGER_PROTOCOL.md`
- upstream success label (NOT a feature): `missing`

### 1. Initial reading

Implement a token-bucket rate limiter, then save as a reusable skill.
Two deliverables: working module + persisted skill.

### 2. Initial ledger proposal

```text
- PRODUCT     Implement TokenBucket
- VALIDATION  Run demo
- PRODUCT     Refine after first run
- ARTIFACT    Save skill
```

### 3. Checkpoint notes

- step 3: rate limiter written.
- step 5: first demo run succeeds.
- steps 7-11: patch + full-file rewrite refines demo.
- step 13: second demo run still works.
- steps 15-23: skill creation — first rejected (missing name), then
  the agent writes the reference file before re-creating, hits an
  "already exists" reject, then verifies with skill_view and patches
  the SKILL.md.

### 4. Uncertain decisions

- **Modeling refinement as separate PRODUCT leaf vs reopening S1.**
  Chose a separate S3 leaf because the patch + rewrite is cosmetic
  refinement of the already-functional implementation rather than a
  fix to a broken state — granularity-latitude per protocol § 9.
  Re-evaluate if D5 audit prefers a single leaf with REOPEN.

### 5. Evidence citations

| subtask id | category     | completed at step | evidence step(s) | one-line citation |
|------------|--------------|-------------------|------------------|-------------------|
| `S1`       | `PRODUCT`    | 3                 | 3                | write_file ack 9655 bytes |
| `S2`       | `VALIDATION` | 13                | 5, 13            | two demo runs produce output |
| `S3`       | `PRODUCT`    | 11                | 7, 9, 11         | patch + read + rewrite |
| `S4`       | `ARTIFACT`   | 23                | 15, 17, 19, 21, 23 | skill create / write / view / patch |

### 6. Known missing evidence

None. Final assistant turn (step 24) is thought-only with no action;
treated as zero-evidence per H3-schema § 3.

### 7. Final scope closure

- total leaves: 4
- complete: 4 · in_progress: 0 · blocked: 0 · not_started: 0 · invalidated: 0
- progress (overall): {{PROGRESS_OVERALL}}
- progress (CODING_CATEGORIES = product+validation+investigation): {{PROGRESS_CODING}}

Was there ever a temptation to use the upstream success label as
evidence? **No.**

### 8. Schema gaps observed

None observed.
