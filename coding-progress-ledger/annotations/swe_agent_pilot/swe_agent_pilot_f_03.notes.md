## Run notes — `swe_agent_pilot_f_03` (`asottile__setup-cfg-fmt-132`)

- annotator: Claude (pilot-zero stress-test, third trace)
- annotation pass: pilot-zero (stress-test for D1 stuck-loop +
  harness-submit refinements)
- protocol: `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
- source addendum: `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`
- upstream success label (NOT a feature): `False`

### 1. Initial reading

The issue is that `configparser` downcases keys in unrelated sections
(e.g. `DJANGO_SETTINGS_MODULE` becomes `django_settings_module` even
when it lives in `[tool:pytest]`, not in a section setup-cfg-fmt
processes). The fix is in `setup_cfg_fmt.py`: configure the
parser to preserve case, or process only the sections it should.

### 2. Initial ledger proposal (written before the walk)

```text
- INVESTIGATION  Reproduce the case-folding behavior
- INVESTIGATION  Locate the configparser configuration in setup_cfg_fmt.py
- PRODUCT        Configure the parser to preserve case (or scope it to known sections)
- VALIDATION     Run tests
- ARTIFACT       Submit
```

The walk's ledger has only the first two leaves and the second is
`blocked`. The remaining three never become discovered work.

### 3. Checkpoint notes

- step 7: `python reproduce.py` succeeds; the repro environment is set
  up. Agent moves on without explicitly verifying the bug from the cat
  output (truncated in normalized trace; agent's transition implies it
  considered repro adequate).
- step 10-17: legitimate investigation. `search_dir`, `open`,
  `search_file 'configparser'`, `goto 156`. The agent has located the
  configparser usage at line 156 of setup_cfg_fmt.py.
- step 18: stuck loop begins. `search_file 'test.ini'` returns
  no matches (test.ini lives in /setup-cfg-fmt, not as source).
- step 22: third iteration of the loop begins; pattern is now
  observably stuck per general protocol § 6.
- steps 22-112: 23 more iterations of the same 4-command cycle.
  No edits, no new files, no query variations.
- step 112: last in-trace step is `goto 156` (not a `submit`).
  Trace ends because the SWE-agent harness force-terminated at
  context exhaustion (`exit_status='submitted (exit_context)'`).

### 4. Uncertain decisions

- **When exactly to mark `blocked`.** Alternatives: step 18 (loop
  start), step 22 (third iteration per the protocol's N≥3 rule),
  step 30 (when the loop is unmistakable to a casual reader), step
  50 (when even a generous observer concedes). Chose step 22 because
  the general protocol explicitly defines the threshold as "third
  iteration begins". Re-evaluate if D5 audit suggests the threshold
  is too lenient or too strict.

### 5. Evidence citations

| subtask id | category        | completed at step | evidence step(s) | one-line citation |
|------------|-----------------|-------------------|------------------|-------------------|
| `S1`       | `INVESTIGATION` | 7                 | 5, 7             | reproduce.py created and run; agent moves on |
| `S2`       | `INVESTIGATION` | (blocked at 22)   | 18-112           | identical 4-command cycle repeats 24x without variation |

### 6. Known missing evidence

- `S2` (locate configparser config) **left at `blocked`**. The agent
  surfaced the right region of the right file (line 156 of
  setup_cfg_fmt.py) but could not pivot the query and never made the
  PRODUCT leaf visible.
- **No PRODUCT leaf.** The trace shows no `edit` or `create` of
  product code anywhere from step 8 onward. `reproduce.py` is the
  only file the agent created, and it is investigation/repro, not
  product. Per SWE-agent addendum § 5 pitfall #7, the non-empty
  `final_diff.patch` (560 chars) reflects reproduce.py, not a fix —
  do **not** treat it as PRODUCT evidence.
- **No VALIDATION leaf.** No pytest, tox, or test mock was run
  after step 6 (which only ran reproduce.py).
- **No ARTIFACT leaf.** Per SWE-agent addendum § 5 pitfall #6,
  `exit_status='submitted (exit_context)'` is harness-forced
  termination at context exhaustion. The agent's last command was
  `goto 156` at step 112, not `submit`. No discovered ARTIFACT work.
- **Hidden-work gap (borderline-discoverable).** The agent has the
  configparser-using code visible at line 156 throughout the loop;
  the issue text names the exact behavior; the agent's mental model
  treated `test.ini` (the artifact) as the problem instead of the
  parser configuration. An honest observer reading the trace can
  describe the missing PRODUCT subtask but cannot point to a step
  where the agent surfaced it as discovered work, so per general
  § 2 it stays hidden in the ledger and is recorded only here.

### 7. Final scope closure

- total leaves: 2
- complete: 1 · in_progress: 0 · blocked: 1 · not_started: 0 · invalidated: 0
- progress (overall): {{PROGRESS_OVERALL}}
- progress (CODING_CATEGORIES = product+validation+investigation): {{PROGRESS_CODING}}

Was there ever a temptation to use `final_success` as evidence? **No.**
Knowing the run failed doesn't change a single ledger event:
investigation was visibly stuck, no PRODUCT/VALIDATION/ARTIFACT
leaves became discovered work.

### 8. Schema gaps observed

None observed — both refinements added to the protocols (general §
6 stuck-loop rule; SWE-agent addendum § 5 pitfall #6
harness-forced-submit; pitfall #7 final_diff.patch caveat) covered
the trace cleanly. f_03 was the input that forced these refinements;
this annotation is the validation that they suffice.
