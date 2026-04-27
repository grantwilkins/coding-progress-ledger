# Ledger Audit

Scope: the eight completed task runs under `task_1_*` through `task_8_*`.
Artifacts reviewed for each run: `task.md`, `ledger.jsonl`, `progress.csv`,
`final_diff.patch`, `test_output.txt`, and `run_notes.md`.

Audit standard:

- Externally checkable means a reader can verify the subtask against code,
  tests, logs, diffs, docs, or exported artifacts without trusting narrative.
- Concrete evidence means completion evidence points to a specific artifact,
  test result, diff, command output, or documented decision.
- A progress drop is justified only when the active discovered-work denominator
  grows, completed work is reopened, a vague task is split, or invalidation
  changes active leaf work for a reason visible in artifacts.

## Overall Findings

The suite is directionally strong: every completed run has a replayable ledger,
passing final tests, and at least one non-monotonic event. The best examples are
Task 4 and Task 7, where drops correspond to genuinely expanded validation
surfaces.

The recurring weak spots are:

- Artifact bookkeeping is sometimes modeled as active coding progress, which
  can create drops that are less research-relevant than product or validation
  discovery.
- Some completion evidence cites test names or narrative rather than the exact
  captured command output that proves the claim.
- A few invalidations look like preloaded escape hatches rather than discovered
  work, especially environment/tooling branches.
- Several runs reach final progress 1.00 only after artifact-export subtasks,
  not strictly after product validation. That is acceptable for run completion,
  but it should be distinguished from code-task completion.

## Task 1: Parser Timezone Offset

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Create parser repo | Yes | Concrete: repo files exist. |
| S2 Write parser tests | Yes | Concrete: test file has claim and cases. |
| S3 Confirm baseline compact failure | Mostly | Concrete if baseline output is preserved; current `test_output.txt` only shows final pass. |
| S4 Patch parser implementation | Yes through children | Parent completion is redundant after split. |
| S4.1 Accept compact positive HHMM | Yes | Concrete: diff and tests cover `+0530`. |
| S4.2 Accept compact negative HHMM | Yes | Concrete: tests cover `-0330`. |
| S5 Export artifacts | Yes | Concrete artifact presence, but not product behavior. |
| S6 Preserve colon behavior | Yes | Concrete: final tests include `+05:30`. |

Progress drops:

- Step 7 to 8: justified. Adding S6 captures preservation of old colon behavior
  after compact syntax work.
- Step 9 to 10: justified. Splitting S4 into positive and negative compact
  cases exposes two plausible failure modes.
- Step 14 to 15: weak. Reopening artifact export for `summary.json` metrics is
  checkable, but it is bookkeeping rather than discovered coding work.

Forced operations: the S5 reopen appears somewhat forced because summary metric
generation is a suite-artifact concern, not a parser-work discovery.

Final 1.00 after validation: yes, but the final transition is artifact
completion, not parser validation.

Suggested fixes:

1. Preserve the initial failing pytest output for `+0530` and `-0330` in
   `test_output.txt` or a separate baseline log.
2. Move summary/artifact export outside the active coding ledger or mark it as
   a separate run-management category.
3. Avoid completing split parent S4 after children; the children are the
   auditable leaves.

## Task 2: CLI Output Flag

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Define output contract | Yes | Concrete: `task.md`, tests, and notes state the contract. |
| S2 Create buggy baseline | Yes | Concrete: baseline commit referenced. |
| S3 Test regressions | Yes through children | Split leaves are checkable. |
| S3.1 Default stdout | Yes | Concrete test exists and final pytest passes. |
| S3.2 File output | Yes | Concrete test exists and final pytest passes. |
| S3.3 Dash stdout sentinel | Yes | Concrete test exists and final pytest passes. |
| S4 Implement output behavior | Yes | Concrete diff and tests. |
| S5 Assemble artifact bundle | Yes | Concrete artifact presence, but not product behavior. |
| S9 Install pytest | Yes as an environment observation | Weak: invalidated by using existing venv. |

Progress drops:

- Step 4 to 5: justified by splitting S3 into three destination-specific tests.
- Step 7 to 8: justified by reopening implementation after discovering
  `--output -` behavior.
- Step 10 to 11: weak. Adding then invalidating pytest installation is an
  environment branch, not core coding work.

Forced operations: S9 looks forced or at least over-modeled. It is reasonable
to note in `run_notes.md`, but it should not affect discovered product work.

Final 1.00 after validation: yes; final pytest validates all output modes
before artifact completion.

Suggested fixes:

1. Keep environment setup observations in notes unless they block the task.
2. Add baseline failing output for ignored `--output` to strengthen the before
   and after comparison.
3. Make `S5` explicitly a run-artifact subtask so it is not confused with code
   progress.

## Task 3: Config Error Type

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Create inconsistent exception repo | Yes | Concrete: diff and baseline output show `ValueError` sites. |
| S2 Add ConfigError tests | Yes | Concrete: tests and baseline failures in `test_output.txt`. |
| S3 Patch loader consistently | Yes through children | Reopen/split makes the parent auditable. |
| S3.1 Missing-key ConfigError | Yes | Concrete: final code and tests. |
| S3.2 Timeout-type ConfigError | Yes | Concrete: final code and tests. |
| S4 Run tests | Yes | Concrete: final pytest output. |
| S5 Export artifacts | Yes | Weak wording: evidence says ledger export was pending. |

Progress drops:

- Step 4 to 5: justified. A second `ValueError` site remained after the first
  fix, so reopening and splitting S3 reflects real incompleteness.

Forced operations: none in the exception work. S5 is awkward because artifact
export evidence describes a pending final side effect.

Final 1.00 after validation: mostly. Validation completes before the artifact
subtask, so code correctness reaches evidence-backed completion before final
run completion.

Suggested fixes:

1. Change S5 evidence to cite the actual exported artifact list after export,
   not "pending as final side effect."
2. Include a short diff excerpt or line references in run notes for both fixed
   raise sites.
3. Keep the parent S3 incomplete until after children pass rather than marking a
   partial one-site patch complete first.

## Task 4: CSV Messy Aggregation

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Clean repeated-user regression | Yes | Concrete test exists. |
| S2 Sum repeated user rows | Yes | Concrete implementation diff. |
| S3 Compare sample input and expected output | Yes | Concrete sample files exist. |
| S4 Validate behavior | Yes through children | Split leaves are checkable. |
| S4.1 Clean-row regression | Yes | Concrete final pytest output. |
| S4.2 Messy-row regression | Yes | Concrete test covers whitespace and blanks. |
| S4.3 Row-order determinism | Yes | Concrete permutation-style test. |
| S5 Normalize whitespace | Yes | Concrete diff and test. |
| S6 Blank amounts as zero | Yes | Concrete diff and test. |
| S7 Deterministic user order | Yes | Concrete diff and test. |

Progress drops:

- Step 4 to 5: justified. Messy input introduces three new active requirements.
- Step 5 to 6: justified. The broad validation task is split into three
  independently checkable leaves.

Forced operations: none apparent. This is one of the cleanest non-monotonic
examples.

Final 1.00 after validation: yes. There is no separate artifact-export leaf in
the ledger, and completion follows the final validation leaves.

Suggested fixes:

1. Add a captured baseline failing test run for messy input.
2. Make the sample `expected_output.csv` include the blank-amount user case or
   add a note that the broader test, not the sample file, covers it.
3. State in `task.md` whether blank user IDs should be rejected or ignored.

## Task 5: Reset-State Reducer

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Scaffold reducer repo | Yes | Concrete repo files and baseline commit. |
| S2 Write reducer tests | Mostly | Tests exist, but initial failing output is not preserved in `test_output.txt`. |
| S3 Fix reset behavior | Yes through children | Reopen/split maps to separate leaks. |
| S3.1 Reset submitted/derived state | Yes | Concrete final diff and tests, but intermediate pass is not captured. |
| S3.2 Reset validation error | Yes | Concrete final diff and tests. |
| S4 Capture test output and diff | Yes | Concrete artifacts. |
| S5 Export ledger/progress | Yes | Concrete artifacts. |
| S6 Document run | Yes | Concrete docs. |
| S7 Add TypeScript tooling if needed | Checkable but speculative | Invalidated by JS `node:test` decision. |

Progress drops:

- Step 14 to 15: justified. Reopening reset after validation error remains is
  a real incompleteness signal.
- Step 15 to 16: justified. Splitting reset into submitted/derived and
  validation-error leaves is meaningful.

Forced operations: S7 is somewhat forced because it was speculative initial
work rather than discovered work. It was invalidated cleanly, but should not be
part of the active denominator unless tooling actually blocks progress.

Final 1.00 after validation: yes, with final progress also depending on
artifact/documentation leaves.

Suggested fixes:

1. Preserve the initial failing `npm test` output and the intermediate failure
   after only submitted state was fixed.
2. Remove S7 from the initial ledger and mention the JS-vs-TS decision only in
   notes unless it becomes blocking.
3. Split artifact/documentation work from reducer progress in the summary.

## Task 6: Async Stale Result

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Out-of-order completion test | Yes | Concrete controlled-fetcher tests. |
| S2 Preserve newest result | Yes | Concrete diff and final test. |
| S3 Keep loading tied to newest request | Yes, but status timing is weak | It is marked complete before implementation is fixed, using a test-discovery evidence string. |
| S4 Run validation | Yes through children | Split leaves are checkable. |
| S4.1 Newest-result validation | Yes | Concrete final pytest output. |
| S4.2 Loading-state validation | Yes | Concrete final pytest output. |
| S4.3 Plugin-free pytest validation | Yes | Concrete final pytest output. |
| S5 Track request identity | Yes | Concrete diff. |
| S9 Replace pytest-asyncio marker | Yes | Concrete diff and final pytest output. |

Progress drops:

- Step 3 to 4: partly justified, but noisy. S5 is real newly discovered work,
  while reopening S3 immediately after marking it complete suggests the earlier
  completion was premature.
- Step 4 to 5: justified. Validation splits into result, loading, and test
  harness leaves.
- Step 5 to 6: justified as run self-containment work, but it is test harness
  work rather than product behavior.

Forced operations: the S3 complete-then-reopen pattern appears forced. The
evidence at first completion proves the loading contract was discovered, not
that loading behavior was implemented.

Final 1.00 after validation: yes. All three validation leaves complete only
after final pytest passes.

Suggested fixes:

1. Rename the early S3 completion to a test-discovery subtask, or keep S3 active
   until the implementation passes.
2. Preserve the failed pytest output that showed `pytest-asyncio` was missing.
3. Add a stale-failure test so the "error state" claim in `task.md` is directly
   validated.

## Task 7: Refactor Validation Split

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Create invoice repo | Yes | Concrete repo files. |
| S2 Implement long public function | Yes | Concrete initial tests and baseline output. |
| S3 Refactor into helpers | Yes | Concrete diff and targeted test output. |
| S4 Validate behavior/API | Yes through children | Split leaves are strong. |
| S4.1 Targeted tests | Yes | Concrete test output. |
| S4.2 Broader regression tests | Yes | Concrete regression tests and output. |
| S4.3 API compatibility checks | Yes | Concrete signature/result-shape tests. |
| S5 Export artifacts | Yes | Concrete artifacts, but evidence wording is weak. |

Progress drops:

- Step 5 to 6: justified. Splitting vague validation into targeted,
  regression, and API compatibility checks is exactly the mutable discovered
  work this suite should demonstrate.

Forced operations: the broad S4 completion before split is borderline because
the notes say validation was deliberately vague. The split itself is strong,
but the pre-split "complete" status should be treated as an initial validation
pass, not final validation.

Final 1.00 after validation: yes, though artifact export is also in the active
work set.

Suggested fixes:

1. Rename S4 to "Run targeted validation pass" before adding a separate
   validation-breadth subtask.
2. Make S5 evidence cite actual artifact paths after export.
3. Add a small behavior fixture showing old and refactored outputs match on the
   same hand-checked invoice.

## Task 8: Package Import Failure

Subtask checkability:

| Subtask | Externally checkable? | Evidence quality |
| --- | --- | --- |
| S1 Create buggy package fixture | Yes | Concrete baseline commit. |
| S2 Add import/execution tests | Yes through children | Split leaves are checkable. |
| S2.1 `python -m` behavior | Yes | Concrete command output. |
| S2.2 Direct test invocation | Yes | Concrete unittest output. |
| S2.3 Package import behavior | Yes | Concrete command output. |
| S3 Fix internal imports | Yes | Concrete relative-import diff. |
| S4 Verify command surfaces | Yes | Concrete `test_output.txt`. |
| S5 Export artifacts | Yes | Concrete artifacts. |
| S9 Preserve old script-from-package-dir style | Checkable but out of scope | Invalidated by scope decision. |

Progress drops:

- Step 8 to 9: justified. Splitting the test surface into three command checks
  clarifies the validation denominator.
- Step 14 to 15: mostly justified but a bit narrative. Reopening import work
  for script-style compatibility is plausible, but invalidating it as out of
  scope means the branch was not required by the original task.

Forced operations: S9 may be forced. It is a reasonable compatibility question,
but because it is immediately invalidated as out of scope, it should probably
be documented in notes rather than counted as active work.

Final 1.00 after validation: yes. The final command output validates all
required surfaces.

Suggested fixes:

1. Move S9 to `run_notes.md` unless a task requirement explicitly includes
   script-from-package-directory execution.
2. Include the original failing `python -m widget_runner.module` output.
3. Avoid changing the test framework and import behavior in the same ledger
   step; separate them if both are active work.

## Negative Controls Added

Two negative controls were added after the audit.

| Run | Purpose | Final progress | Largest drop | Test status |
| --- | --- | ---: | ---: | --- |
| `negative_control_monotonic_one_line` | Trivial one-line fix that should not need a drop | 1.00 | 0.000 | passed |
| `negative_control_incomplete_budget_limited` | Budget-limited run that must not reach 1.00 | 0.667 | 0.250 | failed |

`negative_control_monotonic_one_line` is intentionally monotonic. Its progress
curve rises from 0.00 to 1.00 only after the final pytest pass.

`negative_control_incomplete_budget_limited` intentionally ends with unresolved
active work. `test_output.txt` records two failing tests, and the ledger leaves
rounding and full-discount clamping incomplete.

## Audit Recommendations

1. Keep product/validation progress separate from run-artifact bookkeeping in
   future studies, or label artifact work so it cannot be mistaken for coding
   progress.
2. Preserve both baseline failing output and final passing output when a ledger
   completion claims a regression was reproduced.
3. Do not mark a behavior subtask complete when the evidence only proves the
   test or contract was discovered.
4. Treat speculative environment/tooling paths as notes unless they actually
   block progress and require work.
5. Prefer split events over reopen events when the real issue is that a vague
   validation task needs more concrete leaves.
