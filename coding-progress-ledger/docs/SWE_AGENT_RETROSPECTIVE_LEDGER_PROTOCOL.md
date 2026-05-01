# SWE-agent retrospective ledger annotation addendum

This addendum specializes the general protocol
(`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`) to SWE-agent
trajectories normalized per `docs/SWE_AGENT_TRACE_SCHEMA.md`. It is
the only place where SWE-agent-specific shell vocabulary,
SWE-agent artifact filenames, and SWE-agent worked examples are
allowed. Anything you find here that contradicts the general protocol
is a bug — the general protocol wins.

In particular, this file does NOT redefine § 1-9 of the general
protocol; it only:

- maps SWE-agent's shell vocabulary onto the general categories (§ 1),
- names the SWE-agent input artifacts the procedure walks over (§ 2),
- specifies what the upstream success label is (§ 3),
- gives two real worked examples — one good, one bad — for each of
  one successful and one failed pilot run (§ 4),
- catalogues SWE-agent-specific pitfalls (§ 5).

## 1. SWE-agent shell vocabulary → general categories

SWE-agent assistant turns issue a single shell-style command per turn
(see `SWE_AGENT_TRACE_SCHEMA.md` § 5). The general category set in
`ledger_progress/core.py:SubtaskCategory` is unchanged. Map roughly:

| SWE-agent action                                                                 | General category |
|----------------------------------------------------------------------------------|------------------|
| `find_file`, `search_dir`, `search_file`, `grep`, `ls`, `open`, `goto`, `scroll_*` | `INVESTIGATION`  |
| `edit`, `create`                                                                 | `PRODUCT`        |
| `pytest`, `tox`, `python <repro>.py`, ad-hoc reproduction scripts                | `VALIDATION`     |
| `pip install`, `apt-get`, environment fixes                                      | `ENVIRONMENT`    |
| `submit`                                                                         | `ARTIFACT`       |
| `edit` of `*.md` / `docs/*` when the issue requires docs                         | `DOCUMENTATION`  |

A subtask collapses *many* such actions into one unit of discovered
work — e.g. ten investigation actions that locate a file are one
`INVESTIGATION` leaf, not ten. The trace is the witness; the leaf is
the unit of *discovered work*.

**`__init__.py` re-exports and package-level wiring** *(added per
H3 revision 3)*. Default to `PRODUCT` when the wiring change is
required by the issue itself (e.g., the issue's stack trace points
at the symbol the agent's edit exposes, or the issue text names the
broken import). Default to `ENVIRONMENT` only when the wiring is
*purely setup*: the agent's edit doesn't change runtime behavior of
the symbol; it just makes a previously-internal symbol importable
to satisfy a missing dependency the harness demands. When in doubt,
choose `PRODUCT` and note the ambiguity in `run_notes.md` § 4.

## 2. SWE-agent run-dir artifacts the annotator reads

Per `scripts/import_swe_agent_trace.py` (C3), each pilot run dir
contains:

- `task.md` — the issue text. Read once.
- `trajectory_summary.md` — read end-to-end before opening the JSON.
- `normalized_trace.json` — the source of truth for the walk.
- `final_diff.patch` — final-state artifact; **do not read until
  after the walk**, then only as evidence per general § 4.4.
- `eval_output.txt` — final-state artifact, same rule.
- `source_metadata.json` — pilot bookkeeping (pilot_id, instance_id,
  upstream success label, etc.). Annotators do **not** consult
  `final_success` here during the walk.
- `source_trace.json` — the upstream byte-equivalent copy. Auditors
  consult this; annotators normally do not.

## 3. The upstream success label

For SWE-agent / nebius rows, the upstream success label is the
boolean `target` field, mirrored as `final_success` in
`normalized_trace.json` and `source_metadata.json`. General § 4.2
applies verbatim: it is a stratification field, never a feature, and
never used to decide intermediate completion.

## 4. Worked examples (good × 2, bad × 2)

The two pilot runs below are annotated *of the same runs* in good and
bad form so the contrast is clean. Both are real entries in the pilot
cache.

### Example A — `swe_agent_pilot_s_01` / `Melevir__cognitive_complexity-15`

Shape (43 steps, success): the agent reads the issue (binary-logical-
operator counting bug), localizes `cognitive_complexity/utils/ast.py`
with several `find_file` / `search_dir` / `open` / `goto` calls, edits
line 88, runs `pytest`, and submits. Final eval passes.

#### A-good (right way)

```python
session = LedgerSession("Fix incorrect counting for binary logical operators")
inv = session.add(
    "Localize file controlling complexity increments for boolean ops",
    step=2, category=SubtaskCategory.INVESTIGATION,
)
session.complete(
    inv, "step 23: open cognitive_complexity/utils/ast.py shows process_node_itself", step=23,
)
prod = session.add(
    "Edit utils/ast.py to drop the nesting increment for B-op sequences",
    step=24, category=SubtaskCategory.PRODUCT,
)
session.complete(prod, "step 24: edit 88:88 inserted the +2 fix", step=24)
val = session.add(
    "Run pytest and confirm test_real_function expectation",
    step=26, category=SubtaskCategory.VALIDATION,
)
session.complete(val, "step ~30: pytest output shows test_real_function passes", step=30)
art = session.add("Submit final patch", step=42, category=SubtaskCategory.ARTIFACT)
session.complete(art, "step 42: submit issued", step=42)
```

Why this is right:

- Each leaf cites a concrete step index. Evidence is short, traceable.
- Validation is marked complete because the agent itself ran `pytest`
  in-trace — `eval_output.txt` is corroborating evidence, not the
  primary justification.
- The investigation leaf collapses ~10 navigation steps into one
  outcome ("found the right file"); we do not invent ten subtasks.

#### A-bad (wrong way) — DO NOT DO THIS

```python
session = LedgerSession("Fix incorrect counting for binary logical operators")
for desc in [
    "Understand issue", "Find the right file", "Edit the file",
    "Run tests", "Submit",
]:
    s = session.add(desc, step=1, category=SubtaskCategory.PRODUCT)
    session.complete(s, "final_success=True", step=1)
```

What's wrong:

- Every leaf marked complete at step 1: violates "annotate visible
  trace evidence" (general § 4.1) — none of the work is visible at
  step 1.
- Evidence string `"final_success=True"` violates the hard rule
  against using the upstream label as completion evidence
  (general § 4.2).
- All five leaves filed under `PRODUCT`: destroys the
  `(INVESTIGATION, PRODUCT, VALIDATION)` decomposition that
  downstream queries depend on.
- Front-loading produces a flat progress curve from step 1 onward
  with no shape — the observation channel reduces to "agent
  succeeded", which is just `final_success` re-encoded.

### Example B — `swe_agent_pilot_f_01` / `WIPACrepo__iceprod-339`

Shape (17 steps, **failure**): the agent reads the issue (remove a
`getip.php` lookup pointing at a soon-to-be-decommissioned server),
runs `ls` / `find_file` / `grep`, opens `iceprod/core/functions.py`,
edits line 274, and **submits without running any tests**. The
upstream eval (post-submission) records the patch as not resolving
the issue.

#### B-good (right way)

```python
session = LedgerSession("Remove getip.php request")
inv = session.add(
    "Locate getip.php usage in the repo",
    step=2, category=SubtaskCategory.INVESTIGATION,
)
session.complete(
    inv, "step 11: search_file finds getip.php in iceprod/core/functions.py", step=11,
)
prod = session.add(
    "Replace getip.php lookup at functions.py:274",
    step=12, category=SubtaskCategory.PRODUCT,
)
session.complete(prod, "step 14: edit 274:274 issued", step=14)
val = session.add(
    "Verify replacement does not break tests",
    step=14, category=SubtaskCategory.VALIDATION,
)
# DELIBERATELY NOT COMPLETED.
art = session.add("Submit final patch", step=16, category=SubtaskCategory.ARTIFACT)
session.complete(art, "step 16: submit issued", step=16)
```

…and `run_notes.md` records, under "Known missing evidence":

> The agent submitted at step 16 without running any tests in-trace.
> `eval_output.txt` exists (4030 chars, post-hoc) and shows the patch
> did not resolve the issue, but no in-trace pytest / repro / eval-log
> read happened. The validation leaf is left at `not_started`. Final
> progress will therefore be < 1.0 even though the agent submitted,
> which is the correct shape: validation as *discovered* work was
> never performed.

Why this is right:

- Validation leaf left at `not_started` because no validation step is
  visible in the trace. Progress drops as a direct consequence — that
  drop *is* the observation we want.
- `final_success=False` is mentioned only in `run_notes.md`, never as
  ledger evidence. The ledger doesn't change shape based on it.
- Hidden-work gap (the agent never noticed there's a test fixture for
  the same lookup at `tests/core/functions_test.py:…`) noted in
  `run_notes.md` rather than retro-fitted as a discovered subtask.

#### B-bad (wrong way) — DO NOT DO THIS

```python
session = LedgerSession("Remove getip.php request")
inv = session.add("Locate the file", step=2, category=SubtaskCategory.INVESTIGATION)
session.complete(inv, "evidence: agent found it", step=11)
prod = session.add("Edit the file", step=12, category=SubtaskCategory.PRODUCT)
session.complete(prod, "evidence: agent submitted", step=14)
val = session.add("Run tests", step=14, category=SubtaskCategory.VALIDATION)
session.complete(val, "eval_output.txt shows the patch was wrong", step=16)
session.reopen(prod, reason="patch incorrect per eval_output.txt", step=16)
```

What's wrong:

- Validation marked complete because `eval_output.txt` exists. But
  the agent never *ran* validation in-trace; the eval log is post-hoc.
  Violates general § 4.4.
- `prod` reopened "because the patch was wrong per `eval_output.txt`"
  — uses the post-hoc upstream label to retroactively change
  intermediate completion. Violates general § 4.2.
- Evidence strings ("evidence: agent found it") have no step
  citation, so the next annotator cannot audit the call.
- Net effect: the ledger now mirrors `final_success` (validation
  reached then immediately torn down), so the observation channel
  reduces to a noisy proxy for the upstream label. This is exactly
  the failure mode the protocol exists to prevent.

## 5. SWE-agent-specific pitfalls

These supplement (do not replace) the general pitfalls in
`RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` § 10.

1. **Long stretches of repeated `find_file`s.** Sometimes correct as
   `in_progress`; sometimes the right call is `blocked` paired with
   a reason that cites the step where the agent gave up varying the
   query. Default to `in_progress` unless the trace makes the stuck
   condition explicit.
2. **Thought-only assistant turns** (parse warning
   `no_fenced_block` in the normalized trace). Treat as
   zero-evidence, zero-status-change steps. Do not infer planning
   subtasks from them.
3. **`submit` without a preceding `pytest`.** This is the canonical
   "validation never started" pattern (Example B). Resist the urge
   to mark validation complete just because the SWE-agent harness
   ran the eval after submission.
4. **`edit` of a test file.** Default to `PRODUCT` only if the issue
   text explicitly asks for a test change. Otherwise it is suspect:
   the agent may be patching the test to silence the failure, which
   is a hidden-work signal worth a `run_notes.md` entry.
5. **Ambiguous role mapping** (e.g. an `ai` turn whose `command`
   contains a literal here-doc that itself looks like assistant
   output). Trust the upstream `role` field, not the surface text;
   the normalizer already preserved the upstream role under
   `events[].raw.role`.
6. **Harness-forced termination is not an agent submit.** When the
   normalized trace's `exit_status` is `submitted (exit_context)`,
   `submitted (exit_format)`, or any other harness-forced submission
   AND the agent never issued a literal `submit` command in-trace,
   do **not** add an `ARTIFACT` leaf. The submission is environmental,
   not discovered work. Record the forced termination in
   `run_notes.md` § 6 instead. Distinguish from `submitted` (no
   parenthetical), which is the agent's own `submit`.
7. **`final_diff.patch` is a state diff, not an agent-action diff.**
   It captures every file change since the start of the trace,
   including reproduction scripts created by `create` / `edit` very
   early on. If the trace shows the agent never edited
   product code, treat any non-empty `final_diff.patch` as
   investigation/repro residue, not as `PRODUCT` evidence. Always
   cross-check `final_diff.patch` against the trace's `edit` /
   `create` history before citing it.
8. **Bug-fix tasks always have implicit validation work, even when
   the trace doesn't surface it** *(added per H3 revision 1)*. For
   any task whose acceptance bar requires the runtime to behave a
   particular way (every bug-fix issue in this corpus, plus every
   "make this test pass" or "remove this echo" task), validation is
   implicit discovered work: an honest observer can name "verify
   the fix works" as a unit of work the agent could perform. Always
   add a `VALIDATION` leaf for such tasks. If the agent ran tests /
   a repro in-trace, complete the leaf with that evidence; if the
   agent never validated, leave the leaf at `not_started` and
   record "submitted without in-trace validation" in
   `run_notes.md` § 6. Final progress < 1.00 is the correct shape
   in that case — and that shape is what distinguishes a
   submit-without-test trace from a hidden-work-gap trace
   (f_06-style) where the agent did everything they could see.

   *Why this is in the addendum, not the general protocol:* the
   rule is sharp only when the task type is "fix-then-verify". A
   research-style task whose acceptance bar is "produce an answer"
   has no implicit validation; for those, default to the general
   protocol's strict "annotate only visible trace evidence".

## 6. Open questions / known caveats (SWE-agent specific)

1. The `submit` action is filed under `ARTIFACT`. If the corpus
   turns out to contain `submit` actions that *also* perform
   validation (some SWE-agent harnesses run tests on `submit`), the
   addendum may need a "submit-as-validation" rider. Not observed
   in pilot so far; revisit if seen.
2. For traces where the agent itself reads `eval_output.txt` mid-run
   (a `pytest` followed by inspecting failures and editing again),
   the eval-log read is in-trace evidence and the validation leaf
   can transition normally. For traces where the eval log appears
   only after the trace ends, see general § 4.4.
3. We do not require a `BLOCKED` event for every long search loop;
   long unproductive stretches are simply low-progress regions of
   the curve. If they become common enough to drown out signal,
   Workstream K's evidence audit will surface that, and we revise.
