# Annotation protocol revisions (H3)

This document satisfies `TASKS.md` § Workstream H, task **H3**. It
lists **minimal** changes to the protocol docs that are justified by
the H2 inter-annotator data, ranked by severity. The H2 report
(`runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`)
contains the full diff and per-pilot analysis.

The general protocol
(`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`) and the
SWE-agent addendum
(`docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`) are the targets.
Anything that has been a protocol gap *only on SWE-agent traces*
should land in the addendum; gaps that are agnostic of source
should land in the general doc.

**Are changes needed?** Yes — three minimal changes, plus one
explicit acknowledgment of legitimate annotator latitude. Without
revision 1, two reasonable annotators will continue to disagree on
the implicit-validation question on roughly every bug-fix submit-
without-test trace.

---

## Revision 1 — Implicit validation for bug-fix tasks (HIGH severity)

**Where:** SWE-agent addendum
(`docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`), § 5
(SWE-agent-specific pitfalls).

**Justification:** the f_01 disagreement (0.67 vs 1.00 final coding-
progress) is the only "different conclusions" inter-annotator gap
in the H2 data. Both annotators applied the protocol consistently;
the protocol itself underdetermines the call. The f_04 / s_04
pilots show the same pattern (submit-without-test) and are similarly
exposed.

**Proposed text** to add at the end of § 5 (after pitfall #7):

> **Pitfall #8: bug-fix tasks always have implicit validation
> work, even when the trace doesn't surface it.** For any task
> whose acceptance bar requires the runtime to behave a particular
> way (every bug-fix issue in this corpus, plus every "make this
> test pass" or "remove this echo" task), validation is implicit
> discovered work: an honest observer can name "verify the fix
> works" as a unit of work the agent could perform. Always add a
> `VALIDATION` leaf for such tasks. If the agent ran tests / a
> repro in-trace, complete the leaf with that evidence; if the
> agent never validated, leave the leaf at `not_started` and
> record "submitted without in-trace validation" in
> `run_notes.md` § 6. Final progress < 1.00 is the correct shape
> in that case — and that shape is what distinguishes a
> submit-without-test trace from a hidden-work-gap trace
> (`f_06`-style) where the agent did everything they could see.
>
> *Why this is in the addendum, not the general protocol:* the
> rule is sharp only when the task type is "fix-then-verify". A
> research-style task whose acceptance bar is "produce an
> answer" has no implicit validation; for those, default to the
> general protocol's strict "annotate only visible trace
> evidence". Source-specific addenda specialize the general rule
> to the task type they describe; this is one such specialization.

**Effect on existing annotations:** v1 already follows this rule
(my f_01 / f_04 / s_04 specs all have a not_started VALIDATION
leaf). v2 would converge after applying this revision.

**Memory cross-reference:** this codifies the rule in
`feedback_validation_implicit_for_bug_fix.md`.

---

## Revision 2 — Stuck-loop "third iteration begins" wording (LOW severity)

**Where:** general protocol
(`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`), § 6 (BLOCKED
status, stuck-loop rule).

**Justification:** v1 marked `f_03` as `BLOCKED` at step 22; v2 at
step 30. Both following the same rule, with different counting
conventions: v1 counted the third assistant-tool *cycle* starting
(which begins where the third assistant turn that is part of the
pattern fires); v2 counted by complete cycles closed.

**Proposed text change** in § 6 (a) command-loop:

> Replace:
>
>   "Mark `blocked` at the step where the third iteration begins."
>
> With:
>
>   "Mark `blocked` at the **assistant-turn step** where the third
>   iteration begins — i.e. the third occurrence of the
>   pattern-starting command, counted as the first command of each
>   iteration. The third 'begins' here refers to *the assistant
>   issuing the pattern-starting command for the third time*, not
>   the cycle's third tool response."

**Effect:** v1's interpretation becomes the canonical one. The
wording is a clarification, not a semantic change.

---

## Revision 3 — ENVIRONMENT category boundary clarification (LOW severity)

**Where:** SWE-agent addendum, § 1 (shell vocabulary → category
map).

**Justification:** v2 categorized "make `SimpleHeuristicsPlayer`
importable" (a `__init__.py` exports change) as ENVIRONMENT in
`s_03`; v1 called it PRODUCT. Both readings sit inside the general
protocol's ENVIRONMENT description ("setup work that blocks
product work without being part of it"); the line is genuinely
unclear for agent-internal package wiring.

**Proposed text** to add at the bottom of § 1:

> **`__init__.py` re-exports, package-level wiring.** Default to
> `PRODUCT` when the change is required by the issue (e.g., the
> issue's stack trace points at the symbol that the wiring change
> exposes). Default to `ENVIRONMENT` only when the wiring change
> is *purely setup*: the agent's edit doesn't change runtime
> behavior of the symbol; it just makes a previously-internal
> symbol importable to satisfy a missing dependency the harness
> demands. When in doubt, choose `PRODUCT` and note the ambiguity
> in `run_notes.md` § 4.

**Effect:** moves the s_03 ENVIRONMENT-vs-PRODUCT call to a default,
keeping it as an "uncertain decision" rather than an annotator
free choice.

---

## Acknowledgment — Granularity is annotator latitude (no change needed)

**Observation:** v2 systematically produces more leaves than v1
(mean +1.0 leaves per pilot). v2 splits "build repro + observe
output" into two leaves (INV + VAL); v1 collapses them into one
INV. v2 splits multi-stage product work into separate complete
leaves; v1 uses one PRODUCT leaf with REOPEN.

**Why no protocol change:** both readings are inside the protocol's
"discovered work as a unit" wording, and both produce the same
final progress (or, for `f_03`, a small difference that doesn't
flip the success/progress quadrant). Forcing a single granularity
would introduce a new judgment call ("how fine is fine enough?")
that the protocol cannot resolve without arbitrariness.

**Proposed text** to add as a new note in general § 9 (Procedure):

> **Granularity is annotator latitude.** Two reasonable annotators
> will sometimes disagree on whether to model "build repro +
> observe output" as one INVESTIGATION leaf or as
> INVESTIGATION + VALIDATION; or whether to model a multi-stage
> product fix as one PRODUCT leaf with REOPEN events or as
> several complete PRODUCT leaves. Both are legitimate as long as
> they preserve the framework's load-bearing properties: same
> final-progress shape, evidence cited at each transition,
> non-monotonicity preserved when the trace forces it. Do not
> rewrite an existing annotation purely to change granularity.

**Effect:** documents what the inter-annotator data already shows
without forcing convergence on a single mesh size.

---

## Not changed — single-annotator caveat

The H2 report's § 7 ("What this DOES NOT measure") explicitly
notes that both v1 and v2 are LLM passes and their biases are
likely correlated. The remaining methodological hole (real human
inter-annotator gap) is unaddressed by H1-H3 and would be the
work of a future Workstream H expansion. The current revisions
target only the gaps the H2 data actually surfaced.

## Summary

- **Revision 1** (high severity): add Pitfall #8 to the SWE-agent
  addendum so bug-fix submit-without-test traces produce the same
  ledger across annotators.
- **Revision 2** (low severity): tighten general § 6 stuck-loop
  wording to remove the "third iteration begins" off-by-one.
- **Revision 3** (low severity): add `__init__.py` /
  package-wiring guidance to the SWE-agent addendum's category
  map.
- **Acknowledgment** (no change): document granularity latitude
  in general § 9; do not enforce a single mesh size.

After applying revisions 1-3, an inter-annotator re-run should
show f_01 converging to the v1 progress (0.67), f_03's BLOCKED
step converging to step 22, and s_03's ENVIRONMENT-vs-PRODUCT
call shifting to PRODUCT (with explicit uncertainty noted). A
fourth pass with the revisions in place is the right empirical
test of whether the changes close the gaps.
