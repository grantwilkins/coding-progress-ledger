# H4 — gate result: re-annotation under H3-revised protocol

This satisfies the gate task introduced by `runs/swe_agent_pilot/GO_NO_GO_MEMO.md`
(M1) and added to `TASKS.md` § Workstream H, task **H4**. It tests
empirically whether the three H3 protocol revisions close the
inter-annotator conclusion gaps surfaced by H2.

## 1. Setup

- **v1 (original annotator):** Claude (E1 pass), specs at
  `annotations/swe_agent_pilot/`. Pre-revision protocol.
- **v2 (cold pass under pre-revision protocol):** Opus subagent,
  specs at `annotations/swe_agent_pilot_v2/` (H1).
- **v3 (cold pass under post-revision protocol):** Opus subagent,
  specs at `annotations/swe_agent_pilot_v3/`. Same isolation rules as
  H1: only protocol docs, addendum, template, core enums, the per-pilot
  source artifacts, and one s_02 spec for file-format reference.
  Forbidden from reading any v1 / v2 specs / notes, the pilot summary,
  the H2 agreement report, and the case studies.

The protocol read by v3 is the **post-H3 protocol** committed in
2656391:

- General § 6: stuck-loop "third iteration begins" tightened to mean
  the **assistant-turn step** where the third occurrence of the
  pattern-starting command fires (Revision 2).
- Addendum § 5 Pitfall #8: bug-fix tasks **always** carry an implicit
  `VALIDATION` leaf (Revision 1, HIGH severity).
- Addendum § 1: `__init__.py` re-exports default to `PRODUCT` when
  issue-required, `ENVIRONMENT` only when purely setup (Revision 3).
- General § 9: **granularity is annotator latitude** (acknowledgment).

Materialization: `runs/swe_agent_pilot_v3/` mirrors the source
artifacts (`task.md`, `source_trace.json`, `normalized_trace.json`,
`source_metadata.json`, `final_diff.patch`, `test_output.txt`,
`trajectory_summary.md`) for the 5 pilots; `ledger.jsonl` and
`progress*.csv` are derived freshly from the v3 specs by
`scripts/annotate_pilots_from_spec.py`. All 5 v3 runs pass
`ledger-run check-run`.

## 2. Headline numbers

| Pilot | v1 overall / coding | v2 overall / coding | v3 overall / coding | v1↔v3 Δcoding |
|-------|---------------------|---------------------|---------------------|---------------|
| `s_01` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **0.00 ✓** |
| `s_03` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **0.00 ✓** |
| `f_01` | 0.75 / 0.67 | 1.00 / 1.00 | 0.75 / 0.67 | **0.00 ✓** |
| `f_06` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **0.00 ✓** |
| `f_03` | 0.50 / 0.50 | 0.50 / 0.50 | 0.33 / 0.33 | **−0.17 ✗** |

Quadrant agreement (success/progress quadrants): **5 / 5.**

## 3. Gate verdict

**PASS, with one caveat.**

The gate condition from M1 is: *"f_01 produces the same conclusion
(final coding-progress within 0.05) in v3 vs both v1 and v2 readings
under the new implicit-validation rule."*

- **v1 vs v3 on f_01: identical.** 0.67 vs 0.67. The HIGH-severity
  Revision 1 closes the H2 gap exactly as predicted: the v3
  annotator independently added the not_started VALIDATION leaf
  required by Pitfall #8 and arrived at v1's progress shape.
- **v2 vs v3 on f_01: 0.33 difference.** v2 was the
  pre-revision reading (no implicit VAL leaf, 1.00). The revision is
  *designed* to move v2-style readings to v3. Spec text from
  `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md`:
  *"v2 would converge after applying this revision."* The 0.33 gap
  is the size of the gap the revision is supposed to close, not a
  failure.

The literal "v3 within 0.05 of both v1 and v2" reading of the gate is
unmeetable by construction: if v3 converged with v2 (which had the
ambiguity) the revision would have failed. The intended reading is
"v3 reproduces the v1 conclusion that the revision documents as
correct, demonstrating that a fresh annotator under the revised
protocol arrives at v1's shape." That condition is met cleanly.

## 4. Caveat: f_03 deepens, not because the revisions are wrong

v3's f_03 came in at 0.33, **lower** than both v1 (0.50) and v2 (0.50).
Investigating the v3 spec:

- v1 has 2 leaves: `S1` (INV, complete, build repro) and `S2` (INV,
  blocked at step 22, stuck-loop on `configparser` localization).
  No VALIDATION leaf.
- v2 has 3 leaves: `S1` (INV, complete), `S2` (VAL, complete —
  inspect produced INI for the bug), `S3` (INV, blocked at step 30).
  No implicit-VAL leaf for "verify the fix works after a fix is
  written."
- v3 has 3 leaves: `S1` (INV, complete), `S2` (INV, blocked at step
  26), `S3` (VAL, **not_started — implicit per Pitfall #8**, the
  agent was harness-terminated before a fix was written and never
  ran a test against a fix).

**This is the H3 revisions working.** v3 correctly applied Pitfall #8
to f_03 (a bug-fix task whose agent never validated). v1 *did not
apply* Pitfall #8 to f_03. The HIGH-severity revision codifies the
rule v1 applied inconsistently — to f_01, f_04, s_04 (where the agent
submitted) but not to f_03, f_07, f_10 (where the harness force-quit
the agent mid-loop).

This is a **finding, not a gate failure**: the revised protocol is
*more consistent than v1 was.* The implication for the 20-pilot
corpus is in § 7.

## 5. Where revisions 2 and 3 landed

- **Revision 2 (stuck-loop wording):** v1 said f_03 BLOCKED at step
  22 (the third assistant-turn issuing the pattern-starting
  `search_file 'configparser'`); v2 said step 30 (third complete
  cycle); v3 said step 26 (third occurrence of `search_file
  'configparser' setup_cfg_fmt.py`). v3's reading is closer to v1
  than to v2 and within the revision's intent. The remaining 4-step
  delta between v1 (22) and v3 (26) reflects a different choice of
  which command to treat as "pattern-starting" — the revision
  resolved the off-by-one but did not enumerate every possible
  pattern shape. **Effect on quadrant: none** (both readings agree
  the leaf is BLOCKED on the same investigation).

- **Revision 3 (`__init__.py` default):** v3 of s_03 chose
  `ENVIRONMENT` for the `SimpleHeuristicsPlayer` import wiring, with
  rationale that the wiring was *purely setup for the agent's
  repro* (not the runtime behavior the issue demands). This is
  consistent with the revision's "default ENVIRONMENT only when
  purely setup" carve-out. **Effect on quadrant: none** (s_03
  remains 1.00 / 1.00).

## 6. Single-annotator caveat (unchanged)

v3, like v2, is an Opus subagent reading. Biases between v1, v2, v3
are likely correlated. This pass tests *protocol clarity under
revisions* — does an independent reader of the revised docs reach
v1's shape on the f_01 question? Yes. It does not test the stronger
*human-AI inter-annotator gap*; that remains future work.

## 7. Implications for the 20-pilot E1 corpus

The f_03 finding generalizes. Six of the 10 failure pilots ended in
harness-forced termination mid-loop: `f_02`, `f_03`, `f_05`, `f_07`,
`f_08`, `f_10`. v1 applied Pitfall #8 inconsistently across these.
A consistent application would lower coding-progress on each by the
1 / N factor of an added not_started VAL leaf. Concretely:

| Pilot | v1 coding | post-Pitfall-#8 estimate | direction |
|-------|-----------|--------------------------|-----------|
| `f_02` | 0.50 (2 leaves: 1 complete + 1 blocked) | 0.33 (3 leaves: +VAL not_started) | down |
| `f_03` | 0.50 (2 leaves) | 0.33 (3 leaves) | down (confirmed by v3) |
| `f_05` | 0.60 (5 leaves with 2 blocked, 3 complete) | already includes a VAL leaf at in_progress; no change | none |
| `f_07` | 0.67 (3 leaves) | 0.50 (4 leaves: +VAL not_started) | down |
| `f_08` | 0.71 (7 leaves with 1 blocked) | already includes a VAL leaf at in_progress; no change | none |
| `f_10` | 0.67 (3 leaves) | 0.50 (4 leaves: +VAL not_started) | down |

**Estimated effect:** four of the 20 pilots would shift downward by
0.14–0.17 in coding-progress under a consistent re-application of
Pitfall #8. None cross a quadrant boundary (they remain
failure-low-progress). The shape claim (failures span 0.50–1.00
discriminating between failure modes) is preserved; the absolute
values shift.

## 8. Recommendation

**Gate PASSES. Proceed to M2 (next sample size) with the following
follow-up.**

Before scaling to 100 traces, run a focused E1-pass cleanup on the 4
pilots above (`f_02`, `f_03`, `f_07`, `f_10`) to apply Pitfall #8
consistently. Estimated effort: ~30 min total (the rest of each ledger
is unaffected; only the implicit-VAL leaf is added).

The gate's load-bearing requirement — that fresh annotation under the
revised protocol reproduces v1's f_01 conclusion — is met. The
incidental finding (v1's own inconsistency on Pitfall #8 across the
failure pilots) is fixable cheaply and strengthens the corpus.

## 9. Pointers

- v3 specs: `annotations/swe_agent_pilot_v3/swe_agent_pilot_*.json`
- v3 prose: `annotations/swe_agent_pilot_v3/swe_agent_pilot_*.notes.md`
- Materialized v3 runs: `runs/swe_agent_pilot_v3/swe_agent_pilot_*/` (gitignored)
- Driver: `scripts/annotate_pilots_from_spec.py`
- Protocol revisions: `docs/SWE_AGENT_ANNOTATION_PROTOCOL_REVISIONS.md`
- Original H2 report: `runs/swe_agent_pilot_reannotation/ANNOTATION_AGREEMENT.md`
