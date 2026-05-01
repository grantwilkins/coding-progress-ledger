# K2: source-trace evidence gaps — what live instrumentation would close

This satisfies `TASKS.md` § Workstream K, task **K2**. It enumerates
what an honest retrospective annotator *could not* see in the
nebius/SWE-agent-trajectories source artifacts of the 20 pilots, and
which gaps a live SWE-agent instrumentation pass (Workstream N) could
close vs. which gaps no source could ever close.

The audit grounded in K1 (`runs/swe_agent_pilot/EVIDENCE_AUDIT.md`):
**51 / 81 (63%) completion events have at least one strong evidence
type; 30 / 81 (37%) are `manual_note` only.** This document explains
the 37%.

## 1. What the source provides per pilot

Every nebius row materialized into a run dir gives:

```text
task.md                  — the upstream issue text
source_trace.json        — verbatim trajectory rows from the dataset
normalized_trace.json    — per-step (role, thought, action, observation, command, files_touched)
final_diff.patch         — generated_patch (state, not action)
test_output.txt          — eval_logs (post-hoc, not visible to the agent)
source_metadata.json     — instance_id, model_name, success label, length
```

Annotation evidence cited by E1 / E3 in `evidence` arrays draws from
`normalized_trace.json` and the per-step thought/action/observation
fields. K1's classifier maps them onto `test_output | diff |
file_exists | command_output | contract_text | manual_note`.

## 2. Gap classification

Each row marks whether the gap is recoverable from the source vs only
recoverable from live instrumentation vs structurally not recoverable.

| Source data lacks | Recoverable retrospectively? | Closed by live instrumentation? | Pilots where this matters |
|---|---|---|---|
| **Baseline failing test output** (the test that demonstrates the bug *before* the fix) | No — the source omits any pre-fix run of the issue's test | **Yes** — a live agent could run the test before editing | All 20 pilots have an issue with a "test demonstrates the bug" implication. None has a baseline test run captured. |
| **Final passing/failing eval output as the agent saw it** | Partial — `test_output.txt` is the harness's post-hoc eval, not what the agent ran in-trace | **Yes** — a live trace could capture every `pytest` invocation the agent issued | `f_07`, `f_08`, `f_09`, `f_05`, `f_03`, `f_10` (six pilots ran in-trace tests; the source's normalized observations contain partial output, not the full pytest stream) |
| **Final patch as a verified diff against a known commit** | Yes (the `generated_patch` field is byte-stable) | Yes (same) | All 20 pilots have a patch; F4 surfaced the f_06 caveat that "patch present" ≠ "patch is correct". |
| **File-open context** (line N±k around an edit, the file's prior state) | Partial — the trace may cite line numbers but rarely shows surrounding context | **Yes** — a live trace could capture pre-edit/post-edit windows | `f_07`, `f_05`, `f_10` heavily — these pilots edit the same range repeatedly without the surrounding code visible |
| **Command output beyond the harness's truncation** | Mostly no — observations are truncated in the source | **Yes** — live trace can capture full stdout/stderr | `f_02`'s 509 `find_file` calls all show "No matches found" but the trace can't show e.g. the directory listing the agent never asked for |
| **Tool observations the agent didn't display** | No — only what the agent's harness rendered is recorded | **Partial** — depends on agent-vs-sidecar instrumentation choice | `f_02`'s "agent never tried `ls`" observation is invisible structurally; we know it because the trace doesn't contain `ls`, not because the source surfaces a "missed tool" signal. Live couldn't materialize "what the agent didn't do" either. |
| **Agent's in-context reasoning at decision boundaries** | Partial — `thought` field exists in normalized but is sparse | **Partial** — depends on whether the live agent emits explicit reasoning | Across pilots; the `f_06` "agent didn't notice the repro returned no error" decision is visible only via the absence of follow-up action |
| **The "would `submit` have happened" intent** | No — harness-forced terminations (`exit_status == 'submitted (exit_context)'`) are indistinguishable from agent-issued submits without context | **Yes** — a live `submit` event has clear provenance | Six pilots: `f_02`, `f_03`, `f_05`, `f_07`, `f_08`, `f_10`. K1 recorded all six as missing ARTIFACT leaves per pitfall #6. |
| **Whether the agent's repro actually exercised the bug** | No — needs ground truth about which code path is buggy | **Partial** — live could run the *issue's reference test* in addition to the agent's repro | `f_06` (canonical hidden-work-gap shape); `f_01`, `f_04` to a lesser degree |

## 3. The 37% manual-only completions in K1, by class

K1's 30 manual-only completions (37% of 81) split as:

| Class | Approximate count | Why this lands as `manual_note` | Could live close it? |
|---|---:|---|---|
| Single-line `edit` acks ("step N: edit M:M acknowledged by tool") | ~18 | Classifier sees no `pytest`, no path-pattern match, no stdout — the tool-ack text is too short for the heuristic | **Yes** — instrumented edits would emit a structured event with file path; classifier could extend a `tool_action` strong type |
| `submit` acks | ~5 | Same — "submit issued; exit_status 'submitted'" is parsed as `manual_note` | Yes — live submit has a structured provenance |
| Multi-step PRODUCT completions where the cited evidence summarizes activity | ~5 | The cited evidence is annotator prose like "mid-edit syntax error rejected, retry succeeded" rather than a tool-output quote | Partial — better citations would help; live capture of each individual edit would too |
| INVESTIGATION completions on issue-text reading | ~2 | `contract_text` heuristic didn't match because the cited evidence quoted a step number rather than the contract text itself | Yes — improve K1 classifier OR cite the contract text directly |

The classifier is correctly refusing to call these "strong"; the
*signal* of "the agent did something visible" is in the source but the
heuristic doesn't surface it because it's looking for stdout-style
patterns. **Class 1 (~18 / 30 = 60% of the manual-only) is closable by a
classifier extension that recognizes structured edit-ack patterns;
this does not require live instrumentation.**

## 4. What live instrumentation closes that retrospective cannot

In rough order of impact:

1. **Hidden-work gaps that the trace surfaces but doesn't act on**
   (`f_06` repro succeeds spuriously; `f_02` agent never tried
   `ls`). Live could at least capture the *structured* observation
   that the test/repro returned X, allowing a downstream check
   "did this repro actually exercise the issue's bug?"
2. **Agent-vs-harness submit provenance.** The six harness-terminated
   pilots cannot tell us "would the agent have submitted given
   another N steps?". Live instrumentation captures the difference.
3. **Pre-fix baseline runs.** None of the 20 pilots has a captured
   baseline test result. Live could run the issue's test before any
   edit; the resulting failure is the strongest possible
   `test_output` evidence.
4. **Per-edit before/after file state.** The `final_diff.patch` is
   a state-diff, not an action-diff (pitfall #7). Live could
   capture file state at each edit, recovering a true action-diff.

## 5. What live instrumentation cannot close

- **What the agent *didn't* do.** "Agent never tried `ls`" is a
  structural observation about the absence of a tool call; live
  can't materialize a hypothetical alternative path.
- **Whether the agent's choice was correct given the issue text.**
  Requires ground truth about correctness, which only the upstream
  evaluator has. Live instrumentation can capture the evaluator's
  output but not its semantic justification.
- **Annotator latitude for granularity.** "Build repro + observe"
  as one INV leaf vs INV+VAL is a modeling choice; no source can
  resolve it (per H3 acknowledgment).

## 6. Recommendations

- **Cheapest win: classifier extension.** Add a `tool_action`
  strong-evidence type that matches `edit \d+:\d+ ` / `submit ` /
  `goto ` / `search_file ` patterns. Closes ~18 of the 30
  manual-only completions on the existing pilot data, no
  re-annotation needed. ~30 min of work in
  `scripts/rescore_suite_by_category.py:EVIDENCE_PATTERNS`.
- **Next win: cite tool acks directly.** Annotator pass updating
  evidence to quote the tool's edit-ack text verbatim closes the
  remaining ~7 manual-only PRODUCT completions. ~30 min total.
- **Live instrumentation (Workstream N):** justified primarily for
  closing class 1 (hidden-work gap visibility), class 2 (submit
  provenance), and class 3 (baseline runs). Do not pursue solely
  to fix manual-only counts; the cheaper wins above will close
  most of those without engineering effort.

## 7. Pointers

- K1 audit: `runs/swe_agent_pilot/EVIDENCE_AUDIT.md`,
  `runs/swe_agent_pilot/EVIDENCE_AUDIT.json`
- K1 script: `scripts/audit_pilot_evidence.py`
- Classifier: `scripts/rescore_suite_by_category.py:classify_evidence`
- Workstream N (live instrumentation, not yet started): `TASKS.md` § N
- F4 caveat about `final_success` heuristic: `datasets/observation_distribution_comparison.md` § 3.6
