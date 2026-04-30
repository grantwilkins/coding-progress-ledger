#!/usr/bin/env python3
"""Pilot-zero annotation (D4 + D5) for swe_agent_pilot_s_01 and _f_01.

This is NOT a reusable CLI helper (that would be D3, deferred). It is
the source-controlled record of the pilot-zero annotation. Each
pilot's events are hand-encoded against its `normalized_trace.json`
following the binding general protocol at
`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` and the SWE-agent
addendum at `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`.

Running this script materializes, for each of the two pilot-zero
runs:

  * <run_dir>/ledger.jsonl
  * <run_dir>/progress.csv               (via `ledger-run export-run`)
  * <run_dir>/progress_by_category.csv   (via `ledger-run export-run`)
  * <run_dir>/summary_by_category.json   (via `ledger-run export-run`)
  * <run_dir>/run_notes.md               (extended from D2 template)
  * <run_dir>/annotation_quality.json    (per D5)

It is idempotent: re-running overwrites with byte-identical output.

Auditors confirming an annotation only need this script + the
already-cached `normalized_trace.json` to reproduce.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress import LedgerSession, SubtaskCategory  # noqa: E402
from ledger_progress.queries import CODING_CATEGORIES  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "runs" / "swe_agent_pilot"


# ---------- per-pilot annotation specs ----------


def annotate_s_01() -> Tuple[LedgerSession, "AnnotationMeta"]:
    """swe_agent_pilot_s_01 / Melevir__cognitive_complexity-15 (43 steps, success).

    Walked end-to-end from `normalized_trace.json`. The trace shape
    is: locate `utils/ast.py` (steps 2-23), edit line 88 (24-25),
    run pytest (26-27) -> observe expected value mismatch, update
    test fixture per issue text at lines 125 and 147 with one
    syntax-error retry (28-39), re-run pytest (40-41), submit (42).
    """
    s = LedgerSession("Fix incorrect counting for binary logical operators")

    inv = s.add(
        "Locate file controlling complexity increments for boolean operator sequences",
        step=2,
        category=SubtaskCategory.INVESTIGATION,
    )
    s.complete(
        inv,
        [
            "step 21: search_dir 'def process_node_itself(' surfaces utils/ast.py",
            "step 23: open cognitive_complexity/utils/ast.py confirms file at 92 lines",
        ],
        step=23,
    )

    prod = s.add(
        "Edit utils/ast.py:88 to drop nesting increment for B-op sequences",
        step=24,
        category=SubtaskCategory.PRODUCT,
    )
    s.complete(
        prod,
        "step 25: edit 88:88 acknowledged by tool",
        step=25,
    )

    val1 = s.add(
        "Run pytest #1 to observe behavior after utils/ast.py fix",
        step=26,
        category=SubtaskCategory.VALIDATION,
    )
    s.complete(
        val1,
        "step 27: pytest output observed; subsequent agent action (step 28) edits the test fixture, indicating the run revealed the issue's stated expected-value update was still required",
        step=27,
    )

    fixture = s.add(
        "Update test_real_function expected values per issue spec (+4 -> +2)",
        step=28,
        category=SubtaskCategory.PRODUCT,
    )
    s.complete(
        fixture,
        [
            "step 28: edit 125:125 attempted; tool reports introduced syntax error (step 29) -- attempt rejected, no state change",
            "step 34: edit 125:125 retry succeeded (step 35 tool ack)",
            "step 38: edit 147:147 succeeded (step 39 tool ack)",
        ],
        step=39,
    )

    val2 = s.add(
        "Re-run pytest after fixture update to confirm passing state",
        step=40,
        category=SubtaskCategory.VALIDATION,
    )
    s.complete(
        val2,
        [
            "step 41: pytest tool output observed in-trace",
            "test_output.txt corroborates: post-submission eval reports passing",
        ],
        step=41,
    )

    art = s.add(
        "Submit final patch",
        step=42,
        category=SubtaskCategory.ARTIFACT,
    )
    s.complete(art, "step 42: submit issued; exit_status 'submitted'", step=42)

    meta = AnnotationMeta(
        pilot_id="swe_agent_pilot_s_01",
        instance_id="Melevir__cognitive_complexity-15",
        upstream_success_label=True,
        annotation_time_minutes=35,
        number_of_uncertain_events=1,  # see run_notes § 4
        number_of_evidence_gaps=0,
        whether_final_success_used_only_at_end=True,
        whether_progress_forced=False,
        whether_schema_gap_found=False,
        run_notes_body=_run_notes_body_s_01,
    )
    return s, meta


def annotate_f_01() -> Tuple[LedgerSession, "AnnotationMeta"]:
    """swe_agent_pilot_f_01 / WIPACrepo__iceprod-339 (17 steps, failure).

    Trace shape: ls/find_file/grep (2-7) -> open functions.py and
    locate getip.php at line 274 (8-13) -> edit (14-15) -> submit
    (16). NO in-trace pytest. Validation leaf is therefore
    deliberately left at not_started.
    """
    s = LedgerSession("Remove getip.php request to soon-decommissioned SL6 server")

    inv = s.add(
        "Locate getip.php usage in the repo",
        step=2,
        category=SubtaskCategory.INVESTIGATION,
    )
    s.complete(
        inv,
        [
            "step 7: grep -r 'getip.php' . returns hits in iceprod/core/functions.py and tests/core/functions_test.py",
            "step 11: search_file confirms getip.php at functions.py:274",
        ],
        step=11,
    )

    prod = s.add(
        "Replace getip.php lookup at iceprod/core/functions.py:274",
        step=12,
        category=SubtaskCategory.PRODUCT,
    )
    s.complete(
        prod,
        "step 15: tool ack confirms edit 274:274 applied",
        step=15,
    )

    # Discovered (because grep at step 7 named it) but never acted on.
    s.add(
        "Verify replacement does not break the test mock at tests/core/functions_test.py",
        step=14,
        category=SubtaskCategory.VALIDATION,
    )
    # DELIBERATELY NOT COMPLETED. See run_notes § 6.

    art = s.add(
        "Submit final patch",
        step=16,
        category=SubtaskCategory.ARTIFACT,
    )
    s.complete(art, "step 16: submit issued; exit_status 'submitted'", step=16)

    meta = AnnotationMeta(
        pilot_id="swe_agent_pilot_f_01",
        instance_id="WIPACrepo__iceprod-339",
        upstream_success_label=False,
        annotation_time_minutes=20,
        number_of_uncertain_events=0,
        number_of_evidence_gaps=2,  # validation leaf + hidden-work tests/core gap
        whether_final_success_used_only_at_end=True,
        whether_progress_forced=False,
        whether_schema_gap_found=False,
        run_notes_body=_run_notes_body_f_01,
    )
    return s, meta


# ---------- run_notes bodies (per pilot) ----------


def _run_notes_body_s_01(progress_overall: float, progress_coding: float) -> str:
    return f"""\
## Run notes — `swe_agent_pilot_s_01` (`Melevir__cognitive_complexity-15`)

- annotator: Claude (pilot-zero, AI-driven first pass)
- annotation pass: pilot-zero
- protocol: `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
- source addendum: `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`
- upstream success label (NOT a feature): `True`

### 1. Initial reading

The issue cites `test_real_function()` and asks the cognitive-complexity
calculation for a multiline `if` condition with binary logical
operators to drop from +4 to +2 — i.e. the B-op sequence should
receive only the B1 fundamental increment, not the B3 nesting
increment. The fix is in the calculator (`utils/ast.py`), and the
issue *also* explicitly tells the agent the expected value in the
existing test must be updated. Both edits are legitimate per the
issue text.

### 2. Initial ledger proposal (written before the walk)

```text
- INVESTIGATION  Locate the file that adds the nesting increment
- PRODUCT        Edit calculator to skip nesting increment for B-op chains
- VALIDATION     Run pytest after fix
- ARTIFACT       Submit
```

The walk added one PRODUCT subtask the proposal missed (test
fixture update) and one extra VALIDATION leaf (pytest after fixture
update). Both came from the trace, not the issue.

### 3. Checkpoint notes

- step 23: investigation closes — `cognitive_complexity/utils/ast.py`
  is open and `process_node_itself` is the right function (cited at
  step 21).
- step 24: production edit at line 88.
- step 27: pytest #1 output observed; the very next agent action is
  to edit the test fixture, so the run revealed (without us seeing
  the failure text) that the test still expected +4.
- steps 28-29: a syntax-error attempt; tool rejects the edit with no
  state change. Treated as zero-evidence (general § 6).
- steps 34-39: fixture edits at lines 125 and 147 succeed.
- step 41: pytest #2 observed in-trace.
- step 42: submit.

### 4. Uncertain decisions

- **Treating "edit a test file" as legitimate PRODUCT vs as a
  silence-the-failure anti-pattern (SWE-agent addendum pitfall #4).**
  Chose legitimate PRODUCT because the issue text explicitly says
  the existing `test_real_function` expected value should be +2 not
  +4, so the test edit follows the issue spec. Without that issue
  context this would have read as the anti-pattern.

### 5. Evidence citations

| subtask id | category        | completed at step | evidence step(s) | one-line citation |
|------------|-----------------|-------------------|------------------|-------------------|
| `S1`       | `INVESTIGATION` | 23                | 21, 23           | search_dir surfaces process_node_itself; open ast.py confirms |
| `S2`       | `PRODUCT`       | 25                | 25               | tool ack of edit 88:88 |
| `S3`       | `VALIDATION`    | 27                | 27               | pytest output triggered fixture-update branch |
| `S4`       | `PRODUCT`       | 39                | 28, 34, 38       | retry after syntax error; both fixture line edits ack'd |
| `S5`       | `VALIDATION`    | 41                | 41               | pytest #2 in-trace; test_output.txt corroborates |
| `S6`       | `ARTIFACT`      | 42                | 42               | submit issued |

### 6. Known missing evidence

None for this run. All discovered subtasks reached `complete` with
in-trace evidence; `test_output.txt` was used only as corroborating
evidence for the validation leaf, never as the primary justification.

### 7. Final scope closure

- total leaves: 6
- complete: 6 · in_progress: 0 · blocked: 0 · not_started: 0 · invalidated: 0
- progress (overall): {progress_overall:.2f}
- progress (CODING_CATEGORIES = product+validation+investigation): {progress_coding:.2f}

Was there ever a temptation to use `final_success` as evidence? **No.**
The fixture-update / silence-the-failure ambiguity in § 4 was
resolved from the issue text, not from the upstream label.

### 8. Schema gaps observed

None observed. The framework's category set
(`INVESTIGATION / PRODUCT / VALIDATION / ARTIFACT`) and event types
(`ADD_SUBTASK`, `UPDATE_STATUS`) covered the trace cleanly. The
`syntax error rejected` attempt at step 28 was naturally absorbed as
zero-evidence and the retry at step 34 carried the actual edit
evidence; no special convention was required.

(An earlier pilot-zero run flagged an `eval_output.txt` /
`test_output.txt` artifact-name divergence between C3's SWE-agent
importer and the framework's `ledger-run check-run`. That was a real
gap, fixed at the importer level: C3 now writes the framework name
`test_output.txt` directly, sourced from upstream `eval_logs`. This
re-annotation runs against the post-fix run dirs.)
"""


def _run_notes_body_f_01(progress_overall: float, progress_coding: float) -> str:
    return f"""\
## Run notes — `swe_agent_pilot_f_01` (`WIPACrepo__iceprod-339`)

- annotator: Claude (pilot-zero, AI-driven first pass)
- annotation pass: pilot-zero
- protocol: `docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`
- source addendum: `docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`
- upstream success label (NOT a feature): `False`

### 1. Initial reading

The issue asks the agent to remove a `getip.php` request that points
at an SL6 server slated for decommission, "or replace this lookup
with something else". The acceptance bar is therefore "the runtime
no longer hits that URL" — note this is a behavioral requirement,
which means the test mock at `tests/core/functions_test.py` (which
also references the URL) must be revisited too, otherwise the test
suite either silently passes against a removed code path or fails
because the mock no longer matches actual behavior.

### 2. Initial ledger proposal (written before the walk)

```text
- INVESTIGATION  Locate the getip.php usage
- PRODUCT        Remove or replace the lookup
- PRODUCT        Update test mock at tests/core/functions_test.py
- VALIDATION     Run tests to confirm
- ARTIFACT       Submit
```

The walk's ledger has only 4 leaves (no test mock update, no
in-trace validation), exactly matching the *failure* hypothesis: the
agent never opened the test file even though grep surfaced its
existence.

### 3. Checkpoint notes

- step 7: `grep -r 'getip.php' .` surfaces TWO hits — the
  production file and `tests/core/functions_test.py`. The agent
  proceeds with only the production file.
- step 11: `search_file` confirms the production-file location.
- step 14: edit 274:274 issued.
- step 16: submit, with no preceding test run.

### 4. Uncertain decisions

None. The trace is sparse enough that every ledger event has
unambiguous evidence; the debate was about what *not* to record.

### 5. Evidence citations

| subtask id | category        | completed at step | evidence step(s) | one-line citation |
|------------|-----------------|-------------------|------------------|-------------------|
| `S1`       | `INVESTIGATION` | 11                | 7, 11            | grep + search_file localize getip.php |
| `S2`       | `PRODUCT`       | 15                | 15               | tool ack of edit 274:274 |
| `S3`       | `VALIDATION`    | (not reached)     | —                | left at `not_started` deliberately |
| `S4`       | `ARTIFACT`      | 16                | 16               | submit issued |

### 6. Known missing evidence

- `S3` (validation) **left at `not_started`**. The agent submitted at
  step 16 without running pytest, tox, a repro script, or any
  in-trace eval read. `test_output.txt` (4030 chars; sourced by C3
  from upstream `eval_logs`) exists post-hoc and per the upstream
  eval reports the patch did not resolve the issue, but per general
  § 4.4 a post-hoc artifact cannot complete a validation leaf the
  agent never started. Final progress is < 1.0 by design.
- **Hidden-work gap.** Step 7's grep explicitly surfaced
  `tests/core/functions_test.py` as containing `getip.php`. The
  agent did not open this file. Whether the test mock update was
  required to resolve the issue is conditional on what the runtime
  expects — but the trace makes the absence of that work *visible*
  to an honest observer. We do not retro-fit a discovered subtask
  for the test mock; we only record the gap here. This is exactly
  the kind of datum the framework exists to surface.

### 7. Final scope closure

- total leaves: 4
- complete: 3 · in_progress: 0 · blocked: 0 · not_started: 1 · invalidated: 0
- progress (overall): {progress_overall:.2f}
- progress (CODING_CATEGORIES = product+validation+investigation): {progress_coding:.2f}

Was there ever a temptation to use `final_success` as evidence? **No.**
We knew throughout the walk that the run failed, but the relevant
fact for annotation is "validation leaf was never started", which is
visible in the trace independent of the upstream label.

### 8. Schema gaps observed

None observed. The combination of "leave validation at not_started" +
"record the hidden-work gap in run_notes.md" expressed the entire
shape of this failure cleanly. No category, status, or event-type
gap.

(See `swe_agent_pilot_s_01` § 8 for the resolved-and-no-longer-
applicable `eval_output.txt` / `test_output.txt` history.)
"""


# ---------- D5 quality artifact ----------


@dataclass(frozen=True)
class AnnotationMeta:
    pilot_id: str
    instance_id: str
    upstream_success_label: Optional[bool]
    annotation_time_minutes: int
    number_of_uncertain_events: int
    number_of_evidence_gaps: int
    whether_final_success_used_only_at_end: bool
    whether_progress_forced: bool
    whether_schema_gap_found: bool
    run_notes_body: Callable[[float, float], str]


def write_quality_json(meta: AnnotationMeta, run_dir: Path, n_subtasks: int) -> None:
    body = {
        "annotation_time_minutes": meta.annotation_time_minutes,
        "number_of_subtasks": n_subtasks,
        "number_of_uncertain_events": meta.number_of_uncertain_events,
        "number_of_evidence_gaps": meta.number_of_evidence_gaps,
        "whether_final_success_used_only_at_end": meta.whether_final_success_used_only_at_end,
        "whether_progress_forced": meta.whether_progress_forced,
        "whether_schema_gap_found": meta.whether_schema_gap_found,
    }
    (run_dir / "annotation_quality.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------- driver ----------


def emit_one(annotator: Callable[[], Tuple[LedgerSession, AnnotationMeta]]) -> None:
    session, meta = annotator()
    run_dir = PILOT_ROOT / meta.pilot_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir {run_dir} not found; run import_swe_agent_trace first")

    session.export_jsonl(str(run_dir / "ledger.jsonl"))

    # ledger-run export-run produces progress.csv, progress_by_category.csv,
    # summary_by_category.json from ledger.jsonl.
    subprocess.run(
        ["uv", "run", "ledger-run", "export-run", str(run_dir)],
        check=True,
        cwd=str(REPO_ROOT),
    )

    overall = session.score()
    coding = session.score(categories=CODING_CATEGORIES)
    (run_dir / "run_notes.md").write_text(
        meta.run_notes_body(overall.progress, coding.progress),
        encoding="utf-8",
    )

    write_quality_json(meta, run_dir, n_subtasks=len(session.ledger.subtasks))

    print(
        f"[annotate_pilot_zero] {meta.pilot_id}: "
        f"{len(session.ledger.subtasks)} subtasks, "
        f"progress={overall.progress:.3f} (coding={coding.progress:.3f})"
    )


def main() -> int:
    emit_one(annotate_s_01)
    emit_one(annotate_f_01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
