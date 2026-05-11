# Solved Subagent Replay Audit

Primary question: On a solved SWE-Agent trace, does a blind turn-by-turn evaluator recover a successful progress curve?

Trace: raw SWE-Agent row `349`, `bihealth__biomedsheets-23`, from the local cached `nebius/SWE-agent-trajectories` train split.

High-level task: fix biomedsheets' germline TSV reader so SODAR sample sheet identifiers can contain hyphens in `patientName`, `fatherName`, and `motherName`.

This is a solved benchmark example. The dataset row has `target=True`, and the hidden evaluator log reports `tests/test_io_tsv_germline.py` with `7 passed`.

## Anti-Leakage Rules

- The subagent was run without inherited parent context.
- It received one canonical turn at a time.
- It was not told the final trace length until the final available turn.
- Per-turn prompts removed final-status metadata and did not reveal future observations.
- The task description exposed only the issue-level goal, not the dataset success label or evaluator logs.

## Artifacts

- `replay_progress.csv`: per-turn subagent estimates and confidence.
- `replay_progress.png`: percent-finished curve with confidence shading. Narrower shading means higher confidence.

## Run Summary

- turns replayed: 19
- peak estimated progress: 86%
- final estimated progress: 86%
- hidden evaluator result: `7 passed`
- main qualitative signal: progress rises sharply when the exact three validation checks are found, dips on rejected partial edits, then rises after each successful targeted edit. It does not reach 100% because the visible trace submits without running tests.
