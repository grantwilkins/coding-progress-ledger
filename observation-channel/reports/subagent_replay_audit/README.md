# Subagent Replay Audit

Primary question: If a blind evaluator sees one SWE-Agent turn at a time, how does its believed progress move during the trace?

Trace: `swe-agent:000001:AnalogJ__lexicon-336` from `observation-channel/data/turns/swe-agent/AnalogJ__lexicon-336__2.jsonl`.

High-level task: fix dns-lexicon's Memset provider path where create can return a raw record id string and crash default table formatting, then validate and submit.

This replay is a negative example, not a successful solve. The canonical trace ends in repeated inspection/navigation with no observed correct fix, validation, cleanup, or submit action.

## Anti-Leakage Rules

- The subagent was run without inherited parent context.
- It received one canonical turn at a time.
- It was not told the final trace length until the final available turn.
- Per-turn prompts removed final-status metadata and did not reveal future observations.
- The task description exposed only the issue-level goal, not the trace outcome.

## Artifacts

- `replay_progress.csv`: per-turn subagent estimates and confidence.
- `replay_progress.png`: percent-finished curve with confidence shading. Narrower shading means higher confidence.

## Run Summary

- turns replayed: 51
- peak estimated progress: 35%
- final estimated progress: 30%
- main qualitative signal: progress jumped when the root cause became visible, dropped during failed irrelevant debug edits, then plateaued during repeated file-navigation turns.
