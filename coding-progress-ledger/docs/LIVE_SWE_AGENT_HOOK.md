# N3 — SWE-agent live sidecar hook

This documents the N3 hook from SWE-agent-shaped execution into the live ledger sidecar.

## What shipped

`scripts/run_swe_agent_live_sidecar.py` reads a run directory containing `normalized_trace.json`, converts assistant/tool turns into the N1 wire format, and streams those events through `LedgerSidecar` with the `swe_agent` adapter.

The hook writes:

```text
runs/swe_agent_live/<instance_id>/
  wire_events.jsonl
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  live_instrumentation.json
  plus copied source context artifacts
```

The source run is never mutated. Existing live ledgers are not overwritten.

## Why this is the N3 hook

The adapter boundary is the stable N1 JSONL protocol, not SWE-agent internals. A real SWE-agent process can replace the normalized-trace reader by printing the same `wire_events.jsonl` lines to stdout:

```bash
agent | python -m ledger_progress.sidecar --run-dir runs/swe_agent_live/<id> --adapter swe_agent
```

For N3, existing known-success and known-failure SWE-agent pilot traces supply the SWE-agent-shaped step stream. The script emits fresh wall-clock timestamps while streaming, so the resulting ledger exercises the live sidecar path and unlocks Workstream V timestamp features without editing SWE-agent source code.

## N3 artifacts

Generated runs:

```text
runs/swe_agent_live/Melevir__cognitive_complexity-15   # upstream success
runs/swe_agent_live/WIPACrepo__iceprod-339             # upstream failure
```

Both pass:

```bash
uv run ledger-run check-run <run_dir>
```

Both ledgers have non-null timestamps on every event.

## Submit-without-validation policy

Raw-step live instrumentation does not invent discovered-but-unattempted validation obligations. If an agent edits and submits without running validation, the sidecar records the emitted product/artifact actions and the resulting shape is `no_validation_frontier`, not a synthetic `validation_gap`.

Agents that can declare intent may still emit explicit `ledger_ops` to add an unattempted validation leaf. That higher-fidelity mode is opt-in; the default sidecar path stays a measurement layer over visible actions.

## Non-goals

- This hook does not estimate completion probability.
- It does not use `final_success` as a feature or ledger event source.
- It does not patch SWE-agent internals.
- It does not implement long-lived file tailing; stdin streaming is already supported by `ledger_progress.sidecar`, and finite `--input-file` remains batch-only.
