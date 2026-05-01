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

For N3, existing known-success and known-failure SWE-agent pilot traces supply the SWE-agent-shaped step stream. The script emits per-event timestamps while streaming, so the resulting ledger exercises the live sidecar path and unlocks Workstream V timestamp features without editing SWE-agent source code.

## Wall-clock vs replay-time timestamps

The default replay path stamps each event with `datetime.now(UTC)` at materialization, which produces microsecond-spaced timestamps because the replay loop runs in-process. Those timestamps satisfy the schema requirement that every live event carry a non-null `timestamp`, but they do not reflect actual SWE-agent execution time and so V1's `seconds_since_progress_increase` reads near zero on those runs.

For N6, the script gained a synthetic-clock override:

```bash
uv run python scripts/run_swe_agent_live_sidecar.py \
  --source-run-dir runs/swe_agent_pilot/<pilot> \
  --output-run-dir runs/swe_agent_live_wallclock/<instance_id> \
  --synthetic-clock-start 2026-05-01T00:00:00+00:00 \
  --synthetic-step-seconds 30
```

Each wire event advances the synthetic clock by `--synthetic-step-seconds`, so per-step intervals fall into a regime where V1's wall-clock columns are physically informative (>1 second per step). `live_instrumentation.json` records `timestamp_source` (`replay` or `synthetic`), `first_event_timestamp`, `last_event_timestamp`, and `timestamp_span_seconds` so consumers can tell which mode produced a run dir.

The synthetic-clock mode is **not** real wall-clock data; it is a calibration-friendly stand-in for traces that lack per-step timestamps in the upstream metadata. A future hook against a freshly-running SWE-agent process would replace `_utc_now` with real `datetime.now()` calls between steps and produce true wall-clock data via the same code path.

## N3 artifacts

Generated runs:

```text
runs/swe_agent_live/Melevir__cognitive_complexity-15   # upstream success (replay-time timestamps)
runs/swe_agent_live/WIPACrepo__iceprod-339             # upstream failure (replay-time timestamps)
runs/swe_agent_live_wallclock/<instance_id>            # 20 runs, synthetic-clock timestamps (N6)
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
