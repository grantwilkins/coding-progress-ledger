# Stage 1C follow-up

The coding run completed all 648 scenarios and 1,296 session moves. The local
raw results are in `queue-haul/outputs/coding-run`. Do not
launch another large GPU run to replace data that can be corrected during
reduction.

Before using the run:

```bash
uv run pytest queue-haul/tests
uv run python queue-haul/stage1c_controller.py reduce \
  --run-root queue-haul/outputs/coding-run
uv run python queue-haul/stage1c_profile_fit.py \
  --run-root queue-haul/outputs/coding-run \
  --profile queue-haul/profiles/gpt_oss_20b_a100_tp1.json
uv run python queue-haul/stage1c_controller.py reduce \
  --run-root queue-haul/outputs/coding-run
```

Use `migrations.csv` for per-session measured prompt tokens, processed tokens,
new catch-up tokens, initial KV bytes, catch-up cache hits, time to first
response, response time, service pause, and route-switch time. Use
`scenarios.csv` for exact proxy KV
bytes, achieved rate, completion, continuation, power, and energy. Catch-up
cache-hit bytes are not transferred bytes.

Inspect these aggregate figures:

- `initial_time`: measured work versus request to first response.
- `throughput`: achieved KV and replay rates.
- `concurrency_scaling`: paired concurrency comparisons only.
- `service_effects`: catch-up work, pause, and continuation time.
- `power_energy`: migration time, power, and energy relative to idle.
- `model_check`: current timing equation versus measured time, restricted to
  concurrency one, no activity, and the profile's recorded valid range.
- `profile_evaluation`: the serial model fitted on repeats 0–1 versus repeat 2.

## Next targeted GPU job

`DATA_TO_COLLECT.md` is the complete measurement list; this section is the next
job only.

Reuse `outputs/coding-manifest.json`. Pin
`codex:e381cc89-38ef-e67e-79b9-4b800369b4f5` at turns 0 and 60 and run the
30-scenario serial KV/replay, 1/10 Gbps crossover documented in `README.md`.
The batch profiles two 60-second empty-awake/sleep pairs before migrations on
the same loaded stack. Do not include parallel KV until independent
connections pass a separate concurrency-two smoke.

Any new run must first pass two-model smoke, exact cache-hit checks, complete
reduction, and `uv run pytest queue-haul/tests` from a clean commit.
