# Stage 1C follow-up

The coding run completed all 648 scenarios and 1,296 session moves. The local
raw results are in `queue-haul/outputs/queue-haul/outputs/coding-run`. Do not
launch another large GPU run to replace data that can be corrected during
reduction.

Before using the run:

```bash
uv run pytest queue-haul/tests
uv run python queue-haul/stage1c_controller.py reduce \
  --run-root queue-haul/outputs/queue-haul/outputs/coding-run
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

Run a new targeted GPU profile only if one of these decisions requires it:

- Exact-size prompts are needed beyond the measured trace range.
- Methods or bandwidths must be compared as isolated effects; then pair the
  same sessions across those conditions.
- The paper must claim simultaneous KV traffic; then use independent connector
  connections and verify aggregate proxy traffic.

Any new run must first pass two-model smoke, exact cache-hit checks, complete
reduction, and `uv run pytest queue-haul/tests` from a clean commit.
