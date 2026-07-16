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

## TODO: next targeted GPU job

`DATA_TO_COLLECT.md` is the complete measurement list; this section is the next
job only.

Do not submit the next large job until its small plan and two-model smoke pass.
Keep each missing measurement as a separate plan group so a failed group does
not contaminate the others.

- Catch-up: add controlled turns of 32, 128, 512, and 2,048 measured prompt
  tokens. Record processed tokens, cache-hit bytes, request-to-first-response
  time, response time, and service pause for each method.
- Job types: run the serial timing cases for interactive coding, coding, and
  agentic tool-loop sessions. Fit and validate each job type separately before
  deciding whether they can share one curve.
- Full drain and final state: move all eight sessions, then measure awake,
  sleep, and shutdown completion separately. Record the last route switch,
  transition start, transition end, and the complete source power trace.
- Paired comparisons: choose each session and turn without using method or
  bandwidth in the random seed. Run the same session, turn, and repeat for both
  methods and every bandwidth before estimating a method or bandwidth effect.
- Parallel KV transfer: give each simultaneous lookup an independent LMCache
  connection. First prove overlapping transfer intervals and correct aggregate
  proxy bytes in concurrency-two smoke; then test concurrency four. Keep the
  simulator and checked profile at one KV transfer until this passes.
- Datasets: add broader measured prompt sizes and longer natural catch-up turns;
  do not infer these from requested context size.

Any new run must first pass two-model smoke, exact cache-hit checks, complete
reduction, and `uv run pytest queue-haul/tests` from a clean commit.
