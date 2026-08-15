# Live multiaxis repair hardware results (2026-08-15)

This is the reduced evidence bundle from the completed 36-episode A100 run at
`/datadrive/qh-repair-20260815-r1`.  The frozen launch plan is
[`../repair-stress-multiaxis-hardware-20260814/plan.json`](../repair-stress-multiaxis-hardware-20260814/plan.json),
created from implementation commit `9a139cdb` and committed as `09de43be`.

The primary cutoff result is unambiguous: repair reached the modeled
completion-credited 30.928 W shed target by 25 seconds in 18/18 episodes,
while the paired no-repair control missed it in 18/18 episodes.  Every fault
acknowledgment, live frozen-plan check, solver/apply timestamp, HTTP response,
and TTFT record passed its integrity gate.  The run contains 414 request TTFT
rows and 186 common-session TTFT rows.

`validation.json` intentionally retains `passed: false`.  This is caused by
secondary preregistered safety-margin gates, not by the primary 25-second
endpoint:

- all six joint pairs passed;
- five of six prefill pairs passed, with the remaining repair reaching target
  at 20.079 s rather than the stricter 20.000 s qualification margin;
- bandwidth seed 553 repaired by 22.47--22.69 s and its controls reached target
  at 41.78--44.37 s, but repair exceeded the 20 s qualification margin;
- bandwidth seed 819 was weakly discriminating: repair reached target near
  22.8 s versus control near 27.0 s, and the control cutoff shortfall was only
  1.027 W (3.32%).

The bundle includes the final validation and summary tables plus all 36
per-episode `result.json` files.  Large diagnostic source-power, sink-load,
proxy, and engine traces remain in the durable `/datadrive` run root and are
not duplicated in git.  Shed values are workload-model completion credits;
direct A100 board-power samples are diagnostic and remain labeled separately.
