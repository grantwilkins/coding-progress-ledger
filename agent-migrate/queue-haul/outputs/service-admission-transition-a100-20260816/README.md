# A100 service-admission transition confirmation

This is a nine-cell confirmation of three discrete destination-service
recipes, not a fitted frontier or an end-to-end migration experiment. The
frozen treatment is:

- prefill-heavy `(w_F,w_D)=(0.3078,0.1919)` GPU-s/s;
- balanced `(w_F,w_D)=(0.1996,0.3003)` GPU-s/s;
- decode-heavy `(w_F,w_D)=(0.08163,0.41806)` GPU-s/s;
- three new restart blocks per mix;
- 60 seconds of incumbent-only traffic;
- synchronized cold materialization of eight new prefixes within a fixed
  30-second window;
- 240 seconds of sustained incumbent and added-cohort traffic.

All three blocks of a recipe must preserve the frozen 1-second P90 TTFT and
100-ms P90 mean-TPOT rules for both cohorts, complete and cache every offered
request, retain exact token timing and complete telemetry, and satisfy the
strict queue-stability rule. Actual materialization request starts and spread
are checked from raw records. Every invalid instrumentation attempt remains in
its own attempt directory; a failed materialization is a valid failed outcome
and is not retried away.

The campaign may establish only that eight prefixes were materialized and
their offered workload was sustained for 240 seconds at each passing recipe.
Each recipe's coordinates sum to approximately 0.50 by construction, but that
scalar is neither measured utilization nor an admission limit.
It does not establish a planner decision, reservation lease, source power
action, migration, route switch, arbitrary session count, indefinite service
guarantee, interpolated service envelope, or universal utilization limit.

Raw evidence is written under
`/datadrive/qh-service-admission-transition-a100-20260816-r1`. The reduced
result will be `summary.json`; it always retains `planner_usable=false` and
`supported_envelope=null`.
