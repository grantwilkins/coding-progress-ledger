# Queue-Haul open measurements

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. It is
usable only over its recorded context, load, and concurrency ranges.

- `TODO(profile)`: measure action power, route switch, sleep, and shutdown.
- `TODO(concurrency)`: fix LMCache 0.3.3 concurrency-2 metadata reads, then
  measure replay and KV concurrency.
- `TODO(context)`: measure request and replay rates beyond 31.6k tokens.
- `TODO(public-profiles)`: add a cited public-benchmark importer for scaled and
  sensitivity-only model profiles.
- `TODO(routes)`: measure external-log, heterogeneous destination, rack, and
  site paths.
- `TODO(wake)`: fit first-request timing from complete session traces.
- `TODO(load-cycle)`: include measured service time in expected request load.
- `TODO(request-power)`: determine whether sampled requests require dynamic
  power updates.
- `TODO(transition-power)`: replace sleep and shutdown steps with measured
  power traces.
- `TODO(workloads)`: replace the small assumed workload records with held-out
  complete session traces.
- `TODO(tp-topology)`: build and validate multi-GPU model and network layouts.
