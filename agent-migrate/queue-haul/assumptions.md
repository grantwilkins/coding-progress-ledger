# Queue-Haul open measurements

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. It is
usable only over its recorded context, load, and concurrency ranges.

- `TODO(profile)`: fit action power and timing from the corrected Stage 1C
  tables; still measure sleep and shutdown.
- `TODO(profile-datasets)`: collect exact-size prompt datasets only if the
  measured trace range is insufficient for the simulator or paper.
- `TODO(pairing)`: pair identical sessions across methods or bandwidths only if
  those effects must be separated rather than reported as associations.
- `TODO(parallel-kv)`: determine whether the paper requires simultaneous KV
  traffic; if it does, profile independent connector connections.
- `TODO(context)`: measure request and replay rates beyond 31.6k tokens.
- `TODO(public-profiles)`: add a cited public-benchmark importer for scaled and
  sensitivity-only model profiles.
- `TODO(routes)`: measure external-log, heterogeneous destination, rack, and
  site paths.
- `TODO(wake)`: fit first-request timing from complete session traces.
- `TODO(load-cycle)`: include measured service time in expected request load.
- `TODO(max-ell)`: validate the 5-second latency admission point against held-out
  multi-session SLO runs; measured window sensitivity spans 0.439 to 0.623.
- `TODO(prefix-sharing)`: derive exact shared-prefix blocks from tokenized traces;
  current capacity sizing conservatively counts every active context in full.
- `TODO(request-power)`: determine whether sampled requests require dynamic
  power updates.
- `TODO(transition-power)`: replace sleep and shutdown steps with measured
  power traces.
- `TODO(workloads)`: replace the small assumed workload records with held-out
  complete session traces.
- `TODO(tp-topology)`: build and validate multi-GPU model and network layouts.
