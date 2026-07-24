# Queue-Haul findings

## Decision

The recovered 2026-07-23 archive changes the diagnosis but still does not
justify an accepted destination profile.

- Do not launch the current reserve; its task list is ignored and it would
  repeat the full campaign.
- Do not collect more migration timing for the measured low-concurrency case.
  The raw records support conservative replay and KV timing envelopes and a
  method-affinity rule for live traffic.
- Before any service rerun, replace the six invalid forced-token signatures and
  prevent repeated appended prompts from surviving in APC across cells. If a
  measured boundary is required, rerun only the unresolved frontier.

All 767 artifacts listed in `SHA256SUMS` match; the archive contains those
artifacts plus the manifest itself. The local `data/` copy is ignored by Git
and is not part of the evidence commit.

## Service evidence

The 47 summaries classified infeasible contain 50 requests with HTTP status
200, no recorded error, and zero prompt and output usage. The client requested
a forced token with `ignore_eos=true`, so these are not legitimate early model
stops. The stream ended without the required work or usage record. The other
9,131 of 9,181 requests produced exactly their planned output.

Every empty response is one of six deterministic
`(session, request_index, forced_token)` signatures. Their forced token IDs are
200110–200952; no successful request uses a forced ID at or above 200000. This
is a harness/token-eligibility defect, not evidence of overload. The
descriptive incidence is 11/470 agentic requests, 18/5,065 coding requests, and
21/3,646 interactive-coding requests.

This separates two questions that the old reduction conflated:

1. **Measurement validity:** an empty forced-token stream invalidates the
   probe; it does not establish production availability.
2. **Consumable capacity:** a run with missing work is not a capacity-boundary
   observation.

The 66 complete-work runs all pass normal, emergency, and stability, but
request completeness is not sufficient for service-capacity evidence. The
runner prewarms only each session's historical prefix, while repeated request
indices produce the same appended prompts across cells and vLLM APC is not
reset. Later cells can therefore reuse nominally future append blocks.

For 16-token blocks, a request is consistent with the intended private-prefix
state only when

```text
cached_tokens <= floor((prompt_tokens - input_tokens) / 16) * 16
```

The run is the independent unit: one append-hot request contaminates the whole
run. The reproducible `service_cache_state` reduction gives:

| Run state | Runs | Meaning |
|---|---:|---|
| measurement-invalid | 47 | at least one missing-work response |
| append-hot | 60 | at least one cached block extends into the new append |
| exact private prefix | 5 | every request matches the intended warmed prefix |
| prefix under-hit | 1 | no append reuse, but one request lost intended prefix blocks |

Across all 9,181 requests, the corresponding request counts are 50 invalid,
8,020 append-hot, 1,110 exact-private-prefix, and one prefix under-hit. Request
counts inside a contaminated run are descriptive only; they are not additional
independent evidence.

Only six runs are usable as private-prefix-or-colder service observations:

| Affinity | Split/cell | Radius | Requests | Cache state |
|---|---|---:|---:|---|
| interactive coding | fit/emergency | 0.114063 | 83 | exact private prefix |
| coding | tune/normal | 0.096953 | 78 | exact private prefix |
| agentic tool loop | tune/normal | 0.096953 | 27 | exact private prefix |
| interactive coding | validation/normal | 0.096953 | 48 | exact private prefix |
| coding | validation/normal | 0.096953 | 18 | one prefix under-hit |
| agentic tool loop | validation/normal | 0.096953 | 11 | exact private prefix |

All six pass every policy. Their worst p90 TTFT is 0.587 s, worst p90 mean
TPOT is 0.0248 s/token, and largest queue-drift upper bound is 0.00163
requests/s. The common held-out observation is 0.096953. Interactive coding
also has one fit-only success at 0.114063. These are observed inner points, not
an accepted envelope: there are only two usable physical runs per affinity and
no cache-valid failure.

The earlier 0.108677–0.182805 affinity maxima came from append-hot runs and are
withdrawn as private-prefix capacity evidence. The data do not justify more
service facets or a learned cache-work model. For current analysis, keep
measured cold `F(T)`, `G(T)`, and KV capacity, and report:

- **observed inner:** at or below a cache-valid point in the same measured
  affinity/domain;
- **possible/sensitivity:** dependent on an assumed service envelope or
  interpolation; or
- **unsupported:** outside the measured domain or based on an invalid or
  append-hot probe.

## Migration timing

Twelve of 18 migration treatments overlap a foreground request. Ten have one
request active at migration start, and two start a request during migration.
All 36 control and treatment foreground requests reuse cached prefixes; the 20
treatment requests reuse 2,608–3,168 cached tokens. Twenty-six of 27 first
engine samples record the intended 264,699 prompt tokens, 18 generated tokens,
and 18 successful prewarm requests. One control contains additional prior state
and is not cache-state-identical to its treatments. The data proves reuse, not
exact idle residency: the GPU cache gauge reports zero for evictable cached
blocks. The intended 264,699-token prewarm is 21.79% of declared capacity but
must not be modeled as measured resident occupancy. Six treatments have no
foreground overlap; four complete no foreground request at all.

The recorded `achieved_rho` remains the wrong timing covariate. It is a
preceding 30-second average divided by the rejected service bound, and it
counts cached prompt tokens as new compute. Subtracting cached prompts gives
prewindow compute rho from 0 to 0.898. Exact migration-interval foreground work
is not recoverable: aggregate request totals have no per-token timestamps,
engine counters include migration work, and 8/18 samplers end 39–222 ms before
route switch. The archive identifies overlap topology, not a load curve.

The smallest reliable duration models remain method-specific:

- Keep the reused replay compute-plus-completion context curve. On the six 16K
  rows, central and conservative calibration factors are 0.564571 and
  0.586660 after separating route bytes. Applied to the untouched 24K curve,
  they have 5.5% and 9.6% held-out median error and never underpredict the
  three held-out runs. One fitting context cannot identify a separate replay
  intercept.
- Model KV duration as `sealed_bytes / route_bytes_per_s + c`. The six 16K
  rows give `c=0.961186 s` centrally and `c=1.133822 s` conservatively. The
  24K held-out median errors are 2.4% and 7.8%; the conservative form never
  underpredicts.

A categorical overlap split improves central replay held-out median error from
5.5% to 4.2%, but its 16K idle fit contains one row. The analogous KV split
underpredicts two held-out rows. That is insufficient evidence for another
planner coefficient; the global conservative envelopes already cover both
idle and observed busy executions.

Exact route time must remain outside runtime calibration. The current
`LoadedCoefficients` multiplies the complete migration duration and can predict
less than `bytes/link_rate`, so it cannot safely encode the KV model. Replay
log transfer and switch time likewise remain physical components.

## Foreground impact and affinity

Control and treatment runs provide ten exactly matched foreground requests per
method, six of which overlap migration.

- For five already-active replay requests, paired TPOT increased by a median
  3.45 ms/token and at most 6.79 ms/token. The five KV counterparts increased
  by a median 0.42 ms/token and at most 0.67 ms/token.
- The one request arriving during replay reconstruction incurred 1.084 s
  additional TTFT. The same scheduled request arriving during KV transfer
  incurred 4.7 ms additional TTFT.

These are observations, not percentile bounds. They support a simple
eligibility rule: prefer KV transfer for latency-sensitive busy destinations;
allow replay only on an idle/drained destination or when explicit TTFT slack
covers the empirical replay penalty plus uncertainty. Foreground impact should
not be hidden inside a service-capacity facet.

## Remaining evidence

No additional migration campaign is needed for component timing or method
ranking on the recorded concurrency-one schedules in the measured
16K/10-Gbps and 24K/5-Gbps cells. Higher concurrency, a new foreground
schedule, continuous load claims, or an interference percentile requires new
evidence.

Service capacity remains right-censored. Retain the six cache-valid runs and
exclude append-hot cells from capacity reduction. No rerun is needed for
explicit sensitivity modeling at the observed 0.096953 inner point. If an
accepted boundary is required, first select safe forced tokens, reset APC or
make appended prompts unique across cells, and hard-fail missing work or cache
reuse beyond the warmed prefix. Start near 0.096953, expand until a failure is
bracketed, and collect three independent runs only around each affinity's
boundary. Compute normal, emergency, and stability from every physical run.
