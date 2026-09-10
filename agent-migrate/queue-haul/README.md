# Queue-Haul

Queue-Haul plans request-boundary migration of stateful LLM sessions to reduce
source accelerator power before a deadline. It jointly chooses sessions,
replay or full-KV transfer, and compatible destination serving pools.

For live experiments, the planner emits a fixed plan naming each migrated session, replay or KV
transfer, destination pool and replica, route, and order. Execution adds phase
timing, commit, observed first-token timing, source transitions, resource use,
debt, recovery, achieved shed, and unmet shed. A requirement frontier summarizes
plans across source-power targets. Destination capacity is an advertised pool
contract, not an inferred GPU inventory.

## Pooled 2 MW agentic GPT-OSS/A100 simulation

`pool_shed_campaign.py` compares QH LP, QH greedy, KV-only, replay-only, and
isolated-fastest. Sweden Central and each destination (East US 2 and Germany
West Central) each have 6,666 A100 GPUs at 300 W nameplate: 1.9998 MW per site,
excluding CPUs and peripherals. This is the requested A100 study, not the
original H100 configuration. Eight GPUs share each modeled node. The next campaign
uses the recorded `coding` and `coding_long` agentic trajectories; `measured_pack`
remains a hardware-validation workload. The default output is
`outputs/a100-pooled-agentic-2mw`. The full campaign remains stopped: the replay
diagnosis below is complete, but fleet resident latency is not validated.

The [GPU-local queue pilot](outputs/a100-resident-queues-20260910/report.json)
now reconstructs all 2,865 offered service requests from 20 episodes, including
33 unfinished or failed requests, and reproduces all 112 existing service
windows. It keeps resident histories on their GPU, serializes dependent turns,
and shares the 8,192-token iteration budget between partial prefills and decode.
Timing fits use seed 7101; seed 7102 remains outside fitting. Native cache hits
and incoming dispatches are observed inputs to this pilot, not fleet forecasts.

The model predicts **20.09 s resident P90 TTFT versus 21.43 s measured** during
the difficult long-replay burst. It still fails six of 266 held-out metric checks,
and the successful high-load coding scout also misses its TTFT check. These
overlapping checks are not independent statistical trials. In one 1,233-token
resident turn, a small decode-rate error advances completion by 2.1–4.4 s and
then advances dependent turns. Some client streams also compress token delivery:
an 18-token response contains nine sub-millisecond gaps. Client TPOT therefore
does not identify physical GPU iteration timing, even with individual token
frames. The fit remains frozen; the pilot is **not installed in fleet execution**.

Reproduce the compact extraction and bounded CPU validation with:

```bash
uv run python pool_shed_resident_data.py
uv run python pool_shed_resident_validation.py
```

The remaining timing ambiguity is server scheduling/generation versus buffered
client delivery. The [bounded GPU handoff](#bounded-resident-server-timing-handoff)
freezes twelve warm-decode cells and four agentic episodes to resolve it. Another
GPU measurement is only essential if existing timing uncertainty changes the
end-to-end conclusions; GPU-local KV placement, causal source queues and bounded
validation still need integration either way. Failed transport/dependency scout
cases do not establish GPU capacity. Full `run` calls hard-fail while resident
latency is unvalidated; `prepare`, historical reduction, smoke runs and bounded
`run_cell` audits remain available.

Resident service shares compute with migration. Replay uses the existing measured
resident-throughput loss; spare capacity repays resident debt before migrated
request buffers. Requests arriving after source quiescence are buffered and move
to the destination at handoff. There is no default isolation reservation or
pre-handoff buffer gate. The planner reserves recovery work over every remaining
time interval so earlier idle service cannot pay for later arrivals. This is an
aggregate recovery approximation: executed handoff, `service_ready_s`,
`service_recovered_by_deadline`, and resident latency validity are separate
outputs. **Resident TTFT/TPOT is not validated.** KV network waiting uses no
compute; KV ingest is omitted.

The [completed replay diagnosis](outputs/a100-replay-resolution-20260910/report.json)
reproduces all 20 archived 2/2-MW policy evaluations and adds eight controlled CPU
comparisons. Run `uv run python pool_shed_replay_resolution.py --out /tmp/replay-audit`
with a fresh output directory to reproduce it. Timing coefficients, full-context
replay and the effective 0.80-decimal-GB/32K wire assumption are unchanged. The
audit distinguishes these causes of apparently fast replay:

- **Handoff is earlier than service recovery.** At the 120-second coding deadline,
  replay finishes ownership transfers at 15.15 s but aggregate service recovers at
  58.15 s. QH reaches both at 50.58 s. At 30 seconds both hand off 100%, while QH
  retains 148.3 and replay 906.9 normalized work-seconds of request buffers.
- **Resident service is ideally pooled across GPUs.** Spare GPUs compensate the
  throughput lost on migration GPUs before resident debt is reported. For long
  replay at 30 seconds, the model displaces 80,941 work-seconds of resident service
  and cancels 63,482 through this assumption. Resident histories on the measured
  GPU cannot use that spare capacity elsewhere without moving their KV. Execution
  and campaign tables now expose `resident_displaced_work_s`,
  `resident_pool_compensation_work_s`, `peak_migration_replicas` and the explicit
  aggregate-only `service_recovery_scope`. These are diagnostics, not a new
  queue-delay penalty. The remaining modeled debt can recover while individual
  residents have already missed their latency target.
- **The resource ratio favors parallel rebuilds.** Two 2-MW destinations at half
  nominal load have more standing spare service than the 2-MW source uses at 80%.
  The source's eight sessions/GPU can spread over twice its GPU inventory. At the
  0.80-GB wire anchor, the initial coding and long snapshots contain 18.94 and
  36.28 TB of KV versus 1.56 and 2.98 GB of replay text. The KV payloads alone take
  151.5 and 290.3 s at a shared 1-Tbit/s cap if those snapshots must all move;
  evolving snapshots, endpoints and catch-up require separate accounting.
- **The load and timing transfers are generous.** Nominal half-load offers
  0.2213 coding or 0.1882 long requests/s/GPU. The inherited timing conversion
  increases replay time by just 0.68% or 0.52%; the latest coding hardware used
  twice that coding RPS. Regional calibration also speeds replay by about 22%.
  Removing only that regional discount lowers long replay's 30-second handoff
  from 87.31% to 76.40%, and QH from 93.97% to 83.39%.
- **Completion protocols and context lifecycles differ.** Unloaded width-eight
  cold replay has 0.35-s median absolute held-out error; a blanket replay slowdown
  is unsupported. Loaded probes often generate 512 tokens, while simulator replay
  has a short measured completion overhead. Natural warm-prefix hits reduce
  rebuild work even when the entire context is submitted. The full-rebuild
  simulator can overcharge warm catch-up. Source request durations remain roughly
  0.26/0.30-s throughput proxies, and cyclic traces can wrap to short histories;
  `coding_long` describes the initial state, not a permanent context floor.

At 30 seconds with the requested 2/2-MW inventory, QH hands off 93.97% of the long
workload and replay 87.31%. Halving only each destination to 1 MW changes coding
to QH 60.44% / replay 56.76%, and long to 55.92% / 49.97%; the standing-service
ceiling becomes 62.5%. This is a labeled capacity sensitivity, not the primary
configuration. QH need not strictly dominate executed replay: its temporal LP
optimizes a forecast and commits admissions incrementally. In the no-regional-
discount coding sensitivity, replay still exceeds QH (99.87% versus 97.50%).

The new hardware itself has replay ownership handoffs in 42.9–60.2 s versus
105.6–135.8 s for KV under its measured 1-Gbit/s GET cap. Its long replay resident
P90 TTFT reaches 14.07 s in one repeat versus 0.335 s in the matched control.
That is the discrepancy that needs a replica-local service model: fast handoff
can coexist with poor resident service. The existing paired traces are sufficient
to establish this diagnosis; another broad GPU campaign is not needed for it.

The review also reproduced a pre-existing HiGHS `Unknown` failure in the refined
60-second coding/isolated-fastest check. Both original solver attempts disabled
matrix scaling. A final equilibrated-simplex attempt now solves the same bounded
LP, retaining the original resource and primary-objective certificate. The
captured 888-variable regression has a primal residual below 1.3e-13 and a dual
gap below 5e-9; the [previously failing policy case](outputs/a100-replay-resolution-20260910/resolution-regression.json)
now completes with a resource residual below 5.8e-11. This is a numerical fix;
the default successful solver path and timing coefficients are unchanged.
[Focused tests](outputs/a100-replay-resolution-20260910/focused-tests.log) cover
execution, planning and a complete five-policy single-cell reduction. The three
broader integration tests remain excluded from that focused run; the full
multi-setting resolution sweep is not certified by this bounded regression.

`Fleet.metadata["destination_gpus"]` independently sets the GPU count at each of
the two equally sized destinations; it defaults to the source count. The bounded
audit also uses 33,333 GPUs (10 MW nameplate) per destination with the source
unchanged at 66,666. Destination compute, resident memory, and endpoint inventory
scale together. Route endpoints use the smaller source/destination node inventory;
the shared source egress and configured WAN caps remain in force.

The [bounded realism audit at commit 0614039d](outputs/a100-replay-realism/audit.json)
uses the width-eight packing, loaded/full-drain, regional interference, and SLO
campaigns without refitting their coefficients. Run
`python pool_shed_replay_audit.py` at that revision to reproduce it; the large campaign remains
stopped. The prior v8 sensitivity/audit artifacts remain unchanged.
[181 focused tests passed at that revision](outputs/a100-replay-realism/focused-tests.log), with
the same three broader integration tests excluded. A separate
[single-cell prepare/run/reduce check](outputs/a100-replay-realism/reduction-smoke.log)
completed all five policies in temporary outputs; the audit also reruns the
existing hardware timing and regional resident-deficit checks.

The corrections remove synchronized starts, delta-length replay timing and
implicit resident isolation. For a 256-token append, the old catch-up rule had
the same cost at a 2K or 30K retained context. Full-context catch-up now costs
about **0.557 s at 2,304 tokens and 3.489 s at 30,256 tokens** on the East route,
including the existing measured regional factor. Its request overhead remains
intact. Native cache hits in a real warm-prefix catch-up can change those times;
no synthetic reuse fraction is added.

The 30-second comparisons below use the 0.80 GB/32K effective KV-wire anchor,
shared 1,000 Gbit/s WAN, the existing coding service rates, and a common candidate
library and planning clock within each pair. Numbers are **raw handoff fractions**:

| Source / each destination GPU IT | Workload | Replay-only | QH LP |
|---|---|---:|---:|
| 20 / 20 MW | coding | 100.00% | 99.82% |
| 20 / 10 MW | coding | 58.68% | 59.72% |
| 2 / 2 MW | coding | 96.30% | 100.00% |
| 20 / 20 MW | coding, initially 24K+ | 87.61% | 86.03% |

Every one of these paired outcomes retains buffered work at 30 seconds. For
example, the 20/20-MW replay case hands off by 26.04 s but still has about
2,611 normalized work-seconds of buffered requests. The 20/10-MW standing-service
ceiling is 62.5%, shared by all methods. The audit also runs KV-only, QH greedy,
and isolated-fastest in the two 20-MW-source scenarios. No baseline result is
substituted for QH. These outcomes do not establish SLO-feasible shedding or a
reliable QH advantage; even the revised temporal recovery constraints can leave
queues after execution changes the forecast's timing.
The 2/2-MW replay and QH outcomes retain approximately 1,183 and 54.6 normalized
work-seconds of buffered requests, respectively. Equal reductions in all three
fleets preserve their standing-service capacity ratio; keeping the total WAN
budget fixed increases available bandwidth per GPU.

The stronger measurement warning is the resident load definition. Coding at its
simulated 50% offers about **0.221 requests/s/GPU**, or 112.9 prompt and 8.85
output tokens/s/GPU. The newer A100 SLO campaign's 4K normalizer assigns those
token rates about **0.0092 total offered phase work**, while its confirmed
transition recipes are near 0.50. This comparison is not a coding utilization
estimate: contexts, mixtures and normalizations differ. The SLO confirmation
supports three discrete 4K recipes at 1-second P90 TTFT and 100-ms P90 mean TPOT;
it did not confirm a universal scalar headroom bound. The current simulator
retains the older contextual coding normalization and its explicit transfer gap.

The independent cold-burst evidence also shows real queueing: at 32,256 prompt
tokens, the A100 had at most ten GPT-OSS requests running concurrently, and a
width-16 burst had 65.03-second P90 TTFT. Those requests generated 32 output
tokens; their timing cannot be substituted directly for a migration probe.
Source request durations in the simulator remain throughput-derived proxies.
They are not validated per-request latencies or a source queue model.

The primary references support separating prefill contention, decode latency and
actual prefix reuse. [vLLM's scheduler documentation](https://docs.vllm.ai/en/stable/configuration/optimization/)
describes decode priority with chunked prefill and latency effects from prefill
budgets and recomputation. [Automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
skips cached-prefix prefill work without eliminating decode.
[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
measures the throughput/latency tradeoff from interleaving prefill and decode.
These sources explain mechanisms; they supply no new simulator coefficients.

The requested matched acquisition and bounded policy comparison are complete.
Before claiming SLO-feasible fleet shedding, the remaining work is a model of
resident queues tied to their serving replicas, checked against the collected
control/replay/KV traces with exact RPS, history count, output length and prefix
reuse. Separate rebuild, generation and source quiescence; do not fit a whole
512-token probe as prefill or replace local queues with a global protection pause.
Source durations and arrival timing also require an explicit contract: the
1,655 coding records contain no arrival timestamps or observed cycle restarts.
Keep those assumptions and the roughly 32K context support visible. Acceptance
depends on held-out timing and service fidelity, not QH's ranking. The native KV
geometry and the 0.80-GB effective-wire scenario remain distinct; a private-KV
fraction must not discount the same measurement twice.

Load is a fraction of the measured coding **normal serving envelope**, not
FLOPs or GPU busy time. Request work uses the original context-dependent
prefill/decode rates and live anchors from the destination service campaign,
normalized by its 0.1140625 normal bound. Four recorded coding tune/validation
probes bracket this bound and meet their recorded RPS/latency classifications.
The full destination profile was not accepted: this is a retrospective coding
contract, not a general TTFT/TPOT guarantee. Transferring it to other mixtures
and to shared migration occupancy is explicit. Contexts outside the serving
curves use the slowest measured phase rate and are flagged as extrapolated.

The [2026-09-09 bounded A100 validation](outputs/a100-replay-live-20260909T1920/report.json)
collected 24 unloaded trial attempts, eight scout attempts, 16 destination-only
control/replay episodes and four targeted sensitivities. Acquisition, startup,
reboot and cleanup took **148.6 minutes**, within the 150-minute cap. The GPU
runtime was stopped afterward. Previous results remain preserved.

Full-message catch-up naturally reused native prefixes. The second repeat at
width eight produced the following elapsed times; the final column is the
unchanged simulator's local full-rebuild catch-up prediction:

| Retained tokens | Initial replay | Warm +32 | Cold updated | Full-rebuild prediction |
|---:|---:|---:|---:|---:|
| 2,048 | 2.07 s | 1.26 s | 2.12 s | 1.72 s |
| 8,192 | 5.72 s | 1.12 s | 5.81 s | 6.07 s |
| 30,000 | 32.50 s | 1.41 s | 32.43 s | 35.44 s |

At 30K/width eight, a 2,048-token append took 5.03 s warm versus 35.94 s cold.
[Per-condition observations](outputs/a100-replay-live-20260909T1920/unloaded-observations.csv)
and [held-out errors](outputs/a100-replay-live-20260909T1920/unloaded-heldout.csv)
retain invalid phases: 64 of 72 phases and 18 complete triplets were valid.
Unloaded probes retained full messages and the 512-token generation limit but
used a different state-code format; they do not calibrate reference completion
overhead. Engine computed-token counters and derived prompt-minus-cache counts
are separate, and missing cache usage fields remain unknown.

Cheap warm work does not establish service feasibility. During width-eight replay,
observed resident P90 arrival TTFT reached roughly 12–24 s. In both width-16
sensitivities, zero sessions were admitted by the 30-second checkpoint; only
4/16 and 5/16 were admitted by the observation boundary. Respectively 34/44 and
33/46 post-migration resident arrivals remained unfinished. Catch-ups completing
after the boundary are censored, including completions during cleanup.
[Paired service windows](outputs/a100-replay-live-20260909T1920/service-paired.csv)
report latency, outstanding work and recovery while arrivals continue; failed
admissions prevent claims about equivalent full serving populations.

The run used native vLLM 0.22.0/LMCache 0.5.1 with the reference configuration,
plus a recorded FIFO token-event delivery patch. A node reboot changed the A100
UUID; both identities and runtime differences are recorded. Early timing gaps
and interrupted attempts remain in the package. Resident histories evolve through
recorded turns and resets, with assumed causal arrivals. The 24-history mixture
per GPU is not a validated physical placement of the modeled inventory. Tested
resident rates were 0.110650/0.221300 RPS for coding and 0.188178/0.376357 for
coding_long; neither scout established a capacity boundary.

The [twenty-policy diagnostic](outputs/a100-replay-live-20260909T1920/policy-comparison.csv)
uses the nominal tested rates, 30/120-second deadlines, 6,666 source GPUs and
6,666 GPUs at each destination, and the shared 1,000-Gbit/s WAN assumption.
Completed-request control screens passed, with censoring disclosed; this does
not validate a fleet SLO operating point. The 0.80-GB decimal effective-wire
anchor remains unvalidated on this path, separate from native KV geometry and
resident memory. No timing, cache, normalization or policy-resource coefficient
changed. That single-GPU run left live source quiescence, paired KV transfer,
physical cache placement and latency/recovery mappings open. A single destination
GPU cannot validate source-active ownership transfer; the paired follow-up below
addresses the source and transfer checks.

The [focused tests](outputs/a100-replay-live-20260909T1920/focused-simulator-tests.log)
passed 181 tests with the three requested exclusions. Host instrumentation checks
passed 258 tests with one 50-ms dispatch-timing assertion failure during concurrent
CPU verification; that test passed once in isolation, and both results are retained.
The frozen plan, input/model hashes, exact launches, raw token/request events,
metrics, power traces and reproduction commands are linked from the run report.
`pool_replay_validation.py prepare` freezes inputs; `pool_replay_measure.py` and
`pool_replay_resident.py` execute the bounded stages against a prepared runtime.
`pool_replay_report.py --out RUN_DIRECTORY --policies` reproduces the reductions
and the twenty evaluations. The saved 96.3% audit uses a different candidate union
and is not a controlled reproduction of this bounded comparison.

The [2026-09-10 supplemental completion](outputs/a100-replay-completion-20260910T0116/report.json)
used separate A100s in Sweden and Germany under a separately frozen 30-minute
cutoff. Eight controlled source-active replay/KV attempts were retained. Valid
replay cases confirmed full-message catch-up with source-evolved history; these
controlled append checks do not replace the recorded agentic service episodes.
Two data-supported correctness fixes were made: failed source activity now blocks
ownership transfer, and the LMCache connector preserves native-prefix hit counts
when external lookup misses. The latter bug exported only 1,792 tokens from a
2,048-token request with 64 native hits. It changes neither simulator timing
coefficients nor an assumed cache fraction.

The first KV attempts had a broken destination Redis connection pool after a
source startup correction. Reconnecting it and fixing native-prefix export
produced complete source chunks and an observed 8,192-token external retrieval.
The corrected validation response was unfinished at the cutoff; the second
corrected repeat was unmeasured. Neither is a passing KV handoff or latency result.
Observed storage chunks contain 12,582,912 payload bytes per 256 tokens, matching
native serialized geometry; this does not validate the 0.80-GB effective-wire
anchor for a complete migration. Both GPUs were released within the supplemental
cutoff. Original failed attempts and the initial 148.6-minute acquisition remain
unchanged. The twenty-policy diagnostic was subsequently reproduced in the
completed replay diagnosis above; the final paired follow-up below supersedes
this supplemental report's acquisition checklist. Fleet SLO validation remains open.
The nine per-scenario `events.jsonl` files are preserved losslessly in
[scenario-events.tar.gz](outputs/a100-replay-completion-20260910T0116/scenario-events.tar.gz).
Their [archive manifest](outputs/a100-replay-completion-20260910T0116/scenario-events-archive.json)
records verified original hashes and the extraction command needed before reproducing
the reductions from a Git checkout; the existing ignore rule excludes the raw files.

After extraction, all 298 artifact-manifest files match their hashes; both paired
reductions and the combined findings reproduce after normalizing path provenance.
Extraction is required: the archived reducer treats missing logs as empty and can
overwrite request counts with zeros while still labeling completed scenarios valid.

The [2026-09-10 paired follow-up](outputs/a100-replay-final-20260910T0352/report.json)
completed four clean width-eight KV conditions, eight resident scouts and twelve
300-second control/replay/KV episodes on separate Sweden and Germany A100s.
Acquisition, including startup and cleanup, took 113.0 minutes, within the two-hour cap.
All 32 controlled KV continuations and all 96 main-episode handoffs/continuations
validated. The main episodes retain 2,310 exact completed service requests and nine
censored arrivals. Both replay and KV exercised a verified source token stream
across the pause, followed by actual-history catch-up and destination continuation;
no extra diagnostic was needed.

Eight physical resident sessions used the frozen trajectories at nominal rates of
0.442600 coding and 0.188178 coding_long requests/s/GPU. Actual rates, original-arrival
latency, request-mean TPOT, sample counts, timing coverage, queues at migration +30/+120
seconds and recovery under continuing arrivals are in the
[service table](outputs/a100-replay-final-20260910T0352/service-observations.csv).
The following P90 TTFT values are seconds, shown as resident/destination-incoming;
destination timing excludes requests still dispatched on the source, while the
report retains the entire incoming population and its outstanding work.

| Workload / seed | Control | Replay | KV |
| --- | ---: | ---: | ---: |
| coding / 7101 | 0.353 / 0.339 | 0.689 / 0.505 | 0.355 / 19.753 |
| coding / 7102 | 0.445 / 0.412 | 0.607 / 0.548 | 0.490 / 19.850 |
| coding_long / 7101 | 0.335 / 0.490 | 14.070 / 1.124 | 0.322 / 23.819 |
| coding_long / 7102 | 0.314 / 0.465 | 0.345 / 0.631 | 0.318 / 36.496 |

Warm prefix reuse does not eliminate migration overhead: the verified loaded KV
catch-up took 62.38 seconds overall, with a 0.227-second destination TTFT and two
new KV chunks fetched in 0.428 seconds; source export, the controller/prefetch interval
and the unchanged 512-token validation limit remain separate measurements.
[Wire accounting](outputs/a100-replay-final-20260910T0352/kv-observations.csv)
separates unique payload, repeated GET payload, RESP framing and source-local reads.
Native serialization is 1.611 decimal GB per 32,768 tokens, distinct from the earlier
0.80-GB effective-wire anchor and resident memory accounting. The hardware proxy
limits aggregate GET responses to 1,000 Mbit/s; it is not the simulator's 1,000-Gbit/s
WAN scenario.

Loaded migration probes often generate all 512 allowed output tokens despite EOS
being enabled. The verified 30,777-token replay catch-up reached its first token in
0.377 seconds but took 15.215 seconds to complete 512 output tokens. KV also runs a
source export probe before prefetch and destination validation. Keep these adapter
generation costs separate from prefix rebuild work when calibrating the simulator.

Arrival times and finite-trajectory cycling are assumed: all 111 main-episode context
restarts were cycle wraps, not observed production resets. Histories retain actual
generated tokens between those declared restarts, using content-free recorded shapes.
The two nodes match vLLM 0.22.0/LMCache 0.5.1 but differ in CUDA/Torch build and
Transformers version; the [runtime builds](outputs/a100-replay-final-20260910T0352/runtime-builds.json)
record those differences. Client token timestamps do not identify server execution
or GPU queue time. Small windows and censored arrivals do not certify latency tails.
The acquisition changed no simulator coefficients. The completed replay diagnosis
above adds CPU comparisons and exposes ideal pooling; **the full campaign stays
stopped until replica-local service behavior is validated**.

The [raw archive manifest](outputs/a100-replay-final-20260910T0352/raw-telemetry-archive.json)
provides lossless restoration commands and verified member hashes, including ignored
events. The [artifact manifest](outputs/a100-replay-final-20260910T0352/artifact-sha256.json)
maps every raw file to its archived copy; authoritative Germany originals also reside
in [destination-final-raw.tar.gz](outputs/a100-replay-final-20260910T0352/destination-final-raw.tar.gz).
The frozen plan, amendments, exact launches, verified/open modeling gaps and
[test results](outputs/a100-replay-final-20260910T0352/verification.json) accompany the report.

Source load is 0.8; destination loads are 0.25, 0.50, 0.75, 0.90, and 0.95.
At equal source/destination sizes, the standing-service shed ceiling
is `min(1, 2 * (1 - destination_load) / 0.8)`: 100% at 50% destination load,
62.5% at 75%, 25% at 90%, and 12.5% at 95%. Methods can therefore tie at long
deadlines after reaching that common ceiling.
There are eight source sessions per GPU at every load; destination residents
are represented by their aggregate standing demand. Under the corrected
contract, coding snapshot 0 generates about 0.353 requests/s/GPU; the old
normalization implied about 7.46. Recorded coding trajectories cycle with a
reset on wrap. A separate `coding_long` cohort starts at measured contexts of
at least 24,576 tokens and follows the complete supported recorded trajectories.
Source request duration uses a contextual phase-work proxy, separate from
interarrival spacing; quiescence waits for an active request. Missing timestamps
require an explicit equal-cadence assumption. Seeded, stratified phases spread
cohort starts across the period, including requests started before time zero.
These phases remove the common quiet interval; they are not measured arrival
statistics. Buffer accounting includes arrivals to already paused cohorts while
other members of a batch finish quiescing.
Each migration wave captures the last completed source request state when it
enters the initial-transfer window, including waves that waited after admission.
It retains that snapshot during its copy. Changed or reset replay state resubmits
the full current context, using the existing context-dependent full-replay timing,
packing, and completion overhead; unchanged state needs no second replay. KV
catch-up still transfers the changed sealed blocks and computes the partial tail.
No retained-prefix hit is assumed for replay catch-up. This full-rebuild model is
an explicit transfer assumption, not a measured warm-prefix catch-up curve. Planner candidates use their proposed start times,
and shared execution determines when queued waves actually capture.
Trajectories exceeding replay
context support are excluded and counted. `measured_pack` repeats the measured
request shapes. Conservative peak-cycle KV reservations remain at every load.

`pool_shed_planner.py` uses receding-horizon batch admissions with common queue
and migration-phase feedback for all five methods. Only the next admissions
are committed. The default uses 64 execution waves per selected batch pattern
per admission and feedback resolution 0.5
(half the original geometric intervals). This resolution jointly changes
geometric decision anchors, reservation bins, and candidate starts; its sensitivity check
does not isolate feedback cadence. Observed buffer-recovery boundaries also
trigger decisions and candidate starts. Every policy uses the same event rule;
its timestamps can differ because policies create different queues.
Future rates use central calibration, never hidden execution
draws. Planning runtime is reported separately and is not charged to the
modeled migration deadline; these are offline policy simulations.
Recorded future request shapes and resets are known to every planner;
unknown future prompts and arrival-phase uncertainty are not sampled.
A central-calibration copy of the executor forecasts already admitted work at
every decision, retaining observed source captures and phase progress. New
candidates reserve compute time, route/shared-WAN volume, standing service, and
resident/buffer recovery. Recovery constraints cover each remaining suffix of
the horizon; unused early capacity cannot be saved for a later queue. Actual
queues retain their per-batch service limits. Source quiescence, resets and
full-context catch-up remain causal.

New admissions can change sharing and invalidate a forecast's finish or recovery
time. All unfinished admissions remain reserved, and actual debt, buffered work,
and recovery misses remain in the results. The executor does not certify resident
latency from these work budgets. There are no individual GPU, packet, or token
scheduler objects. Replay and KV policies share the same batch projections, while
QH can jointly select both action populations.

This is a **temporal LP approximation**, not a globally optimal dynamic
schedule. Load interactions and source resets are nonlinear. Static LP
baseline containment is checked on a common matrix; it does not certify the
ranking of different executed feedback trajectories. Raw executed losses,
iteration residuals, and deadline regressions remain in the audit. No baseline
outcome is substituted for QH. Isolated-fastest chooses each session type's
current isolated singleton action, which can differ from the best aggregate
throughput choice; this ranking uses unrounded isolated times, not the
candidate network-bin barriers. Figures distinguish admitted work from completed handoffs.
Greedy ranks shed per dominant remaining resource use; compute per unit shed
breaks equal-score ties.

Shed power is a proxy: completed source workload fraction times the measured
phase-power model's active-to-awake-idle difference at the declared request
rates. The supported coding points yield about 169 W active and 102 W idle,
so full migration corresponds to roughly **0.45 MW** at the 2-MW installed fleet
(about 4.5 MW in the archived 20-MW scenarios). The old
11.935 MW result used an active anchor inconsistent with the corrected source
cadence. The phase-power model's grouped cross-validation RMSE is 12.76 W;
bootstrap bands do not include all this model error. Proportional allocation
of the full power difference is a separate approximation, not evaluation of
nonlinear remaining-load power. No GPU shutdown trajectory is claimed.

Networking separates measured endpoint throughput from assumed shared WAN
budgets. Paired bulk endpoints are approximately 2.280/8.733 Gbps; application
KV limits are approximately 1.304/4.204 Gbps per node. Per-node throughput is
shared by its eight GPUs and both routes, with additional shared route/source
WAN limits. The WAN sweep is 40, 100, 400, and 1000 Gbps plus the measured
single-node reference. These are sensitivity scenarios, not asserted Azure
region-pair capacities. KV uses measured native sealed blocks (256 tokens,
12,582,912 bytes), with partial tails and live catch-up charged separately.
Large replay rebuilds use the full-context profile.
If any session in a replay batch exceeds 16,384 context tokens, the entire
batch uses the conservative serial-work model instead of the calibrated
packing curve. Long-batch measurements support this serial family, but do not
establish a physical discontinuity at 16,384 tokens. This switching rule is a
material modeling assumption: crossing it can sharply increase batch work.
The sweep spans the production inter-DC capacity scale reported in
[SWAN, §6.1 (SIGCOMM 2013)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/swan-sigcomm13.pdf);
it does not identify current available capacity for these Azure routes.
[Lai et al. (HotCloud 2018)](https://www.usenix.org/conference/hotcloud18/presentation/lai)
report stable inter-VM WAN rates across public-cloud measurements, supporting
the distinction between measured endpoint rates and shared WAN capacity.

Hardware reproduction runs the pooled engine against 440 loaded replay/KV
holdouts, 160 recorded policy cases, and 72 long-context batches, plus the
24 regional episodes. Runtime-matched KV calibration is fitted only on the
original local training split; the regional-to-local mismatch is also reported.
These historical cases retain their original service contracts and do not
validate fleet-scale resident latency or live source/recovery timing. The validation report records per-case
errors, false-feasible deadlines, original reference errors, library expansion,
execution refinement, feedback sensitivity, and proportional scaling.

The stable-greedy [hardware and resolution validation](outputs/a100-pooled-service-validation/stable-greedy/critic-summary.json)
passes all required gates and [233 focused tests](outputs/a100-pooled-service-validation/stable-greedy/focused-tests.log).
A [strict three-case platform comparison](outputs/a100-pooled-service-validation/stable-greedy/platform-comparison.json)
passes for all five policies after stabilizing greedy's secondary ties.
Across six central
cases, doubling dispatch resolution changes shed or the QH gap by at most
0.0743 percentage points. Changing the planning grid changes results by up to
8.152 points, so small policy advantages are not grid-robust. Runtime-matched
loaded replay/KV p90 relative errors are 2.28%/11.92%; the regional cases have
1.243 s mean absolute error and retain one false-feasible 25 s deadline.
The unmatched regional-to-local KV transfer diagnostic remains above its 15%
error threshold at 16.83%; the accepted local fit uses only local training data.
Recorded policy-case p90 relative errors are 18.67% for replay, 13.13% for KV,
26.37% for QH, and 29.04% for greedy. The 72 long-context batches have 16.01%
p90 error; their serial model family was selected after inspecting those cases,
so this is a calibration check rather than an untouched holdout.
A [matched hardware-ranking audit](outputs/a100-pooled-service-validation/historical-ranking-audit.json)
reproduces nine of ten QH/replay ranking signs when their recorded actions
differ. One reversal persists across all three hardware repeats: seven replays
plus one KV transfer take 3.03–3.88 s longer than eight replays, while the model
predicts a 1.09 s advantage on the held-out repeat. The
[phase audit](outputs/a100-pooled-service-validation/historical-mixed-phase-audit.json)
confirms KV completion is charged compute, but seven-request packing and mixed
execution on one GPU remain imperfect transfers. The fixture also lacks an
explicit 135-token verification prefix. These checks assess recorded actions,
not the new planner's decisions, and do not establish universal ranking fidelity.

The archived stable-greedy [2 MW comparison](outputs/a100-pooled-service-pre-certificate-retry/scale-diagnostic.json)
scales all three GPU fleets together; [30 s](outputs/a100-pooled-service-pre-certificate-retry/scale-comparison-30s.png)
and [300 s](outputs/a100-pooled-service-pre-certificate-retry/scale-comparison-300s.png) figures show handoff and KV fractions.
At 50% destination load and a 300 s
deadline, QH completes coding KV handoffs for 1.41% of the original source at
20 MW with a 1000 Gbps WAN budget, versus 11.80% at 2 MW with the same budget.
Scaling WAN capacity proportionally reduces the latter to 1.23%. The 1000 Gbit/s
shared budget supplies about 15 Mbit/s per source GPU at 20 MW, versus 150 Mbit/s
at 2 MW. These fleet allocations are much smaller than the measured per-node
rates. Downsizing with fixed WAN changes bandwidth per GPU; GPU count alone
does not explain action selection. Across all 24 central scale cases, QH has 11 wins,
7 losses, and 6 ties against replay; all 120 policy executions protect the
resident service budget.

`pool_shed_cache_sensitivity.py` runs a separate 36-case, central-calibration
comparison: coding snapshot 0, 20 MW and 2 MW A100 fleets, a 1000 Gbps shared WAN,
50% of the serving envelope, and 30/120/300 s deadlines. Six variants separate
the old payload, a user-reported **800,000,000 bytes per 32,768 tokens** effective
wire anchor, and assumed replay-prefix/private-KV savings. The anchor is not
reclassified as a measured full-cache geometry: additional private fractions
describe initial snapshots and are sensitivities, since the anchor may already
describe a private transfer. Prefix lengths are rounded down to sealed blocks.

Replay reuse removes the assumed cached tokens' prefill work at the original
full-context rate, retaining completion overhead and packing rules. Historical
native cache hits are not fully observed, so these are additional assumed
savings relative to the fitted timing. Shared prefixes are assumed already
present at the destinations and reusable after resets, capped by current
context; newly appended tokens beyond those fixed prefixes remain private.
Wire savings do not change resident memory, source demand, or source power.
Every variant receives the same union of candidate batches; reduction rejects
any change in replay-only behavior caused solely by KV assumptions. A fixed
1 s geometric base clock, subdivided at resolution 0.5, prevents KV durations
from changing replay-only's decision times. Observed recovery feedback remains
available to every method. The comparison's baseline therefore uses this common
library and clock, rather than reproducing the stopped campaign's decisions.
Run `python pool_shed_cache_sensitivity.py --case N` for indices 0–35, then
`PYTHONPATH=. python outputs/a100-cache-sensitivity/render.py` to validate,
reduce, and render the frozen grid with complete percentage axes.

The completed [36-case results](outputs/a100-cache-sensitivity/summary.json)
contain 180 policy evaluations, with [deadline/handoff](outputs/a100-cache-sensitivity/handoff_fraction.png),
[deadline/power](outputs/a100-cache-sensitivity/power_mw.png), and
[KV action fractions](outputs/a100-cache-sensitivity/kv_source_fraction.png).
At 30 s, QH's completed KV fraction of original source workload is:

| Initial cache assumption | 20 MW | 2 MW |
| --- | ---: | ---: |
| Original wire payload | 0.74% | 7.57% |
| 0.80 GB/32K effective wire anchor | 1.25% | 11.70% |
| Anchor + 25% replay reuse + 50% private KV | 2.68% | 18.96% |
| Anchor + 25% replay reuse + 10% private KV | 10.76% | 35.81% |

Replay-only reaches approximately 100% handoff in all 36 cases, so this shows
changed actions and compute use, not additional shed over replay. At 20 MW and
300 s, KV-only handoff increases from 28.42% to 44.34% with the wire anchor alone,
and to 94.92% in the 10%-private sensitivity. The largest retained QH loss versus
replay is 0.0394 percentage points; greedy's is 0.1918 points. All 24 paired
replay-only controls are exactly equal. The final run spanned 11.6 minutes
(28.2 summed case-minutes with three workers; setup and discarded diagnostic
runs excluded). These are central scenario sensitivities, not confidence bounds.
The [validation record](outputs/a100-cache-sensitivity/validation.json) includes
169 focused passing tests, 24 exact replay controls, 12 exact KV controls,
resource residuals, and hashes for the preserved cases and corrected figures.

The archived v8 cache sensitivity does not validate fast replay or resident SLOs.
The following v8 audit records its original synchronized/isolation assumptions;
its results are historical and do not describe the current executor.
The simulated coding 50% point offers about 0.221 resident requests/s/GPU,
versus 1.318 at the loaded experiment's 50% reference, with different request
shapes. The extra fitted replay slowdown is approximately 0.7% versus 15.3%,
respectively. Source requests also start synchronously every 22.63 s in this
snapshot. Matching resident RPS, shape, context, and arrival timing remains
necessary before interpreting the fleet results as deployment performance.

The subsequent [replay/resident audit](outputs/a100-replay-audit/audit.json)
kept the large campaign stopped. Its archived twelve policy executions and
hardware timing checks retain their original source hashes. It changes no simulator calibration or production policy.
**The fast replay is explainable under this scenario, but protected resident
latency and realistic arrival behavior remain unvalidated.**

- The 20 MW source has **40 MW of installed destination GPUs**. At 50% load,
  the two destinations supply 66,666 spare GPU equivalents, while imported
  source service requires 53,332.8. Increasing all three fleets preserves
  replay work per spare GPU. Holding total WAN fixed makes KV progressively
  harder; increasing GPU count alone does not impose an additional replay
  slowdown. Free pooling of fractional replicas also assumes schedulable
  capacity without per-GPU placement, preemption, or kernel-level contention.
- Coding snapshot 0 averages 14,690 initial context tokens. Its initial replay
  logs are 15.67 GB; the effective 0.80 GB/32K anchor still gives **189.82 TB**
  of initial KV snapshots. At 1000 Gbps these volumes alone require 0.125 s
  and 1,518.6 s respectively. These are initial-snapshot byte bounds, not
  predictions for a migration whose snapshots change with time and resets.
  The preserved replay run transfers 16.06 GB and completes at **19.064 s**,
  before the next 22.633 s source arrivals, with **zero buffered requests**.
- The coding destination offers 0.221 requests/s/GPU at its 50% point, versus
  1.318 for the loaded 2048/32 reference and 3.688 for the regional 604/64
  reference. Different shapes prevent interpreting those ratios as utilization
  ratios. The coding load conversion reduces the fitted extra replay slowdown
  from 15.31% at offered-reference rho=0.5 to **0.68%**. The additional measured
  regional factors are 0.779/0.776; these are timing scales, not cache fractions.
- **Zero resident debt is enforced by construction.** The executor does not
  produce resident TTFT or TPOT. Separate regional holdouts show median resident
  completion losses of 91.03% during replay and 2.81% during KV; those checks use
  the original unprotected execution contract. They do not validate the fleet's
  fractional protected-compute model. The independent
  [A100 service-headroom confirmation](outputs/service-headroom-a100-20260815/confirmed.json)
  also leaves `planner_usable=false` and `supported_bound=null`.
  Resident latency depends on actual prefill/decode scheduling; vLLM documents
  the TTFT/ITL tradeoff from changing prefill chunk size in its
  [tuning guide](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill).
- Initial replay charges full context at the measured context-dependent rate.
  **Incremental replay catch-up is weaker:** appending 256 tokens costs the same
  0.0692 s after a 2,048-token or 30,000-token prefix. Its rate and packing depend
  only on the append length; below measured support, interpolation from zero
  scales down the 0.5256 s singleton completion overhead too. No matched
  context-by-append calibration validates that rule. Hardware catch-up submits
  full messages, whereas the simulator additionally assumes delta-log transport.
  Source request durations are throughput-based proxies without an executed
  source queue. Cyclic resets and exclusion of trajectories beyond approximately
  32K also prevent indefinitely growing contexts.

The audit's 30-second completed workload fractions are:

| Scenario or timing ablation | Replay-only | QH LP |
| --- | ---: | ---: |
| Effective wire anchor, current assumptions | 100.00% | 99.99% |
| Offered-reference timing factor set to one | 96.09% | 99.83% |
| Regional replay speedup removed | 96.00% | 99.26% |
| All replay batches serialized | 100.00% | 100.00% |
| Destination load 75% (62.5% standing-service ceiling) | 59.01% | 60.59% |
| Recorded long-context cohort | 87.88% | 88.70% |

All use the effective wire anchor without further private-KV or replay-cache
discounts. The first five share candidate batches and the fixed planning clock;
the long-context pair shares its own library. Timing ablations leave resident
traffic and capacity fixed, so setting the timing factor to one is **not a
hardware-load match**. These results do not select a replacement calibration
or certify small policy gaps against the known planning-grid sensitivity.
Serializing batches alone does not remove the fast replay outcome.

Existing timing reproduction passes again: 220 loaded replay holdouts, 440
loaded replay/KV cases, 160 recorded policy cases, 72 long-context batches, and
24 regional episodes. The regional check retains one false-feasible 25-second
deadline. Resident-debt reproduction covers six held-out routes. Those frozen
source checks leave protected service, live catch-up, and arrival-phase behavior
outside their validation scope. [176 focused tests pass](outputs/a100-replay-audit/focused-tests.log),
with the same three campaign integration tests excluded as in the cache audit.
Before resuming the campaign, specify actual
destination availability and resident traffic, then validate concurrent resident
TTFT/TPOT and source quiescence/buffer recovery under that same contract.

The follow-up measurement inventory finds 336 historical replay catch-ups in
[`coding-run`](outputs/coding-run/migrations.csv) and
[`bounded-hardware-campaign-run`](outputs/bounded-hardware-campaign-run/migrations.csv).
Both record a vLLM 0.10.1.1 runtime, so they are diagnostics for the current
0.22 stack. The 108 concurrency-one coding cases append only 31–35 tokens but
have a median catch-up duration of 2.681 s. Their raw per-scenario `result.json`
files retain catch-up request records; the flattened CSV omits processed-token
counts, and historical inferred counts do not establish native cache hits.
The current coding manifest has no arrival timestamp for any of its 1,655
records. Independent seeded arrival phases can remove artificial synchrony,
but measured burstiness requires source request timestamps. Across 23,324
regional resident request records from all retained attempts, 17,279 have
`first_byte_ns == end_ns`. Their stream chunks retain times and byte counts;
the meaning of the first-token field, including reasoning output, needs checking
before fitting TTFT/TPOT. Continued-arrival recovery also needs resident traffic
that remains active after migration. The existing transition study covers its
three discrete 4K recipes, not this general long-context recovery model.

The bounded 20 MW source with two 10 MW destinations has a **62.5%**
standing-service handoff ceiling at the current source/destination loads of
80%/50%. The current bounded audit executes this capacity scenario using the independent
destination count; compute, memory and endpoint inventory shrink without changing
source population or enlarging the shared WAN. Reducing
destination size can leave both methods tied at that common service ceiling.

After the live replay and service checks above, use the new output directory:

```bash
uv run python pool_shed_campaign.py validate --out outputs/a100-pooled-agentic-2mw-validation
uv run python pool_shed_campaign.py prepare --out outputs/a100-pooled-agentic-2mw
uv run python pool_shed_campaign.py run --out outputs/a100-pooled-agentic-2mw
uv run python pool_shed_campaign.py reduce --out outputs/a100-pooled-agentic-2mw
```

The default grid contains 11,250 scenarios / 56,250 policy results: four coding
snapshots, one long-context cohort, five destination loads, five WAN settings,
ten deadlines from 1 to 3600 seconds, and central plus eight paired timing/network
draws. `prepare --smoke` selects 24 scenarios. These defaults pin fleet size and
workloads; they do not certify replay or resident-service fidelity.
`run --shard N --shards K` supports process shards and checkpoint resume;
code, solver, calibration, and grid identities must match. A reviewed numerical
recovery retries a nonoptimal or independently uncertified simplex solve once
with a fresh interior-point solver; both must satisfy the same original resource
and objective checks. Native solver status alone is insufficient for acceptance.
The [certificate-retry tests](outputs/a100-pooled-service-validation/certificate-retry/qh-certificate-focused-tests.json)
pass all 242 cases. The [independent numerical review](outputs/a100-pooled-service-validation/certificate-retry/success-path-review.json)
verifies unchanged acceptance predicates and bitwise-identical prior successful
fixtures, plus the corrected solver result for the newly captured failure.
Successful earlier checkpoints may be reused only through an explicit manifest
pinning their original identities and exact bytes; the summary reports this
execution lineage separately from the current source identity. Bands show empirical
p05–p95 sensitivity across execution draws, snapshots, and measured power curves,
not coverage of unmeasured transfer error or formal SLO compliance. Power curves
show medians; action stacks show mean fractions of the original source workload,
not action shares conditional on completed handoffs.

The earlier 3,124 campaign checkpoints are archived intact in
`outputs/a100-pooled-service-pre-stable-ties`. The subsequent campaign's 9,486
completed checkpoints are preserved in `outputs/a100-pooled-service-pre-certificate-retry`.
Their bytes and original identity are retained through explicit inheritance;
the remaining cases used the certificate-retry implementation until the campaign
was stopped at the user's request with 12,011 of 13,500 checkpoints written.
The campaign is incomplete and its saved checkpoints are preserved; it has not
been restarted. The retry changes only previously rejected solver attempts.
Greedy treats primary and secondary scores within the same relative
1e-12 tolerance as tied and then selects the earliest candidate. This prevents
a one-ULP work-cost difference from postponing an otherwise equivalent current
admission. The failed Linux comparison is retained as regression evidence.

`outputs/a100-pooled-feedback` archives the v7 campaign, which permitted
resident displacement and used the old service/power normalization. Its results
use an obsolete normalization; they do not validate the current shared-service model. The earlier fixed-plan
campaign is archived under `outputs/a100-pooled-execution`.

### Bounded resident server timing handoff

The node agent should pull `policy-hardware-width8-pilot` and use
[this frozen acquisition plan](outputs/a100-resident-server-timing-plan/plan.json).
`pool_replay_server_timing.py` generates six guarded patches for vLLM 0.22.0;
install them in an isolated reference runtime with the preserved FIFO collector.
`pool_replay_server_acquire.py --out RUN --inventory RUN/inventory.json` attaches
to an owned two-GPU stack with its original startup clock and cleanup commands.
It enforces the frozen overhead gate, cell order and time limits.
`pool_replay_server_reduce.py` checks raw server/client joins after shutdown;
launcher completion alone does not establish telemetry acceptance.
These tools reuse the current harness and token-stream collector. This follow-up measures
resident decode and queueing; the fleet study remains 2 MW source / 2 MW at each
of two destinations. Do not rerun scouts, the broad KV/WAN sweep or fleet policies.

1. **Verify the GPUs and runtime.** Azure is optional. Both archived GPUs are
   **A100 80GB PCIe**; matching full GPUs outside Azure can run this probe.
   Use exclusive GPUs without MIG partitioning. One GPU suffices for warm-decode
   cells; source-active episodes require two separate physical GPUs. They may
   share a host if CPU/service interference is recorded. Prefer the archived
   Germany destination and Sweden source for direct timing reproduction; a new
   environment provides new calibration, not proof of Germany's old timing.
   Record network/endpoint differences. SXM or 40GB variants require a separately
   labeled hardware calibration; do not silently pool their timing fits with
   PCIe 80GB. NVIDIA documents the variant differences in its
   [A100 specifications](https://www.nvidia.com/en-us/data-center/a100/).
   Use an exclusive test stack: the existing episode setup resets native/LMCache state and its
   owned Redis database. Pin GPT-OSS-20B revision
   `6cee5e81ee83917806bbde320786a8fb61efebee` and verify weights/tokenizer against
   the two reference model manifests linked in the plan. Keep TP1, vLLM 0.22.0,
   LMCache 0.5.1, BF16/MXFP4, TRITON_ATTN, eager execution, 32,768 maximum context,
   256 sequences, 8,192 scheduled tokens, 16-token blocks, KV dtype `auto`, memory utilization
   0.75, prefix caching and chunked prefill enabled, hybrid KV manager disabled,
   prompt-token details enabled and `VLLM_USE_FLASHINFER_SAMPLER=0`.
   Keep the installed Torch/CUDA/Transformers builds; record their differences.
   Record GPU UUIDs and actual `nvidia-smi` power limits on both nodes; use 300 W
   on the owned test GPUs and record any change. The archived final Germany
   inventory does not prove its old power cap. Preserve the hardware proxy's
   1,000 **Mbit/s** setting and record actual transport delays on the new hosts;
   this is distinct from the fleet's 1,000-Gbit/s WAN.

2. **Instrument generation separately from delivery.** Log stable iteration IDs,
   external/internal request-ID mappings and token ordinals at these boundaries:
   scheduler (prompt length, computed tokens before scheduling, scheduled tokens,
   running/waiting requests, cache hits, remote-KV waits and preemptions); worker
   (CUDA-event forward/logits/sampling elapsed time and the host output-ready
   timestamp); frontend (ingress, output receipt, collector put/pop or SSE yield);
   and the existing client SSE receiver. Capture partial/failed requests too.
   Read the scheduler's **pre-update** computed count: the live request has already
   advanced when `schedule()` returns. Associate the correct iteration with each
   asynchronous worker result. Reuse output-copy synchronization when resolving
   CUDA events; do not add a global device synchronization on every iteration.
   Host `execute_model(non_block=True)` duration measures enqueueing. CUDA events
   measure relative GPU-stream elapsed time, which can include waits; they are
   not host timestamps. SSE yield does not prove wire flush. Record clock domains,
   hosts, boot/time namespaces and paired wall/monotonic anchors. Subtract host
   timestamps only after verifying a common clock domain; never subtract Sweden
   and Germany monotonic clocks.
   These boundaries follow the pinned vLLM
   [scheduler](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/v1/core/sched/scheduler.py),
   [engine](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/v1/engine/core.py) and
   [worker](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/v1/worker/gpu_model_runner.py).

3. **Check measurement overhead before collecting.** Run matched warm 8K,
   concurrency-eight, 256-output smoke bursts with instrumentation off/on/off.
   Use the warm-cell protocol below, reusing the same eight prompts and clearing
   owned native cache before each mode's prewarm. Disable only the new timing
   hooks; preserve the FIFO collector patch. Completion duration is the median
   client dispatch-to-DONE duration; output rate is total actual output tokens
   divided by the interval from first dispatch to last DONE. Require the two off
   baselines to agree within 5% in both metrics, and on to agree within 5% of their
   time-interpolated values.
   Permit one repeat only for an inconclusive check within the setup budget;
   stop acquisition if instrumentation fails. Use buffered process-local logs,
   require zero dropped records and hash every actually imported patched module.
   A short native profile can verify the device boundaries: use the v0.22
   `--profiler-config` with `profiler: torch`, `max_iterations: 32` and
   `ignore_frontend: true`, and disable stack/shape/memory/FLOP collection. Verify
   the trace actually contains CUDA kernels and exclude it from fitting. Keep
   the profiler off for the measurement cells.

4. **Run these four complete agentic episodes first, in this order.** Each has
   eight resident histories, eight source histories, 60 seconds of baseline,
   migration at 60 seconds and continuing arrivals until 300 seconds. Preserve
   causal turns, resets, source quiescence and captured-state continuations.

   | Episode | Resident requests/s | Incoming requests/s/session |
   |---|---:|---:|
   | `episodes-coding-0-control-7102` | 0.4426002548363285 | 0.04426002548363285 |
   | `episodes-coding-0-replay-7102` | 0.4426002548363285 | 0.04426002548363285 |
   | `episodes-coding_long-0-control-7101` | 0.1881783279213518 | 0.03763566558427036 |
   | `episodes-coding_long-0-replay-7101` | 0.1881783279213518 | 0.03763566558427036 |

   The JSON contains the exact archived specs and hashes. Use the original
   `outputs/a100-replay-final-20260910T0352/plan.json` workload objects with
   `ResidentAcquisition.episode(spec, workload, 300, True)`, including for control.
   Create a fresh output directory, `runtime-launch.json` and validated inventory
   with a new deadline before constructing the acquisition object. The harness
   uses one coordinator address: expose separate source/destination serving and
   LMCache ports there, directly or via forwarding, with coordinator-local proxy
   timestamps and attached source/sink stack logs. Do not pass
   the already sampled `physical-workload.json`: `episode()` samples internally.
   The offered traces must reproduce 132 resident + 106 incoming arrivals for
   coding and 56 + 92 for long, with the frozen hashes. Preserve the current
   control mirror behavior after releasing trajectory ownership; tag this work
   separately. Keep the full retained-token `/v1/completions` migration probe
   with a 512-token maximum and normal EOS; do not substitute the separate chat
   state-code probe. Service turns retain their exact trajectory outputs.
   In particular, retain coding resident session 1, turn 7 (1,233 outputs), and
   its dependent turns; a 90-second scout misses this event.

   Use a thin direct launcher: the old `episodes` CLI recalculates incoming rates
   from the current fleet model, `--workload` only filters scouts, and `followups`
   adds hardcoded unrelated cases. Do not run an archived `start-stack.py`
   unchanged: its paths, endpoints and deadlines belong to the old acquisition.

5. **Run the twelve controlled warm-decode cells.** Cross total prompt lengths
   **8,192 / 30,000**, independent concurrent sessions **1 / 8 / 16**, and two
   repeats, each generating **1,536 tokens**. The JSON fixes order and seeds
   8101/8102; repeat one is fitting data and repeat two is held out. Before each
   cell, verify the owned GPU/stack is idle and reset its native prefix cache.
   Prewarm each unique session's `C - 32` prefix with one output token, then submit full `C`
   with 32 fresh suffix tokens. Preserve native cache between warmup and measure.
   Reuse `destination_runner.Session(..., force_output=False)`, `prewarm`,
   `prepare_issue`/`completion_payload` and `headroom.async_completion`; set
   `ignore_eos=True`, temperature zero and bypass LMCache reads/writes. Do not
   restrict allowed output tokens. Require 1,536 actual outputs, native cached
   prefixes at least 8,160 / 29,968 tokens and zero external-cache hits. Verify
   actual scheduled prefill work. Both prompt-plus-output lengths fit 32,768.
   Retain failed warm conditions, preemptions and incomplete outputs explicitly;
   a width-16 cell that cannot keep its prefixes resident is useful evidence,
   but cannot be fitted as a successful 32-token warm append.

6. **Bound the acquisition and return all evidence.** The hard wall cap is
   **90 minutes**, including setup, warmup, failures and cleanup: setup/smoke
   at most 360 s, each episode 420 s, each warm cell 270 s (all prewarming at most
   90 s, measurement at most 180 s), plus a final 120 s cleanup reserve. These
   are limits, not runtime predictions. Prepare/test the instrumentation before
   that clock starts. Existing prewarm timeouts are per request; enforce the
   whole-cell and global deadlines too. Start a cell only with its full reserve
   plus cleanup available. Save failures, partial outputs and unstarted cells;
   do not retry slow results or relax the protocol to fill the matrix.

   Return `outputs/a100-resident-server-timing-<UTC>/` with the frozen plan,
   checkout, new inventory and exact launch commands; model/tokenizer and imported
   module hashes, instrumentation patch/source copies; raw scheduler, worker,
   frontend and client events; offered/physical traces, migration/request history,
   engine and GPU telemetry; overhead comparison and optional short profile;
   per-cell summaries, completeness report and a SHA256 manifest. Compressed raw
   archives are fine if member hashes and a fresh reduction are verified.
   Reconcile request/iteration/token ordinals, generated counts and final usage,
   including partial requests; check each iteration's scheduled-token total is
   at most 8,192. Report causal queue wait, device execution, server output-ready
   intervals and client delivery separately. Explain the long coding response
   and long-replay resident burst from those records. Keep the existing fit and
   tolerances frozen while collecting; all four new agentic episodes are
   validation only. Data quality is the GPU acceptance test, not a QH advantage.

After the data returns, CPU work remains: calibrate only identifiable timing
terms, integrate GPU-local resident queues and KV placement plus causal source
queues, then run bounded end-to-end policy and uncertainty checks. Remove the
full-campaign guard only after those checks support resident-service validity.

## Current evidence

The repository contains:

- a measured GPT-OSS-20B/H100 NVL TP=1 occupancy and GPU-power curve;
- working replay and compatible KV handoff on two A100s, with 24/24 serial and
  90/90 bounded-campaign migrations completing by their deadlines;
- 105/105 passing bounded-campaign gates and 6/6 passing parallel-KV gates;
- conservative replay and KV duration fits;
- exact full and incremental KV block/wire accounting;
- measured request-boundary replay and KV Gantt charts through the first
  destination token;
- a one-pool requirement-frontier solver;
- LP, static `greedy`, and dual-priced `greedy_lagrangian` optimizers;
- pool-aware planning and internal packing checks; and
- a deterministic migration, network, request, queue, and power simulator.

The completed planner-driven policy campaign executes mixed replay/KV choices
with ordered eager-parallel launch. The archived destination service campaign
does not provide an accepted shared-load capacity boundary, so simulator
service headroom remains a sensitivity. That boundary is not required for the
dedicated two-A100 migration claim.

`service_holdout_analysis.py` reproduces the prefill/decode staircase audit,
leaky context-bundle retrospective diagnostic, and matched-work
request-simulation falsification. It keeps pooled-token ITL separate from
per-request decode duration and reports why those traces do not yet support a
TTFT/ITL or decode-hold admission guarantee. Supplying `--powertrace-root`
adds the raw disaggregated GPT-OSS empirical sensitivity reference; it is
explicitly not used as a Queue-Haul latency model.

```bash
uv run python service_holdout_analysis.py \
  --powertrace-root ~/powertrace-sim \
  --out outputs/service-holdout-20260814/summary.json
```

The audit's primary profile is the A100 staircase profile; its default H100
comparison input is `profiles/gpt_oss_20b_h100_tp1.json`. The latter's 2026-08-11
H100 NVL measurements give `F=11415.78` prefill tok/s, `G=451.32` decode tok/s,
1,205,376 production KV-cache tokens, and a concave GPU-power envelope reaching 168.39 W
and measured through offered load `ell=12.566`. Admission remains bounded at
`ell=0.96647`. Raw benchmark and power samples are under
`outputs/h100-profile-20260811/`. Replay, KV-transfer, and transition timings
remain clearly marked A100-derived estimates until rerun on H100.

The separate boundary-aligned H100 power campaign retains 111 measured cells
(90 discovery, 18 unseen confirmation, and three idle anchors) with no cached
tokens or counter/window violations. Its frozen rational model is
`P=P0+A z/(1+z)`, `z=alpha f+beta g`, without a cross term. The unseen cells
have 2.37 W MAE and 4.34 W p90 absolute error; the fit remains explicitly
`holdout_failed` because its `R^2=0.942` misses the declared 0.95 gate. The
pooled parity view also applies that frozen fit to 38 valid cells from the prior
physical H100. It plots all 149 points while folding every non-idle workload
composition into one "Sessions" series and reports their aggregate 3.56 W MAE
and `R^2=0.919`. Interrupted and
offered-work-attributed sweeps are excluded.

```bash
uv run python queue-haul/plot_h100_power_parity.py \
  --run-root /datadrive/queue-haul-power/h100-realized-20260814-005 \
  --history-run-root /datadrive/queue-haul-power/h100-realized-20260814-004 \
  --out queue-haul/outputs/h100_power_model_parity
```

The H100 full-migration parity view pools all 295 historical queue critical
paths and reports 1.42 s MAE and `R^2=0.954`. Its fixed 1--200 s log axes retain
the two observations above 100 seconds while labeling decades from `10^0`
through `10^2`. This view and the H100 power parity view use native 1.65 x
1.75 inch canvases for side-by-side placement within one USENIX column.

The A100 timing campaign completed all 120 prespecified scenarios: 40 replay,
40 KV-transfer, and 40 mixed queues, varying context, width, destination,
background load, and move order. All 1,241 migration requests passed HTTP, token,
cache, and timestamp accounting. Two failed attempts (a background Harmony parser
error and a midnight prompt-date change) are preserved alongside successful retries.

The original frozen model overpredicts: its full 120-scenario comparison has
9.161 s MAE and R²=0.32294, failing the 3 s / 0.8 gates. This baseline remains in
`outputs/a100_live_queue_makespan_parity.{png,pdf,csv}` and its summary JSON.
A separately prespecified correction fits one positive least-squares scale through
the origin per action using the first 80 scenarios: replay 0.53599, KV 0.73814,
and mixed 0.79805. Predictions were frozen before the final 24 balanced scenarios
(eight per action) began. Those unseen scenarios achieve **1.310 s MAE and
R²=0.98378**, passing both gates. The held-out view uses the H100 plot format in
`outputs/a100_heldout_queue_makespan_parity.{png,pdf,csv}`. The intervening 16
scenarios remain in the baseline and are excluded from this calibration/validation
split. No observations were selected by prediction error.

Evidence is archived in `outputs/a100-parity-20260907/`: all scenario attempts,
node logs, raw timing/power traces, setup logs, and an SHA-256 archive manifest.
`timing/scale-protocol.json` records the split before validation; `scale-fit.json`
records the frozen coefficients and predictions. Reduction verifies their hashes,
reproduces the training coefficients, and checks that all training ended before
freezing and every held-out measurement began afterward.

Queue makespan runs from the shared release timestamp to the last first streamed
response, including dispatch delay. KV requests must retrieve their full planned
context. Replay may reuse at most 64 framing tokens: a local synthetic rendering
audit found 69 common tokens before different state codes, or four 16-token cache
blocks. The reducer requires that audit and rejects cached session context.
`--timing-only` records missing state answers without rejecting otherwise valid
generation; 17 of 1,241 requests lacked a verified answer (four of 261 in the
held-out block). These figures validate timing, not semantic restoration.

East US 2 was restored after its reimage, retaining the existing data disk and
model snapshot. All three nodes used commit `7fc0cda1`, vLLM 0.22.0, LMCache 0.5.1,
torch 2.11.0+cu129, and transformers 5.15.1. Use `QH_RUNTIME=native`,
`QH_LMCACHE_MODE=mp`, `QH_NATIVE_RUNTIME_VERSIONS=0.22.0,0.5.1`,
`HF_HOME=/datadrive`, and `QH_CACHE_ROOT=/datadrive/queue-haul-cache` on every node.
Their login environments now pin `VLLM_SYSTEM_START_DATE=2026-09-06`, matching the
campaign: vLLM's automatic date otherwise changes cache hashes at midnight.
Keep `UV_NO_SYNC=1` so `uv run` preserves the separately installed serving runtime.
The raw run remains at `/datadrive/queue-haul-network/a100-timing-r3`.

The A100 power graph uses the H100 renderer on all 111 verified Sweden cells,
including training and idle cells: **1.65 W MAE, R²=0.990**. Artifacts are
`outputs/a100_power_model_parity.{png,pdf,csv}`. This descriptive plot is separate
from power holdout acceptance: the 18 unseen cells had 1.129 W MAE, R²=-0.3633,
and 61.63% coefficient variation, failing the R² and 20% stability gates. Raw
Sweden power evidence and its holdout-only diagnostic remain under
`outputs/a100-parity-20260905/`. The earlier 111+117-cell Germany replication
remains under `outputs/a100-parity-20260904/`; its stability gate also failed
(29.94%). Earlier interrupted timing and parser diagnostics are retained.

Reproduce the figures from the archive (run from `agent-migrate`):

```bash
export UV_NO_SYNC=1
uv run python queue-haul/a100_parity_campaign.py reduce-timing \
  --run-root queue-haul/outputs/a100-parity-20260907/timing --prospective-scale \
  --out queue-haul/outputs/a100_heldout_queue_makespan_parity
uv run python queue-haul/a100_parity_campaign.py plot-power \
  --run-root queue-haul/outputs/a100-parity-20260905/power \
  --out queue-haul/outputs/a100_power_model_parity
```

Omit `--prospective-scale` and use `outputs/a100_live_queue_makespan_parity` to
reproduce the full frozen baseline; that reduction writes its artifacts and exits
nonzero because its accuracy gates fail. Collection uses the unchanged archived
`timing-plan.json` with `run-timing --timing-only` and a fresh run root.

The power command is the same 90-discovery, 18-confirmation, three-idle
realized-token protocol used for H100. It hard-fails unless exactly one NVIDIA
A100 80GB PCIe is visible at a 300 W limit, all request/counter accounting is
exact, and the unseen power gates pass. Do not overlap the power and timing
campaigns on the source A100.

The completed 72-scenario H100 hardware-gap campaign has no failed or missing
runs. It scales the constrained East KV reserve to 96% of the measured
1,205,376-token capacity, reserves 65% of each destination's migration window
for KV, and requests 41.4% of removable source power. All three deadline-blind
controls missed 45 seconds and reached the target in 55.5--59.5 seconds.
The reduced evidence and raw scenario attempts are retained in
`outputs/east-germany-hardware-gap-h100-20260812/`, copied from node run root
`/datadrive/queue-haul-network/hardware-gap-h100-002`. Its artifact manifest
also addresses the uncommitted reusable stack logs.

## System boundary

A handoff prepares state in the background, quiesces at a request boundary,
performs final catch-up, switches routing, and succeeds when the destination
returns the first token. Mid-token migration, return migration, cold model
placement, unrelated destination arrivals, provider fleet policy, and facility
power are out of scope.
Destinations use a 32 GB bidirectional LMCache L1 so resumed sessions retain
and write back KV instead of remaining read-only migration consumers.
Same-source migrations may overlap and share the bandwidth of every link on
their route; there is no fixed per-source migration-count limit.

The public candidate is:

```text
(session, replay-or-KV, compatible destination pool)
```

The planner lowers the logical pool choice to a deterministic replica assignment.
A V1 pool advertises ongoing and stable service envelopes, temporary queued-work
allowance, live-KV blocks, route identity, allowed methods, compatibility, and
evidence status. The scenario supplies link rates. The pool planner enforces
ongoing event capacity and conservative replica-second debt and reports required
recovery.

`repair_controller.py` is an optional in-memory feasibility latch; fixed planning
remains the default. Progress, route rate, replay rate, and observed per-pool
prefill capacity update the versioned ledger. Two consecutive deadline-miss
forecasts request one residual repair, while hard failures request one
immediately. `repair_destination` lands committed work on its concrete replica,
keeps explicitly locked running attempts fixed, prices their measured remainder
with the regional timing components, and minimizes changes to repairable work.
It proposes a diff only when that residual schedule restores the target;
otherwise it reports the attainable shed and leaves execution unchanged.

`repair_shadow_campaign.py` is the narrow RAMR validation: three seeded repeats
each of a sustained 10-to-1 Gbps cut, replay load from rho 0 to 0.8, and both.
It requests two A100-SXM4-80GB GPUs, records but does not apply proposed diffs,
and validates the trigger policy rather than live redirection or performance.

The artifact in `outputs/repair-plan-shift-sim-20260812/` is superseded: that
campaign independently solved degraded snapshots and cannot be interpreted as
within-plan repair evidence. `repair_plan_shift_campaign.py` now starts from the
passing regional A100 timing plan, runs one four-worker schedule to 25% aggregate
planned work, sends two observations through the real repair ledger, and applies
only target-restoring pending-work diffs. Its 16 cells cross bandwidth and
prefill-capacity locations in `{none, east, germany, both}` at 0.1x. Bandwidth-cut
cells remain explicitly labeled sensitivity evidence until the live 0.1x timing
gate passes. The replacement output is in
`outputs/repair-scheduled-sim-20260814/`.

`repair_hardware_campaign.py` prepares the three-region Azure run. A source-side
proxy changes live API/KV route rates without restarting connections, while a
destination gateway imposes an aggregate uncached-prefill completion cap on
background and replay requests. The job first runs 36 regional timing checks
(two destinations, two migration methods, three contexts, three repeats) at
0.1x bandwidth. It launches the 48 repair episodes only if median and p90
relative error are at most 15%, p90 absolute error is at most one second, and KV
reuse is verified. Every proposed diff is shadow-checked before application;
active work cannot be redirected, changes cannot move work toward an impaired
destination, and applied repairs must meet the measured hardware target.

The pinned launch bundle is `outputs/repair-scheduled-hardware-20260814/`:

```bash
export QH_AZURE_SSH_KEY=/path/to/azure-key
export QH_REPAIR_RUN_ROOT=/datadrive/queue-haul-repair-20260814
outputs/repair-scheduled-hardware-20260814/run.sh
```

## Evidence flow

```text
archived raw logs
  -> checksum-pinned reduced measurements
  -> versioned model/workload/pool inputs
  -> result tables with provenance
  -> deterministic figures
```

Every input and result must say whether it is measured, fitted, assumed, or
simulated. Assumed values are sensitivities, never admission guarantees.
The destination campaign's ShareGPT-derived conversation profile pins revision
`192ab2185289094fc556ec8ce5ce1e8e587154ca` and stores only token/turn shapes.

## How to read a workload output

An output directory name does not identify what ran. In particular, `8` and
`28` mean sessions in one episode or modeled pack; they do not mean GPUs,
models, repetitions, or total requests. Read the artifact in this order:

1. `plan.json` defines a hardware campaign's model profile, session rows,
   policies, deadlines, bandwidths, and scenario matrix. `run_metadata.json`
   records the runtime that actually executed it: model, container, ports,
   LMCache mode, Git commit, and other launch settings. Per-scenario results or
   reduced migration CSVs establish what completed.
2. A modeled sweep without `plan.json` uses its `*_metadata.json` as the
   authority. Its `claim`, `inputs`, `limitations`, `sessions_per_pack`, and
   input hashes distinguish modeled reuse of measurements from new model
   execution.
3. A `*-plan` directory is an unexecuted launch bundle. A `simulation/`
   directory contains predictions. PNG/PDF/summary-only directories are
   derived views. Do not promote any of these to hardware evidence merely
   because they reuse a measured profile.

"Workload" has three separate meanings that must not be collapsed:

- **Shape source:** the dataset or manifest from which session identity and
  token/turn counts came.
- **Model payload:** the text or token IDs actually sent to the serving engine.
- **Offered activity:** whether requests continue during migration, their
  arrival schedule, and their concurrency.

The two commonly confused outputs instantiate those layers differently.

### `policy-hardware-width8-packing-20260730`: live two-A100 migration

This is hardware timing evidence. Its frozen plan and run metadata say:

| Dimension | What ran |
|---|---|
| Model/runtime | `openai/gpt-oss-20b`, BF16, TP=1, vLLM 0.22.0 and LMCache 0.5.1 MP in the pinned CUDA 12.9 container |
| Hardware | one RAMR node with two A100 SXM4 80 GB GPUs: one source engine and one destination engine |
| Sessions | the same eight coding-session identities from `outputs/coding-manifest.json` in every scenario |
| Payload | deterministic synthetic calibration messages: a state-code system message, a labeled body of repeated `x` tokens sized from `initial_tokens`, and a state-code probe; the original coding text was not sent |
| Pack shapes | `tiny=8x2048`, `small=8x4096`, `medium=8x8192`, `large=8x16384`, and `mixed=(2048,4096,4096,8192,8192,12288,12288,14336)` nominal context tokens |
| Network/deadline cells | 1, 2.5, 5, and 10 Gbit/s crossed with 19- and 30-second scoring deadlines |
| Policies | Queue-Haul LP, static greedy, KV-only, and replay-only, each paired with a no-migration control |
| Repetition | three repeats per pack/bandwidth/deadline cell |
| Activity | no appended requests and no destination background load during migration; the source and destination remain awake |
| Concurrency | source warming is serialized; all eight migrations are then started in deterministic planned order with width-eight concurrency; destination continuation probes run concurrently |

The matrix is `5 packs x 4 bandwidths x 2 scoring deadlines x 3 repeats =
120` matched cells. Each cell has one control and four migration policies, for
600 scenarios total. All 480 migration scenarios completed all eight moves,
producing 3,840 measured migrations. The per-scenario `deadline_s=180` is the
execution timeout; `required_deadline_s` is the 19- or 30-second value used for
policy admission and scoring.

Each scenario first warms GPT-OSS-20B on the source to materialize session KV.
Replay sends the deterministic prompt to the destination while explicitly
bypassing LMCache, so GPT-OSS recomputes it. KV transfer moves LMCache blocks
and requires the destination request to prove the corresponding cache hits.
After cutover, a continuation request verifies the committed state and route.
The measured evidence is reconstruction, cutover, continuation, and GPU timing;
the reported source-power attainment is projected from the pinned A100 power
profile rather than obtained by keeping eight natural conversations active.

The manifest's `claude:` and `codex:` IDs are labels and state-code seeds here.
Because every plan row supplies `initial_tokens`, the trace's original turn
contents and recorded turn sizes do not define the live prompt. The nominal
pack size is retained in `policy_migrations.csv` as `context_tokens`; raw model
usage at the original run root is the place to inspect tokenizer-reported
prompt and completion counts.

### `workload-power-frontier-20260814`: 28-session modeled sensitivity

This output did not launch GPT-OSS-20B and did not run 28 live requests. It is
a deterministic planner/model sweep whose metadata labels the claim
`modeled`. GPT-OSS-20B enters in two ways: its pinned tokenizer produced the
archived token shapes, and its fitted A100 PCIe 300 W service, migration, KV,
and phase-power measurements parameterize the model.

The content-free manifest combines three shape families:

- coding sessions from `trace-commons/agent-traces`;
- interactive coding conversations from `allenai/WildChat-1M`; and
- agentic tool loops from NVIDIA SWE-Hero OpenHands trajectories.

Only shapes are retained. For each eligible turn, the modeled state is
`context = total input - newly appended input`, `prefill work = newly appended
input`, and `decode work = output tokens`. States outside the 1,536--31,562
context timing support or the phase-power direction support are rejected. The
frozen metadata records 63 supported conversation templates and 3,021
supported turn states.

The evaluated OpenHands split has 24 sessions and 1,383 turns. Session turn
counts have min/p25/median/p75/p95/max `35/52/55.5/67.25/77.95/81`; retained
context across turns has the corresponding token distribution
`0/8,455/15,258/21,591.5/28,855.4/32,415`. Session duration and inter-turn
user/tool waits are not identifiable: all 1,383 selected rows have
`time_s=null`, and the source trajectories provide roles and contents but no
event timestamps. These timing distributions must not be inferred from the
generated request schedule.

For each of 100 draws, the sampler chooses 28 template IDs with replacement,
then one supported turn state from each chosen template. It keeps the 28 raw
contexts but rescales all prefill/decode rates by one common factor so
`sum(f/F + g/G)=0.4`. This is a simultaneous source-state pack, not a timed
request trace: it has no arrivals, think time, prompt text, or generated model
responses. A template can appear more than once in a pack.

Each draw also selects one joint phase-power bootstrap tuple and refits the
regional migration timing from a stratified bootstrap. That complete draw is
then reused across all eight combinations of:

- destination HBM occupancy: 0% or 98%;
- route bandwidth: measured natural East/Germany rates (2.280/8.733 Gbit/s) or
  a 1-Gbit/s cap on both; and
- pre-existing destination compute: 25% or 95% of the modeled envelope.

The virtual topology has one single-GPU source and single-GPU destination pools
in East and Germany. For each of the `100 draws x 8 constraint states = 800`
paired cases, Queue-Haul's HiGHS LP is evaluated at removable-power fractions
`0, 1/8, 1/4, 3/8, 1/2, 5/8, 2/3, 3/4, 7/8, 1`, producing 8,000 raw rows. A
separate integer maximum-shed solve supplies the capacity endpoint. Every point
uses a 30-second power deadline. The profile reserves its final five seconds for
the trailing power window, leaving a 25-second migration budget. The retained
CSV field `target_met_by_30s` therefore means target met by that 30-second
scenario deadline.

Power is steady awake **source** GPU power only; destination power and energy
are excluded. Therefore the figure is a distribution across paired synthetic
workload/calibration draws, not a confidence interval, 800 hardware runs, or
2,800 executed GPT-OSS sessions.

## Three-region Azure A100 campaign

The implemented campaign powers down the Sweden Central source GPU and
reconstructs sessions in East US 2 and West Europe. It uses private IPs over Global VNet
Peering; Azure routes peered-VNet traffic over the Microsoft backbone, not the
public Internet. No Azure CLI access is needed on the VMs. The relevant Azure
contracts are [Global VNet Peering](https://learn.microsoft.com/en-us/azure/networking/design-guide/cross-region),
[Linux PTP/chrony](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/time-sync),
and [Spot Scheduled Events](https://learn.microsoft.com/en-us/azure/virtual-machines/windows/scheduled-events).

The H100 path uses West US 3 (`10.11.0.4`) as source and Australia East
(`10.12.0.4`) plus South Central US (`10.13.0.4`) as destinations via
`azure_network_cluster_australia_southcentral.json`. Set
`QH_MODEL_PROFILE=gpt_oss_20b_h100_tp1.json`; the A100 profile remains the
network campaign default so archived plans retain their original meaning.

The node map across the provided cluster files is:

| role | region | private IP |
|---|---|---|
| source/power-down | Sweden Central | `10.0.0.4` |
| destination | East US 2 | `10.1.0.4` |
| destination | West Europe | `10.2.0.4` |
| optional destination | Germany West Central | `10.3.0.4` |

Use `azure_network_cluster_germany.json` for an isolated Germany run when West
Europe is unavailable, or `azure_network_cluster_east_germany.json` for the
two-destination East US 2 and Germany campaign.

`check` compares every selected entry with Azure IMDS and hard-fails before
calibration if an address assignment differs. Correct the selected node record;
do not bypass the check.

### One-time portal work for the Azure account owner

The account owner must complete these items. The experiment operator does not
need `az` permissions.

1. Use `Standard_NC24ads_A100_v4` Spot VMs with one visible A100 each,
   Azure Linux 3.0, persistent `/datadrive`, eviction policy `Deallocate`, and
   no delete-on-eviction data disk.
2. Configure bidirectional Global VNet Peering between the Sweden Central VNet and
   each selected destination VNet. Each peering must show `Connected`; address spaces
   must not overlap. Destination-to-destination peering is unnecessary.
3. Do not add a public data-plane address, NAT gateway, VPN, load balancer, or
   TLS terminator. SSH can use the existing private access path. Private Azure
   backbone traffic is not application-layer encryption; that is acceptable for
   this measurement-only, private-VNet deployment.
4. Restrict NSGs to these experiment flows. Source egress goes only to each
   destination on TCP `22,5201,8081,8200` and ICMP. Each selected destination
   permits source `10.0.0.4/32` on those ports. Sweden Central
   permits TCP `8301` from East `10.1.0.4/32`, West `10.2.0.4/32`, or Germany
   `10.3.0.4/32`, according to the selected isolated cluster. The joint cluster
   also uses TCP `8302` from West. Ports `5556,5557,5655,8080,8100,8401,8402` remain
   host-local. Do not expose any experiment port to `0.0.0.0/0`.
5. Confirm all selected VMs have the repository at
   `/home/azureuser/coding-progress-ledger/agent-migrate`, the same commit, and
   the source has `~/.ssh/azrs` plus verified host keys for each destination.

### Install each selected host

Run from the `agent-migrate` repository on every VM as `azureuser`, not root:

```bash
bash queue-haul/setup.sh
source ~/.bashrc
```

This installs Valkey, `chrony`, and `iperf3`, configures chrony against Azure's
stable `/dev/ptp_hyperv` device, waits for synchronization, installs the pinned
Python 3.12/vLLM 0.22.0/LMCache 0.5.1 CUDA 12.9 runtime, and stores the pinned
GPT-OSS-20B model and caches under `/datadrive`. Setup hard-fails without
`nvidia-smi`, the persistent data mount, PTP device, or pinned runtime.
For the A100 three-model architecture/drain campaign, install its isolated
compatibility contract on every selected host instead:

```bash
QH_VLLM_VERSION=0.24.0 QH_TRANSFORMERS_VERSION=5.15.1 bash queue-haul/setup.sh
source ~/.bashrc
```

Setup records the selected vLLM/LMCache pair in the managed shell block and
always rebuilds the native planner extension from the checked-out source.

From the Sweden Central source, establish and verify SSH host keys once, then confirm
that the same commit is checked out everywhere:

```bash
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 true
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 true
ssh -i ~/.ssh/azrs azureuser@10.3.0.4 true
git rev-parse HEAD
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
ssh -i ~/.ssh/azrs azureuser@10.3.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
```

### Calibrate, smoke-test, and run

Run every command below from the Sweden Central `agent-migrate` directory. Do not
start a formal run with a dirty tracked worktree.

```bash
source ~/.bashrc
mkdir -p /datadrive/queue-haul-network/control

uv run python queue-haul/network_campaign.py check \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs

uv run python queue-haul/network_campaign.py calibrate \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --out /datadrive/queue-haul-network/control/calibration.json
```

Formal calibration takes three 60-second repeats. It records 200 RTT samples per
path, isolated one- and eight-stream `iperf3`, simultaneous eight-stream
receiver goodput to both destinations, all raw iperf JSON, host fingerprints,
and clock uncertainty. Controlled 40% and 80% rates come from simultaneous—not
isolated—receiver goodput, with an aggregate source-NIC cap. Clock uncertainty
above 2 ms is a hard failure.

Run both an unshaped and shaped end-to-end gate before planning:

```bash
uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --bandwidth natural \
  --run-root /datadrive/queue-haul-network/smoke-natural

uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --bandwidth controlled_40 \
  --run-root /datadrive/queue-haul-network/smoke-controlled-40
```

Each smoke must prove nonzero KV wire bytes and cached tokens at both remote
destinations, then sleep and wake the source GPU. Use new smoke directories;
existing directories are rejected.

Prepare and run the targeted campaign:

```bash
# Validate one route first with 54 paired replay/KV migrations.
uv run python queue-haul/network_campaign.py prepare \
  --design isolated \
  --cluster queue-haul/azure_network_cluster_east.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-post-west-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-east-validation.json

uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east.json \
  --ssh-key ~/.ssh/azrs \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-post-west-001.json \
  --plan /datadrive/queue-haul-network/control/plan-east-validation.json \
  --run-root /datadrive/queue-haul-network/validation-east-001

# Prepare the joint campaign after both routes validate.
uv run python queue-haul/network_campaign.py prepare \
  --design joint \
  --cluster queue-haul/azure_network_cluster.json \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-joint.json

uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --current-calibration /datadrive/queue-haul-network/control/calibration.json \
  --plan /datadrive/queue-haul-network/control/plan-joint.json \
  --run-root /datadrive/queue-haul-network/formal-001
```

An isolated plan pairs replay and KV at 2K, 8K, and 32K contexts, route-relative
40%, 80%, and natural bandwidth, and three repeats: 54 migrations per site. The
West run uses `azure_network_cluster_west.json` with its one-path calibration.
Multi-site destination services start concurrently for each bandwidth stack.
Resume permits a synchronized commit update while keeping all other run identity
fields pinned and retaining both commits in the metadata checks.
joint design is 7 targeted agentic-trace conditions x 3 repeats x 6 policies =
126 physical scenarios. The policies are Queue-Haul, greedy, Lagrangian greedy,
KV-only, replay-only, and seeded feasible random. Its 20%-compute/20%-KV anchor
uses controlled 80% bandwidth and a 30-second deadline. The other cells are
idle destinations; crossed 20/40 and 40/20 compute/KV pressure in both
directions; controlled 40%; natural bandwidth; and a 19-second deadline.

Joint scenarios do not preassign a destination. After five seconds of seeded
background warmup and five one-second vLLM metric samples, the selected policy
chooses destination, replay or KV, and order for every session. Declared work
comes from pinned agentic turn rates and prefill/decode tokens; measured live KV
usage is also a planner input. Trace demand is normalized to the campaign's
existing total source-load contract of `ell=0.4`. A live-state deviation over five
percentage points or any waiting request is retained as a warning, not a failed
measurement. Missing metrics, invalid reconstruction, or missing KV evidence
remain hard failures.

The runner retries one malformed state-code probe, then hard-fails with its
response excerpt. It checkpoints each decision and scenario result atomically and fsyncs
them. `progress.json` records completed scenario IDs and counts after every
attempt. Rerun the same command and run root to skip every completed scenario;
an interrupted or failed `attempt-NNNN` remains intact and resume uses the next
attempt number. A failed attempt stops immediately instead of repeatedly cold
loading the models. After any Spot deallocation, restart the VMs, rerun `setup.sh`
and `check`, write a fresh formal
calibration file, and resume with that file as `--current-calibration`. Resume
hard-fails if RTT or simultaneous goodput drifts more than 10%, or if the plan,
node identity, model, or runtime changed. A synchronized commit update remains
visible in the audit checks. Azure Scheduled Events are
logged on all three hosts and an active Spot event fails the attempt.

The handoff accepts any validated two-destination cluster. For East US 2 and
Germany West Central, create one fresh calibration and matched plan after East
comes online, then pass the same repeat to Queue-Haul, KV-only, and replay-only:

```bash
uv run python queue-haul/network_campaign.py check \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs
uv run python queue-haul/network_campaign.py calibrate \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs \
  --out /datadrive/queue-haul-network/control/calibration-east-germany-001.json
uv run python queue-haul/network_campaign.py prepare --design joint \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-east-germany-001.json
uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --bandwidth natural \
  --run-root /datadrive/queue-haul-network/smoke-east-germany-natural-001

for policy in queue_haul kv_only replay_only; do
  uv run python queue-haul/network_campaign.py handoff \
    --cluster queue-haul/azure_network_cluster_east_germany.json \
    --ssh-key ~/.ssh/azrs \
    --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
    --plan /datadrive/queue-haul-network/control/plan-east-germany-001.json \
    --manifest queue-haul/outputs/coding-manifest.json \
    --policy "$policy" --repeat 0 \
    --run-root "/datadrive/queue-haul-network/handoff-east-germany-$policy-001"
  uv run python queue-haul/plot_handoff_power.py \
    --run-root "/datadrive/queue-haul-network/handoff-east-germany-$policy-001"
done
```

The East/Germany frontier campaign uses measured natural bandwidth only. Its
185-episode pilot is six movement packs (4x16K, 8x16K, 16x16K, 8x8K, 8x24K,
and 8x31K), destination loads 0, 0.5, 0.85, 0.9, and 0.95, plus the seven-cell
8x16K asymmetric slice. Every matched cell runs Queue-Haul LP, greedy,
replay-only, KV-only, and power-blind Queue-Haul against a 30-second deadline
and an 80% modeled removable-power target. The source load is 80%; replay
requests explicitly bypass LMCache and KV requests require positive cache
evidence only as a warning.

The H100 frontier keeps the 4x16K, 8x16K, and 16x16K width bridge, expands the
8K, 24K, and 31K packs to width 16, and adds one 32x31K red-zone tail. Its 304
matched scenarios cover Queue-Haul, greedy, Lagrangian greedy,
isolated-fastest, KV-only, replay-only, power-blind, and deadline-blind using
only the two measured natural WAN paths; no bandwidth cap or fixed destination
split is applied. H100 destination load is normalized to the measured
604-prefill/64-decode service rates used by the live background generator.

```bash
uv run python queue-haul/network_campaign.py prepare --design frontier \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/frontier-pilot.json
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --run-root /datadrive/queue-haul-network/frontier-pilot-001
uv run python queue-haul/network_campaign.py refine \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --run-root /datadrive/queue-haul-network/frontier-pilot-001 \
  --out /datadrive/queue-haul-network/control/frontier-refinement.json
uv run python queue-haul/network_campaign.py deadline-blind \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --plan /datadrive/queue-haul-network/control/frontier-refinement.json \
  --out /datadrive/queue-haul-network/control/frontier-deadline-blind.json
```

Reduction writes raw episode CSV plus PNG/PDF prefill--network mechanism and
target-attainment figures. Refinement adds 0.875 or 0.925 load midpoints where
an action or attainment boundary appears and caps the first adaptive stage at
65 matched episodes, for 250 pilot-plus-refinement episodes total. New midpoint
cells receive repeats first. A second refinement takes unstable cells to ten
repeats when the action changes or the 95% shed-width exceeds 10 W. Deadline
misses, target misses, individual request failures, load drift, queueing, and
missing secondary telemetry remain observations. Invalid identity or inputs,
unusable primary outcomes, and failure of more than half of the planned
episodes stop the campaign.

The deadline-blind plan selects one Queue-Haul ablation for every unique
condition/repeat block in the pilot and capped refinement. It plans against a
nonbinding 600-second horizon while execution and attainment retain the measured
30-second deadline.

`plot_network_power_attainment_cdf.py` combines the pilot, refinement, and
deadline-blind phases into matched 50-episode-per-policy ECDFs. Each event is the
earliest common-epoch completion at which nonlinear modeled source-power shed
reaches the 80% target; late events remain and unattained targets are missing
mass. Run `uv run python queue-haul/plot_network_power_attainment_cdf.py`; it
writes CSV, PNG, and PDF outputs to the frontier campaign root using Tab10
policy colors and the longest observed episode runtime as the plotting horizon.

The constrained East/Germany diagnostic is one frozen 24-episode run, not a
sweep. It uses the measured simultaneous natural paths (2.280 Gb/s East and
8.733 Gb/s Germany), 80% source load, and the original generator's 50% East
and 95% Germany source-rate-normalized load labels. Those labels are not
destination service utilization, so this campaign remains a migration-window
capacity diagnostic rather than evidence that destination service binds. Its
19-second cell has 22 exact recorded-support contexts selected with disclosed
seed 15 (513,650 tokens); its 30-second cell has 28 contexts from seed 8
(648,131 tokens); and its 60-second cell has eight copies of each trace at
14,042 tokens (898,688 tokens). Every cell requests the full 61.86 W removable
pack power, so the attained result is a deadline-constrained capacity point
rather than an easy target. The fourth cell reuses the exact 30-second pack
while limiting Germany replay to 25% of its migration window, or 6.25
replica-seconds. This quota is an operator counterfactual, not a measured
load-slowdown coefficient.

Every cell runs exact maximum-shed Queue-Haul, Queue-Haul greedy, KV-only,
replay-only, per-session-fastest, and power-blind Queue-Haul once. The exact
binary solver jointly chooses sessions, methods, and destinations to maximize
removed single-source load without a secondary tie-break. In
the pinned simulation at `outputs/east-germany-constraint-20260808/`, it sheds
49.25, 55.92, 60.48, and 51.69 W at 19, 30, 60, and quota-constrained 30
seconds. Greedy reaches 46.99, 51.69, 60.48, and 51.69 W. The full-pack request
is intentionally unattainable in every cell. Under the replay quota,
Queue-Haul chooses 5 KV→Germany, 7 replay→East, 3 KV→East, and 1
replay→Germany. All four method--destination migration windows have positive
Phase-I additive-surrogate prices; direct route and destination-service prices
are zero. Exact nonlinear bundle shed is recomputed after integral packing.

```bash
uv run python queue-haul/network_campaign.py prepare --design constraint \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/constraint.json
uv run python queue-haul/network_campaign.py simulate-constraint \
  --plan /datadrive/queue-haul-network/control/constraint.json \
  --out /datadrive/queue-haul-network/constraint-simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/constraint.json \
  --run-root /datadrive/queue-haul-network/constraint-001
```

The runner keeps collecting after an episode failure but hard-fails after final
reduction unless all 24 episodes complete and meet their deadline, every
intentionally oversized target remains unmet, and no episode contains a
request, KV-evidence, load-drift, or queueing warning.
Migration timing ends when parallel reconstruction finishes; draining background
load happens afterward and is excluded from `migration_s` and `deadline_met`.
Each background generator caps pending work at its eight request workers, so
overload cannot create a stale client-side queue that lengthens episode cleanup.
Reduction writes matched Tab10 attainment and action-composition figures; the
simulator additionally writes the Phase-I dual table and figure. There is no
adaptive refinement or CDF for this single-block diagnostic.

The separation campaign is the service-binding follow-up. It pins the same
28-session recorded-support pack (648,131 tokens, seed 8) in every matched
cell, uses measured destination prefill and decode rates to generate exact
604-prompt/64-output-token background requests, warms that traffic for 30
seconds, and then launches all selected migrations concurrently. Each policy
plans inside 30 seconds and is measured against a 45-second hardware deadline.
Background prompts stay fixed while unique cache salts prevent reuse, and
forced-length decoding makes every service-normalized request exactly 604/64.
Attainment credits only reconstructions ending by the common migration
start plus 45 seconds; reduction recomputes this value from raw request
timestamps, including for results written by an earlier runner revision.

| cell | paths | East/Germany service load | target | Queue-Haul | greedy | strongest losing baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Germany service | natural | 25% / 95% | 37.11 W | 47.72 W | 41.77 W | 32.51 W |
| East slow path | natural | 90% / 25% | 47.63 W | 55.92 W | 53.72 W | 41.88 W |
| joint shaped | 0.9 / 3.4 Gb/s | 50% / 85% | 33.40 W | 51.06 W | 37.22 W | 29.72 W |

The 63 episodes are three cells by three repeats by seven policies: exact
maximum-shed Queue-Haul, target-aware greedy, KV-only, replay-only,
per-session-fastest, power-blind, and deadline-blind Queue-Haul. The simulator
hard-fails unless Queue-Haul and greedy both use at least two KV and two replay
actions, span both destinations, and attain at least 110% of target; each of
the five ablations must remain at or below 90%. Deadline-blind must claim feasibility at
600 seconds and then fail the 45-second evaluation. At least three Queue-Haul
resources per cell must reach 89--98% utilization, including Germany service
at 98%, all four action windows in the East cell at 92--98%, and both services
plus all four action windows in the joint cell at 89--97%. Their Phase-I duals
must also be positive, except for degenerate East-service pricing. Thus the
certified separation is at least 20 percentage points, with 50% extra hardware
time beyond the planning window; the hardware reducer applies the same attainment gates and also
requires clean requests, cache evidence, destination load, and queue telemetry.

```bash
uv run python queue-haul/network_campaign.py prepare --design separation \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/separation.json
uv run python queue-haul/network_campaign.py simulate-separation \
  --plan /datadrive/queue-haul-network/control/separation.json \
  --out /datadrive/queue-haul-network/separation-simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/separation.json \
  --run-root /datadrive/queue-haul-network/separation-001
```

The frozen local certificate is in
`outputs/east-germany-separation-20260809/`. Simulation proves the selected
model points have the intended separation. The completed 63-episode A100
hardware run had no request or load failures. Across Germany-service,
East-slow-path, and joint-shaped cells, Queue-Haul's median attainment was
128.6%, 117.4%, and 152.9%; greedy reached 112.5%, 112.8%, and 111.4%.
KV-only, replay-only, isolated-fastest, and power-blind stayed below 90% in
every cell. Deadline-blind finished the full move set after 45 seconds in all
three cells, but the joint-shaped cell already crossed its power target, so it
is not a clean deadline-blind negative control. One greedy repeat also lacked
clean KV evidence. Consequently all 63 executions completed, but the strict
certificate is invalid in nine episodes. The measured run further confirms
that destination KV occupancy was near zero, leaving KV capacity untested.
The compact reduction, plots, run identity, and verified 438-entry raw-artifact
manifest are retained in `outputs/east-germany-separation-hardware-20260809/`;
the large per-request and power traces remain in the hardware run root.

`simulate-oracle-stale` is the no-new-campaign constraint and stale-information
certificate. It reuses that exact 28-session recorded pack, measured East and
Germany paths, measured destination service profile, and the measured 75%
service-load support point. The constructed all-bind corner reserves 90% of
East's profiled A100 KV capacity, applies 75% Germany service load, and uses
40%-controlled caps derived from the measured paths (0.9/3.4 Gbit/s). The
request is 40.21 W, or 65% of removable pack power. Exact restricted max-shed
oracles all use forced normal admission and change only the allowed action or
destination: Queue-Haul reaches 49.13 W, versus 27.31 W for KV only, 30.90 W
for replay only, 22.73 W for East only, and 24.31 W for Germany only. East KV
and Germany service reach 95% and 98% utilization and both have positive
Phase-I duals.

The eight corners independently release East KV, Germany service, and path
bandwidth. A single exact worst-corner plan reaches 49.13 W by the deadline in
all eight, while fresh replanning reaches 49.13--55.92 W. The all-release plan
is capacity-invalid at the all-bind corner because it exceeds both East KV and
Germany service. As a negative control, Germany-only reaches 41.88 W once all
constraints are released. Deadline-aware Queue-Haul crosses 40.21 W at 23.33
seconds; the 90-second deadline-blind ablation eventually reaches 54.20 W but
has only 31.37 W at 45 seconds and first crosses the same target at 58.19
seconds. Generated Tab10 figures, exact moves, nonlinear attainment curves,
resource use, duals, and checksums are in
`outputs/east-germany-oracle-stale-20260809/`.

```bash
uv run python queue-haul/network_campaign.py simulate-oracle-stale \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --out queue-haul/outputs/east-germany-oracle-stale-20260809
```

The 72-episode hardware-gap follow-up turns those missing mechanisms into five
matched operating points using the same 28-session recorded pack, profiled
destination service, source-power model, and measured routes. Its all-bind
point physically limits East to 10% of profiled vLLM KV blocks, loads Germany
to 75% service utilization, and applies the measured 40% route caps. It tests
Queue-Haul against greedy, four exact restricted oracles, isolated-fastest,
power-blind, deadline-blind, and a frozen all-release plan. One-at-a-time
release controls restore East KV, Germany service, or path bandwidth; the
all-release control admits the previously stale plan. Every block has three
matched repeats, forced normal admission, and concurrent migrations. The fixed
worst-corner robust plan also runs in every released state, so its cross-state
guarantee is hardware-tested rather than inferred only from simulation.

The frozen simulation requests 44.54 W, or 72% of removable pack power.
All-bind Queue-Haul reaches 49.13 W while every losing baseline reaches at most
39.60 W; the stale plan is rejected because East KV and Germany service both
overflow. Deadline-blind reaches enough eventual power only after 59.31
seconds. Releasing KV, service, and bandwidth expands the corresponding exact
restricted oracle by 4.58, 12.91, and 7.44 W. The runner parses each vLLM
engine's reported KV-token capacity and hard-fails unless it matches the
planned fraction within one percentage point, preventing a labeled-only quota.
Planning additionally reserves 668 KV tokens per destination for one continuing
background request. Eventual attainment stops at the frozen 90-second horizon;
responses after it cannot satisfy the deadline-blind control.

```bash
uv run python queue-haul/network_campaign.py hardware-gap \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --oracle-plans queue-haul/outputs/east-germany-oracle-stale-20260809/plans.json \
  --out queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json
uv run python queue-haul/network_campaign.py simulate-hardware-gap \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-20260809/simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration queue-haul/outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --run-root /datadrive/queue-haul-network/hardware-gap-001
```

The completed 72-episode A100 hardware run had no failed or missing scenarios,
and all 585 retained raw artifacts verify. At all-bind, robust Queue-Haul shed
49.13 W (110.3% of target) in 24.56 seconds, while greedy shed 39.60 W
(88.9%). KV-only, replay-only, East-only, Germany-only, isolated-fastest, and
power-blind all stayed below target. Releasing KV, service, and bandwidth
increased their corresponding restricted oracle by 4.58, 12.91, and 7.43 W;
the robust plan continued to shed 49.13 W in every state. The stale plan was
rejected at all-bind and reached 55.92 W at all-release. Deadline-blind was not
a valid negative control: all three repeats crossed the target in 38.89--43.29
seconds, before the 45-second cutoff, although their full migrations ended in
54.37--56.67 seconds. The strict reducer therefore reports three invalid
episodes. The compact reduction, plots, run identity, and verified raw-artifact
manifest are retained in
`outputs/east-germany-hardware-gap-hardware-20260810/`; large traces remain in
the raw run root.

`plot_hardware_constraint_timeline.py` reconstructs the all-bind repeat-0
Queue-Haul, power-blind, and deadline-blind resource accounting. KV and service
curves are residual headroom consumed after measured cutover; migration curves
are modeled work charged to Queue-Haul's 30-second planning budget in measured
completion order, not sampled instantaneous utilization.

```bash
uv run python queue-haul/plot_hardware_constraint_timeline.py \
  --raw-root /datadrive/queue-haul-network/hardware-gap-001 \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-hardware-20260810/constraint_timeline
```

The requested-shed frontier is a model sweep over that frozen all-bind hardware
scenario, not additional hardware observations. It retains raw overshed above
the requested-equals-attained diagonal and carries the last safe attainment
forward when a larger request is unsafe. Its second panel reports Queue-Haul
LP's modeled resource pressure and action/destination mix; those are diagnostic
rather than the primary comparison.

```bash
uv run python queue-haul/plot_hardware_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-frontier-20260810/shed_frontier
```

The pooled publication view removes those diagnostic panels and standardizes
twelve constraint, separation, and hardware-gap operating points to a common
30-second cutoff. Requested and attained shed are normalized by each case's
removable power before equal-weight pooling; raw overshed remains visible above
the diagonal. Lines are medians and ribbons are the interquartile spread across
designed cases, not repeated-run confidence intervals. Deadline-blind plans
against 90 seconds but receives credit only for shed attained by the common
30-second cutoff. The x-axis ends at 80%, just beyond the nonzero intersection
of Queue-Haul LP's median frontier with requested-equals-attained parity; the
y-axis retains the full 0--100% removable-power range.

The companion attainment-time ECDF uses the same twelve cases at the common
67% stress point. Each event is the first modeled target crossing plus the
profile's power-window delay; misses retain missing CDF mass. The 30-second
line is the common evaluation deadline, while the 90-second horizon exposes
late deadline-blind attainment:

```bash
uv run python queue-haul/plot_pooled_attainment_cdf.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_attainment_cdf
```

The H100 counterpart pools the five completed hardware-gap states using the
measured H100 profile. Its star is the median of 15 matched Queue-Haul robust
hardware runs at the common 41.4% request; the curves remain modeled 30-second
frontiers rather than interpolated hardware measurements:

```bash
QH_MODEL_PROFILE=gpt_oss_20b_h100_tp1.json uv run python \
  queue-haul/plot_pooled_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-hardware-gap-h100-20260812/plan.json \
  --hardware-results queue-haul/outputs/east-germany-hardware-gap-h100-20260812/results.csv \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-h100-20260812/pooled_shed_frontier
```

```bash
uv run python queue-haul/plot_pooled_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier
```

The companion hardware target-attainment plot retains uncapped overshed for the
six deadline-safe policies in the 63 separation episodes. It divides realized
shed by requested shed within each episode before pooling the three repeats, so
values above 100% explicitly show target overshoot rather than extra physical
efficiency. Deadline-blind is omitted because its recorded shed is eventual,
not shed attained by the deadline.

The matched A100 migration-timing parity view compares the pre-run modeled
episode makespan with the measured hardware migration duration for Queue-Haul
LP and Queue-Haul Greedy across the nine separation conditions/repeats:

```bash
uv run python queue-haul/plot_migration_timing_parity.py \
  --predictions queue-haul/outputs/east-germany-separation-20260809/simulation/separation_predictions.csv \
  --measurements queue-haul/outputs/east-germany-separation-hardware-20260809/results.csv \
  --out queue-haul/outputs/east-germany-migration-timing-parity-20260813/migration_timing_parity
```

```bash
uv run python queue-haul/plot_hardware_target_attainment.py \
  --results queue-haul/outputs/east-germany-separation-hardware-20260809/results.csv \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-target-attainment-20260810/target_attainment
```

The companion resource-pressure view compares the same cases at two-thirds of
removable power, where Queue-Haul usually succeeds and the restricted policies
usually fail. Its four facets sum physical use and capacity across destinations
for prefill service, KV headroom, replay time, and bandwidth-sensitive KV
transfer time. Points are designed cases, filled when the target is met by 30
seconds; diamonds and whiskers are the equal-case mean and 95% case-bootstrap
interval.

```bash
uv run python queue-haul/plot_pooled_resource_pressure.py \
  --cases queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier_cases.csv \
  --out queue-haul/outputs/east-germany-pooled-resource-pressure-20260810/resource_pressure
```

The action-adaptation views use the same equal-case sweep. The primary chart
shows Queue-Haul's total replay/KV composition for the three single bottlenecks,
all bottlenecked, and none bottlenecked at a common 67% target. Gray reports
sessions left at the source, so every 100%-stacked bar accounts for the same
28-session pack; destination identities are intentionally omitted. The raw
tables retain all eight HBM/bandwidth/destination-compute combinations.

```bash
uv run python queue-haul/plot_pooled_action_adaptation.py \
  --cases queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier_cases.csv \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out-dir queue-haul/outputs/east-germany-action-adaptation-20260811
```

`bootstrap_action_adaptation.py` repeats those eight matched cases under 1,000
paired calibration draws. It stratifies timing by destination, method,
bandwidth, and context and samples each fitted phase-power tuple jointly. This
is a modeled calibration-sensitivity distribution for the fixed 28-session
pack, not 8,000 independent hardware observations. The stacked bars show joint-
bootstrap mean shares; black whiskers mark the 5--95% Replay and total-moved
boundaries, with a three-facet interval companion for the full composition.

```bash
uv run python queue-haul/bootstrap_action_adaptation.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out-dir queue-haul/outputs/east-germany-action-adaptation-20260811
```

`workload_adaptation_campaign.py` adds workload-shape sensitivity without
claiming new hardware observations. Each of 1,000 paired draws resamples
conversation templates and then one whole supported state tuple per template,
normalizes the 28-session source pack to the
campaign's 0.4 load, refits the balanced regional timing cells, and samples one
joint phase-power bootstrap tuple. All eight HBM/bandwidth/destination-compute
states share
that draw. The planner separates each region's measured physical route from its
calibrated effective migration pipeline. The bandwidth state caps both physical
routes at 1 Gbit/s, the predeclared lower boundary of the existing A100 loaded-
migration validation, and retains each region's controlled pipeline fit. Network
transfer overlaps destination migration work,
while Replay and KV endpoint work share one conservative capacity envelope;
endpoint replica-seconds remain a physical capacity row while isolated
candidate duration is the common action objective. Queue-Haul can therefore
avoid route-heavy KV transfers and leave the physical route slack after the
bottleneck has reduced the available opportunity. The HBM stress state
uses 98% baseline occupancy on both destinations; the producer rejects any
single-factor label unless it activates in at least 90% of paired draws and at
least 10% of paired plans respond.
In the regenerated ensemble, the cap replaces the no-bound KV share with Replay;
the source-power frontier stays close to none bottlenecked because Replay transfers
little network state. That is the modeled adaptation, not an inactive link.
Background
inference consumes shared prefill/decode destination-compute headroom. Replay
endpoint work is multiplied by the measured relative factor
`exp(0.284963 * rho)` at the incumbent normalized destination load; KV is
load-neutral centrally because its paired bootstrap spans zero. The regional
concurrency-one fit remains the exact rho=0 anchor. Every enabled
factor is applied to both destinations, using region-specific route rates.
Within the measured regional 1,536--32,256-token migration support, Replay uses
the base rate curve where available and its conservative minimum rate outside
that narrower curve; candidate duration and shared migration work use the same
timing components but retain distinct objective and capacity roles.
The stacked chart and companion boxplot show the three single bottlenecks, all
bottlenecked, and none bottlenecked; intermediate two-factor states remain in the raw tables.
Each stacked bar is the mean modeled source phase-load share across paired
draws, the additive quantity used by the exact nonlinear power target. The
separate `action_mix_boxplot.pdf` shows session-count
variation for HBM, bandwidth, destination compute, all bottlenecked, and none bottlenecked:
each x-position has Replay, KV-transfer, and not-moved boxes spanning the
25th--75th percentiles, median lines, and 5th--95th-percentile whiskers. Raw
session counts and count shares remain in the CSV; they intentionally differ
from phase-load weighting because a few sessions can carry most of the load.
Target misses and one-factor-release checks remain in the output tables.
The eight states are independent branches. Their fractional LP opportunity
sets must expand on every release. If greedy LP rounding misses the target, an
integral recovery minimizes migration work under the same target and resource
constraints. Noisy bootstrap route draws
are minimally projected to preserve natural bandwidth at or above the measured
40%-route condition, and the projection rate is recorded.

The same command writes `action_choice_oat_bandwidth.pdf`,
`action_choice_oat_prefill.pdf`, and `action_choice_oat_density.pdf`. Both
50-level sweeps use the same 1,000 seeded packs of eight OpenHands sessions.
Each pack samples trajectories without replacement and then one supported turn
per trajectory; that exact pack is planned at every resource level. Calibration
and source load (0.4) remain fixed, and the planner must meet 100% of removable
session-induced source power. The bandwidth sweep fixes prefill at the median
of nine A100 saturation-throughput observations: three repeats at each of
4,096, 16,384, and 24,576 context tokens. This pooled median
(5,342.4 token/s) is a chosen scalar control equal to 91.5% of the modeled
7,680-token rate, not a direct measurement at 7,680 tokens or an estimate of
generic A100 capacity. The prefill sweep fixes routes at natural capacity,
includes that exact shared operating point, and extends from a synthetic 1% of
the raw observed upper anchor through that anchor (5,690.2 token/s). The upper
anchor is one 4,096-token observation, not full A100 capacity. These are
conditional one-factor sensitivities; they do not identify a
bandwidth-by-prefill interaction or general main effects.

The clean stacked plots report Monte Carlo mean modeled session shares over
8,000 dependent session decisions at each resource level; their lower panels
report `Deadline-Met (%)`: the fraction of packs that attain the full 100%
removable-power target by the 30-second deadline, including its trailing
five-second power window, not merely the fraction whose admitted partial moves
finish by the deadline. The density
figure reports the discrete pack distributions that directly support the two
claims:
`P(KV count | bandwidth)` and `P(migrated count | prefill)`. Color, rather than
a smoothed percentile ribbon, represents density. Pack definitions are in
`action_choice_oat_packs.csv`, the 100,000 paired results are in
`action_choice_oat_plans.csv`, and plotted frequencies are in
`action_choice_oat_distribution.csv`. Regional pipeline timing is interpolated
between measured endpoint fits, so these are modeled workload sensitivities
rather than hardware action observations.

```bash
uv run python queue-haul/workload_adaptation_campaign.py
uv run python queue-haul/workload_adaptation_campaign.py --oat-only
```

To rerun with the bandwidth sweep's fixed prefill throughput halved to
2,671.188 token/s, preserving the original outputs:

```bash
uv run python queue-haul/workload_adaptation_campaign.py --oat-only --oat-fixed-prefill-tps 2671.1881753955605 --out queue-haul/outputs/workload-action-adaptation-half-prefill-20260907
```

This uses the same seeded packs and bandwidth levels. The paired prefill sweep
keeps its original endpoints and includes the new shared operating point;
metadata retains the measured median separately from the fixed control.
At the maximum 8.733-Gbit/s bandwidth, halving prefill increases modeled KV
transfer share from 13.8% to 34.125%; full-target deadline attainment is
97.3--97.4% across the bandwidth sweep, compared with 99.8% originally.

The main factorial's 1,000 draws are a modeled calibration/workload
sensitivity, not a confidence interval or 1,000 independent workloads. The
OAT sweeps instead fix calibration across 1,000 Monte Carlo workload packs from
the 24 manifest-selected OpenHands trajectories. The
regional single-move timing holdout passes its recorded gate; grouped local
and width-8 timing audits remain retrospective. Route and endpoint work use
the pipeline overlap already present in the calibrated effective rate;
cross-method endpoint work remains
fully shared rather than fitting partial overlap from KV-heavy mixed evidence.
The width-8 audit has zero false-feasible cases at the 25-second migration
horizon. The separate local c1--c4 audit has 24/162 false-feasible cases,
including 5/66 in its grouped split, so the bars are a regional modeled action-
mix sensitivity rather than a generic hardware deadline guarantee.
Two otherwise timing-supported trace states are excluded
because their prefill/decode direction lies outside the phase-power calibration
cone; the output records that exclusion and retains a post-plan hull hard gate.

`loaded_service_model.py` fits that factor from 160 equal-cell-weighted,
fixed-width forced-action A100 episodes. A 440-episode, 1--10-Gbit/s check
validates the relative width-8 factor, not the deployed regional loaded model.
At rho=.95, Replay endpoint work is 1.311x its idle value. The fitted width-8
intercept is diagnostic and is never applied to regional timing. The loaded
pack covers 2,048--14,336 tokens and is prefill-heavy, so context and load-shape
transport remain explicit sensitivities. The output counts every selected,
candidate, and sampled use outside that context range or the validated bandwidth
range. A support-restricted table resamples only 2,048--14,336-token states with
the same timing and power draws; because it changes the workload population, it
is not a within-pack counterfactual. The action ensemble fixes the load slope at
its central fit rather than propagating its bootstrap. Resume TTFT is retained
as a diagnostic but is not a planner resource: its adverse 1-Gbit/s cells fail
the prediction gate. The existing service-debt machinery remains disabled until
a completed headroom campaign identifies an SLO budget. HBM remains
method-independent because either method leaves the same resident KV state.

`workload_power_frontier.py` carries 100 deterministic paired draws through all
eight constraint states. For each draw and state, the main figure uses an
integer MILP that maximizes removed phase load `sum(a*f + b*g)` under the same
session, route, HBM, migration, and destination-compute constraints. The MILP
has a certified 0.25% relative gap; exact nonlinear source watts are evaluated
after packing. A feasible result from a tighter paired state is retained after
constraint release, so the reported capacity cannot physically decrease; an
independent-solve inversion above 1% hard-fails. The figure shows the three
single bottlenecks, all bottlenecked, and none bottlenecked, while the raw table retains all
eight states and every target-specific Queue-Haul LP solve. Watts are normalized
by each draw's removable source power. The current phase profile is
`outputs/azure-compact-calibration-20260813/gpt_oss_20b_a100_tp1_azure_300w_phase.json`;
one joint `(p0, delta, a, b)` bootstrap tuple is shared by all eight states in a
draw and recorded in every row. The companion `_power.csv` retains 5th, median,
and 95th percentile watts. These are modeled sensitivity ranges, not confidence
intervals or new hardware observations. The frontier and action-mix PDFs use
compact canvases intended for side-by-side `0.49\\columnwidth` placement.

```bash
uv run python queue-haul/workload_power_frontier.py
```

The time-to-binding view uses the same two-thirds stress point. It estimates
completion-ordered slack for VRAM, network-transfer, and prefill constraints;
each class reports its tightest component without exposing destination names.
Thin step curves are the twelve cases, thick curves are policy medians, and a
cross marks the first time a case reaches at most 5% residual slack. The
deadline-blind trajectory likewise shows only its first 30 seconds.

```bash
uv run python queue-haul/plot_pooled_resource_slack.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-resource-slack-20260810/resource_slack
```

In the separate standard handoff experiment, each policy uses the same eight
pinned agentic sessions. An independent 80%-load stream serves on Sweden while
both destinations sustain 50% background inference.
The 30-second handoff clock includes live metrics, policy planning, and parallel
reconstruction, and hard-fails unless all sessions are admitted and complete.
Destination background service never pauses. At the traffic switch, destination
service starts immediately while source admissions stop and Sweden drains and
sleeps concurrently. Defaults retain five-minute pre/post windows and 100-ms
power sampling.

Handoff processes pin `kv_both`, 33 GB L1 pools, a 32 GB Redis cap, and disabled
vLLM prefix caching. Unrelated source and destination loads bypass LMCache so
they cannot evict migration state. The reducer shades migration, switch, and
source power-fall windows, writes phase queue depth, and requires Sweden to
start at or above 200 W and shed at least 50 W. The measured Sweden window
itself populates the migrating KV state; there is no separate warmup before it.

Only the Azure campaign uses
`profiles/gpt_oss_20b_a100_tp1_azure_300w.json`, calibrated on an Azure
NVIDIA A100 80GB PCIe whose `nvidia-smi` power limit is 300 W. The generic
`gpt_oss_20b_a100_tp1.json` retains the original non-Azure A100 calibration.
The Azure profile uses the cache-cold fixed-rate sweep at
`/datadrive/queue-haul-network/control/power-cal-300w-rate-001`: 18 rates from
0.25 to 20 requests/s, 20-second windows, a 1,100-word synthetic prompt body
with a unique leading hash, 64-token outputs, and zero cached prompt tokens. Its
conservative concave envelope reaches 300.24 W at `ell=10.0543`; the 14--20
requests/s deep-queue points independently remain near 299 W. The 98.1 W
model-resident idle anchor comes from
`power-cal-300w-002`; bare GPU idle is outside this curve. Reproduce and reduce
the sweep with the commands below. The runner hard-fails unless `nvidia-smi`
reports exactly one NVIDIA A100 80GB PCIe with a 300 W power limit.

```bash
uv run python power_rate_sweep.py --out PATH --window-s 20 --warmup-s 5 --workers 512 --rates 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 12 14 16 20
uv run python power_rate_sweep.py --out PATH --reduce-only --prefill-capacity-tps 1448.32 --decode-capacity-tps 1260.38 --idle-power-w 98.11623555 --curve-max-rate 12
```

The scalar Azure curve remains estimated evidence. The promotion campaign uses
`phase_power_calibration.py` to measure five prefill/decode mixtures at six load
levels and three repeats, fit `z = a f + b g`, and validate by holding out whole
mixtures. A v5 profile keeps destination service load (`f/F + g/G`) separate
from phase-aware power load and hard-fails outside the calibrated `(f,g)` hull.

```bash
uv run python phase_power_calibration.py prepare --out /datadrive/queue-haul-network/phase-power-v1
uv run python phase_power_calibration.py run --plan /datadrive/queue-haul-network/phase-power-v1/plan.json --profile profiles/gpt_oss_20b_a100_tp1_azure_300w.json --out /datadrive/queue-haul-network/phase-power-v1/run
uv run python phase_power_calibration.py fit --base-profile profiles/gpt_oss_20b_a100_tp1_azure_300w.json --measurements /datadrive/queue-haul-network/phase-power-v1/run/measurements.csv --idle-power-w 98.11623555 --out-profile profiles/gpt_oss_20b_a100_tp1_azure_300w_phase.json --summary outputs/azure-calibration/power-summary.json
uv run python testbed_calibration_campaign.py prepare --parent outputs/east-germany-separation-20260809/plan.json --out outputs/azure-calibration/testbed-plan.json
```

`evidence_catalog.py` writes an immutable-raw, checksum-bound sidecar at
`/datadrive/queue-haul-network/evidence-catalog.json`. Existing
`realized_shed_w` hardware-gap fields are cataloged as `model_credited`;
`trailing-power` separately derives direct five-second Sweden power windows.
The final `stress_frontier_campaign.py` plan contains 40 equal-weight states and
runs the Queue-Haul LP and its six baselines independently at 10--60 second
deadlines. Queue-Haul plans this suite with `lp_work_first`, guarded by its
MILP-free unrestricted greedy result after both are simulated to the deadline;
an LP-infeasible target skips impossible integral recovery, so this campaign
does not run a max-shed MILP.
Reduction carries each state's tighter-deadline attainment forward, uses the
fifth-smallest of 40 values, and keeps modeled-versus-empirical status in the
reduced JSON rather than a figure title.

The checked-in sensitivity uses the Azure-300W baseline-gated fixed-pack
profile and its 200 whole-repeat bootstrap curves. Baseline gating improves the
pack fit, but grouped-repeat validation still fails (12.76 W RMSE; 32.1% within
5 W), so this artifact remains modeled and is not an empirical promotion.

```bash
uv run python stress_frontier_campaign.py prepare --parent outputs/azure-compact-calibration-20260813/separation-calibrated.json --profile outputs/azure-compact-calibration-20260813/gpt_oss_20b_a100_tp1_azure_300w_pack_power_gated.json --out outputs/azure-compact-calibration-20260813/stress-pack-power-plan.json
for shard in 0 1 2 3; do uv run python stress_frontier_campaign.py run --plan outputs/azure-compact-calibration-20260813/stress-pack-power-plan.json --shard "$shard" --shards 4 --out "outputs/azure-compact-calibration-20260813/stress-pack-power-$shard.csv"; done
uv run python stress_frontier_campaign.py reduce --results outputs/azure-compact-calibration-20260813/stress-pack-power-{0,1,2,3}.csv --modeled-only --out outputs/azure-compact-calibration-20260813/frontier-pack-power.json
```

`outputs/network-campaign-20260805` retains the complete 54/54 East and West
single-link campaigns plus the successful `handoff-009` and bidirectional-cache
`handoff-010` three-node evidence, including raw 100-ms power, request,
transfer, decision, plot, and checksum artifacts. It also retains
`joint-queue-002-partial-086`: 86/126 completed joint scenarios, three raw stack
captures, and interrupted-attempt audit evidence. The first 17 scenarios used
consumer-only destination caches; the following 69 used bidirectional caches.
Reconstruction requests end with an explicit state-code probe and reserve 128
output tokens; a successful HTTP response without the code hard-fails the attempt.
The 600-second HTTP timeout is independent of the measured scenario deadline;
slow reconstruction completes with `deadline_met=false` instead of restarting.

`plot_network_action_breakdown.py` writes stacked bars for the completed joint
planner's Queue-Haul LP, Queue-Haul Greedy, KV-only, and replay-only recorded
destination-method selections. Run
`uv run python queue-haul/plot_network_action_breakdown.py`.

`plot_hardware_power_parity.py` audits the 840 matched Queue-Haul LP, Queue-Haul
Greedy, True Greedy, KV-only, replay-only, power-blind, and deadline-blind raw
traces. These runs do not provide the required settled pre-migration window:
the median gap from the final source request to migration is 0.075 s, and only
3/840 episodes cover a one-second window plus a one-second settling guard. The
script therefore hard-fails direct parity reduction instead of averaging
migration warm-up power. `--audit-only --out PATH` writes the episode-level
window audit without claiming measured shed. `--raw-delta --out PATH` writes an
explicitly `warmup_contaminated_immediate_pre` exploratory delta from the final
second before migration to the one-second post-switch window after a one-second
guard. A new hardware run with an explicit pre-migration hold is required for a
settled-state parity claim.

`power_parity_experiment.py` supplies that focused run without campaign controls:
50 matched random migration sizes per policy (350 runs total), a five-second
settled source window at load 0.4, and a five-second settled window at the
remaining source load. It reduces the direct GPU-0 means against the pinned
RAMR A100 power curve and writes the CSV plus y=x PNG/PDF.

```bash
uv run python queue-haul/power_parity_experiment.py prepare \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json \
  --out queue-haul/outputs/power-parity-random-plan
sbatch queue-haul/outputs/power-parity-random-plan/run.sbatch
```

The completed 350-scenario run is retained in scratch. The descriptive
phase-aware refit under `outputs/power-parity-phase-aware-20260813/` uses the
same observations for fitting and parity, so it is not held-out evidence. Its
shed regression has a 0.997 through-origin slope, effectively zero aggregate
bias, and 8.66 W RMSE; grouped five-fold episode cross-validation retains a
0.999 slope, 0.11 W bias, and 8.78 W RMSE. The CSV preserves all policy repeats.
The parity x-axis reports shed from the fitted
`P0 + delta_p * z / (1 + z)` model with `z = a*f + b*g`.
The publication plot shows Queue-Haul LP and Queue-Haul Greedy; its CSV retains
all seven measured policy arms.

Reduction runs automatically and can also be repeated without hardware:

```bash
uv run python queue-haul/network_campaign.py reduce \
  --plan /datadrive/queue-haul-network/control/plan.json \
  --run-root /datadrive/queue-haul-network/formal-001
sha256sum -c /datadrive/queue-haul-network/formal-001/artifacts.sha256
```

`summary.json` is valid only with all planned latest attempts complete. `results.csv`
contains status, deadline, migration time, API/log bytes, KV bytes, TCP RTT, and
retransmissions per scenario. Every stack retains 250-ms directional byte logs,
per-connection duration/bytes/RTT/RTT variance/congestion window/retransmits,
per-RESP-transfer payload and wire bytes, source and destination GPU power,
service logs, scheduled events, and loaded-sink request traces. Scenario results
retain streaming chunks, prompt/cached token counts, TTFT timestamps, source
sleep timestamps, state-code validation, and every attempt. Power scope is GPU,
not whole-node. These files become evidence only after the hardware run passes;
their presence in the implementation is not a measurement claim.

The normative formulation and executable-contract mapping are in:

- `formulation_nsdi.md`;
- `PROPOSED_DESTINATION_ARCH.md`.

## Main commands

Run commands from the parent `agent-migrate` directory:

```bash
uv run pytest

uv run python queue-haul/power_drain_experiment.py \
  --workload-profile queue-haul/profiles/agentic_tool_loop.json \
  --sessions 6 --seed 3 --power-limit 500 --deadline 5 --end 5 \
  --link-bytes-per-s 125000000 --intra-dc-bytes-per-s 12500000000 \
  --solver greedy --workers 2 --out queue-haul/outputs/profile_smoke

uv run python queue-haul/plot_simulator_validation.py
uv run python queue-haul/plot_simulator_evaluation.py
uv run python queue-haul/plot_scaling_results.py
uv run python queue-haul/plot_testbed_kv_timeline.py
uv run python queue-haul/plot_testbed_kv_timeline.py --method replay
uv run python queue-haul/mechanism_validation_campaign.py prepare \
  --out queue-haul/outputs/mechanism-validation-plan
# Run the matched current-stack KV/replay campaign on one 2xA100 node:
sbatch queue-haul/outputs/mechanism-validation-plan/run.sbatch
uv run python queue-haul/plot_migration_ttft_cdf.py
uv run python queue-haul/plot_fixed_contract_residuals.py
uv run python queue-haul/policy_hardware_campaign.py prepare \
  --out queue-haul/outputs/policy-hardware-width8-pilot-plan
QH_POLICY_RUN_ROOT=/scratch/users/$USER/qh-policy-run-width8-pilot \
  bash queue-haul/outputs/policy-hardware-width8-pilot-plan/run.sh
# Or submit the resumable two-A100 Slurm job:
sbatch queue-haul/outputs/policy-hardware-width8-pilot-plan/run.sbatch
uv run python queue-haul/migration_profiler.py make-crossover \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/policy-hardware-crossover-plan/plan.json \
  --context-sizes 2048,4096,8192,16384,24576,32768 \
  --bandwidth-mbps 1000,2500,5000,10000 --repeats 3 --seed 1
uv run python queue-haul/policy_hardware_campaign.py plot-reduced \
  --out queue-haul/outputs/policy-hardware-width8-frontier-20260730
uv run python queue-haul/canonical_simulator_campaign.py
uv run python queue-haul/greedy_lagrangian_experiment.py
uv run python queue-haul/simulated_pareto_campaign.py prepare
for shard in {0..63}; do
  uv run python queue-haul/simulated_pareto_campaign.py run-shard --shard "$shard"
done
uv run python queue-haul/simulated_pareto_campaign.py reduce
uv run python queue-haul/paper_evaluation.py \
  --out queue-haul/outputs/paper-evaluation
```

The exact 10K Pareto campaign includes Queue-Haul, greedy, isolated-fastest,
feasible-random, replay-only, and KV-only. The combinatorial
`greedy_lagrangian` recovery is excluded and remains a separate scale-limited
experiment. Trace-derived context anchors stop at the measured 31,562-token
prefill/decode boundary. `pareto-hero.png` shows one explicitly scoped example:
interactive-coding seed 1 at 10 Gb/s. Repeated identical frontier points are
collapsed, and the endpoint shared by all four frontier policies is labeled.

`requirement_frontier.py` computes destination requirements without constructing
a destination inventory. `pool_planner.py` compares those requirements with
concrete pool contracts and emits physical use/capacity rows. Pool admission
shares route capacity across methods but tracks replay and KV aggregate work
separately, matching their distinct measured throughput caps. `simulate.py`
independently schedules routes, reconstruction endpoints, requests, commits,
and power. It separately traces declared pool-service demand and debt from
realized replay and commit times. `evaluation_config.py` is the canonical
source for assumed paper operating points and their replacement evidence.
The default pool LP remains the Clarabel implementation. The experimental
`lp_highs` solver runs the same relaxation and rounder through SciPy/HiGHS;
resource rows are assembled directly from candidate nonzeros, and maximum-gain
fallback uses a scale-relative normalized feasibility margin. Its rounder uses
exact integral recovery only when its heuristic selection misses the target.
`lp_column_generation` is a Phase-I/Phase-II prototype with a reported
primal-dual certificate. It materializes the full candidate table and rebuilds
restricted masters, so it is a correctness reference rather than the
million-session implementation.
`lp_column_generation_persistent` keeps one native HiGHS master and basis while
adding session rows and priced columns, and reuses one column-oriented resource
matrix across insertion batches. Candidate construction uses compact
immutable records and reuses identical session physics across equivalent pool
type/route signatures while retaining pool-specific capacity rows.
`destination_bench.py --pool-counts`
splits fixed replica inventory across pools to vary alternatives without adding
hardware or route capacity. All column-generation solvers remain experimental;
`lp_column_generation_lazy` instead streams the complete implicit action
universe, retains only generated master columns, stops on a global certified
gap, and consults the oracle again during integral completion. It preserves the
same LP but regenerates Python candidate physics on every pricing sweep.
`lp_column_generation_native` runs the identical Phase-I/Phase-II master and
certificate with a long-lived Rust pricing oracle while keeping HiGHS in Python.
The native boundary factors candidates into float64 session/signature features,
packed per-session feasibility masks, and distinct pool/method sparse templates;
bounded chunks load directly into Rust-owned storage. It is limited to 16
options and hard-fails outside that scope. `uv sync` installs it for development;
rustup Cargo must precede any system Cargo, and the toolchain is pinned under
`native/`. Pools reuse admission physics only when type, route, replica count,
baseline, bounds, and methods are exactly equal; variables and capacity rows
remain pool-specific. Indexed replica placement and a compact execution verifier are still
required for million-session operation.
Static greedy uses a separate uncapped compact boundary: Python emits numeric
session/option features without constructing every `Candidate`, Rust computes
fixed scarcity prices and runs the dependent feasibility scans, and Python
materializes only selected moves. A successful primary scan is unchanged; on a
miss, footprint-first, gain-first, and dynamic least-peak option scans compete
by achieved gain. Packing or deadline cuts expand the exhaustive table only on
that repair path. `PlanResult` reports candidate generation,
selection, MILP recovery, packing, and validation time separately; greedy's
MILP recovery time is always zero.
`outputs/native-lp-scale-20260801/one-million.json` records the post-optimization
one-seed 1M-session LP/rounding sensitivity and its hashes; it explicitly excludes
replica packing, DES, prediction, and execution validation.
`greedy_lagrangian_experiment.py` compares the two supported greedies on paired
trace-derived targets; infeasible plans receive zero validated shed. Static
`greedy` fixes one scarcity price; its extra orders run only after the primary
order misses its target.
`greedy_lagrangian` iterates aggregate-resource prices, retains a bounded set of
exact nonlinear source prefixes, performs target-capped recovery, and packs the
final set. Recovery caches sparse candidate columns and accumulates each retained
prefix without constructing sparse submatrices, and reuses those statistics in
packing fallback. Equal normal/emergency pool bounds reuse one admission solve.
Feasible-random groups each session's candidates in one pass, so its
setup is linear in candidate count. Immutable single-policy scale runs under `outputs/dual-lagrangian-*`
record the former `greedy_prefix` name; mixed bundles containing retired
optimizers were removed rather than rewritten.
Replica-packing repair considers actions from best to worst normalized resource
per watt, keeps feasible placements fixed, and rejects actions that cannot fit.
It reports rejected-action count and complete repair-pass time.
The campaign also computes a fractional source-chord LP lower bound on migration
work and plots each feasible greedy's excess work over that bound. The bound is
valid for the concave GPU-scoped power profile and relaxes integrality, replica
packing, and exact timing; it is not the unknown integral optimum.
The plot reports completion over every case and compares work only on the paired
common-feasible cohort to avoid survivor bias.
`paper_evaluation.py` writes the legacy Q1–Q9 result/plot registry while the
paper evaluation is reorganized into mechanism validation, fixed-contract
coordination, multi-pool contracts, and planner quality/scale. It rejects
tables with missing provenance.
`mechanism_validation_campaign.py` replaces the mechanism Gantts with matched
current-stack 28K-context, 10-Gb/s, concurrency-one KV and replay measurements.
It runs five repetitions of the same four-turn source workload and plots the
trace nearest each mechanism's median request-boundary wait.
Timestamped arrivals execute independently of migration. Quiescence drains only
requests admitted before the pause; later arrivals wait for the destination
route. The committed `fix1` charts predate this correction and must be replaced
by the fresh `quiesce` run before using their gray intervals as drain evidence.
Warm prefetch permits concurrent destination L1 fills but requires a full L1
hit before inference.
Source-store completion uses unique RESP keys, not LMCache summary token counts.
Its expected key count comes from the request's retained-cache miss.
Traces with no catch-up plot a zero-length catch-up at the measured idle boundary.
Destination inference forbids L2 reads but may recompute an unstorable boundary chunk.
`plot_fixed_contract_residuals.py` caches the canonical 100K-session,
120-second fixed-contract requirement sweep and compares mixed greedy,
GPU-work-first, replay-only, and KV-only resource headroom at common requested
source-power shed levels. Use `--refresh` when its pinned inputs change.
`policy_hardware_campaign.py` creates a resumable paired idle-session campaign
that launches every session concurrently for Queue-Haul, both greedies,
KV-only, and replay-only. Static greedy retains its established hardware
planning path. Lagrangian greedy uses a one-pool idle dedicated-sink adapter,
then emits the same frozen move schema. Its default truncated grid uses coding,
interactive-coding, and
agentic-tool-loop context-length profiles; uniform-over-support and
uniform-over-range token distributions, or named exact context packs; configurable
bandwidths and deadlines; and full-episode migration width. The requirement and
cell bandwidth are passed to the Queue-Haul planner. The runner remains eager and
does not execute planner pacing or measure
planning latency. A policy-infeasible deadline retains its admitted prefix and
appends an explicitly marked independently-fastest tail so runtime width remains
the episode size. Failed episodes remain in denominators. Reduction writes
timing CDFs, including a 30-second full-target attainment CDF whose event time
includes the trailing five-second power window; missing mass is deadline
failure. The same 30-second cohort also produces standalone bandwidth plots for
episode attainment and Queue-Haul's deadline-admitted replay/KV action mix.
Power attainment is trailing-five-second average modeled source-power
shed divided by the 100% source-power target. MP
runs require bounded RESP quiescence between scenarios so late cache writes
cannot cross scenario boundaries. Run from a clean committed checkout with two
A100 80GB GPUs. The pinned frontier plan under
`outputs/policy-hardware-width8-frontier-plan/` uses seed 1, three episodes per
cell, eight sessions and moves, 5/10 Gbit/s, and 19/30-second requirements for
360 scenarios. Its fresh default root is
`/scratch/users/$USER/qh-policy-run-width8-frontier`; 1 Gbit/s is excluded.
Its completed checksum-pinned reduced bundle is retained under
`outputs/policy-hardware-width8-frontier-20260730/`, including move-admission
provenance and raw GPU samples. `plot-reduced` writes a pooled
migration-to-destination-first-token CDF, CDFs of the slowest session in each
complete episode before and after normalization by watts shed, and median
modeled source-power shed over
elapsed time with an interquartile band, plus paired attainment–completion
points and a CDF of measured session downtime per modeled watt shed. This idle
evidence also includes an episode migration-makespan-per-modeled-watt CDF and
supports timing and projected, not realized, power attainment.
`plot-reduced --pooled-with` adds supplied reduced campaigns to every pooled,
bandwidth, condition, attainment, power, and Pareto plot.
The pinned 2026-07-30 bundles predate `greedy_lagrangian`; they do not constitute
hardware evidence for it. A new two-A100 run is required for that claim.
Matched reruns of either reduced bundle use its frozen plan as the cohort source.
The three network baselines use resource-aware per-session fastest, uniform-gain
power-blind LP, and a 600-second deadline-blind planning horizon while retaining
the source plan's 19/30-second scoring deadlines:

```bash
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-frontier-20260730/plan.json \
  --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_20260730.json \
  --out queue-haul/outputs/policy-hardware-width8-frontier-network-baselines-plan
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --out queue-haul/outputs/policy-hardware-width8-packing-network-baselines-plan
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --policies queue_haul greedy isolated_fastest queue_haul_power_blind queue_haul_deadline_blind \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-plan
for shard in 0 1; do uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --policies queue_haul greedy isolated_fastest queue_haul_power_blind queue_haul_deadline_blind \
  --condition-shard "$shard" 2 \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-shard${shard}-plan; done
```

The 720-scenario contemporaneous packing rerun completed as RAMR jobs 38607705
and 38607709. Both 360-scenario condition shards validate independently; their
checksum-pinned reduced evidence is under
`outputs/policy-hardware-width8-packing-contemporaneous-20260811/`, with pooled
graphs at the top level and shard-1 provenance nested under `shard1/`. The
pooled migration CDF compares Queue-Haul, greedy, isolated-fastest, power-blind,
and deadline-blind on this single contemporaneous cohort. Rebuild the pooled
and per-condition graphs with:

```bash
uv run python queue-haul/policy_hardware_campaign.py plot-reduced \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-20260811 \
  --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_crossover.json \
  --pooled-with queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-20260811/shard1 \
  --cdf-policies queue_haul isolated_fastest queue_haul_power_blind queue_haul_deadline_blind
```

The completed reduced baselines are checksum-pinned under each 2026-07-30
bundle in `network-baselines-20260811/`; the parent graphs pool both campaigns.
The packing bundle's 30-second full-attainment CDF pools packing and frontier
parent and baseline episodes. It uses an 8-by-4-inch plot, 17-point axis text,
Okabe–Ito colors, distinct line styles, and a collision-free legend inside the
lower-right axes. The deadline is labeled vertically in italics on its line.

The canonical output style is `plot_style.py`: 8-by-5 inches, 15-point titles,
labels, and ticks, 11-point legends and annotations, 3-point lines, and 220 DPI.
Plot-specific layouts may use the shared compact size. New and modified plot
producers must inherit it. Policy identities are:

`plot_workload_policy_attainment.py` pools all eight constraint states and all
paired workload, timing, and power draws into one modeled time-to-target CDF.
It compares Queue-Haul with greedy and fixed-action policies, includes the
trailing power window, and retains misses as missing CDF mass. Every policy
appends the same relaxed-horizon
independent-fastest tail for unadmitted sessions after the scoring deadline,
analogous to the hardware campaign tail. Queue-Haul uses the target-aware
HiGHS LP with integral target recovery. Queue-Haul Greedy uses only
deterministic fixed-price scans and reports a miss if none reaches the target.
Regenerate it with
`uv run python plot_workload_policy_attainment.py`.

| Internal name | Display name | Okabe–Ito | Line |
|---|---|---:|---|
| `queue_haul` | Queue-Haul LP | `#0072B2` | solid |
| `greedy` | Queue-Haul Greedy | `#E69F00` | dashed |
| `greedy_lagrangian` | Queue-Haul Lagrangian Greedy | `#F0E442` | dash-dot-dot |
| `isolated_fastest` | Isolated Fastest | `#D55E00` | long dash |
| `kv_only` | KV Migrate Only | `#56B4E9` | dash-dot |
| `replay_only` | Replay Context Only | `#CC79A7` | dotted |
| `queue_haul_power_blind` | Queue-Haul Power Blind | `#009E73` | short dash |
| `queue_haul_deadline_blind` | Queue-Haul Deadline Blind | `#000000` | fine dotted |

`isolated_fastest` picks each session's fastest method in isolation but may
route to any destination offering it and is displayed as "True Greedy" in the
stress frontier. The older destination-locked variant is `isolated_myopic`
(displayed as "Myopic fastest (method+route)"), which the network campaign's
separation cells keep as their deliberately weak baseline. Older pinned
outputs may also use "True Greedy" for that destination-locked behavior; their
policy IDs distinguish the two.

Stress-frontier figures omit the modeled MILP reference and normalize every
displayed policy to the shared maximum 90%-coverage shed. The y-axis reports
this quantity as normalized power shed; every curve carries an unlabelled 95%
regime-stratified trajectory-bootstrap confidence ribbon.

Rebuild them with:

```bash
uv run python queue-haul/policy_hardware_campaign.py plot-reduced --out queue-haul/outputs/policy-hardware-width8-frontier-20260730 --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_20260730.json --pooled-with queue-haul/outputs/policy-hardware-width8-frontier-20260730/network-baselines-20260811
uv run python queue-haul/policy_hardware_campaign.py plot-reduced --out queue-haul/outputs/policy-hardware-width8-packing-20260730 --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_crossover.json --pooled-with queue-haul/outputs/policy-hardware-width8-packing-20260730/network-baselines-20260811 queue-haul/outputs/policy-hardware-width8-frontier-20260730 queue-haul/outputs/policy-hardware-width8-frontier-20260730/network-baselines-20260811
```

capacity_sweep_campaign.py keeps the completed two-point load run as a
pilot and builds the publication load curve at
0,.25,.50,.65,.75,.80,.85,.875,.90,.925,.95,.975. Normalized offered load is
scheduled prefill GPU-seconds plus decode GPU-seconds per second, using the
checksum-pinned independent destination calibration; it is not achieved
throughput or RPS. Each load has ten randomized, paired repeats with the same
pre-generated deterministic trace for Queue-Haul LP, static greedy,
replay-only, and KV-only. All eight source sessions are offered to each policy,
the destination warms for 30 seconds before migration, and arrivals continue
through the full 30-second deadline. A session earns shed credit only when its
route commits and first destination continuation token both arrive by the
deadline. The live figure uses trace-derived load, median shed, bootstrap 95%
confidence bands, and the 147.2 W requested-shed line; LP and Greedy component
panels report replay, KV, and unmet watts. After the dense base run, add
the midpoint of every adjacent pair where any policy's measured median shed
changes by more than 5 W. The empirical LP knee is the last base load whose
median reaches requested shed (within 1e-6 W) and the following load; failure
to bracket it is a hard error. Phase 2a runs repeats 10--19 for both knee loads
and every base load where any policy's 95% interval is wider than 5 W, plus
repeats 0--9 for the midpoints. Phase 2b runs repeats 20--29 only where the
combined base and phase-2a interval remains wider than 5 W. Every selected
load/repeat remains a four-policy common-trace cell with all eight source
sessions. Plans, hardware runs, and the final merged bundle use separate roots;
plans record the exact repeat map, selection reasons, and prior-plan hashes.
All 480 base episodes are checkpointed independently and completed scenario IDs
are skipped on resume.

```
uv run python queue-haul/capacity_sweep_campaign.py load --out queue-haul/outputs/capacity-load-publication-20260807 --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807
uv run python queue-haul/capacity_sweep_campaign.py load --adaptive-stage phase2a --prior-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 --out queue-haul/outputs/capacity-load-publication-20260807-phase2a --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a
uv run python queue-haul/capacity_sweep_campaign.py load --adaptive-stage phase2b --prior-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a --out queue-haul/outputs/capacity-load-publication-20260807-phase2b --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2b
uv run python queue-haul/capacity_sweep_campaign.py load --merge-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2b --out queue-haul/outputs/capacity-load-publication-20260807-final
uv run python queue-haul/capacity_sweep_campaign.py goodput --out queue-haul/outputs/capacity-goodput-20260806 --calibration goodput-calibration.json --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json
```

If phase 2b selects no cells, omit its run root from the merge command. Base and
phase plan roots keep `plan.json`, `modeled_capacity.csv`, `live_plan.json`, and
`summary.json`; hardware and final roots add `live_capacity.csv`,
`live_summary.json`, and the live PNG/PDF figures.

`plot_capacity_operating_curves.py` compares the completed 2xA100 load campaign,
balanced full-drain bandwidth blocks 0--1, and width-8 bandwidth scheduler
campaign. It writes scheduler-colored time-to-full-power and deadline-shed
curves plus Queue-Haul replay/KV/not-moved action shares. Full-drain bandwidth
figures split by destination load; width-8 figures split by context profile and
19/30-second campaign. The primary width-8 ECDFs pool the balanced context
profiles at equal frequency within each bandwidth panel (15 episodes per
scheduler); the faceted curves and CSV retain the context-specific results.
The combined action heatmap shows the observed L-shaped 30-second design:
zero-prefill bandwidth cells and positive-prefill 10-Gbit/s load cells, with
unmeasured combinations left gray rather than interpolated. Prefill throughput
uses the load campaign's pinned service-rate calibration.
Coincident scheduler curves are horizontally offset slightly for visibility;
the CSV retains exact operating points. Run
`uv run python queue-haul/plot_capacity_operating_curves.py`; outputs go to
`queue-haul/outputs/capacity-operating-curves-20260808/`.

The forced-full-drain appendix crosses
.85,.875,.8875,.90,.9125,.925,.9375,.95,.9625,.975 offered load with
1, 2.5, 5, and 10 Gbit/s. Five repeat-block shards each contain two repeats at
all four bandwidths (160 episodes), preventing bandwidth from being confounded
with allocation time. Bandwidth block order rotates from a deterministic random
base, and each block randomizes its ten loads and two policies. Arrival traces
are identical across policy and bandwidth for a load/repeat. Every episode
attempts all eight sessions, credits shed at 30 seconds, records the last route
commit and continuation token, and keeps arrivals active until the drain
completes, up to the 180-second timeout. Normalized load still uses only the
30-second measurement window.

The retained 2026-08-08 hardware evidence includes the complete standalone
10-Gbit/s appendix (200 episodes and all 1,600 sessions credited by 30 seconds)
and balanced blocks 0--1 (320 episodes and four repeats per exact
load/bandwidth/policy cell). Across the balanced blocks, all 40 replay-only
episodes at 1 Gbit/s drain in 32.919--33.501 seconds and miss the all-session
deadline, while all 40 KV-only episodes drain in 27.256--28.267 seconds and
meet it. Every episode at 2.5, 5, and 10 Gbit/s meets the deadline. These two
blocks support the qualitative bandwidth boundary but not a ten-repeat
confidence claim; they are retained separately rather than presented as the
five-block merged result.

Run blocks 0 through 4 from the same commit, one per allocation, then merge the
five roots:

```
uv run python queue-haul/capacity_sweep_campaign.py full-drain --repeat-block 0 --out queue-haul/outputs/capacity-full-drain-block0 --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-full-drain-block0
uv run python queue-haul/capacity_sweep_campaign.py full-drain --merge-run-root /scratch/users/$USER/qh-capacity-full-drain-block0 /scratch/users/$USER/qh-capacity-full-drain-block1 /scratch/users/$USER/qh-capacity-full-drain-block2 /scratch/users/$USER/qh-capacity-full-drain-block3 /scratch/users/$USER/qh-capacity-full-drain-block4 --out queue-haul/outputs/capacity-full-drain-final
```

Replace `0` in the first command with each block number. Reusing its plan and
run root resumes a block only when the commit and run metadata still match; an
audited code change additionally requires `--resume-from-git-sha OLD_SHA`.
`--bandwidth-mbps 10000` remains available for the standalone ten-repeat knee
appendix. Each shard writes `full_drain_capacity.csv` and its PNG/PDF figure;
the merge accepts exactly five complete blocks, checks profile, calibration,
manifest, context, and trace provenance, and emits the 800-row final result.
The separate live power-drain evidence in
`outputs/power_drain_live_20260714/` includes planned and measured source-power
reductions. The contemporaneous packing traces retain raw `power.csv`,
`result.json`, and plans, but their warm-up-to-migration timing is insufficient
for the same direct measurement.
`outputs/live-power-shed/` retains the 2026-08-06 two-A100 seamless full-shed
run. The Queue-Haul LP arm moved all eight sessions under continuous 4 rps
source and 1 rps destination agentic load with `kv_both` and 33 GB L1 pools;
there is no pre-migration flush, pause, or drain. After 300 s with both sites
serving, the parallel migration took 4.656 s, the traffic switch took 70 us,
and the source drained naturally in 3.002 s before a stable five-minute
destination hold. Mean source power fell from 255.8 W to 85.8 W while mean
destination power rose from 128.2 W to 283.4 W. The three replay moves
recomputed 10,662 tokens; the five KV moves computed 713 tokens while reusing
2.72 GB of KV. The plot shows the 300–400 s handoff region as fixed 500 ms mean
power in a compact, outlined view. The retained paired 10 Hz samples have 95.6%
coverage and the bundle includes phase-level engine queue depth and
checksum-pinned raw data.
`outputs/live-power-handoff-east-germany-20260807/` retains the three-A100
Sweden-to-East-US-2/West-Germany campaign with 80% source and 50% load at each
destination. Queue Haul admitted all eight sessions in 29.669 s and KV-only in
25.159 s; replay-only admitted six before the fixed 30-second deadline. The
bundle includes raw `power.csv`, load and transfer telemetry, and separate 500
ms mean regional-power plots cropped from session-state preparation through GPU
sleep. Each trace reports percent of the 300 W per-GPU TDP and marks Migration,
Switch, Barrier, and Sleep. The bundle also retains the exact plan and composed
non-formal calibration used.
`migration_profiler.py make-crossover` creates paired single-session replay/KV
measurements for each nominal context, bandwidth, and repeat. The synthetic body
reserves 192 tokens for message overhead, and the first 32K replay is a fail-fast
model-limit smoke. Bandwidths remain contiguous to avoid unnecessary MP-stack
restarts. Its 6-context, 4-bandwidth, 3-repeat grid contains
144 migrations. Use the reduced measurements to establish the method crossover
before freezing the 2K–16K width-8 packing plan; the legacy replay profile
starts at 3,473 tokens and must not be extrapolated to 2K. The completed 144/144
crossover bundle is checksum-pinned under
`outputs/policy-hardware-crossover-20260730/`. Replay is faster across the tested
range at 1/2.5 Gbit/s, through 8K at 5 Gbit/s, and through 4K at 10 Gbit/s; the
8K 10-Gbit/s cell is effectively tied. The derived profile
`profiles/gpt_oss_20b_a100_tp1_crossover.json` replaces serial replay rate,
replay/KV completion, KV ingestion lower bound, and route-switch timing while
preserving the existing catch-up, power, and capacity evidence. `--policies isolated_fastest` enables the per-session-fastest policy without
changing the default policy set. The pinned
`outputs/policy-hardware-width8-packing-plan/` runs three paired width-8 episodes
for Tiny, Small, Medium, Mixed, and Large packs at 1/2.5/5/10 Gbit/s and 19/30-s
requirements: 600 scenarios in total. Job 36822272 completed all 600 scenarios
without failures in 8:08:22. Its checksum-pinned reduced bundle is under
`outputs/policy-hardware-width8-packing-20260730/`; compressed results, GPU
samples, proxy byte counters, and RESP transfer records retain the raw evidence
without runtime debug logs. The matched 120-episode, 240-scenario baseline plan is pinned under
`outputs/policy-hardware-width8-isolated-fastest-plan/`. Its bandwidth-grouped
order has four bandwidth blocks instead of 93 bandwidth runs; the 30-scenario
hygiene rotation yields eight model-stack starts. Its launcher reduces all
episodes, hard-validates three complete repetitions per condition and clean
plan/profile provenance, writes checksums, and rebuilds the comparison from the
common packing cohort only (120 observations per method). The comparison uses
±5 percentage points relative to Queue-Haul LP for better/similar/worse and
shows every sample count; frontier and partial-network rows are excluded. The
completed allocation 37874352 produced 240/240 scenarios with no failures and
120 validated attainment rows across 40 conditions with three repetitions each.
Its reduced, checksum-pinned bundle is under
`outputs/policy-hardware-width8-isolated-fastest-20260806/`, including the
common-cohort chart and compressed result, power, proxy-byte, and RESP-transfer
evidence.
Its bandwidth-faceted destination-TTFT CDF pools
all five workloads, both deadlines, and three episodes within each bandwidth;
the companion pooled CDF combines all four bandwidths. Regenerate that CDF
with the earlier trace-sampled 5/10-Gbit/s frontier rows included at raw-sample
weight using `plot-reduced --out <packing-results> --pooled-with
<frontier-results>`; this pooling also applies to the maximum-session,
maximum-session-per-watt, and existing per-watt CDFs.
`workload_adaptation_campaign.py` resamples the measured coding templates and
paired timing/power calibrations across the eight HBM, bandwidth, and
destination-compute states. Its phase-aware planner is scoped to one awake
source: it converts requested watts exactly into additive removed phase load,
hard-fails multiple-source phase topologies, and verifies nonlinear source
watts after packing. The fixed 0.4 workload normalization is service load
`sum(f/F + g/G)`, not sampled phase load `af + bg`. The action and
`workload_power_frontier.py` outputs report steady source-region power only;
destination power and net fleet energy are outside their claim. The bandwidth
state caps both physical destination routes at 1 Gbit/s while retaining the
region-specific controlled effective-pipeline fits. Every single-factor state
must activate in at least 90% of paired draws and change at least 10% of paired
plans.
`simulated_pareto_campaign.py` creates 64 deterministic shards for 14 exact
10K-session idle snapshots: three trace seeds for each workload and five
trace-derived context anchors. It compares Queue-Haul, static and Lagrangian
greedy, isolated-fastest, feasible-random, replay-only, and KV-only over fixed
1/2.5/5/10-Gbit/s site links and 30-second through four-hour deadlines. Replay
is a divisible destination-fleet service fitted from successful width-8 10G
episodes; width 8 is calibration evidence, not an execution cap. KV bytes use
both fixed WAN and per-replica ingest capacity. All methods use identical idle
destination packing and report simulated sensitivity, trailing-five-second
target attainment, last commit, censoring, source hashes, and dirty Git state.
Reduction hard-fails missing or duplicate shards and emits separate trace and
anchor small multiples. Conservative and optimistic fits run on sentinel cells;
the full grid uses the central fit. The pinned stage-span evidence fits
p25/median/p75 replay capacity factors of 0.963/0.984/1.026. These are used
without clamping: the lower two represent a small effective slowdown, while
width 8 remains only the upper validity bound.
`canonical_simulator_campaign.py` runs a four-target paired 10K-session
Queue-Haul, both greedies, per-session-fastest, replay-only, and KV-only comparison
under one assumed dedicated-pool contract. Its compact 10K/100K/1M scale check
uses both greedies, an equivalent pooled-destination topology,
10 Gbps per 10K sessions, and summary-only prediction. Sampled future requests
are disabled; measured two-A100 results provide continuation first-token
evidence.

Software results are authoritative for optimizer intent, aggregate feasibility,
and fleet-scale sensitivities. Hardware results are authoritative for measured
execution and timing of the frozen plan. A mismatch fails validation; neither
result silently overwrites the other.
Override `QH_APPTAINER_IMAGE` if the pinned LMCache image is not at the default
scratch path; set `QH_RESUME_FROM_GIT_SHA` when resuming after a code change.
MP catch-up completeness uses exact rendered-token prefix chunks; observed key
sets may also contain a generated-tail chunk.
Stacked MP runs retain per-scenario byte, connection, and RESP traces.
Re-submit after a time limit; the stable run root reuses its original port
offset and completed scenarios. Set `QH_RESUME_FROM_GIT_SHA` after code changes.

## Planner selection latency

`planner_latency.py` times the selection step alone against fleet size, for the
pure LP and the pure greedy, and writes `outputs/planner-latency/`.

```bash
uv run python planner_latency.py                      # 28 .. 100k sessions
uv run python planner_latency.py --sessions 28 50000  # one point at each end
```

A whole `plan` call includes candidate physics, packing, and execution-model
validation. On the 100,000-session/400,000-candidate latency case, native greedy
selection takes about 0.13 s versus 8.90 s for LP, while production compact
candidate generation and selected-move materialization still take about 9.5 s.
Both solvers are held to their pure form: greedy has no integral recovery, and
the target is one both can attain, so the LP runs its target-first solve instead
of its max-shed fallback. A selection that
misses the target hard-fails rather than reporting a fallback's timing, which is
what makes `outputs/scaling_1_to_100k_20260720_greedy` unusable for this
comparison: past 32 sessions its 50% request is out of reach.

`planner_scaling_campaign.py` is the apples-to-apples production-front-end
comparison. It gives LP and greedy the same deterministic fleet and attainable
25% removable-power target, times candidate generation plus selection in fresh
processes, disables LP integral recovery, and excludes common fleet setup,
packing, and DES. Three repeats run through 100K sessions and one thereafter.

```bash
uv run python planner_scaling_campaign.py run
```

The pinned 24 GiB campaign in `outputs/planner-scaling-greedy-vs-lp/` records
both production-front-end and pure-selection time; its figure plots selection
only. Greedy/LP selection medians are 1.25/9.05 s at 100K, 13.27/87.08 s at 1M,
and 26.97/326.38 s at 2M. Greedy selection remains nearly linear at 70.14 s for
5M and 142.14 s for 10M. LP's combined candidate-generation and selection arm
reaches the declared 1,800 s timeout at both larger sizes, so no selection-only
LP value is plotted there. No monitored 24 GiB breach was classified. The figure
marks those combined-path timeouts in a top outcome strip, not at a selection-time
y-value. At 10M, greedy also spends 936.30 s generating compact inputs, which
remains the dominant optimization target outside the plotted selection step.

`planner_quality.py` pairs full greedy and shipped LP plans over workload,
fleet-size, deadline, destination-scarcity, seed, and target grids. It checks
exact modeled power, packing, and deadlines, and separately solves an exact
binary oracle for the aggregate candidate model. The oracle is not an
integrated replica-packing or nonlinear-power optimum.

```bash
uv run python planner_quality.py --out outputs/planner-quality
uv run python planner_quality.py --workloads agentic_tool_loop coding \
  --target-basis removable --out outputs/planner-quality-operational
```

## Measurement programs

The focused GPT-OSS A100/H100 SLO curve, request-block error bars, standard
request-level TPOT, pinned runtime, and exact site commands are in
`AGENTIC_RPS_SWEEP.md`. It is separate from the legacy RPS plan below.

`h100_serving_campaign.py` coordinates optimized-H100 calibration for the
pinned Qwen3.8-27B and Gemma-4-26B checkpoints, with GPT-OSS-20B accepted as an
apples-to-apples prefill reference. Prefill, RPS/SLO, and power
evidence all require native BF16 TP1 execution with compilation and CUDA graphs;
the campaign hard-fails eager fallback or runtime drift. It retains raw evidence
and reduced CSV/JSON only, without constructing a migration profile. The power
fit exports its explicit saturating simulation envelope through `ell=16`, beyond
the prior GPT-OSS campaign's measured overload extent of about 12.57.
Matched prefill runs fix `max_num_batched_tokens=8192` for every model; older
Qwen evidence collected with its 1,567-token architecture-campaign limit is not
hardware-comparable. They omit the unused LMCache connector because Qwen's
hybrid-state connector requires one 784-token cache block per scheduler step.
Set `QH_NATIVE_RUNTIME_VERSIONS=vllm,lmcache` only with an isolated native
environment when a model requires a separately pinned runtime.
Pass `prepare --hardware a100` to collect the identical optimized prefill grid
on A100 without changing the default H100 plan.

```bash
uv run python h100_serving_campaign.py prepare --out runs/h100-serving/plan.json
QH_RUNTIME=native QH_LMCACHE_MODE=mp uv run python h100_serving_campaign.py run-prefill --plan runs/h100-serving/plan.json --model MODEL --root /datadrive/h100-serving/MODEL/prefill
QH_RUNTIME=native QH_LMCACHE_MODE=mp uv run python agentic_rps_sweep_campaign.py run --plan runs/h100-serving/rps-plan.json --model MODEL --run-root /datadrive/h100-serving/rps
uv run python power_model_campaign.py --hardware h100 --model MODEL --out /datadrive/h100-serving/MODEL/power
```

Run `reduce-prefill`, the agentic sweep's `reduce`, then `validate` to emit the
final alignment record. Gemma uses 2 s TTFT/0.2 s TPOT; Qwen uses twice its
0.125-RPS baseline. The older single-repeat H100 curves are reference-only.
The same power program accepts `--hardware a100`, hard-gates the Azure 300 W
A100 identity, and uses runtime-specific model overrides. Managed phase-power
launches likewise accept either `a100` or `h100`.

If the original rational A100 fit completed but failed its frozen holdout gates,
collect a prospective replication extension instead of refitting or relabeling
that evidence:

```bash
uv run python power_model_campaign.py --hardware a100 --model MODEL \
  --replication-base /datadrive/ORIGINAL-FAILED-RUN \
  --out /datadrive/NEW-REPLICATION-RUN
```

The extension pins hashes of the original metadata, cells, and fit; requires the
same checkpoint revision and physical GPU; and acquires all training data before
the new holdout. Training adds two complete repeats of the original 45-cell grid
and three idle anchors. The untouched prospective holdout contains 18 newly
measured active cells plus six interspersed idle anchors, so its R2 covers the
full idle-to-saturation envelope. The report also retains an active-only
diagnostic, excludes the original failed confirmation cells from validation,
and applies the existing MAE, p90, R2, family, coefficient-stability,
repeatability, and zero-cache gates without changing their thresholds.

For a profile-free two-region mechanism check, `migration-timing` starts the
pinned model on one source A100 and one destination A100, then measures both KV
transfer and replay at exact token counts. It requires a formal calibration for
that two-node cluster, validates matching host/runtime identities, checks the
model's cache-block alignment and exact token timestamps on every repetition,
and finishes with a source sleep/wake health gate. This timing-only path pins
vLLM's stream interval to one and disables asynchronous scheduling on both
engines, because asynchronous engine updates may coalesce multiple generated
tokens into one SSE event and therefore cannot provide literal per-token
arrival timestamps. Other architecture, power, and throughput campaigns retain
their normal scheduler configuration.

```bash
QH_RUNTIME=native QH_LMCACHE_MODE=mp \
QH_NATIVE_RUNTIME_VERSIONS=0.24.0,0.5.1 \
uv run python network_campaign.py migration-timing \
  --cluster azure_network_cluster_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-germany-001.json \
  --model Qwen/Qwen3.8-27B --run-root /datadrive/qwen-regional-timing
```

Use `phase_power_calibration.py` for simulation power. Its open-loop five-ray
grid, grouped holdouts, and bootstrap fit distinguish offered service load from
the saturated-kernel envelope measured by `power_model_campaign.py`. H100 runs
name the pinned model and same-stack prefill/decode capacities; resumable run
metadata freezes those inputs.
With `--vllm`, the runner owns the optimized server lifecycle and additionally
pins its command, git revision, GPU UUID, and same-launch resident-idle anchors.
Each power window must retain at least seven synchronized samples per second.
Decode-only cells pace 512-token requests so load changes offered work rather
than saturated batch occupancy.
`run-suite` consumes an ordered target JSON and writes a durable completion
marker only after each optimized runtime validates. The supplied user unit
restarts failures and resumes the suite after host reboots.

`matched_power_fit.py` freezes non-monotone, model-specific H100 power curves
for the matched East/Germany coding-session load path. It hard-gates repeat
holdouts and carries 200 resampled measured curves into action simulations;
identical bootstrap curves are stored once with an explicit weight.

```bash
uv run python phase_power_calibration.py prepare --out runs/h100-phase-power
uv run python phase_power_calibration.py run --plan runs/h100-phase-power/plan.json --model MODEL --hardware h100 --prefill-tps F --decode-tps G --vllm VLLM --out /datadrive/h100-phase-power/MODEL
systemctl --user enable --now "$(pwd)/queue-haul-phase-power.service"
```

- `power_window_sensitivity.py` and `power_profile_reduce.py`: source power.
- `migration_profiler.py` and `migration_profile_fit.py`: replay/KV handoff.
- `destination_campaign.py` and `destination_runner.py`: targeted destination
  service and loaded-migration evidence.
- `service_headroom_campaign.py`: exact-stack A100/H100 incumbent-latency curves
  against Queue-Haul's offered normalized prefill/decode service work.
- `fixed_shape_slo_campaign.py`: H100 fixed-shape RPS curves. Its TPOT column is
  `p90_tpot_s`, pooled over every exact post-first-token interval in each cell.
  Launches leave CUDA architecture selection to the visible GPU.
- `service_surface_runner.py` and `service_profile_reduce.py`: isolated and
  mixed service profiles.

Prepare the frozen cell matrix, follow its per-hardware run order, reduce one
exact-stack normalization per hardware, then run and reduce the discovery cells:

```bash
uv run python service_headroom_campaign.py prepare --out runs/service-headroom/plan.json
uv run python service_headroom_campaign.py run-cell --plan runs/service-headroom/plan.json --cell-id CELL --out /datadrive/service-headroom
uv run python service_headroom_campaign.py reduce-calibration --plan runs/service-headroom/plan.json --hardware a100 --runs /datadrive/service-headroom --out /datadrive/service-headroom/a100-normalization.json
uv run python service_headroom_campaign.py run-cell --plan runs/service-headroom/plan.json --cell-id CELL --normalization /datadrive/service-headroom/a100-normalization.json --out /datadrive/service-headroom
uv run python service_headroom_campaign.py reduce --plan runs/service-headroom/plan.json --hardware a100 --runs /datadrive/service-headroom --ttft-target-s 1 --tpot-target-s .1 --out /datadrive/service-headroom/a100-scout.json
```

The discovery result cannot update P1/P2. Generate and execute its unseen
confirmation plan, then reduce it:

```bash
uv run python service_headroom_campaign.py prepare-confirmation --plan runs/service-headroom/plan.json --scout /datadrive/service-headroom/a100-scout.json --hardware a100 --out runs/service-headroom/a100-confirmation.json
uv run python service_headroom_campaign.py run-cell --plan runs/service-headroom/a100-confirmation.json --cell-id CELL --normalization /datadrive/service-headroom/a100-normalization.json --out /datadrive/service-headroom-confirmation
uv run python service_headroom_campaign.py reduce-confirmation --plan runs/service-headroom/a100-confirmation.json --core-plan runs/service-headroom/plan.json --scout /datadrive/service-headroom/a100-scout.json --runs /datadrive/service-headroom-confirmation --out /datadrive/service-headroom/a100-confirmed.json
uv run python plot_service_headroom.py --plan runs/service-headroom/plan.json --normalization /datadrive/service-headroom/a100-normalization.json --scout /datadrive/service-headroom/a100-scout.json --confirmation-plan runs/service-headroom/a100-confirmation.json --confirmed /datadrive/service-headroom/a100-confirmed.json --transition outputs/service-admission-transition-a100-20260816/summary.json --out outputs/service-headroom-a100/service-headroom
```

Only a confirmation result accepted by `supported_bound()` supplies a service
limit; that loader rechecks the exact core plan, scout, confirmation plan, and
all held-out decisions. It returns a total normalized-load cap, so P1/P2 use
`b_f + b_g + sum_i(w_i,f + w_i,g) <= rho_safe`; available added headroom is
`rho_safe - (b_f + b_g)`.
`validate_confirmation_evidence()` separately authenticates rejected held-out
evidence for analysis and plotting; it never supplies a planner bound.
The paper figure is P90, matching the DistServe-style comparison; P99 remains
null unless a cell has at least 1,000 incumbent completions. TTFT and TPOT
targets are declared evaluation inputs; raw P90 curves, joint
offered-request attainment, physical stability, and censoring remain in the
outputs. The harness preserves exact token IDs/events, labeled Prometheus
scrapes, queue/KV/power series, partial failures, cache proof, and complete
runtime identity, including the exact vLLM, LMCache, and Redis commands. It
uses one asynchronous task per offered request across 32 fixed event-loop
shards, pins the aiohttp version, and disables cyclic GC during the trace to
prevent periodic scheduler stalls. It restores GC afterward and invalidates
any cell whose launch schedule slips by more than 50 ms or whose exact token
timing coverage falls below 99%. TPOT quantiles use only exact streams, report
that coverage, and treat an ambiguous stream as a joint-SLO miss. It
fits total in-system requests over the final two thirds of the measurement
window and calls growth material only above one fitted accumulated request;
the older block-bootstrap upper bound remains a diagnostic field and does not
decide feasibility. It
enforces the frozen randomized cell order and hashes an unchanged image once
per stage. The balanced confirmation workload is derived from each hardware's
normalization and must have a 40--60% prefill share. Live KV capacity and the
planned block-rounded prefix stock are bound into the normalization. Exact
successful uncached prewarm tokens must differ from the incumbent-only control
by that planned stock, and confirmation must reproduce the full prewarm count;
the vLLM active-KV gauge is retained only as a diagnostic because it does not
measure reclaimable APC blocks. Use the exact
vLLM 0.22/LMCache 0.5.1 MP stack provisioned by `setup.sh`
(`QH_RUNTIME=native`, `QH_LMCACHE_MODE=mp`) or the checksum-pinned Apptainer
equivalent. The selected runtime mode, versions, and semantic commands become
part of the cross-cell service identity. Discovery and confirmation may use
different collector commits only when that service identity is unchanged;
both complete runtime identities and collector SHAs remain in the evidence.
Submit A100 and H100 separately.
Retry an invalid measurement immediately before starting the next frozen cell;
a later retry violates the audited order and stops the stage. A valid service
failure is never retried away.
See `DATA_TO_COLLECT.md` for the 54-cell discovery and 18-cell confirmation
matrix per hardware and the claim boundary.

The completed A100 run is in `outputs/service-headroom-a100-20260815/`. All 54
discovery cells and all 18 unseen confirmation cells completed in frozen order;
the confirmation service had zero restarts, invalid cells, or cache mismatches.
The isolated normalization was 16,758.93 prefill tok/s and 3,597.59 decode
tok/s. Discovery selected `rho=0.70` as the candidate and `rho=0.85` as the
first fail for both directional slices. Held out, however, every `rho=0.70`
mix passed only two of three blocks under the one-fitted-request stability
contract. Prefill-heavy `rho=0.85` failed TPOT in all three blocks; decode-heavy
`rho=0.85` passed one of three blocks and failed stability in two. The balanced
`rho=0.70` check also passed only two of three. The reducer therefore correctly
reports `planner_usable=false` and no supported scalar bound.

Interpret `rho` only as offered normalized phase work,
`rho_p + rho_d`, computed from exact offered tokens and isolated phase rates;
it is not measured GPU utilization and is not composition-invariant. The main
figure follows the prefill-heavy ray in two raw-latency panels: TTFT versus
measured `rho_p` and TPOT versus measured `rho_d`. Total `rho` parameterizes
both trajectories. Crosses and vertical projections mark the first measured
misses, TPOT at `rho=0.85` and TTFT at `rho=1.10`. The phase figure and
committed CSVs retain the other composition and restart-block detail, so this
single ray is not a two-dimensional response model.

New fixed-profile RPS sweeps can remain conventional at the paper boundary.
`destination.ProfileRateLimit` converts a measured total-RPS limit into the
existing `(1,1)` service facet, including baseline and added headroom, and
checks the exact destination type, context, and prefill/decode ray. The helper
is opt-in: it changes no destination schema, planner default, power model, or
historical result. A different request class requires a different measured
limit rather than extrapolation through the helper.

At total `rho` near 0.70, held-out prefill-heavy, balanced, and decode-heavy P90
TTFT/mean-TPOT medians are respectively 234.7/59.7 ms, 152.9/42.7 ms, and
131.7/37.2 ms. This composition dependence proves that total work is not a
latency-response model. It does not prevent using a lower total-work row as a
conservative admission certificate when every tested composition passes. A
two-dimensional campaign is needed only to claim extra headroom for unseen
phase mixtures; the present evidence does not make that stronger claim.

`service_admission_transition_campaign.py` is the deliberately small live
follow-up. It does not fit an admission surface or exercise an end-to-end
Queue-Haul migration. It freezes three discrete normalized phase-work recipes
`(w_F,w_D)` in GPU-s/s: prefill-heavy `(0.3078,0.1919)`, balanced
`(0.1996,0.3003)`, and decode-heavy `(0.08163,0.41806)`. Their sums are near
0.50 by construction; that scalar is neither utilization nor an admission
limit. Each of three fresh restart blocks per recipe begins with 60 seconds of
incumbent-only traffic,
cold-materializes eight added session prefixes during a fixed 30-second
window, and then offers both cohorts for 240 seconds. A recipe passes only if
all three blocks preserve the frozen 1-second TTFT and 100-ms TPOT rules for
both incumbents and the added cohort, exactly complete and cache every request,
retain complete telemetry, and meet the strict queue-stability contract.
Admission failure is a measured failure, not a retryable invalid cell. Invalid
instrumentation attempts are retained in immutable per-attempt directories.
The result certifies only the three tested recipes for the 240-second declared
horizon; `planner_usable` remains false and no interpolation is allowed.

The transition completed 9/9 cells successfully, making `W=0.50` a tested safe
floor for this exact A100/4K serving profile. It is not installed as a planner
cap. At `W=0.70`, all held-out repeats remained inside both latency SLOs, but
each mix passed the strict stability contract in only two of three blocks.
Thus the data brackets a higher stable cap in `[0.50, 0.70)` without selecting
one. If deployment needs more headroom, the minimal next measurement is one
preregistered `W=0.60` transition point. This interpretation uses the
campaign's 16,758.928 prefill-token/s and 3,597.591 decode-token/s normalization
and must not be copied into profiles with different rates. No planner/schema
change or historical-profile edit is required; the current `(1,1)` service
normal already represents the row, while KV, bandwidth, migration, and power
stay separate.
The result follows the load-sweep methodology used by
[DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) and
[vLLM's serving benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli.html):
hold the workload profile fixed, vary offered load, and retain the last point
whose latency and completion contract passes.

```bash
uv run python service_admission_transition_campaign.py prepare \
  --source-plan outputs/service-headroom-a100-20260815/plan.json \
  --normalization outputs/service-headroom-a100-20260815/normalization.json \
  --scout outputs/service-headroom-a100-20260815/scout.json \
  --confirmation-plan outputs/service-headroom-a100-20260815/confirmation-plan.json \
  --confirmed outputs/service-headroom-a100-20260815/confirmed.json \
  --out outputs/service-admission-transition-a100-20260816/plan.json
QH_RUNTIME=native QH_LMCACHE_MODE=mp uv run python \
  service_admission_transition_campaign.py run \
  --plan outputs/service-admission-transition-a100-20260816/plan.json \
  --normalization outputs/service-headroom-a100-20260815/normalization.json \
  --run-root /datadrive/qh-service-admission-transition-a100-20260816-r1 \
  --summary outputs/service-admission-transition-a100-20260816/summary.json
```

The verified 2026-07-23 destination bundle is retained under
`outputs/destination-v7-20260722/`. Do not treat its service rows as an accepted
capacity profile. See `FINDINGS.md` and `DATA_TO_COLLECT.md` for the forensic
audit and retained evidence.

## Development rules

- Keep implementation and documentation small.
- Hard-fail unsupported domains and invalid evidence.
- Add semantic tests for every source change.
- Run `uv run pytest` after every change.
- Commit each completed task separately with a descriptive message.

## Model-architecture campaign

`single_gpu_capacity_campaign.py` is the non-gating A100 precursor. It discovers
how much of each pinned checkpoint can actually be served on one physical GPU;
it does not require the requested width or context to pass. Each of the three
models gets a fresh engine at five contexts. Synchronized geometric bursts
from 1 through 256 requests distinguish the largest eventually completed burst
from the maximum number vLLM reports running simultaneously. Launch rejection,
OOM, queue saturation, timeout, or a service crash is retained as the capacity
outcome rather than retried away. Only infrastructure and runtime-contract
failures are retryable.

The runtime is one A100, BF16 KV, TP1, 32K `max_model_len`, 256
`max_num_seqs`, 90% memory, chunked prefill, APC, eager execution, and the
hybrid KV manager. Qwen contexts are multiples of its measured 784-token
unified block and its LMCache server uses separate object groups. The pinned
vLLM 0.22 Qwen cache is exposed as a K/V-major transpose of a contiguous
page-major allocation; `connector_patch.py` restores that page-major view
without copying and delegates its 784-token logical-page re-view to LMCache.
The live launch must prove that group geometry. Qwen and Gemma use LMCache's
GPU-visible `lmcache_driven` transport because its engine-driven gather path
does not support hybrid KV cache groups. On a sleep-enabled source, their KV
backing uses PyTorch's standard CUDA allocator so LMCache can export it over
CUDA IPC; model weights remain in vLLM's CuMem pool for level-1 sleep/wake.
GPT-OSS retains `engine_driven`.
Results are descriptive limits, not admission gates.

```bash
uv run python single_gpu_capacity_campaign.py prepare --seed 1 --out runs/single-gpu-capacity-a100/plan.json
CUDA_VISIBLE_DEVICES=0 QH_RUNTIME=native QH_LMCACHE_MODE=mp uv run python single_gpu_capacity_campaign.py run --plan runs/single-gpu-capacity-a100/plan.json --run-root /datadrive/single-gpu-capacity-a100
uv run python single_gpu_capacity_campaign.py reduce --plan runs/single-gpu-capacity-a100/plan.json --run-root /datadrive/single-gpu-capacity-a100 --out outputs/single-gpu-capacity-a100/summary.json
uv run python plot_single_gpu_capacity.py outputs/single-gpu-capacity-a100/summary.json outputs/single-gpu-capacity-a100/single-gpu-capacity.pdf
```

The raw evidence includes exact token events, complete request responses,
Prometheus scrapes, queue/running/KV traces, power samples, server info, launch
logs, runtime-reported KV-token capacity, available KV GiB, and the complete
hashed runtime command identity. An open marker in the reduced plot means the
last tested synchronized burst completed and is therefore a right-censored
lower bound, whether the geometric sweep ended at 256 or stopped after a
repeated running-capacity plateau. An `x` means that model/context did not
launch.

The completed A100 run is under
`outputs/single-gpu-capacity-a100-20260815/`. All 15 model/context cells
launched on their first attempt, all tested bursts completed and drained, and
the persistent service exited successfully without a restart. Maximum observed
simultaneous running requests across increasing contexts were
`65/33/17/12/10` for both GPT-OSS and Gemma, but only `8/5/3/2/2` for Qwen.
This equality does not make GPT-OSS and Gemma equivalent: runtime KV capacity
was 1,952,597 tokens for GPT-OSS versus 286,068 for Gemma, and first-saturation
KV usage was 11--13% versus 45--89%. Qwen exposed 285,354 KV tokens, proved its
784-token hybrid alignment, and saturated at 16--20% KV usage. Treat service
flow, KV capacity, prefill, decode, and power as separate measured constraints;
the observed request plateau is not a scalar utilization bound.

`model_architecture_campaign.py` reuses the migration profiler and base planner
for the pinned GPT-OSS-20B, Qwen3.8-27B, and Gemma-4-26B-A4B checkpoints. It
keeps BF16 KV, TP1, 32K context, eight sessions, 90% GPU memory, and exact token
shapes fixed across A100 and H100 arms. A100 arms use native vLLM 0.24.0,
Transformers 5.15.1, and LMCache 0.5.1 so Gemma's heterogeneous attention
geometry is explicit; H100 arms retain vLLM 0.22.0/LMCache 0.5.1. Use
`QH_RUNTIME=native` and `QH_LMCACHE_MODE=mp`.

```bash
uv run python model_architecture_campaign.py prepare --out-dir runs/model-architecture/plans
uv run python model_architecture_campaign.py run-profile --plan PLAN.json --run-root RUN-smoke --smoke-only
uv run python model_architecture_campaign.py run-profile --plan PLAN.json --run-root RUN-full
uv run python model_architecture_campaign.py freeze-profile --base-profile BASE.json --run-root RUN-full --smoke-root RUN-smoke --geometry kv_geometry.csv --out PROFILE.json
```

`model_hardware_drain_campaign.py` reuses those frozen profiles and the Azure
network executor for the repeated drain experiment. Each arm runs ten matched
eight-session context packs five times: Queue-Haul greedy plans a complete
30-second evacuation, then all eight actions are released together. The A100
command runs GPT-OSS, Qwen, and Gemma end to end; H100 GPT-OSS is a separate
command. Profiles must be repository-local so the same pinned file exists on
every node; each must retain the adjacent passing `.gate.json` written by
`freeze-profile`; both are snapshotted into the portable arm output. The
calibration must be formal and match every cluster node's region, IP, and GPU.
A100 model order rotates inside each of five fresh-stack blocks. The first
pending episode in every block is a fail-fast smoke.
Reconstruction forces 128 output tokens and records exact token-event TTFT and
mean TPOT.

```bash
uv run python model_hardware_drain_campaign.py a100 --profiles A100_GPT.json A100_QWEN.json A100_GEMMA.json --cluster azure_network_cluster_east_germany.json --calibration A100_CALIBRATION.json --manifest MANIFEST.json --run-root /datadrive/model-hardware-drain-a100
uv run python model_hardware_drain_campaign.py h100 --profile H100_GPT.json --cluster azure_network_cluster_australia_southcentral.json --calibration H100_CALIBRATION.json --manifest MANIFEST.json --run-root /datadrive/model-hardware-drain-h100
uv run python model_hardware_drain_campaign.py reduce --run-root /datadrive/model-hardware-drain-a100 --run-root /datadrive/model-hardware-drain-h100 --out outputs/model-hardware-drain
```

Each arm retains raw requests and exact token timestamps, power and utilization
samples, route bytes and TCP telemetry, planner decisions, and checkpoints.
`results.csv` adds TTFT, mean TPOT, dispatch skew, four-way action counts, and
modeled power-attainment time. Failures are never retried into successes: they
remain immutable non-attainment observations with raw failure evidence, and
interrupted attempts are sealed as failures on resume.
Reduction writes the combined episode table with physical route regions,
method mix, and deadline ECDF. Interpret arm differences as configuration
effects: model/runtime geometry, compute, power, capacity, and the H100 routes
all differ, so action mix alone does not isolate KV encoding.
Raw GPU power traces are observational; shed and power-attainment fields are
explicitly model-derived.

`matched_action_campaign.py` is the narrow cross-hardware/cross-model decision
demonstration. It freezes completed A100 East/Germany frontier scenario
`4ce7626a1f20a5c3`: eight 16K sessions, 80% requested shed, a 30-second
deadline, measured natural links, and zero destination background. The
harmonized A100 solve must reproduce the executed eight-KV-to-Germany decision.
Each H100 arm uses its own measured prefill, decode, and non-monotone
coding-path power fit. Atomic per-arm checkpoints make reruns resumable, and
the GPT-OSS/H100 checkpoint is shared by both comparisons.

```bash
uv run python matched_action_campaign.py
```

The pinned result is eight Germany KV moves for GPT-OSS/A100; five Germany plus
one East Replay move for GPT-OSS/H100; seven Germany KV moves for Qwen/H100;
and six Germany Replay moves for Gemma/H100. The first three reach the target.
Gemma remains on its measured high-power plateau after the six admissible moves
and therefore cannot meet the same 80% target. All 200 calibration-bootstrap
draws reproduce each arm's central action mix and feasibility result.

The H100 repeat-holdout RMSE is 2.73 W for GPT-OSS, 2.24 W for Qwen, and
1.32 W for Gemma. GPT-OSS and Gemma use their completed 108-work-cell power
runs; Qwen uses the 43 cells durably committed before the host reboot, of which
eight cover the matched mixed-load path. Only the archived A100 migration is a
physical migration. The H100 bars are planner decisions, BF16 KV bytes remain
analytic, endpoint residuals are zero, and the bootstrap covers calibration
uncertainty rather than workload-population uncertainty.

Run the smoke command for all six arms before any full run. `BASE.json` must be
the model/hardware arm's measured service and power profile. The geometry CSV
has one row per context, repeat, and runtime-reported cache group with columns
`context_tokens,repeat,group,resident_bytes,capacity_bytes,transfer_bytes`.
Profile freezing requires all five contexts and three repeats, checks group
totals against measured LMCache payloads, and rejects held-out median/P90 timing
error above 10%/15%, P90 absolute error above one second, or any false-feasible
19/30-second deadline.

Pass six `--arm MODEL HARDWARE PROFILE GATE` arguments to `screen`. It writes
the paired fixed-arrival and 40%-utilization-matched analysis, cache-only and
compute-only counterfactuals, the canonical action-mix figure, and six live
plans totaling 36 runs. Screening proceeds only with an in-grid crossover and
either a confidence-separated 10-point action-share change or a feasibility
flip. Execute each live plan with `run-profile`, then pass the six
`--run MODEL HARDWARE ROOT` arguments to `validate-live`. Interpret accepted
differences as architecture/deployment behavior, not a causal sparsity effect.


## 2×A100 marginal bottleneck comparison

`marginal_state_campaign.py` replaces the exhaustive 7,400-episode sweep with
six offline-selected points: slack control, WAN, prefill, HBM, serving, and
combined WAN/prefill. Each uses the fixed eight-session `large-r0` pack, all
five policies, and three paired repeats: **90 live episodes**. The repeat order
covers every point once before the next repeat. Policy order is seeded and
randomized within each point/repeat.

Preparation reuses the existing frozen profile and measured background
capacities. It places residual budgets just below full-pack demand, verifies
actual LP and greedy action changes offline, and requires a full mixed-action
plan at the combined bottleneck. These are predicted capacities; every live
policy measures its actual background and records the resulting decisions.
Live outcomes never determine which points are retained. HBM and serving may
produce ties because their demands do not depend on migration action.

The shared runner preserves loaded models while WAN and HBM settings stay
fixed. Each policy still resets caches and starts a fresh background with a
30-second observation window. Fractional background rates and HBM units allow
points between the former integer rungs. HBM allocations are actual held worker
memory, with corresponding destination KV blocks removed.

```bash
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
export QH_LMCACHE_MODE=mp
uv run python marginal_state_campaign.py prepare \
  --source-plan outputs/constrained-resource-a100-20260905/prepared/plan.json \
  --out outputs/marginal-resource-a100-20260905/prepared
sbatch marginal_state_campaign.sbatch
```

The batch job has a six-hour limit and runs the repeated comparison followed
by live action-shift validation. `offline_decisions.csv` records predicted
policy actions. The run retains every episode, actual capacities, request and
power evidence, attainment plots, and the final per-point robustness summary.
A failed action-shift validation remains visible rather than being discarded.
Rerunning with the same run root resumes only the exact hashed schedule prefix;
interrupted attempts remain intact. The old `constrained_state_campaign.py
prepare` command still reproduces the exhaustive historical design and is not
used by this comparison.

## Calibrated WAN/replay contention deadline sensitivity

`contention_campaign.py` searches heterogeneous eight-session packs, shaped
WAN rates, and explicit per-case deadlines in simulation before selecting live
cases. It fits aggregate replay throughput and the KV completion tail from the
completed marginal campaign. The simulation shares WAN bandwidth across
transfers and replay capacity across simultaneous replays; both resources can
operate concurrently. The planner profile remains unchanged.

Selection requires predicted full-target attainment by both QH variants,
prefers full attainment across all three timing variants, then maximizes
improvement over per-session greedy. Both QH variants must tie or beat all
three baselines in windowed relief at nominal and ±15% replay/tail durations. The live schedule contains
the selected case, three nearby qualifying settings, and a slack control, each
with all five policies and three fresh repeats: at most 75 episodes. Every
simulated candidate, including baseline winners, remains in `simulation.csv`;
`calibration.json` identifies the measurements and fitted values. Calibration
uses equal-context measurements, so heterogeneous performance remains a
prediction to be tested on hardware.

This snapshot experiment tests WAN/replay contention, not persistent HBM
residency or sustained serving capacity. The preceding marginal experiment
confirmed action shifts but did not beat per-session greedy, and its HBM and
serving admission assumptions did not match physical snapshot execution.

```bash
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
uv run python contention_campaign.py prepare \
  --source-plan outputs/constrained-resource-a100-20260905/prepared/plan.json \
  --calibration-plan outputs/marginal-resource-a100-20260905/prepared/plan.json \
  --calibration-raw outputs/marginal-resource-a100-20260905/run/raw_episodes.jsonl \
  --out outputs/contention-a100-20260905/prepared
sbatch contention_campaign.sbatch
```

The six-hour batch job runs the frozen cases, then writes `comparisons.csv`
and `validation.json` alongside the episode evidence and canonical plots.
Comparisons pair each QH variant with each baseline at the same case and repeat;
losses remain in the outputs. The primary validation requires three repeats,
no paired full-target attainment losses, and at least one strict attainment win
over per-session greedy for each QH variant. Partial windowed relief is a
secondary diagnostic. Completion time alone is insufficient: full relief must
hold over the final five-second window of the declared deadline.
The shared runner uses each state's deadline for planning and each scheduled
policy's deadline for measurement reduction; the historical default is 30s.

The frozen `outputs/contention-a100-20260905/prepared` plan uses paired 16K,
24K, 28K, and 31,562-token sessions. Its five settings are 5Gbps/20s,
4Gbps/20s, 6Gbps/20s, 5Gbps/22s, and the 10Gbps/25s slack control. These are
simulation-selected predictions; the batch job independently tests their
attainment on hardware. The search retained 5,670 timing/policy outcomes.

## Original 30-second contract

`contention_campaign.py prepare --original-contract` fixes every candidate and
control to the original 30-second deadline and five-second relief window.
It retains eight sessions, the two-A100 runtime, the model profile, and the
full-shed power target. Only context lengths and shaped WAN rates change.
The shorter-deadline experiment above is a separate sensitivity study, not
validation of this original contract.

The `outputs/contention-original-a100-20260906/prepared` plan uses eight
31,562-token contexts at 2.5, 3, 3.25, 3.5, and 10Gbps, all at 30 seconds.
Each of the five policies gets three fresh repeats (75 episodes). Concurrent
execution is calibrated from the completed heterogeneous-context campaign;
all candidate outcomes and calibration measurements remain in the prepared
output. Simulation predicts mixed QH actions meet the full-target deadline
while isolated greedy's eight replays exceed the 25-second migration cutoff
needed for five seconds of full relief. The completed 75-episode run
validated both QH variants at all four constrained rates (3/3 each), with
isolated greedy missing every constrained case and tying at the 10Gbps
control. The original deadline stayed fixed.

```bash
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
uv run python contention_campaign.py prepare --original-contract \
  --source-plan outputs/constrained-resource-a100-20260905/prepared/plan.json \
  --calibration-plan outputs/contention-a100-20260905/prepared/plan.json \
  --calibration-raw outputs/contention-a100-20260905/run/raw_episodes.jsonl \
  --out outputs/contention-original-a100-20260906/prepared
sbatch --job-name=qh-original-30s \
  --output=outputs/contention-original-a100-20260906/job-%j.log \
  contention_campaign.sbatch outputs/contention-original-a100-20260906
```

## Controlled prefill-pressure probe

`prefill_pressure_campaign.py` tests a narrow simulation-predicted transition
at fixed 4Gbps WAN, eight 27,360-token sessions, the original 30-second deadline,
and the five-second relief window. Only the background prefill rate changes:
0, 0.4, 0.5, and 0.6 requests/second. All five policies run three fresh repeats
at each rate (60 episodes). HBM allocations and serving backgrounds remain zero.

The timing model uses prior heterogeneous migration measurements and the
observed effect of prefill load from the earlier marginal campaign, rather
than treating inverse remaining prefill capacity as measured slowdown.
The completed 60-episode probe passed all 12 trials for each QH variant,
but isolated greedy missed all three zero-load controls by 0.35–0.47 seconds.
The control failure prevents attributing its misses specifically to prefill.
`prefill_validation.json` explicitly checks the zero-load control before
attributing any isolated-greedy miss to background prefill. A robust transition
requires all three control trials to pass and all three loaded trials to favor
both QH variants; unsuccessful probes and baseline wins remain in the outputs.

```bash
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
uv run python prefill_pressure_campaign.py prepare \
  --out outputs/prefill-pressure-a100-20260906/prepared
sbatch --job-name=qh-prefill-probe \
  --output=outputs/prefill-pressure-a100-20260906/job-%j.log \
  contention_campaign.sbatch outputs/prefill-pressure-a100-20260906 \
  prefill_pressure_campaign.py
```

## Additional fixed-case repeats

`outputs/robustness-a100-20260907` freezes ten additional paired repeats
(IDs 3–12) for every original-contract WAN and prefill case and all five
policies: 450 new episodes, giving 13 observations per case and policy with
the existing runs. Workloads, capacities, deadline, and relief window are
unchanged. Policy order is randomized within each repeat. Five sequential
array tasks each run two repeats of both campaigns, with a six-hour limit
per task; sequential execution avoids shared testbed port conflicts.

Submit with `sbatch outputs/robustness-a100-20260907/run.sbatch`.
Each task retains raw episodes and produces normalized CSVs and existing
diagnostic plots. Runs resume from their saved schedule prefix. These are
distribution measurements, so execution does not require QH wins or a passing
prefill causal control. Final CDFs and descriptive error bars should group by
case and policy, retain deadline misses, and treat each eight-session episode
as one repeat. The original three trials should remain identifiable as pilot
measurements when combined with the ten fresh repeats.

Job array `42334565` completed all 450 additional episodes without action
errors in 14h47m of batch runtime. Both QH variants attained the full target
in all 90 new trials each. Isolated greedy passed the ten WAN controls and
ten highest-prefill trials, missing the other 70. KV-only and replay-only
completed admitted subsets but never attained the full eight-session target.
The prefill zero-load control still fails for isolated greedy, so these
results do not establish a separate causal prefill transition.

The combined `outputs/robustness-a100-20260907/episodes.csv` contains all 585
initial and additional episodes, with `campaign`, `phase`, and `source_csv`
columns identifying provenance. Per-batch raw JSONL, normalized CSVs, plan
hashes, and diagnostic PNGs are included alongside the initial campaign data.
Existing `target_attainment.png` plots show full-target attainment over time;
the pooled episode-completion ECDFs count each episode once (65 WAN or 52
prefill episodes per policy). An episode finishes when all eight source sessions
complete their route switches; its time is the last completion measured from
the common migration start, without adding the five-second power window.
Episodes with unsubmitted or failed sessions remain in the denominator as
incomplete mass. Completions after the 30-second deadline remain visible in
the tail but do not count as deadline successes.

Regenerate the two single-panel ECDFs and the action mix split by bottleneck with
`uv run python plot_wan_prefill_results.py`. PNG/PDF figures and
`pooled_summary.csv` are under `outputs/robustness-a100-20260907/pooled/`.
WAN pools all five cases (including the slack control); prefill pools all four
rates (including zero load). Each case has 13 repeats per policy. The
`wan_action_mix` and `prefill_action_mix` figures use horizontal stacked bars
in separate 2.1 × 1.75 inch PDFs/PNGs for placement side by side, scaled to the
paper column width. All four figures omit titles. The episode ECDFs also use
2.1 × 1.75 inch canvases, compact labels, and legends below the axes for
side-by-side column placement. A single-row action legend uses the shared paper action palette
(purple replay, green KV transfer) and canonical method hatches.
Each figure pools cases within its class: 520 WAN and 416 prefill source sessions per policy. `per_session_greedy` is displayed as Isolated
Fastest because it picks each session's fastest isolated action. Unselected
sessions remain visible as a separate action category if present. Boundary error
bars show percentile 95% bootstrap confidence intervals for mean replay share,
using 10,000 whole-episode resamples within each class/policy (seed 0). They
include variation across pooled cases, not just repeat noise within a case.
These are descriptive episode ECDFs, not evidence of a causal prefill transition.

`uv run python plot_wan_prefill_tradeoff.py` writes separate
`wan_action_attainment` and `prefill_action_attainment` PDFs/PNGs plus
`action_attainment.csv` in the same `pooled/` directory. Each point is one
full-plan episode: 156 WAN and 117 prefill points, with 13 repeats per case
for QH LP, QH Greedy, and True Greedy. Each class is pooled into one
2.1 × 1.6 inch scatter, excluding the 10 Gb/s WAN and zero-load prefill controls.
Small translucent markers have deterministic horizontal display offsets of
at most ±1 percentage point; attainment times and CSV KV shares remain exact.
The x-axis is KV transfer as a percentage of selected actions; the y-axis is
time to the requested full power reduction. The frozen target requires all
eight sessions, so attainment is the last successful completion plus the
five-second power window. This agrees with archived on-time attainment and
extends late completions beyond the horizontal 30-second deadline.

True Greedy's 0% and 100% KV episodes are the observed all-eight-replay
and all-eight-KV executions for those same cases. Pure-action markers overlay those same
endpoint observations, sharing both their times and their display offsets.
The CSV records each source episode once; the overlays are reused timings,
not extra baseline measurements. The separately recorded deadline-admitted
KV-only/replay-only policies move only subsets and are excluded from this
full-plan comparison. Four all-KV WAN endpoints are back-of-the-envelope
estimates, shown as hollow squares labeled `KV only` and recorded separately
in `wan_kv_estimates.csv`. The frozen pack has 12,381,585,408 KV bytes. The
13 all-KV 10 Gb/s control runs average 13.4219 s to completion, giving a fixed
overhead of 3.5166 s after subtracting byte-transfer time. Thus estimated
attainment is `8 * KV_bytes / bandwidth_bps + 3.5166 + 5` seconds: 48.14,
41.53, 38.99, and 36.82 s at 2.5, 3, 3.25, and 3.5 Gb/s. This assumes the
same byte volume, saturated shared WAN, and bandwidth-independent overhead;
these four estimates are not additional measured repeats. The control anchors
the estimate rather than providing an independent validation. Colors and
distinct policy markers come from `plot_style.py`.
