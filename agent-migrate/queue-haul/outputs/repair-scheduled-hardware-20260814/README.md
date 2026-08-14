# Guarded Azure scheduled-repair launch

`plan.json` pins the corrected regional timing parent, workload manifest, A100
profile, cluster, network calibration, timing summary, and all nine
implementation files. It contains 36 mandatory 0.1x regional timing checks and
48 main episodes (16 location combinations times three repeats).

The main grid runs only if the timing gate passes. During each episode the job
applies live route and uncached-prefill controls at 25% aggregate planned work,
observes a second sample one second later, shadow-validates the residual diff,
and applies only pending-work changes that reduce impaired-resource work. An
unstaged KV choice, an active-session change, or an applied repair that misses
the measured target fails validation.

Client-observed time to first content token (TTFT) is mandatory evidence. The
calibration gate records it in `calibration/timing_rows.csv`; every episode
records the raw monotonic start/first-token/end timestamps and `ttft_s` in its
result. The final reducer writes one row per migration to `repair_ttft.csv` and
reports TTFT p50, p90, and maximum in `validation.json`. Missing TTFT fails the
calibration gate or final validation.

Launch from the repository checkout:

```bash
export QH_AZURE_SSH_KEY=/path/to/azure-key
export QH_REPAIR_RUN_ROOT=/datadrive/queue-haul-repair-20260814
outputs/repair-scheduled-hardware-20260814/run.sh
```

No hardware evidence exists in this directory yet; `run.sh` writes it to
`QH_REPAIR_RUN_ROOT`.
