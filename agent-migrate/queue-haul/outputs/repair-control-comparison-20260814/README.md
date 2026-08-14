# Live repair comparison

This figure uses the paired A100 hardware episodes with online repair enabled
and disabled. Every pair starts from the same hashed 15-session plan and sees
the same simultaneous 10x reduction in Germany bandwidth and prefill
throughput. The plotted trace is repeat 0; the repair decision and action mix
are identical in all three repeats.

Suggested caption:

> Queue-Haul repairs a live schedule after a 10x drop in Germany bandwidth and
> prefill throughput. (a) Both executions begin with the same 15-session plan.
> Queue-Haul replans at 6.2 s, executes one additional East replay, stops seven
> planned Germany-bound actions, and retains six sessions at the source; the
> same repair was selected in all three runs. (b) Representative measured A100
> execution. Both policies reach the 30.9 W shed target, but Queue-Haul reaches
> it at 17.5 s rather than 21.7 s. Its ninth and final action completes at
> 17.5 s with a 14.3 s maximum TTFT; without repair, the fifteenth action
> completes at 141 s with a 91.7 s maximum TTFT.

The evidence supports a faster, lower-work repair claim. It does not support a
binary feasibility claim at this operating point: the no-repair control also
reaches the power target before 30 seconds.

Regenerate with:

```bash
MPLCONFIGDIR=/tmp/qh-mpl-cache UV_CACHE_DIR=/tmp/qh-uv-cache \
  uv run python plot_repair_control_comparison.py
```

Machine-readable inputs to the graphic are copied into `action_mix.csv` and
`attainment.csv`; aggregate and per-repeat metrics are in `summary.json`.
