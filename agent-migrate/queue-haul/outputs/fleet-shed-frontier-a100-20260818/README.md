# Fleet shed frontier, 12,000 sessions

> **Partial run.** 822 of 975 headline rows are present; 153 never completed. Cells whose seeds are missing are aggregated over fewer seeds, and any cell missing a single-action baseline is absent from the advantage table.

One shedding source site (swedencentral) and two equally sized destination sites (eastus2, germanywestcentral), gpt-oss-20b on A100. Each source node owns its egress pipe at the measured per-pipeline rate, and reaches only the destination pools its own path serves.

Every cell reports the **largest shed the policy actually executed** inside every contract: committed by the deadline, destinations within the measured offered-RPS envelope, and pool service contracts honoured. Seeds are aggregated by the median.

Destination admission is capped at the measured 5 offered RPS per replica, the last swept rate whose median p90 TTFT meets the 2.0 s SLO. Across 822 headline rows, 0 exceeded that envelope under normal admission.

Power is accelerator-scoped: the sum of a measured per-GPU curve over the modeled fleet. No facility, cooling, or host power is claimed.

## Executed shed at rho=0.38

| Deadline (s) | Policy | Median executed shed | Median shed (kW) | KV share of commits |
|---|---|---|---|---|
| 30 | greedy | 6% | 3.7 | 15% |
| 30 | isolated_fastest | 6% | 3.7 | 15% |
| 30 | kv_only | 4% | 2.3 | 100% |
| 30 | queue_haul | 6% | 3.7 | 15% |
| 30 | replay_only | 6% | 3.3 | 0% |
| 60 | greedy | 12% | 7.1 | 34% |
| 60 | isolated_fastest | 12% | 6.8 | 39% |
| 60 | kv_only | 9% | 5.1 | 100% |
| 60 | queue_haul | 12% | 7.1 | 34% |
| 60 | replay_only | 10% | 6.1 | 0% |
| 120 | greedy | 28% | 16.5 | 40% |
| 120 | isolated_fastest | 25% | 15.0 | 53% |
| 120 | kv_only | 21% | 12.2 | 100% |
| 120 | queue_haul | 28% | 16.5 | 40% |
| 120 | replay_only | 24% | 13.9 | 0% |
| 180 | greedy | 42% | 24.6 | 42% |
| 180 | isolated_fastest | 39% | 22.9 | 59% |
| 180 | kv_only | 34% | 19.8 | 100% |
| 180 | queue_haul | 42% | 24.6 | 42% |
| 180 | replay_only | 37% | 21.6 | 0% |
| 240 | greedy | 55% | 32.4 | 43% |
| 240 | isolated_fastest | 52% | 30.8 | 62% |
| 240 | kv_only | 46% | 26.8 | 100% |
| 240 | queue_haul | 55% | 32.4 | 43% |
| 240 | replay_only | 50% | 29.5 | 0% |
| 300 | greedy | 62% | 36.7 | 0% |
| 300 | isolated_fastest | 65% | 38.5 | 64% |
| 300 | kv_only | 10% | 5.6 | 100% |
| 300 | queue_haul | 70% | 40.9 | 44% |
| 300 | replay_only | 63% | 36.8 | 0% |
| 375 | greedy | 79% | 46.2 | 0% |
| 375 | isolated_fastest | 80% | 46.9 | 66% |
| 375 | kv_only | 73% | 42.9 | 100% |
| 375 | queue_haul | 87% | 51.1 | 44% |
| 375 | replay_only | 79% | 46.2 | 0% |
| 450 | greedy | 93% | 54.7 | 0% |
| 450 | isolated_fastest | 86% | 50.7 | 64% |
| 450 | kv_only | 10% | 5.6 | 100% |
| 450 | queue_haul | 99% | 58.0 | 33% |
| 450 | replay_only | 93% | 54.7 | 0% |
| 600 | greedy | 100% | 58.8 | 0% |
| 600 | isolated_fastest | 99% | 58.3 | 67% |
| 600 | kv_only | 7% | 4.1 | 100% |
| 600 | queue_haul | 100% | 58.8 | 41% |
| 600 | replay_only | 100% | 58.8 | 0% |
| 750 | greedy | 100% | 58.8 | 0% |
| 750 | isolated_fastest | 100% | 58.8 | 67% |
| 750 | kv_only | 98% | 57.8 | 100% |
| 750 | queue_haul | 100% | 58.8 | 41% |
| 750 | replay_only | 100% | 58.8 | 0% |
| 900 | greedy | 100% | 58.8 | 0% |
| 900 | isolated_fastest | 100% | 58.8 | 67% |
| 900 | kv_only | 100% | 58.8 | 100% |
| 900 | queue_haul | 100% | 58.8 | 41% |
| 900 | replay_only | 100% | 58.8 | 0% |
| 1800 | greedy | 100% | 58.8 | 0% |
| 1800 | isolated_fastest | 100% | 58.8 | 67% |
| 1800 | kv_only | 100% | 58.8 | 100% |
| 1800 | queue_haul | 100% | 58.8 | 41% |
| 1800 | replay_only | 100% | 58.8 | 0% |
| 2700 | greedy | 100% | 58.8 | 0% |
| 2700 | isolated_fastest | 100% | 58.8 | 67% |
| 2700 | kv_only | 100% | 58.8 | 100% |
| 2700 | queue_haul | 100% | 58.8 | 41% |
| 2700 | replay_only | 100% | 58.8 | 0% |

## What multiple actions buy at rho=0.38

| Deadline (s) | rho | Best single action | Best flexible | Gain |
|---|---|---|---|---|
| 30 | 0.38 | 6% | 6% | +0.7% |
| 60 | 0.38 | 10% | 12% | +1.7% |
| 120 | 0.38 | 24% | 28% | +4.5% |
| 180 | 0.38 | 37% | 42% | +5.2% |
| 240 | 0.38 | 50% | 55% | +5.1% |
| 300 | 0.38 | 63% | 70% | +7.1% |
| 375 | 0.38 | 79% | 87% | +8.3% |
| 450 | 0.38 | 93% | 99% | +5.5% |
| 600 | 0.38 | 100% | 100% | +0.0% |
| 750 | 0.38 | 100% | 100% | +0.0% |
| 900 | 0.38 | 100% | 100% | +0.0% |
| 1800 | 0.38 | 100% | 100% | +0.0% |
| 2700 | 0.38 | 100% | 100% | +0.0% |
