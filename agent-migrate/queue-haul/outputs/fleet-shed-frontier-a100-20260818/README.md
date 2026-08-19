# Fleet shed frontier, 12,000 sessions

One shedding source site (swedencentral) and two equally sized destination sites (eastus2, germanywestcentral), gpt-oss-20b on A100. Each source node owns its egress pipe at the measured per-pipeline rate, and reaches only the destination pools its own path serves.

Every cell reports the **largest shed the policy actually executed** inside every contract: committed by the deadline, destinations within the measured offered-RPS envelope, and pool service contracts honoured. Seeds are aggregated by the median.

Destination admission is capped at the measured 5 offered RPS per replica, the last swept rate whose median p90 TTFT meets the 2.0 s SLO. Across 975 headline rows, 0 exceeded that envelope under normal admission.

Power is accelerator-scoped: the sum of a measured per-GPU curve over the modeled fleet. No facility, cooling, or host power is claimed.

## Executed shed at rho=0.38

| Deadline (s) | Policy | Median executed shed | Median shed (kW) | KV share of commits |
|---|---|---|---|---|
| 30 | greedy | 6% | 3.4 | 5% |
| 30 | isolated_fastest | 6% | 3.4 | 5% |
| 30 | kv_only | 0% | 0.0 | 100% |
| 30 | queue_haul | 6% | 3.4 | 5% |
| 30 | replay_only | 6% | 3.6 | 0% |
| 60 | greedy | 8% | 4.5 | 3% |
| 60 | isolated_fastest | 8% | 4.5 | 3% |
| 60 | kv_only | 2% | 1.0 | 100% |
| 60 | queue_haul | 7% | 4.4 | 3% |
| 60 | replay_only | 10% | 6.1 | 0% |
| 120 | greedy | 12% | 7.1 | 15% |
| 120 | isolated_fastest | 13% | 7.5 | 19% |
| 120 | kv_only | 0% | 0.1 | 100% |
| 120 | queue_haul | 15% | 9.0 | 25% |
| 120 | replay_only | 24% | 13.9 | 0% |
| 180 | greedy | 15% | 8.5 | 5% |
| 180 | isolated_fastest | 15% | 8.9 | 10% |
| 180 | kv_only | 4% | 2.1 | 100% |
| 180 | queue_haul | 19% | 11.0 | 21% |
| 180 | replay_only | 36% | 21.4 | 0% |
| 240 | greedy | 17% | 10.1 | 3% |
| 240 | isolated_fastest | 22% | 12.9 | 20% |
| 240 | kv_only | 5% | 2.7 | 100% |
| 240 | queue_haul | 22% | 12.9 | 20% |
| 240 | replay_only | 50% | 29.5 | 0% |
| 300 | greedy | 71% | 41.7 | 39% |
| 300 | isolated_fastest | 25% | 14.9 | 19% |
| 300 | kv_only | 6% | 3.3 | 100% |
| 300 | queue_haul | 27% | 15.7 | 16% |
| 300 | replay_only | 63% | 36.8 | 0% |
| 375 | greedy | 85% | 50.0 | 34% |
| 375 | isolated_fastest | 29% | 16.8 | 15% |
| 375 | kv_only | 6% | 3.4 | 100% |
| 375 | queue_haul | 85% | 49.9 | 35% |
| 375 | replay_only | 79% | 46.2 | 0% |
| 450 | greedy | 93% | 54.7 | 0% |
| 450 | isolated_fastest | 86% | 50.7 | 64% |
| 450 | kv_only | 6% | 3.6 | 100% |
| 450 | queue_haul | 99% | 58.0 | 29% |
| 450 | replay_only | 93% | 54.7 | 0% |
| 600 | greedy | 100% | 58.8 | 0% |
| 600 | isolated_fastest | 99% | 58.3 | 67% |
| 600 | kv_only | 6% | 3.8 | 100% |
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
| 30 | 0.38 | 6% | 6% | -0.3% |
| 60 | 0.38 | 10% | 8% | -2.8% |
| 120 | 0.38 | 24% | 15% | -8.3% |
| 180 | 0.38 | 36% | 19% | -17.8% |
| 240 | 0.38 | 50% | 22% | -28.1% |
| 300 | 0.38 | 63% | 71% | +8.4% |
| 375 | 0.38 | 79% | 85% | +6.5% |
| 450 | 0.38 | 93% | 99% | +5.5% |
| 600 | 0.38 | 100% | 100% | +0.0% |
| 750 | 0.38 | 100% | 100% | +0.0% |
| 900 | 0.38 | 100% | 100% | +0.0% |
| 1800 | 0.38 | 100% | 100% | +0.0% |
| 2700 | 0.38 | 100% | 100% | +0.0% |
