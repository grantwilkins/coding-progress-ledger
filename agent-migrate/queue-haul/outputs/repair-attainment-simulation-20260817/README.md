# Paired repair-attainment simulation

This artifact contains 242 paired actual plan shifts drawn from 512 independently generated workload packs crossed with bandwidth, prefill, and joint 10x resource drops. The original and repaired residual schedules are both replayed under the same degraded timing. Curves retain failures to attain the target by 120 s in the denominator; they are not bootstrap replicas of the hardware cases.

`repair_response.pdf` shows the paired attainment CDF. `repair_actions.pdf` classifies each original pending action as retained, changed, redirected, or removed. Exact pooled counts are in `plan_changes.csv`.
