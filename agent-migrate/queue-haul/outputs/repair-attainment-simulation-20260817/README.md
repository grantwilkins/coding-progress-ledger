# Paired repair-attainment simulation

This artifact contains 2,061 paired actual plan shifts drawn from 4,096 independently generated workload packs crossed with bandwidth, prefill, and joint 10x resource drops. The original and repaired residual schedules are both replayed under the same degraded timing. Curves retain failures to attain the target by 120 s in the denominator; they are not bootstrap replicas of the hardware cases.

`repair_response.pdf` is the single-column paper figure. Panel (a) shows the
paired attainment CDF. Panel (b) classifies every original action pending at
the repair decision as retained, method-only changed, destination changed, or
removed from the revised plan. Destination changes take precedence over
simultaneous method changes; newly added replacement actions are not part of
the pending-action denominator. Exact pooled counts are in `plan_changes.csv`.
