# Scheduled repair simulation

This replaces `../repair-plan-shift-sim-20260812/`. One calibrated regional
A100 plan is executed with four migration workers. At 25% aggregate planned
work, bandwidth and/or observed prefill capacity is reduced to 0.1x at East,
Germany, both, or neither. Running attempts are locked; only pending work can be
changed. Two missed forecasts are required before `repair_destination` runs.

Validation passes for all 16 cells. One cell (`bandwidth-germany` plus
`prefill-germany`) applies a target-restoring pending-work diff: one replay moves
from Germany to East and five no-longer-required pending moves are removed. Ten
cells report a revised attainable maximum, four degraded cells remain feasible
without repair, and the control remains feasible. The 12 bandwidth-cut cells
are sensitivity results pending the live 0.1x timing gate.

`plans.json` contains the event ledger, spliced schedule, discarded/retained
work, action mix, directional checks, and resource utilization/slack.
`repair_grid.png` and `repair_grid.pdf` use the shared plot style registry.
