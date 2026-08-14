# Paired repair-disabled hardware control

This bundle runs the three accepted `bandwidth-germany__prefill-germany`
episodes again with the same 0.1x disturbances and original initial schedule.
The controller computes and shadow-validates the repair, but the apply policy
keeps the original pending schedule. Each episode is hash-paired with the
corresponding successful applied-repair result in
`/datadrive/queue-haul-repair-20260814-r3`.

The run records per-request TTFT, power, bandwidth and prefill acknowledgements,
the unapplied repair proposal, and deadline/eventual power attainment. The run
is rejected if the initial schedule differs from its paired repair episode, the
shadow repair would not pass, any request is not HTTP 200, or any TTFT is
missing.

The dedicated hardware run root is
`/datadrive/queue-haul-repair-disabled-control-20260814-r1`.
