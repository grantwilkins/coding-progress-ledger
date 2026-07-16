# Queue-Haul findings

The live two-model testbed demonstrates reliable session transfer and matching
continuations, but the existing full-drain data do not validate deadline or
power prediction. The checked-in profile therefore remains `estimated`.

The active simulator now models whole-session placement, background transfer,
shared links, serving and destination queues, catch-up, route switch, node
state, request wait, incomplete bytes, and trailing-window source power. Raw
tables stream to disk one run at a time; plot data are bounded.

Local validation on the M1 MacBook Pro passes the complete test suite and the
six-session command-line smoke in `README.md`. Long runs hard-fail when sampled
contexts exceed the measured 31.6k-token range. They are not paper results
until the GPU profiling and held-out validation in `handoff.md` complete.
