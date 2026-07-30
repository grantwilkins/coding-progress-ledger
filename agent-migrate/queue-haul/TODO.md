# Queue-Haul TODO

The normative behavior is in `formulation_nsdi.md` and
`PROPOSED_DESTINATION_ARCH.md`. The primary executor launches a fixed plan in
order and overlaps moves up to its configured width.

## Required

- [x] Validate replay and KV correctness, continuation, and exact byte/block
  accounting on two A100s.
- [x] Complete the ordered eager-parallel width-eight policy campaign.
- [x] Keep measured, fitted, simulated, and assumed evidence labels explicit.
- [x] Validate realized destination service debt during summary prediction.
- [ ] Regenerate canonical simulator tables and figures from a clean checkout.
- [x] Add `greedy_bundle` to pool-aware fleet-policy comparisons.
- [ ] Verify checksums and paper tables against the committed code revision.
- [ ] Run `uv run pytest`.

## Optional

- Measure an accepted destination normal/stable service boundary.
- Validate planner pacing before making it part of the hardware contract.
- Add live leases and commit-time contract revalidation.
- Extend hardware coverage beyond the dedicated two-A100 sink.
- Evaluate disaggregated pools and exhaustive diversity sweeps.
