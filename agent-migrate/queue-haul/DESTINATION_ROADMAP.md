# Queue-Haul evidence roadmap

`formulation_nsdi.md` defines the algorithm and
`PROPOSED_DESTINATION_ARCH.md` maps it to the executable contract.

## Established evidence

- Replay and KV handoff preserve state and continue serving on the pinned
  two-A100 setup.
- Serial and bounded-concurrency mechanism campaigns pass their correctness and
  deadline gates.
- The completed width-eight campaign validates planner-generated replay/KV
  choices under ordered eager-parallel, unpaced execution on a dedicated sink.
- Source power, migration timing, and KV accounting are measured or fitted with
  explicit provenance.

## Simulation boundary

Pool choice, replica packing, multi-pool contention, destination service debt,
and fleet-scale source-power shed are simulated using explicit contracts.
Destination service headroom, route reservations, and multi-pool inventories
remain assumed sensitivities until measured.

## Remaining work

1. Regenerate the fixed-contract and scale results from the committed code.
2. Compare static greedy, experimental bundle greedy, LP, replay-only, and
   KV-only policies on matched scenarios.
3. Report requested and achieved shed, action mix, last commit, route and pool
   resource use, exposed sessions, and complete binding-resource sets.
4. Keep the dedicated hardware result separate from simulated pool admission.

## Supported claim

> On the pinned two-A100 setup, Queue-Haul chooses replay or KV transfer and
> executes the fixed plan with ordered eager-parallel launch while retaining
> source ownership until request-boundary commit. Its simulator evaluates
> deadline-constrained source accelerator-power shed using measured, fitted,
> simulated, and explicitly assumed inputs.

The evidence does not establish production destination headroom, facility
power, live leasing, arbitrary mid-token migration, or hardware execution of
planner pacing.
