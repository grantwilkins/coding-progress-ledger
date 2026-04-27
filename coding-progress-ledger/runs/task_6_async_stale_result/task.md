# Task 6: Async Stale-Result Bug

The repository contains a small async search controller. Starting a newer
request should make that request the only one allowed to update public state.

Fix the controller so a slower old request cannot overwrite the result, loading
state, or error state for a newer request. Include deterministic tests for
out-of-order completion.
