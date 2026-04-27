# Run Notes

Progress increased when the deterministic async tests encoded the out-of-order
scenario and when request identity was added to the controller.

Progress decreased twice. First, the loading-state check showed that simply
setting `loading = False` on any completion was not enough. Second, the local
pytest run showed a validation dependency problem: `pytest.mark.asyncio` made
the repo less self-contained, so test execution itself became newly active work.

Completed subtasks cite concrete evidence from `async_result.py`, the
`tests/test_async_result.py` diff, and the final pytest output in
`test_output.txt`.

The ledger was useful because the task looked like a single stale-result fix,
but the validation surface split into result protection, loading protection, and
self-contained async test execution. The awkward part was that a test-harness
repair counted as discovered work even though it was not part of the product
bug.
