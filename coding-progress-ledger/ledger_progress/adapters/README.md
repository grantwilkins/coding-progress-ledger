# ledger_progress/adapters

Source-specific adapter utilities that map external traces/events into package-native structures.

## Files

- `generic.py`: shared adapter helpers.
- `swe_agent.py`: SWE-agent trace-specific mapping helpers.

Adapters should normalize structure, not change ledger semantics.
