# TASK 1: Parser timezone offset bug

Create a tiny Python parser package whose initial `parse_offset(s)` accepts
`+05:30` but rejects compact offsets such as `+0530`. Add deterministic pytest
coverage, fix the parser so `+05:30`, `+0530`, and `-0330` work, and reject
invalid inputs.

The simulated coding run is tracked with `LedgerSession`; exported artifacts
live at this run root.
