# Tool Affordance Gaps

Evidence comes from the six GPT-5.4 / GPT-5.4-mini completed-failure runs.

## Supported Fixes

- Add head+tail default file reads and line-range chunked reads.
- Add first-class `find_files` and `grep` tools with bounded output.
- Add first-class `apply_patch` so agents can make targeted edits without whole-file rewrites.
- Classify network and dependency dead ends with visible controller messages.
- Record truncation, repeated file inspection, and chunked-read observation events.

## Failure Mix

- `agent_image_missing_runtime`: 3
- `no_network_install_mismatch`: 2
- `tool_affordance`: 1

## Remaining Work

- Run oracle-hidden success checks for the same three tasks before treating verifier failures as task difficulty.
- Add task compatibility tags for solve-time network, package install, service bootstrap, long-file context, and visible-test availability.
- Run a 6-task / 12-run calibration mini-pilot before the 12-task / 24-run pilot.
