# H5b — real workspace bytes on the H5a trajectories

H5a (`tests/test_h5a_multi_trajectory.py`) builds a multi-session SWE-style
fixture from five distinct cached pilot-zero trajectories with **synthetic**
1 GB workspace bytes per session. H5b graduates the byte source from
"synthetic int" to "real working-tree byte sum" by setting
`SessionSpec.workspace_path` to a freshly-cloned upstream repo per session.

## Setup

```bash
scripts/h5b/clone_repos.sh                  # default: /tmp/h5b_workspaces
scripts/h5b/clone_repos.sh /custom/dest     # or pick your own
```

Clones five upstream repos at HEAD (shallow, ~50 MB total network):

| sid | repo                          | trajectory                   |
| --- | ----------------------------- | ---------------------------- |
| cog | Melevir/cognitive_complexity  | swe_agent_pilot_s_01.json    |
| pok | hsahovic/poke-env             | swe_agent_pilot_s_03.json    |
| dcj | lidatong/dataclasses-json     | swe_agent_pilot_s_05.json    |
| ice | WIPACrepo/iceprod             | swe_agent_pilot_f_01.json    |
| scf | asottile/setup-cfg-fmt        | swe_agent_pilot_f_03.json    |

## Run the tests

```bash
uv run pytest tests/test_h5b_real_bytes.py     # uses /tmp/h5b_workspaces
VAGRANT_H5B_WORKSPACES=/custom/dest uv run pytest tests/test_h5b_real_bytes.py
```

Tests auto-skip if any of the five sub-directories is missing.

## Caveats

1. **HEAD vs base_commit.** The trajectories were collected against each
   instance's SWE-bench pre-fix `base_commit`, but the cached trajectory
   JSON does not surface that commit and the SWE-bench dataset metadata
   is not loaded locally. HEAD is defensible as "real bytes from the same
   upstream repo at a real commit"; the H1<D2 mechanism is bytes-layer
   (`gap = 8*B/bps`), so the byte magnitude regime is what matters, not
   the exact commit.

2. **Real bytes drift over time.** Repo HEADs grow. The H5b tests range-
   check each repo's bytes (e.g., "ice between 5 MB and 50 MB") rather
   than pinning exact values, so a +20% growth doesn't break the suite,
   but a +1000% growth does.

3. **No committed fixture.** Unlike H5a (`examples/traces/h5a_*.jsonl` is
   committed), H5b generates the trace dynamically per-environment. A
   committed trace would pin byte counts that drift with HEAD and would
   fail the byte-deterministic regenerator test on any future re-clone.
