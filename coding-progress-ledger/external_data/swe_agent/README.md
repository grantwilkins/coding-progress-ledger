# external_data/swe_agent

Holding area for raw SWE-agent trajectories used by the retrospective pilot
(see `TASKS.md` § Workstream A).

## Layout

```
raw/         immutable raw inputs from upstream sources (parquet, jsonl, .traj)
manifests/   derived inventories and pilot sample CSVs
samples/     small human-readable excerpts for documentation
SOURCE_FORMAT.md  upstream schema reference (filled in by A2)
```

## Rules

- `raw/` is **immutable**. Never edit, normalize, or reformat files inside it.
  Normalization is a separate downstream artifact (see Workstream C).
- `raw/` is **out of scope for git**. It is excluded by the repo-root
  `.gitignore`. Large parquet/jsonl dumps must not be committed; if a small
  fixture is genuinely useful, place an excerpt under `samples/` instead and
  cite the row index it came from.
- Treat the upstream dataset (e.g. `nebius/SWE-agent-trajectories` on Hugging
  Face) as the source of truth. If you need to recover `raw/`, redownload from
  that source rather than restoring from a backup of this directory.
- `manifests/` and `samples/` are **committed** — they are derived artifacts
  small enough to live in the repo and they document what was sampled.

## License / usage

License and usage constraints for the chosen upstream source are recorded in
`SOURCE_FORMAT.md` (created by task A2). Do not redistribute raw traces from
this directory; share manifests and pilot summaries instead.
