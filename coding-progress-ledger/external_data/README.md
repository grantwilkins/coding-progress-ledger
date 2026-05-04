# external_data

Source-side data organization for imported corpora used by the retrospective pipeline.

## Subfolders

- `swe_agent/`: SWE-agent source manifests, policies, and cache tooling inputs.
- `hermes/`: Hermes source manifests, policies, raw samples, and pilot caches.

## Policy

- Raw upstream data is treated as immutable input.
- Commit manifests/summaries and tiny schema samples.
- Avoid committing large raw dumps unless explicitly required.
