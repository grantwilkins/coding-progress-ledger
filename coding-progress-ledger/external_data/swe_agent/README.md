# external_data/swe_agent

SWE-agent source-data workspace for inventorying, sampling, and reproducible pilot reconstruction.

## Layout

- `SOURCE_FORMAT.md`: verified upstream schema and provenance notes.
- `PILOT_SAMPLING_POLICY.md`: fixed inclusion/dedupe/sampling rules.
- `manifests/`: deterministic inventory and pilot sample outputs.
- `raw/`: immutable local source rows (excluded from git by policy).
- `samples/`: tiny documentation excerpts only.

## Data handling rules

- Do not mutate files under `raw/`.
- Do not commit large raw trajectory dumps.
- Commit manifests and summaries; they are the auditable source pointers.
- If raw cache is missing, reacquire from the declared upstream source in `SOURCE_FORMAT.md`.

## Typical workflow

1. Build inventory with `scripts/swe_agent_inventory.py`.
2. Generate deterministic pilot sample with `scripts/sample_swe_agent_pilot.py`.
3. Populate pilot cache and import runs via `scripts/populate_swe_agent_pilot_cache.py` and `scripts/import_swe_agent_trace.py`.
