# external_data/hermes

Hermes source-data workspace: provenance docs, inventories, pilot sampling manifests, and local pilot cache outputs.

## Layout

- `SOURCE_FORMAT.md`: upstream field/schema notes.
- `PILOT_SAMPLING_POLICY.md`: deterministic sample rules.
- `manifests/`: inventory and pilot sample CSVs.
- `raw/`: small schema/sample row artifacts.
- `pilot_cache/`, `pilot_cache_h5/`: cached per-source-row JSON used by import scripts.

## Notes

The cache directories are operational artifacts for deterministic replay and should be treated as source-derived inputs, not hand-edited data.
