# Queue-Haul assumptions

The checked-in GPT-OSS-20B/A100 profile is estimated, not validated. It is
usable only over its recorded context, load, and concurrency ranges. The
simulator hard-fails outside those ranges. The single list of missing
measurements is `DATA_TO_COLLECT.md`.

Open `??` values for the destination-pool architecture
(`PROPOSED_DESTINATION_ARCH.md`); proposed defaults stand until replaced:

- Normal SLOs: p90 TTFT ≤ 2 s, p90 TPOT ≤ 100 ms ??
- Emergency SLOs: p90 TTFT ≤ 10 s, p90 TPOT ≤ 250 ms ??
- Trace sources for the prefill-heavy, decode-heavy, interactive-coding,
  and agentic workload manifests (coding uses
  `outputs/coding-manifest.json`) ??
