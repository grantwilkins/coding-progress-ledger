# Queue-Haul additive model (archived)

This directory preserves the earlier additive power-shed model, its serial-link
simulator, plots, tests, and result files. It is historical research code and is
not imported or tested by the active `agent-migrate/queue-haul` implementation.

The active profile-driven controller and simulator are documented in
`agent-migrate/README.md`.

The frozen tests can still be run from the repository root:

```bash
uv run pytest _archive/queue-haul-additive-v0/tests
```
