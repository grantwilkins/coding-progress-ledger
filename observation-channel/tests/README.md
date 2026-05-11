# tests

Tests cover semantic contracts, not snapshots:

- reader extraction and hard-fail behavior
- classifier priority
- product target splitting
- stuck-unit accounting and v1.6 prefix features
- empirical-Bayes support, feature fallback, censoring, and calibration-pair rules
- CLI smoke paths
- gated cached Hugging Face integration

Run fast tests with:

```sh
uv run pytest observation-channel
```

Run cached HF integration with:

```sh
OBSERVATION_CHANNEL_HF_CACHE=data/raw/hf_cache uv run --project observation-channel --extra dev pytest observation-channel/tests/test_hf_integration.py
```
