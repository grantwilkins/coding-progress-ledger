from __future__ import annotations

import os
from pathlib import Path

import pytest

from observation_channel.hf import load_hf_rows
from observation_channel.readers import rows_to_turns


@pytest.mark.skipif(
    not os.environ.get("OBSERVATION_CHANNEL_HF_CACHE"),
    reason="set OBSERVATION_CHANNEL_HF_CACHE to run cached Hugging Face integration tests",
)
@pytest.mark.parametrize("source", ["swe-agent", "hermes", "terminalbench"])
def test_cached_hf_rows_convert_to_ordered_turns(source: str) -> None:
    cache_dir = Path(os.environ["OBSERVATION_CHANNEL_HF_CACHE"])
    rows = load_hf_rows(source=source, cache_dir=cache_dir, limit=1, local_files_only=True)
    converted = list(rows_to_turns(rows, source=source))

    assert converted
    _instance_id, turns = converted[0]
    assert turns
    assert [turn.step for turn in turns] == sorted(turn.step for turn in turns)
    assert all("<think>" not in (turn.command or "") for turn in turns)
