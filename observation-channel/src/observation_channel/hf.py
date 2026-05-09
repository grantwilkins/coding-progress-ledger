from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


DATASETS = {
    "swe-agent": ("nebius/SWE-agent-trajectories", None),
    "hermes": ("lambda/hermes-agent-reasoning-traces", "kimi"),
    "terminalbench": ("yoonholee/terminalbench-trajectories", None),
}


def expand_sources(source: str) -> list[str]:
    if source == "all":
        return list(DATASETS)
    if source not in DATASETS:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(DATASETS)} or 'all'")
    return [source]


def load_hf_rows(
    *,
    source: str,
    cache_dir: Path,
    split: str = "train",
    limit: int | None = None,
    local_files_only: bool = False,
    config: str | None = None,
    agent_filter: str | None = None,
) -> list[dict[str, Any]]:
    return list(
        iter_hf_rows(
            source=source,
            cache_dir=cache_dir,
            split=split,
            limit=limit,
            local_files_only=local_files_only,
            config=config,
            agent_filter=agent_filter,
        )
    )


def iter_hf_rows(
    *,
    source: str,
    cache_dir: Path,
    split: str = "train",
    limit: int | None = None,
    local_files_only: bool = False,
    config: str | None = None,
    agent_filter: str | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError("install the `datasets` package to use Hugging Face cache/preprocess commands") from exc

    dataset_name, default_config = DATASETS[source]
    selected_config = config if config is not None else default_config
    if source == "terminalbench" and agent_filter is None:
        agent_filter = "mini-swe-agent"
    kwargs: dict[str, Any] = {
        "split": split,
        "cache_dir": str(cache_dir),
    }
    if selected_config is not None:
        kwargs["name"] = selected_config
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    ds = load_dataset(dataset_name, **kwargs)
    if limit is None or agent_filter:
        iterator = ds
    else:
        iterator = ds.select(range(min(limit, len(ds))))
    emitted = 0
    for row in iterator:
        row_dict = dict(row)
        if agent_filter and str(row_dict.get("agent", "")).lower() != agent_filter.lower():
            continue
        yield row_dict
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def write_raw_sample(path: Path, rows: list[dict[str, Any]] | Iterator[dict[str, Any]]) -> int:
    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(row, default=str))

    count = 0

    def converted() -> Iterator[dict[str, Any]]:
        nonlocal count
        for row in rows:
            count += 1
            yield convert(row)

    write_jsonl(path, converted())
    return count


def read_raw_sample(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
