from __future__ import annotations

import argparse
import re
from pathlib import Path

from .hf import expand_sources, iter_hf_rows, load_hf_rows, read_raw_sample, write_raw_sample
from .io import write_turns
from .readers import rows_to_turns
from .runner import annotate_corpus, annotate_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="observation-channel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache", help="materialize Hugging Face rows into data/raw samples")
    cache_parser.add_argument("--source", default="all", choices=["all", "swe-agent", "hermes", "terminalbench"])
    cache_parser.add_argument("--split", default="train")
    cache_parser.add_argument("--limit", type=int, default=None)
    cache_parser.add_argument("--config", help="optional Hugging Face config override for the selected source")
    cache_parser.add_argument("--agent-filter", default="mini-swe-agent", help="TerminalBench agent filter")
    cache_parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "raw" / "hf_cache")
    cache_parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "raw")

    preprocess_parser = subparsers.add_parser("preprocess", help="convert raw HF rows to canonical turn JSONL")
    preprocess_parser.add_argument("--source", required=True, choices=["swe-agent", "hermes", "terminalbench"])
    preprocess_parser.add_argument("--split", default="train")
    preprocess_parser.add_argument("--limit", type=int, default=None)
    preprocess_parser.add_argument("--config", help="optional Hugging Face config override for the selected source")
    preprocess_parser.add_argument("--agent-filter", default="mini-swe-agent", help="TerminalBench agent filter")
    preprocess_parser.add_argument("--input-jsonl", type=Path)
    preprocess_parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "raw" / "hf_cache")
    preprocess_parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "turns")
    preprocess_parser.add_argument("--local-files-only", action="store_true")

    annotate_parser = subparsers.add_parser("annotate", help="annotate one canonical turn JSONL file")
    annotate_parser.add_argument("turn_file", type=Path)
    annotate_parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "outputs")

    corpus_parser = subparsers.add_parser("annotate-corpus", help="annotate a directory of canonical turn JSONL files")
    corpus_parser.add_argument("turn_dir", type=Path)
    corpus_parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "outputs")

    args = parser.parse_args(argv)
    if args.command == "cache":
        return _cache(args)
    if args.command == "preprocess":
        return _preprocess(args)
    if args.command == "annotate":
        annotate_file(args.turn_file, args.out_dir)
        return 0
    if args.command == "annotate-corpus":
        annotate_corpus(args.turn_dir, args.out_dir)
        return 0
    raise AssertionError(args.command)


def _cache(args: argparse.Namespace) -> int:
    for source in expand_sources(args.source):
        rows = iter_hf_rows(
            source=source,
            cache_dir=args.cache_dir,
            split=args.split,
            limit=args.limit,
            config=args.config if args.source != "all" else None,
            agent_filter=args.agent_filter if source == "terminalbench" else None,
        )
        out_path = args.out_dir / source / f"{args.split}.jsonl"
        count = write_raw_sample(out_path, rows)
        print(f"cached {count} {source} rows to {out_path}")
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    if args.input_jsonl:
        rows = read_raw_sample(args.input_jsonl)
    else:
        rows = load_hf_rows(
            source=args.source,
            cache_dir=args.cache_dir,
            split=args.split,
            limit=args.limit,
            local_files_only=args.local_files_only,
            config=args.config,
            agent_filter=args.agent_filter if args.source == "terminalbench" else None,
        )
    out_dir = args.out_dir / args.source
    seen: dict[str, int] = {}
    for instance_id, turns in rows_to_turns(rows, source=args.source):
        safe = _safe_name(instance_id)
        seen[safe] = seen.get(safe, 0) + 1
        suffix = "" if seen[safe] == 1 else f"__{seen[safe]}"
        write_turns(out_dir / f"{safe}{suffix}.jsonl", turns)
    return 0


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean[:180] or "trace"


if __name__ == "__main__":
    raise SystemExit(main())
