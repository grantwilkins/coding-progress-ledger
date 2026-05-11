from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .empirical_bayes import DEFAULT_BOOTSTRAP_RESAMPLES, EmpiricalBayesLookup, evaluate, query_json, write_diagnostics
from .hf import expand_sources, iter_hf_rows, load_hf_rows, read_raw_sample, write_raw_sample
from .io import ROW_FIELDS, read_jsonl, write_turns
from .readers import rows_to_turns
from .runner import annotate_corpus, annotate_file, annotate_turns


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

    cached_diag = subparsers.add_parser("cached-annotator-diagnostic", help="write combined cached annotator diagnostic CSVs")
    cached_diag.add_argument("--source", default="all", choices=["all", "swe-agent", "hermes", "terminalbench"])
    cached_diag.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    cached_diag.add_argument("--split", default="train")
    cached_diag.add_argument("--out-dir", type=Path, default=DATA_DIR / "diagnostics" / "cached_annotator")

    eb_eval = subparsers.add_parser("empirical-bayes-eval", help="evaluate empirical-Bayes final-unit lookup")
    eb_eval.add_argument("--turns-csv", type=Path, default=DATA_DIR / "diagnostics" / "cached_annotator" / "turns.csv")
    eb_eval.add_argument("--traces-csv", type=Path, default=DATA_DIR / "diagnostics" / "cached_annotator" / "traces.csv")
    eb_eval.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "empirical_bayes_v1")
    eb_eval.add_argument("--bundle-path", type=Path, default=DATA_DIR / "estimators" / "empirical_bayes_v1" / "lookup.json")
    eb_eval.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    eb_eval.add_argument("--seed", type=int, default=1729)
    eb_eval.add_argument("--min-support", type=int, default=25)

    eb_diag = subparsers.add_parser("empirical-bayes-diagnostics", help="write follow-up empirical-Bayes diagnostics")
    eb_diag.add_argument("--heldout-csv", type=Path, default=PROJECT_ROOT / "reports" / "empirical_bayes_v1" / "heldout_predictions.csv")
    eb_diag.add_argument("--turns-csv", type=Path, default=DATA_DIR / "diagnostics" / "cached_annotator" / "turns.csv")
    eb_diag.add_argument("--traces-csv", type=Path, default=DATA_DIR / "diagnostics" / "cached_annotator" / "traces.csv")
    eb_diag.add_argument(
        "--conditional-cohorts-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "cached_annotator_diagnostic" / "conditional_prefix_cohorts.csv",
    )
    eb_diag.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "empirical_bayes_v1")
    eb_diag.add_argument("--bundle-path", type=Path, default=DATA_DIR / "estimators" / "empirical_bayes_v1" / "lookup.json")

    eb_query = subparsers.add_parser("empirical-bayes-query", help="query a saved empirical-Bayes lookup")
    eb_query.add_argument("--bundle-path", type=Path, default=DATA_DIR / "estimators" / "empirical_bayes_v1" / "lookup.json")
    eb_query.add_argument("--source", required=True)
    eb_query.add_argument("--total", type=int, required=True)
    eb_query.add_argument("--current-category", default="NONE")
    eb_query.add_argument("--step", type=int, required=True)
    eb_query.add_argument("--current-unit-age", type=int, required=True)
    eb_query.add_argument("--had-stuck-episode", action="store_true")

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
    if args.command == "cached-annotator-diagnostic":
        result = _cached_annotator_diagnostic(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "empirical-bayes-eval":
        result = evaluate(
            args.turns_csv,
            args.traces_csv,
            args.report_dir,
            args.bundle_path,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            min_support=args.min_support,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "empirical-bayes-query":
        lookup = EmpiricalBayesLookup.load(args.bundle_path)
        result = query_json(
            lookup,
            source=args.source,
            total=args.total,
            current_category=args.current_category,
            step=args.step,
            current_unit_age=args.current_unit_age,
            had_stuck_episode=args.had_stuck_episode,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "empirical-bayes-diagnostics":
        result = write_diagnostics(
            args.heldout_csv,
            args.turns_csv,
            args.traces_csv,
            args.conditional_cohorts_csv,
            args.report_dir,
            args.bundle_path,
        )
        print(json.dumps(result, sort_keys=True))
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


TRACE_FIELDS = [
    "trace_key",
    "source",
    "raw_row_index",
    "instance_id",
    "model",
    "agent",
    "success_or_reward",
    "exit_status",
    "task",
    "category",
    "subcategory",
    "final_total",
    "final_done",
    "had_stuck_episode",
    "total_turns",
    "action_turns",
    "terminal_category",
    "first_stuck_step",
    "censored_right_tail",
    "censor_reason",
    "parse_error",
]


def _cached_annotator_diagnostic(args: argparse.Namespace) -> dict[str, int]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_count = 0
    turn_count = 0
    with (args.out_dir / "turns.csv").open("w", encoding="utf-8", newline="") as turns_handle, (
        args.out_dir / "traces.csv"
    ).open("w", encoding="utf-8", newline="") as traces_handle:
        turn_writer = csv.DictWriter(turns_handle, fieldnames=["trace_key", "source", "instance_id", *ROW_FIELDS])
        trace_writer = csv.DictWriter(traces_handle, fieldnames=TRACE_FIELDS)
        turn_writer.writeheader()
        trace_writer.writeheader()
        for source in expand_sources(args.source):
            raw_path = args.raw_dir / source / f"{args.split}.jsonl"
            for raw_index, raw in enumerate(read_jsonl(raw_path)):
                trace_count += 1
                try:
                    [(instance_id, turns)] = list(rows_to_turns([raw], source=source))
                    trace_key = f"{source}:{raw_index:06d}:{instance_id}"
                    rows, summary = annotate_turns(turns, instance_id=instance_id, exit_status=str(turns[0].metadata.get("exit_status", "unknown")))
                    for row in rows:
                        turn_writer.writerow({"trace_key": trace_key, "source": source, "instance_id": instance_id} | row.to_csv_row())
                    turn_count += len(rows)
                    trace_writer.writerow(_trace_row(trace_key, source, raw_index, instance_id, raw, rows, summary))
                except Exception as exc:
                    instance_id = str(raw.get("instance_id") or raw.get("task_name") or f"{source}-{raw_index:06d}")
                    trace_key = f"{source}:{raw_index:06d}:{instance_id}"
                    trace_writer.writerow(_parse_error_trace_row(trace_key, source, raw_index, instance_id, raw, exc))
    return {"traces": trace_count, "turns": turn_count}


def _trace_row(trace_key: str, source: str, raw_index: int, instance_id: str, raw: dict, rows: list, summary: object) -> dict[str, object]:
    terminal_category = next((row.current_category for row in reversed(rows) if row.current_category), "")
    first_stuck_step = next((row.step for row in rows if row.had_stuck_episode), "")
    censored = source == "swe-agent" and summary.final_total > 103
    return {
        "trace_key": trace_key,
        "source": source,
        "raw_row_index": raw_index,
        "instance_id": instance_id,
        "model": raw.get("model", ""),
        "agent": raw.get("agent", ""),
        "success_or_reward": raw.get("success", raw.get("reward", "")),
        "exit_status": summary.exit_status,
        "task": raw.get("task", raw.get("task_name", "")),
        "category": raw.get("category", ""),
        "subcategory": raw.get("subcategory", ""),
        "final_total": summary.final_total,
        "final_done": summary.final_done,
        "had_stuck_episode": summary.had_stuck_episode,
        "total_turns": rows[-1].step if rows else 0,
        "action_turns": sum(1 for row in rows if row.kind == "action"),
        "terminal_category": terminal_category,
        "first_stuck_step": first_stuck_step,
        "censored_right_tail": censored,
        "censor_reason": "swe_agent_final_total_gt_103" if censored else "",
        "parse_error": "",
    }


def _parse_error_trace_row(trace_key: str, source: str, raw_index: int, instance_id: str, raw: dict, exc: Exception) -> dict[str, object]:
    return {
        "trace_key": trace_key,
        "source": source,
        "raw_row_index": raw_index,
        "instance_id": instance_id,
        "model": raw.get("model", ""),
        "agent": raw.get("agent", ""),
        "success_or_reward": raw.get("success", raw.get("reward", "")),
        "exit_status": raw.get("exit_status", "unknown"),
        "task": raw.get("task", raw.get("task_name", "")),
        "category": raw.get("category", ""),
        "subcategory": raw.get("subcategory", ""),
        "final_total": 0,
        "final_done": 0,
        "had_stuck_episode": False,
        "total_turns": 0,
        "action_turns": 0,
        "terminal_category": "",
        "first_stuck_step": "",
        "censored_right_tail": False,
        "censor_reason": "",
        "parse_error": str(exc),
    }


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean[:180] or "trace"


if __name__ == "__main__":
    raise SystemExit(main())
