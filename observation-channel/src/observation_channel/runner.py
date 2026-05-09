from __future__ import annotations

from pathlib import Path

from .annotator import Annotator
from .io import read_turns, write_rows_csv, write_summaries_csv
from .models import Row, Summary, Turn


def annotate_turns(turns: list[Turn], *, instance_id: str = "", exit_status: str = "unknown") -> tuple[list[Row], Summary]:
    annotator = Annotator(instance_id=instance_id, exit_status=exit_status)
    rows = [annotator.feed(turn) for turn in turns]
    return rows, annotator.finalize()


def annotate_file(path: Path, out_dir: Path) -> Summary:
    turns = read_turns(path)
    instance_id = turns[0].instance_id if turns else path.stem
    exit_status = str(turns[0].metadata.get("exit_status", "unknown")) if turns else "unknown"
    rows, summary = annotate_turns(turns, instance_id=instance_id or path.stem, exit_status=exit_status)
    write_rows_csv(out_dir / f"{path.stem}.csv", rows)
    write_summaries_csv(out_dir / f"{path.stem}.summary.csv", [summary])
    return summary


def annotate_corpus(turn_dir: Path, out_dir: Path) -> list[Summary]:
    summaries: list[Summary] = []
    for path in sorted(turn_dir.glob("*.jsonl")):
        summaries.append(annotate_file(path, out_dir))
    write_summaries_csv(out_dir / "summaries.csv", summaries)
    return summaries
