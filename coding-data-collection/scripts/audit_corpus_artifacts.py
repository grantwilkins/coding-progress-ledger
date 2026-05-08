from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.audits import corpus_hardening_report, write_report


def _run_dirs_from_args(paths: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for text in paths:
        path = Path(text)
        if (path / "run_manifest.json").is_file():
            run_dirs.append(path)
        else:
            run_dirs.extend(sorted(child for child in path.iterdir() if child.is_dir()))
    return run_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit corpus artifact completeness and redaction safety.")
    parser.add_argument("paths", nargs="+", help="Run directories or corpus directories.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = corpus_hardening_report(_run_dirs_from_args(args.paths))
    write_report(Path(args.out), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
