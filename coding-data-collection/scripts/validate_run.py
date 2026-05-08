from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.validation import validate_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a protocol-shaped run directory.")
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)

    issues = validate_run_dir(Path(args.run_dir))
    if issues:
        for issue in issues:
            print(f"{issue.artifact}: {issue.message}", file=sys.stderr)
        return 2
    print(f"{args.run_dir}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
