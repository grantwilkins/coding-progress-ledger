#!/usr/bin/env bash
set -euo pipefail
cat > pipeline.py <<'PY'
import csv, sys


def parse_orders(path):
    with open(path, newline="") as f:
        return [
            {"id": row["id"], "qty": int(row["qty"]),
             "unit_price": float(row["unit_price"])}
            for row in csv.DictReader(f)
        ]


def total_revenue(orders):
    return float(sum(o["qty"] * o["unit_price"] for o in orders))


if __name__ == "__main__":
    print(f"total_revenue={total_revenue(parse_orders(sys.argv[1]))}")
PY
