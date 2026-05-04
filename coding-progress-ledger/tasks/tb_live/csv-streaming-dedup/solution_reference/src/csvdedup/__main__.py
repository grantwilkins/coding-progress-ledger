import csv
import sys
from csvdedup import dedup


def main():
    if len(sys.argv) > 1:
        f = open(sys.argv[1], newline="")
    else:
        f = sys.stdin

    out = sys.stdout
    reader = csv.reader(f)
    writer = csv.writer(out, lineterminator="\n")
    dedup(reader, writer)

    if len(sys.argv) > 1:
        f.close()


main()
