import csv
import io


def dedup(reader, writer):
    seen = set()
    for row in reader:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            writer.writerow(row)
