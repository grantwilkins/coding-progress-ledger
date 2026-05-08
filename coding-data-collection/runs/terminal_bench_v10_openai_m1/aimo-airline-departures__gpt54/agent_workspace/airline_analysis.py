from math import gcd
from functools import reduce

PERIODS = (100, 120, 150)


def lcm(a, b):
    return a * b // gcd(a, b)


def lcm_many(values):
    return reduce(lcm, values)


def circular_gaps(days, cycle_length):
    ordered = sorted(set(days))
    if not ordered:
        return [cycle_length]
    gaps = []
    for i in range(len(ordered)):
        current = ordered[i]
        nxt = ordered[(i + 1) % len(ordered)]
        if i + 1 < len(ordered):
            gaps.append(nxt - current - 1)
        else:
            gaps.append((cycle_length - current) + nxt - 1)
    return gaps


def best_gap_for_offsets(offsets):
    cycle = lcm_many(PERIODS)
    departures = set()
    for period, offset in zip(PERIODS, offsets):
        departures.update(range(offset % period, cycle, period))
    return max(circular_gaps(departures, cycle))


def compute_guaranteed_gap():
    worst_best_gap = None
    worst_offsets = None
    for a in range(PERIODS[0]):
        for b in range(PERIODS[1]):
            for c in range(PERIODS[2]):
                gap = best_gap_for_offsets((a, b, c))
                if worst_best_gap is None or gap < worst_best_gap:
                    worst_best_gap = gap
                    worst_offsets = (a, b, c)
    return worst_best_gap, worst_offsets


def main():
    answer, _ = compute_guaranteed_gap()
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(str(answer))


if __name__ == "__main__":
    main()
