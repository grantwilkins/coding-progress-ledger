from itertools import product


def max_no_flight_days(periods=(100, 120, 150)):
    """
    Return the greatest integer d such that, regardless of phase offsets,
    there are always at least d consecutive no-flight days.
    """
    p1, p2, p3 = periods
    lcm = 600  # lcm(100, 120, 150)

    best_guaranteed = float("inf")

    for a, b, c in product(range(p1), range(p2), range(p3)):
        flight_days = set()

        day = a
        while day < lcm + a:
            flight_days.add(day % lcm)
            day += p1

        day = b
        while day < lcm + b:
            flight_days.add(day % lcm)
            day += p2

        day = c
        while day < lcm + c:
            flight_days.add(day % lcm)
            day += p3

        days = sorted(flight_days)

        max_gap = 0
        for i in range(len(days)):
            cur = days[i]
            nxt = days[(i + 1) % len(days)]
            gap = (nxt - cur) % lcm
            if gap == 0:
                gap = lcm
            max_gap = max(max_gap, gap)

        no_flight_streak = max_gap - 1
        best_guaranteed = min(best_guaranteed, no_flight_streak)

    return int(best_guaranteed)


def main():
    answer = max_no_flight_days()
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(str(answer))


if __name__ == "__main__":
    main()
