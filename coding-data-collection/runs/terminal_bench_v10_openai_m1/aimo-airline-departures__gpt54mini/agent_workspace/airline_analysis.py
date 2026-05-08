def max_consecutive_no_flight_days() -> int:
    # All departure periods are multiples of 10, so flights can only occur on
    # up to three residue classes modulo 10. To minimize the longest run of
    # no-flight days, place these residues as evenly as possible around the
    # 10-day cycle: gaps 2, 2, and 3. Thus the guaranteed maximum gap is 3.
    return 3


if __name__ == "__main__":
    answer = max_consecutive_no_flight_days()
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(str(answer))
    print(answer)
