def solve(input_grid):
    """Transform a 2x2 grid into a 6x6 patterned grid."""
    a, b = input_grid[0]
    c, d = input_grid[1]

    row1 = [a, b] * 3
    row2 = [c, d] * 3
    row3 = [b, a] * 3
    row4 = [d, c] * 3

    return [row1, row2, row3, row4, row1[:], row2[:]]
