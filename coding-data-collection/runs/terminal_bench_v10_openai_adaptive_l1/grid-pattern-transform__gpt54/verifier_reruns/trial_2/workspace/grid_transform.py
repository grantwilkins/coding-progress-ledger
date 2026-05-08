def solve(input_grid):
    """Expand a 2x2 grid into the required 6x6 alternating pattern."""
    if len(input_grid) != 2 or any(len(row) != 2 for row in input_grid):
        raise ValueError("input_grid must be a 2x2 grid")

    a, b = input_grid[0]
    c, d = input_grid[1]

    return [
        [a, b, a, b, a, b],
        [c, d, c, d, c, d],
        [b, a, b, a, b, a],
        [d, c, d, c, d, c],
        [a, b, a, b, a, b],
        [c, d, c, d, c, d],
    ]
