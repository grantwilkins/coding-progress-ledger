def solve(input_grid):
    """Transform a 2x2 input grid into a 6x6 patterned output grid."""
    if not input_grid or len(input_grid) != 2 or any(len(row) != 2 for row in input_grid):
        raise ValueError("solve expects a 2x2 grid")

    original = [row[:] for row in input_grid]
    mirrored = [row[::-1] for row in input_grid]

    output_grid = []
    for block_row in range(3):
        tile = original if block_row % 2 == 0 else mirrored
        for row in tile:
            output_grid.append(row * 3)

    return output_grid
