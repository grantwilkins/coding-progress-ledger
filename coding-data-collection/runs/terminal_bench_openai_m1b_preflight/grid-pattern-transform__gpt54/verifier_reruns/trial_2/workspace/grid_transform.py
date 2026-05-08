def solve(input_grid):
    """Expand a 2x2 grid into a 6x6 patterned grid.

    The output repeats the input values across a 6x6 canvas. Every two output
    rows, the repeated 2-column pattern is shifted by one position, matching
    the example pattern in task.md.
    """
    if not input_grid or not input_grid[0]:
        return []

    height = len(input_grid)
    width = len(input_grid[0])

    if height != 2 or width != 2:
        raise ValueError("solve expects a 2x2 input grid")

    output_height = height * 3
    output_width = width * 3
    output_grid = []

    for r in range(output_height):
        row_source = input_grid[r % height]
        shift = (r // height) % width
        output_row = [row_source[(c + shift) % width] for c in range(output_width)]
        output_grid.append(output_row)

    return output_grid
