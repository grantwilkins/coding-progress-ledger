from typing import List


def solve(input_grid: List[List[int]]) -> List[List[int]]:
    # Tile the 2x2 input grid 3 times in each dimension -> 6x6 output.
    return [row_tile * 3 for row_tile in ([input_grid[r % 2] for r in range(2)] * 3)]
