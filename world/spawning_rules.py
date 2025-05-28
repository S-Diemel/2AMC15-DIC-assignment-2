"""
Function rules for spawning obstacles in the environment.
"""
def not_on_border(grid, pos):
    """
    Example rule for spawning obstacles. We could create many more or combine
    them to simulate different testing environments.

    Checks if the position is not on the border of the grid.
    Args:
        grid: The grid to check the position on.
        pos: The position to check.
    Returns:
        True if the position is not on the border, False otherwise.
    """
    x, y = pos
    return 1 < x < grid.shape[0]-2 and 1 < y < grid.shape[1]-2