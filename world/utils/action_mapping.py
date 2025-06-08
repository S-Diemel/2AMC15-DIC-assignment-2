def action_to_values(action):
    """
    Define what the action integers mean in terms of speed and orientation change. First index of tuple indicates speed, 
    second index indicates change in orientation in 45 degree increments.
    """
    values = {
        0: (1, 0),      # go
        1: (0, -2),     # left 90 
        2: (0, -1),     # left 45 
        3: (0, 1),      # right 45
        4: (0, 2),      # right 90
        5: (0, 0)       # pickup/drop
    }
    return values[action]


def orientation_to_directions(orientation):
    """What does the orientation in degree mean in terms of x and y direction of a one-unit step."""
    directions = {
        0: (0, 1),         # Up
        45: (1, 1),        # Up-Right
        90: (1, 0),        # Right
        135: (1, -1),      # Down-Right
        180: (0, -1),      # Down
        225: (-1, -1),     # Down-Left
        270: (-1, 0),      # Left
        315: (-1, 1),      # Up-Left
    }
    return directions[orientation]