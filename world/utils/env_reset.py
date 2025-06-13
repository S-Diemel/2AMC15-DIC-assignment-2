import numpy as np
from .action_mapping import orientation_to_directions

def _point_in_rectangle(x, y, rect):
    """Checks if a point occurs in a single rectangle or not."""
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def is_inside_set_of_rectangles(x, y, set_of_rectangles):
    """Checks whether a point (x, y) occurs within a set of rectangles and returns True if it occurs in at least one of the rectangles"""
    for rect in set_of_rectangles:
        if _point_in_rectangle(x, y, rect):
            return True
    return False


def sample_points_in_rectangles(rectangles, number_of_items, radius, obstacles, difficulty_region=None):
    """
    Sample package points to spawn packages in package pickup area, and delivery points in drop-off areas around storage racks. So we provide
    rectangles to sample within. Then to determine whether a sampled location is valid we check three conditions:

    1. If a difficulty region is given, the point must be within this difficulty region
    2. The point cannot be within an obstacle, but it can overlap with obstacles as long as the agent can reach the item or delivery point
    3. The point cannot overlap with each other. The reason for this is that it negatively impacts the rendering (GUI)
    
    The optional difficulty region allows us to sample only points that are within a certain difficulty region of the
    warehouse. The easier the difficulty region, the closer the delivery points will be to the package spawn area. 
    """
    points = []
    min_dist = 2 * radius  # between points

    for _ in range(number_of_items):
        while True:
            # pick a random rect and uniform point inside it (with `radius` margin)
            rect = rectangles[np.random.randint(len(rectangles))]
            x = np.random.uniform(rect[0] + radius, rect[2] - radius)
            y = np.random.uniform(rect[1] + radius, rect[3] - radius)

            # Three conditions for a sampled point to be valid
            # If difficulty_region is given, the sampled data point must be within it
            if difficulty_region is not None:
                if not _point_in_rectangle(x, y, difficulty_region):
                    continue

            # The sampled data point cannot be inside an obstacle (but it can overlap) as long as agent can reach it
            if is_inside_set_of_rectangles(x, y, obstacles):
                continue

            # The sampled data point cannot overlap with another already sampled datapoint, as this negatively impacts
            # the rendering (GUI).
            too_close = False
            for (px, py) in points:
                if np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue

            # Append the point if all conditions are satisfied, and the point is thus valid
            points.append((x, y))
            break

    return points


def sample_one_point_outside(rectangles, radius, bounding_rect, difficulty_region=None):
    """
    Sample agents starting position, that it not too close to any of the obstacles. 
    Gives a set of rectangles, a distance around these rectangles (radius), and a total bounding rectangle within which to sample.
    Furthermore, the optional difficulty regrion allows us to sample only points that are within a certain difficulty region of the
    warehouse. The easier the difficulty region, the closer the agents spawn will be to the package spawn area. 
    Detail: the agent can spawn in forbidden zones to make it learn to leave them quickly, and it can spawn in the package spawn area.
    """
    xmin_b, ymin_b, xmax_b, ymax_b = bounding_rect

    # Pre‐compute the 'inflated' rectangles
    inflated = []
    for (xmin, ymin, xmax, ymax) in rectangles:
        inflated.append((xmin - radius, ymin - radius, xmax + radius, ymax + radius))

    # Keep sampling until we find a point that is not inside any of the inflated rectangles
    while True:
        x_cand = np.random.uniform(xmin_b, xmax_b)
        y_cand = np.random.uniform(ymin_b, ymax_b)
        if difficulty_region is not None:
            if not is_inside_set_of_rectangles(x_cand, y_cand, inflated) and _point_in_rectangle(x_cand, y_cand, difficulty_region):
                return (x_cand, y_cand)
        else:
            if not is_inside_set_of_rectangles(x_cand, y_cand, inflated):
                return (x_cand, y_cand)
            

def set_difficulty_of_env(item_spawn, width, height, difficulty=None):
        """
        For CURRICULUM LEARNING: define invisible easy, medium and hard zones within the environment,
        which divide the warehouse to the right of the package spawn place into 3 evenly sized sections.
        The agent and the delivery points will exclusively be sampled according to the selected difficulty level. 
        """
        # Only values None, 0, 1, 2 are accepted the rest is not accepted
        assert difficulty in [None, 0, 1, 2], f"Only values None (no level), 0 (easy), 1 (medium), 2 (hard) are accepted the rest is not accepted. not: {difficulty}"
        item_spawn_width = item_spawn[0][2]
        width_difficulty_region = (width - item_spawn_width) / 3
        # Create difficulty region
        difficulty_region = (item_spawn_width + difficulty * width_difficulty_region, 
                             0, 
                             item_spawn_width + (difficulty + 1) * width_difficulty_region, 
                             height) if difficulty is not None else None
        return difficulty_region

def calc_vision_triangle(agent_pos, orientation, max_range, agent_radius):
    max_range=3
    """Calc the corners of the vision triangle"""
    left_side_triangle = (orientation-45)%360
    right_side_triangle = (orientation+45)%360
    left_direction = np.array(orientation_to_directions(left_side_triangle))
    right_direction = np.array(orientation_to_directions(right_side_triangle))
    left_direction  = left_direction / np.linalg.norm(left_direction)
    right_direction = right_direction / np.linalg.norm(right_direction)
    left_point = agent_pos + left_direction*(max_range + agent_radius)
    right_point = agent_pos + right_direction*(max_range + agent_radius)
    return agent_pos, left_point, right_point