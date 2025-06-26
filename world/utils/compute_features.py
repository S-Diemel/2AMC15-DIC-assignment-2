import numpy as np
from .action_mapping import orientation_to_directions


def compute_distance_to_obstacle(orientation, max_range, agent_radius, agent_pos, width, height, all_obstacles):
    """
    simulates a lidar sensor and thus calcultes the distance to an obstacle/wall
    """
    max_distance = max_range + agent_radius
    direction = np.array(orientation_to_directions(orientation))
    direction = direction / np.linalg.norm(direction)

    def is_collision(pos):
        if not (0 <= pos[0] <= width and 0 <= pos[1] <= height):
            return True
        for (xmin, ymin, xmax, ymax) in all_obstacles:
            if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
                return True
        return False

    approx_d = _ray_march_collision(direction, agent_pos, max_distance, step_size=0.5, is_collision_fn=is_collision)
    if approx_d is None:
        return max_range

    refined_d = _binary_search_collision(direction, agent_pos, max(0, approx_d - 0.5), approx_d, is_collision_fn=is_collision)
    return min(max(refined_d - agent_radius, 0), max_range)


def compute_target(carrying, delivery_points, item_spawn_center):
    """
    When an item is carrying an item/package the target is the delivery point for this package, but when the agent is not carrying any items the target 
    will be the center of item/package pickup area. This is because we want the agent to find packages, and not have the direct location. Once it finds 
    a package, it scans the package and knows where this package should be delivered.
    """

    if carrying >= 0:
        target_x, target_y = delivery_points[carrying]  # Location of delivery point of the item that the agent is carrying
    else:
        target_x, target_y = item_spawn_center  # Center of the area where items spawn
    return target_x, target_y


def _ray_march_collision(direction, origin, max_distance, step_size, is_collision_fn):
    """
    Moves from origin in given direction, step by step, and returns distance to first collision.
    """
    steps = int(np.ceil(max_distance / step_size))

    for i in range(steps + 1):
        d = i * step_size
        test_pos = origin + direction * d
        if is_collision_fn(test_pos):
            return d
    return None


def _binary_search_collision(direction, origin, low, high, is_collision_fn, iterations=5):
    """Performs binary search between low and high distances to find the exact distance to the first collision."""
    for _ in range(iterations):
        mid = (low + high) / 2
        test_pos = origin + direction * mid
        if is_collision_fn(test_pos):
            high = mid
        else:
            low = mid
    return (low + high) / 2

def _calc_distances_item(max_range, agent_radius, agent_pos, all_obstacles, item_pos):
    """
    calculates the distance from agent to item and agent to obstacle in the same direction.
    With these distances it can be determined whether the vision on an item is blocked by an obstacle
    """
    x, y = item_pos
    x_agent, y_agent = agent_pos
    item_distance = np.sqrt((x - x_agent) ** 2 + (y - y_agent) ** 2) - agent_radius

    max_distance = max_range + agent_radius
    item_direction = np.array((item_pos[0]-agent_pos[0], item_pos[1]-agent_pos[1]))
    item_direction = item_direction / np.linalg.norm(item_direction)

    def is_collision(pos):
        for (xmin, ymin, xmax, ymax) in all_obstacles:
            if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
                return True
        return False

    approx_d = _ray_march_collision(item_direction, agent_pos, max_distance, step_size=0.5, is_collision_fn=is_collision)
    if approx_d is None:
        return item_distance, max_range

    refined_d = _binary_search_collision(item_direction, agent_pos, max(0, approx_d - 0.5), approx_d, is_collision_fn=is_collision)
    obstacle_dist = min(max(refined_d - agent_radius, 0), max_range)
    return item_distance, obstacle_dist

def is_point_in_triangle(point, triangle):
    """
    Return True if point lies inside or on the edge of triangle.
    """
    A, B, C = triangle

    # compute signed areas or cross-products of the three sub-triangles
    d1 = np.cross(point - A, B - A)
    d2 = np.cross(point - B, C - B)
    d3 = np.cross(point - C, A - C)

    # point is inside if all have the same sign
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)

def calc_barcode_sensor_features(agent_pos, agent_radius, item_starts, delivered, carrying, vision_triangle, all_obstacles, delivery_points):
    """
    returns a binary whether the agent can scan the correct barcode in its vision triangle.
    """
    max_range = np.sqrt(2)

    # check only items that are valid for the state of the agent (not delivered packages or deliver point of package carrying)
    if carrying == -1:
        valid_items = [pos for i, pos in enumerate(item_starts) if not delivered[i] and carrying != i]
    elif carrying >= 0:
        valid_items = [pos for i, pos in enumerate(delivery_points) if not delivered[i] and carrying == i]
    else:
        valid_items = []

    if not valid_items:
        return 0

    # calc whether an item in valid_items is in the vision triangle and whether it is not blocked by an obstacle.
    min_distance=max_range
    for item in valid_items:
        if is_point_in_triangle(item, vision_triangle):
            distance, obstacle_distance = _calc_distances_item(max_range, agent_radius, agent_pos, all_obstacles, item)
            if distance < obstacle_distance:
                if distance < min_distance:
                    return 1
    return 0















