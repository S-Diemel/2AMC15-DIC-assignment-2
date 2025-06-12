import numpy as np
from .action_mapping import orientation_to_directions


def compute_distance_to_wall(orientation, max_range, agent_radius, agent_pos, width, height, all_obstacles):
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
#
#
# def item_sensor(orientation, max_range, agent_radius, agent_pos, item_starts, item_radius, delivered, carrying):
#     max_distance = max_range + agent_radius
#     direction = np.array(orientation_to_directions(orientation))
#
#     valid_items = [
#         (i, pos) for i, pos in enumerate(item_starts)
#         if not delivered[i] and carrying != i
#     ]
#     if not valid_items:
#         return max_range
#
#     # Closure to bind `collision_item`
#     collision_item = {'pos': None}
#
#     def is_collision(pos):
#         for _, (x, y) in valid_items:
#             if (x - item_radius) <= pos[0] <= (x + item_radius) and \
#                (y - item_radius) <= pos[1] <= (y + item_radius):
#                 collision_item['pos'] = (x, y)
#                 return True
#         return False
#
#     approx_d = _ray_march_collision(direction, agent_pos, max_distance, step_size=0.5, is_collision_fn=is_collision)
#     if approx_d is None:
#         return max_range
#
#     # Use only the closest item's refined bounds
#     x, y = collision_item['pos']
#     def is_collision_single_item(pos):
#         return (x - item_radius) <= pos[0] <= (x + item_radius) and \
#                (y - item_radius) <= pos[1] <= (y + item_radius)
#
#     refined_d = _binary_search_collision(direction, agent_pos, max(0, approx_d - 0.5), approx_d, is_collision_fn=is_collision_single_item)
#     return max(refined_d - agent_radius, 0)


def compute_dist_to_target(agent_pos, battery, battery_value_reward_charging, charger_center, 
                           carrying, delivery_points, item_spawn_center):
    """
    Compute distance between the agent and the next target on the x-axis and the y-axis. Note that when an item is carrying an item/package the target is the delivery point 
    for this package, but when the agent is not carrying any items the target will be the center of item/package pickup area. This is because we want the agent
    to find packages, and not have the direct location. Once it finds a package, it scans the package and knows where this package should be delivered. 
    """
    target_x, target_y = compute_target(battery, battery_value_reward_charging, charger_center, 
                                        carrying, delivery_points, item_spawn_center)
    x, y = agent_pos
    dist_target_x = target_x - x
    dist_target_y = target_y - y
    return dist_target_x, dist_target_y  # return distance on x and y axis to target


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

def point_to_rectangle_distance(px, py, rect):
    xmin, ymin, xmax, ymax = rect

    # Clamp point to rectangle bounds
    closest_x = max(xmin, min(px, xmax))
    closest_y = max(ymin, min(py, ymax))

    # Compute Euclidean distance from point to closest point on rectangle
    dx = px - closest_x
    dy = py - closest_y
    return np.sqrt((dx)**2 + (dy)**2)

def find_closest_rectangle_to_edge(rectangles, point):
    px, py = point
    min_distance = float('inf')
    closest_index = -1

    for i, rect in enumerate(rectangles):
        dist = point_to_rectangle_distance(px, py, rect)
        if dist < min_distance:
            min_distance = dist
            closest_index = i

    return closest_index+1

def compute_area_code(carrying, delivery_points, item_spawn_center, aisles):

    target_x, target_y = compute_target(carrying, delivery_points, item_spawn_center)

    if carrying>=0:
        area = find_closest_rectangle_to_edge(aisles, (target_x, target_y))
        return area
    else:
        return 0


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

def _calc_signed_angle(a, b):
    dot = np.dot(a, b)
    cross = a[0]*b[1] - a[1]*b[0]  # 2D scalar cross product
    angle_rad = np.arctan2(cross, dot)  # signed angle in radians
    angle_deg = np.degrees(angle_rad)
    return -angle_deg

def _calc_properties_item(max_range, agent_radius, agent_pos, all_obstacles, item_pos, orientation):
    x, y = item_pos
    x_agent, y_agent = agent_pos
    item_distance = np.sqrt((x - x_agent) ** 2 + (y - y_agent) ** 2) - agent_radius

    max_distance = max_range + agent_radius
    item_direction = np.array((item_pos[0]-agent_pos[0], item_pos[1]-agent_pos[1]))
    item_direction = item_direction / np.linalg.norm(item_direction)
    agent_direction = orientation_to_directions(orientation)
    agent_direction = agent_direction / np.linalg.norm(agent_direction)
    angle = _calc_signed_angle(agent_direction, item_direction)

    def is_collision(pos):
        for (xmin, ymin, xmax, ymax) in all_obstacles:
            if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax:
                return True
        return False

    approx_d = _ray_march_collision(item_direction, agent_pos, max_distance, step_size=0.5, is_collision_fn=is_collision)
    if approx_d is None:
        return item_distance, max_range, angle

    refined_d = _binary_search_collision(item_direction, agent_pos, max(0, approx_d - 0.5), approx_d, is_collision_fn=is_collision)
    obstacle_dist = min(max(refined_d - agent_radius, 0), max_range)
    return item_distance, obstacle_dist, angle

def is_point_in_triangle(point, triangle):
    """
    Return True if pt lies inside or on the edge of triangle tri = (A, B, C).
    """
    A, B, C = triangle

    # Compute signed areas (cross-products) of the three sub-triangles
    d1 = np.cross(point - A, B - A)
    d2 = np.cross(point - B, C - B)
    d3 = np.cross(point - C, A - C)

    # Point is inside if all have the same sign (or zero)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)



def calc_vision_triangle_features(agent_pos, max_range, agent_radius, item_starts, delivered, carrying, vision_triangle, all_obstacles, delivery_points, orientation):
    """
    calc min distance to item in the vision triangle
    """
    angle = 0

    if carrying==-1:
        valid_items = [pos for i, pos in enumerate(item_starts) if not delivered[i] and carrying != i]
    elif carrying>=0:
        valid_items = [pos for i, pos in enumerate(delivery_points) if not delivered[i] and carrying == i]
    else:
        valid_items = []
    if not valid_items:
        return max_range, angle

    min_distance=max_range
    for item in valid_items:
        if is_point_in_triangle(item, vision_triangle):
            distance, obstacle_distance, angle = _calc_properties_item(max_range, agent_radius, agent_pos, all_obstacles, item, orientation)
            if distance < obstacle_distance:
                if distance < min_distance:
                    min_distance = distance
    return max(0, min_distance), angle

def calc_can_interact(agent_pos, agent_radius, items, item_radius, delivery_points, delivery_radius, delivered, carrying, charger):
    for i, (pos, delivered_status) in enumerate(zip(items, delivered)):
        # Iterate over all items that can be picked up, and make sure we have info on whether these items have been delivered yet.
        if not delivered_status and carrying==-1 and np.linalg.norm(agent_pos - pos) < agent_radius + item_radius:
            return 1

    for i, point in enumerate(delivery_points):
        if carrying == i and np.linalg.norm(agent_pos - point) < agent_radius + delivery_radius:
            return 1

    x, y = agent_pos
    xmin, ymin, xmax, ymax = charger
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return 1

    return 0














