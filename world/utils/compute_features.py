import numpy as np
from .action_mapping import orientation_to_directions


def compute_distance_to_wall(orientation, max_range, agent_radius, agent_pos, width, height, all_obstacles):
    max_distance = max_range + agent_radius
    direction = np.array(orientation_to_directions(orientation))
    
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
    return max(refined_d - agent_radius, 0)


def item_sensor(orientation, max_range, agent_radius, agent_pos, item_starts, item_radius, delivered, carrying):
    max_distance = max_range + agent_radius
    direction = np.array(orientation_to_directions(orientation))
    
    valid_items = [
        (i, pos) for i, pos in enumerate(item_starts)
        if not delivered[i] and carrying != i
    ]
    if not valid_items:
        return max_range

    # Closure to bind `collision_item`
    collision_item = {'pos': None}

    def is_collision(pos):
        for _, (x, y) in valid_items:
            if (x - item_radius) <= pos[0] <= (x + item_radius) and \
               (y - item_radius) <= pos[1] <= (y + item_radius):
                collision_item['pos'] = (x, y)
                return True
        return False

    approx_d = _ray_march_collision(direction, agent_pos, max_distance, step_size=0.5, is_collision_fn=is_collision)
    if approx_d is None:
        return max_range

    # Use only the closest item's refined bounds
    x, y = collision_item['pos']
    def is_collision_single_item(pos):
        return (x - item_radius) <= pos[0] <= (x + item_radius) and \
               (y - item_radius) <= pos[1] <= (y + item_radius)

    refined_d = _binary_search_collision(direction, agent_pos, max(0, approx_d - 0.5), approx_d, is_collision_fn=is_collision_single_item)
    return max(refined_d - agent_radius, 0)


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


def compute_target(battery, battery_value_reward_charging, charger_center, carrying, delivery_points, item_spawn_center):
    """
    When an item is carrying an item/package the target is the delivery point for this package, but when the agent is not carrying any items the target 
    will be the center of item/package pickup area. This is because we want the agent to find packages, and not have the direct location. Once it finds 
    a package, it scans the package and knows where this package should be delivered.
    """
    if battery < battery_value_reward_charging:
        target_x, target_y = charger_center
    else:
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
