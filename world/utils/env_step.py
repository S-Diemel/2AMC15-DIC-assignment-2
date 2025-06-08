import numpy as np
from .action_mapping import action_to_values, orientation_to_directions


def calc_new_position(action, speed, orientation, agent_angle, agent_pos, step_size):
    """Calculate the new position and orientation of the agent within the environment."""
    # TODO: remove speed, no longer used
    new_speed, sign_orientation = action_to_values(action)

    if speed == 0 and action==0:
        new_orientation = (orientation + sign_orientation*agent_angle) % 360
        direction = orientation_to_directions(orientation)
        new_position = np.array([agent_pos[0] + step_size*direction[0], agent_pos[1] + step_size*direction[1]])
        return new_orientation, new_position
    else:
        new_orientation = (orientation + sign_orientation*agent_angle) % 360
        new_position = agent_pos
        return new_orientation, new_position


def calc_collision(old_pos, new_position, agent_radius, width, height, all_obstacles):
    """Compute if any collisions happened with walls or obstacles."""
    new_pos = new_position.copy()
    collided = False

    # Wall collisions
    if new_pos[0] - agent_radius < 0:
        new_pos[0] = agent_radius
        collided = True
    if new_pos[0] + agent_radius > width:
        new_pos[0] = width - agent_radius
        collided = True
    if new_pos[1] - agent_radius < 0:
        new_pos[1] = agent_radius
        collided = True
    if new_pos[1] + agent_radius > height:
        new_pos[1] = height - agent_radius
        collided = True

    # Obstacle collisions
    for (xmin, ymin, xmax, ymax) in all_obstacles:
        closest = np.clip(new_pos, [xmin, ymin], [xmax, ymax])
        delta = new_pos - closest
        dist = np.linalg.norm(delta)
        if dist < agent_radius:
            collided = True
            overlap = agent_radius - dist
            if dist > 1e-10:
                new_pos += (delta / dist) * overlap
            else:
                new_pos = old_pos.copy()
            break

    return new_pos, collided


def update_delivery(action, carrying, speed, items, delivered, agent_pos, agent_radius, item_radius, delivery_points, delivery_radius):
    """
    Update all agent attribute regarding the delivery. Whether the agent is carrying an item, and whether each item is delivered. 
    Furthermore, for a step it saves whether an item was picked up or delivered, which is important for computing the reward.
    """
    item_delivered = False
    item_picked_up=False
    if action == 5 and carrying == -1 and speed == 0:  
        # If we are performing the pickup action, we are not yet carrying any item and we are standing still we can pick up an item.
        for i, (pos, delivered_status) in enumerate(zip(items, delivered)):  
            # Iterate over all items that can be picked up, and make sure we have info on whether these items have been delivered yet.
            if not delivered_status and np.linalg.norm(agent_pos - pos) < agent_radius + item_radius:  
                # if item is not yet delivered and the radius of the agent and the item are overlapping, we can pick up the item
                carrying = i
                item_picked_up = True
                break

    if carrying != -1:  # If we are carrying an item then we should move the item with the agent
        items[carrying] = agent_pos.copy()

    # If we are carrying an item, we do pickup/dropoff action and we are standing still
    if carrying != -1 and action == 4 and speed == 0:
        for i, point in enumerate(delivery_points):  
            if carrying == i and np.linalg.norm(agent_pos - point) < agent_radius + delivery_radius:
                # Check if item that is being carried and its delivery point correspond
                # and check if the radius of agent and dropoff point overlap
                delivered[carrying] = True
                carrying = -1
                item_delivered = True
                break

    return carrying, items, delivered, item_picked_up, item_delivered


def update_battery(battery, battery_drain_per_step, agent_pos, charger, speed, battery_value_reward_charging):
    """All logic for reducing battery level during steps, recharging by standing still in the charging area, and rewarding charging at low battery level"""
    old_battery = battery
    battery -= battery_drain_per_step  # Decrease the battery of the agent at each timestep
    x, y = agent_pos
    xmin, ymin, xmax, ymax = charger
    if xmin <= x <= xmax and ymin <= y <= ymax and speed == 0:  # if robot stands still in charging stop the battery is full again.
        battery = 100
        if old_battery <= battery_value_reward_charging:  # only reward charging if battery was actually low
            return battery, True
    return battery, False
