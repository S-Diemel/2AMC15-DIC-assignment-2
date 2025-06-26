import numpy as np


def charging_reward(battery_level, max_reward, battery_level_max_reward=15, min_battery_for_charging_reward=50, exp_factor=3.0):
    """
    Exponential reward for battery charging given some maximum battery level to receive a reward and
    some minimum battery level from which reward is big but does not change
    """
    if battery_level >= min_battery_for_charging_reward:
        return 0.0
    elif battery_level <= battery_level_max_reward:
        return max_reward
    else:
        # Scale battery to [0,1] between b_low and b_high
        x = (min_battery_for_charging_reward - battery_level_max_reward) / (min_battery_for_charging_reward - battery_level_max_reward)
        # Exponential mapping normalized to [0,1]
        return max_reward * (np.exp(exp_factor * x) - 1) / (np.exp(exp_factor) - 1)


def default_reward_function(pickup, delivered, collided, charged_battery_level, battery_died, old_pos, agent_pos, agent_radius, forbidden_zones, old_speed):
    """
    Reward function for the agent. It has the following rewards:
    - negative reward in general for taking a step (we want to obtain an optimal route and thus have a minimal number of steps)
    - bigger negative reward for staying in the same cell (we want to encourage movement)
    - positive reward for charging when battery is low (below som given value self.battery_value_reward_charging)
    - positive reward for pickup up an item/package (logically the agent should pick up items in order to deliver them)
    - positive reward for delivering an item (this is the main goal of the agent so this receives a high reward)
    - negative reward for colliding with walls or objects (this can be dangerous for the robot and general workplace safety)
    - negative reward for being in forbidden places (the agent should just not be in certain areas, altough it can physically move there)
    - the potential based reward shaping is applied outside this function and provides a reward for moving closer to the target
    """
    reward = -0.5
    if np.array_equal(old_pos, agent_pos):  # Punish agent for staying in the same position
        reward -= 0.5
    if charged_battery_level is not None:  # charging when below certain battery value
        reward += charging_reward(charged_battery_level, max_reward=5)
    if battery_died:
        reward -= 20
    if pickup:  # picking up an item
        reward += 5
    if delivered:  # delivering an item
        reward += 10
    if collided: # colliding with a wall or object
        collision_penalty = 2
        if old_speed > 0:
            reward -= collision_penalty*old_speed
        else:
            reward -= collision_penalty
    return reward



def shaping_reward(old_pos, old_target, agent_pos):
    """Potential based shaping of the reward inspired by (g, Harada, & Russell, 1999)"""
    gamma = 0.99  # gamma value we use

    # Use euclidean distance in a continuous space
    old_distance_to_target = -np.sqrt((old_pos[0] - old_target[0])**2 + (old_pos[1] - old_target[1])**2)
    new_distance_to_target = -np.sqrt((agent_pos[0] - old_target[0])**2 + (agent_pos[1] - old_target[1])**2)

    shaping_reward = gamma*new_distance_to_target - old_distance_to_target
    return shaping_reward

