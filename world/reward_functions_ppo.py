import numpy as np


def default_reward_function(pickup, delivered, collided, charged, old_pos, agent_pos, agent_radius, forbidden_zones):
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
    if charged:  # charging when below certain battery value
        reward += 1
    if pickup:  # picking up an item
        reward += 250
    if delivered:  # delivering an item
        reward += 1000
    if collided:  # colliding with a wall or object
        reward -= 1
    if _agent_in_forbidden_zone(agent_pos, agent_radius, forbidden_zones):  # being in a forbidden zone
        reward -= 1
    return reward


# def shaping_reward(old_pos, old_target, agent_pos):
#     """Potential based shaping of the reward inspired by (g, Harada, & Russell, 1999)"""
#     gamma = 0.99  # gamma value we use
#
#     # Use Manhatten distance when you do no allow diagonal moves:
#     # old_distance_to_target = -(abs(old_pos[0]-old_target[0]) + abs(old_pos[1]-old_target[1])) # negative manhattan distance
#     # new_distance_to_target = -(abs(self.agent_pos[0]-old_target[0]) + abs(self.agent_pos[1]-old_target[1]))  # negative manhattan distance
#
#     # Use Chebyshev distance when you do allow diagonal moves which are equivalent in number of steps as a 'straight' move:
#     old_distance_to_target = -max(abs(old_pos[0] - old_target[0]), abs(old_pos[1] - old_target[1]))
#     new_distance_to_target = -max(abs(agent_pos[0] - old_target[0]), abs(agent_pos[1] - old_target[1]))
#
#     shaping_reward = gamma*new_distance_to_target - old_distance_to_target
#     return shaping_reward


def _agent_in_forbidden_zone(agent_pos, agent_radius, forbidden_zones):
    """Check if the agent is in one of the forbidden zones to properly assign a negative reward to this."""
    in_forbidden_zone = False
    x, y = agent_pos
    r = agent_radius
    for xmin, ymin, xmax, ymax in forbidden_zones:  # Iterate over forbidden zones
        if (x + r > xmin and x - r < xmax and y + r > ymin and y - r < ymax):
            in_forbidden_zone = True
    return in_forbidden_zone  # If not in a forbidden zone then return false