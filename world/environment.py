import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt

from world.utils.env_reset import (
    sample_points_in_rectangles, 
    sample_one_point_outside, 
    set_difficulty_of_env,
    calc_vision_triangle
)
from world.utils.env_step import (
    calc_new_position,
    calc_collision,
    update_delivery,
    update_battery,
)
from world.utils.compute_features import (
    compute_distance_to_obstacle,
    compute_target,
    calc_vision_triangle_features,
)
from world.utils.env_init import (
    create_delivery_zones,
)
from .gui import render_gui
from .reward_functions import default_reward_function, shaping_reward


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        number_of_items=3,
        agent_radius=0.3,
        step_size=0.2,
        sigma=0,
        difficulty=None,  # Difficulty level represented by None (random no particular difficulty level), 0 (easy), 1 (medium), 2 (hard)  
        extra_obstacles=None,  # List of additional obstacles, if any --> useful for experimenting with humans or boxes in random places
    ):
        super().__init__()

        self.width = 15.0
        self.height = 10.0
        self.sigma = sigma 
        # Environment stochasticity interpreted as slippery-ness. The agent's current move is duplicated. 
        # Unless it has a collision with a wall or obstacle, in this case the action is not repeated.
        self.agent_radius = agent_radius
        self.step_size = step_size
        self.speed = 0
        self.orientation = 0
        self.agent_angle = 45  # the agents angle step-size
        self.max_range = 5 # Maximum range for sensors
        self.battery_drain_per_step = 0.25  # In reset, the battery is always reset to 100%
        self.battery_value_reward_charging = 50  # From which battery level to reward the agent for going to the charging station.

        # Define the layout based on the image
        width_spawn_area = 3
        self.item_spawn = [(0, 0, width_spawn_area, self.height)]  # Yellow Area: where packages spawn
        assert len(self.item_spawn) == 1, "The environment allows only a single item spawn, due to some inherent design choices and scope limitation." 

        half_width_of_rack = 0.5
        self.racks = [  # Blue Areas: storage racks
            # xmin, ymin, xmax, ymax
            # Storage racks around along the walls
            (5, self.height - half_width_of_rack, self.width, self.height),  # top
            (7, 0, self.width, half_width_of_rack),  # bottom
            (self.width-half_width_of_rack, half_width_of_rack, self.width, self.height-half_width_of_rack),  # right
            # First row of storage racks
            (5, self.height - 6 * half_width_of_rack, 13, self.height - 4 * half_width_of_rack),
            # Second row of storage racks
            (5, self.height - 11 * half_width_of_rack, 13, self.height - 9 * half_width_of_rack),
            # Third row of storage racks
            (5, self.height - 16 * half_width_of_rack, 13, self.height - 14 * half_width_of_rack),
        ]

        if extra_obstacles is not None: 
            self.extra_obstacles = extra_obstacles  # List of additional obstacles, if any
        else:
            self.extra_obstacles = []

        self.forbidden_zones = []  # self.forbidden_zones = [(14, self.height - 12 * half_width_of_rack, 17.5, self.height - 8 * half_width_of_rack)]  # Red Area: forbidden zones, where the agent can but should not go
        self.charger = (3.5, 0, 6, 1)  # Green Area: charging area
        self.charger_center = ((self.charger[0] + self.charger[2]) / 2, (self.charger[1] + self.charger[3]) / 2)

        # Create delivery zones (grey areas) around the racks
        self.delivery_zones = create_delivery_zones(self.racks, self.width, self.height, margin=0.5)

        # For CURRICULUM LEARNING: define the difficulty of the environment by constraining the agent starting position and delivery points
        self.difficulty = difficulty

        # Define all items (= packages), there spawn points and the delivery points
        self.number_of_items = number_of_items
        self.item_radius = agent_radius
        self.item_spawn_center = ((self.item_spawn[0][0] + self.item_spawn[0][2]) / 2, (self.item_spawn[0][1] + self.item_spawn[0][3]) / 2)
        self.delivery_radius = agent_radius
        # Both items and delivery points are linked by index, so item 0 is delivered at delivery point 0, etc.

        # Initialize some Gym environment paramters: Necessary for Gym-compatible trainers
        self.action_space = spaces.Discrete(5)
        # Give the range of values that the agents state space can take for each feature
        low = np.array([
            0.0,   # x / width
            0.0,   # y / height
            0.0,   # orientation / 45
            0.0,   # carrying flag
            0.0,   # steps_left / max_range
            0.0,   # steps_fw_left / max_range
            0.0,   # steps_fw / max_range
            0.0,   # steps_fw_right / max_range
            0.0,   # steps_right / max_range
            0.0,    # battery / 100
            0.0, #triangle_vision
            0.0, # speed
        ], dtype=np.float32)

        high = np.array([
            1.0,  # x / width
            1.0,  # y / height
            7.0,  # orientation / 45  (possible values: 0…7)
            1.0,  # carrying flag
            1.0,  # steps_left / max_range
            1.0,  # steps_fw_left / max_range
            1.0,  # steps_fw / max_range
            1.0,  # steps_fw_right / max_range
            1.0,  # steps_right / max_range
            1.0,   # battery / 100
            1.0, # triangle vision
            2.0, # speed
        ], dtype=np.float32)

        # Give possible values of observational space
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Maintaining statistics
        self.cumulative_reward = 0
        self.total_nr_collisions = 0
        self.total_nr_steps = 0

        # Call reset to finish initializing the environment
        self.reset()
    
    def reset(self, no_gui=True, seed=None, agent_start_pos=False, difficulty=None, extra_obstacles=None, number_of_items=None, battery_drain_per_step=None, options=None, difficulty_mode='eval'):
        """
        Resetting the environment for a new task for the agent. This involves spawning packages/items, delivery points, the agent itself. 
        It also involves initializing some attributes to the environment and the agent, such as: that it is not carrying any items/packages, 
        it has not delivered any packages yet, it is at full battery, etc. Finally it computes the initial agent state observation vector.
        """
        # parse options
        if options is not None:
            self.difficulty = options.get("difficulty", self.difficulty)
            if self.difficulty==3:
                self.difficulty=None
            self.number_of_items = options.get("number_of_items", self.number_of_items)
            self.battery_drain_per_step = options.get("battery_drain_per_step", self.battery_drain_per_step)
            difficulty_mode = options.get("difficulty_mode", difficulty_mode)

        if difficulty==3:
            difficulty=None
            self.difficulty=difficulty

        super().reset(seed=seed)
        info = {} # required for Gymnasium (parallel environments), but unused

        if extra_obstacles is not None:
            self.extra_obstacles = extra_obstacles
            # Implicit else: we keep the self.extra_obstacles from the initialization above, which is an empty list for None
        # Combine all obstacles for collision detection 
        self.all_obstacles = self.racks + self.extra_obstacles

        if difficulty is not None:
            self.difficulty = difficulty

        if number_of_items is not None:
            self.number_of_items = number_of_items

        if battery_drain_per_step is not None:
            self.battery_drain_per_step = battery_drain_per_step

        self.difficulty_region = set_difficulty_of_env(
            self.item_spawn, self.width, self.height, self.difficulty, difficulty_mode)  # For curriculum learning set the difficulty of the environment

        self.item_starts = sample_points_in_rectangles(
            self.item_spawn, self.number_of_items, self.item_radius, self.all_obstacles)  # spawn/initialize packages/items

        self.delivery_points = sample_points_in_rectangles(
            self.delivery_zones, self.number_of_items, self.delivery_radius, self.all_obstacles, self.difficulty_region)  # choose delivery spots

        if not agent_start_pos:  # randomly sample agent position if none is supplied.
            self.agent_pos = np.array(sample_one_point_outside(
                self.all_obstacles, self.agent_radius, (0, 0, self.width, self.height), self.difficulty_region))
        else:
            self.agent_pos = np.array(agent_start_pos)  # Use given starting position

        self.vision_triangle = calc_vision_triangle(self.agent_pos, self.orientation, self.agent_radius)
        self.items = [np.array(pos, dtype=np.float32) for pos in self.item_starts] 
        self.delivered = [False] * len(self.items)
        self.carrying = -1  # -1 = not carrying any items; otherwise this is the index of the item that is at that moment carried by the agent.
        self.battery = 100  # initializes battery
        self.no_gui = no_gui

        return self._compute_features(), info
    
    
    def step(self, action):
        """
        All logic that has to be executed upon performing a single time-step in the environment. For this, we use all the logic previously defined for the:
        - Agents old position
        - Agents new position (after with potential collisions)
        - Pick-up/drop-off logic for delivering and carrying items
        - Updates to battery life of the agent
        - Computing a reward for the step
        - Computing when an episode is finished --> if all items are delivered or the battery died
        - Computing new state (observational feature vector)
        """
        env_stochasticity = np.random.choice([True, False], p=[self.sigma, 1-self.sigma])  # determine whether to do apply environment stochasticity (slippery)
        # This means duplication the effect of the current action. Therefore the outcome is a more exeggarated situation of the inteded move. 
        info = {"env_stochasticity": env_stochasticity}  # we need to supply as step "info" parameter, we add information regarding stochasticity of the step

        # Check move for certainty
        assert self.action_space.contains(action)

        # Save old values
        old_pos = self.agent_pos.copy()
        old_target = compute_target(self.carrying, self.delivery_points, self.item_spawn_center)
        
        # Compute new position after action
        self.orientation, new_pos, self.speed = calc_new_position(action, self.speed, self.orientation, self.agent_angle, self.agent_pos, self.step_size)
        self.agent_pos, collided = calc_collision(old_pos, new_pos, self.agent_radius, self.width, self.height, self.all_obstacles)
        
        # Apply environment stochasticity -> new position due to slippery-ness
        if env_stochasticity and not collided:  # if not collided, new_pos and correct_new_pos are the same --> intuition slippery-ness only has effect when no collisions happen
            old_pos_extra = self.agent_pos.copy()
            # Repeat same code as above
            slippage_action = 4 if action in [0,1,4] else action  
            # To simulate slipping, we want the repeat action to be "do nothing" if action was "do nothing", "accelerate" or "decelerate".
            # If the action was a "turn", then slipping would cause the turn to be repeated. 
            self.orientation, new_pos, self.speed = calc_new_position(slippage_action, self.speed, self.orientation, self.agent_angle, self.agent_pos, self.step_size)
            self.agent_pos, collided = calc_collision(old_pos_extra, new_pos, self.agent_radius, self.width, self.height, self.all_obstacles)      

        # if collided set speed to 0
        old_speed=self.speed
        if collided:
            self.speed = 0

        # Update delivery and battery information
        self.vision_triangle = calc_vision_triangle(self.agent_pos, self.orientation, self.agent_radius)

        self.carrying, self.item, self.delivered, pickup, delivered = update_delivery(self.carrying, self.items,
        self.delivered, self.agent_pos, self.agent_radius, self.item_radius, self.delivery_points, self.delivery_radius
        )

        self.battery, charged = update_battery(self.battery, self.battery_drain_per_step, self.agent_pos, self.charger, 
                                 self.speed, self.battery_value_reward_charging, action)

        if self.battery <= 0:
            battery_died = True
        else:
            battery_died=False

        # Compute the reward for this step
        reward = default_reward_function(pickup, delivered, collided, charged, battery_died, old_pos,
                                         self.agent_pos, self.agent_radius, self.forbidden_zones, old_speed)
        reward += shaping_reward(old_pos, old_target, self.agent_pos)

        # Bookkeeping for ending an episode
        terminated = all(self.delivered) and self.battery==100 # terminated: relates to success/failure
        truncated = battery_died # Truncated: relates to early stopping

        # Update some statistics
        self.cumulative_reward += reward
        self.total_nr_steps += 1
        if collided:
            self.total_nr_collisions += 1

        return self._compute_features(), reward, terminated, truncated, info
    

    def _compute_features(self):
        """
        Compute the complete observation feature vector that defines the state space of the agent. This consists of:
        - A distance to wall/object at 5 angles from the agent, with some maximum distance that can be measured
        - Distance to item that can be picked up at 5 angles from the agent, with some maximum distance that can be measured
        - A binary indicator indicating if the agent is carrying any items
        - A distance between agent and the target on x-axis and y-axis
        """
        x, y = self.agent_pos

        # Distance to wall or object at 5 angles from the agent
        steps_fw = compute_distance_to_obstacle(self.orientation, self.max_range,
            self.agent_radius, self.agent_pos, self.width, self.height, self.all_obstacles)
        steps_left = compute_distance_to_obstacle((self.orientation-2*self.agent_angle)%360,
            self.max_range, self.agent_radius, self.agent_pos, self.width, self.height, self.all_obstacles) # Left 90
        steps_right = compute_distance_to_obstacle((self.orientation+2*self.agent_angle)%360,
            self.max_range, self.agent_radius, self.agent_pos, self.width, self.height, self.all_obstacles)  # Right 90
        steps_fw_left = compute_distance_to_obstacle((self.orientation-self.agent_angle)%360,
            self.max_range, self.agent_radius, self.agent_pos, self.width, self.height, self.all_obstacles) # Left 45
        steps_fw_right = compute_distance_to_obstacle((self.orientation+self.agent_angle)%360,
            self.max_range, self.agent_radius, self.agent_pos, self.width, self.height, self.all_obstacles)  # Right 45

        vision_triangle_sensor = calc_vision_triangle_features(self.agent_pos, self.agent_radius, self.item_starts, self.delivered, self.carrying, self.vision_triangle, self.all_obstacles, self.delivery_points)

        if self.carrying >= 0:
            carrying = 1
        else:
            carrying = 0

        feature_vector = [
            x/self.width, 
            y/self.height, 
            self.orientation/self.agent_angle,
            carrying,
            steps_left/self.max_range,
            steps_fw_left/self.max_range,
            steps_fw/self.max_range,
            steps_fw_right/self.max_range,
            steps_right/self.max_range,
            self.battery/100.0,
            vision_triangle_sensor,
            self.speed,
        ]
        return feature_vector


    def render(self, mode="human", show_full_legend=True, show_difficulty_region=False):
        """
        Render the environment in a GUI. This allows us to visually inspect the agents behaviour.
        """
        render_gui(self, mode=mode, show_full_legend=show_full_legend, show_difficulty_region=show_difficulty_region)

    def close(self):
        plt.close(fig='all')
