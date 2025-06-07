import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.lines import Line2D


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


def point_in_rectangle(x, y, rect):
    """Checks if a point occurs in a single rectangle or not."""
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def is_inside_set_of_rectangles(x, y, set_of_rectangles):
    """Checks whether a point (x, y) occurs within a set of rectangles and returns True if it occurs in at least one of the rectangles"""
    for rect in set_of_rectangles:
        if point_in_rectangle(x, y, rect):
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
                if not point_in_rectangle(x, y, difficulty_region):
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
            if not is_inside_set_of_rectangles(x_cand, y_cand, inflated) and point_in_rectangle(x_cand, y_cand, difficulty_region):
                return (x_cand, y_cand)
        else:
            if not is_inside_set_of_rectangles(x_cand, y_cand, inflated):
                return (x_cand, y_cand)


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        number_of_items=3,
        agent_radius=0.25,
        step_size=0.2,
        difficulty=None,  # Difficulty level represented by None (random no particular difficulty level), 0 (easy), 1 (medium), 2 (hard)  
        extra_obstacles=None,  # List of additional obstacles, if any --> useful for experimenting with humans or boxes in random places
    ):
        super().__init__()

        self.width = 20.0
        self.height = 10.0
        self.agent_radius = agent_radius
        self.step_size = step_size
        self.speed = 0
        self.orientation = 0
        self.agent_angle = 45
        self.max_range = 5 # Maximum range for sensors
        self.battery_drain_per_step = 0.5  # In reset, the battery is always reset to 100%
        self.battery_value_reward_charging = 20  # From which battery level to reward the agent for going to the charging station. 

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
            (5, self.height - 6 * half_width_of_rack, 12, self.height - 4 * half_width_of_rack),
            (13.5, self.height - 6 * half_width_of_rack, 18, self.height - 4 * half_width_of_rack),
            # Second row of storage racks
            (5, self.height - 11 * half_width_of_rack, 9, self.height - 9 * half_width_of_rack),
            (10, self.height - 11 * half_width_of_rack, 12, self.height - 9 * half_width_of_rack),
            # Third row of storage racks
            (5, self.height - 16 * half_width_of_rack, 18, self.height - 14 * half_width_of_rack)
        ]
        if extra_obstacles is not None: 
            self.extra_obstacles = extra_obstacles  # List of additional obstacles, if any
        else:
            self.extra_obstacles = []

        self.forbidden_zones = [(14, self.height - 12 * half_width_of_rack, 17.5, self.height - 8 * half_width_of_rack)]  # Red Area: forbidden zones, where the agent can but should not go
        self.charger = (3.5, 0, 6, 1)  # Green Area: charging area
        self.charger_center = ((self.charger[0] + self.charger[2]) / 2, (self.charger[1] + self.charger[3]) / 2)

        # Create delivery zones (grey areas) around the racks
        self.delivery_zones = self._create_delivery_zones(self.racks, margin=0.5)

        # For CURRICULUM LEARNING: define the difficulty of the environment by constraining the agent starting position and delivery points
        self.difficulty = difficulty

        # Define all items (= packages), there spawn points and the delivery points
        self.number_of_items = number_of_items
        self.item_radius = 0.2
        self.item_spawn_center = ((self.item_spawn[0][0] + self.item_spawn[0][2]) / 2, (self.item_spawn[0][1] + self.item_spawn[0][3]) / 2)
        self.delivery_radius = agent_radius
        # Both items and delivery points are linked by index, so item 0 is delivered at delivery point 0, etc.

        # Initialize some Gym environment paramters: Necessary for Gym-compatible trainers
        self.action_space = spaces.Discrete(6)
        # Give the range of values that the agents state space can take for each feature
        low = np.array([
            0.0,   # x / width
            0.0,   # y / height
            0.0,   # orientation / 45
            0.0,   # carrying flag
            -1.0,  # dist_target_x / width
            -1.0,  # dist_target_y / height
            0.0,   # steps_left / max_range
            0.0,   # steps_fw_left / max_range
            0.0,   # steps_fw / max_range
            0.0,   # steps_fw_right / max_range
            0.0,   # steps_right / max_range
            0.0,   # item_left / max_range
            0.0,   # item_fw_left / max_range
            0.0,   # item_fw / max_range
            0.0,   # item_fw_right / max_range
            0.0,   # item_right / max_range
            0.0    # battery / 100
        ], dtype=np.float32)

        high = np.array([
            1.0,  # x / width
            1.0,  # y / height
            7.0,  # orientation / 45  (possible values: 0…7)
            1.0,  # carrying flag
            1.0,  # dist_target_x / width
            1.0,  # dist_target_y / height
            1.0,  # steps_left / max_range
            1.0,  # steps_fw_left / max_range
            1.0,  # steps_fw / max_range
            1.0,  # steps_fw_right / max_range
            1.0,  # steps_right / max_range
            1.0,  # item_left / max_range
            1.0,  # item_fw_left / max_range
            1.0,  # item_fw / max_range
            1.0,  # item_fw_right / max_range
            1.0,  # item_right / max_range
            1.0   # battery / 100
        ], dtype=np.float32)
        # Give possible values of observational space
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Maintaining statistics
        self.cumulative_reward = 0
        self.total_nr_collisions = 0
        self.total_nr_steps = 0

        # Call reset to finish initializing the environment
        self.reset()
    
    def reset(self, no_gui=True, seed=None, agent_start_pos=False, difficulty=None, extra_obstacles=None):
        """
        Resetting the environment for a new task for the agent. This involves spawning packages/items, delivery points, the agent itself. 
        It also involves initializing some attributes to the environment and the agent, such as: that it is not carrying any items/packages, 
        it has not delivered any packages yet, it is at full battery, etc. Finally it computes the initial agent state observation vector.
        """
        super().reset(seed=seed)
        if extra_obstacles is not None:
            self.extra_obstacles = extra_obstacles
            # Implicit else: we keep the self.extra_obstacles from the initialization above, which is an empty list for None
        # Combine all obstacles for collision detection 
        self.all_obstacles = self.racks + self.extra_obstacles
        if difficulty is not None:
            self.difficulty = difficulty
        self.difficulty_region = self._set_difficulty_of_env(self.difficulty)  # For curriculum learning set the difficulty of the environment
        self.item_starts = sample_points_in_rectangles(self.item_spawn, self.number_of_items, self.item_radius, self.all_obstacles)  # spawn/initialize packages/items
        self.delivery_points = sample_points_in_rectangles(self.delivery_zones, self.number_of_items, self.delivery_radius, self.all_obstacles, self.difficulty_region)  # choose delivery spots
        if not agent_start_pos:  # randomly sample agent position if none is supplied.
            self.agent_pos = np.array(sample_one_point_outside(self.all_obstacles, self.agent_radius, (0, 0, self.width, self.height), self.difficulty_region))
        else:
            self.agent_pos = np.array(agent_start_pos)  # Use given starting position
        self.items = [np.array(pos, dtype=np.float32) for pos in self.item_starts] 
        self.delivered = [False] * len(self.items)
        self.carrying = -1  # -1 = not carrying any items; otherwise this is the index of the item that is at that moment carried by the agent.
        self.battery = 100  # initializes battery
        self.no_gui = no_gui
        return self._compute_features()
    
    def _set_difficulty_of_env(self, difficulty=None):
        """
        For CURRICULUM LEARNING: define invisible easy, medium and hard zones within the environment,
        which divide the warehouse to the right of the package spawn place into 3 evenly sized sections.
        The agent and the delivery points will exclusively be sampled according to the selected difficulty level. 
        """
        # Only values None, 0, 1, 2 are accepted the rest is not accepted
        assert difficulty in [None, 0, 1, 2], "Only values None (no level), 0 (easy), 1 (medium), 2 (hard) are accepted the rest is not accepted"
        item_spawn_width = self.item_spawn[0][2]
        width_difficulty_region = (self.width - item_spawn_width) / 3
        # Create difficulty region
        difficulty_region = (item_spawn_width + difficulty * width_difficulty_region, 
                             0, 
                             item_spawn_width + (difficulty + 1) * width_difficulty_region, 
                             self.height) if difficulty is not None else None
        return difficulty_region
    
    def _create_delivery_zones(self, racks, margin):
        """
        Creates rectangular delivery zones around a list of storage racks, but make sure that the rectangular delivery areas 
        do not extend outside the environment boundaries.
        """
        delivery_zones = []
        
        # Helper function to process each potential zone
        def add_clipped_zone(xmin, ymin, xmax, ymax):
            """The created delivery zone rectangles are clipped such that do not extend outside the environment boundaries."""
            # Ensure delivery rectangle coordinates do not violate the warehouse dimensions (0, 0, width, height)
            clipped_xmin = max(0, xmin)
            clipped_ymin = max(0, ymin)
            clipped_xmax = min(self.width, xmax)
            clipped_ymax = min(self.height, ymax)

            # Only add the zone if it has a valid, positive area after clipping. This prevents zero-width/height delivery rectangles.
            if clipped_xmax > clipped_xmin and clipped_ymax > clipped_ymin:
                delivery_zones.append((clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax))

        # Iterate over the storage racks to create its surrounding zones
        for (xmin, ymin, xmax, ymax) in racks:
            # Calculate rectangle coordinates for all four rectangles along the storage racks
            
            add_clipped_zone(xmin - margin, ymax, xmax + margin, ymax + margin)  # Above
            add_clipped_zone(xmin - margin, ymin - margin, xmax + margin, ymin)  # Below
            add_clipped_zone(xmin - margin, ymin, xmin, ymax)  # Left
            add_clipped_zone(xmax, ymin, xmax + margin, ymax)  # Right
            
        return delivery_zones

    def _calc_new_position(self, action):
        """Calculate the new position and orientation of the agent within the environment."""
        new_speed, sign_orientation = action_to_values(action)

        if self.speed == 0 and action==0:
            self.orientation = (self.orientation + sign_orientation*self.agent_angle) % 360
            direction = orientation_to_directions(self.orientation)
            new_position = np.array([self.agent_pos[0] + self.step_size*direction[0], self.agent_pos[1] + self.step_size*direction[1]])
            return new_position
        else:
            self.orientation = (self.orientation + sign_orientation*self.agent_angle) % 360
            new_position = self.agent_pos
            return new_position

    def _calc_collision(self, old_pos, new_position):
        """Compute if any collisions happened with walls or obstacles."""
        new_pos = new_position.copy()
        collided = False

        # Wall collisions
        if new_pos[0] - self.agent_radius < 0:
            new_pos[0] = self.agent_radius
            collided = True
        if new_pos[0] + self.agent_radius > self.width:
            new_pos[0] = self.width - self.agent_radius
            collided = True
        if new_pos[1] - self.agent_radius < 0:
            new_pos[1] = self.agent_radius
            collided = True
        if new_pos[1] + self.agent_radius > self.height:
            new_pos[1] = self.height - self.agent_radius
            collided = True

        # Obstacle collisions
        for (xmin, ymin, xmax, ymax) in self.all_obstacles:
            closest = np.clip(new_pos, [xmin, ymin], [xmax, ymax])
            delta = new_pos - closest
            dist = np.linalg.norm(delta)
            if dist < self.agent_radius:
                collided = True
                overlap = self.agent_radius - dist
                if dist > 1e-10:
                    new_pos += (delta / dist) * overlap
                else:
                    new_pos = old_pos.copy()
                break

        return new_pos, collided

    def _update_delivery(self, action):
        """
        Update all agent attribute regarding the delivery. Whether the agent is carrying an item, and whether each item is delivered. 
        Furthermore, for a step it saves whether an item was picked up or delivered, which is important for computing the reward.
        """
        item_delivered = False
        item_picked_up=False
        if action == 5 and self.carrying == -1 and self.speed == 0:  
            # If we are performing the pickup action, we are not yet carrying any item and we are standing still we can pick up an item.
            for i, (pos, delivered_status) in enumerate(zip(self.items, self.delivered)):  
                # Iterate over all items that can be picked up, and make sure we have info on whether these items have been delivered yet.
                if not delivered_status and np.linalg.norm(self.agent_pos - pos) < self.agent_radius + self.item_radius:  
                    # if item is not yet delivered and the radius of the agent and the item are overlapping, we can pick up the item
                    self.carrying = i
                    item_picked_up = True
                    break

        if self.carrying != -1:  # If we are carrying an item then we should move the item with the agent
            self.items[self.carrying] = self.agent_pos.copy()

        # Deliver
        if self.carrying != -1 and action == 4 and self.speed == 0:  # If we are carrying an item, we do pickup/dropoff action and we are standing still
            for i, point in enumerate(self.delivery_points):  
                if self.carrying == i and np.linalg.norm(self.agent_pos - point) < self.agent_radius + self.delivery_radius:
                    # Check if item that is being carried and its delivery point correspond, and check if the radius of agent and dropoff point overlap
                    self.delivered[self.carrying] = True
                    self.carrying = -1
                    item_delivered = True
                    break

        return item_picked_up, item_delivered

    def _compute_distance_to_wall(self, orientation):
        """
        Computing the distance between the agent and the walls and obstacles in the environment. These are important sensors to inform the agent in its task.
        If there are no walls or obstacles in the agent's line of vision, the max_range that the agent can see with the sensor is returned. 
        """
        max_range_from_agent_center = self.max_range + self.agent_radius  # We want to measure from edge of agent
        direction = np.array(orientation_to_directions(orientation))
        current_pos = self.agent_pos.copy()

        for d in np.arange(0, max_range_from_agent_center, 0.05):
            test_pos = current_pos + direction * d

            # Check wall boundaries of the environmnet
            if not (0 <= test_pos[0] <= self.width and 0 <= test_pos[1] <= self.height):  # if the test position is not in the environment boundaries
                return max(d - self.agent_radius, 0)

            # Check obstacle collision (racks + box obstacles)
            for (xmin, ymin, xmax, ymax) in self.all_obstacles:
                if xmin <= test_pos[0] <= xmax and ymin <= test_pos[1] <= ymax:
                    return max(d - self.agent_radius, 0)

        return self.max_range  # There is nothing within a self.max_range from the edge of the agent (where sensors are)

    def _item_sensor(self, orientation):
        """
        Computing the distance between the agent and some items that the agent needs to pick-up. These are important sensors to inform the agent in its task.
        If there are no items in the line of vision, the max_range that the agent can see with the sensor is returned. 
        """
        max_range_from_agent_center = self.max_range + self.agent_radius  # We want to measure from edge of agent
        direction = np.array(orientation_to_directions(orientation))
        current_pos = self.agent_pos.copy()

        for d in np.arange(0, max_range_from_agent_center, 0.05):
            test_pos = current_pos + direction * d

            # Check targets
            for i, item_pos in enumerate(self.item_starts):
                x, y = item_pos
                if x-self.item_radius <= test_pos[0] <= x+self.item_radius and y-self.item_radius <= test_pos[1] <= y+self.item_radius and not self.delivered[i] and self.carrying != i:
                    return max(d - self.agent_radius, 0)

        return self.max_range  # There is nothing within a self.max_range from the edge of the agent (where sensors are)

    def _update_battery(self):
        """All logic for reducing battery level during steps, recharging by standing still in the charging area, and rewarding charging at low battery level"""
        self.battery -= self.battery_drain_per_step  # Decrease the battery of the agent at each timestep
        old_battery = self.battery
        x, y = self.agent_pos
        xmin, ymin, xmax, ymax = self.charger
        if xmin <= x <= xmax and ymin <= y <= ymax and self.speed == 0:  # if robot stands still in charging stop the battary is full again.
            self.battery = 100
            if old_battery <= self.battery_value_reward_charging:  # only reward charging if battery was actually low
                return True
        return False

    def _compute_dist_to_target(self):
        """
        Compute distance between the agent and the next target on the x-axis and the y-axis. Note that when an item is carrying an item/package the target is the delivery point 
        for this package, but when the agent is not carrying any items the target will be the center of item/package pickup area. This is because we want the agent
        to find packages, and not have the direct location. Once it finds a package, it scans the package and knows where this package should be delivered. 
        """
        target_x, target_y = self._compute_target()
        x, y = self.agent_pos
        dist_target_x = target_x - x
        dist_target_y = target_y - y
        return dist_target_x, dist_target_y  # return distance on x and y axis to target
    
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
        steps_fw = self._compute_distance_to_wall(self.orientation)
        steps_left = self._compute_distance_to_wall((self.orientation-2*self.agent_angle)%360) # Left 90
        steps_right = self._compute_distance_to_wall((self.orientation+2*self.agent_angle)%360)  # Right 90
        steps_fw_left = self._compute_distance_to_wall((self.orientation-self.agent_angle)%360) # Left 45
        steps_fw_right = self._compute_distance_to_wall((self.orientation+self.agent_angle)%360)  # Right 45
        # Distance to item that can be picked up at 5 angles from the agent
        item_fw = self._item_sensor(self.orientation)
        item_left = self._item_sensor((self.orientation-2*self.agent_angle)%360)
        item_right = self._item_sensor((self.orientation+2*self.agent_angle)%360)
        item_fw_left = self._item_sensor((self.orientation-self.agent_angle)%360)
        item_fw_right = self._item_sensor((self.orientation+self.agent_angle)%360)
        # Binary indicator whether agent is carrying an item
        if self.carrying >= 0:
            carrying = 1
        else:
            carrying = 0
        # Distance between agent and target on x and y axis.
        dist_target_x, dist_target_y = self._compute_dist_to_target()

        # Combining everything into a single vector
        feature_vector = [x/self.width, y/self.height, self.orientation/self.agent_angle, carrying, dist_target_x/self.width, dist_target_y/self.height,
                          steps_left/self.max_range, steps_fw_left/self.max_range, steps_fw/self.max_range, steps_fw_right/self.max_range, steps_right/self.max_range,
                          item_left/self.max_range, item_fw_left/self.max_range, item_fw/self.max_range, item_fw_right/self.max_range, item_right/self.max_range]
        
        return feature_vector

    def _compute_target(self):
        """
        When an item is carrying an item/package the target is the delivery point for this package, but when the agent is not carrying any items the target 
        will be the center of item/package pickup area. This is because we want the agent to find packages, and not have the direct location. Once it finds 
        a package, it scans the package and knows where this package should be delivered.
        """
        if self.battery < self.battery_value_reward_charging:
            target_x, target_y = self.charger_center
        else:
            if self.carrying >= 0:
                target_x, target_y = self.delivery_points[self.carrying]  # Location of delivery point of the item that the agent is carrying
            else:
                target_x, target_y = self.item_spawn_center  # Center of the area where items spawn
        return target_x, target_y
    
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
        assert self.action_space.contains(action)
        old_pos = self.agent_pos.copy()
        old_target = self._compute_target()
        new_pos = self._calc_new_position(action)
        correct_new_pos, collided = self._calc_collision(old_pos, new_pos)
        self.agent_pos = correct_new_pos
        pickup, delivered = self._update_delivery(action)
        charged = self._update_battery()
        reward = self._reward_function(pickup, delivered, collided, charged, old_pos)
        reward += self._shaping_reward(old_pos, old_target)  # Used include the improvement with respect to the target in the reward function (g, Harada, & Russell, 1999).
        done = self.battery <= 0 or all(self.delivered)
        # Update some statistics
        self.cumulative_reward += reward
        self.total_nr_steps += 1
        if collided:
            self.total_nr_collisions += 1
        return self._compute_features(), reward, done

    def _agent_in_forbidden_zone(self):
        """Check if the agent is in one of the forbidden zones to properly assign a negative reward to this."""
        in_forbidden_zone = False
        x, y = self.agent_pos
        r = self.agent_radius
        for xmin, ymin, xmax, ymax in self.forbidden_zones:  # Iterate over forbidden zones
            if (x + r > xmin and x - r < xmax and y + r > ymin and y - r < ymax):
                in_forbidden_zone = True
        return in_forbidden_zone  # If not in a forbidden zone then return false

    def _reward_function(self, pickup, delivered, collided, charged, old_pos):
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
        if np.array_equal(old_pos, self.agent_pos):  # Punish agent for staying in the same position
            # old_pos[0]==self.agent_pos[0] and old_pos[1]==self.agent_pos[1] 
            reward -= 0.5
        if charged:  # charging when below certain battery value
            reward += 5
        if pickup:  # picking up an item
            reward += 10
        if delivered:  # delivering an item
            reward += 100
        if collided:  # colliding with a wall or object
            reward -= 2
        if self._agent_in_forbidden_zone():  # being in a forbidden zone
            reward -= 2
        return reward

    def _shaping_reward(self, old_pos, old_target):
        """Potential based shaping of the reward inspired by (g, Harada, & Russell, 1999)"""
        gamma = 0.99  # gamma value we use 
        
        # Use Manhatten distance when you do no allow diagonal moves:
        # old_distance_to_target = -(abs(old_pos[0]-old_target[0]) + abs(old_pos[1]-old_target[1])) # negative manhattan distance
        # new_distance_to_target = -(abs(self.agent_pos[0]-old_target[0]) + abs(self.agent_pos[1]-old_target[1]))  # negative manhattan distance
        
        # Use Chebyshev distance when you do allow diagonal moves which are equivalent in number of steps as a 'straight' move:
        old_distance_to_target = -max(abs(old_pos[0] - old_target[0]), abs(old_pos[1] - old_target[1]))
        new_distance_to_target = -max(abs(self.agent_pos[0] - old_target[0]), abs(self.agent_pos[1] - old_target[1]))

        shaping_reward = gamma*new_distance_to_target - old_distance_to_target
        return shaping_reward

    def render(self, mode="human", show_full_legend=True, show_difficulty_region=False):
        """This function renders the GUI of the environement allowing us to visually inspect the agents behaviour."""
        # Return nothing if GUI is turned off
        if self.no_gui:
            return
        
        # Initializing the environement.
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')

        # Turn of the ticks and number on the axes, because they are not relevant for the GUI
        ax.set_xticks([])
        ax.set_yticks([])

        # Give room for a side legend
        fig = ax.get_figure()
        if show_full_legend:
            fig.subplots_adjust(left=0.2, right=0.8)
        else:
            fig.subplots_adjust(right=0.8)

        # Draw Areas
        # Item Spawn (Yellow)
        for (xmin, ymin, xmax, ymax) in self.item_spawn:
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#fdeeac", alpha=0.75))

        # Delivery Zones (Grey)
        for (xmin, ymin, xmax, ymax) in self.delivery_zones:
             # Clip zones to be within warehouse boundaries for rendering
            draw_xmin = max(xmin, 0)
            draw_ymin = max(ymin, 0)
            draw_xmax = min(xmax, self.width)
            draw_ymax = min(ymax, self.height)
            if draw_xmax > draw_xmin and draw_ymax > draw_ymin:
                 ax.add_patch(Rectangle((draw_xmin, draw_ymin), draw_xmax - draw_xmin, draw_ymax - draw_ymin, color="#eeeded"))

        # Forbidden Zone (Red)
        for (xmin, ymin, xmax, ymax) in self.forbidden_zones:
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#fa6e6e", alpha=0.75))

        # Charger (Green)
        xmin, ymin, xmax, ymax = self.charger
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#81fd8b", alpha=0.75))

        # Racks (Blue)
        for (xmin, ymin, xmax, ymax) in self.racks:
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#7881ff"))

        # Extra obstacles (Dark grey)
        for (xmin, ymin, xmax, ymax) in self.extra_obstacles:
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#636363"))

        # Draw Delivery Points --> with numbers and correct dynamics
        for i, point in enumerate(self.delivery_points):
            if self.carrying == i:  # Make delivery point more clearly visible when carrying the item for that delivery point
                ax.add_patch(Circle(point, self.delivery_radius, color='darkred', alpha=0.85))
            else:
                ax.add_patch(Circle(point, self.delivery_radius, color='darkred', alpha=0.3))
            ax.text(
                point[0], point[1],       # x, y
                str(i),                   # number itself
                color='white',            # text color
                ha='center', va='center', 
                fontsize=7,              
                fontweight='bold',
                zorder=10                 # make sure it overlays the delivery point patch
            )

        # Draw Items (Packages)
        for i, point in enumerate(self.items):
            if not self.delivered[i] or self.carrying == i:
                ax.add_patch(Circle(point, self.item_radius, color="orange"))
                ax.text(
                    point[0], point[1],       # x, y
                    str(i),                   # number itself
                    color='white',            # text color
                    ha='center', va='center', 
                    fontsize=7,              
                    fontweight='bold',
                    zorder=10                 # make sure it overlays the delivery point patch
                )

        # Potentially show difficulty region for sampling with curriculum learning      
        if show_difficulty_region:
            if self.difficulty_region is not None:
                xmin, ymin, xmax, ymax = self.difficulty_region
                ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="green", alpha=0.2))

        # Draw Agent
        if self.carrying > -1:  # Give orange edge to agent when carrying an item
            ax.add_patch(Circle(self.agent_pos, self.agent_radius, facecolor="#00A800", edgecolor="orange", linewidth=2))
        else:
            ax.add_patch(Circle(self.agent_pos, self.agent_radius, color="#00A800"))
        # Compute where the white dot which expresses the agents orientation should be 
        # -> for 45 degree orientations we need to use the unit circle to make sure that white circle is on the agent. 
        dir_vec = np.array(orientation_to_directions(self.orientation), dtype=float)
        dir_unit = dir_vec / np.linalg.norm(dir_vec)
        dot_offset = self.agent_radius * 0.75 
        dot_pos = self.agent_pos + dir_unit * dot_offset
        ax.add_patch(Circle(dot_pos, self.agent_radius * 0.15, color="white"))

        # Add a legend to the GUI for better understanding:
        # battery, cum hazard, bumps into obstacles, total steps
        legend_stats = [
            Line2D([0], [0], linestyle="None", label=f"Battery: {self.battery:.1f}%"),
            Line2D([0], [0], linestyle="None", label=f"Total Steps: {self.total_nr_steps:.0f}"),
            Line2D([0], [0], linestyle="None", label=f"Cumulative Reward: {self.cumulative_reward:.0f}"),
            Line2D([0], [0], linestyle="None", label=f"Total Collisions: {self.total_nr_collisions:.0f}")
        ]
        legend1 = ax.legend(
            handles=legend_stats,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="Episode statistics:",               
            title_fontsize="medium",      
            borderaxespad=0,
            handlelength=0,  
            handletextpad=0  
        )
        ax.add_artist(legend1)

        if show_full_legend:
            legend_env_info = [
                Line2D([0], [0],
                    marker='o', markersize=10,
                    markerfacecolor="#00A800",
                    markeredgecolor="orange" if self.carrying > -1 else "#00A800",
                    linestyle="None",
                    label="Agent"
                ),
                Patch(facecolor="#fdeeac", edgecolor="none", alpha=0.75, label="Item Spawn"),
                Patch(facecolor="#eeeded", edgecolor="none", label="Delivery Zone"),
                Patch(facecolor="#fa6e6e", edgecolor="none", alpha=0.75, label="Forbidden Zone"),
                Patch(facecolor="#81fd8b", edgecolor="none", alpha=0.75, label="Charger"),
                Patch(facecolor="#7881ff", edgecolor="none", label="Storage racks"),
                Patch(facecolor="darkred", edgecolor="none", alpha=0.3, label="Delivery Point"),
                Patch(facecolor="orange", edgecolor="none", label="Item"),
            ]
            if len(self.extra_obstacles) > 0:
                legend_env_info += [Patch(facecolor="#636363", edgecolor="none", label="Extra Obstacle")]
            # ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15,1))
            legend2 = ax.legend(
                handles=legend_env_info,
                loc="upper right",
                bbox_to_anchor=(-0.02, 1),   
                title="Environment info:",               
                title_fontsize="medium",       
                frameon=True,
                borderaxespad=0,
                labelspacing=0.5
            )
            ax.add_artist(legend2)

        plt.title("Warehouse Simulation")
        plt.pause(1 / self.metadata["render_fps"])
        if mode == "rgb_array":
            return np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8).reshape(
                plt.gcf().canvas.get_width_height()[::-1] + (3,)
            )

    def close(self):
        plt.close()
