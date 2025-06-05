import gymnasium as gym
from gymnasium import spaces
import numpy as np

def action_to_values(action):
    values = {
        0: (1, 0),   # go
        1: (0, 0),  # stop
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0,0)
    }
    return values[action]

def orientation_to_directions(orientation):
    directions = {
        0: (0, 1),       # Up
        45: (1, 1),      # Up-Right
        90: (1, 0),       # Right
        135: (1, -1),      # Down-Right
        180: (0, -1),      # Down
        225: (-1, -1),     # Down-Left
        270: (-1, 0),     # Left
        315: (-1, 1),    # Up-Left
    }
    return directions[orientation]
class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            width=10.0,
            height=10.0,
            rack_obstacles=None,
            box_obstacles=None,
            item_starts=None,
            delivery_points=None,
            agent_radius=0.2,
            step_size=0.2,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.agent_radius = agent_radius
        self.step_size = step_size
        self.speed = 1
        self.orientation = 0
        # Obstacles: list of (x_min, y_min, x_max, y_max)
        self.racks = rack_obstacles or [
            (4.0, 0.0, 4.5, 6.0),  # example rack
            (6.0, 4.0, 6.5, 10.0),
        ]
        self.box_obs = box_obstacles or [
            (1.0, 5.0, 2.0, 6.0),
            (7.0, 1.0, 8.0, 2.0),
        ]
        self.charger = (4,9,6,10)
        self.forbidden_zone = (0,8,2,10)
        self.agent_angle = 90
        self.item_starts = item_starts or [(2.0, 2.0), (1.0, 1.0)]
        self.item_radius = 0.2
        self.delivery_points = delivery_points or [(8.0, 8.0), (9.0, 2.0)]
        self.delivery_radius = agent_radius * 1.5

        # Actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(5)

        # Observation: [agent_x, agent_y, orientation, speed, carrying (0/1), target_x, target_y, dist_fw, dist_left, dist_right, item_fw, item_fw_left, item_fw_right, battery]
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array(
            [width, height, 3, 1, 1, width, height, 10, 10, 10, 10, 10, 10, 1], dtype=np.float32
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([self.agent_radius + 0.1, self.agent_radius + 0.1], dtype=np.float32)
        self.items = [np.array(pos, dtype=np.float32) for pos in self.item_starts]
        self.delivered = [False] * len(self.items)
        self.carrying = -1  # -1 = none, otherwise index of carried item
        self.battery = 100
        return self._compute_features()


    def _calc_new_position(self, action):
        new_speed, sign_orientation = action_to_values(action)
        if action == 0 or action == 1:
            self.speed = new_speed

        if self.speed == 1:
            self.orientation = (self.orientation + sign_orientation*self.agent_angle) % 360
            direction = orientation_to_directions(self.orientation)
            new_position = np.array([self.agent_pos[0] + self.step_size*direction[0], self.agent_pos[1] + self.step_size*direction[1]])
            return new_position
        else:
            self.orientation = (self.orientation + sign_orientation*self.agent_angle) % 360
            new_position = self.agent_pos
            return new_position

    def _calc_collision(self, old_pos, new_position):
        new_pos = new_position.copy()
        collided = False
        # wall collisions
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

        # obstacle collisions
        for (xmin, ymin, xmax, ymax) in self.racks + self.box_obs:
            closest = np.clip(new_pos, [xmin, ymin], [xmax, ymax])
            delta = new_pos - closest
            dist = np.linalg.norm(delta)
            if dist < self.agent_radius:
                collided = True
                overlap = self.agent_radius - dist
                if dist > 1e-10:
                    new_pos += (delta / dist) * overlap
                else:
                    # fallback: cancel
                    new_pos = old_pos.copy()
                break
        return new_pos, collided

    def _update_delivery(self, action):
        delivered = False
        pickup=False
        if action == 4 and self.carrying == -1 and self.speed == 0:
            for i, (pos, delivered) in enumerate(zip(self.items, self.delivered)):
                if not delivered and np.linalg.norm(self.agent_pos - pos) < self.agent_radius + self.item_radius:
                    self.carrying = i
                    pickup=True
                    break

        if self.carrying != -1:
            self.items[self.carrying] = self.agent_pos.copy()

        # Deliver
        if self.carrying != -1 and action == 4 and self.speed == 0:
            for point in self.delivery_points:
                if np.linalg.norm(self.agent_pos - point) < self.delivery_radius:
                    self.delivered[self.carrying] = True
                    delivered =True
                    self.carrying = -1
                    break

        return pickup, delivered

    def _compute_distance_to_wall(self, orientation):
        max_range = 10 - self.agent_radius
        direction = np.array(orientation_to_directions(orientation))
        current_pos = self.agent_pos.copy()

        for d in np.arange(0, max_range, 0.05):
            test_pos = current_pos + direction * d

            # Check wall boundaries
            if (test_pos[0] < 0 or test_pos[0] > self.width or test_pos[1] < 0 or test_pos[1] > self.height):
                return d - self.agent_radius

            # Check obstacle collision (racks + box obstacles)
            for (xmin, ymin, xmax, ymax) in self.racks + self.box_obs:
                if xmin <= test_pos[0] <= xmax and ymin <= test_pos[1] <= ymax:
                    return d - self.agent_radius

        return max_range - self.agent_radius

    def _item_sensor(self, orientation):
        max_range = 10 + self.agent_radius
        direction = np.array(orientation_to_directions(orientation))
        current_pos = self.agent_pos.copy()
        for d in np.arange(0, max_range, 0.05):
            test_pos = current_pos + direction * d

            # Check targets
            for i, item_pos in enumerate(self.item_starts):
                x, y = item_pos
                if x-self.item_radius <= test_pos[0] <= x+self.item_radius and y-self.item_radius <= test_pos[1] <= y+self.item_radius and not self.delivered[i] and self.carrying != i:
                    return max(d - self.agent_radius, 0)

        return max_range - self.agent_radius

    def _update_battery(self):
        self.battery -= 0.1
        old_battery = self.battery
        x, y = self.agent_pos
        xmin, ymin, xmax, ymax = self.forbidden_zone
        if xmin <= x <= xmax and ymin <= y <= ymax and self.speed == 0:
            self.battery = 100
            if old_battery <= 10:
                return True
        return False


    def _compute_features(self):
        x, y = self.agent_pos

        steps_fw = self._compute_distance_to_wall(self.orientation)
        steps_left = self._compute_distance_to_wall((self.orientation-self.agent_angle)%360) # Left
        steps_right = self._compute_distance_to_wall((self.orientation+self.agent_angle)%360)  # Right
        item_fw = self._item_sensor(self.orientation)
        item_left = self._item_sensor((self.orientation-self.agent_angle)%360)
        item_right = self._item_sensor((self.orientation+self.agent_angle)%360)
        if self.carrying >= 0:
            carrying = 1
            target_x, target_y = self.delivery_points[self.carrying]
        else:
            carrying = 0
            target_x, target_y = (0,0)


        feature_vector = [x, y, self.orientation/self.agent_angle, self.speed, carrying, target_x, target_y,
                          steps_fw, steps_left, steps_right, item_fw, item_left, item_right,
                          self.battery/100]
        # Observation: [agent_x, agent_y, orientation, speed, carrying (0/1), target_x, target_y, dist_fw, dist_left, dist_right, item_fw, item_fw_left, item_fw_right, battery]
        return feature_vector
    def step(self, action):
        assert self.action_space.contains(action)
        old_pos = self.agent_pos.copy()
        new_pos = self._calc_new_position(action)
        correct_new_pos, collided = self._calc_collision(old_pos, new_pos)
        self.agent_pos = correct_new_pos
        pickup, delivered = self._update_delivery(action)
        charged = self._update_battery()
        reward = self._reward_function(pickup, delivered, collided, charged)
        if self.battery == 0 or all(self.delivered):
            done = True
        else:
            done = False

        return self._compute_features(), reward, done

    def _check_forbidden_zone(self):
        x, y = self.agent_pos
        r = self.agent_radius
        xmin, ymin, xmax, ymax = self.forbidden_zone
        return (x + r > xmin and x - r < xmax and
                y + r > ymin and y - r < ymax)

    def _reward_function(self, pickup, delivered, collided, charged):
        #TODO: shaping reward if agent is carrying an item
        reward = -1
        if charged:
            reward+=5
        if pickup:
            reward+=10
        if delivered:
            reward+=20
        if collided:
            reward-=5
        if self._check_forbidden_zone():
            reward-=5
        return reward

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        # draw racks and box obstacles
        xmin, ymin, xmax, ymax = self.forbidden_zone
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="pink"))
        xmin, ymin, xmax, ymax = self.charger
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="lightgreen"))
        for (xmin, ymin, xmax, ymax) in self.racks:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="saddlebrown"))
        for (xmin, ymin, xmax, ymax) in self.box_obs:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="gray"))

        for point in self.delivery_points:
            ax.add_patch(Circle(point, self.delivery_radius, fill=False, linestyle="--"))

        for i, pos in enumerate(self.items):
            if not self.delivered[i] or self.carrying == i:
                ax.add_patch(Circle(pos, self.item_radius, color="orange"))


        # draw agent
        ax.add_patch(Circle(self.agent_pos, self.agent_radius, color="blue"))
        direction = np.array(orientation_to_directions(self.orientation))
        dot_pos = self.agent_pos + direction * (self.agent_radius * 0.8)

        ax.add_patch(Circle(dot_pos, self.agent_radius * 0.15, color="yellow"))

        plt.pause(1 / self.metadata["render_fps"])
        if mode == "rgb_array":
            return np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8).reshape(
                plt.gcf().canvas.get_width_height()[::-1] + (3,)
            )

    def close(self):
        pass


# Example usage:
if __name__ == "__main__":
    env = WarehouseEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # random
        obs, rew, done, _, _ = env.step(action)
        env.render()
    print("Episode finished.")
