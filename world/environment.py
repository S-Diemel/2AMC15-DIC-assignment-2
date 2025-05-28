import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            width=10.0,
            height=10.0,
            rack_obstacles=None,
            box_obstacles=None,
            box_start=(2.0, 2.0),
            delivery_point=(8.0, 8.0),
            robot_radius=0.45,
            step_size=0.8,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.step_size = step_size

        # Obstacles: list of (x_min, y_min, x_max, y_max)
        self.racks = rack_obstacles or [
            (4.0, 0.0, 4.5, 6.0),  # example rack
            (6.0, 4.0, 6.5, 10.0),
        ]
        self.box_obs = box_obstacles or [
            (1.0, 5.0, 2.0, 6.0),
            (7.0, 1.0, 8.0, 2.0),
        ]

        # Pick-up box and delivery target
        self.box_start = np.array(box_start, dtype=np.float32)
        self.delivery_point = np.array(delivery_point, dtype=np.float32)
        self.delivery_radius = robot_radius * 1.2

        # Actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)

        # Observation: [robot_x, robot_y, carrying (0/1), box_x, box_y]
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array(
            [width, height, 1, width, height], dtype=np.float32
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start robot in lower-left corner (just away from walls)
        self.robot_pos = np.array(
            [self.robot_radius + 0.1, self.robot_radius + 0.1], dtype=np.float32
        )
        self.box_pos = self.box_start.copy()
        self.carrying = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [*self.robot_pos, float(self.carrying), *self.box_pos],
            dtype=np.float32,
        )

    def step(self, action):
        assert self.action_space.contains(action)
        old_pos = self.robot_pos.copy()
        move = np.zeros(2, dtype=np.float32)
        if action == 0:
            move[1] += self.step_size
        elif action == 1:
            move[1] -= self.step_size
        elif action == 2:
            move[0] -= self.step_size
        else:
            move[0] += self.step_size

        # attempt move
        new_pos = old_pos + move
        reward = -0.1  # step cost
        collided = False

        # wall collisions
        if new_pos[0] - self.robot_radius < 0:
            new_pos[0] = self.robot_radius
            collided = True
        if new_pos[0] + self.robot_radius > self.width:
            new_pos[0] = self.width - self.robot_radius
            collided = True
        if new_pos[1] - self.robot_radius < 0:
            new_pos[1] = self.robot_radius
            collided = True
        if new_pos[1] + self.robot_radius > self.height:
            new_pos[1] = self.height - self.robot_radius
            collided = True

        # obstacle collisions
        for (xmin, ymin, xmax, ymax) in self.racks + self.box_obs:
            closest = np.clip(new_pos, [xmin, ymin], [xmax, ymax])
            delta   = new_pos - closest
            dist    = np.linalg.norm(delta)
            if dist < self.robot_radius:
                collided = True
                overlap = self.robot_radius - dist
                if dist > 1e-6:
                    new_pos += (delta / dist) * overlap
                else:
                    # fallback: cancel
                    new_pos = old_pos.copy()
                reward -= 1.0
                break

        self.robot_pos = new_pos

        # pick up
        if (not self.carrying) and np.linalg.norm(self.robot_pos - self.box_pos) < self.robot_radius * 1.1:
            self.carrying = True
            reward += 1.0

        # if carrying, box follows robot
        if self.carrying:
            self.box_pos = self.robot_pos.copy()

        # delivery
        done = False
        if self.carrying and np.linalg.norm(self.robot_pos - self.delivery_point) < self.delivery_radius:
            reward += 10.0
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        # draw racks and box obstacles
        for (xmin, ymin, xmax, ymax) in self.racks:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="saddlebrown"))
        for (xmin, ymin, xmax, ymax) in self.box_obs:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="gray"))

        # draw delivery point
        ax.add_patch(Circle(self.delivery_point, self.delivery_radius, fill=False, linestyle="--"))

        # draw pick-up box (if not yet picked)
        if not self.carrying:
            ax.add_patch(Circle(self.box_pos, self.robot_radius*0.8, color="orange"))

        # draw robot
        ax.add_patch(Circle(self.robot_pos, self.robot_radius, color="blue"))

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
