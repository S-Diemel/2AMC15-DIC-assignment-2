import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from world.environment import Environment  # Assuming environment.py is in same directory


class GridNavigationEnv(gym.Env):
    """Gym-compatible environment for grid navigation with orientation"""

    def __init__(self,
                 grid_fp: str,
                 no_gui: bool,
                 sigma: float = 0.0,
                 target_fps: int = 30,
                 random_seed: Optional[int] = 0,
                 agent_start_pos: Optional[Tuple[int, int]] = None,
                 target_position: Optional[Tuple[int, int]] = None):
        self.elapsed_steps = 0
        super().__init__()

        # Initialize the custom environment
        self.env = Environment(
            grid_fp=Path(grid_fp),
            no_gui=no_gui,  # Disable GUI for training efficiency
            sigma=sigma,
            agent_start_pos=agent_start_pos,
            target_fps=target_fps,
            random_seed=random_seed,
            target_positions=target_position
        )

        # Reset to get grid dimensions
        self.reset()
        grid = self.env.grid
        max_row, max_col = grid.shape
        print(grid.shape)

        # Define action space: 4 discrete actions
        self.action_space = gym.spaces.Discrete(4)  # 0-3 as defined in action_to_values

        # Define observation space (9 features)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                0,  # min x position
                0,  # min y position
                0,  # min speed
                -1,  # min orientation (normalized)
                0,  # min forward steps
                0,  # min forward-left steps
                0,  # min forward-right steps
                -max_row,  # min dx
                -max_col  # min dy
            ]),
            high=np.array([
                max_row - 1,  # max x position
                max_col - 1,  # max y position
                1,  # max speed
                1,  # max orientation (normalized)
                2 * max(max_row, max_col),  # max obstacle distance
                2 * max(max_row, max_col),
                2 * max(max_row, max_col),
                max_row,  # max dx
                max_col  # max dy
            ]),
            dtype=np.float32
        )

    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return initial observation"""
        self.elapsed_steps = 0
        obs = self.env.reset_env(**kwargs)
        return obs, {}  # Convert to float32 for SB3 compatibility

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one action in the environment"""
        obs, reward, terminated, info = self.env.step(action)
        # print("Distance to target: ", ((abs(obs[7]) + abs(obs[8])) / 10))
        # obs, reward, terminated, info = self.env.step(0)    # Keep the agent moving in a direction
        if reward == -1:    # Not hit a wall
            reward = -1 * ((abs(obs[7]) + abs(obs[8])) / 10)   # Distance to target
        self.elapsed_steps += 1
        if self.elapsed_steps == 500:   # Terminal state not reached
            truncated = True
        else:
            truncated = False
        """Early termination of episode if reward not reached within 500 timesteps"""
        return (
            obs,
            reward,
            terminated,
            truncated,  # ending episode early if target not found within 500 timesteps
            info
        )