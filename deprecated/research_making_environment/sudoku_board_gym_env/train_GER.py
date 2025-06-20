from sb3_contrib import MaskablePPO
from datetime import datetime
# from sb3_contrib.common.maskable.utils import get_action_masks

# import gymnasium as gym
from gymnasium import spaces

# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sudoku_gym import SudokuEnv
# from team05_A3.SudokuSolver import solver
from sudoku_gym import TensorboardCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


# env = DummyVecEnv([lambda: env])

# Now wrap the environment with VecNormalize, setting normalize_images=False
# env = VecNormalize(env, norm_obs=False, norm_reward=False, normalize_images=False)


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor class for processing image observations.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract the shape of the input image from observation_space
        channels, height, width = observation_space.shape
        print(observation_space.shape)
        
        # Dynamically set the kernel size and number of channels based on m and n
        self.m = M
        self.n = N # Assuming m*n is height and width, and m*n is channels
        # Define CNN layers (adjust based on m and n)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.m * self.n, self.m * self.n, kernel_size=(self.m, self.n), stride=self.m, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.m * self.n , self.m * self.n, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output size after convolution layers to determine the input size for the linear layer
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Define a fully connected layer after convolutional layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass observations through the CNN layers and then the linear layers
        return self.linear(self.cnn(observations))
    
# Create the environment
M = 3
N = 3
k_timesteps = 250
reward_structure = 'GER'
# reward_structure = 'vanilla'

env = make_vec_env(lambda: SudokuEnv(M,N, reward_structure, eval_env=False), n_envs=4)

eval_env = make_vec_env(lambda: SudokuEnv(M,N, reward_structure, eval_env=True), n_envs=1)


total_timesteps = k_timesteps * 1000

curr_time = datetime.now()
start_of_day = curr_time.replace(hour=0, minute=0, second=0, microsecond=0)

# Calculate the time difference
seconds_from_start_of_day = int((curr_time - start_of_day).total_seconds())
timestamp = f"{curr_time.date()}_{seconds_from_start_of_day}"

model_details = f"cnn_sudoku_{M}x{N}_reward_{reward_structure}_{k_timesteps}k_{timestamp}"


eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=f"./logs/{model_details}",
                             log_path=f"./logs/{model_details}", eval_freq=k_timesteps, n_eval_episodes=20,
                             deterministic=True, render=False)


 # Train a PPO agentA
# action_masks = get_action_masks(env)
# print(action_masks)
# print(action_masks)
# print("Shape is: ", action_masks.shape)
policy_kwargs = dict(
    normalize_images = True,
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=3 * (M * N)**2),
    net_arch=dict(pi=[3 * (M * N)**2, (M * N)**2], vf=[3 * (M * N)**2, (M * N)**2]),
    )
model = MaskablePPO("CnnPolicy", env=env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="../cnn_sudoku_tensorboard/")
print(model.policy)

model.learn(total_timesteps=500_000, use_masking=True, tb_log_name=model_details, callback=[TensorboardCallback(), eval_callback])
model.save(model_details)