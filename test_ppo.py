"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
from world.environment_ppo import Environment
from agents.ppo import PPOAgent
import numpy as np
from evaluate_trained_dqn import evaluate_agent_training
import torch

import os
print(os.getcwd())  # shows your current working directory
print(os.listdir())

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="",
                   help="Name of the model to save. ")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=5_000,
                   help="Number of episodes to train the agent for. Each episode is completed by either reaching the target, or putting `iters` steps.")
    p.add_argument("--iters", type=int, default=1_000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()

def make_env(difficulty=None):
    def _thunk():
        return Environment(difficulty=difficulty)
    return _thunk

def set_difficulty(episode, phase_len):
    # Set difficulty based on curriculum phase (applies to all envs in batch)
    if episode < phase_len:
        difficulty = 0  # easy
        number_of_items = 1
        battery_drain_per_step = 0
    elif episode < 2 * phase_len:
        difficulty = 0  # medium
        number_of_items = 3
        battery_drain_per_step = 0
    elif episode < 3 * phase_len:
        difficulty = 0  # medium
        number_of_items = 3
        battery_drain_per_step = 0.2
    else:
        difficulty = 0  # no difficulty, just train on any problem
        number_of_items = 3
        battery_drain_per_step = 0.5

    return difficulty, number_of_items, battery_drain_per_step

def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""

    agent = PPOAgent(state_size=11, action_size=6, seed=random_seed, num_envs=1)
    agent.load('best_ppo_agent_1600_100.pth')
    # Evaluate every few episodes
    difficulty = 0  # no difficulty, just train on any problem
    number_of_items = 3
    battery_drain_per_step = 0.0

    # Set difficulty for curriculum learning
    opts = {"difficulty": difficulty, 'number_of_items': number_of_items,
            'battery_drain_per_step': battery_drain_per_step}

    evaluate_agent_training(agent=agent, iters=500, no_gui=False, difficulty=difficulty,number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step)

if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)