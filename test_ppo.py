"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
from world.environment import Environment
from agents.ppo import PPOAgent
import numpy as np
from evaluate_trained_ppo import evaluate_agent_training
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

def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""

    agent = PPOAgent(state_size=15, action_size=6, seed=random_seed, num_envs=1)
    agent.load(f"models/ppo_after_training_10000_v2.pth")

    difficulty, number_of_items, battery_drain_per_step = 1, 3, 0.25

    # Set difficulty for curriculum learning
    opts = {"difficulty": difficulty, 'number_of_items': number_of_items,
            'battery_drain_per_step': battery_drain_per_step}

    evaluate_agent_training(agent=agent, iters=500, no_gui=False, difficulty=difficulty,number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step)

if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)