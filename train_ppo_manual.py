"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.greedy_agent import GreedyAgent
    from agents.dqn import DQNAgent
    from agents import ppo
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to train the agent for. Each episode is completed by either reaching the target, or putting `iter` steps.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_min", type=float, default=0.01,
                   help="Minimum epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_decay", type=float, default=0.9985,  # Value for 3500 episodes.
                   help="Decay factor for the epsilon value in the epsilon-greedy policy.")
    return p.parse_args()


def main(grid: list[Path], no_gui: bool, episodes: int, iters: int, fps: int,
         sigma: float, random_seed: int, epsilon: float,
         epsilon_min: float, epsilon_decay: float):
    """Main loop of the program."""

    assert len(grid) == 1, "Provide exactly one grid for training"
    grid = grid[0]

    # Set up the environment
    env = Environment(grid, no_gui=False, sigma=sigma, target_fps=fps,
                      random_seed=random_seed, agent_start_pos=(11, 3), target_positions=[(9, 13)])

    agent = ppo.PPO(state_dim=9, action_dim=4)

    t_truncate = 500    # Stop an episode for going on forever

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

        total_reward = agent.collect_rollout(env, iters, t_truncate)
        print(f"Total reward: {total_reward:.2f}")

    grid_name = grid.stem  # Get the grid name from the path
    # after all episodes for this grid
    model_path = f"models/ppo_{grid_name}_test.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    # Evaluate the agent
    Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.episodes, args.iter, args.fps, args.sigma, args.random_seed, args.epsilon,
         args.epsilon_min, args.epsilon_decay)