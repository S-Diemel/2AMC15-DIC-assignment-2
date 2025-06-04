"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

from agents.ppo import PPOAgent

try:
    from world.environment_dqn import Environment
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
    from world.environment_dqn import Environment
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
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of episodes to train the agent for. Each episode is completed by either reaching the target, or putting `iter` steps.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid: list[Path], no_gui: bool, episodes: int, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    assert len(grid) == 1, "Provide exactly one grid for training"
    grid = grid[0]

    # Set up the environment
    env = Environment(grid, no_gui=True, sigma=sigma, target_fps=fps,
                      random_seed=random_seed, agent_start_pos=(9, 13), target_positions=[(11, 3)])

    agent = ppo.PPOAgent(state_size=10, action_size=4, seed=random_seed)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

        # if episode%500 == 0:
        #     no_gui = False
        # else:
        #     no_gui = True

        # Reset rewards and states at beginning of episode
        state = env.reset_env(no_gui=True)
        total_reward = 0

        for timestep in trange(iters):

            # Select action based on PPO agent's policy
            action = agent.take_action(state)
            next_state, reward, terminated, info = env.step(action)  # Updated to handle truncated

            # Update PPO agent's buffer
            agent.update(state, reward, terminated)
            total_reward += reward
            state = next_state

            if terminated:
                break

        print(f"Total reward: {total_reward:.2f}")

    grid_name = grid.stem  # Get the grid name from the path
    # after all episodes for this grid
    model_path = f"models/ppo_{grid_name}_test.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    # Evaluate the agent
    Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=None, agent_start_pos=(9, 13), target_positions=[(11, 3)])


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.episodes, args.iter, args.fps, args.sigma, args.random_seed)