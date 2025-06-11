"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
from world.environment import Environment
from agents.ppo import PPOAgent
import numpy as np

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

    # Initialize vector envs
    num_envs = 4  # Number of parallel environments
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    # Initialize agent
    agent = PPOAgent(state_size=11, action_size=6, seed=random_seed, num_envs=num_envs)

    # Curriculum schedule: split episodes based on difficulty, lower to higher
    phase_len = episodes // 4

    for episode in range(episodes):

        print(f"Episode batch {episode + 1}/{episodes}")

        # Evaluate every few episodes
        difficulty, number_of_items, battery_drain_per_step = set_difficulty(episode, phase_len)

        # Set difficulty for curriculum learning
        opts = {"difficulty": difficulty, 'number_of_items': number_of_items,
                'battery_drain_per_step': battery_drain_per_step}

        # Resetting all parallel envs

        states, _ = envs.reset(options=opts)
        states = np.array(states, dtype=np.float32)

        # Initialize tracking variables
        episode_rewards = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)

        for timestep in range(iters):
            # Get actions for all environments at once
            actions, log_probs, values = agent.take_action_training(states)

            # Step all environments
            next_states, rewards, terminated, truncated, infos = envs.step(actions)

            # Update agent with batch of experiences
            agent.update(
                states=states,  # All states
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                terminated=terminated
            )

            # Track rewards
            episode_rewards += rewards
            active_envs &= ~(terminated | truncated)

            # Prepare for next step
            states = next_states

            # Check if we should learn (now handled inside update())
            if not np.any(active_envs):
                break

        # Print average reward across environments
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward across {num_envs} envs: {avg_reward:.2f}")
        terminated_envs = [not i for i in active_envs]
        print(terminated_envs)

        # Optional: Save model periodically
        if episode % 100 == 0:
            agent.save(f"ppo_agent_{episode}.pth")

        # if env_gui:
        #     env.close()

    # grid_name = grid.stem  # Get the grid name from the path
    # # after all episodes for this grid
    # model_path = f"models/ppo_{grid_name}_test.pth"
    # agent.save(model_path)
    # print(f"Saved trained model to -> {model_path}")

    # # Evaluate the agent
    # Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=None, agent_start_pos=(9, 13), target_positions=[(11, 3)])


if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)