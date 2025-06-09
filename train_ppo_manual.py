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
    p.add_argument("--episodes", type=int, default=5000,
                   help="Number of episodes to train the agent for. Each episode is completed by either reaching the target, or putting `iters` steps.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()

def make_env(difficulty):
    def _thunk():
        return Environment(difficulty=difficulty)
    return _thunk


def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""
    num_envs = 4  # Number of parallel environments
    agent = PPOAgent(state_size=16, action_size=6, seed=random_seed, num_envs=num_envs)
    # Curriculum schedule: split episodes into 4 equal parts
    phase_len = episodes // 4

    for episode_batch in range(episodes // num_envs):

        # Set difficulty based on curriculum phase
        if episode_batch < phase_len:
            difficulty = 0  # easy
        elif episode_batch < 2 * phase_len:
            difficulty = 1  # medium
        elif episode_batch < 3 * phase_len:
            difficulty = 2  # hard
        else:
            difficulty = None  # no difficulty, just train on any problem

        print(f"Episode batch {episode_batch + 1}/{episodes // num_envs}")

        # Resetting all parallel envs
        envs = AsyncVectorEnv([make_env(difficulty) for _ in range(num_envs)])
        states, _ = envs.reset()
        states = np.array(states, dtype=np.float32)

        # Initialize tracking variables
        episode_rewards = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)

        for timestep in range(iters):
            # Select actions for all active environments
            actions = np.zeros(num_envs, dtype=np.int32)
            log_probs = np.zeros(num_envs, dtype=np.float32)
            values = np.zeros(num_envs, dtype=np.float32)
            for i in range(num_envs):
                if active_envs[i]:
                    actions[i], log_probs[i], values[i] = agent.take_action_training(states[i], i)

            # Step all environments
            next_states, rewards, terminated, truncated, infos = envs.step(actions)

            # Update agent and track rewards for each environment
            for i in range(num_envs):
                if active_envs[i]:
                    agent.update(
                        state=states[i],
                        action = actions[i],
                        reward=rewards[i],
                        log_prob=log_probs[i],
                        value=values[i],
                        terminated=terminated[i],
                        env_idx=i
                    )
                    episode_rewards[i] += rewards[i]

                    # Mark environment as done if terminated or truncated
                    if terminated[i] or truncated[i]:
                        active_envs[i] = False

            # Check if we've completed rollout steps and need to learn
            if agent.step_counter >= agent.rollout_steps:
                agent.learn()

            # Prepare for next step
            states = next_states

            # If all environments are done, break early
            if not np.any(active_envs):
                break

        # Print average reward across environments
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward across {num_envs} envs: {avg_reward:.2f}")

        # Optional: Save model periodically
        if episode_batch % 10 == 0:
            agent.save(f"ppo_agent_{episode_batch}.pth")

        # if env_gui:
        #     env.close()

    # grid_name = grid.stem  # Get the grid name from the path
    # # after all episodes for this grid
    # model_path = f"models/ppo_{grid_name}_test.pth"
    # agent.save(model_path)
    # print(f"Saved trained model to -> {model_path}")
    #
    # # Evaluate the agent
    # Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=None, agent_start_pos=(9, 13), target_positions=[(11, 3)])


if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)