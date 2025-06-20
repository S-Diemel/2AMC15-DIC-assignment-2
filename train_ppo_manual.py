"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
from world.environment import Environment
from agents.ppo import PPOAgent
import numpy as np
# from evaluate_trained_ppo import evaluate_agent_training
from torch.utils.tensorboard import SummaryWriter
import time

# Simple tensorboard logging
writer = SummaryWriter(log_dir=f"logs/ppo_training_10k")

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="",
                   help="Name of the model to save. ")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=10_000,
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

def get_curriculum_parameters(episode, phase_len, base_entropy=0.1, min_entropy=0.01):
    # Define curriculum phases:
    # (entropy_start, entropy_end, difficulty, num_items, battery_drain)
    curriculum = [
        (0.1, 0.01, 0, 1, 0.0),
        (0.1, 0.01, 0, 1, 0.25),
        (0.1, 0.01, 0, 3, 0.25),
        (0.1, 0.05, 1, 3, 0.25),
        (0.05, 0.01, 1, 3, 0.25),
        (0.1, 0.05, 2, 3, 0.25),
        (0.05, 0.01, 2, 3, 0.25),
        (0.05, 0.01, 3, 3, 0.25)  # Final fallback phase
    ]

    phase = min(episode // phase_len, len(curriculum) - 1)
    entropy_start, entropy_end, difficulty, number_of_items, battery_drain = curriculum[phase]
    phase_episode = episode - phase * phase_len

    # Linear entropy decay within the phase
    entropy_coef = entropy_start - ((entropy_start - entropy_end) / (0.7 * phase_len)) * phase_episode
    entropy_coef = max(entropy_end, entropy_coef)
    entropy_coef = max(min_entropy, entropy_coef)

    return difficulty, number_of_items, battery_drain, entropy_coef


def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""

    start_time = time.time()

    # Initialize vector envs
    num_envs = 5  # Number of parallel environments
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    # Initialize agent
    agent = PPOAgent(state_size=12, action_size=5, seed=random_seed, num_envs=num_envs)

    # Curriculum schedule: split episodes based on difficulty, lower to higher
    phase_len = episodes // 8

    for episode in range(episodes):

        print(f"Episode batch {episode + 1}/{episodes}")

        # Evaluate every few episodes
        difficulty, number_of_items, battery_drain_per_step, entropy_coef = get_curriculum_parameters(episode, phase_len)
        agent.entropy_coef = entropy_coef

        # if (episode + 1) % 100_000 == 0:
        #     evaluate_agent_training(agent=agent, iters=500, no_gui=False, difficulty=difficulty,
        #                             number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step)

        # Set difficulty for curriculum learning
        opts = {"difficulty": difficulty, 'number_of_items': number_of_items,
                'battery_drain_per_step': battery_drain_per_step}

        # Resetting all parallel envs

        states, _ = envs.reset(options=opts)
        states = np.array(states, dtype=np.float32)

        # Initialize tracking variables
        episode_rewards = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)
        terminated_envs = np.ones(num_envs, dtype=bool)

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
                dones=terminated | truncated
            )

            # Track rewards
            episode_rewards += rewards
            active_envs &= ~(terminated | truncated)
            terminated_envs &= ~terminated

            # Prepare for next step
            states = next_states

            # Check if we should learn (now handled inside update())
            if not np.any(active_envs):
                break

        # Print average reward across environments
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward across {num_envs} envs: {avg_reward:.2f}")
        # Log to TensorBoard
        writer.add_scalar("Reward/AverageEpisodeReward", avg_reward, global_step=episode)
        # Print number of envs where objective was completed
        print(~terminated_envs)

        # Optional: Save model periodically
        if (episode + 1) % 500 == 0:
            agent.save(f"models/ppo/ppo_{episode + 1}.pth")


    model_path = f"models/ppo_after_training_{episodes}_v3.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    end_time = time.time()
    print(f"Took {end_time - start_time:.2f} seconds")
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)