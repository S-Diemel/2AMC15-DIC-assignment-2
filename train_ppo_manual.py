"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from gymnasium.vector import AsyncVectorEnv
from world.environment import Environment
from agents.ppo import PPOAgent
import numpy as np
from train_curriculum_utils import setup_curriculum, get_curriculum_parameters, evaluate_agent_metrics, save_metrics_to_csv
import time

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="",
                   help="Name of the model to save.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=2_500,
                   help="Number of episodes to train the agent for.")
    p.add_argument("--iters", type=int, default=1_000,
                   help="Number of iterations per batch.")
    p.add_argument("--random_seed", type=int, default=2,
                   help="Random seed value for the environment.")
    return p.parse_args()

# Define curriculum phases as:
# (percent_of_total, entropy_start, entropy_end, difficulty, num_items, battery_drain)
curriculum = [
        (0.10, 0.1, 0.05, 0, 1, 0.0),
        (0.10, 0.1, 0.05, 0, 1, 0.25),
        (0.20, 0.1, 0.05, 0, 3, 0.25),
        (0.30, 0.1, 0.01, 1, 3, 0.25),
        (0.30, 0.1, 0.01, 2, 3, 0.25),
    ]

def make_env(difficulty=None):
    def _thunk():
        return Environment(difficulty=difficulty)
    return _thunk


def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""

    start_time = time.time()

    # Initialize vector envs
    num_envs = 5  # Number of parallel environments
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    # Initialize agent
    agent = PPOAgent(state_size=12, action_size=5, seed=random_seed, num_envs=num_envs)

    curriculum_phases = setup_curriculum(episodes, curriculum, 3)
    print(curriculum_phases)
    metrics_by_stage = {}

    for episode in range(episodes):

        # Get curriculum settings
        phase_number, entropy_coef, difficulty, number_of_items, \
            battery_drain_per_step, should_evaluate = get_curriculum_parameters(episode, curriculum_phases)

        if should_evaluate:
            eval_index = curriculum_phases[phase_number-1]['eval_points'].index(episode)
            print(f"Evaluating agent at episode {episode} (eval point {eval_index}) - Phase: {phase_number}")
            # TODO: no_gui False is not working
            metrics = evaluate_agent_metrics(agent, difficulty, number_of_items, battery_drain_per_step, no_gui=False)
            metrics_by_stage[(phase_number, eval_index)] = metrics

        print(f"Episode batch {episode + 1}/{episodes} - Entropy: {entropy_coef:.4f} - Phase: {phase_number}")

        # Assign entropy coef to PPO agent
        agent.entropy_coef = entropy_coef

        # Set difficulty for curriculum learning
        opts = {"difficulty": difficulty, 'number_of_items': number_of_items, 'battery_drain_per_step': battery_drain_per_step, 'difficulty_mode': "train"}

        # Resetting all parallel envs

        states, _ = envs.reset(options=opts)
        states = np.array(states, dtype=np.float32)

        # Initialize tracking variables
        episode_rewards = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)
        terminated_envs = np.zeros(num_envs, dtype=bool)

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
            terminated_envs |= terminated
            active_envs &= ~(terminated | truncated)

            # Prepare for next step
            states = next_states
            #
            if not np.any(active_envs):
                break

        # Print average reward across environments
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward across {num_envs} envs: {avg_reward:.2f}")

        # Print number of envs where objective was completed
        print(terminated_envs)

        # Optional: Save model periodically
        if (episode + 1) % 1_000 == 0:
            agent.save(f"models/ppo/ppo_{episode + 1}.pth")


    model_path = f"models/ppo_after_training_{episodes}_final.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    # Save updated metrics to csv
    csv_filename = f"metrics/ppo_{name}_metrics.csv"
    save_metrics_to_csv(metrics_by_stage, csv_filename)
    print(f"Saved evaluation metrics to -> {csv_filename}")

    end_time = time.time()
    print(f"Took {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)