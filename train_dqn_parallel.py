"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from tqdm import trange
from gymnasium.vector import AsyncVectorEnv
from world.environment import Environment
from agents.dqn import DQNAgent
from train_curriculum_utils import setup_curriculum, get_curriculum_parameters, evaluate_agent_metrics, save_metrics_to_csv
import numpy as np


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="", 
                   help="Name of the model to save.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=10000,
                   help="Number of episodes to train the agent for.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations per batch.")
    p.add_argument("--random_seed", type=int, default=2,
                   help="Random seed value for the environment.")
    return p.parse_args()


# Define curriculum phases as:
# (percent_of_total, eps_start, eps_end, difficulty, num_items, battery_drain)
curriculum = [
    (0.10, 1.0, 0.1,  0, 1, 0.0),
    (0.10, 0.5, 0.05, 0, 1, 0.25),
    (0.20, 0.3, 0.05, 0, 3, 0.25),
    (0.30, 0.3, 0.05, 1, 3, 0.25),
    (0.30, 0.3, 0.01, 2, 3, 0.25),
]

def make_env():
    def _thunk():
        return Environment()
    return _thunk


def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int):
    """Main loop of the program."""
    num_envs = 5  # Set this to the number of parallel environments you want
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    agent = DQNAgent(state_size=12, action_size=5, seed=random_seed)

    total_episodes = episodes // num_envs

    curriculum_phases = setup_curriculum(total_episodes, curriculum, 3)
    print(curriculum_phases)
    metrics_by_stage = {}

    for episode in range(episodes // num_envs):

        phase_number, epsilon, difficulty, number_of_items, \
            battery_drain_per_step, should_evaluate = get_curriculum_parameters(episode, curriculum_phases)
        
        if should_evaluate:
            eval_index = curriculum_phases[phase_number-1]['eval_points'].index(episode)
            print(f"Evaluating agent at episode {episode} (eval point {eval_index}) - Phase: {phase_number}")
            # TODO: no_gui False is not working
            metrics = evaluate_agent_metrics(agent, difficulty, number_of_items, battery_drain_per_step, no_gui=False)
            metrics_by_stage[(phase_number, eval_index)] = metrics

        print(f"Episode batch {episode + 1}/{episodes // num_envs} - Epsilon: {epsilon:.4f} - Phase: {phase_number}")

        agent.epsilon=epsilon
        opts = {"difficulty": difficulty, 'number_of_items': number_of_items, 
                'battery_drain_per_step': battery_drain_per_step, 'difficulty_mode': "train"}
        states, _ = envs.reset(options=opts)

        # Initialize tracking variables
        episode_rewards = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)
        terminated_envs = np.zeros(num_envs, dtype=bool)

        for _ in range(iters):
            # take action + step in `num_envs` parallel environments
            actions = agent.take_actions_batch(states)
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = terminateds | truncateds

            for i in range(num_envs):
                # Update for all environments - including final step with terminal reward
                agent.update(states[i], actions[i], rewards[i], next_states[i], dones[i])

            # Track rewards
            episode_rewards += rewards
            terminated_envs |= terminateds
            active_envs &= ~(terminateds | truncateds)

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

    # Save the trained model
    model_path = f"models/dqn_{name}_new.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    # Save updated metrics to csv
    csv_filename = f"metrics/dqn_{name}_metrics_new.csv"
    save_metrics_to_csv(metrics_by_stage, csv_filename)
    print(f"Saved evaluation metrics to -> {csv_filename}")


if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed)
