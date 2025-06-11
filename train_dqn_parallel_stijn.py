"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from tqdm import trange
from gymnasium.vector import AsyncVectorEnv
from world.environment import Environment
from agents.dqn import DQNAgent
from evaluate_trained_dqn import evaluate_agent_training
import time

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="", 
                   help="Name of the model to save. ")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=2400,
                   help="Number of episodes to train the agent for. " \
                   "Each episode is completed by either reaching the target, or putting `iters` steps.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=2,
                   help="Random seed value for the environment.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_min", type=float, default=0.01,
                   help="Minimum epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_decay_proportion", type=float, default=0.7,
                   help="Proportion of training to decay epsilon over. " \
                   "0.5 means that halfway of the training procedure we the epsilon has reached in minimum value of 0.1.")
    return p.parse_args()


def make_env():
    def _thunk():
        return Environment()
    return _thunk

def get_epsilon(episode, phase_len):
    if episode < phase_len:
        # Phase 1: Easy
        eps_start, eps_end = 1.0, 0.1
        phase_episode = episode
    elif episode < 2 * phase_len:
        # Phase 2
        eps_start, eps_end = 0.5, 0.1
        phase_episode = episode - phase_len
    elif episode < 3 * phase_len:
        # Phase 3
        eps_start, eps_end = 0.3, 0.05
        phase_episode = episode - 2 * phase_len
    elif episode < 4 * phase_len:
        # Phase 3
        eps_start, eps_end = 0.3, 0.05
        phase_episode = episode - 2 * phase_len
    elif episode < 5 * phase_len:
        # Phase 3
        eps_start, eps_end = 0.3, 0.05
        phase_episode = episode - 2 * phase_len
    else:
        # Phase 4+
        eps_start, eps_end = 0.2, 0.01
        phase_episode = episode - 3 * phase_len

    # Linear decay within phase
    epsilon = eps_start - ((eps_start - eps_end) / (0.7*phase_len)) * phase_episode
    return max(eps_end, epsilon)  # prevent going below eps_end

def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int, epsilon: float,
         epsilon_min: float, epsilon_decay_proportion: float):  
    """Main loop of the program."""
    num_envs = 5  # Set this to the number of parallel environments you want
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    agent = DQNAgent(state_size=14, action_size=6, seed=random_seed)

    # Number of episodes to decay the epsilon linearly
    decay_steps = int(epsilon_decay_proportion * (episodes//num_envs) * iters)
    initial_epsilon = epsilon  # Store initial epsilon for decay calculation

    # Curriculum schedule: split episodes into 4 equal parts
    phase_len = episodes // (6*num_envs)

    total_steps = 0

    for episode in range(episodes // num_envs):
        epsilon = get_epsilon(episode, phase_len)
        agent.epsilon = epsilon
        # Set difficulty based on curriculum phase (applies to all envs in batch)
        if episode < phase_len:
            difficulty = 0
            number_of_items = 0
            battery_drain_per_step = 0
        elif episode < 2 * phase_len:
            difficulty = 0
            number_of_items = 3
            battery_drain_per_step = 0
        elif episode < 3 * phase_len:
            difficulty = 0
            number_of_items = 3
            battery_drain_per_step = 0.2
        elif episode < 4 * phase_len:
            difficulty = 1
            number_of_items = 3
            battery_drain_per_step = 0.2
        elif episode < 5 * phase_len:
            difficulty = 2
            number_of_items = 3
            battery_drain_per_step = 0.2
        else:
            difficulty = 3
            number_of_items = 3
            battery_drain_per_step = 0.2

        print(f"Episode batch {episode + 1}/{episodes // num_envs} - Epsilon: {epsilon:.4f}")

        if not no_gui and (episode+1) % 50 == 0 and episode != 0:
            evaluate_agent_training(agent=agent, iters=500, no_gui=False, difficulty=difficulty, number_of_items= number_of_items, battery_drain_per_step= battery_drain_per_step, epsilon=0.1)
        agent.epsilon=epsilon
        opts = {"difficulty": difficulty, 'number_of_items': number_of_items, 'battery_drain_per_step': battery_drain_per_step}
        states, _ = envs.reset(options=opts)
        done_flags = num_envs*[False]
        terminated_flags = num_envs*[False]
        for _ in trange(iters):
            # take action + step in `num_envs` parallel environments
            actions = [agent.take_action(state) for state in states]
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            for j in range(num_envs):
                done = terminateds[j] or truncateds[j]
                if terminateds[j]:
                    terminated_flags[j] = True
                if done_flags[j]==False:
                    agent.update(states[j], actions[j], rewards[j], next_states[j], done)
                    if done:
                        done_flags[j]=True
            states = next_states
            if all(done_flags):
                break

        print(terminated_flags)
    model_path = f"models/dqn_{name}_final.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")



if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed, args.epsilon, 
         args.epsilon_min, args.epsilon_decay_proportion)
