"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from tqdm import trange
from gymnasium.vector import AsyncVectorEnv
from world.environment import Environment
from agents.dqn import DQNAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--name", type=str, default="", 
                   help="Name of the model to save. ")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--episodes", type=int, default=10000,
                   help="Number of episodes to train the agent for. " \
                   "Each episode is completed by either reaching the target, or putting `iters` steps.")
    p.add_argument("--iters", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
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


def main(name: str, no_gui: bool, episodes: int, iters: int, random_seed: int, epsilon: float,
         epsilon_min: float, epsilon_decay_proportion: float):  
    """Main loop of the program."""
    num_envs = 5  # Set this to the number of parallel environments you want
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    agent = DQNAgent(state_size=16, action_size=6, seed=random_seed)

    # Number of episodes to decay the epsilon linearly
    decay_steps = int(epsilon_decay_proportion * episodes * iters)
    initial_epsilon = epsilon  # Store initial epsilon for decay calculation

    # # Curriculum schedule: split episodes into 4 equal parts
    # phase_len = episodes // 4

    states, _ = envs.reset()
    total_steps = 0

    for episode in range(episodes // num_envs):
        # # Set difficulty based on curriculum phase (applies to all envs in batch)
        # if episode < phase_len:
        #     difficulty = 0  # easy
        # elif episode < 2 * phase_len:
        #     difficulty = 1  # medium
        # elif episode < 3 * phase_len:
        #     difficulty = 2  # hard
        # else:
        #     difficulty = None  # no difficulty, just train on any problem

        print(f"Episode batch {episode + 1}/{episodes // num_envs} - Epsilon: {epsilon:.4f}")

        # if no_gui:
        #     env_gui = False
        # else:
        #     env_gui = episode % 10 == 0 and episode != 0

        for _ in trange(iters):
            # take action + step in `num_envs` parallel environments
            actions = [agent.take_action(state) for state in states]
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            for j in range(num_envs):
                done = terminateds[j] or truncateds[j]
                agent.update(states[j], actions[j], rewards[j], next_states[j], done)
            states = next_states

            # Update the agent's epsilon value by `num_envs` steps
            total_steps += num_envs
            if total_steps < decay_steps:
                frac = total_steps / decay_steps
                epsilon = initial_epsilon - frac * (initial_epsilon - epsilon_min)
            else:
                epsilon = epsilon_min
            agent.epsilon = epsilon
            if all(terminateds):
                break

    # model_path = f"models/dqn_{name}_final.pth"
    # agent.save(model_path)
    # print(f"Saved trained model to -> {model_path}")

    # agent.epsilon = 0  # Set epsilon to 0 for evaluation
    # Environment.evaluate_agent(agent, iters)


if __name__ == '__main__':
    args = parse_args()
    main(args.name, args.no_gui, args.episodes, args.iters, args.random_seed, args.epsilon, 
         args.epsilon_min, args.epsilon_decay_proportion)
