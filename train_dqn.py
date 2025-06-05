"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from world.environment import WarehouseEnv
from agents.dqn import DQNAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--episodes", type=int, default=1000,  # 1000
                   help="Number of episodes to train the agent for. Each episode is completed by either reaching the target, or putting `iter` steps.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=None,
                   help="Random seed value for the environment.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_min", type=float, default=0.01,
                   help="Minimum epsilon value for the epsilon-greedy policy.")
    p.add_argument("--epsilon_decay_proportion", type=float, default=0.7,
                   help="Proportion of training to decay epsilon over. " \
                   "0.5 means that halfway of the training procedure we the epsilon has reached in minimum value of 0.1.")
    return p.parse_args()


def main(no_gui: bool, episodes: int, iters: int, fps: int,
         sigma: float, random_seed: int, epsilon: float,
         epsilon_min: float, epsilon_decay_proportion: float):  
    """Main loop of the program."""

        
    # Set up the environment
    env = WarehouseEnv()  # , agent_start_pos=(1, 1), target_positions=[(1, 12)])
    
    # Initialize dqn agent
    agent = DQNAgent(state_size=13, action_size=5, seed=random_seed)  # note we have set the state features and actions ourselves so hardcoded here

    # Number of episodes to decay the epsilon linearly
    decay_episodes = int(epsilon_decay_proportion * episodes)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes} - Epsilon: {epsilon:.4f}")
        # Always reset the environment to initial state
        # state = env.reset()

        env_gui = episode % 1 == 0 and episode != 0
        #env_gui = True
        state = env.reset(no_gui=not env_gui)

        for i in trange(iters):
            env.render()
            # Agent takes an action based on the latest observation and info.
            action = agent.take_action(state)
            # The action is performed in the environment
            next_state, reward, terminated = env.step(action)

            # Flag terminated upon reaching target or if the episode limit is reached
            termination_flag = terminated or i == iters-1

            # Flag terminated only upon reaching target
            # termination_flag = terminated

            agent.update(state, action, reward, next_state, termination_flag)

            state = next_state

            # If the final state is reached, stop. But before stopping, we want to incorportate the reward in the Q-value update.
            if terminated:
                break
        
        # Decay epsilon (exploration rate)
        if episode < decay_episodes:
            frac = episode / decay_episodes
            epsilon = args.epsilon - frac * (args.epsilon - args.epsilon_min)
        else:
            epsilon = args.epsilon_min

        # Update the agent's epsilon    
        agent.epsilon = epsilon

    # after all episodes for this grid
    model_path = f"models/dqn_{1}_test.pth"
    agent.save(model_path)
    print(f"Saved trained model to -> {model_path}")

    agent.epsilon = 0  # Set epsilon to 0 for evaluation
    # Evaluate the agent



if __name__ == '__main__':
    args = parse_args()
    main(args.no_gui, args.episodes, args.iter, args.fps, args.sigma, args.random_seed, args.epsilon, args.epsilon_min, args.epsilon_decay_proportion)