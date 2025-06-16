import torch
from pathlib import Path
from world.environment import Environment
from agents.dqn import DQNAgent
from tqdm import trange
import time

# def full_evaluation(env, agent, n_episodes=10, max_steps=1000):
#     """More extensive evaluation of the agent's performance."""
#     returns = []
#     successes = 0
#     for _ in range(n_episodes):
#         state = env.reset_env(no_gui=False)
#         total_r = 0
#         for t in range(max_steps):
#             action = agent.take_action(state)
#             next_state, reward, done, info = env.step(action)
#             total_r += reward
#             state = next_state
#             if done:
#                 if info.get("target_reached", False):
#                     successes += 1
#                 break
#         returns.append(total_r)
#     avg_return = sum(returns) / n_episodes
#     success_rate = successes / n_episodes
#     print(f"Avg return over {n_episodes} eps: {avg_return:.2f}")
#     print(f"Success rate: {success_rate*100:.1f}%")
#     return returns, success_rate


def evaluate(model_path: Path):
    # Load agent
    # NOTE: Must pass the same state_size and action_size used in training
    # 3, 6
    #4, 5
    #3, 8
    agent = DQNAgent.load(str(model_path),
                          state_size=11,
                          action_size=5, seed=6)
    for i in range(5):
        evaluate_agent_training(agent, 1000, False, 3, 1, 0.25, 0)


def evaluate_agent_training(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step, epsilon):
    next_state = 0
    env = Environment()
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step)
    agent.epsilon=epsilon
    for i in trange(iters):
        # env.render()
        # time.sleep(2)
        # print(state)

        action = agent.take_action(state)
        # print(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        # print(reward)
        # print('\n')
        termination_flag = terminated or i == iters-1
        state = next_state
        if terminated or truncated:
            break


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)

    # Command-line example: python evaluate_trained_dqn.py models/dqn_A1_grid_test.pth grid_configs/A1_grid.npy

