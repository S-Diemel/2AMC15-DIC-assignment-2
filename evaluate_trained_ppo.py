import torch
from pathlib import Path
from world.environment_ppo import Environment
from agents.dqn import DQNAgent
from tqdm import trange
import time


def evaluate(model_path: Path):
    # Load agent
    # NOTE: Must pass the same state_size and action_size used in training
    agent = DQNAgent.load(str(model_path),
                          state_size=14,
                          action_size=6)
    for i in range(5):
        evaluate_agent_training(agent, 1000, False, 3, 4, 0.25, 0)


def evaluate_agent_training(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step):
    next_state = 0
    env = Environment()
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step, agent_start_pos=(2,2))
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
    env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)

    # Command-line example: python evaluate_trained_dqn.py models/dqn_A1_grid_test.pth grid_configs/A1_grid.npy

