import torch
from pathlib import Path
from world.environment import Environment
from agents.dqn import DQNAgent
from tqdm import trange

def evaluate(model_path: Path):
    agent = DQNAgent.load(str(model_path),
                          state_size=12,
                          action_size=5, seed=5)
    for i in range(5):
        evaluate_agent_training(agent, 1000, False, 3, 4, 0.25, 0)


def evaluate_agent_training(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step, epsilon):

    env = Environment()
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step, agent_start_pos=(2,2))
    agent.epsilon=epsilon
    for _ in trange(iters):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
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

