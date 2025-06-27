from pathlib import Path
from world.environment import Environment
from agents.dqn import DQNAgent
from tqdm import trange


def evaluate(model_path: Path):
    l =[]
    agent = DQNAgent.load(str(model_path),
                          state_size=12,
                          action_size=5, seed=60)
    for _ in range(100):
        l.append(evaluate_agent_training(agent, 1000, False, 3, 3, 0.25, 0, 0))
    print(l)
    print(sum(l))


def evaluate_agent_training(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step, epsilon, sigma):

    env = Environment(sigma=sigma)
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, 
                         battery_drain_per_step=battery_drain_per_step)
    agent.epsilon=epsilon
    for _ in trange(iters):
        action = agent.take_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        env.render()
        state = next_state
        if terminated:
            return True
        if truncated:
            return False
    return False


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)
    # Command-line example: python evaluate_trained_dqn.py models/dqn_A1_grid_test.pth grid_configs/A1_grid.npy