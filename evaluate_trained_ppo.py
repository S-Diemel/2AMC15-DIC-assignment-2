import torch
from pathlib import Path
from world.environment import Environment
from agents.ppo import PPOAgent
from tqdm import trange

def evaluate(model_path: Path):
    l =[]
    agent = PPOAgent(state_size=12, action_size=5, seed=60, num_envs=1)
    agent.load(model_path)
    for i in range(100):
        l.append(evaluate_agent_training(agent, 1000, True, 3, 3, 0.25, 0))
    print(l)
    print(sum(l))



def evaluate_agent_training(agent, iters, no_gui, difficulty, number_of_items, battery_drain_per_step, sigma):

    env = Environment(sigma=sigma)
    state, _ = env.reset(no_gui=no_gui, difficulty=difficulty, number_of_items=number_of_items, battery_drain_per_step=battery_drain_per_step)
    for _ in trange(iters):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
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

    # Command-line example: python evaluate_trained_ppo.py models/ppo_after_training_2000_final.pth