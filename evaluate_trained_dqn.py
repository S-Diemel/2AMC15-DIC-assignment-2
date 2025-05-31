import torch
from pathlib import Path
from world import Environment
from agents.dqn import DQNAgent


def full_evaluation(env, agent, n_episodes=10, max_steps=1000):
    """More extensive evaluation of the agent's performance."""
    returns = []
    successes = 0
    for _ in range(n_episodes):
        state = env.reset(no_gui=True)
        total_r = 0
        for t in range(max_steps):
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            total_r += reward
            state = next_state
            if done:
                if info.get("target_reached", False):
                    successes += 1
                break
        returns.append(total_r)
    avg_return = sum(returns) / n_episodes
    success_rate = successes / n_episodes
    print(f"Avg return over {n_episodes} eps: {avg_return:.2f}")
    print(f"Success rate: {success_rate*100:.1f}%")
    return returns, success_rate


def evaluate(model_path: Path, grid_path: Path,
             iters: int = 1000, sigma: float = 0.0,
             random_seed: int = 0):
    # Load your environment
    env = Environment(grid_path, no_gui=True, sigma=sigma,
                      random_seed=random_seed,
                      agent_start_pos=(1,1),
                      target_positions=[(1,12)])
    # Load agent
    # NOTE: Must pass the same state_size and action_size used in training
    agent = DQNAgent.load(str(model_path),
                          state_size=9,
                          action_size=4,
                          seed=random_seed)
    # Run evaluation
    # Basic evaluation
    # Environment.evaluate_agent(grid_path, agent, iters, sigma, random_seed=random_seed)
    # Own evaluation function
    returns, success_rate = full_evaluation(env, agent, n_episodes=10, max_steps=iters)
    return returns, success_rate


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    p.add_argument("grid",  type=Path, help="Grid file")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--seed",  type=int, default=0)
    args = p.parse_args()
    evaluate(args.model, args.grid, args.iters, args.sigma, args.seed)

    # Command-line example: python evaluate_trained_dqn.py models/dqn_A1_grid_test.pth grid_configs/A1_grid.npy

