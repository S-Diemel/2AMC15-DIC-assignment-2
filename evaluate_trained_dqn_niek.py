# evaluate_trained_dqn.py

import torch
from pathlib import Path
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent
from world.environment import Environment
from tqdm import trange
import numpy as np
import argparse

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TXT = RESULTS_DIR / "stochasticity_results.txt"


def run_episode(env, agent, max_steps=1000):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    for _ in range(max_steps):
        action = agent.take_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return terminated, steps


def experiment_stochasticity(agent, levels=(0.0, 0.1, 0.3, 0.5)):
    success_rates = []
    avg_steps = []
    for sigma in levels:
        successes = 0
        steps_record = []
        for _ in range(10):
            env = Environment(sigma=sigma)
            success, steps = run_episode(env, agent)
            if success:
                successes += 1
            steps_record.append(steps)
        success_rates.append(successes / 10)
        avg_steps.append(np.mean(steps_record))
    plot_results(levels, success_rates, avg_steps, "Stochasticity", "stochasticity_plot.png")
    append_to_txt("Stochasticity", levels, success_rates, avg_steps)


def experiment_difficulty(agent):
    # Custom obstacle layouts simulating difficulty
    levels = [0, 1, 2, 3, 4, 5]
    success_rates = []
    avg_steps = []
    for i in levels:
        # Generate i boxes as extra obstacles to simulate increasing difficulty
        extra_obstacles = [(15 + 0.2 * j, 4 + 0.2 * j, 15.3 + 0.2 * j, 4.3 + 0.2 * j) for j in range(i)]
        successes = 0
        steps_record = []
        for _ in range(10):
            env = Environment(extra_obstacles=extra_obstacles)
            env.reset(extra_obstacles=extra_obstacles)
            success, steps = run_episode(env, agent)
            if success:
                successes += 1
            steps_record.append(steps)
        success_rates.append(successes / 10)
        avg_steps.append(np.mean(steps_record))
    plot_results(levels, success_rates, avg_steps, "Difficulty", "difficulty_plot.png")
    append_to_txt("Difficulty", levels, success_rates, avg_steps)


def experiment_target_distance(agent):
    # Custom levels that adjust item and delivery separation
    levels = [0, 1, 2]
    success_rates = []
    avg_steps = []
    for i in levels:
        margin = 2 + i * 1.5
        delivery_zones = [
            (15 - margin, 1, 16 - margin, 2),
            (15 + margin, 8, 16 + margin, 9)
        ]
        successes = 0
        steps_record = []
        for _ in range(10):
            env = Environment()
            env.delivery_zones = delivery_zones
            env.reset()
            success, steps = run_episode(env, agent)
            if success:
                successes += 1
            steps_record.append(steps)
        success_rates.append(successes / 10)
        avg_steps.append(np.mean(steps_record))
    plot_results(levels, success_rates, avg_steps, "TargetDistance", "target_distance_plot.png")
    append_to_txt("TargetDistance", levels, success_rates, avg_steps)


def plot_results(x_values, success_rates, avg_steps, xlabel, filename):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_values, success_rates, marker="o")
    plt.title(f"Success Rate vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_values, avg_steps, marker="o", color="orange")
    plt.title(f"Average Steps vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Average Steps")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.close()


def append_to_txt(title, x, success_rates, steps):
    with open(RESULTS_TXT, "a") as f:
        f.write(f"\n\n{title} Experiment\n")
        f.write(f"{title}\tSuccessRate\tAvgSteps\n")
        for a, sr, s in zip(x, success_rates, steps):
            f.write(f"{a}\t\t{sr:.2f}\t\t{s:.2f}\n")


def evaluate(model_path: Path):
    agent = DQNAgent.load(str(model_path), state_size=11, action_size=6)
    agent.epsilon = 0.0
    experiment_stochasticity(agent)
    experiment_difficulty(agent)
    experiment_target_distance(agent)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)
