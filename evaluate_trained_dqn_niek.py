# evaluate_trained_dqn.py
import matplotlib.patches as patches
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent
from world.environment import Environment
from tqdm import tqdm
from tqdm import trange
import numpy as np
import argparse
import random
from world.utils.env_init import (
    create_delivery_zones,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TXT = RESULTS_DIR / "stochasticity_results.txt"

#note change enviroment to experiment = True to run rewards

def run_episode(env, agent, name_exp, delivery_zones=None, max_steps=350, no_gui = True, agent_start_pos= False):
    """Changed max steps to 500, we need to decide on a good value
    change enviroment to experiment = True to run rewards"""
    if name_exp != "target_distance":
        state, _ = env.reset(no_gui= no_gui)
    else:
        state, _ = env.reset(no_gui=no_gui, agent_start_pos=agent_start_pos, delivery_zones=delivery_zones)
    total_reward = 0
    steps = 0
    for _ in range(max_steps):
        action = agent.take_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if not no_gui:
            env.render()
        if terminated or truncated:
            break
    return terminated, steps, total_reward  # <-- Return total_reward


def experiment_stochasticity(agent, levels=(0.01, 0.2, 0.4, 0.6,0.8, 0.99), reps=20):
    success_rates = []
    avg_steps = []
    std_steps = []
    avg_rewards = []
    std_rewards = []
    all_steps = []
    all_rewards = []
    for sigma in tqdm(levels, desc="Processing Stochasticity Levels"):
        successes = 0
        steps_record = []
        rewards_record = []
        steps_successful = []
        rewards_successful = []
        for _ in range(reps):
            env = Environment(sigma=sigma)
            success, steps, total_reward = run_episode(env, agent, "stochasticity")
            if success:
                successes += 1
                steps_record.append(steps)
                rewards_record.append(total_reward)
                steps_successful.append(steps)
                rewards_successful.append(total_reward)
        success_rates.append(successes / reps)
        avg_steps.append(np.mean(steps_record) if steps_record else 0)
        std_steps.append(np.std(steps_record) if steps_record else 0)
        avg_rewards.append(np.mean(rewards_record) if rewards_record else 0)
        std_rewards.append(np.std(rewards_record) if rewards_record else 0)
        all_steps.append(steps_successful)
        all_rewards.append(rewards_successful)
    plot_results(levels, np.array(success_rates), np.array(avg_steps), np.array(std_steps),
                 np.array(avg_rewards), np.array(std_rewards), "Stochasticity", "stochasticity_plot.png",
                 all_steps, all_rewards)
    append_to_txt("Stochasticity", levels, success_rates, avg_steps, avg_rewards)

def experiment_difficulty(agent, reps =20):
    levels = [0, 1, 2, 3, 4, 5]
    success_rates = []
    avg_steps = []
    std_steps = []
    avg_rewards = []
    std_rewards = []
    all_steps = []
    all_rewards = []

    for i in tqdm(levels, desc="Processing Difficulty Levels"):
        extra_obstacles = [(15 + 0.2 * j, 4 + 0.2 * j, 15.3 + 0.2 * j, 4.3 + 0.2 * j) for j in range(i)]
        successes = 0
        steps_record = []
        rewards_record = []
        steps_successful = []
        rewards_successful = []
        for _ in range(reps):
            env = Environment(extra_obstacles=extra_obstacles)
            env.reset(extra_obstacles=extra_obstacles)
            success, steps, total_reward = run_episode(env, agent, "difficulty")
            if success:
                successes += 1
                steps_record.append(steps)
                rewards_record.append(total_reward)
                steps_successful.append(steps)
                rewards_successful.append(total_reward)
        success_rates.append(successes / reps)
        avg_steps.append(np.mean(steps_record) if steps_record else 0)
        std_steps.append(np.std(steps_record) if steps_record else 0)
        avg_rewards.append(np.mean(rewards_record) if rewards_record else 0)
        std_rewards.append(np.std(rewards_record) if rewards_record else 0)
        all_steps.append(steps_successful)
        all_rewards.append(rewards_successful)

    plot_results(levels, np.array(success_rates), np.array(avg_steps), np.array(std_steps),
                 np.array(avg_rewards), np.array(std_rewards), "Difficulty", "difficulty_plot.png",
                 all_steps, all_rewards)
    append_to_txt("Difficulty", levels, success_rates, avg_steps, avg_rewards)

def experiment_target_distance(agent, reps=20):
    levels = [0, 1, 2]
    success_rates = []
    avg_steps = []
    std_steps = []
    avg_rewards = []
    std_rewards = []
    all_steps = []
    all_rewards = []

    for i in tqdm(levels, desc="Processing Target Distance Levels"):
        env = Environment()
        zones = create_delivery_zones(env.racks, env.width, env.height, margin=0.5 + i * 0.1)
        random.seed(15)
        np.random.seed(15)
        delivery_zones = random.sample(zones, 3)
        eval = False
        if eval:
            plot_delivery_zones(delivery_zones, env.width, env.height)
        successes = 0
        steps_record = []
        rewards_record = []
        steps_successful = []
        rewards_successful = []
        for _ in trange(reps, desc=f"Level {i}"):
            env = Environment()
            env.reset(agent_start_pos= (2,6), delivery_zones=delivery_zones)
            success, steps, total_reward = run_episode(env, agent, "target_distance", agent_start_pos= (2,6))
            if success:
                successes += 1
                steps_record.append(steps)
                rewards_record.append(total_reward)
                steps_successful.append(steps)
                rewards_successful.append(total_reward)
        success_rates.append(successes / reps)
        avg_steps.append(np.mean(steps_record) if steps_record else 0)
        std_steps.append(np.std(steps_record) if steps_record else 0)
        avg_rewards.append(np.mean(rewards_record) if rewards_record else 0)
        std_rewards.append(np.std(rewards_record) if rewards_record else 0)
        all_steps.append(steps_successful)
        all_rewards.append(rewards_successful)

    plot_results(levels, np.array(success_rates), np.array(avg_steps), np.array(std_steps),
                 np.array(avg_rewards), np.array(std_rewards), "TargetDistance", "target_distance_plot.png",
                 all_steps, all_rewards)
    append_to_txt("TargetDistance", levels, success_rates, avg_steps, avg_rewards)

def plot_results(x_values, success_rates, avg_steps, std_steps, avg_rewards, std_rewards, xlabel, filename,
                 all_steps=None, all_rewards=None):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x_values, success_rates, marker="o")
    plt.title(f"Success Rate vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    # Boxplot for steps (successful episodes)
    if all_steps is not None:
        plt.boxplot(all_steps, positions=x_values, widths=0.06*(max(x_values)-min(x_values)), patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='orange'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    flierprops=dict(markerfacecolor='blue', marker='o', alpha=0.2))
        # Mean of successful episodes (blue line/dots)
        mean_success_steps = [np.mean(steps) if len(steps) > 0 else np.nan for steps in all_steps]
        #plt.plot(x_values, mean_success_steps, marker="D", color="blue", label="Mean (successes)")
    # Trend line for average successful steps (orange)
    plt.plot(x_values, avg_steps, marker="o", color="orange", label="Mean (successes)")
    plt.title(f"Steps per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Steps per Successful Episode")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(1, 3, 3)
    # Boxplot for rewards (successful episodes)
    if all_rewards is not None:
        plt.boxplot(all_rewards, positions=x_values, widths=0.06*(max(x_values)-min(x_values)), patch_artist=True,
                    boxprops=dict(facecolor='mistyrose', color='red'),
                    medianprops=dict(color='green'),
                    whiskerprops=dict(color='red'),
                    capprops=dict(color='red'),
                    flierprops=dict(markerfacecolor='red', marker='o', alpha=0.2))
        # Mean of successful episodes (blue line/dots)
        mean_success_rewards = [np.mean(rewards) if len(rewards) > 0 else np.nan for rewards in all_rewards]
        #plt.plot(x_values, mean_success_rewards, marker="D", color="blue", label="Mean (successes)")
    # Trend line for average reward (green)
    plt.plot(x_values, avg_rewards, marker="o", color="green", label="Mean (successes)")
    plt.title(f"Reward (collisions) per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Reward (collisions) per Successful Episode")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.close()


def append_to_txt(title, x, success_rates, steps, rewards):
    with open(RESULTS_TXT, "a") as f:
        f.write(f"\n\n{title} Experiment\n")
        f.write(f"{title}\tSuccessRate\tAvgSteps\tAvgReward\n")
        for a, sr, s, r in zip(x, success_rates, steps, rewards):
            f.write(f"{a}\t\t{sr:.2f}\t\t{s:.2f}\t\t{r:.2f}\n")

def plot_delivery_zones(zones, env_width, env_height):
    fig, ax = plt.subplots()
    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    for zone in zones:
        x1, y1, x2, y2 = zone
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title("Delivery Zones")
    plt.show()


def evaluate(model_path: Path):
    agent = DQNAgent.load(str(model_path), state_size=15, action_size=6)
    agent.epsilon = 0.0
    reps = 50
    print("Agent Loaded Successfully!")
    experiment_stochasticity(agent, reps = reps)
    print("Experiment Stochasticity Finished!")
    experiment_difficulty(agent, reps = reps)
    print("Experiment Difficulty Finished!")
    experiment_target_distance(agent, reps = reps)
    print("Experiment Target Distance Finished!")


if __name__ == "__main__":  
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)
