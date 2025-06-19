# evaluate_trained_dqn.py

import matplotlib.patches as patches
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent
from world.environment import Environment
from tqdm import tqdm, trange
import numpy as np
import argparse
import random
from world.utils.env_init import create_delivery_zones
from world.utils.env_reset import sample_one_point_outside
import math
from agents.ppo import PPOAgent

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TXT = RESULTS_DIR / "stochasticity_results.txt"

# note: change environment to experiment=True to run rewards

def run_episode(env, agent, name_exp, delivery_zones=None, max_steps=350, no_gui=True, agent_start_pos=False):
    if name_exp != "target_distance":
        state, _ = env.reset(no_gui=no_gui)
    else:
        state, _ = env.reset(no_gui=no_gui,
                             agent_start_pos=agent_start_pos,
                             delivery_zones=delivery_zones)
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
    return terminated, steps, total_reward  # original reward return


def experiment_stochasticity(agent, levels=(0.01, 0.2, 0.4, 0.6, 0.8, 0.99), reps=20):
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

    plot_results(
        levels,
        np.array(success_rates),
        np.array(avg_steps),
        np.array(std_steps),
        np.array(avg_rewards),
        np.array(std_rewards),
        "Stochasticity",
        "stochasticity_plot.png",
        all_steps,
        all_rewards
    )
    append_to_txt("Stochasticity", levels, success_rates, avg_steps, avg_rewards)


def experiment_difficulty(agent, levels=(0,1,2,3,4,5), reps=20):
    from world.utils.env_reset import sample_one_point_outside
    xs, srates, avg_steps, avg_rews = [], [], [], []

    for k in tqdm(levels, desc="Difficulty"):
        # reproducible per level
        random.seed(100 + k)
        np.random.seed(100 + k)

        # 1) sample k obstacle centers in free space
        obstacle_centers = []
        for _ in range(k):
            center = np.array(
                sample_one_point_outside(
                    Environment().all_obstacles,        # existing obstacles
                    Environment().agent_radius,
                    (0, 0, Environment().width, Environment().height)
                )
            )
            obstacle_centers.append(center)

        # 2) build small square obstacles around each center
        extra_obstacles = []
        size = 0.5  # 0.5m x 0.5m boxes
        for (cx, cy) in obstacle_centers:
            half = size / 2
            extra_obstacles.append((cx-half, cy-half, cx+half, cy+half))

        # 3) run reps trials
        successes, steps_list, rew_list = 0, [], []
        for _ in range(reps):
            env = Environment(extra_obstacles=extra_obstacles)
            env.reset(extra_obstacles=extra_obstacles)
            succ, st, rw = run_episode(env, agent, "difficulty")
            if succ:
                successes += 1
                steps_list.append(st)
                rew_list.append(rw)

        xs.append(k)
        srates.append(successes / reps)
        avg_steps.append(np.mean(steps_list) if steps_list else 0)
        avg_rews.append(np.mean(rew_list) if rew_list else 0)

    # plot & log exactly as before
    plot_results(
        xs, np.array(srates), np.array(avg_steps), np.array([0]*len(xs)),
        np.array(avg_rews), np.array([0]*len(xs)),
        "Difficulty", "difficulty_plot.png"
    )
    append_to_txt("Difficulty", xs, srates, avg_steps, avg_rews)



def experiment_target_distance(agent, reps=20, max_steps=350):
    """
    For each distance level, attempt up to 20 different start positions.
    For each start, try up to 100 goal samples at approx. dist ± tol.
    If no valid goal after 20 starts, record failure.
    """
    from world.utils.env_reset import sample_one_point_outside

    levels = [2, 4, 6, 8, 10, 12]
    success_rates, avg_steps, avg_rewards = [], [], []

    for dist in tqdm(levels, desc="Processing Target Distance Levels"):
        successes = 0
        steps_list = []
        rew_list = []

        for _ in trange(reps, desc=f"Dist {dist}"):
            env = Environment()
            goal = None
            start = None

            # Try up to 20 different start positions
            for _ in range(20):
                _, _ = env.reset()
                start = env.agent_pos.copy()

                # Try up to 100 goal samples for this start
                tol = 0.5
                for _ in range(100):
                    candidate = np.array(
                        sample_one_point_outside(
                            env.all_obstacles,
                            env.agent_radius,
                            (0, 0, env.width, env.height)
                        )
                    )
                    if abs(np.linalg.norm(candidate - start) - dist) <= tol:
                        goal = candidate
                        break

                if goal is not None:
                    break  # found a valid goal for this start

            # If no goal found after 20 starts => immediate failure
            if goal is None:
                succ = False
                st = max_steps
                rw = 0.0 - st
            else:
                # override to single-item delivery
                env.item_starts     = [start]
                env.delivery_points = [goal]
                env.delivered       = [False]
                env.carrying        = -1

                # run from custom initial state
                state = env._compute_features()
                succ, st, _ = _run_episode_override(env, agent, state, max_steps=max_steps)
                rw = (100.0 if succ else 0.0) - st

            if succ:
                successes += 1
            steps_list.append(st)
            rew_list.append(rw)

        success_rates.append(successes / reps)
        avg_steps.append(np.mean(steps_list))
        avg_rewards.append(np.mean(rew_list))

    # plot & log as before
    plot_results(
        levels,
        np.array(success_rates),
        np.array(avg_steps),
        np.zeros(len(levels)),       # std unused
        np.array(avg_rewards),
        np.zeros(len(levels)),       # std unused
        "TargetDistance",
        "target_distance_plot.png"
    )
    append_to_txt("TargetDistance", levels, success_rates, avg_steps, avg_rewards)



def _run_episode_override(env, agent, initial_state, max_steps=350):
    state = initial_state
    steps = 0
    for _ in range(max_steps):
        action = agent.take_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    success = terminated
    # preserve original reward accumulation for logging
    return success, steps, (100.0 if success else 0.0) - float(steps)


def plot_results(x_values, success_rates, avg_steps, std_steps, avg_rewards, std_rewards,
                 xlabel, filename, all_steps=None, all_rewards=None):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x_values, success_rates, marker="o")
    plt.title(f"Success Rate vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)

    plt.subplot(1, 3, 2)
    if all_steps is not None:
        plt.boxplot(all_steps,
                    positions=x_values,
                    widths=0.06 * (max(x_values) - min(x_values)),
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='orange'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    flierprops=dict(markerfacecolor='blue', marker='o', alpha=0.2))
    plt.plot(x_values, avg_steps, marker="o", color="orange", label="Mean (successes)")
    plt.title(f"Steps per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Steps per Successful Episode")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(1, 3, 3)
    if all_rewards is not None:
        plt.boxplot(all_rewards,
                    positions=x_values,
                    widths=0.06 * (max(x_values) - min(x_values)),
                    patch_artist=True,
                    boxprops=dict(facecolor='mistyrose', color='red'),
                    medianprops=dict(color='green'),
                    whiskerprops=dict(color='red'),
                    capprops=dict(color='red'),
                    flierprops=dict(markerfacecolor='red', marker='o', alpha=0.2))
    plt.plot(x_values, avg_rewards, marker="o", color="green", label="Mean (successes)")
    plt.title(f"Reward per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Reward per Successful Episode")
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



    print('Loading agent from:', model_path)
    agent = DQNAgent.load(str(model_path), state_size=12, action_size=5)
    agent.epsilon = 0.0


        
    # agent = PPOAgent(state_size=12, action_size=5, seed=12, num_envs=1)
    # agent.load(f"models/best_ppo_yet.pth")
    # waiting for a ppo.
    reps = 50
    print("Agent Loaded Successfully!")
    experiment_stochasticity(agent, reps=reps)
    print("Experiment Stochasticity Finished!")
    experiment_difficulty(agent, reps=reps)
    print("Experiment Difficulty Finished!")
    experiment_target_distance(agent, reps=reps)
    print("Experiment Target Distance Finished!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model)
