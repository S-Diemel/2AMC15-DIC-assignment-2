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

def run_episode(env, agent, name_exp, delivery_zones=None, max_steps=1000, no_gui=True, agent_start_pos=False):
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


def _run_episode_override(env, agent, initial_state, max_steps=1000):
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

def experiment_stochasticity(agents, levels=(0.1, 0.3, 0.5, 0.7, 0.9,
                                             ), reps=20):
    all_results = []
    for agent in agents:
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
        all_results.append((success_rates, avg_steps, std_steps, avg_rewards, std_rewards, all_steps, all_rewards))
    plot_results(
        levels, all_results, "Stochasticity", "stochasticity_plot.png", boxplot=True
    )
    append_to_txt("Stochasticity", levels, all_results)

def experiment_difficulty(agents, levels=(0,1,2,3,4,5), reps=20):
    all_results = []
    for agent in agents:
        xs, srates, avg_steps, avg_rews, all_steps, all_rewards = [], [], [], [], [], []
        for k in tqdm(levels, desc="Difficulty"):
            random.seed(100 + k)
            np.random.seed(100 + k)
            obstacle_centers = []
            for _ in range(k):
                center = np.array(
                    sample_one_point_outside(
                        Environment().all_obstacles,
                        Environment().agent_radius,
                        (0, 0, Environment().width, Environment().height)
                    )
                )
                obstacle_centers.append(center)
            extra_obstacles = []
            size = 0.5
            for (cx, cy) in obstacle_centers:
                half = size / 2
                extra_obstacles.append((cx-half, cy-half, cx+half, cy+half))
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
            all_steps.append(steps_list)
            all_rewards.append(rew_list)
        all_results.append((srates, avg_steps, [0]*len(xs), avg_rews, [0]*len(xs), all_steps, all_rewards))
    plot_results(
        xs, all_results, "Difficulty", "difficulty_plot.png", boxplot=False
    )
    append_to_txt("Difficulty", xs, all_results)

def experiment_target_distance(agents, reps=20, max_steps=350):
    levels = [2, 4, 6, 8, 10, 12]
    all_results = []
    for agent in agents:
        success_rates, avg_steps, avg_rewards, all_steps, all_rewards = [], [], [], [], []
        for dist in tqdm(levels, desc="Processing Target Distance Levels"):
            successes = 0
            steps_list = []
            rew_list = []
            for _ in trange(reps, desc=f"Dist {dist}"):
                env = Environment()
                goal = None
                start = None
                for _ in range(20):
                    _, _ = env.reset()
                    start = env.agent_pos.copy()
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
                        break
                if goal is None:
                    succ = False
                    st = max_steps
                    rw = 0.0 - st
                else:
                    env.item_starts     = [start]
                    env.delivery_points = [goal]
                    env.delivered       = [False]
                    env.carrying        = -1
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
            all_steps.append(steps_list)
            all_rewards.append(rew_list)
        all_results.append((success_rates, avg_steps, [0]*len(levels), avg_rewards, [0]*len(levels), all_steps, all_rewards))
    plot_results(
        levels, all_results, "TargetDistance", "target_distance_plot.png", boxplot=False
    )
    append_to_txt("TargetDistance", levels, all_results)

def plot_results(x_values, all_results, xlabel, filename, boxplot=True):
    plt.figure(figsize=(15, 4))
    agent_colors = ['orange', 'green']
    agent_labels = ['DQN', 'PPO']
    plot_positions = np.arange(len(x_values))  # Evenly spaced positions

    # Success Rate
    plt.subplot(1, 3, 1)
    for idx, (success_rates, *_ ) in enumerate(all_results):
        plt.plot(plot_positions, success_rates, marker="o", label=agent_labels[idx], color=agent_colors[idx])
    plt.title(f"Success Rate vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xticks(plot_positions, [str(x) for x in x_values])
    plt.legend()

    # Steps
    plt.subplot(1, 3, 2)
    if boxplot:
        for idx, (*_, all_steps) in enumerate([r[1:6] + (r[5],) for r in all_results]):
            plt.boxplot(all_steps, positions=plot_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color=agent_colors[idx]),
                        whiskerprops=dict(color='blue'), capprops=dict(color='blue'),
                        flierprops=dict(markerfacecolor='blue', marker='o', alpha=0.2))
    for idx, (_, avg_steps, *_ ) in enumerate(all_results):
        plt.plot(plot_positions, avg_steps, marker="o", color=agent_colors[idx], label=agent_labels[idx])
    plt.title(f"Steps per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Steps per Successful Episode")
    plt.grid(True)
    plt.xticks(plot_positions, [str(x) for x in x_values])
    plt.legend()

    # Rewards
    plt.subplot(1, 3, 3)
    if boxplot:
        for idx, (*_, all_rewards) in enumerate([r[1:7] + (r[6],) for r in all_results]):
            plt.boxplot(all_rewards, positions=plot_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor='mistyrose', color='red'),
                        medianprops=dict(color=agent_colors[idx]),
                        whiskerprops=dict(color='red'), capprops=dict(color='red'),
                        flierprops=dict(markerfacecolor='red', marker='o', alpha=0.2))
    for idx, (*_, avg_rewards, _) in enumerate([r[1:5] + (r[3], r[5]) for r in all_results]):
        plt.plot(plot_positions, avg_rewards, marker="o", color=agent_colors[idx], label=agent_labels[idx])
    plt.title(f"Reward per Successful Episode vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Reward per Successful Episode")
    plt.grid(True)
    plt.xticks(plot_positions, [str(x) for x in x_values])
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.close()

def append_to_txt(title, x, all_results):
    with open(RESULTS_TXT, "a") as f:
        f.write(f"\n\n{title} Experiment\n")
        for idx, (success_rates, avg_steps, _, avg_rewards, _, _, _) in enumerate(all_results):
            f.write(f"Agent {idx+1}\n")
            f.write(f"{title}\tSuccessRate\tAvgSteps\tAvgReward\n")
            for a, sr, s, r in zip(x, success_rates, avg_steps, avg_rewards):
                f.write(f"{a}\t\t{sr:.2f}\t\t{s:.2f}\t\t{r:.2f}\n")

def evaluate(model_path1: Path, model_path2: Path):
    print('Loading dqn: agent 1 from:', model_path1)
    agent1 = DQNAgent.load(str(model_path1), state_size=12, action_size=5)
    agent1.epsilon = 0.0
    print('Loading ppo: agent 2 from:', model_path2)
    # agent2 = DQNAgent.load(str(model_path2), state_size=12, action_size=5)
    # agent2.epsilon = 0.0
    
    agent2 = PPOAgent(state_size=12, action_size=5, seed=12, num_envs=1)
    agent2.load(model_path2)

    agents = [agent1, agent2]
    reps = 20
    print("Agents Loaded Successfully!")
    experiment_stochasticity(agents, reps=reps)
    print("Experiment Stochasticity Finished!")
    experiment_difficulty(agents, reps=reps)
    print("Experiment Difficulty Finished!")
    experiment_target_distance(agents, reps=reps)
    print("Experiment Target Distance Finished!")


    # python experiments_dqn_ppo.py .\models\best_dqn.pth .\models\ppo_after_training_10000_v3.pth 


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model1", type=Path, help="Path to first .pth checkpoint")
    p.add_argument("model2", type=Path, help="Path to second .pth checkpoint")
    args = p.parse_args()
    evaluate(args.model1, args.model2)
