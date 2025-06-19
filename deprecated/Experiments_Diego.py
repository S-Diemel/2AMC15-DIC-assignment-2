import torch
import os
import sys
from pathlib import Path
from world import Environment
from agents.dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
# CODE is alpha, and for illusdtration purposes only,
# it needs to be polished and adaptated for mulitple agents,
# this is just a start
# model_path = f"models/dqn_{grid_path}_{episodes}_test.pth" , change this also in the dqn code.

# use this commant for the grids:
#python3 -m experiments_Diego_Niek_Timo.Experiments_Diego models/dqn_A1_grid_test.pth grid_configs/A1_grid.npy grid_configs/large_grid.npy 
#python3 -m experiments_Diego_Niek_Timo.Experiments_Diego models/dqn_A1_grid_test.pth grid_configs/lvl0.npy grid_configs/lvl1.npy grid_configs/lvl2.npy grid_configs/lvl4.npy grid_configs/lvl6.npy

# Define editable variables for episode counts
EPISODES_5 = 5
EPISODES_15 = 500

def evaluate_and_train_multi_grid(trainer_path: Path, grid_paths: list, episodes_5: int, episodes_15: int):
    """
    Evaluate and train the agent on multiple grids and visualize success rates and average successful steps.
    
    Args:
        trainer_path (Path): Path to the trainer script.
        grid_paths (list): List of paths to grid files.
        episodes_5 (int): Number of episodes for the first agent.
        episodes_15 (int): Number of episodes for the second agent.
    """
    results = {episodes_5: [], episodes_15: []}  # Dictionary to store success rates for each episode count
    avg_steps = {episodes_5: [], episodes_15: []}  # Dictionary to store average successful steps for each episode count
    episode_counts = [episodes_5, episodes_15]  # List of episode counts to iterate over

    # Wrap episode_counts with tqdm for progress tracking
    for episodes in tqdm(episode_counts, desc="Episode Counts"):
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ESTIMATED total time ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # Wrap grid_paths with tqdm for progress tracking
        for grid_path in tqdm(grid_paths, desc=f"Training with {episodes} episodes"):
            print('Time for grids to complete')
            print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
            print(f"Training on grid: {grid_path} with {episodes} episodes")
            # Run the training script
            cmd = f"{sys.executable} {trainer_path} {grid_path} --episodes {episodes}"  # Dynamically use the current Python interpreter
            print(f"Running command: {cmd}")
            exit_code = os.system(cmd)
            if exit_code != 0:
                print(f"Error running command: {cmd}")
                break
            # Evaluate the trained agent
            model_path = Path(f"models/dqn_{grid_path.stem}_{episodes}_test.pth")
            if model_path.exists():
                print(f"Evaluating model: {model_path}")
                returns, success_rate, avg_succes_steps = evaluate(model_path, grid_path, random_seed = np.random.randint(0, 1000))
                print(f"Returns: {returns}, Success Rate: {success_rate:.2f}, Avg Successful Steps: {avg_succes_steps:.2f}")
                results[episodes].append((grid_path.stem, success_rate))  # Store grid name and success rate
                avg_steps[episodes].append((grid_path.stem, avg_succes_steps))  # Store grid name and avg successful steps

    print(results[episodes_5])
    print(results[episodes_15])
    # Visualize and save success rates and average successful steps
    visualize_results(results[episodes_5], results[episodes_15], avg_steps[episodes_5], avg_steps[episodes_15])


def visualize_results(results_5, results_15, avg_steps_5, avg_steps_15):
    """
    Visualize and save success rates and average successful steps for two agents across different grids.
    
    Args:
        results_5 (list): List of tuples (grid_name, success_rate) for agent trained with episodes_5.
        results_15 (list): List of tuples (grid_name, success_rate) for agent trained with episodes_15.
        avg_steps_5 (list): List of tuples (grid_name, avg_successful_steps) for agent trained with episodes_5.
        avg_steps_15 (list): List of tuples (grid_name, avg_successful_steps) for agent trained with episodes_15.
    """
    # Create results directory if it doesn't exist
    results_dir = Path("results/experimentsDiegoresults")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Unpack success rates
    grids_5, success_rates_5 = zip(*results_5)
    grids_15, success_rates_15 = zip(*results_15)

    # Unpack average successful steps
    grids_steps_5, avg_steps_5_values = zip(*avg_steps_5)
    grids_steps_15, avg_steps_15_values = zip(*avg_steps_15)

    # Plot success rates
    plt.figure(figsize=(10, 6))
    plt.plot(grids_5, success_rates_5, marker='o', label=f'Agent ({EPISODES_5} episodes)', color='blue')
    plt.plot(grids_15, success_rates_15, marker='o', label=f'Agent ({EPISODES_15} episodes)', color='red')
    plt.title(f'Success Rates Across Grids\nCreated on {timestamp}')
    plt.xlabel('Grid')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / f"success_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")  # Save plot with timestamp
    plt.show()

    # Plot average successful steps
    plt.figure(figsize=(10, 6))
    plt.plot(grids_steps_5, avg_steps_5_values, marker='o', label=f'Agent ({EPISODES_5} episodes)', color='green')
    plt.plot(grids_steps_15, avg_steps_15_values, marker='o', label=f'Agent ({EPISODES_15} episodes)', color='orange')
    plt.title(f'Average Successful Steps Across Grids\nCreated on {timestamp}')
    plt.xlabel('Grid')
    plt.ylabel('Average Successful Steps')
    plt.legend()
    plt.grid()
    plt.savefig(results_dir / f"average_successful_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")  # Save plot with timestamp
    plt.show()

    # Save numerical results to a file
    with open(results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(f"Results generated on {timestamp}\n\n")
        f.write(f"Success Rates ({EPISODES_5} episodes):\n")
        for grid, rate in results_5:
            f.write(f"{grid}: {rate:.2f}%\n")
        f.write(f"\nSuccess Rates ({EPISODES_15} episodes):\n")
        for grid, rate in results_15:
            f.write(f"{grid}: {rate:.2f}%\n")
        f.write(f"\nAverage Successful Steps ({EPISODES_5} episodes):\n")
        for grid, steps in avg_steps_5:
            f.write(f"{grid}: {steps:.2f}\n")
        f.write(f"\nAverage Successful Steps ({EPISODES_15} episodes):\n")
        for grid, steps in avg_steps_15:
            f.write(f"{grid}: {steps:.2f}\n")


def full_evaluation(env, agent, n_episodes=10, max_steps=1000):
    """More extensive evaluation of the agent's performance."""
    returns = []
    successes = 0
    succesfull_steps = []
    for _ in range(n_episodes):
        state = env.reset_env(no_gui=True)
        total_r = 0
        for t in range(max_steps):
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            total_r += reward
            state = next_state
            if done:
                if info.get("target_reached", False):
                    successes += 1
                    succesfull_steps.append(env.world_stats["total_steps"])
                break
        returns.append(total_r)
    avg_return = sum(returns) / n_episodes
    success_rate = successes / n_episodes
    avg_succes_steps = sum(succesfull_steps)/ len(succesfull_steps) if succesfull_steps else 0
    print(f"Avg return over {n_episodes} eps: {avg_return:.2f}")
    print(f"Success rate: {success_rate*100:.1f}%")
    return returns, success_rate, avg_succes_steps


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
    Environment.evaluate_agent(grid_path, agent, iters, sigma, random_seed=random_seed)
    # Own evaluation function
    returns, success_rate, avg_succes_steps = full_evaluation(env, agent, n_episodes=100, max_steps=iters)
    return returns, success_rate, avg_succes_steps


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .pth checkpoint")
    p.add_argument("grid", type=Path, nargs="+", help="Grid file(s)")  # Accept multiple grid files
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    
    # Fix the path to train_dqn.py to point to the parent directory
    evaluate_and_train_multi_grid(
        Path(__file__).resolve().parent.parent / "train_dqn.py",  # Corrected path
        args.grid,  # Pass the list of grid files
        EPISODES_5,  # Pass the number of episodes for the first agent
        EPISODES_15  # Pass the number of episodes for the second agent
    )

