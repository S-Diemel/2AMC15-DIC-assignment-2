# 2AMC15-DIC-assignment-2

This repository contains the code and experiments for the Data Intelligence Challenge (2AMC15) assignment 2, focused on reinforcement learning (RL) with DQN and PPO agents in a custom warehouse environment.

## Project Structure

```
├── agents/                # DQN, PPO, and base agent implementations
├── metrics/               # training metrics collected during curriculum learning phases (CSV, plots)
├── models/                # Saved models (DQN, PPO)
├── results/               # Output results and experiment figures
├── world/                 # Main environment code
│   └── utils/             # Utility functions for environment setup and reset methods
```

## Main Scripts

- **train_dqn_parallel.py**: Train a DQN agent using parallel environments and curriculum learning. Handles curriculum scheduling, evaluation, and metrics logging.
- **train_ppo_manual.py**: Train a PPO agent with parallel environments and curriculum learning. Supports entropy scheduling and evaluation.
- **train_curriculum_utils.py**: Utility functions for curriculum setup, parameter scheduling, evaluation, and metrics saving.
- **evaluate_trained_dqn.py**: Evaluate a trained DQN agent on an environment or curriculum stage, reporting success and other metrics.
- **evaluate_trained_ppo.py**: Evaluate a trained PPO agent similarly to the DQN evaluation script.
- **plot_training_metrics.py**: Plot training and evaluation metrics (e.g., success rate, items delivered) from CSV files generated during training.
- **plot_experiment_metrics.py**: Plot results from experimental setups (e.g., stochasticity, obstacle density, target distance) for comparing agent robustness and adaptability.

## Usage

- **Training**

  - **DQN (Parallel, Curriculum):**
    ```sh
    python train_dqn_parallel.py --name test --episodes 10000 --iters 1000 --random_seed 2
    ```
    - `--name`: Name for the saved model and metrics files (e.g., `test`).
    - `--episodes`: Total number of training episodes (default: 10000).
    - `--iters`: Number of environment steps per episode batch (default: 1000).
    - `--random_seed`: Random seed for reproducibility (default: 2).

  - **PPO (Parallel, Curriculum):**
    ```sh
    python train_ppo_manual.py --name test --episodes 2500 --iters 1000 --random_seed 2
    ```
    - `--name`: Name for the saved model and metrics files (e.g., `test`).
    - `--episodes`: Total number of training episodes (default: 2500).
    - `--iters`: Number of environment steps per episode batch (default: 1000).
    - `--random_seed`: Random seed for reproducibility (default: 2).

- **Evaluation**

  - **DQN:**
    ```sh
    python evaluate_trained_dqn.py models/dqn.pth
    ```
    - First argument: Path to the trained DQN model checkpoint (`.pth` file).
    - Evaluates the agent for 100 episodes and prints success statistics.
    - Note: Renders the GUI for every run.

  - **PPO:**
    ```sh
    python evaluate_trained_ppo.py models/ppo.pth
    ```
    - First argument: Path to the trained PPO model checkpoint (`.pth` file).
    - Evaluates the agent for 100 episodes and prints success statistics.
    - Note: Renders the GUI for every run.

- **Plotting**

  - **Training Metrics:**
    ```sh
    python plot_training_metrics.py --file_dqn_metrics metrics/dqn_metrics.csv --file_ppo_metrics metrics/ppo_metrics.csv
    ```
    - `--file_dqn_metrics`: Path to DQN metrics CSV (default: `metrics/dqn__metrics.csv`).
    - `--file_ppo_metrics`: Path to PPO metrics CSV (default: `metrics/ppo__metrics.csv`).
    - Note: Saves plots to `metrics/training_metrics_plot.png`.

  - **Experiment Metrics:**
    ```sh
    python plot_experiment_metrics.py models/dqn.pth models/ppo.pth
    ```
    - First argument: Path to DQN model checkpoint.
    - Second argument: Path to PPO model checkpoint.
    - Note: Runs robustness experiments (stochasticity, obstacles, target distance) and saves plots in `results/`.

Plots and metrics are saved in the `metrics/` and `results/` folders.

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```

## Notes
- The `deprecated/` folder contains legacy code and is not used for the project.
- All main scripts are in the top-level directory for easy access.
- Models and metrics are saved in the `models/` and `metrics/` folders, respectively.

For more details on the curriculum setup, evaluation protocol, and metrics, see the report and code comments.