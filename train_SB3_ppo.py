"""
Train your RL Agent using Stable Baselines3 PPO in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import time
import numpy as np
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from world.gym import GridNavigationEnv  # Our custom environment from gym.py

def parse_args():
    p = ArgumentParser(description="PPO Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--episodes", type=int, default=600,
                   help="Number of episodes to train the agent for.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--learning_rate", type=float, default=3e-4,
                   help="Learning rate for PPO.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for PPO.")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount factor.")
    return p.parse_args()

def main(grid: list[Path], no_gui: bool, episodes: int, iters: int, fps: int,
         sigma: float, random_seed: int, learning_rate: float,
         batch_size: int, gamma: float):
    """Main loop of the program."""

    assert len(grid) == 1, "Provide exactly one grid for training"
    grid = grid[0]
    total_timesteps = episodes * iters

    # Setup logging
    log_dir = "./logs/"
    models_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Create environment
    env = GridNavigationEnv(
        grid_fp=str(grid),
        no_gui=True,
        sigma=sigma,
        random_seed=random_seed,
        agent_start_pos=(9, 13),
        target_position=[(11, 3)]
    )

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    eval_env = DummyVecEnv([lambda: env])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(iters, 1000),  # Save every episode or 1000 steps
        save_path=log_dir,
        name_prefix="ppo_grid"
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5000,  # Evaluate every 5000 steps
        n_eval_episodes=5,  # Run 5 episodes per evaluation
        log_path=log_dir,  # Where to save evaluation results
        best_model_save_path=models_dir,  # Save best model
        deterministic=True,  # Use deterministic actions during eval
        render=False,  # Don't render eval episodes
    )

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        n_steps=iters,           # Steps per environment per update
        batch_size=batch_size,
        gamma=gamma,
        n_epochs=10,
        seed=random_seed,
        policy_kwargs={"net_arch": [64, 64]}  # Two hidden layers
    )

    # Training
    print(f"Starting training for {episodes} episodes ({total_timesteps} timesteps)...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_run"
    )

    # Save final model
    model.save(os.path.join(log_dir, "ppo_grid_nav_final"))
    print(f"Training completed in {time.time() - start_time:.2f}s")

    # Evaluation
    print("Evaluating model...")
    # Create environment
    env = GridNavigationEnv(
        grid_fp=str(grid),
        no_gui=False,
        sigma=sigma,
        random_seed=random_seed,
        agent_start_pos=(9, 13),
        target_position=[(11, 3)]
    )

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()
    episode_reward = 0
    episode_length = 0
    terminated = False

    for step in trange(iters, desc="Evaluation"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_reward += reward
        episode_length += 1

        if done:
            print(f"Episode finished at step {step}! Reward: {episode_reward}, Length: {episode_length}")
            terminated = True
            break

    if not terminated:
        print(f"Reached max steps ({iters}). Reward: {episode_reward}")

    # Save evaluation results
    with open(os.path.join(log_dir, "evaluation.txt"), "w") as f:
        f.write(f"Final Reward: {episode_reward}\n")
        f.write(f"Episode Length: {episode_length}\n")
        f.write(f"Terminated: {terminated}\n")

    print(f"Results saved to {log_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(
        grid=args.GRID,
        no_gui=args.no_gui,
        episodes=args.episodes,
        iters=args.iter,
        fps=args.fps,
        sigma=args.sigma,
        random_seed=args.random_seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma
    )