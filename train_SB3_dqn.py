# """
# Train your RL Agent using Stable Baselines3 PPO in this file.
# """

# from argparse import ArgumentParser
# from pathlib import Path
# import os
# import time
# import numpy as np
# from tqdm import trange
# from stable_baselines3 import DQN
# from world.gym import GridNavigationEnv  # Our custom environment from gym.py

# def parse_args():
#     p = ArgumentParser(description="PPO Reinforcement Learning Trainer.")
#     p.add_argument("GRID", type=Path, nargs="+",
#                    help="Paths to the grid file to use. There can be more than "
#                         "one.")
#     p.add_argument("--no_gui", action="store_true",
#                    help="Disables rendering to train faster")
#     p.add_argument("--sigma", type=float, default=0,
#                    help="Sigma value for the stochasticity of the environment.")
#     p.add_argument("--fps", type=int, default=30,
#                    help="Frames per second to render at. Only used if "
#                         "no_gui is not set.")
#     p.add_argument("--episodes", type=int, default=600,
#                    help="Number of episodes to train the agent for.")
#     p.add_argument("--iter", type=int, default=1000,
#                    help="Max steps per episode.")
#     p.add_argument("--random_seed", type=int, default=0,
#                    help="Random seed value for the environment.")
#     p.add_argument("--learning_rate", type=float, default=3e-4,
#                    help="Learning rate for PPO.")
#     p.add_argument("--batch_size", type=int, default=64,
#                    help="Batch size for PPO.")
#     p.add_argument("--gamma", type=float, default=0.9,
#                    help="Discount factor.")
#     return p.parse_args()

# def main(grid: list[Path], no_gui: bool, episodes: int, iters: int, fps: int,
#          sigma: float, random_seed: int, learning_rate: float,
#          batch_size: int, gamma: float):
#     """Main loop of the program."""

#     assert len(grid) == 1, "Provide exactly one grid for training"
#     grid = grid[0]
#     total_timesteps = episodes * iters

#     # Setup logging
#     log_dir = "./logs/"
#     models_dir = "./models/"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(models_dir, exist_ok=True)

#     # Create environment
#     env = GridNavigationEnv(
#         grid_fp=str(grid),
#         no_gui=True,
#         sigma=sigma,
#         random_seed=random_seed,
#         agent_start_pos=(9, 13),
#         target_position=[(11, 3)]
#     )

#     # Create vectorized environment
#     vec_env = DummyVecEnv([lambda: env])
#     eval_env = DummyVecEnv([lambda: env])

#     # Callbacks
#     checkpoint_callback = CheckpointCallback(
#         save_freq=max(iters, 1000),  # Save every episode or 1000 steps
#         save_path=log_dir,
#         name_prefix="ppo_grid"
#     )

#     eval_callback = EvalCallback(
#         eval_env,
#         eval_freq=5000,  # Evaluate every 5000 steps
#         n_eval_episodes=5,  # Run 5 episodes per evaluation
#         log_path=log_dir,  # Where to save evaluation results
#         best_model_save_path=models_dir,  # Save best model
#         deterministic=True,  # Use deterministic actions during eval
#         render=False,  # Don't render eval episodes
#     )

#     # Create DQN model
#     print("Creating DQN model...")
#     model = DQN(
#         policy="MlpPolicy",
#         env=env,
#         learning_rate=learning_rate,
#         buffer_size=buffer_size,
#         learning_starts=learning_starts,
#         batch_size=batch_size,
#         gamma=gamma,
#         train_freq=train_freq,
#         target_update_interval=target_update_interval,
#         verbose=1,
#         tensorboard_log=log_dir,
#         seed=random_seed,
#         policy_kwargs={"net_arch": [64, 64]}  # Two hidden layers, as before
#     )

#     # Training
#     print(f"Starting training for {episodes} episodes ({total_timesteps} timesteps)...")
#     start_time = time.time()

#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=[checkpoint_callback, eval_callback],
#         tb_log_name="ppo_run"
#     )

#     # Save final model
#     model.save(os.path.join(log_dir, "ppo_grid_nav_final"))
#     print(f"Training completed in {time.time() - start_time:.2f}s")

#     # Evaluation
#     print("Evaluating model...")
#     # Create environment
#     env = GridNavigationEnv(
#         grid_fp=str(grid),
#         no_gui=False,
#         sigma=sigma,
#         random_seed=random_seed,
#         agent_start_pos=(9, 13),
#         target_position=[(11, 3)]
#     )

#     # Create vectorized environment
#     vec_env = DummyVecEnv([lambda: env])
#     obs = vec_env.reset()
#     episode_reward = 0
#     episode_length = 0
#     terminated = False

#     for step in trange(iters, desc="Evaluation"):
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = vec_env.step(action)
#         episode_reward += reward
#         episode_length += 1

#         if done:
#             print(f"Episode finished at step {step}! Reward: {episode_reward}, Length: {episode_length}")
#             terminated = True
#             break

#     if not terminated:
#         print(f"Reached max steps ({iters}). Reward: {episode_reward}")

#     # Save evaluation results
#     with open(os.path.join(log_dir, "evaluation.txt"), "w") as f:
#         f.write(f"Final Reward: {episode_reward}\n")
#         f.write(f"Episode Length: {episode_length}\n")
#         f.write(f"Terminated: {terminated}\n")

#     print(f"Results saved to {log_dir}")

# if __name__ == '__main__':
#     args = parse_args()
#     main(
#         grid=args.GRID,
#         no_gui=args.no_gui,
#         episodes=args.episodes,
#         iters=args.iter,
#         fps=args.fps,
#         sigma=args.sigma,
#         random_seed=args.random_seed,
#         learning_rate=args.learning_rate,
#         batch_size=args.batch_size,
#         gamma=args.gamma
#     )

"""
Train your RL Agent using Stable Baselines3 DQN in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import time
import numpy as np
from tqdm import trange

# Import DQN from Stable Baselines3
from stable_baselines3 import DQN

# Callbacks remain usable:
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Your custom environment stays the same:
from world.gym import GridNavigationEnv  # Our custom environment from gym.py

def parse_args():
    p = ArgumentParser(description="DQN Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of episodes to train the agent for.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Max steps per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Learning rate for DQN.")  # You can adjust as needed
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for DQN.")  # Typically smaller than PPO's
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor.")  # Often 0.99 for DQN
    p.add_argument("--buffer_size", type=int, default=100_000,
                   help="Replay buffer size for DQN.")
    p.add_argument("--learning_starts", type=int, default=1_000,
                   help="Number of steps before learning starts.")
    p.add_argument("--train_freq", type=int, default=4,
                   help="Frequency (in env steps) to update the model.")
    p.add_argument("--target_update_interval", type=int, default=10_000,
                   help="Number of steps between target network updates.")
    return p.parse_args()

def main(grid: list[Path],
         no_gui: bool,
         episodes: int,
         iters: int,
         fps: int,
         sigma: float,
         random_seed: int,
         learning_rate: float,
         batch_size: int,
         gamma: float,
         buffer_size: int,
         learning_starts: int,
         train_freq: int,
         target_update_interval: int):
    """Main loop of the program, now using DQN instead of PPO."""
    assert len(grid) == 1, "Provide exactly one grid for training"
    grid = grid[0]

    # For DQN, we'll compute total_timesteps as episodes * iters, but note
    # that DQN steps through the env one step at a time (no vectorization).
    total_timesteps = episodes * iters

    # Setup logging folders (same as before)
    log_dir = "./logs/"
    models_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = GridNavigationEnv(
        grid_fp=str(grid),
        no_gui=True,        # We force no_gui=True during training for speed
        sigma=sigma,
        random_seed=random_seed,
        agent_start_pos=(9, 13),
        target_position=[(11, 3)]
    )

    # Keep using same callback because we are not using vectorizing as we do for PPO
    checkpoint_callback = CheckpointCallback(
        save_freq=max(iters, 1000),  # Save every episode or 1000 steps
        save_path=log_dir,
        name_prefix="dqn_grid"
    )

    # For evaluation:
    eval_env = GridNavigationEnv(
        grid_fp=str(grid),
        no_gui=True,
        sigma=sigma,
        random_seed=random_seed,
        agent_start_pos=(9, 13),
        target_position=[(11, 3)]
    )
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5000,          # Evaluate every 5000 steps
        n_eval_episodes=5,       # Run 5 episodes per evaluation
        log_path=log_dir,        # Where to save evaluation results
        best_model_save_path=models_dir,  # Save best model
        deterministic=True,
        render=False
    )

    # DQN model
    print("Creating DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        verbose=1,
        tensorboard_log=log_dir,
        seed=random_seed,
        policy_kwargs={"net_arch": [64, 64]}  # Two hidden layers for simple MLP
        exploration_initial_eps=1.0,  # initial epsilon value
        exploration_final_eps=0.1,  # final epsilon value
        exploration_fraction=0.5  # means that in first half of training, epsilon will decay from 1.0 to 0.1
    )

    # Training
    print(f"Starting training for {episodes} episodes (~{total_timesteps} timesteps)...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="dqn_run"
    )
    print(f"Training completed in {time.time() - start_time:.2f}s")

    # Save final model
    model.save(os.path.join(models_dir, "dqn_grid_nav_final"))
    print(f"Final model saved to {os.path.join(models_dir, 'dqn_grid_nav_final.zip')}")

    # evaluation
    print("Evaluating model...")
    # Create a render‐enabled environment for “watching” the policy in action
    eval_env = GridNavigationEnv(
        grid_fp=str(grid),
        no_gui=False,      # Turn on rendering
        sigma=sigma,
        random_seed=random_seed,
        agent_start_pos=(9, 13),
        target_position=[(11, 3)]
    )

    obs, _ = eval_env.reset()
    
    episode_reward = 0
    episode_length = 0
    terminated = False

    for step in trange(iters, desc="Evaluation"):
        action, _states = model.predict(obs, deterministic=True)
        action = action.item()  # DQN returns a single action as an array, but we want an integer
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1
        obs = next_obs

        if done:
            print(f"Episode finished at step {step}! Reward: {episode_reward}, Length: {episode_length}")
            terminated = True
            break

    if not terminated:
        print(f"Reached max steps ({iters}). Reward: {episode_reward}")

    # Save evaluation results
    eval_fp = os.path.join(log_dir, "evaluation_dqn.txt")
    with open(eval_fp, "w") as f:
        f.write(f"Final Reward: {episode_reward}\n")
        f.write(f"Episode Length: {episode_length}\n")
        f.write(f"Terminated: {terminated}\n")
    print(f"Evaluation results saved to {eval_fp}")


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
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval
    )
    # main(
    #     grid=[Path("2AMC15-DIC-assignment-2/grid_configs/A1_grid.npy")],   # Default GRID (nargs="+") must be provided as a list
    #     no_gui=False,              # --no_gui default
    #     episodes=10,               # --episodes default
    #     iters=1000,                # --iter default
    #     fps=30,                    # --fps default
    #     sigma=0.0,                 # --sigma default
    #     random_seed=0,             # --random_seed default
    #     learning_rate=1e-4,        # --learning_rate default
    #     batch_size=32,             # --batch_size default
    #     gamma=0.99,                # --gamma default
    #     buffer_size=100_000,       # --buffer_size default
    #     learning_starts=1_000,     # --learning_starts default
    #     train_freq=4,              # --train_freq default
    #     target_update_interval=10_000  # --target_update_interval default
    # )
