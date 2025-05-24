# import sudoku_gym

from team05_A3.sudoku_gym import SudokuEnv
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from random import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, \
    SudokuSettings, allowed_squares
import os
import competitive_sudoku.sudokuai
from sb3_contrib.common.maskable.utils import get_action_masks


# Call the model
model = MaskablePPO.load("team05_A3/best_model")

M = 3
N = 3
# k_timesteps = 250
reward_structure = 'GER'

env = SudokuEnv(M,N, reward_structure, eval_env=False)
# Test the trained model
print(env)
obs, info = env.reset()


def play_rl_agent_action(gamestate: GameState, env):
    env.game_state = gamestate
    obs = env._get_obs()
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
    print("HELLO LOOK HERE")
    print("Action is: ", action)
    print("Previous game state is: ")
    print(env.game_state)
    obs, rewards, done, truncated, info = env.step(action)
    print("New game state is: ")
    print(env.game_state)
    return env.game_state, action

# def get_eval_agent_action(gamestate?, env):
#     action from second agent, greedy/hueristics
#     return env.gamestatewiththismoveadded, action

# Initialize game_state
initial_board = SudokuBoard(M, N)
allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
game_state_current = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        


for i in range(1):
    #20 games

    agent_plays_first = True if random() > 0.5 else False

    while not env.done:
    
        if agent_plays_first:
            gamestate, test = play_rl_agent_action(game_state_current, env)
        
        else:
            gamestate = play_rl_agent_action(game_state_current, env)
        
    else:
        env.reset
        
    
    # get actions from other agent
    # play that action

    print(env.game_state)