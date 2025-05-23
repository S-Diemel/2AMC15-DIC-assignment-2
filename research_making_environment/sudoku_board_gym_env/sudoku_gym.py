from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, \
    SudokuSettings, allowed_squares
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Optional
from typing import List
# from solver import solve_sudoku, str2grid, grid2str
# from Sudoku import SudokuException
from team05_A3.PythonSolverUnique import numpy_to_sudoku_format, SudokuPuzzle, depth_first_solve
from stable_baselines3.common.callbacks import BaseCallback


# print("Hello")

class SudokuEnv(gym.Env):
    
    def __init__(self, m, n, reward_shape, eval_env: bool = False):
        
        super(SudokuEnv, self).__init__()
        initial_board = SudokuBoard(m, n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((m*n, m*n), dtype=int)
        
        self.action_space = spaces.Discrete(1 + (m*n * m*n * m*n))
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(m*n, m*n, m*n), dtype=np.uint8)
        self.rewards = [0,0]
        self.infos = {}
        self.win = 0
        self.eval_env = eval_env
        self.done = False
        self.reward_shape = reward_shape
        self.get_logables()

    def get_logables(self):
        self.infos = {}
        self.infos['player1_score'], self.infos['player2_score']= self.game_state.scores
        self.infos['done'] = self.done
        self.infos['fill_percentage'] = (1 - (np.sum(self.board == 0) / self.game_state.board.N**2)) * 100
        self.infos['reward_1'] = self.rewards[0]
        self.infos['reward_2'] = self.rewards[1]
    
    def convert_to_move(self, action):
        action_adjusted = action - 1
        val = action_adjusted // (self.game_state.board.N * self.game_state.board.N)
        i = (action_adjusted % ((self.game_state.board.N)**2)) // (self.game_state.board.N)
        j = (action_adjusted % ((self.game_state.board.N)**2)) % (self.game_state.board.N) # action -> move

        # Now use these components to create the move
        move = Move((i, j), val + 1)
        
        return move
    
    def _get_obs(self):
        board_interim = np.zeros((self.game_state.board.N, self.game_state.board.N, self.game_state.board.N), dtype=np.uint8)
        for i in range(self.game_state.board.N):
            for j in range(self.game_state.board.N):
                if self.board[i,j] != 0:
                    board_interim[self.board[i,j] - 1, i,j] = 1
        return board_interim
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        self.get_logables()
        '''calculcate filled% of board before reseting it'''
        num = self.game_state.board.N**2 - np.sum(self.board==0)
        perc = (num / self.game_state.board.N**2) * 100
        
        if self.eval_env == False:
            print("Percentage filled is: ", perc, "%")
            print(self.game_state.board)
            print(self.game_state.scores)
        
        initial_board = SudokuBoard(self.game_state.board.m, self.game_state.board.n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])

        self.win = 0
        self.rewards = [0,0]
        
        self.board = np.zeros((self.game_state.board.N, self.game_state.board.N), dtype=int)
        self.done = False
        return self._get_obs(), {}
    
    def action_masks(self):
        
        action_mask_interim = np.full((self.game_state.board.N, self.game_state.board.N, self.game_state.board.N), False)
        for move in self.get_legal_moves():
            action_mask_interim[move.value-1, move.square[0], move.square[1]] = True
        action_mask_interim = action_mask_interim.flatten()
        action_mask_interim = action_mask_interim.tolist()
        if True not in action_mask_interim:
            action_mask_interim = [True] + action_mask_interim
        else:
            action_mask_interim = [False] + action_mask_interim
        return action_mask_interim
    
    def step(self, action):

        self.get_logables()
        
        if self.reward_shape == 'GER':
            
        # Game State Encoded Reward

            reward = 0

            if (np.sum(self.board == 0) != 0):
                if action == 0:

                    self.game_state.current_player = 3 - self.game_state.current_player
                    if not self.get_legal_moves():
                        reward -= 20
                        self.done = True
                    self.game_state.current_player = 3 - self.game_state.current_player

                else:
                    move = self.convert_to_move(action)
                    self.add_to_game_state(move)

                reward += (self.game_state.scores[self.game_state.current_player - 1] - self.game_state.scores[2 - self.game_state.current_player]) / 10

            # Board is full
            if (np.sum(self.board == 0) == 0):

                # Game is tied
                if self.game_state.scores[0] == self.game_state.scores[1]:
                    self.done = True

                # Rewards are updated to reflect win
                else:
                    if self.win < 2:    # Winner/loser scores are not updated

                        winner = self.game_state.scores.index(max(self.game_state.scores)) + 1
                        # diff = abs(self.game_state.scores[self.game_state.current_player - 1] - self.game_state.scores[2 - self.game_state.current_player])

                        if winner == self.game_state.current_player:
                            reward += 2 * self.game_state.scores[winner - 1]
                            self.win += 1
                        else:
                            reward -= 2 * self.game_state.scores[2 - winner]
                            self.win += 1

                    else:
                        self.done = True

            self.rewards[self.game_state.current_player - 1] += reward
            self.game_state.current_player = 3 - self.game_state.current_player
        
        if self.reward_shape == 'vanilla':

            reward = 0

            if (np.sum(self.board == 0) != 0):
                if action == 0:

                    self.game_state.current_player = 3 - self.game_state.current_player
                    if not self.get_legal_moves():
                        reward -= 20
                        self.done = True
                    self.game_state.current_player = 3 - self.game_state.current_player

                else:
                    move = self.convert_to_move(action)
                    self.add_to_game_state(move)
            
            # Board is full
            if (np.sum(self.board == 0) == 0):

                if self.win < 2:    # Winner/loser scores are not updated
                    winner = self.game_state.scores.index(max(self.game_state.scores)) + 1
                    # diff = abs(self.game_state.scores[self.game_state.current_player - 1] - self.game_state.scores[2 - self.game_state.current_player])
                    if winner == self.game_state.current_player:
                        reward = (reward + 50) + self.game_state.scores[winner - 1]
                        self.win += 1
                    else:
                        reward = (reward - 50) - self.game_state.scores[2 - winner]
                        self.win += 1
                else:
                    self.done = True

            self.rewards[self.game_state.current_player - 1] += reward
            self.game_state.current_player = 3 - self.game_state.current_player
        
        return self._get_obs(), reward, self.done, False, {}
    
    def get_constraints(self, move: Move):

        N = self.game_state.board.m * self.game_state.board.n
        row_values = [self.game_state.board.get((move.square[0], j)) for j in range(
            N) if self.game_state.board.get((move.square[0], j)) != SudokuBoard.empty]
        col_values = [self.game_state.board.get((i, move.square[1])) for i in range(
            N) if self.game_state.board.get((i, move.square[1])) != SudokuBoard.empty]
        block_i = (move.square[0] // self.game_state.board.m) * self.game_state.board.m
        block_j = (move.square[1] // self.game_state.board.n) * self.game_state.board.n
        block_values = [
            self.game_state.board.get((i, j))
            for i in range(block_i, block_i + self.game_state.board.m)
            for j in range(block_j, block_j + self.game_state.board.n)
            if self.game_state.board.get((i, j)) != SudokuBoard.empty
        ]
        return row_values, col_values, block_values
    
    def add_to_game_state(self, move):

        reward = self.evaluate_score(move)
        self.game_state.scores[self.game_state.current_player - 1] += reward
        self.game_state.board.put(move.square, move.value)
        self.game_state.moves.append(move)
        self.game_state.occupied_squares().append(move.square)
        self.board[move.square[0], move.square[1]] = move.value
        # self.game_state.current_player = 3 - self.game_state.current_player  # Toggle between player 1 and 2
        # return reward
    
    def evaluate_score(self, move):

        row_values, col_values, block_values = self.get_constraints(move)
        solves_row = len(row_values) == self.game_state.board.N - 1
        solves_col = len(col_values) == self.game_state.board.N - 1
        solves_block = len(block_values) == self.game_state.board.N - 1
        sum = solves_row + solves_col + solves_block
        score = 0
        if sum == 0:
            score = 0
        elif sum == 1:
            score = 1
        elif sum == 2:
            score = 3
        elif sum == 3:
            score = 7
        return score
    
    # Check if a move is valid
    def is_valid_move(self, i, j, value):
        return (
            self.game_state.board.get((i, j)) == SudokuBoard.empty
            and TabooMove((i, j), value) not in self.game_state.taboo_moves
            and (i, j) in self.game_state.player_squares()
        )
        
    # Check allowed moves to generate legal_moves
    def get_legal_moves(self):
        moves = []
        for i in self.game_state.player_squares():
            for value in range(1, self.game_state.board.N + 1):
                if self.is_valid_move(i[0], i[1], value):
                    row_values, col_values, block_values = self.get_constraints(Move(i, value))
                    if value not in row_values + col_values + block_values:
                        moves.append(Move(i, value))

        
        num = self.game_state.board.N**2 - np.sum(self.board == 0)
        if num > 16:

            moves_solvable = []
            for move in moves:
                
                board_copy = copy.deepcopy(self.board)
                board_copy[move.square[0], move.square[1]] = move.value

                '''Solving Sudoku'''

                N = len(board_copy)
                symbol_set = {str(i) for i in range(1, N + 1)}
                board_copy = numpy_to_sudoku_format(board_copy)
                puzzle = SudokuPuzzle(n = N, symbols=board_copy, symbol_set=symbol_set)
                solution = depth_first_solve(puzzle)
                
                if solution:
                    moves_solvable.append(move)

            return moves_solvable
        
        return moves
    
    def split(self, num_envs):
        # Return num_envs copies of the environment
        return [SudokuEnv() for _ in range(num_envs)]

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.game_scores = [0, 0]

    def _on_step(self) -> bool:
        
        if (self.training_env.get_attr("infos")[0]['done']):
            
            print(self.locals['rewards'])
            self.logger.record("player1_score", self.training_env.get_attr("infos")[0]['player1_score'])
            self.logger.record("player2_score", self.training_env.get_attr("infos")[0]['player2_score'])
            self.logger.record("fill percentage", self.training_env.get_attr("infos")[0]['fill_percentage'])
            self.logger.record("reward_1", self.training_env.get_attr("infos")[0]['reward_1'])
            self.logger.record("reward_2", self.training_env.get_attr("infos")[0]['reward_2'])
            
            self.logger.dump(self.num_timesteps)
        return True

if __name__ == "__main__":
    print('hell naw')