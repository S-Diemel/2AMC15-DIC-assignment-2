from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from team05_A3.sudoku_gym import SudokuEnv
from sb3_contrib import MaskablePPO
import numpy as np
from team05_A3.PythonSolverUnique import numpy_to_sudoku_format, SudokuPuzzle, depth_first_solve
import copy
  
# Call the model
model = MaskablePPO.load("team05_A3/best_model")

size = 0

def game_state_to_board(game_state: GameState):
    board = np.zeros((game_state.board.N, game_state.board.N), dtype=int)
    for i in range(game_state.board.N):
        for j in range(game_state.board.N):
            board[i,j] = game_state.board.get((i, j))
    return board

def convert_to_move(action):
    action_adjusted = action - 1
    val = action_adjusted // (size * size)
    i = (action_adjusted % ((size)**2)) // (size)
    j = (action_adjusted % ((size)**2)) % (size) 
    move = Move((i, j), val + 1)
    return move

def get_obs(game_state, board):
    board_interim = np.zeros((game_state.board.N, game_state.board.N, game_state.board.N), dtype=np.uint8)
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i,j] != 0:
                board_interim[board[i,j] - 1, i,j] = 1
    return board_interim

def action_masks(game_state: GameState):
        
    action_mask_interim = np.full((game_state.board.N, game_state.board.N, game_state.board.N), False)
    for move in get_legal_moves(game_state):
        action_mask_interim[move.value-1, move.square[0], move.square[1]] = True
    action_mask_interim = action_mask_interim.flatten()
    action_mask_interim = action_mask_interim.tolist()
    if True not in action_mask_interim:
        action_mask_interim = [True] + action_mask_interim
    else:
        action_mask_interim = [False] + action_mask_interim
    return action_mask_interim

def get_constraints(board, move: Move):
    
    row_values = [board.get((move.square[0], j)) for j in range(
        size) if board.get((move.square[0], j)) != SudokuBoard.empty]
    col_values = [board.get((i, move.square[1])) for i in range(
        size) if board.get((i, move.square[1])) != SudokuBoard.empty]
    block_i = (move.square[0] // board.m) * board.m
    block_j = (move.square[1] // board.n) * board.n
    block_values = [
        board.get((i, j))
        for i in range(block_i, block_i + board.m)
        for j in range(block_j, block_j + board.n)
        if board.get((i, j)) != SudokuBoard.empty
    ]
    return row_values, col_values, block_values

def get_legal_moves(game_state: GameState):
    moves = []
    for i in game_state.player_squares():
        for value in range(1, size + 1):
            if is_valid_move(game_state, i[0], i[1], value):
                row_values, col_values, block_values = get_constraints(
                    game_state.board, Move(i, value))
                if value not in row_values + col_values + block_values:
                    moves.append(Move(i, value))
    board = game_state_to_board(game_state)
    num = game_state.board.N**2 - np.sum(board == 0)
    print("Number of played moves in game state is: ", num)
    if num > 16:
        moves_solvable = []
        for move in moves:
            
            board_copy = copy.deepcopy(board)
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


def is_valid_move(game_state: GameState, i, j, value):
    return (
        game_state.board.get((i, j)) == SudokuBoard.empty
        and TabooMove((i, j), value) not in game_state.taboo_moves
        and (i, j) in game_state.player_squares()
    )

def play_rl_agent_action(game_state: GameState):
    board = game_state_to_board(game_state)
    obs = get_obs(game_state, board)
    action_mask = action_masks(game_state)
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
    return game_state, action