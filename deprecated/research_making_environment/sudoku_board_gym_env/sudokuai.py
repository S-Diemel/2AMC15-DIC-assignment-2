from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import team05_A3.sudoku_calc_move as sudoku_calc_move

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given Sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    
    def compute_best_move(self, game_state: GameState) -> None:
        sudoku_calc_move.size = game_state.board.N
        game_state, action = sudoku_calc_move.play_rl_agent_action(game_state)
        move = sudoku_calc_move.convert_to_move(action)
        self.propose_move(move)