import random
from .board import BLACK, WHITE

class RandomAgent:
    def __init__(self, color):
        self.color = color

    def select_move(self, board):
        moves = list(board.valid_moves(self.color).keys())
        if not moves:
            return None
        return random.choice(moves)

class GreedyAgent:
    def __init__(self, color):
        self.color = color

    def select_move(self, board):
        moves = board.valid_moves(self.color)
        if not moves:
            return None
        # choose move that flips the most stones
        best = None
        best_count = -1
        for m, caps in moves.items():
            if len(caps) > best_count:
                best_count = len(caps)
                best = m
        return best
