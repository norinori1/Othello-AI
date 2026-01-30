import copy

EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [[EMPTY]*8 for _ in range(8)]
        self.grid[3][3] = WHITE
        self.grid[3][4] = BLACK
        self.grid[4][3] = BLACK
        self.grid[4][4] = WHITE

    def copy(self):
        b = Board()
        b.grid = copy.deepcopy(self.grid)
        return b

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def opponent(self, color):
        return BLACK if color == WHITE else WHITE

    def _captures_in_dir(self, r, c, dr, dc, color):
        r += dr; c += dc
        captured = []
        while self.in_bounds(r, c) and self.grid[r][c] == self.opponent(color):
            captured.append((r,c))
            r += dr; c += dc
        if not self.in_bounds(r, c) or self.grid[r][c] != color:
            return []
        return captured

    def valid_moves(self, color):
        moves = {}
        for r in range(8):
            for c in range(8):
                if self.grid[r][c] != EMPTY:
                    continue
                caps = []
                for dr, dc in DIRECTIONS:
                    caps += self._captures_in_dir(r, c, dr, dc, color)
                if caps:
                    moves[(r,c)] = caps
        return moves  # dict: move -> list of captured positions

    def apply_move(self, move, color):
        # move: (r,c) or None for pass
        if move is None:
            return
        r, c = move
        moves = self.valid_moves(color)
        if (r,c) not in moves:
            raise ValueError(f"Invalid move {(r,c)} for color {color}")
        self.grid[r][c] = color
        for (rr,cc) in moves[(r,c)]:
            self.grid[rr][cc] = color

    def is_full(self):
        for r in range(8):
            for c in range(8):
                if self.grid[r][c] == EMPTY:
                    return False
        return True

    def score(self):
        black = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == BLACK)
        white = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == WHITE)
        return {BLACK: black, WHITE: white}

    def game_over(self):
        # game over if both players have no moves or board full
        if self.is_full():
            return True
        if not self.valid_moves(BLACK) and not self.valid_moves(WHITE):
            return True
        return False

    def winner(self):
        s = self.score()
        if s[BLACK] > s[WHITE]:
            return BLACK
        elif s[WHITE] > s[BLACK]:
            return WHITE
        else:
            return None  # draw

    def render(self):
        chars = {EMPTY: '.', BLACK: '●', WHITE: '○'}
        lines = []
        for r in range(8):
            lines.append(''.join(chars[self.grid[r][c]] for c in range(8)))
        return '\n'.join(lines)

    def to_tensor(self, to_play):
        # returns 8x8x3 like spec: self, opp, legal
        import numpy as np
        arr = np.zeros((3,8,8), dtype=int)
        for r in range(8):
            for c in range(8):
                v = self.grid[r][c]
                if v == to_play:
                    arr[0,r,c] = 1
                elif v == self.opponent(to_play):
                    arr[1,r,c] = 1
        for (r,c) in self.valid_moves(to_play):
            arr[2,r,c] = 1
        return arr
    
    def is_corner(self, r, c):
        """Check if position (r, c) is a corner."""
        return (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def is_edge(self, r, c):
        """Check if position (r, c) is on the edge (but not a corner)."""
        if self.is_corner(r, c):
            return False
        return r == 0 or r == 7 or c == 0 or c == 7
    
    def count_corners(self, color):
        """Count how many corners are occupied by the given color."""
        count = 0
        for r, c in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            if self.grid[r][c] == color:
                count += 1
        return count
    
    def count_edges(self, color):
        """Count how many edge positions (not corners) are occupied by the given color."""
        count = 0
        for r in range(8):
            for c in range(8):
                if self.is_edge(r, c) and self.grid[r][c] == color:
                    count += 1
        return count
