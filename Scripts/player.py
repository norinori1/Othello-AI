"""
Player classes for Othello game.
Includes HumanPlayer and AIPlayer.
"""
from .board import BLACK, WHITE


class HumanPlayer:
    """Human player that inputs moves via console."""
    
    def __init__(self, color):
        self.color = color
        self.color_name = "黒(●)" if color == BLACK else "白(○)"
    
    def select_move(self, board):
        """
        Prompts the human player to enter a move.
        
        Args:
            board: Board object
            
        Returns:
            tuple: (row, col) or None for pass
        """
        moves = board.valid_moves(self.color)
        
        if not moves:
            print(f"{self.color_name}はパスです")
            input("Enterキーを押して続行...")
            return None
        
        print(f"\n{self.color_name}のターンです")
        print("着手可能な位置:")
        for i, (r, c) in enumerate(sorted(moves.keys())):
            print(f"  {i+1}: ({r}, {c})", end="")
            if (i + 1) % 4 == 0:
                print()
        print()
        
        while True:
            try:
                user_input = input("着手位置を入力してください (例: 2 3) または番号 (例: 1): ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                
                # If single number, treat as index
                if len(parts) == 1:
                    idx = int(parts[0]) - 1
                    move_list = sorted(moves.keys())
                    if 0 <= idx < len(move_list):
                        return move_list[idx]
                    else:
                        print(f"番号は 1 から {len(move_list)} の範囲で入力してください")
                        continue
                
                # If two numbers, treat as row and col
                elif len(parts) == 2:
                    r, c = int(parts[0]), int(parts[1])
                    if (r, c) in moves:
                        return (r, c)
                    else:
                        print(f"({r}, {c}) は無効な位置です")
                        continue
                else:
                    print("入力形式が正しくありません")
                    continue
                    
            except (ValueError, IndexError):
                print("入力エラー。もう一度入力してください")
                continue


class AIPlayer:
    """AI player that wraps an agent."""
    
    def __init__(self, agent):
        """
        Args:
            agent: An agent object with select_move(board) method
        """
        self.agent = agent
        self.color = agent.color
        self.color_name = "黒(●)" if self.color == BLACK else "白(○)"
    
    def select_move(self, board):
        """
        Gets a move from the AI agent.
        
        Args:
            board: Board object
            
        Returns:
            tuple: (row, col) or None for pass
        """
        moves = board.valid_moves(self.color)
        
        if not moves:
            print(f"{self.color_name}(AI)はパスです")
            return None
        
        move = self.agent.select_move(board)
        print(f"{self.color_name}(AI)が ({move[0]}, {move[1]}) に着手しました")
        return move
