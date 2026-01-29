"""
Main game interface for Othello.
Allows humans and AIs to play Othello interactively.

Usage:
    From repo root:
    python -m Scripts.game

    Or:
    python Scripts/game.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.board import Board, BLACK, WHITE, EMPTY
from Scripts.player import HumanPlayer, AIPlayer
from Scripts.agents import RandomAgent, GreedyAgent
import os


class OthelloGame:
    """Main Othello game controller."""
    
    def __init__(self, player_black, player_white):
        """
        Initialize the game.
        
        Args:
            player_black: Player object for black
            player_white: Player object for white
        """
        self.board = Board()
        self.players = {BLACK: player_black, WHITE: player_white}
        self.current_player = BLACK
        self.passes = 0
        
    def display_board(self):
        """Display the current board state."""
        print("\n" + "="*40)
        print("現在のボード:")
        print("  0 1 2 3 4 5 6 7")
        for i, line in enumerate(self.board.render().split('\n')):
            print(f"{i} {' '.join(line)}")
        
        score = self.board.score()
        print(f"\nスコア: 黒(●) {score[BLACK]} - {score[WHITE]} 白(○)")
        print("="*40)
    
    def play_turn(self):
        """Execute one turn of the game."""
        player = self.players[self.current_player]
        
        # Check if player has valid moves
        moves = self.board.valid_moves(self.current_player)
        
        if not moves:
            # Pass - display message
            color_name = "黒(●)" if self.current_player == BLACK else "白(○)"
            print(f"{color_name}はパスです")
            if hasattr(player, '__class__') and player.__class__.__name__ == 'HumanPlayer':
                input("Enterキーを押して続行...")
            self.passes += 1
            return True
        
        # Reset pass counter when a move is made
        self.passes = 0
        
        # Get move from player
        move = player.select_move(self.board)
        
        # Apply move
        try:
            self.board.apply_move(move, self.current_player)
            return True
        except ValueError as e:
            color_name = "黒(●)" if self.current_player == BLACK else "白(○)"
            print(f"エラー ({color_name}): {e}")
            return False
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = WHITE if self.current_player == BLACK else BLACK
    
    def is_game_over(self):
        """Check if the game is over."""
        # Game over if board is full
        if self.board.is_full():
            return True
        
        # Game over if both players passed
        if self.passes >= 2:
            return True
        
        # Game over if neither player has valid moves
        if not self.board.valid_moves(BLACK) and not self.board.valid_moves(WHITE):
            return True
        
        return False
    
    def display_result(self):
        """Display the final game result."""
        print("\n" + "="*40)
        print("ゲーム終了！")
        self.display_board()
        
        winner = self.board.winner()
        score = self.board.score()
        
        if winner is None:
            print("\n引き分けです！")
        elif winner == BLACK:
            print("\n黒(●)の勝利です！")
        else:
            print("\n白(○)の勝利です！")
        
        print(f"最終スコア: 黒(●) {score[BLACK]} - {score[WHITE]} 白(○)")
        print("="*40 + "\n")
    
    def play(self):
        """Main game loop."""
        print("\n" + "="*40)
        print("オセロゲームを開始します！")
        print("="*40)
        
        while not self.is_game_over():
            self.display_board()
            
            success = self.play_turn()
            if success:
                self.switch_player()
        
        self.display_result()


def select_player_type(color_name):
    """
    Let the user select player type.
    
    Args:
        color_name: "黒" or "白"
        
    Returns:
        Player object
    """
    print(f"\n{color_name}のプレイヤーを選択してください:")
    print("  1: 人間")
    print("  2: ランダムAI")
    print("  3: 貪欲AI (最も多く石を取る手を選ぶ)")
    print("  4: 学習済みDQN AI (Deep Q-Network)")
    
    while True:
        try:
            choice = input("選択 (1-4): ").strip()
            color = BLACK if color_name == "黒" else WHITE
            
            if choice == "1":
                return HumanPlayer(color)
            elif choice == "2":
                agent = RandomAgent(color)
                return AIPlayer(agent)
            elif choice == "3":
                agent = GreedyAgent(color)
                return AIPlayer(agent)
            elif choice == "4":
                # Try to load DQN agent
                try:
                    from Scripts.dqn_agent import DQNAgent
                    import torch
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Models')
                    model_file = f'dqn_{"black" if color == BLACK else "white"}_final.pth'
                    model_path = os.path.join(models_dir, model_file)
                    
                    if not os.path.exists(model_path):
                        print(f"\nエラー: モデルファイルが見つかりません: {model_path}")
                        print("先に学習を実行してください: python -m Scripts.train")
                        continue
                    
                    agent = DQNAgent(color, epsilon=0.0)  # No exploration during play
                    agent.load(model_path)
                    print(f"学習済みDQNモデルを読み込みました (エピソード: {agent.episode_count})")
                    return AIPlayer(agent)
                except ImportError as e:
                    print(f"エラー: PyTorchがインストールされていません: {e}")
                    print("pip install torch を実行してください")
                    continue
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"エラー: モデルの読み込みに失敗しました: {e}")
                    continue
            else:
                print("1, 2, 3, または 4 を入力してください")
        except (ValueError, KeyboardInterrupt):
            print("\n入力エラー。もう一度入力してください")


def main():
    """Main function to start the game."""
    print("="*40)
    print("オセロゲーム")
    print("="*40)
    
    # Select players
    player_black = select_player_type("黒")
    player_white = select_player_type("白")
    
    # Create and play game
    game = OthelloGame(player_black, player_white)
    game.play()
    
    # Ask if player wants to play again
    while True:
        play_again = input("もう一度プレイしますか？ (y/n): ").strip().lower()
        if play_again in ['y', 'yes', 'はい']:
            # Use loop instead of recursion to avoid stack overflow
            print("\n" + "="*40)
            print("新しいゲームを開始します")
            print("="*40)
            
            # Select players again
            player_black = select_player_type("黒")
            player_white = select_player_type("白")
            
            # Create and play game
            game = OthelloGame(player_black, player_white)
            game.play()
        elif play_again in ['n', 'no', 'いいえ']:
            print("ありがとうございました！")
            break
        else:
            print("y または n を入力してください")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nゲームを中断しました")
        sys.exit(0)
