"""
Pygame GUI for Othello game.
Provides a graphical interface for playing Othello with mouse clicks.

Usage:
    From repo root:
    python -m Scripts.gui_game

    Or:
    python Scripts/gui_game.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from Scripts.board import Board, BLACK, WHITE, EMPTY
from Scripts.player import AIPlayer
from Scripts.agents import RandomAgent, GreedyAgent

# Colors
COLOR_BACKGROUND = (34, 139, 34)  # Forest green
COLOR_GRID = (0, 0, 0)  # Black grid lines
COLOR_BLACK = (0, 0, 0)  # Black pieces
COLOR_WHITE = (255, 255, 255)  # White pieces
COLOR_VALID_MOVE = (128, 128, 128)  # Gray for valid moves
COLOR_UI_BG = (240, 240, 240)  # Light gray for UI background
COLOR_UI_TEXT = (0, 0, 0)  # Black text
COLOR_HIGHLIGHT = (255, 215, 0)  # Gold for highlighting current player

# Game settings
CELL_SIZE = 80
BOARD_SIZE = 8
BOARD_WIDTH = CELL_SIZE * BOARD_SIZE
UI_HEIGHT = 100
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_WIDTH + UI_HEIGHT
FPS = 30


class GUIPlayer:
    """Human player that uses mouse clicks to select moves."""
    
    def __init__(self, color):
        self.color = color
        self.color_name = "黒" if color == BLACK else "白"


class OthelloGUI:
    """Pygame-based graphical interface for Othello."""
    
    def __init__(self, player_black, player_white):
        """
        Initialize the GUI.
        
        Args:
            player_black: Player object for black
            player_white: Player object for white
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("オセロ / Othello")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.board = Board()
        self.players = {BLACK: player_black, WHITE: player_white}
        self.current_player = BLACK
        self.passes = 0
        self.move_number = 1  # Total moves in the game
        self.game_over = False
        self.winner = None
        self.valid_moves_cache = None  # Cache for valid moves
        self.ai_move_time = 0  # Timestamp for AI move delay
        
    def get_valid_moves(self):
        """Get valid moves for current player, using cache."""
        if self.valid_moves_cache is None:
            self.valid_moves_cache = self.board.valid_moves(self.current_player)
        return self.valid_moves_cache
    
    def get_cell_from_pos(self, pos):
        """
        Convert mouse position to board cell coordinates.
        
        Args:
            pos: (x, y) mouse position
            
        Returns:
            tuple: (row, col) or None if outside board
        """
        x, y = pos
        
        # Check if click is on the board area
        if y >= BOARD_WIDTH:
            return None
        
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)
        return None
    
    def draw_board(self):
        """Draw the game board."""
        # Fill background
        self.screen.fill(COLOR_BACKGROUND)
        
        # Draw grid lines
        for i in range(BOARD_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (0, i * CELL_SIZE),
                (BOARD_WIDTH, i * CELL_SIZE),
                2
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, BOARD_WIDTH),
                2
            )
        
        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = self.board.grid[row][col]
                if cell != EMPTY:
                    center_x = col * CELL_SIZE + CELL_SIZE // 2
                    center_y = row * CELL_SIZE + CELL_SIZE // 2
                    radius = CELL_SIZE // 2 - 5
                    
                    color = COLOR_BLACK if cell == BLACK else COLOR_WHITE
                    pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
        
        # Draw valid move indicators for current player
        if not self.game_over:
            moves = self.get_valid_moves()
            for (row, col) in moves.keys():
                center_x = col * CELL_SIZE + CELL_SIZE // 2
                center_y = row * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // 2 - 5
                
                # Draw gray circle outline
                pygame.draw.circle(self.screen, COLOR_VALID_MOVE, (center_x, center_y), radius, 3)
    
    def draw_ui(self):
        """Draw the UI panel showing game info."""
        # Draw UI background
        ui_rect = pygame.Rect(0, BOARD_WIDTH, WINDOW_WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_UI_BG, ui_rect)
        
        # Get scores
        score = self.board.score()
        
        # Draw move number
        move_text = self.small_font.render(f"Move: {self.move_number}", True, COLOR_UI_TEXT)
        self.screen.blit(move_text, (10, BOARD_WIDTH + 10))
        
        # Draw current player indicator
        if not self.game_over:
            player_name = "黒のターン" if self.current_player == BLACK else "白のターン"
            player_text = self.font.render(player_name, True, COLOR_UI_TEXT)
            
            # Highlight current player
            highlight_x = 10
            highlight_y = BOARD_WIDTH + 35
            pygame.draw.rect(
                self.screen, COLOR_HIGHLIGHT,
                (highlight_x, highlight_y, player_text.get_width() + 10, player_text.get_height() + 5),
                0, 5
            )
            self.screen.blit(player_text, (highlight_x + 5, highlight_y + 2))
        else:
            # Game over message
            if self.winner is None:
                result_text = "引き分け!"
            elif self.winner == BLACK:
                result_text = "黒の勝利!"
            else:
                result_text = "白の勝利!"
            
            result_surface = self.font.render(result_text, True, COLOR_UI_TEXT)
            self.screen.blit(result_surface, (10, BOARD_WIDTH + 35))
        
        # Draw scores
        black_score_text = self.small_font.render(f"黒: {score[BLACK]}", True, COLOR_UI_TEXT)
        white_score_text = self.small_font.render(f"白: {score[WHITE]}", True, COLOR_UI_TEXT)
        
        score_x = WINDOW_WIDTH - 150
        self.screen.blit(black_score_text, (score_x, BOARD_WIDTH + 20))
        self.screen.blit(white_score_text, (score_x, BOARD_WIDTH + 50))
    
    def handle_click(self, pos):
        """
        Handle mouse click event.
        
        Args:
            pos: (x, y) mouse position
        """
        if self.game_over:
            return
        
        cell = self.get_cell_from_pos(pos)
        if cell is None:
            return
        
        row, col = cell
        
        # Check if it's a valid move
        moves = self.get_valid_moves()
        if (row, col) in moves:
            # Apply the move
            self.board.apply_move((row, col), self.current_player)
            self.passes = 0
            self.switch_player()
            self.move_number += 1
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self.valid_moves_cache = None  # Invalidate cache
    
    def check_game_over(self):
        """Check if the game is over."""
        # Game over if board is full
        if self.board.is_full():
            return True
        
        # Game over if both players passed
        if self.passes >= 2:
            return True
        
        return False
    
    def handle_ai_turn(self):
        """Handle AI player's turn."""
        player = self.players[self.current_player]
        
        # Check if it's an AI player
        if not isinstance(player, AIPlayer):
            return False
        
        # Check if player has valid moves
        moves = self.get_valid_moves()
        
        if not moves:
            # Pass
            self.passes += 1
            self.switch_player()
            return True
        
        # Get move from AI
        move = player.select_move(self.board)
        
        if move:
            # Apply move
            self.board.apply_move(move, self.current_player)
            self.passes = 0
            self.switch_player()
            self.move_number += 1
            return True
        
        return False
    
    def handle_human_pass(self):
        """Handle pass for human player when no moves available."""
        player = self.players[self.current_player]
        
        # Only for human players (GUIPlayer)
        if not isinstance(player, GUIPlayer):
            return
        
        moves = self.get_valid_moves()
        if not moves:
            # Pass
            self.passes += 1
            self.switch_player()
            # Show pass message in UI (will be visible on next frame)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            self.clock.tick(FPS)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if isinstance(self.players[self.current_player], GUIPlayer):
                        self.handle_click(event.pos)
            
            # Check for game over
            if self.check_game_over():
                if not self.game_over:
                    self.game_over = True
                    self.winner = self.board.winner()
            
            # Handle AI turn if current player is AI
            if not self.game_over:
                if isinstance(self.players[self.current_player], AIPlayer):
                    # Use timestamp to delay AI moves
                    current_time = pygame.time.get_ticks()
                    if self.ai_move_time == 0:
                        self.ai_move_time = current_time + 500  # Set delay
                    elif current_time >= self.ai_move_time:
                        self.handle_ai_turn()
                        self.ai_move_time = 0  # Reset for next AI move
                else:
                    self.ai_move_time = 0  # Reset when human player
                    # Handle human pass
                    self.handle_human_pass()
            
            # Draw everything
            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
        
        pygame.quit()


def select_player_type(color_name):
    """
    Let the user select player type.
    
    Args:
        color_name: "黒" or "白"
        
    Returns:
        Player object
    """
    print(f"\n{color_name}のプレイヤーを選択してください:")
    print("  1: 人間 (マウスクリック)")
    print("  2: ランダムAI")
    print("  3: 貪欲AI (最も多く石を取る手を選ぶ)")
    print("  4: 学習済みDQN AI (Deep Q-Network)")
    
    while True:
        try:
            choice = input("選択 (1-4): ").strip()
            
            color = BLACK if color_name == "黒" else WHITE
            
            if choice == "1":
                return GUIPlayer(color)
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
                except Exception as e:
                    print(f"エラー: DQNモデルの読み込みに失敗しました: {e}")
                    continue
            else:
                print("1, 2, 3, または 4 を入力してください")
        except KeyboardInterrupt:
            print("\n\nプログラムを中断しました")
            pygame.quit()
            sys.exit(0)
        except ValueError:
            print("\n入力エラー。もう一度入力してください")


def main():
    """Main function to start the GUI game."""
    print("="*40)
    print("オセロゲーム (GUI版)")
    print("="*40)
    
    # Select players
    player_black = select_player_type("黒")
    player_white = select_player_type("白")
    
    # Create and run GUI game
    game = OthelloGUI(player_black, player_white)
    game.run()
    
    print("\nゲームを終了しました")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nゲームを中断しました")
        pygame.quit()
        sys.exit(0)
