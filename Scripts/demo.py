#!/usr/bin/env python3
"""
Demo script showing different game modes for Othello.
This script runs automated demonstrations of the game.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.board import Board, BLACK, WHITE
from Scripts.player import AIPlayer
from Scripts.agents import RandomAgent, GreedyAgent
from Scripts.game import OthelloGame


def demo_random_vs_greedy():
    """Demonstrate Random AI vs Greedy AI."""
    print("\n" + "="*60)
    print("デモ: ランダムAI（黒） vs 貪欲AI（白）")
    print("="*60)
    
    agent_black = RandomAgent(BLACK)
    agent_white = GreedyAgent(WHITE)
    
    player_black = AIPlayer(agent_black)
    player_white = AIPlayer(agent_white)
    
    game = OthelloGame(player_black, player_white)
    
    # Play the game
    turn_count = 0
    while not game.is_game_over():
        if turn_count % 10 == 0:
            game.display_board()
        
        success = game.play_turn()
        if success:
            game.switch_player()
            turn_count += 1
    
    # Display final result
    game.display_result()


def demo_greedy_vs_greedy():
    """Demonstrate Greedy AI vs Greedy AI."""
    print("\n" + "="*60)
    print("デモ: 貪欲AI（黒） vs 貪欲AI（白）")
    print("="*60)
    
    agent_black = GreedyAgent(BLACK)
    agent_white = GreedyAgent(WHITE)
    
    player_black = AIPlayer(agent_black)
    player_white = AIPlayer(agent_white)
    
    game = OthelloGame(player_black, player_white)
    
    # Play the game
    turn_count = 0
    while not game.is_game_over():
        if turn_count % 10 == 0:
            game.display_board()
        
        success = game.play_turn()
        if success:
            game.switch_player()
            turn_count += 1
    
    # Display final result
    game.display_result()


def run_statistics():
    """Run multiple games and show statistics."""
    print("\n" + "="*60)
    print("統計: 100試合の結果")
    print("="*60)
    
    wins = {BLACK: 0, WHITE: 0, 'draw': 0}
    total_games = 100
    
    for i in range(total_games):
        agent_black = RandomAgent(BLACK)
        agent_white = GreedyAgent(WHITE)
        
        player_black = AIPlayer(agent_black)
        player_white = AIPlayer(agent_white)
        
        game = OthelloGame(player_black, player_white)
        
        # Suppress output
        game.display_board = lambda: None
        
        while not game.is_game_over():
            success = game.play_turn()
            if success:
                game.switch_player()
        
        winner = game.board.winner()
        if winner is None:
            wins['draw'] += 1
        else:
            wins[winner] += 1
        
        if (i + 1) % 20 == 0:
            print(f"進行中... {i + 1}/{total_games} 試合完了")
    
    print(f"\n結果 ({total_games}試合):")
    print(f"  黒（ランダムAI）の勝利: {wins[BLACK]} ({wins[BLACK]/total_games*100:.1f}%)")
    print(f"  白（貪欲AI）の勝利: {wins[WHITE]} ({wins[WHITE]/total_games*100:.1f}%)")
    print(f"  引き分け: {wins['draw']} ({wins['draw']/total_games*100:.1f}%)")
    print("="*60)


def main():
    """Main demo function."""
    print("\n" + "="*60)
    print("オセロゲーム - デモンストレーション")
    print("="*60)
    print("\nこのスクリプトは以下のデモを実行します:")
    print("1. ランダムAI vs 貪欲AI (1試合)")
    print("2. 貪欲AI vs 貪欲AI (1試合)")
    print("3. 統計: ランダムAI vs 貪欲AI (100試合)")
    
    input("\nEnterキーを押して開始...")
    
    # Run demos
    demo_random_vs_greedy()
    input("\nEnterキーを押して次のデモへ...")
    
    demo_greedy_vs_greedy()
    input("\nEnterキーを押して統計を実行...")
    
    run_statistics()
    
    print("\n" + "="*60)
    print("デモ終了")
    print("="*60)
    print("\nゲームをプレイするには以下を実行してください:")
    print("  python -m Scripts.game")
    print("または:")
    print("  python Scripts/game.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nデモを中断しました")
        sys.exit(0)
