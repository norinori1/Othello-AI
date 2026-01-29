"""
Evaluation script for trained DQN Othello AI.
Tests the trained agent against random and greedy opponents.

Usage:
    From repo root:
    python -m Scripts.evaluate
    
    Or:
    python Scripts/evaluate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from Scripts.board import Board, BLACK, WHITE
from Scripts.dqn_agent import DQNAgent
from Scripts.agents import RandomAgent, GreedyAgent
from Scripts.game import OthelloGame
from Scripts.player import AIPlayer


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models')


def play_evaluation_game(agent1, agent2, color1=BLACK):
    """
    Play one evaluation game between two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        color1: Color for agent1 (BLACK or WHITE)
        
    Returns:
        tuple: (winner, score_dict)
    """
    board = Board()
    to_play = BLACK
    passes = 0
    
    # Map agents to colors
    agents = {color1: agent1, (WHITE if color1 == BLACK else BLACK): agent2}
    
    while True:
        moves = board.valid_moves(to_play)
        
        if moves:
            passes = 0
            agent = agents[to_play]
            
            # Select move (not in training mode)
            if isinstance(agent, DQNAgent):
                move = agent.select_move(board, training=False)
            else:
                move = agent.select_move(board)
            
            board.apply_move(move, to_play)
        else:
            passes += 1
            if passes >= 2:
                break
        
        if board.is_full():
            break
        
        to_play = WHITE if to_play == BLACK else BLACK
    
    winner = board.winner()
    score = board.score()
    
    return winner, score


def evaluate_against_opponent(dqn_agent, opponent_name, opponent_factory, 
                               num_games=100, dqn_color=BLACK):
    """
    Evaluate DQN agent against an opponent.
    
    Args:
        dqn_agent: Trained DQN agent
        opponent_name: Name of opponent for display
        opponent_factory: Function that creates opponent agent
        num_games: Number of games to play
        dqn_color: Color for DQN agent
        
    Returns:
        dict: Evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating against {opponent_name} ({num_games} games)")
    print(f"DQN Agent: {'Black' if dqn_color == BLACK else 'White'}")
    print(f"{'='*60}")
    
    wins = 0
    losses = 0
    draws = 0
    total_score_dqn = 0
    total_score_opponent = 0
    
    for i in range(num_games):
        # Create opponent
        opponent_color = WHITE if dqn_color == BLACK else BLACK
        opponent = opponent_factory(opponent_color)
        
        # Play game
        winner, score = play_evaluation_game(dqn_agent, opponent, color1=dqn_color)
        
        # Record results
        if winner == dqn_agent.color:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        
        total_score_dqn += score[dqn_agent.color]
        total_score_opponent += score[opponent_color]
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_games} games")
    
    # Calculate statistics
    win_rate = wins / num_games * 100
    avg_score_dqn = total_score_dqn / num_games
    avg_score_opponent = total_score_opponent / num_games
    
    print(f"\n結果:")
    print(f"  勝利: {wins}/{num_games} ({win_rate:.1f}%)")
    print(f"  敗北: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"  引き分け: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"  平均獲得石数 (DQN): {avg_score_dqn:.1f}")
    print(f"  平均獲得石数 ({opponent_name}): {avg_score_opponent:.1f}")
    print(f"{'='*60}")
    
    return {
        'opponent': opponent_name,
        'games': num_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_score_dqn': avg_score_dqn,
        'avg_score_opponent': avg_score_opponent
    }


def evaluate_model(model_path, num_games=100):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the model file
        num_games: Number of games to play against each opponent
    """
    print("="*60)
    print("DQN Model Evaluation")
    print("="*60)
    print(f"Loading model: {model_path}")
    
    # Load agent
    agent = DQNAgent(BLACK, epsilon=0.0)  # No exploration during evaluation
    try:
        agent.load(model_path)
        print(f"Model loaded successfully!")
        print(f"  Episodes trained: {agent.episode_count}")
        print(f"  Training steps: {agent.training_step}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate against different opponents
    results = []
    
    # 1. Against Random AI
    result_random = evaluate_against_opponent(
        agent, "Random AI", 
        lambda c: RandomAgent(c),
        num_games=num_games,
        dqn_color=BLACK
    )
    results.append(result_random)
    
    # 2. Against Greedy AI
    result_greedy = evaluate_against_opponent(
        agent, "Greedy AI",
        lambda c: GreedyAgent(c),
        num_games=num_games,
        dqn_color=BLACK
    )
    results.append(result_greedy)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for result in results:
        print(f"\nvs {result['opponent']}:")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Avg Score: {result['avg_score_dqn']:.1f}")
    print("="*60)
    
    return results


def main():
    """Main evaluation function."""
    print("\n" + "="*60)
    print("DQN Othello AI - Evaluation")
    print("="*60)
    
    # Check for available models
    if not os.path.exists(MODELS_DIR):
        print(f"\nエラー: Models ディレクトリが見つかりません")
        print("先に学習を実行してください: python -m Scripts.train")
        return
    
    # List available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth') and 'black' in f]
    
    if not model_files:
        print(f"\nエラー: 学習済みモデルが見つかりません")
        print("先に学習を実行してください: python -m Scripts.train")
        return
    
    print("\n利用可能なモデル:")
    for i, model in enumerate(sorted(model_files)):
        print(f"  {i+1}: {model}")
    
    # Select model
    print("\n評価するモデルを選択してください")
    print("(Enterキーのみで最新のfinalモデルを使用)")
    
    try:
        choice = input("選択: ").strip()
        if choice:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                model_name = sorted(model_files)[idx]
            else:
                print("無効な選択です。finalモデルを使用します")
                model_name = 'dqn_black_final.pth'
        else:
            model_name = 'dqn_black_final.pth'
    except (ValueError, KeyboardInterrupt):
        print("finalモデルを使用します")
        model_name = 'dqn_black_final.pth'
    
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nエラー: {model_path} が見つかりません")
        return
    
    # Get number of evaluation games
    try:
        num_games_input = input("\n評価ゲーム数 (デフォルト: 100): ").strip()
        if num_games_input:
            num_games = int(num_games_input)
        else:
            num_games = 100
    except (ValueError, KeyboardInterrupt):
        print("デフォルト値 (100) を使用します")
        num_games = 100
    
    print(f"\n評価を開始します...")
    input("Enterキーを押して開始...")
    
    # Evaluate
    evaluate_model(model_path, num_games=num_games)
    
    print("\n評価が完了しました！")


if __name__ == '__main__':
    main()
