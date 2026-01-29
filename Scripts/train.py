"""
Training script for DQN Othello AI.
Implements self-play training using Deep Q-Learning.

Usage:
    From repo root:
    python -m Scripts.train
    
    Or:
    python Scripts/train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from Scripts.board import Board, BLACK, WHITE
from Scripts.dqn_agent import DQNAgent


# Create necessary directories
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Logs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def play_training_game(agent_black, agent_white, training=True):
    """
    Play one game between two agents and collect experiences.
    
    Args:
        agent_black: DQN agent playing black
        agent_white: DQN agent playing white
        training: Whether agents are in training mode
        
    Returns:
        tuple: (winner, experiences_black, experiences_white)
    """
    board = Board()
    to_play = BLACK
    passes = 0
    
    # Store experiences: (state, action, reward, next_state, done)
    experiences_black = []
    experiences_white = []
    
    # Game states for each player
    prev_state_black = None
    prev_action_black = None
    prev_state_white = None
    prev_action_white = None
    
    while True:
        moves = board.valid_moves(to_play)
        
        if moves:
            passes = 0
            
            # Select agent
            agent = agent_black if to_play == BLACK else agent_white
            
            # Get current state
            state = board.to_tensor(to_play)
            
            # Select action
            move = agent.select_move(board, training=training)
            
            # Store previous state and action
            if to_play == BLACK:
                # Store previous experience if exists
                if prev_state_black is not None:
                    # Intermediate reward is 0 (only terminal reward matters)
                    experiences_black.append((prev_state_black, prev_action_black, 0.0, state, False))
                prev_state_black = state.copy()
                prev_action_black = move
            else:
                if prev_state_white is not None:
                    experiences_white.append((prev_state_white, prev_action_white, 0.0, state, False))
                prev_state_white = state.copy()
                prev_action_white = move
            
            # Apply move
            board.apply_move(move, to_play)
        else:
            passes += 1
            if passes >= 2:
                break
        
        if board.is_full():
            break
        
        to_play = WHITE if to_play == BLACK else BLACK
    
    # Game is over, calculate rewards
    winner = board.winner()
    score = board.score()
    
    # Reward design: +1 for win, -1 for loss, 0 for draw
    reward_black = 0.0
    reward_white = 0.0
    
    if winner == BLACK:
        reward_black = 1.0
        reward_white = -1.0
    elif winner == WHITE:
        reward_black = -1.0
        reward_white = 1.0
    # else: draw, both get 0
    
    # Add terminal experiences
    if prev_state_black is not None:
        # Final state is the current board state
        final_state = board.to_tensor(BLACK)
        experiences_black.append((prev_state_black, prev_action_black, reward_black, final_state, True))
    
    if prev_state_white is not None:
        final_state = board.to_tensor(WHITE)
        experiences_white.append((prev_state_white, prev_action_white, reward_white, final_state, True))
    
    return winner, experiences_black, experiences_white


def train_dqn(num_episodes=10000, batch_size=32, target_update_freq=1000,
              save_freq=1000, eval_freq=1000, print_freq=100):
    """
    Train DQN agent using self-play.
    
    Args:
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        target_update_freq: Frequency to update target network
        save_freq: Frequency to save model
        eval_freq: Frequency to evaluate model
        print_freq: Frequency to print progress
    """
    print("="*60)
    print("DQN Othello AI - Training")
    print("="*60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Target update frequency: {target_update_freq}")
    print("="*60)
    
    # Create agents
    agent_black = DQNAgent(BLACK)
    agent_white = DQNAgent(WHITE)
    
    # Share experience replay buffer for efficiency (both learn from all games)
    shared_buffer = agent_black.replay_buffer
    agent_white.replay_buffer = shared_buffer
    
    # Training statistics
    episode_rewards_black = []
    episode_rewards_white = []
    losses = []
    wins = {BLACK: 0, WHITE: 0, 'draw': 0}
    
    for episode in range(num_episodes):
        # Play one game
        winner, exp_black, exp_white = play_training_game(agent_black, agent_white, training=True)
        
        # Track wins
        if winner is None:
            wins['draw'] += 1
        else:
            wins[winner] += 1
        
        # Add experiences to replay buffer
        for exp in exp_black:
            shared_buffer.push(*exp)
        for exp in exp_white:
            shared_buffer.push(*exp)
        
        # Track rewards
        total_reward_black = sum(exp[2] for exp in exp_black)  # reward is index 2
        total_reward_white = sum(exp[2] for exp in exp_white)
        episode_rewards_black.append(total_reward_black)
        episode_rewards_white.append(total_reward_white)
        
        # Train both agents
        if len(shared_buffer) >= batch_size:
            loss_black = agent_black.train_step(batch_size)
            loss_white = agent_white.train_step(batch_size)
            losses.append((loss_black + loss_white) / 2)
        
        # Decay epsilon
        agent_black.decay_epsilon()
        agent_white.decay_epsilon()
        
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent_black.update_target_network()
            agent_white.update_target_network()
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward_black = np.mean(episode_rewards_black[-print_freq:])
            avg_reward_white = np.mean(episode_rewards_white[-print_freq:])
            avg_loss = np.mean(losses[-print_freq:]) if losses else 0
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Epsilon: {agent_black.epsilon:.4f}")
            print(f"  Avg Reward (Black): {avg_reward_black:.4f}, (White): {avg_reward_white:.4f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Wins (last {print_freq}): Black={wins[BLACK]}, White={wins[WHITE]}, Draw={wins['draw']}")
            wins = {BLACK: 0, WHITE: 0, 'draw': 0}  # Reset for next window
        
        # Save model
        if (episode + 1) % save_freq == 0:
            model_path_black = os.path.join(MODELS_DIR, f'dqn_black_ep{episode+1}.pth')
            model_path_white = os.path.join(MODELS_DIR, f'dqn_white_ep{episode+1}.pth')
            agent_black.save(model_path_black)
            agent_white.save(model_path_white)
            print(f"  Models saved: {model_path_black}, {model_path_white}")
    
    # Save final models
    final_model_black = os.path.join(MODELS_DIR, 'dqn_black_final.pth')
    final_model_white = os.path.join(MODELS_DIR, 'dqn_white_final.pth')
    agent_black.save(final_model_black)
    agent_white.save(final_model_white)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Final models saved: {final_model_black}, {final_model_white}")
    print("="*60)
    
    # Plot learning curves
    plot_learning_curves(episode_rewards_black, episode_rewards_white, losses)
    
    return agent_black, agent_white


def plot_learning_curves(rewards_black, rewards_white, losses):
    """
    Plot learning curves for rewards and losses.
    
    Args:
        rewards_black: List of episode rewards for black
        rewards_white: List of episode rewards for white
        losses: List of training losses
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    window = 100
    if len(rewards_black) >= window:
        avg_rewards_black = [np.mean(rewards_black[max(0, i-window):i+1]) 
                             for i in range(len(rewards_black))]
        avg_rewards_white = [np.mean(rewards_white[max(0, i-window):i+1]) 
                             for i in range(len(rewards_white))]
        
        ax1.plot(avg_rewards_black, label='Black Agent (Avg)', color='black', alpha=0.8)
        ax1.plot(avg_rewards_white, label='White Agent (Avg)', color='gray', alpha=0.8)
    else:
        ax1.plot(rewards_black, label='Black Agent', color='black', alpha=0.5)
        ax1.plot(rewards_white, label='White Agent', color='gray', alpha=0.5)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curve - Episode Rewards (100-episode moving average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        if len(losses) >= window:
            avg_losses = [np.mean(losses[max(0, i-window):i+1]) for i in range(len(losses))]
            ax2.plot(avg_losses, label='Training Loss (Avg)', color='blue', alpha=0.8)
        else:
            ax2.plot(losses, label='Training Loss', color='blue', alpha=0.5)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss (100-episode moving average)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(LOGS_DIR, f'learning_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Learning curve saved: {plot_path}")
    
    # Also save with generic name for easy access
    plot_path_generic = os.path.join(LOGS_DIR, 'learning_curve_latest.png')
    plt.savefig(plot_path_generic, dpi=150)
    print(f"Learning curve saved: {plot_path_generic}")


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("DQN Othello AI Training")
    print("="*60)
    print("\nThis will train a DQN agent through self-play.")
    print("\nTraining parameters:")
    print("  - Algorithm: Deep Q-Network (DQN)")
    print("  - Episodes: 10000 (default)")
    print("  - Learning rate: 0.001")
    print("  - Discount factor (gamma): 0.99")
    print("  - Epsilon: 1.0 → 0.1 (decay)")
    print("  - Batch size: 32")
    print("  - Replay buffer: 10000")
    print("  - Target network update: every 1000 steps")
    
    # Ask user for number of episodes
    print("\nカスタマイズオプション:")
    try:
        episodes_input = input("エピソード数を入力 (デフォルト: 10000, 推奨: 1000-50000): ").strip()
        if episodes_input:
            num_episodes = int(episodes_input)
        else:
            num_episodes = 10000
    except (ValueError, KeyboardInterrupt):
        print("デフォルト値を使用します")
        num_episodes = 10000
    
    print(f"\n学習を開始します... (エピソード数: {num_episodes})")
    print("注意: 学習には時間がかかる場合があります")
    input("Enterキーを押して開始...")
    
    # Train
    try:
        agent_black, agent_white = train_dqn(
            num_episodes=num_episodes,
            batch_size=32,
            target_update_freq=1000,
            save_freq=1000,
            eval_freq=1000,
            print_freq=100
        )
        
        print("\n学習が正常に完了しました！")
        print(f"モデルは {MODELS_DIR} に保存されました")
        print(f"学習曲線は {LOGS_DIR} に保存されました")
        
    except KeyboardInterrupt:
        print("\n\n学習を中断しました")
        print("中間モデルは保存されています")
        sys.exit(0)


if __name__ == '__main__':
    main()
