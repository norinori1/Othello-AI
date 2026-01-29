"""
Simple self-play runner.

Usage (from repo root):
python -m Scripts.self_play

This script runs `num_games` between two agents and saves trajectories to Logs/selfplay.pkl
"""
import os
import pickle
from .board import Board, BLACK, WHITE
from .agents import RandomAgent, GreedyAgent

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

def play_game(agent_black, agent_white, return_trajectory=True):
    board = Board()
    to_play = BLACK
    passes = 0

    traj = []  # list of (state_tensor, action, to_play)

    while True:
        moves = board.valid_moves(to_play)
        if moves:
            passes = 0
            if to_play == BLACK:
                move = agent_black.select_move(board)
            else:
                move = agent_white.select_move(board)
            # record state and action
            state = board.to_tensor(to_play)
            traj.append((state, move, to_play))
            board.apply_move(move, to_play)
        else:
            passes += 1
            if passes >= 2:
                break
        if board.is_full():
            break
        to_play = WHITE if to_play == BLACK else BLACK

    winner = board.winner()
    result = {BLACK: -1, WHITE: -1}
    if winner is None:
        result = {BLACK: 0, WHITE: 0}
    else:
        result[winner] = 1
        result[board.opponent(winner)] = -1

    if return_trajectory:
        # assign final reward to each step for the player who made it
        labeled = []
        for state, action, player in traj:
            labeled.append((state, action, result[player]))
        return labeled, winner, board
    return None, winner, board


def run_selfplay(num_games=100, agent_type='random'):
    if agent_type == 'random':
        def make(c): return RandomAgent(c)
    else:
        def make(c): return GreedyAgent(c)

    all_games = []
    wins = {BLACK:0, WHITE:0, 'draw':0}
    for i in range(num_games):
        black = make(BLACK)
        white = make(WHITE)
        traj, winner, board = play_game(black, white, return_trajectory=True)
        all_games.extend(traj)
        if winner is None:
            wins['draw'] += 1
        else:
            wins[winner] += 1
        if (i+1) % 10 == 0:
            print(f"Played {i+1}/{num_games} games")

    out_path = os.path.join(LOG_DIR, 'selfplay.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'trajectories': all_games, 'wins': wins}, f)

    print('Self-play finished')
    print('Wins:', wins)
    print('Saved trajectories to', out_path)
    return out_path


if __name__ == '__main__':
    # quick demo
    run_selfplay(num_games=50, agent_type='random')
