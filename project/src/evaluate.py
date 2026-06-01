"""Evaluation utilities for DQN/DDQN agents and MCTS."""

import numpy as np
from minichess.games.abstract.piece import PieceColor
from minichess.games.abstract.board import AbstractBoardStatus

from src.env import MiniChessEnv, encode_state, encode_legal_mask
from src.mcts_agent import MCTSAgent, deep_copy_board


# ---------------------------------------------------------------------------
# Evaluate a DQN / DDQN agent against an opponent
# ---------------------------------------------------------------------------

def evaluate_agent(agent, opponent, config, num_games=50):
    """Play *num_games*, alternating colours.

    Returns (win_rate, loss_rate, draw_rate).
    """
    env = MiniChessEnv(opponent=opponent, max_plies=config.max_plies)
    wins = losses = draws = 0

    for g in range(num_games):
        color = PieceColor.WHITE if g % 2 == 0 else PieceColor.BLACK
        state, mask, done = env.reset(color)
        reward = 0.0

        while not done:
            action = agent.select_action_greedy(state, mask)
            state, reward, done, mask = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    n = num_games
    return (wins / n, losses / n, draws / n)


# ---------------------------------------------------------------------------
# Evaluate MCTS against an opponent
# ---------------------------------------------------------------------------

def evaluate_mcts(mcts_agent, opponent, config, num_games=50, verbose=False):
    """Play MCTS as the agent vs *opponent*, alternating colours.

    Returns (win_rate, loss_rate, draw_rate).
    """
    env = MiniChessEnv(opponent=opponent, max_plies=config.max_plies)
    wins = losses = draws = 0

    for g in range(num_games):
        color = PieceColor.WHITE if g % 2 == 0 else PieceColor.BLACK
        state, mask, done = env.reset(color)

        while not done:
            # MCTS works directly on the board
            mcts_action = mcts_agent.select_action(env.board)
            if mcts_action is None:
                break
            # convert to action_id so we can call env.step
            from src.env import _canonical_pos
            fr, fc = _canonical_pos(*mcts_action.from_pos, color)
            tr, tc = _canonical_pos(*mcts_action.to_pos, color)
            action_id = (fr * 5 + fc) * 25 + (tr * 5 + tc)
            state, reward, done, mask = env.step(action_id)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        if verbose:
            print(f"  MCTS game {g+1}/{num_games}  "
                  f"result={'W' if reward>0 else 'L' if reward<0 else 'D'}")

    n = num_games
    return (wins / n, losses / n, draws / n)
