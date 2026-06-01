"""Opponent agents: Random, Heuristic, and frozen-DQN for self-play."""

import random
import numpy as np
import torch

from minichess.games.abstract.action import AbstractActionFlags
from minichess.games.abstract.board import AbstractBoardStatus

from src.env import encode_state, encode_legal_mask, decode_action


# ---------------------------------------------------------------------------
# Random opponent
# ---------------------------------------------------------------------------

class RandomOpponent:
    """Picks a uniformly random legal move."""
    def select_action(self, board):
        actions = board.legal_actions()
        if not actions:
            return None
        return random.choice(actions)


# ---------------------------------------------------------------------------
# Heuristic opponent  (checks > captures > random)
# ---------------------------------------------------------------------------

class HeuristicOpponent:
    """Priority: checking moves > captures > random legal move."""

    def select_action(self, board):
        actions = board.legal_actions()
        if not actions:
            return None

        # 1. check-giving moves
        checks = []
        for act in actions:
            if self._gives_check(act, board):
                checks.append(act)
        if checks:
            return random.choice(checks)

        # 2. captures
        captures = [a for a in actions
                    if AbstractActionFlags.CAPTURE in a.modifier_flags]
        if captures:
            return random.choice(captures)

        # 3. random
        return random.choice(actions)

    @staticmethod
    def _gives_check(action, board):
        """Simulate action and test whether opponent's king is attacked."""
        board.push(action, check_for_check=False)
        # after push, active_color has flipped to opponent
        # check if the mover can now capture the king
        mover_color = action.agent.color
        opponent_actions = board.legal_actions_for_color(mover_color,
                                                         filter_for_check=False)
        in_check = any(AbstractActionFlags.KING_CAPTURE in a.modifier_flags
                       for a in opponent_actions)
        board.pop()
        return in_check


# ---------------------------------------------------------------------------
# Frozen DQN opponent (for self-play training)
# ---------------------------------------------------------------------------

class DQNOpponent:
    """Uses a frozen Q-network.  Canonical encoding adapts automatically."""

    def __init__(self, net, device, epsilon=0.05):
        self.net = net            # frozen copy, eval mode
        self.device = device
        self.epsilon = epsilon

    def select_action(self, board):
        actions = board.legal_actions()
        if not actions:
            return None

        state = encode_state(board)
        mask = encode_legal_mask(board)
        legal_ids = np.where(mask > 0)[0]

        if random.random() < self.epsilon:
            action_id = int(np.random.choice(legal_ids))
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.net(s).squeeze(0).cpu().numpy()
            q[mask == 0] = -np.inf
            action_id = int(np.argmax(q))

        return decode_action(action_id, board)
