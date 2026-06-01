"""
MiniChess Gym-like environment wrapper for Gardner 5x5 Chess.

State  : 12 x 5 x 5  (own/opp x {P,N,B,R,Q,K}), canonical perspective
Action : 625           (from_sq * 25 + to_sq), canonical coordinates
Reward : terminal only (+1 win, -1 loss, 0 draw / move-limit)
"""

import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'minichess-main'))

from minichess.games.gardner.board import GardnerChessBoard
from minichess.games.gardner.pieces import Pawn, Knight, Bishop, Rook, Queen, King
from minichess.games.abstract.board import AbstractBoardStatus
from minichess.games.abstract.piece import PieceColor
from minichess.games.abstract.action import AbstractActionFlags
from minichess.games.gardner.action import GardnerChessAction

ACTION_SIZE = 625   # 25 from-squares x 25 to-squares
STATE_CHANNELS = 12
BOARD_SIZE = 5

# Piece-type → channel index
_PIECE_CHANNEL = {Pawn: 0, Knight: 1, Bishop: 2, Rook: 3, Queen: 4, King: 5}


# ---------------------------------------------------------------------------
# State / action encoding helpers
# ---------------------------------------------------------------------------

def encode_state(board):
    """Return 12x5x5 float32 tensor from *active player's* canonical perspective.

    Channels 0-5 : own {P,N,B,R,Q,K}
    Channels 6-11: opponent {P,N,B,R,Q,K}
    When active_color is Black the board is rotated 180 degrees so the network
    always sees the game as-if it were White.
    """
    state = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    cur = board.active_color
    flip = (cur == PieceColor.BLACK)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board._board[r][c].peek()
            if piece is None:
                continue
            ch = _PIECE_CHANNEL[type(piece)]
            own = (piece.color == cur)
            offset = 0 if own else 6
            if flip:
                state[offset + ch, 4 - r, 4 - c] = 1.0
            else:
                state[offset + ch, r, c] = 1.0
    return state


def _canonical_pos(row, col, color):
    """Flip position for Black so canonical view is always White-oriented."""
    if color == PieceColor.BLACK:
        return (4 - row, 4 - col)
    return (row, col)


def encode_legal_mask(board):
    """Return shape-(625,) float32 mask in canonical coordinates."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    color = board.active_color
    for act in board.legal_actions():
        fr, fc = _canonical_pos(*act.from_pos, color)
        tr, tc = _canonical_pos(*act.to_pos, color)
        idx = (fr * 5 + fc) * 25 + (tr * 5 + tc)
        mask[idx] = 1.0
    return mask


def decode_action(action_id, board):
    """Convert canonical action_id (0-624) to a GardnerChessAction on *board*.

    Finds the matching legal action.  Defaults to queen promotion when a pawn
    reaches the back rank.
    """
    from_sq = action_id // 25
    to_sq = action_id % 25
    fr, fc = from_sq // 5, from_sq % 5
    tr, tc = to_sq // 5, to_sq % 5

    color = board.active_color
    if color == PieceColor.BLACK:
        fr, fc = 4 - fr, 4 - fc
        tr, tc = 4 - tr, 4 - tc

    from_pos = (fr, fc)
    to_pos = (tr, tc)

    # prefer queen promotion among matching legal actions
    queen_promo = None
    fallback = None
    for act in board.legal_actions():
        if act.from_pos == from_pos and act.to_pos == to_pos:
            if AbstractActionFlags.PROMOTE_QUEEN in act.modifier_flags:
                queen_promo = act
            if fallback is None:
                fallback = act
    return queen_promo if queen_promo is not None else fallback


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MiniChessEnv:
    """Two-player environment: agent acts, then an opponent auto-responds."""

    def __init__(self, opponent, max_plies=200):
        """
        Parameters
        ----------
        opponent : object with ``select_action(board) -> GardnerChessAction``
        max_plies : int  – half-move limit (both players combined)
        """
        self.opponent = opponent
        self.max_plies = max_plies
        self.board = None
        self.agent_color = None
        self.ply = 0

    # ---- public API -------------------------------------------------------

    def reset(self, agent_color=PieceColor.WHITE):
        """Reset board. Returns (state, legal_mask, done)."""
        self.board = GardnerChessBoard()
        self.agent_color = agent_color
        self.ply = 0

        # if agent is Black, opponent (White) moves first
        if self.agent_color == PieceColor.BLACK:
            self._opponent_move()

        state = encode_state(self.board)
        mask = encode_legal_mask(self.board)
        done = self._is_terminal()
        return state, mask, done

    def step(self, action_id):
        """Apply agent's action, let opponent respond.

        Returns (next_state, reward, done, next_legal_mask).
        """
        assert self.board.active_color == self.agent_color, "Not agent's turn"

        # --- agent moves ---
        action = decode_action(action_id, self.board)
        assert action is not None, f"No matching legal action for id {action_id}"
        self.board.push(action, check_for_check=False)
        self.ply += 1

        if self._is_terminal():
            return self._terminal_return()

        # --- opponent responds ---
        self._opponent_move()

        if self._is_terminal():
            return self._terminal_return()

        # not done
        state = encode_state(self.board)
        mask = encode_legal_mask(self.board)
        return state, 0.0, False, mask

    # ---- internals --------------------------------------------------------

    def _opponent_move(self):
        action = self.opponent.select_action(self.board)
        if action is not None:
            self.board.push(action, check_for_check=False)
            self.ply += 1

    def _is_terminal(self):
        return (self.board.status != AbstractBoardStatus.ONGOING
                or self.ply >= self.max_plies)

    def _compute_reward(self):
        st = self.board.status
        if st == AbstractBoardStatus.WHITE_WIN:
            return 1.0 if self.agent_color == PieceColor.WHITE else -1.0
        if st == AbstractBoardStatus.BLACK_WIN:
            return 1.0 if self.agent_color == PieceColor.BLACK else -1.0
        return 0.0   # draw or move-limit

    def _terminal_return(self):
        state = encode_state(self.board)
        mask = encode_legal_mask(self.board)
        return state, self._compute_reward(), True, mask
