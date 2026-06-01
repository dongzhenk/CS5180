"""
Plain MCTS with random rollout policy for Gardner 5x5 Mini-Chess.

Values are stored from White's perspective at every node so that
selection can simply maximise or minimise depending on whose turn it is.
"""

import math, random
import numpy as np

from minichess.games.gardner.board import GardnerChessBoard
from minichess.games.abstract.board import AbstractBoardStatus, AbstractChessTile
from minichess.games.abstract.piece import PieceColor


# ---------------------------------------------------------------------------
# Deep copy (the library's copy() is shallow on tiles)
# ---------------------------------------------------------------------------

def deep_copy_board(board):
    """Properly deep-copy a GardnerChessBoard (tiles + pieces)."""
    new_grid = []
    for r in range(5):
        row = []
        for c in range(5):
            row.append(board._board[r][c].copy())
        new_grid.append(row)
    new_board = GardnerChessBoard(board=new_grid)
    new_board.active_color = board.active_color
    new_board.move_history = []
    return new_board


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ('board', 'parent', 'action', 'children',
                 'untried_actions', 'visits', 'white_value')

    def __init__(self, board, parent=None, action=None):
        self.board = board
        self.parent = parent
        self.action = action          # action that led here
        self.children = []
        self.visits = 0
        self.white_value = 0.0        # cumulative value from White POV

        if board.status == AbstractBoardStatus.ONGOING:
            self.untried_actions = list(board.legal_actions())
            random.shuffle(self.untried_actions)
        else:
            self.untried_actions = []

    def is_terminal(self):
        return self.board.status != AbstractBoardStatus.ONGOING

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def ucb1_score(self, c):
        """Score used during selection.  White nodes maximise, Black minimises."""
        if self.visits == 0:
            return float('inf')
        exploit = self.white_value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        # parent picks: maximise if parent is White, minimise if Black
        if self.parent.board.active_color == PieceColor.WHITE:
            return exploit + explore
        else:
            return -exploit + explore

    def best_child(self, c):
        return max(self.children, key=lambda ch: ch.ucb1_score(c))

    def expand(self):
        action = self.untried_actions.pop()
        child_board = deep_copy_board(self.board)
        child_board.push(action, check_for_check=False)
        child = MCTSNode(child_board, parent=self, action=action)
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# Random rollout
# ---------------------------------------------------------------------------

def _rollout(board, max_depth=200):
    """Play random moves to termination.  Returns value from White's POV."""
    b = deep_copy_board(board)
    depth = 0
    while b.status == AbstractBoardStatus.ONGOING and depth < max_depth:
        actions = b.legal_actions()
        if not actions:
            break
        b.push(random.choice(actions), check_for_check=False)
        depth += 1
    st = b.status
    if st == AbstractBoardStatus.WHITE_WIN:
        return 1.0
    if st == AbstractBoardStatus.BLACK_WIN:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------

def mcts_search(board, num_simulations=200, c=1.414):
    """Run MCTS from *board* and return the best GardnerChessAction."""
    root = MCTSNode(deep_copy_board(board))

    for _ in range(num_simulations):
        node = root

        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(c)

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation (random rollout)
        value = _rollout(node.board)

        # 4. Back-propagation (all values stored from White's POV)
        while node is not None:
            node.visits += 1
            node.white_value += value
            node = node.parent

    if not root.children:
        return None
    # pick the most-visited child (robust selection)
    best = max(root.children, key=lambda ch: ch.visits)
    return best.action


# ---------------------------------------------------------------------------
# Agent wrapper (matches opponent interface)
# ---------------------------------------------------------------------------

class MCTSAgent:
    """Opponent / agent interface: ``select_action(board) -> action``."""
    def __init__(self, num_simulations=200, c=1.414):
        self.num_simulations = num_simulations
        self.c = c

    def select_action(self, board):
        return mcts_search(board, self.num_simulations, self.c)
