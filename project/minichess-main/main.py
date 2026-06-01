import numpy as np
from minichess.games.atomic.board import AtomicChessBoard
from minichess.games.dark.board import DarkChessBoard
from minichess.games.rifle.board import RifleChessBoard
from minichess.games.abstract.board import AbstractBoardStatus
from minichess.players.gardner import RandomPlayer
from minichess.games.gardner.board import GardnerChessBoard
from minichess.games.gardner.action import GardnerChessAction

if __name__ == "__main__":
    g = GardnerChessBoard()
    r = RifleChessBoard()
    d = DarkChessBoard()
    a = AtomicChessBoard()
    p = RandomPlayer()

    game = g

    while game.status == AbstractBoardStatus.ONGOING:
        print(game)
        # print(game.no_fog_board())

        actions = game.legal_actions()

        # for action in actions:
        #     print(action)

        # for action in actions:
        #     print(action)
        
        _,proposed = p.propose_action(game, None, game.legal_action_mask())
        print(proposed, proposed.modifier_flags)
        input()
        # proposed = GardnerChessAction.decode(proposed, game)
        game.push(proposed)

        print('+---------------+')

    print(game)

    print(game.status)



