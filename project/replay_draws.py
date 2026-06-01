import sys, random
sys.path.insert(0, '.')
sys.path.insert(0, 'minichess-main')
import torch, numpy as np
from minichess.games.gardner.board import GardnerChessBoard
from minichess.games.abstract.board import AbstractBoardStatus
from minichess.games.abstract.piece import PieceColor
from minichess.games.gardner.pieces import King
from minichess.games.abstract.action import AbstractActionFlags
from src.env import encode_state, encode_legal_mask, decode_action
from src.model import QNetwork

device = torch.device('cpu')
net = QNetwork().to(device)
ckpt = torch.load('results/DQN_seed42/checkpoint.pt', map_location=device, weights_only=False)
net.load_state_dict(ckpt['online'])
net.eval()

def pn(p):
    return ('W' if p.color == PieceColor.WHITE else 'B') + type(p).__name__[0]

def act(board):
    s = encode_state(board)
    m = encode_legal_mask(board)
    with torch.no_grad():
        q = net(torch.FloatTensor(s).unsqueeze(0)).squeeze(0).numpy()
    q[m == 0] = -np.inf
    return decode_action(int(np.argmax(q)), board)

cases = []
for gid in range(500):
    board = GardnerChessBoard()
    hist = []
    ply = 0
    while board.status == AbstractBoardStatus.ONGOING and ply < 200:
        n = sum(1 for t in board if t.peek())
        c = 'W' if board.active_color == PieceColor.WHITE else 'B'
        a = act(board)
        if a is None:
            break
        cap = ''
        if AbstractActionFlags.CAPTURE in a.modifier_flags and a.captured_piece:
            cap = ' x' + pn(a.captured_piece)
        hist.append((ply, c, pn(a.agent) + ' ' + str(a.from_pos) + '->' + str(a.to_pos) + cap, n))
        board.push(a, check_for_check=False)
        ply += 1

    ko = all(type(t.peek()) == King for t in board if t.peek() is not None)
    st = board.status
    draw = st in [AbstractBoardStatus.DRAW, AbstractBoardStatus.ONGOING]
    if draw and ply < 200 and len(cases) < 3:
        fp = [pn(t.peek()) for t in board if t.peek()]
        cases.append((gid, ply, ko, fp, hist))
    if len(cases) >= 3:
        break

for i, (gid, ply, ko, fp, hist) in enumerate(cases):
    print(f'===== Case {i+1} (game #{gid}, {ply} plies, kings_only={ko}) =====')
    print(f'Final pieces on board: {fp}')
    print()
    if len(hist) <= 24:
        for p, c, mv, n in hist:
            print(f'  {p:3d}. [{c}] {mv:42s} ({n} pcs)')
    else:
        for p, c, mv, n in hist[:10]:
            print(f'  {p:3d}. [{c}] {mv:42s} ({n} pcs)')
        print(f'  ... ({len(hist)-20} moves skipped) ...')
        for p, c, mv, n in hist[-10:]:
            print(f'  {p:3d}. [{c}] {mv:42s} ({n} pcs)')
    print()
