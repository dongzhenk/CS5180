"""
Generate a 4x3 academic poster (34" x 33") and split into 12 letter-size pages.

Usage:
    py -3.11 poster/make_poster.py

Output:
    poster/poster_full.png       - full poster image
    poster/pages/page_01.png ... - 12 individual pages for printing
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# ── Dimensions ────────────────────────────────────────────────────────
COLS, ROWS = 4, 3
PAGE_W, PAGE_H = 8.5, 11          # inches (letter)
POSTER_W = COLS * PAGE_W          # 34"
POSTER_H = ROWS * PAGE_H          # 33"
DPI = 150

# ── Colours ───────────────────────────────────────────────────────────
C_TITLE_BG  = '#1a365d'
C_TITLE_FG  = '#ffffff'
C_SECTION   = '#2c5282'
C_BODY      = '#1a202c'
C_BG        = '#ffffff'
C_ACCENT    = '#e2e8f0'
C_DQN       = '#3182ce'
C_DDQN      = '#dd6b20'
C_MCTS      = '#38a169'

# ── Result figure paths ───────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# ── Helper functions ──────────────────────────────────────────────────

def add_bg(fig, x, y, w, h, color, alpha=1.0):
    fig.patches.append(mpatches.FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure, facecolor=color,
        edgecolor='none', alpha=alpha, zorder=0,
        boxstyle="square,pad=0"))

def section_header(fig, x, y, text, fontsize=32):
    fig.text(x, y, text, fontsize=fontsize, fontweight='bold',
             color=C_SECTION, va='top', ha='left')

def body_text(fig, x, y, text, fontsize=18, **kw):
    fig.text(x, y, text, fontsize=fontsize, color=C_BODY,
             va='top', ha='left', linespacing=1.5, **kw)

def embed_image(fig, img_path, extent):
    """extent = [left, bottom, width, height] in figure coords."""
    ax = fig.add_axes(extent)
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')
    return ax

def mono_text(fig, x, y, text, fontsize=14):
    fig.text(x, y, text, fontsize=fontsize, color=C_BODY,
             va='top', ha='left', family='monospace', linespacing=1.6,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=C_ACCENT,
                       edgecolor='#cbd5e0', alpha=0.7))


# ── Build poster ──────────────────────────────────────────────────────

def _draw_chessboard(save_path):
    """Draw a 5x5 Gardner Mini-Chess initial position and save as PNG."""
    fig_cb, ax = plt.subplots(figsize=(5, 5))

    # board colors
    light, dark = '#f0d9b5', '#b58863'
    for r in range(5):
        for c in range(5):
            color = light if (r + c) % 2 == 0 else dark
            ax.add_patch(mpatches.Rectangle((c, 4 - r), 1, 1, facecolor=color, edgecolor='none'))

    # pieces: row 0 = black back rank, row 1 = black pawns, row 3 = white pawns, row 4 = white back rank
    pieces = {
        (0, 0): '\u265C', (0, 1): '\u265E', (0, 2): '\u265D', (0, 3): '\u265B', (0, 4): '\u265A',
        (1, 0): '\u265F', (1, 1): '\u265F', (1, 2): '\u265F', (1, 3): '\u265F', (1, 4): '\u265F',
        (3, 0): '\u2659', (3, 1): '\u2659', (3, 2): '\u2659', (3, 3): '\u2659', (3, 4): '\u2659',
        (4, 0): '\u2656', (4, 1): '\u2658', (4, 2): '\u2657', (4, 3): '\u2655', (4, 4): '\u2654',
    }
    for (r, c), p in pieces.items():
        ax.text(c + 0.5, 4 - r + 0.5, p, fontsize=36, ha='center', va='center')

    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#333'); spine.set_linewidth(2)
    fig_cb.tight_layout(pad=0.2)
    fig_cb.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_cb)


def make_poster():
    # pre-generate chessboard image
    board_img_path = os.path.join(os.path.dirname(__file__), '_chessboard.png')
    _draw_chessboard(board_img_path)

    fig = plt.figure(figsize=(POSTER_W, POSTER_H), facecolor=C_BG)

    # ============================================================
    # ROW 0 — Title banner  (top 1/3 height → y: 0.667 – 1.0)
    # ============================================================
    add_bg(fig, 0, 0.82, 1, 0.18, C_TITLE_BG)

    fig.text(0.50, 0.97,
             'Deep Q-Learning and Planning in Self-Play Mini-Chess',
             fontsize=52, fontweight='bold', color=C_TITLE_FG,
             ha='center', va='top')
    fig.text(0.50, 0.92,
             'A Comparison of DQN, Double DQN, and MCTS',
             fontsize=38, color='#bee3f8', ha='center', va='top')
    fig.text(0.50, 0.875,
             'Zhenkang Dong  |  CS 5180 Reinforcement Learning  |  Northeastern University',
             fontsize=24, color='#a0aec0', ha='center', va='top')

    # ============================================================
    # ROW 1 — Methods  (middle 1/3 → y: 0.333 – 0.667)
    # ============================================================
    col_w = 1.0 / COLS
    row_mid_top = 0.80
    row_mid_bot = 0.40

    # light background for method row
    add_bg(fig, 0, row_mid_bot, 1, row_mid_top - row_mid_bot, '#f7fafc')

    # ── Section 1: Problem & MDP ──
    x0 = 0.02
    section_header(fig, x0, 0.78, '1. Problem & MDP Formulation')

    # chessboard image (right side of Section 1)
    if os.path.exists(board_img_path):
        embed_image(fig, board_img_path, [0.13, 0.56, 0.10, 0.12])

    body_text(fig, x0, 0.75,
        'Gardner Mini-Chess\n'
        '5×5 board, 10 pieces per side\n'
        '(K, Q, R, B, N, 5P)\n\n'
        'State: 12 × 5 × 5 binary planes\n'
        '  • Ch 0-5: own {P,N,B,R,Q,K}\n'
        '  • Ch 6-11: opponent pieces\n'
        '  • Canonical view (mover\'s POV)',
        fontsize=17)

    body_text(fig, x0, 0.48,
        'Actions: 625 (from_sq × to_sq)\n'
        '  • Legal-action masking\n\n'
        'Rewards: Terminal only\n'
        '  • +1 win, -1 loss, 0 draw\n'
        '  • Move limit: 200 half-moves',
        fontsize=17)

    # ── Section 2: DQN & Double DQN ──
    x1 = col_w + 0.02
    section_header(fig, x1, 0.78, '2. DQN & Double DQN')

    body_text(fig, x1, 0.75,
        'CNN Architecture:\n'
        '  Conv2d(12→64, 3×3) → BN → ReLU\n'
        '  Conv2d(64→128, 3×3) → BN → ReLU\n'
        '  Conv2d(128→128, 3×3) → BN → ReLU\n'
        '  FC(3200→512) → ReLU → FC(512→625)',
        fontsize=17)

    mono_text(fig, x1, 0.65,
        'DQN Target:\n'
        '  y = r + γ · max_a\' Q(s\', a\'; θ⁻)\n'
        '\n'
        'Double DQN Target:\n'
        '  a* = argmax_a\' Q(s\', a\'; θ)   ← online\n'
        '  y  = r + γ · Q(s\', a*; θ⁻)     ← target',
        fontsize=16)

    body_text(fig, x1, 0.52,
        'Key Hyperparameters:\n'
        '  • Learning rate: 1e-4 (Adam)\n'
        '  • Replay buffer: 100K transitions\n'
        '  • Target update: every 1000 steps\n'
        '  • ε-greedy: 1.0 → 0.05 (linear, 30K eps)\n'
        '  • Discount γ = 0.99\n'
        '  • Loss: Huber (Smooth L1)',
        fontsize=17)

    # ── Section 3: MCTS ──
    x2 = 2 * col_w + 0.02
    section_header(fig, x2, 0.78, '3. MCTS (Planning)')

    body_text(fig, x2, 0.75,
        'Monte Carlo Tree Search with random rollout\n'
        'policy — no training required.\n\n'
        '200 simulations per move decision.',
        fontsize=18)

    mono_text(fig, x2, 0.65,
        'function MCTS(state, N_sim):\n'
        '  root = Node(state)\n'
        '  for i = 1 ... N_sim:\n'
        '    node = SELECT(root)   // UCB1\n'
        '    node = EXPAND(node)\n'
        '    v = ROLLOUT(node)     // random\n'
        '    BACKPROP(node, v)\n'
        '  return most_visited(root)',
        fontsize=15)

    body_text(fig, x2, 0.51,
        'UCB1 Selection:\n'
        '  score = Q/N + c · √(ln N_parent / N)\n'
        '  c = √2 ≈ 1.414\n\n'
        'Rollout: uniform random legal moves\n'
        'until terminal state.\n\n'
        'Value stored from White\'s perspective\n'
        'at every node (no negation needed).',
        fontsize=17)

    # ── Section 4: Self-Play Training ──
    x3 = 3 * col_w + 0.02
    section_header(fig, x3, 0.78, '4. Self-Play Training')

    body_text(fig, x3, 0.75,
        'Training Pipeline:\n\n'
        '① Warmup phase (ep 1-1000):\n'
        '   Agent vs Random opponent\n\n'
        '② Self-play phase (ep 1001+):\n'
        '   Agent vs Frozen copy of itself\n\n'
        '③ Frozen opponent refreshed\n'
        '   every 500 episodes\n\n'
        '④ Colors alternate each episode\n'
        '   (mitigate first-move bias)\n\n'
        '⑤ Canonical encoding:\n'
        '   Board always from mover\'s POV\n'
        '   → same network plays both sides\n\n'
        'Training budget:\n'
        '  20,000 episodes × 2 seeds\n'
        '  (RTX 4060 Laptop GPU)',
        fontsize=17)

    # ============================================================
    # ROW 2 — Results  (bottom 1/3 → y: 0 – 0.333)
    # ============================================================
    row_bot_top = 0.39

    # ── Section 5: Training Win Rate ──
    section_header(fig, 0.02, 0.38, '5. Training Win Rate')
    fig_path = os.path.join(RESULTS, 'fig_winrate.png')
    if os.path.exists(fig_path):
        embed_image(fig, fig_path, [0.01, 0.12, 0.24, 0.25])

    # ── Section 6: Q-Value Overestimation ──
    section_header(fig, col_w + 0.02, 0.38, '6. Q-Value Analysis')
    fig_path = os.path.join(RESULTS, 'fig_qvalues.png')
    if os.path.exists(fig_path):
        embed_image(fig, fig_path, [0.255, 0.12, 0.24, 0.25])

    # ── Section 7: Evaluation Results ──
    section_header(fig, 2*col_w + 0.02, 0.38, '7. Evaluation Results')
    fig_path = os.path.join(RESULTS, 'fig_eval_bars.png')
    if os.path.exists(fig_path):
        embed_image(fig, fig_path, [0.505, 0.12, 0.24, 0.25])

    # Result summary table
    body_text(fig, 2*col_w + 0.02, 0.11,
        '           vs Random (W/L/D)  vs Heuristic (W/L/D)\n'
        'DQN       13% / 7%  / 80%     0% / 27% / 73%\n'
        'DDQN     18% / 12% / 70%     5% / 27% / 68%\n'
        'MCTS     93% /  0%  /  7%    63% /  3%  / 33%',
        fontsize=14, family='monospace')

    # ── Section 8: Conclusions & Future Work ──
    section_header(fig, 3*col_w + 0.02, 0.38, '8. Conclusions')
    body_text(fig, 3*col_w + 0.02, 0.35,
        'Key Findings:\n\n'
        '• MCTS (93% win) greatly outperforms\n'
        '  DQN (13%) and DDQN (18%) vs Random\n\n'
        '• DDQN shows lower Q-value\n'
        '  overestimation than DQN\n'
        '  (avg Q: 1.84 vs 1.89)\n\n'
        '• DDQN has lower variance across\n'
        '  seeds (±0.8% vs ±1.4%)\n\n'
        '• Both DQN/DDQN struggle with\n'
        '  sparse terminal rewards\n'
        '  (most games end in draw)\n\n'
        'Future Work:\n\n'
        '• Reward shaping (captures, checks)\n'
        '• AlphaZero-style MCTS + NN\n'
        '• Longer training (50K+ episodes)\n'
        '• Policy gradient methods (PPO)',
        fontsize=17)

    # ── Grid lines (faint, to help alignment when taping) ──
    for i in range(1, COLS):
        x = i / COLS
        fig.add_artist(plt.Line2D([x, x], [0, 0.82], transform=fig.transFigure,
                 color='#e2e8f0', linewidth=0.5, zorder=1))
    for j in range(1, ROWS):
        y = j / ROWS
        fig.add_artist(plt.Line2D([0, 1], [y, y], transform=fig.transFigure,
                 color='#e2e8f0', linewidth=0.5, zorder=1))

    return fig


# ── Save and split ────────────────────────────────────────────────────

def save_and_split(fig):
    out_dir = os.path.dirname(__file__)
    pages_dir = os.path.join(out_dir, 'pages')
    os.makedirs(pages_dir, exist_ok=True)

    # save full poster
    full_path = os.path.join(out_dir, 'poster_full.png')
    fig.savefig(full_path, dpi=DPI, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved full poster: {full_path}')

    # split into 12 tiles
    img = Image.open(full_path)
    w, h = img.size
    tile_w = w // COLS
    tile_h = h // ROWS

    for row in range(ROWS):
        for col in range(COLS):
            idx = row * COLS + col + 1
            box = (col * tile_w, row * tile_h,
                   (col + 1) * tile_w, (row + 1) * tile_h)
            tile = img.crop(box)

            # resize to exact letter size at DPI
            letter_px = (int(PAGE_W * DPI), int(PAGE_H * DPI))
            tile = tile.resize(letter_px, Image.LANCZOS)

            path = os.path.join(pages_dir, f'page_{idx:02d}.png')
            tile.save(path, dpi=(DPI, DPI))

    print(f'Saved 12 pages to: {pages_dir}/')
    print(f'\nPrint all pages in {pages_dir}/ and arrange in 4×3 grid:')
    print('  [01] [02] [03] [04]')
    print('  [05] [06] [07] [08]')
    print('  [09] [10] [11] [12]')


if __name__ == '__main__':
    fig = make_poster()
    save_and_split(fig)
    print('\nDone!')
