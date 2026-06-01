"""
Comprehensive quantitative analysis for the Mini-Chess RL project.

Produces all metrics needed for the paper:
  1. Training win-rate curves (with variance across seeds)
  2. Q-value overestimation comparison (DQN vs DDQN)
  3. Training loss convergence
  4. Evaluation win rates (bar chart + table)
  5. Average game length over training
  6. Summary statistics table (LaTeX-ready)

Usage:
    py -3.11 -m src.analyze --results-dir results
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'minichess-main'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(data, window):
    if len(data) < window:
        window = max(1, len(data) // 5)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def load_all(results_dir, seeds):
    """Load metrics for DQN and DDQN across all seeds."""
    out = {}
    for tag in ['DQN', 'DDQN']:
        out[tag] = []
        for s in seeds:
            path = os.path.join(results_dir, f"{tag}_seed{s}", "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    out[tag].append(json.load(f))
    return out


# -----------------------------------------------------------------------
# 1. Training Win-Rate  (rolling window, with seed variance)
# -----------------------------------------------------------------------
def plot_winrate(data, results_dir, window=500):
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
        curves = []
        for m in data[tag]:
            wr = np.array([1.0 if r > 0 else 0.0 for r in m['episode_rewards']])
            curves.append(smooth(wr, window))
        minl = min(len(c) for c in curves)
        arr = np.array([c[:minl] for c in curves])
        x = np.arange(minl) + window
        ax.plot(x, arr.mean(0), color=color, label=tag)
        ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                        color=color, alpha=0.2)
    ax.set_xlabel('Episode'); ax.set_ylabel(f'Win Rate (rolling {window})')
    ax.set_title('Training Win Rate'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_winrate.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_winrate.png')


# -----------------------------------------------------------------------
# 2. Q-Value Overestimation  (THE key DQN vs DDQN metric)
# -----------------------------------------------------------------------
def plot_qvalues(data, results_dir, window=2000):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Mean Q-value over training
    ax = axes[0]
    for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
        curves = []
        for m in data[tag]:
            if 'mean_q_values' in m and m['mean_q_values']:
                curves.append(smooth(np.array(m['mean_q_values']), window))
        if curves:
            minl = min(len(c) for c in curves)
            arr = np.array([c[:minl] for c in curves])
            x = np.arange(minl)
            ax.plot(x, arr.mean(0), color=color, label=tag)
            ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                            color=color, alpha=0.2)
    ax.set_xlabel('Optimisation Step'); ax.set_ylabel('Mean Q(s,a)')
    ax.set_title('(a) Average Q-Value During Training')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Overestimation = mean_q - mean_target
    ax = axes[1]
    for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
        curves = []
        for m in data[tag]:
            if 'mean_q_values' in m and 'mean_targets' in m \
                    and m['mean_q_values'] and m['mean_targets']:
                q = np.array(m['mean_q_values'])
                t = np.array(m['mean_targets'])
                minl2 = min(len(q), len(t))
                curves.append(smooth(q[:minl2] - t[:minl2], window))
        if curves:
            minl = min(len(c) for c in curves)
            arr = np.array([c[:minl] for c in curves])
            x = np.arange(minl)
            ax.plot(x, arr.mean(0), color=color, label=tag)
            ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                            color=color, alpha=0.2)
    ax.axhline(0, color='black', ls='--', lw=0.8)
    ax.set_xlabel('Optimisation Step'); ax.set_ylabel('Q(s,a) - Target')
    ax.set_title('(b) Overestimation Bias (Q - Target)')
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Q-Value Analysis: DQN vs Double DQN', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_qvalues.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_qvalues.png')


# -----------------------------------------------------------------------
# 3. Training Loss
# -----------------------------------------------------------------------
def plot_loss(data, results_dir, window=2000):
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
        curves = []
        for m in data[tag]:
            curves.append(smooth(np.array(m['losses']), window))
        minl = min(len(c) for c in curves)
        arr = np.array([c[:minl] for c in curves])
        x = np.arange(minl)
        ax.plot(x, arr.mean(0), color=color, label=tag)
        ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                        color=color, alpha=0.2)
    ax.set_xlabel('Optimisation Step'); ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_loss.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_loss.png')


# -----------------------------------------------------------------------
# 4. Average Game Length
# -----------------------------------------------------------------------
def plot_game_length(data, results_dir, window=500):
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
        curves = []
        for m in data[tag]:
            if 'game_lengths' in m and m['game_lengths']:
                curves.append(smooth(np.array(m['game_lengths'], dtype=float), window))
        if curves:
            minl = min(len(c) for c in curves)
            arr = np.array([c[:minl] for c in curves])
            x = np.arange(minl) + window
            ax.plot(x, arr.mean(0), color=color, label=tag)
            ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                            color=color, alpha=0.2)
    ax.set_xlabel('Episode'); ax.set_ylabel('Moves (agent)')
    ax.set_title('Average Game Length'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_gamelength.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_gamelength.png')


# -----------------------------------------------------------------------
# 5. Evaluation Bar Chart (with MCTS if available)
# -----------------------------------------------------------------------
def plot_eval_bars(data, results_dir, mcts_results=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax_i, (opp, key) in enumerate(
            zip(['Random', 'Heuristic'], ['eval_vs_random', 'eval_vs_heuristic'])):
        ax = axes[ax_i]
        x = np.arange(3)
        width = 0.22
        for i, (tag, label, clr) in enumerate([
                ('DQN', 'DQN', 'tab:blue'), ('DDQN', 'Double DQN', 'tab:orange')]):
            wins, losses, draws = [], [], []
            for m in data[tag]:
                if m[key]:
                    last = m[key][-1]
                    wins.append(last[1]); losses.append(last[2]); draws.append(last[3])
            if wins:
                vals = [np.mean(wins), np.mean(losses), np.mean(draws)]
                errs = [np.std(wins), np.std(losses), np.std(draws)]
                ax.bar(x + i*width, vals, width, yerr=errs, label=label,
                       color=clr, capsize=3)
        mkey = 'vs_random' if opp == 'Random' else 'vs_heuristic'
        if mcts_results and mkey in mcts_results:
            mr = mcts_results[mkey]
            ax.bar(x + 2*width, mr, width, label='MCTS', color='tab:green')
        ax.set_xticks(x + width); ax.set_xticklabels(['Win', 'Loss', 'Draw'])
        ax.set_title(f'vs {opp}'); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Evaluation Win/Loss/Draw Rates', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_eval_bars.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_eval_bars.png')


# -----------------------------------------------------------------------
# 6. Evaluation Win Rate Over Training
# -----------------------------------------------------------------------
def plot_eval_curves(data, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax_i, (opp, key) in enumerate(
            zip(['Random', 'Heuristic'], ['eval_vs_random', 'eval_vs_heuristic'])):
        ax = axes[ax_i]
        for tag, color in [('DQN', 'tab:blue'), ('DDQN', 'tab:orange')]:
            all_x, all_wr = [], []
            for m in data[tag]:
                if m[key]:
                    all_x.append([e[0] for e in m[key]])
                    all_wr.append([e[1] for e in m[key]])
            if all_wr:
                minl = min(len(w) for w in all_wr)
                arr = np.array([w[:minl] for w in all_wr])
                x = all_x[0][:minl]
                ax.plot(x, arr.mean(0), color=color, label=tag, marker='o', ms=3)
                ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                                color=color, alpha=0.2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Win Rate')
        ax.set_title(f'vs {opp}'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Evaluation Win Rate During Training', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'fig_eval_curves.png'), dpi=150)
    plt.close(fig)
    print('  -> fig_eval_curves.png')


# -----------------------------------------------------------------------
# 7. Summary Statistics (printed + saved as txt)
# -----------------------------------------------------------------------
def print_summary(data, results_dir, mcts_results=None):
    lines = []
    lines.append('=' * 70)
    lines.append('QUANTITATIVE SUMMARY')
    lines.append('=' * 70)

    for tag in ['DQN', 'DDQN']:
        lines.append(f'\n--- {tag} ({len(data[tag])} seeds) ---')
        all_rewards = [m['episode_rewards'] for m in data[tag]]

        # overall win/loss/draw
        for i, m in enumerate(data[tag]):
            r = m['episode_rewards']
            n = len(r)
            w = sum(1 for x in r if x > 0)
            l = sum(1 for x in r if x < 0)
            d = n - w - l
            lines.append(f'  Seed {i}: {n} eps  W={w/n:.1%}  L={l/n:.1%}  D={d/n:.1%}')

        # last N win rate (convergence indicator)
        last_n = 2000
        for label, n_last in [(f'Last {last_n}', last_n), ('Last 500', 500)]:
            wrs = []
            for m in data[tag]:
                r = m['episode_rewards'][-n_last:]
                wrs.append(sum(1 for x in r if x > 0) / len(r))
            lines.append(f'  {label} win rate: {np.mean(wrs):.1%} +/- {np.std(wrs):.1%}')

        # average Q-value (overestimation)
        for m in data[tag]:
            if 'mean_q_values' in m and m['mean_q_values']:
                q = m['mean_q_values']
                lines.append(f'  Avg Q-value (last 20%): {np.mean(q[-len(q)//5:]):.4f}')

        # average game length
        for m in data[tag]:
            if 'game_lengths' in m and m['game_lengths']:
                gl = m['game_lengths']
                lines.append(f'  Avg game length (last 20%): {np.mean(gl[-len(gl)//5:]):.1f} moves')

        # final evaluation
        for key, opp in [('eval_vs_random', 'Random'), ('eval_vs_heuristic', 'Heuristic')]:
            ws, ls, ds = [], [], []
            for m in data[tag]:
                if m[key]:
                    last = m[key][-1]
                    ws.append(last[1]); ls.append(last[2]); ds.append(last[3])
            if ws:
                lines.append(f'  vs {opp}: W={np.mean(ws):.0%}+/-{np.std(ws):.0%}  '
                             f'L={np.mean(ls):.0%}+/-{np.std(ls):.0%}  '
                             f'D={np.mean(ds):.0%}+/-{np.std(ds):.0%}')

        # final loss
        for m in data[tag]:
            losses = m['losses']
            if losses:
                lines.append(f'  Final loss (last 20%): {np.mean(losses[-len(losses)//5:]):.6f}')

    # MCTS
    if mcts_results:
        lines.append(f'\n--- MCTS ---')
        for key, opp in [('vs_random', 'Random'), ('vs_heuristic', 'Heuristic')]:
            if key in mcts_results:
                w, l, d = mcts_results[key]
                lines.append(f'  vs {opp}: W={w:.0%}  L={l:.0%}  D={d:.0%}')

    lines.append('\n' + '=' * 70)

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(report)
    print(f'\n  -> summary.txt')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = parser.parse_args()

    print(f'Analysing results in: {args.results_dir}')
    data = load_all(args.results_dir, args.seeds)

    if not data['DQN'] and not data['DDQN']:
        print('No data found. Run training first.')
        return

    # load MCTS results if available
    mcts_path = os.path.join(args.results_dir, 'mcts_results.json')
    mcts_results = None
    if os.path.exists(mcts_path):
        with open(mcts_path) as f:
            mcts_results = json.load(f)

    print('\nGenerating analysis plots...')
    plot_winrate(data, args.results_dir)
    plot_qvalues(data, args.results_dir)
    plot_loss(data, args.results_dir)
    plot_game_length(data, args.results_dir)
    plot_eval_bars(data, args.results_dir, mcts_results)
    plot_eval_curves(data, args.results_dir)
    print_summary(data, args.results_dir, mcts_results)


if __name__ == '__main__':
    main()
