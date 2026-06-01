"""Plotting utilities for comparing DQN, Double DQN, and MCTS results."""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _smooth(data, window=500):
    """Simple moving-average smoother."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_metrics(results_dir, tag, seeds):
    """Load metrics.json for each seed. Returns list of dicts."""
    all_m = []
    for s in seeds:
        path = os.path.join(results_dir, f"{tag}_seed{s}", "metrics.json")
        with open(path) as f:
            all_m.append(json.load(f))
    return all_m


# ---------------------------------------------------------------------------
# 1. Training win-rate curves  (DQN vs DDQN, with seed variance)
# ---------------------------------------------------------------------------

def plot_training_winrate(results_dir, seeds, window=500):
    fig, ax = plt.subplots(figsize=(10, 5))

    for tag, color, label in [("DQN", "tab:blue", "DQN"),
                               ("DDQN", "tab:orange", "Double DQN")]:
        all_curves = []
        for m in load_metrics(results_dir, tag, seeds):
            rewards = np.array(m['episode_rewards'])
            wr = (rewards > 0).astype(float)
            smoothed = _smooth(wr, window)
            all_curves.append(smoothed)

        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(min_len) + window

        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Win Rate (rolling {window})")
    ax.set_title("Training Win Rate: DQN vs Double DQN (self-play)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_winrate.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Training loss curves
# ---------------------------------------------------------------------------

def plot_training_loss(results_dir, seeds, window=1000):
    fig, ax = plt.subplots(figsize=(10, 5))

    for tag, color, label in [("DQN", "tab:blue", "DQN"),
                               ("DDQN", "tab:orange", "Double DQN")]:
        all_curves = []
        for m in load_metrics(results_dir, tag, seeds):
            losses = np.array(m['losses'])
            smoothed = _smooth(losses, window)
            all_curves.append(smoothed)

        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(min_len)

        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Optimisation Step")
    ax.set_ylabel(f"Loss (rolling {window})")
    ax.set_title("Training Loss: DQN vs Double DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_loss.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Evaluation bar chart (DQN vs DDQN vs MCTS against each opponent)
# ---------------------------------------------------------------------------

def plot_eval_bars(results_dir, seeds, mcts_results=None):
    """
    mcts_results : dict with keys 'vs_random' and 'vs_heuristic',
                   each a tuple (win, loss, draw) rate.
    """
    opponents = ['Random', 'Heuristic']
    methods = ['DQN', 'Double DQN', 'MCTS']
    colours = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (opp_name, key) in enumerate(
            zip(opponents, ['eval_vs_random', 'eval_vs_heuristic'])):
        ax = axes[ax_idx]
        x = np.arange(3)  # W, L, D
        width = 0.22

        for i, (tag, json_tag) in enumerate(
                [("DQN", "DQN"), ("DDQN", "Double DQN")]):
            # average final eval across seeds
            wins, losses, draws = [], [], []
            for m in load_metrics(results_dir, tag, seeds):
                if m[key]:
                    last = m[key][-1]  # [episode, w, l, d]
                    wins.append(last[1])
                    losses.append(last[2])
                    draws.append(last[3])
            if wins:
                vals = [np.mean(wins), np.mean(losses), np.mean(draws)]
                errs = [np.std(wins), np.std(losses), np.std(draws)]
            else:
                vals = [0, 0, 0]
                errs = [0, 0, 0]
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=json_tag, color=colours[i], capsize=3)

        # MCTS bar
        mcts_key = 'vs_random' if opp_name == 'Random' else 'vs_heuristic'
        if mcts_results and mcts_key in mcts_results:
            mr = mcts_results[mcts_key]
            ax.bar(x + 2 * width, list(mr), width,
                   label='MCTS', color=colours[2])

        ax.set_xticks(x + width)
        ax.set_xticklabels(['Win', 'Loss', 'Draw'])
        ax.set_title(f'vs {opp_name} Opponent')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle("Evaluation Results: DQN vs Double DQN vs MCTS", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "eval_bars.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Epsilon decay curve
# ---------------------------------------------------------------------------

def plot_epsilon(results_dir, seeds):
    fig, ax = plt.subplots(figsize=(8, 4))
    m = load_metrics(results_dir, "DQN", seeds)[0]
    ax.plot(m['epsilons'])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon Decay Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "epsilon.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Evaluation win-rate over training (line plot)
# ---------------------------------------------------------------------------

def plot_eval_over_training(results_dir, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax_idx, (opp_name, key) in enumerate(
            zip(['Random', 'Heuristic'],
                ['eval_vs_random', 'eval_vs_heuristic'])):
        ax = axes[ax_idx]
        for tag, color, label in [("DQN", "tab:blue", "DQN"),
                                   ("DDQN", "tab:orange", "Double DQN")]:
            all_eps, all_wr = [], []
            for m in load_metrics(results_dir, tag, seeds):
                if m[key]:
                    eps = [e[0] for e in m[key]]
                    wr  = [e[1] for e in m[key]]
                    all_eps.append(eps)
                    all_wr.append(wr)

            if all_wr:
                min_len = min(len(w) for w in all_wr)
                arr = np.array([w[:min_len] for w in all_wr])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                x = all_eps[0][:min_len]
                ax.plot(x, mean, color=color, label=label, marker='o', ms=3)
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=0.2)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Evaluation vs {opp_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Win Rate During Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "eval_over_training.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(results_dir, seeds, mcts_results=None):
    """Convenience: generate every plot."""
    print("Generating plots …")
    plot_training_winrate(results_dir, seeds)
    plot_training_loss(results_dir, seeds)
    plot_eval_bars(results_dir, seeds, mcts_results)
    plot_epsilon(results_dir, seeds)
    plot_eval_over_training(results_dir, seeds)
    print(f"Plots saved to {results_dir}/")
