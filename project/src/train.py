"""Self-play training loop for DQN / Double DQN on 5x5 Mini-Chess."""

import os, time, random, json
import numpy as np
import torch
from tqdm import trange

from minichess.games.abstract.piece import PieceColor

from src.config import Config
from src.env import MiniChessEnv
from src.dqn_agent import DQNAgent
from src.opponents import RandomOpponent, DQNOpponent, HeuristicOpponent
from src.evaluate import evaluate_agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dqn(config: Config, double=False, seed=42, verbose=True):
    """Train one DQN (or DDQN) agent. Returns metrics dict."""
    set_seed(seed)
    tag = "DDQN" if double else "DQN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[{tag}] seed={seed}  device={device}")

    agent = DQNAgent(config, double=double, device=device)

    # opponents
    random_opp = RandomOpponent()
    heuristic_opp = HeuristicOpponent()
    frozen_net = agent.make_frozen_copy()
    frozen_opp = DQNOpponent(frozen_net, device, epsilon=0.05)

    # metrics accumulators
    metrics = {
        'episode_rewards': [],    # per-episode terminal reward
        'losses': [],             # per-update loss
        'mean_q_values': [],      # avg Q(s,a) per update — overestimation metric
        'max_q_values': [],       # max Q per update
        'mean_targets': [],       # avg Bellman target per update
        'game_lengths': [],       # half-moves per game
        'eval_vs_random': [],     # (episode, win, loss, draw)
        'eval_vs_heuristic': [],
        'epsilons': [],
    }

    env = MiniChessEnv(opponent=random_opp, max_plies=config.max_plies)

    pbar = trange(1, config.num_episodes + 1, desc=tag, disable=not verbose)
    for episode in pbar:

        # --- swap opponent after warmup ---
        if episode == config.warmup_episodes + 1:
            env.opponent = frozen_opp

        # --- periodically refresh frozen opponent ---
        if (episode > config.warmup_episodes
                and episode % config.opponent_update_freq == 0):
            frozen_net = agent.make_frozen_copy()
            frozen_opp = DQNOpponent(frozen_net, device, epsilon=0.05)
            env.opponent = frozen_opp

        # --- play one game ---
        agent_color = PieceColor.WHITE if episode % 2 == 0 else PieceColor.BLACK
        state, mask, done = env.reset(agent_color)
        episode_reward = 0.0

        step_count = 0
        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done, next_mask = env.step(action)
            agent.store_transition(state, action, reward, next_state,
                                   float(done), next_mask)
            state, mask = next_state, next_mask
            episode_reward += reward
            step_count += 1

            info = agent.update()
            if info is not None:
                metrics['losses'].append(info['loss'])
                metrics['mean_q_values'].append(info['mean_q'])
                metrics['max_q_values'].append(info['max_q'])
                metrics['mean_targets'].append(info['mean_target'])

        metrics['episode_rewards'].append(episode_reward)
        metrics['game_lengths'].append(step_count)
        agent.decay_epsilon(episode)
        metrics['epsilons'].append(agent.epsilon)

        # --- progress bar ---
        if episode % 200 == 0:
            recent = metrics['episode_rewards'][-200:]
            wr = sum(1 for r in recent if r > 0) / len(recent)
            pbar.set_postfix(eps=f"{agent.epsilon:.3f}", wr200=f"{wr:.2f}")

        # --- periodic evaluation ---
        if episode % config.eval_freq == 0:
            wr_rand = evaluate_agent(agent, random_opp, config,
                                     num_games=config.eval_games)
            wr_heur = evaluate_agent(agent, heuristic_opp, config,
                                     num_games=config.eval_games)
            metrics['eval_vs_random'].append((episode, *wr_rand))
            metrics['eval_vs_heuristic'].append((episode, *wr_heur))
            if verbose:
                print(f"  [{tag}] ep {episode}  "
                      f"vs random W/L/D={wr_rand}  "
                      f"vs heuristic W/L/D={wr_heur}")

    # --- save checkpoint ---
    save_dir = os.path.join(config.results_dir, f"{tag}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, "checkpoint.pt"))

    # serialise metrics (convert tuples to lists for JSON)
    metrics_serialisable = {}
    for k, v in metrics.items():
        if v and isinstance(v[0], tuple):
            metrics_serialisable[k] = [list(t) for t in v]
        else:
            metrics_serialisable[k] = v
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_serialisable, f)

    if verbose:
        print(f"[{tag}] seed={seed} done – saved to {save_dir}")
    return metrics
