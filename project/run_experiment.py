#!/usr/bin/env python3
"""
Main entry point: train DQN & Double DQN, evaluate MCTS, generate plots.

Usage
-----
    python run_experiment.py                # full experiment
    python run_experiment.py --episodes 5000 --eval-games 20  # quick test
    python run_experiment.py --mcts-only    # skip training, only eval MCTS
"""

import sys, os, argparse, json, time

# make project root importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minichess-main'))

from src.config import Config
from src.train import train_dqn
from src.evaluate import evaluate_mcts
from src.opponents import RandomOpponent, HeuristicOpponent
from src.mcts_agent import MCTSAgent
from src.plot_results import generate_all_plots


def parse_args():
    p = argparse.ArgumentParser(description="Mini-Chess RL Experiment")
    p.add_argument('--episodes', type=int, default=None,
                   help='override num_episodes')
    p.add_argument('--eval-games', type=int, default=None,
                   help='override eval_games')
    p.add_argument('--mcts-sims', type=int, default=None,
                   help='override MCTS simulations per move')
    p.add_argument('--mcts-games', type=int, default=50,
                   help='MCTS evaluation games per opponent')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='override seeds (e.g. --seeds 42 123 456)')
    p.add_argument('--results-dir', type=str, default='results',
                   help='directory for outputs')
    p.add_argument('--mcts-only', action='store_true',
                   help='skip DQN training, only run MCTS evaluation')
    p.add_argument('--skip-mcts', action='store_true',
                   help='skip MCTS evaluation (faster)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # apply CLI overrides
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games
    if args.mcts_sims is not None:
        cfg.mcts_simulations = args.mcts_sims
    if args.seeds is not None:
        cfg.seeds = args.seeds
    cfg.results_dir = args.results_dir
    os.makedirs(cfg.results_dir, exist_ok=True)

    # save config
    with open(os.path.join(cfg.results_dir, "config.json"), 'w') as f:
        json.dump(vars(cfg), f, indent=2)

    # ==================================================================
    # Phase 1 – DQN & Double DQN training
    # ==================================================================
    if not args.mcts_only:
        for seed in cfg.seeds:
            print(f"\n{'='*60}")
            print(f"Training DQN  (seed {seed})")
            print(f"{'='*60}")
            t0 = time.time()
            train_dqn(cfg, double=False, seed=seed)
            print(f"  elapsed: {time.time()-t0:.1f}s")

            print(f"\n{'='*60}")
            print(f"Training Double DQN  (seed {seed})")
            print(f"{'='*60}")
            t0 = time.time()
            train_dqn(cfg, double=True, seed=seed)
            print(f"  elapsed: {time.time()-t0:.1f}s")

    # ==================================================================
    # Phase 2 – MCTS evaluation
    # ==================================================================
    mcts_results = {}
    if not args.skip_mcts:
        print(f"\n{'='*60}")
        print(f"Evaluating MCTS ({cfg.mcts_simulations} sims/move)")
        print(f"{'='*60}")
        mcts = MCTSAgent(num_simulations=cfg.mcts_simulations, c=cfg.mcts_c)
        random_opp = RandomOpponent()
        heuristic_opp = HeuristicOpponent()

        print("  MCTS vs Random …")
        t0 = time.time()
        wr = evaluate_mcts(mcts, random_opp, cfg,
                           num_games=args.mcts_games, verbose=True)
        mcts_results['vs_random'] = wr
        print(f"  MCTS vs Random: W/L/D = {wr}  ({time.time()-t0:.1f}s)")

        print("  MCTS vs Heuristic …")
        t0 = time.time()
        wr = evaluate_mcts(mcts, heuristic_opp, cfg,
                           num_games=args.mcts_games, verbose=True)
        mcts_results['vs_heuristic'] = wr
        print(f"  MCTS vs Heuristic: W/L/D = {wr}  ({time.time()-t0:.1f}s)")

        # save MCTS results
        with open(os.path.join(cfg.results_dir, "mcts_results.json"), 'w') as f:
            json.dump({k: list(v) for k, v in mcts_results.items()}, f, indent=2)

    # ==================================================================
    # Phase 3 – Plots
    # ==================================================================
    if not args.mcts_only or os.path.exists(
            os.path.join(cfg.results_dir, "DQN_seed42", "metrics.json")):
        generate_all_plots(cfg.results_dir, cfg.seeds,
                           mcts_results if mcts_results else None)

    print("\nDone.  All outputs in:", cfg.results_dir)


if __name__ == "__main__":
    main()
