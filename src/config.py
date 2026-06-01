"""Hyperparameters and configuration for the Mini-Chess RL project."""

from dataclasses import dataclass, field

@dataclass
class Config:
    # --- Training ---
    num_episodes: int = 50000
    max_plies: int = 200          # half-moves per game (100 full moves)
    batch_size: int = 256
    buffer_size: int = 100000
    lr: float = 1e-4
    gamma: float = 0.99

    # --- Epsilon-greedy ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 30000

    # --- Target network ---
    target_update_freq: int = 1000   # every N optimisation steps

    # --- Self-play ---
    opponent_update_freq: int = 500  # every N episodes, copy online → frozen
    warmup_episodes: int = 1000      # use random opponent for warmup

    # --- Evaluation ---
    eval_freq: int = 2000            # evaluate every N episodes
    eval_games: int = 50             # games per evaluation round

    # --- MCTS ---
    mcts_simulations: int = 200
    mcts_c: float = 1.414

    # --- Reproducibility ---
    seeds: list = field(default_factory=lambda: [42, 123, 456])

    # --- Paths ---
    results_dir: str = "results"
