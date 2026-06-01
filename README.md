# Deep Q-Learning and Planning in Self-Play Mini-Chess

**A Comparison of DQN, Double DQN, and MCTS**

CS 5180 Reinforcement Learning — Final Project — Zhenkang Dong, Northeastern University

---

## Overview

This project trains self-play agents in **5×5 Gardner Mini-Chess** and conducts a
controlled comparison between three methods:

- **DQN** — deep Q-learning baseline
- **Double DQN** — decoupled action selection / evaluation to reduce overestimation
- **MCTS** — plain Monte Carlo Tree Search with random rollouts (no training)

DQN and Double DQN share an identical CNN architecture, hyperparameters, and training
budget, isolating the effect of the target-estimation mechanism. Agents are trained via
**frozen-opponent self-play** and evaluated against fixed **random** and **heuristic**
opponents.

## Key Results

| Method | vs Random (W/L/D) | vs Heuristic (W/L/D) |
|--------|-------------------|----------------------|
| DQN | 13 / 7 / 80 | 0 / 27 / 73 |
| Double DQN | 18 / 12 / 70 | 5 / 27 / 68 |
| MCTS (200 sims) | **93 / 0 / 7** | **63 / 3 / 33** |

- MCTS dominates without any training.
- Double DQN shows **lower Q-value overestimation** (mean Q 1.84 vs 1.89) and
  **lower cross-seed variance** (±0.8% vs ±1.4%) than standard DQN, confirming theory.
- Both learned agents converge to draw-heavy play due to sparse terminal rewards.

## Repository Structure

```
src/                  Implementation
├── config.py         Hyperparameters
├── env.py            5×5 Mini-Chess environment wrapper (state/action/reward encoding)
├── model.py          CNN Q-Network (3 conv + 2 FC layers)
├── replay_buffer.py  Experience replay
├── dqn_agent.py      DQN / Double DQN agent
├── mcts_agent.py     MCTS (UCB1 selection + random rollout)
├── opponents.py      Random / Heuristic / frozen-DQN opponents
├── train.py          Self-play training loop
├── evaluate.py       Evaluation
└── analyze.py        Figure generation + summary statistics

results/              Trained checkpoints, raw metrics, figures, summary
paper/                Final paper (AAAI format) + LaTeX source
poster/               Poster (4×3 tiled), presentation script, Q&A prep
minichess-main/       Open-source 5×5 Gardner rule engine (dependency)
run_experiment.py     Main entry point (training + evaluation)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11 and PyTorch (CUDA recommended).

## Usage

```bash
# Full experiment (2 seeds × 20K episodes + MCTS eval)
python run_experiment.py

# Quick test
python run_experiment.py --episodes 5000 --eval-games 20 --skip-mcts

# Generate all analysis figures from saved results
python -m src.analyze --results-dir results --seeds 42 123
```

## Environment

- State: 12 × 5 × 5 binary planes (own / opponent × {P,N,B,R,Q,K}), canonical view
- Action: 625 discrete (from-square × to-square) with legal-action masking
- Reward: terminal only (+1 win / −1 loss / 0 draw), 200 half-move limit
