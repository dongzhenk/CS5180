# Poster Presentation Script (~5 minutes)

## Opening (30s)

> "Hi, I'm Zhenkang. My project compares three approaches for playing 5×5 Gardner Mini-Chess: DQN, Double DQN, and MCTS. The goal is to study how value-based reinforcement learning behaves in a two-player adversarial setting — specifically convergence stability and overestimation bias."

## 1. Problem — Why Mini-Chess? (40s)

*Point to Section 1 on the poster*

> "Gardner Mini-Chess is a 5×5 variant with 10 pieces per side. It's computationally manageable but strategically rich — it's actually been weakly solved and the game-theoretic value is a draw."

> "I formulated it as an MDP: the state is a 12-channel binary encoding — 6 channels for my pieces, 6 for the opponent's. The action space is 625 from-square-to-square combinations with legal-action masking. Rewards are terminal only — +1 for a win, -1 for a loss, 0 for a draw."

## 2. Algorithms (60s)

*Point to Sections 2 & 3*

> "For DQN, I use a 3-layer CNN feeding into fully connected layers that output Q-values for all 625 actions. I train with experience replay and a target network."

> "Double DQN uses the exact same architecture and hyperparameters — the only difference is the target computation. Standard DQN uses the target network for both action selection and evaluation, which causes overestimation. Double DQN decouples these: the online network selects the action, but the target network evaluates it."

> "MCTS is our planning baseline. It uses UCB1 for tree selection and random rollouts — 200 simulations per move. It requires no training at all."

## 3. Training Setup (30s)

*Point to Section 4*

> "Training uses frozen-opponent self-play. The agent plays against a frozen copy of itself that's updated every 500 episodes. I use canonical state encoding so the same network can play both colors. I trained for 20,000 episodes across 2 random seeds."

## 4. Results — The Key Findings (90s)

*Point to Sections 5-7, walk through the figures*

> "Here's the training win rate curve. Both DQN and DDQN peak around 25% early on when playing against the random opponent, then stabilize around 15% as the frozen opponent gets stronger."

> "The Q-value analysis is the core of this comparison."

*Point to Q-value figure*

> "You can see DQN has a slightly higher average Q-value — 1.89 versus 1.84 for Double DQN. This confirms the overestimation bias that Double DQN was designed to address. The overestimation gap — Q minus target — is consistently positive and slightly larger for DQN."

> "And DDQN shows lower variance across seeds — ±0.8% versus ±1.4% — which suggests more stable training."

*Point to evaluation bar chart*

> "The most striking result is the evaluation. MCTS achieves 93% win rate against the random opponent — it dominates. DQN gets 13%, DDQN gets 18%. Against the heuristic opponent, MCTS still wins 63%, while DQN and DDQN hover near 0% wins with mostly draws."

> "The high draw rate for DQN/DDQN suggests they learned to not lose, but haven't learned to win — likely because terminal rewards are too sparse for the network to learn aggressive strategies."

## 5. Conclusions (30s)

*Point to Section 8*

> "Three takeaways: First, MCTS significantly outperforms learned value-based methods in this setting — planning beats learning with limited training. Second, Double DQN does show lower overestimation and more stable training, confirming the theory. Third, sparse terminal rewards are a major bottleneck for DQN methods in two-player games. Future directions include reward shaping and AlphaZero-style combinations of MCTS with neural networks."

> "Thanks — happy to take questions."
