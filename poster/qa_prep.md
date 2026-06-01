# Q&A Preparation

---

## Part 1: Questions They Might Ask You

### Q1: "Why 5×5 instead of standard 8×8 chess?"
> **Answer:** Computational feasibility. Standard chess has a state space of ~10^47 and action space of ~10^3 per position. 5×5 Gardner Mini-Chess reduces this dramatically — 625 possible actions — making it feasible to train DQN on a single GPU in reasonable time. It's still strategically rich (it's been weakly solved; game-theoretic value is a draw), so it serves as a valid testbed for comparing RL algorithms.

### Q2: "Why is the win rate so low for DQN/DDQN? Only 13-18%?"
> **Answer:** Two main reasons:
> 1. **Sparse terminal rewards** — the agent only gets feedback at the end of the game (after 40-50 moves). This makes credit assignment extremely difficult. Many games reach the 200 move limit without a decisive result.
> 2. **Self-play instability** — when training against a frozen copy of itself, the reward distribution keeps shifting as the opponent changes. This is a known challenge in self-play RL.
>
> The high draw rate (~70-80%) suggests the agents learned a conservative strategy — they avoid losing but haven't learned to win. This is actually not uncommon for value-based methods with sparse rewards.

### Q3: "What's the significance of the Q-value overestimation result?"
> **Answer:** This is the core finding that validates the Double DQN theory. In standard DQN, the max operator in the target computation (max_a' Q(s', a'; θ⁻)) creates a positive bias because it always picks the noisiest high estimate. Double DQN decouples selection from evaluation to reduce this. Our data shows DQN's average Q-value (1.89) is indeed higher than DDQN's (1.84), and the overestimation gap (Q - target) is consistently larger for DQN. This is exactly what the Van Hasselt et al. (2016) paper predicted.

### Q4: "Why does MCTS so drastically outperform DQN/DDQN?"
> **Answer:** MCTS has a fundamental advantage in this setting:
> - It uses **direct lookahead search** — it literally simulates future game states to evaluate moves
> - It doesn't need training — 200 simulations per move is enough to find strong plays
> - In contrast, DQN tries to learn a value function from experience, which requires many thousands of games to converge
>
> This is consistent with the broader finding in game AI that search-based methods tend to outperform pure learning in small, deterministic games. AlphaZero's success came from **combining** MCTS with neural networks.

### Q5: "How does the self-play frozen opponent approach work? Why not play against the latest version?"
> **Answer:** Playing against the *latest* version (online self-play) causes severe instability — the opponent distribution changes every step, violating the stationary MDP assumption that DQN relies on. The frozen opponent approach creates a more stable training signal. The frozen copy is refreshed every 500 episodes to gradually increase difficulty. This is similar to fictitious self-play used in game theory research.

### Q6: "What happens with more training? Would 100K episodes help?"
> **Answer:** Likely yes, but with diminishing returns. The learning curves show the agents are still slowly improving at 20K episodes. With more training (and epsilon fully decayed to 0.05), the agents would have more time to exploit learned strategies. However, the fundamental bottleneck of sparse rewards would remain. Reward shaping or curriculum learning would likely help more than just additional episodes.

### Q7: "Why use MSE/Huber loss instead of something else?"
> **Answer:** I initially used MSE loss (standard in the original DQN paper), but it caused loss divergence due to Q-value explosion. I switched to Huber loss (Smooth L1) which is more robust to outliers — it behaves like MSE for small errors but L1 for large errors. I also added Q-value clamping to [-2, 2] since the reward range is only ±1.

### Q8: "How does the canonical encoding work?"
> **Answer:** The state is always encoded from the current player's perspective, as if they were White. When it's Black's turn, the board is rotated 180° and the piece colors are swapped in the encoding. This way, a single network can play both colors — it always sees "my pieces" in channels 0-5 and "opponent's pieces" in channels 6-11. The action space is similarly canonicalized.

### Q9: "Why only 2 random seeds? Is that enough?"
> **Answer:** Ideally we'd use 3-5 seeds, but each seed takes ~6 hours on an RTX 4060 Laptop GPU. With 2 seeds we can still compute mean and standard deviation, which gives a basic measure of variance. The results are consistent across seeds, which is encouraging. For a full publication, more seeds would be needed.

### Q10: "What's the heuristic opponent?"
> **Answer:** It follows a simple priority: first check if any move gives check → play that. If not, look for captures → play a random capture. Otherwise, play a random legal move. It's stronger than pure random because it actively exploits tactical opportunities, but it has no strategic understanding. It serves as an intermediate baseline between random and trained agents.

---

## Part 2: Questions You Can Ask the Professor

### Q1: "Do you think reward shaping would help, or would it introduce bias?"
**Why ask this:** Shows you're thinking about solutions to the sparse reward problem.
**Expected answer:** "Reward shaping can definitely help with convergence speed, but you need to be careful. If you use potential-based reward shaping (Ng et al., 1999), it preserves the optimal policy. Giving rewards for captures or material advantage is a common approach in chess RL. However, it might bias the agent toward short-term material gain over long-term positional play."

### Q2: "Would a policy gradient method like PPO be more suitable for this type of game?"
**Why ask this:** Shows breadth of RL knowledge.
**Expected answer:** "PPO could work, but it has its own challenges in two-player games. The advantage is it directly optimizes the policy rather than learning Q-values, which can be more stable. The disadvantage is it's on-policy, so it can't reuse old experience like DQN. For games, A3C or IMPALA-style architectures have been used. But the real breakthrough in game AI was AlphaZero combining MCTS with policy+value networks."

### Q3: "Is there a theoretical explanation for why DQN/DDQN produce so many draws?"
**Why ask this:** Shows analytical thinking about your results.
**Expected answer:** "In self-play with frozen opponents, the agent gets rewarded for not losing. If the value function assigns similar values to many positions (which happens with sparse rewards), the agent plays conservatively. The γ=0.99 discount also means future rewards are heavily discounted over 40+ moves, making it hard to distinguish winning trajectories from drawing ones. You might try γ=1.0 for terminal-reward-only games."

### Q4: "If I were to combine MCTS with the DQN value function (like AlphaZero), what would be the simplest way?"
**Why ask this:** Shows ambition and forward thinking. This is your "future work" direction.
**Expected answer:** "The simplest approach would be to use the trained Q-network as the rollout policy instead of random rollouts. A more sophisticated approach (closer to AlphaZero) would be to use the network for both a policy prior (to guide MCTS selection) and a value estimate (to replace rollouts). You'd need to add a policy head to your network and train it with MCTS-generated targets."

---

## Part 3: Difficult Questions & Honest Answers

### "Your DQN barely beats random. Isn't this a failure?"
> "It's a limited result, not a failure. The objective wasn't to solve Mini-Chess — it was to compare the *learning behavior* of DQN vs Double DQN under identical conditions. We confirmed that Double DQN reduces overestimation and improves stability, which is the theoretical contribution. The low absolute performance highlights the challenge of sparse rewards in adversarial games, which is itself an interesting finding."

### "2 seeds isn't statistically significant."
> "You're right that 2 seeds provides limited statistical power. The results are consistent across seeds, which is encouraging, but I acknowledge this is a limitation. Given the computational constraints (each seed takes ~6 hours), this was a practical tradeoff. For the final paper, I plan to run at least one more seed."

### "Why not just use AlphaZero?"
> "AlphaZero would certainly perform better, but it combines MCTS with neural networks — it's a fundamentally different paradigm. The goal of this project was to isolate and compare specific components: pure value-learning (DQN), improved value-learning (DDQN), and pure planning (MCTS). Understanding these building blocks is important before combining them."
