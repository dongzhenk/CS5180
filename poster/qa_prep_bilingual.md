# Q&A 准备 (中英对照版)

---

## Part 1: 可能被问的问题 / Questions They Might Ask

### Q1: "Why 5×5 instead of standard 8×8 chess?"
### 为什么选 5×5 而不是标准 8×8？

> **Answer / 回答:**
> Computational feasibility. Standard chess has a state space of ~10^47 and action space of ~10^3 per position. 5×5 Gardner Mini-Chess reduces this dramatically — 625 possible actions — making it feasible to train DQN on a single GPU in reasonable time. It's still strategically rich (it's been weakly solved; game-theoretic value is a draw), so it serves as a valid testbed for comparing RL algorithms.
>
> 计算可行性。标准国际象棋状态空间约 10^47，每步约 10^3 个动作。5×5 Mini-Chess 大幅缩减到 625 个动作，可以在单个 GPU 上合理时间内训练。但它仍然策略丰富（已被弱解，博弈论值为平局），是比较 RL 算法的有效测试环境。

---

### Q2: "Why is the win rate so low for DQN/DDQN? Only 13-18%?"
### 为什么 DQN/DDQN 胜率这么低？才 13-18%？

> **Answer / 回答:**
> Two main reasons:
> 1. **Sparse terminal rewards** — the agent only gets feedback at the end of the game (after 40-50 moves). This makes credit assignment extremely difficult.
> 2. **Self-play instability** — the reward distribution keeps shifting as the frozen opponent changes.
>
> The high draw rate (~70-80%) suggests the agents learned a conservative strategy — they avoid losing but haven't learned to win.
>
> 两个原因：
> 1. **稀疏终局奖励** — 智能体只在游戏结束时（40-50 步后）获得反馈，信用分配极其困难。
> 2. **自对弈不稳定** — 随着冻结对手更新，奖励分布不断变化。
>
> 高平局率（~70-80%）说明智能体学会了不输，但没学会赢。

---

### Q3: "What's the significance of the Q-value overestimation result?"
### Q 值过高估计结果说明什么？

> **Answer / 回答:**
> This is the core finding. In standard DQN, the max operator in the target (max_a' Q(s', a'; θ⁻)) always picks the noisiest high estimate, creating positive bias. Double DQN decouples selection from evaluation. Our data shows DQN's average Q (1.89) > DDQN's (1.84), and the overestimation gap is consistently larger for DQN. This confirms the Van Hasselt et al. (2016) theory.
>
> 这是核心发现。标准 DQN 的 target 计算中，max 操作总是选择噪声最大的高估值，产生正偏差。Double DQN 将动作选择和价值评估解耦。数据显示 DQN 平均 Q 值（1.89）> DDQN（1.84），过高估计差距持续更大。这验证了 Van Hasselt 等人（2016）的理论。

---

### Q4: "Why does MCTS so drastically outperform DQN/DDQN?"
### 为什么 MCTS 大幅碾压 DQN/DDQN？

> **Answer / 回答:**
> MCTS uses **direct lookahead search** — it simulates future states to evaluate moves. It doesn't need training. In contrast, DQN tries to learn a value function from experience, requiring many thousands of games to converge. This is consistent with the broader finding that search-based methods outperform pure learning in small, deterministic games. AlphaZero's success came from **combining** MCTS with neural networks.
>
> MCTS 使用**直接前瞻搜索**——模拟未来状态来评估走法，不需要训练。相比之下，DQN 尝试从经验中学习价值函数，需要数千局才能收敛。这符合"搜索方法在小型确定性博弈中优于纯学习"的普遍发现。AlphaZero 的成功正是**结合**了 MCTS 和神经网络。

---

### Q5: "How does the frozen opponent self-play work? Why not the latest version?"
### 冻结对手自对弈怎么工作？为什么不用最新版本？

> **Answer / 回答:**
> Playing against the *latest* version causes severe instability — the opponent distribution changes every step, violating DQN's stationary MDP assumption. The frozen copy creates a more stable training signal and is refreshed every 500 episodes to gradually increase difficulty. This is similar to fictitious self-play in game theory.
>
> 对最新版本对弈会导致严重不稳定——对手分布每步都变，违反 DQN 的平稳 MDP 假设。冻结副本创造更稳定的训练信号，每 500 局更新一次逐步增加难度。这类似于博弈论中的虚拟自对弈（fictitious self-play）。

---

### Q6: "Would more training (100K episodes) help?"
### 更多训练（100K episodes）会更好吗？

> **Answer / 回答:**
> Likely yes, but with diminishing returns. The learning curves show agents are still slowly improving at 20K. However, the fundamental bottleneck of sparse rewards would remain. Reward shaping or curriculum learning would likely help more than just additional episodes.
>
> 可能会，但收益递减。学习曲线显示 20K 时智能体仍在缓慢提升。但稀疏奖励的根本瓶颈仍在。奖励塑形（reward shaping）或课程学习（curriculum learning）可能比单纯增加训练量更有效。

---

### Q7: "Why Huber loss instead of MSE?"
### 为什么用 Huber Loss 而不是 MSE？

> **Answer / 回答:**
> I initially used MSE (standard in the original DQN paper), but it caused loss divergence due to Q-value explosion. Huber loss is more robust to outliers — MSE for small errors, L1 for large errors. I also added Q-value clamping to [-2, 2] since the reward range is only ±1.
>
> 最初使用 MSE（原始 DQN 论文标准），但由于 Q 值爆炸导致 loss 发散。Huber loss 对异常值更鲁棒——小误差用 MSE，大误差用 L1。同时加了 Q 值裁剪 [-2, 2]，因为奖励范围只有 ±1。

---

### Q8: "How does the canonical encoding work?"
### 规范编码（canonical encoding）是怎么工作的？

> **Answer / 回答:**
> The state is always encoded from the current player's perspective, as if they were White. When it's Black's turn, the board is rotated 180° and piece colors are swapped. This way, a single network plays both colors — it always sees "my pieces" in channels 0-5 and "opponent's pieces" in channels 6-11.
>
> 状态始终从当前玩家的视角编码，就像他们是白方一样。轮到黑方时，棋盘旋转 180° 并交换棋子颜色。这样同一个网络可以执白执黑——它总是在通道 0-5 看到"我的棋子"，通道 6-11 看到"对手的棋子"。

---

### Q9: "Only 2 seeds — is that statistically significant?"
### 只有 2 个 seed，统计上够吗？

> **Answer / 回答:**
> You're right that 2 seeds provides limited statistical power. The results are consistent across seeds, which is encouraging, but I acknowledge this as a limitation. Each seed takes ~6 hours on an RTX 4060, so this was a practical tradeoff. For the final paper, I plan to run at least one more seed.
>
> 您说得对，2 个 seed 统计能力有限。结果在两个 seed 间一致，令人鼓舞，但我承认这是局限性。每个 seed 在 RTX 4060 上需要 ~6 小时，这是实际的权衡。最终论文中计划至少再跑一个 seed。

---

### Q10: "What is the heuristic opponent exactly?"
### 启发式对手具体是什么？

> **Answer / 回答:**
> It follows a simple priority: first check if any move gives check → play that. If not, look for captures → play a random capture. Otherwise, play a random legal move. It's stronger than pure random because it exploits tactical opportunities, but has no strategic understanding.
>
> 它遵循简单的优先级：首先检查是否有将军的走法→走那步。如果没有，寻找吃子→随机选一个吃子。否则，走随机合法步。它比纯随机强因为会利用战术机会，但没有战略理解。

---

## Part 2: 你可以问教授的问题 / Questions to Ask the Professor

### Q1: "Do you think reward shaping would help, or would it introduce bias?"
### "您认为奖励塑形（reward shaping）会有帮助，还是会引入偏差？"

**为什么问这个 / Why ask:** 展示你在思考稀疏奖励问题的解决方案。

> **预期回答 / Expected answer:**
> "Reward shaping can definitely help with convergence speed, but you need to be careful. If you use potential-based reward shaping (Ng et al., 1999), it preserves the optimal policy. Giving rewards for captures or material advantage is common in chess RL. However, it might bias the agent toward short-term material gain over long-term positional play."
>
> "奖励塑形肯定能帮助加速收敛，但要小心。如果用基于势函数的奖励塑形（Ng 等，1999），可以保持最优策略不变。对吃子或材料优势给奖励在棋类 RL 中很常见。不过可能会让智能体偏向短期物质收益而忽略长期位置优势。"

---

### Q2: "Would a policy gradient method like PPO be more suitable for this game?"
### "像 PPO 这样的策略梯度方法是否更适合这种博弈？"

**为什么问这个 / Why ask:** 展示你对不同 RL 方法的广度了解。

> **预期回答 / Expected answer:**
> "PPO could work, but has its own challenges. The advantage is it directly optimizes the policy, which can be more stable. The disadvantage is it's on-policy, so it can't reuse old experience like DQN. For games, A3C or IMPALA-style architectures have been used. But the real breakthrough was AlphaZero combining MCTS with policy+value networks."
>
> "PPO 可以用，但有自己的挑战。优点是直接优化策略，可能更稳定。缺点是 on-policy 的，不能像 DQN 那样复用旧经验。在游戏中，A3C 或 IMPALA 架构也被使用过。但真正的突破是 AlphaZero 将 MCTS 与策略+价值网络结合。"

---

### Q3: "Is there a theoretical explanation for why DQN/DDQN produce so many draws?"
### "DQN/DDQN 产生大量平局有没有理论解释？"

**为什么问这个 / Why ask:** 展示对结果的分析性思考。

> **预期回答 / Expected answer:**
> "In self-play with frozen opponents, the agent gets rewarded for not losing. If the value function assigns similar values to many positions (which happens with sparse rewards), the agent plays conservatively. The γ=0.99 discount also means future rewards are heavily discounted over 40+ moves. You might try γ=1.0 for terminal-reward-only games."
>
> "在冻结对手自对弈中，智能体因为'不输'就能获得奖励。如果价值函数给很多局面赋予相似值（稀疏奖励下容易发生），智能体就会保守下棋。γ=0.99 的折扣也意味着 40+ 步后的奖励被严重折扣。对于只有终局奖励的游戏，可以尝试 γ=1.0。"

---

### Q4: "What would be the simplest way to combine MCTS with the DQN value function, like AlphaZero?"
### "将 MCTS 和 DQN 价值函数结合（类似 AlphaZero）最简单的方式是什么？"

**为什么问这个 / Why ask:** 展示前瞻性思维，是你的"future work"方向。

> **预期回答 / Expected answer:**
> "The simplest approach would be to use the trained Q-network as the rollout policy instead of random rollouts. A more sophisticated approach (closer to AlphaZero) would be to use the network for both a policy prior (to guide MCTS selection) and a value estimate (to replace rollouts entirely). You'd need to add a policy head to your network and train with MCTS-generated targets."
>
> "最简单的方法是用训练好的 Q 网络作为 rollout 策略，替代随机 rollout。更高级的方法（更接近 AlphaZero）是同时用网络做策略先验（引导 MCTS 选择）和价值估计（完全替代 rollout）。需要给网络加一个策略头，用 MCTS 生成的目标来训练。"

---

## Part 3: 棘手问题的诚实回答 / Tough Questions & Honest Answers

### "Your DQN barely beats random. Isn't this a failure?"
### "你的 DQN 几乎赢不了随机。这不是失败吗？"

> "It's a limited result, not a failure. The objective wasn't to solve Mini-Chess — it was to compare the *learning behavior* of DQN vs Double DQN under identical conditions. We confirmed that Double DQN reduces overestimation and improves stability, which is the theoretical contribution. The low absolute performance highlights the challenge of sparse rewards in adversarial games, which is itself an interesting finding."
>
> "这是受限的结果，不是失败。目标不是解决 Mini-Chess，而是在相同条件下比较 DQN 和 Double DQN 的*学习行为*。我们确认了 Double DQN 降低了过高估计并提高了稳定性，这是理论贡献。较低的绝对性能恰恰突出了稀疏奖励在对抗性游戏中的挑战，这本身就是有趣的发现。"

---

### "2 seeds isn't statistically significant."
### "2 个 seed 没有统计显著性。"

> "You're right that 2 seeds provides limited statistical power. The results are consistent across seeds, which is encouraging, but I acknowledge this as a limitation. Given the computational constraints (~6 hours per seed), this was a practical tradeoff. For the final paper, I plan to run additional seeds."
>
> "您说得对，2 个 seed 统计能力有限。结果在 seed 间一致，这是好的，但我承认这是局限性。考虑到计算约束（每个 seed ~6 小时），这是实际权衡。最终论文中计划跑更多 seed。"

---

### "Why not just use AlphaZero?"
### "为什么不直接用 AlphaZero？"

> "AlphaZero would certainly perform better, but it combines MCTS with neural networks — a fundamentally different paradigm. This project's goal was to isolate and compare specific components: pure value-learning (DQN), improved value-learning (DDQN), and pure planning (MCTS). Understanding these building blocks is important before combining them."
>
> "AlphaZero 肯定性能更好，但它结合了 MCTS 和神经网络——是根本不同的范式。这个项目的目标是隔离比较特定组件：纯价值学习（DQN）、改进价值学习（DDQN）和纯规划（MCTS）。在组合它们之前，理解这些基础组件很重要。"
