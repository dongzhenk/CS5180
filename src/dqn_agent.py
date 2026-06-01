"""DQN and Double DQN agent for Mini-Chess self-play."""

import copy, random
import numpy as np
import torch
import torch.nn.functional as F

from src.model import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    """Handles both standard DQN and Double DQN (controlled by *double* flag)."""

    def __init__(self, config, double=False, device=None):
        self.double = double
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # networks
        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # epsilon schedule
        self.epsilon = config.epsilon_start
        self._eps_start = config.epsilon_start
        self._eps_end = config.epsilon_end
        self._eps_decay = config.epsilon_decay_episodes

        self.opt_steps = 0
        self.target_update_freq = config.target_update_freq

    # ----- action selection ------------------------------------------------

    def select_action(self, state, legal_mask):
        """Epsilon-greedy action in canonical space."""
        if random.random() < self.epsilon:
            legal = np.where(legal_mask > 0)[0]
            return int(np.random.choice(legal))
        return self._greedy_action(state, legal_mask)

    def select_action_greedy(self, state, legal_mask):
        """Pure greedy (for evaluation)."""
        return self._greedy_action(state, legal_mask)

    def _greedy_action(self, state, legal_mask):
        self.online_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(s).squeeze(0).cpu().numpy()
        self.online_net.train()
        q[legal_mask == 0] = -np.inf
        return int(np.argmax(q))

    # ----- learning --------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done, next_mask):
        self.replay_buffer.push(state, action, reward, next_state, done, next_mask)

    def update(self):
        """One gradient step. Returns loss value or None if buffer too small."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, masks = \
            self.replay_buffer.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        m  = torch.FloatTensor(masks).to(self.device)

        # current Q(s, a)
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # target (clamp Q-values to [-2, 2] to prevent divergence)
        with torch.no_grad():
            if self.double:
                # Double DQN: online selects, target evaluates
                online_q = self.online_net(ns)
                online_q[m == 0] = -1e9
                best_a = online_q.argmax(dim=1)
                target_q = self.target_net(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: target does both
                tq = self.target_net(ns)
                tq[m == 0] = -1e9
                target_q = tq.max(dim=1)[0]
            target_q = target_q.clamp(-2.0, 2.0)
            target = r + (1.0 - d) * self.gamma * target_q

        loss = F.smooth_l1_loss(q_values, target)   # Huber loss, more robust
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.opt_steps += 1
        if self.opt_steps % self.target_update_freq == 0:
            self.update_target()

        # return loss and Q-value stats for overestimation analysis
        return {
            'loss': loss.item(),
            'mean_q': q_values.mean().item(),
            'max_q': q_values.max().item(),
            'mean_target': target.mean().item(),
        }

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ----- epsilon decay ---------------------------------------------------

    def decay_epsilon(self, episode):
        frac = min(1.0, episode / max(1, self._eps_decay))
        self.epsilon = self._eps_start + frac * (self._eps_end - self._eps_start)

    # ----- frozen copy for self-play opponent ------------------------------

    def make_frozen_copy(self):
        """Return a *detached* copy of the online network (eval mode)."""
        net = copy.deepcopy(self.online_net)
        net.eval()
        return net

    # ----- save / load -----------------------------------------------------

    def save(self, path):
        torch.save({
            'online': self.online_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'opt_steps': self.opt_steps,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.opt_steps = ckpt['opt_steps']
        self.epsilon = ckpt['epsilon']
