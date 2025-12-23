# agent_a30.py
"""
A30 Rainbow Agent（Hybrid Stage-3）
+ Action Mask for a11 regularization
"""

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_a30_hybrid import A30ConfigHybrid


# ============================================================
# Noisy Linear
# ============================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 / math.sqrt(self.out_features))

    def sample_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

# ============================================================
# PER (Prioritized Experience Replay)
# ============================================================

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def add(self, priority: float, idx: int):
        tree_idx = idx + self.capacity
        self.update(tree_idx, priority)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += change

    def total(self) -> float:
        return float(self.tree[1])

    def get_leaf(self, value: float) -> int:
        parent = 1
        while parent < self.capacity:
            left = 2 * parent
            right = left + 1
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        return parent

class PERReplayBuffer:
    def __init__(
            self,
            capacity: int,
            alpha: float,
            beta_start: float,
            beta_end: float,
            beta_anneal_steps: int,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps

        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0

        self.states = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.max_priority = 1.0
        self.step_counter = 0

    def _ensure_state_shape(self, state_dim: int):
        if self.states is None:
            self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
            self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self._ensure_state_shape(state.shape[-1])

        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, idx)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def beta_by_step(self):
        t = min(self.step_counter, self.beta_anneal_steps)
        ratio = t / float(self.beta_anneal_steps)
        return self.beta_start + ratio * (self.beta_end - self.beta_start)

    def sample(self, batch_size: int, total_steps: int):
        self.step_counter = total_steps

        batch_idx = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float32)

        total_p = self.tree.total() + 1e-8
        segment = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            tree_idx = self.tree.get_leaf(value)
            data_idx = tree_idx - self.capacity
            batch_idx[i] = data_idx
            priorities[i] = max(self.tree.tree[tree_idx], 1e-8)

        prob = priorities / total_p
        beta = self.beta_by_step()

        weights = (self.size * prob) ** (-beta)
        weights /= weights.max() + 1e-6

        return (
            torch.from_numpy(self.states[batch_idx]).float(),
            torch.from_numpy(self.actions[batch_idx]).long(),
            torch.from_numpy(self.rewards[batch_idx]).float(),
            torch.from_numpy(self.next_states[batch_idx]).float(),
            torch.from_numpy(self.dones[batch_idx]).float(),
            torch.from_numpy(weights).float(),
            batch_idx,
        )

    def update_priorities(self, idxs, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        for i, td in zip(idxs, td_errors):
            p = float(abs(td) + 1e-6)
            self.max_priority = max(self.max_priority, p)
            self.tree.add(p ** self.alpha, int(i))


# ============================================================
# Dueling Rainbow Net
# ============================================================

class DuelingRainbowNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.val_fc = NoisyLinear(256, 256)
        self.val_out = NoisyLinear(256, 1)

        self.adv_fc = NoisyLinear(256, 256)
        self.adv_out = NoisyLinear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = self.val_out(F.relu(self.val_fc(x)))
        a = self.adv_out(F.relu(self.adv_fc(x)))

        return v + a - a.mean(dim=1, keepdim=True)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# ============================================================
# A30 Agent（with Action Mask）
# ============================================================

class a30Agent:
    def __init__(self, cfg: A30ConfigHybrid):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.policy_net = DuelingRainbowNet(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net = DuelingRainbowNet(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.buffer = PERReplayBuffer(
            capacity=cfg.buffer_capacity,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            beta_end=cfg.per_beta_end,
            beta_anneal_steps=cfg.per_beta_anneal_steps,
        )

        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.target_update_freq = cfg.target_update_freq
        self.max_grad_norm = cfg.max_grad_norm
        self.grad_accumulate_steps = cfg.grad_accumulate_steps

        self.total_updates = 0
        self.grad_accum_counter = 0

    # ========================================================
    # Action Mask（只管 a11）
    # ========================================================

    def compute_action_mask(self, state):
        """
        state layout（关键字段）：
        idx 7  : alive_cnt_ratio
        idx 9  : pot_bb
        idx 12 : hero_stack_bb
        idx 5  : round_index
        """
        mask = torch.ones(self.cfg.action_dim, device=self.device)

        hero_stack_bb = state[12]
        alive_ratio = state[7]
        pot_bb = state[9]
        round_idx = state[5]

        alive_cnt = alive_ratio * 9.0

        # a11 正则化
        if hero_stack_bb < 12:
            mask[11] = 0
        if alive_cnt >= 4 and round_idx < 2:
            mask[11] = 0
        if pot_bb < 3:
            mask[11] = 0

        return mask

    # ========================================================
    # Action Selection
    # ========================================================

    def select_action(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s).squeeze(0)
            mask = self.compute_action_mask(s.squeeze(0))
            q[mask == 0] = -1e9
        return int(q.argmax().item())

    # ========================================================
    # Learn
    # ========================================================

    def learn(self, total_steps):
        if len(self.buffer) < self.batch_size:
            return None

        if self.grad_accum_counter % self.grad_accumulate_steps == 0:
            self.optimizer.zero_grad()

        self.policy_net.sample_noise()
        self.target_net.sample_noise()

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
            idxs,
        ) = self.buffer.sample(self.batch_size, total_steps)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        q_values = self.policy_net(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.policy_net(next_states)
            for i in range(next_q.size(0)):
                mask = self.compute_action_mask(next_states[i])
                next_q[i][mask == 0] = -1e9

            next_actions = next_q.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q_target_a = next_q_target.gather(1, next_actions).squeeze(1)

            target = rewards + (1.0 - dones) * self.gamma * next_q_target_a

        td_errors = target - q_a
        loss = (weights * td_errors.pow(2)).mean()
        loss.backward()

        self.grad_accum_counter += 1
        if self.grad_accum_counter % self.grad_accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.total_updates += 1

            if self.total_updates % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.buffer.update_priorities(idxs, td_errors)
        return loss.item()



    # ----------------- Buffer -----------------

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
