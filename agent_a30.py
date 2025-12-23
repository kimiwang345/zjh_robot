# agent_a30.py
"""
A30 Rainbow Agent（Hybrid Stage-3）

特性：
- Double DQN
- Dueling 架构
- PER（优先经验回放）
- NoisyNet（固定噪声）
- 支持 Hybrid 配置：
    * cfg.device            : "cpu" / "cuda"
    * cfg.batch_size
    * cfg.learn_interval
    * cfg.grad_accumulate_steps
    * cfg.max_grad_norm
- 支持 train_a30_stage3_safe_stop.py 的断点续训接口
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
# Noisy Linear（Fixed-Noisy 版本）
# ============================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device):
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        if self.weight_epsilon is None or self.bias_epsilon is None:
            return
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
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
# PER（优先经验回放）
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
        self.step_counter = 0  # 用于 beta 退火

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
        if self.size == 0:
            raise ValueError("PERReplayBuffer is empty")

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
        eps = 1e-6

        prob = np.clip(prob, eps, 1.0)
        weights = (self.size * prob) ** (-beta)
        weights = weights / (weights.max() + eps)
        weights = np.clip(weights, 0.0, 10.0)

        weights = torch.from_numpy(weights).float()
        states = torch.from_numpy(self.states[batch_idx])
        actions = torch.from_numpy(self.actions[batch_idx])
        rewards = torch.from_numpy(self.rewards[batch_idx])
        next_states = torch.from_numpy(self.next_states[batch_idx])
        dones = torch.from_numpy(self.dones[batch_idx])

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
            batch_idx,
        )

    def update_priorities(self, idxs, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        for i, td in zip(idxs, td_errors):
            p = float(abs(td) + 1e-6)
            if p > self.max_priority:
                self.max_priority = p
            self.tree.add(p ** self.alpha, int(i))


# ============================================================
# Dueling Rainbow 网络
# ============================================================

class DuelingRainbowNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden1 = 256
        hidden2 = 256

        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.val_fc = NoisyLinear(hidden2, hidden2)
        self.val_out = NoisyLinear(hidden2, 1)

        self.adv_fc = NoisyLinear(hidden2, hidden2)
        self.adv_out = NoisyLinear(hidden2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.val_fc(x))
        v = self.val_out(v)

        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean
        return q

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# ============================================================
# A30 Agent
# ============================================================

class a30Agent:
    def __init__(self, cfg: A30ConfigHybrid, reward_shaping_fn=None):
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

        self.epsilon = cfg.eps_start  # 仅用于日志
        self.reward_shaping_fn = reward_shaping_fn

    @property
    def eval_net(self):
        """
        统一对外评估网络接口
        A30: policy_net
        """
        return self.policy_net

    # ----------------- epsilon（仅用于日志） -----------------

    def epsilon_by_episode(self, ep: int) -> float:
        decay_ep = getattr(self.cfg, "eps_decay_episodes", 1)
        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * \
            max(0.0, (decay_ep - ep) / decay_ep)
        self.epsilon = eps
        return eps

    # ----------------- 动作选择 -----------------

    def select_action(self, state, eps: float = 0.0) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s)
        action = int(q.argmax(dim=1).item())
        return action

    # ----------------- Reward Shaping -----------------

    def shape_reward(self, state, action, raw_reward, next_state, done, info):
        if self.reward_shaping_fn is None:
            return raw_reward
        return self.reward_shaping_fn(state, action, raw_reward, next_state, done, info)

    # ----------------- Buffer -----------------

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ----------------- 学习步骤（支持梯度累积） -----------------

    def learn(self, total_steps: int):
        if len(self.buffer) < self.batch_size:
            return None

        # 仅在一次“累积周期”开头清零梯度
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
            next_actions = next_q.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states)
            next_q_target_a = next_q_target.gather(1, next_actions).squeeze(1)

            target = rewards + (1.0 - dones) * self.gamma * next_q_target_a

        td_errors = target - q_a
        loss = (weights * td_errors.pow(2)).mean()

        loss.backward()

        # 每 grad_accumulate_steps 次更新一次参数
        self.grad_accum_counter += 1
        stepped = False
        if self.grad_accum_counter % self.grad_accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.total_updates += 1
            stepped = True

            if self.total_updates % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.sample_noise()

        # 每次都更新 PER 权重（不会太慢）
        self.buffer.update_priorities(idxs, td_errors)

        return loss.item() if stepped else None

    # ----------------- 保存 / 加载 -----------------

    def save(self, path: str, total_steps: int, eps: float):
        ckpt = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg.__dict__,
            "total_steps": total_steps,
            "epsilon": eps,
            "total_updates": self.total_updates,
            "grad_accum_counter": self.grad_accum_counter,
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> Tuple[int, float]:
        ckpt = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "cfg" in ckpt:
            for k, v in ckpt["cfg"].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)

        total_steps = ckpt.get("total_steps", 0)
        eps = ckpt.get("epsilon", self.cfg.eps_end)
        self.total_updates = ckpt.get("total_updates", 0)
        self.grad_accum_counter = ckpt.get("grad_accum_counter", 0)
        self.epsilon = eps

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        return total_steps, eps
