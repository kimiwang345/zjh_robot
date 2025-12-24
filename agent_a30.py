import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# Q Network
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# A30 Agent (Mask-aware DQN)
# ============================================================

class A30Agent:
    """
    - 支持 Action Mask
    - epsilon-greedy 也 respect mask
    - target Q 计算时使用 next_mask
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=300_000,
        target_update_interval=5_000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = gamma

        # epsilon
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.total_steps = 0
        self.target_update_interval = target_update_interval

        # networks
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    # --------------------------------------------------------
    # Action Selection (Mask-aware)
    # --------------------------------------------------------

    def select_action(self, state, action_mask):
        """
        state: np.ndarray, shape = (state_dim,)
        action_mask: np.ndarray, shape = (action_dim,), values in {0,1}
        """
        self.total_steps += 1
        self._update_epsilon()

        # 保证 mask 至少有一个合法动作
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            raise RuntimeError("No valid action available (mask empty)")

        # epsilon-greedy（respect mask）
        if random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # greedy
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)[0]  # (action_dim,)

        # mask 非法动作
        mask_t = torch.tensor(action_mask, device=self.device)
        masked_q = q_values.masked_fill(mask_t == 0, -1e9)

        return int(torch.argmax(masked_q).item())

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(self, replay_buffer, batch_size=64):
        if replay_buffer.size() < batch_size:
            return
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            return
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            masks,
            next_masks,
            weights,
            indices,
        ) = batch

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        masks = torch.tensor(masks, dtype=torch.float32, device=self.device)
        next_masks = torch.tensor(next_masks, dtype=torch.float32, device=self.device)

        # -------------------------
        # Current Q
        # -------------------------
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # -------------------------
        # Target Q (mask-aware)
        # -------------------------
        with torch.no_grad():
            # Double DQN
            next_q_values = self.q_net(next_states)
            next_q_values = next_q_values.masked_fill(next_masks == 0, -1e9)
            next_actions = torch.argmax(next_q_values, dim=1)

            next_target_q = self.target_q_net(next_states)
            next_target_q = next_target_q.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + self.gamma * (1.0 - dones) * next_target_q

        # -------------------------
        # Loss (PER-weighted)
        # -------------------------
        td_error = target_q - q_sa
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新 PER 优先级
        # replay_buffer.update_priorities(indices, td_error.abs().detach().cpu().numpy())

        # td_error_np: shape (batch,)
        td_error = td_error.detach()

        # ===== NaN 防线 =====
        if not torch.isfinite(td_error).all():
            return  # ⭐ 直接跳过这一步训练


        td_error_np = td_error.abs().detach().cpu().numpy().reshape(-1)

        # ⭐ 强制 indices 为 List[int]
        indices = [int(i) for i in indices]

        replay_buffer.update_priorities(indices, td_error_np)



        # -------------------------
        # Update target network
        # -------------------------
        if self.total_steps % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    # --------------------------------------------------------
    # Utils
    # --------------------------------------------------------

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon - decay)

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            path,
        )

    def set_eval(self):
        self.q_net.eval()
        self.target_q_net.eval()

    def set_train(self):
        self.q_net.train()
        self.target_q_net.train()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt["target_q_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.total_steps = ckpt.get("total_steps", 0)
