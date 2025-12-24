import random
import numpy as np


# ============================================================
# SumTree（PER 核心结构）
# ============================================================

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        # 向上更新
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        """
        s ∈ [0, total)
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ============================================================
# Prioritized Experience Replay Buffer (Mask-aware)
# ============================================================

class PERReplayBuffer:
    """
    每条 transition：
      (state, action, reward, next_state, done, mask, next_mask)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 500_000,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / max(1, beta_frames)

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.frame = 1

    # --------------------------------------------------------
    # 基本接口
    # --------------------------------------------------------

    def size(self):
        return self.tree.size

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        mask,
        next_mask,
    ):
        """
        存储 transition
        """
        data = (
            state,
            action,
            reward,
            next_state,
            done,
            mask,
            next_mask,
        )
        self.tree.add(self.max_priority, data)

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    def sample(self, batch_size):
        """
        Returns:
            (states, actions, rewards, next_states, dones, masks, next_masks)
            or None if sampling failed
        """

        if self.size() < batch_size:
            return None

        batch = []
        idxs = []
        total_p = self.tree.total()
        if not np.isfinite(total_p) or total_p <= 0:
            return None

        segment = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            # ⭐ 关键防御 1：采样失败
            if data is None:
                return None

            batch.append(data)
            idxs.append(idx)

        # ⭐ 关键防御 2：batch 完整性
        if len(batch) != batch_size:
            return None

        # ⭐ 关键防御 3：zip 前兜底
        try:
            states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        except Exception:
            return None

        states = np.stack(states)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.float32)
        masks = np.stack(masks)
        next_masks = np.stack(next_masks)

        # importance-sampling weights
        priorities = []
        for idx in idxs:
            priorities.append(self.tree.tree[idx])

        probs = np.array(priorities) / self.tree.total()
        weights = (self.size() * probs) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            masks,
            next_masks,
            idxs,
            weights,
        )


    # --------------------------------------------------------
    # Update priorities
    # --------------------------------------------------------

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            idx = int(idx)        # ⭐ 核心修复
            p = float(p)

            # ===== PER 数值防线 =====
            if not np.isfinite(p):
                p = 1e-6
            p = max(p, 1e-6)

            self.tree.update(idx, p)


            self.max_priority = max(self.max_priority, p)


    # --------------------------------------------------------
    # Utils
    # --------------------------------------------------------

    def _beta_by_frame(self, frame_idx):
        return min(
            1.0,
            self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames,
        )
