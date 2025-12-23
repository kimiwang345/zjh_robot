# per_buffer.py
import numpy as np
from typing import Tuple

class PrioritizedReplayBuffer:
    def __init__(self,
                 capacity: int,
                 alpha: float = 0.6,
                 eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self.pos = 0
        self.size = 0

        self.states = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # 优先级
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def _init_state_arrays(self, state: np.ndarray):
        if self.states is None:
            state_shape = state.shape
            self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
            self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.float32)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool):
        if self.states is None:
            self._init_state_arrays(state)

        idx = self.pos

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # 新样本给最大优先级，保证被采样
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self,
               batch_size: int,
               beta: float) -> Tuple[np.ndarray, ...]:
        assert self.size > 0, "Buffer is empty"

        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size,
                                   batch_size,
                                   p=probs,
                                   replace=False if batch_size <= self.size else True)

        # importance-sampling 权重
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一到 [0,1]

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]

        return (batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
                indices,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.eps
        self.priorities[indices] = td_errors

    def __len__(self):
        return self.size
