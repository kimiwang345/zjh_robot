from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 动作空间上限（你 env.legal_actions 每步一般不会太多，但我们做上限保护）
MAX_ACTIONS = 64

def obs_to_vector(obs: Dict[str, Any]) -> np.ndarray:
    """
    固定长度特征：这里沿用你 PG 版本的结构，并加入 table_size（2人桌此时为常数，但后续多人数会用到）。
    """
    me = obs["self"]
    num_players = obs.get("num_players", 2)  # 兼容老 obs
    table_size = float(num_players) / 9.0

    v = [
        table_size,
        obs["round"] / 20.0,
        obs["alive_count"] / 9.0,
        obs["pot"] / 5000.0,
        obs["last_bet_mul"] / 20.0,
        1.0 if obs["last_bet_seen"] else 0.0,

        me["stack"] / 5000.0,
        me["bet_total"] / 5000.0,
        1.0 if me["has_seen"] else 0.0,
        obs["min_bet_mul"] / 20.0,
        1.0 if obs["force_allin_or_fold"] else 0.0,
    ]

    # 对手统计：最多 8 个对手 padding（2人桌=1个对手）
    opps = obs["opponents"]
    opps_sorted = sorted(opps, key=lambda x: x["seat"])
    for i in range(8):
        if i < len(opps_sorted):
            o = opps_sorted[i]
            v.extend([
                1.0 if o["alive"] else 0.0,
                1.0 if o["has_seen"] else 0.0,
                o["stack"] / 5000.0,
                o["bet_total"] / 5000.0,
            ])
        else:
            v.extend([0.0, 0.0, 0.0, 0.0])

    return np.asarray(v, dtype=np.float32)

@dataclass
class ActionEncoding:
    actions: List[Dict[str, Any]]
    mask: np.ndarray  # shape=(MAX_ACTIONS,), 0/1

def encode_actions(legal: List[Dict[str, Any]]) -> ActionEncoding:
    actions = legal[:MAX_ACTIONS]
    mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
    mask[:len(actions)] = 1.0
    return ActionEncoding(actions=actions, mask=mask)

def action_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.get("type") != b.get("type"):
        return False
    t = a.get("type")
    if t == "BET":
        return int(a.get("mul")) == int(b.get("mul"))
    if t == "PK":
        return int(a.get("target")) == int(b.get("target"))
    return True

def action_to_index(action: Dict[str, Any], enc: ActionEncoding) -> int:
    for i, a in enumerate(enc.actions):
        if action_equal(action, a):
            return i
    return -1

class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, MAX_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # (B, MAX_ACTIONS)

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buf = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buf)

    def push(
        self,
        s: np.ndarray,
        mask: np.ndarray,
        a_idx: int,
        r: float,
        s2: np.ndarray,
        mask2: np.ndarray,
        done: bool,
    ):
        self.buf.append((s, mask, a_idx, r, s2, mask2, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, batch_size)
        s, m, a, r, s2, m2, d = zip(*batch)
        return (
            np.stack(s),
            np.stack(m),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.stack(s2),
            np.stack(m2),
            np.asarray(d, dtype=np.float32),
        )

@torch.no_grad()
def select_action_epsilon_greedy(
    qnet: QNet,
    obs_vec: np.ndarray,
    enc: ActionEncoding,
    eps: float,
    device: str,
    rng: random.Random,
) -> int:
    # 随机探索：只能在合法动作里随机
    if rng.random() < eps:
        legal_n = int(enc.mask.sum())
        if legal_n <= 0:
            return 0
        return rng.randrange(legal_n)

    x = torch.from_numpy(obs_vec).to(device).unsqueeze(0)  # (1, obs_dim)
    q = qnet(x).squeeze(0).cpu().numpy()  # (MAX_ACTIONS,)

    # mask 非法动作
    q[enc.mask < 0.5] = -1e18
    return int(np.argmax(q))

@torch.no_grad()
def greedy_action(
    qnet: QNet,
    obs_vec: np.ndarray,
    enc: ActionEncoding,
    device: str,
) -> int:
    x = torch.from_numpy(obs_vec).to(device).unsqueeze(0)
    q = qnet(x).squeeze(0).cpu().numpy()
    q[enc.mask < 0.5] = -1e18
    return int(np.argmax(q))
