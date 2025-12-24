from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ACTIONS = 64  # 动作列表最大长度（超出则截断）

def obs_to_vector(obs: Dict[str, Any]) -> np.ndarray:
    """
    把 obs 转成定长向量（可按你需要扩充特征）。
    """
    me = obs["self"]
    v = [
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

    # 对手统计：取最多 8 个对手，padding
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

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, MAX_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)

@dataclass
class ActionEncoding:
    actions: List[Dict[str, Any]]
    mask: np.ndarray  # (MAX_ACTIONS,) 0/1

def encode_actions(legal: List[Dict[str, Any]]) -> ActionEncoding:
    actions = legal[:MAX_ACTIONS]
    mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
    mask[:len(actions)] = 1.0
    return ActionEncoding(actions=actions, mask=mask)

def sample_action(model: PolicyNet, obs_vec: np.ndarray, enc: ActionEncoding, device: str = "cpu") -> Tuple[int, float]:
    """
    返回：选择的 action_index, log_prob
    """
    x = torch.from_numpy(obs_vec).to(device).unsqueeze(0)  # (1, obs_dim)
    logits = model(x).squeeze(0)  # (MAX_ACTIONS,)
    mask = torch.from_numpy(enc.mask).to(device)

    # mask 无效动作：用很小的值
    masked_logits = logits + (mask - 1.0) * 1e9
    probs = torch.softmax(masked_logits, dim=-1)

    dist = torch.distributions.Categorical(probs=probs)
    idx = dist.sample()

    logp = dist.log_prob(idx)   # Tensor，有 grad
    return int(idx.item()), logp

    #logp = dist.log_prob(idx).item()
    #return int(idx.item()), float(logp)

def greedy_action(model: PolicyNet, obs_vec: np.ndarray, enc: ActionEncoding, device: str = "cpu") -> int:
    x = torch.from_numpy(obs_vec).to(device).unsqueeze(0)
    logits = model(x).squeeze(0).detach().cpu().numpy()
    logits[enc.mask < 0.5] = -1e18
    return int(np.argmax(logits))
