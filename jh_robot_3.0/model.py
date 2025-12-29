# model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from action_codec import ACT_DIM

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, ACT_DIM)
        self.v = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value
