# dueling_dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        hidden = 256

        # 公共特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Value 分支 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Advantage 分支 A(s,a)
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)                # (B, 1)
        adv = self.adv_stream(x)                    # (B, A)

        # dueling 汇总：Q = V + (A - mean(A))
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = value + (adv - adv_mean)
        return q
