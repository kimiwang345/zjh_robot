# model_hier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNetHier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # fold head: 2 classes (FOLD / PLAY)
        self.fold_head = nn.Linear(hidden, 2)

        # play head: actions 1..28（不含FOLD=0）
        self.play_head = nn.Linear(hidden, 28)

        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        fold_logits = self.fold_head(h)         # [B,2]
        play_logits = self.play_head(h)         # [B,28]
        value = self.value_head(h).squeeze(-1)  # [B]
        return fold_logits, play_logits, value
