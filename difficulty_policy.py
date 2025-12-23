# difficulty_policy.py
import numpy as np
import torch
from enum import Enum


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# 动作常量（与环境一致）
FOLD = 0
CALL = 1
LOOK = 2
BET_2X = 3
BET_3X = 4
BET_4X = 5
BET_5X = 6
BET_6X = 7
BET_7X = 8
BET_8X = 9
BET_9X = 10
BET_10X = 11
PK_ONE = 12
COMPARE_ALL = 13


# ============================================================
# 难度策略
# ============================================================

class DifficultyPolicy:

    def __init__(self, agent):
        self.agent = agent

    def select(self, state, difficulty: Difficulty) -> int:
        if difficulty == Difficulty.HARD:
            return self._hard(state)
        elif difficulty == Difficulty.MEDIUM:
            return self._medium(state)
        elif difficulty == Difficulty.EASY:
            return self._easy(state)
        else:
            return self._hard(state)

    # ------------------------
    # HARD：最强（纯 argmax）
    # ------------------------
    def _hard(self, state):
        return self.agent.select_action(state, eps=0.0)

    # ------------------------
    # MEDIUM：Top-K 随机（推荐）
    # ------------------------
    def _medium(self, state, k: int = 3):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.agent.policy_net(s).cpu().numpy()[0]

        top_k = np.argsort(q)[-k:]
        return int(np.random.choice(top_k))

    # ------------------------
    # EASY：限制动作空间
    # ------------------------
    def _easy(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.agent.policy_net(s).cpu().numpy()[0]

        # 只允许安全动作
        allowed_actions = [
            CALL,
            LOOK,
            BET_2X,
            BET_3X,
        ]

        # 从允许集合中选 Q 最大的
        best = max(allowed_actions, key=lambda a: q[a])
        return int(best)
