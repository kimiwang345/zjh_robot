# ai_service.py
from typing import Dict, Any
import numpy as np
import torch

from agent_a30 import a30Agent
from config_a30_hybrid import A30ConfigHybrid
from difficulty_policy import DifficultyPolicy, Difficulty
from state_builder_normalized import ZJHStateBuilder49Normalized


# ============================================================
# 动作常量（必须与训练环境一致）
# ============================================================

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


ACTION_NAMES = {
    0: "FOLD",
    1: "CALL",
    2: "LOOK",
    3: "BET_2X",
    4: "BET_3X",
    5: "BET_4X",
    6: "BET_5X",
    7: "BET_6X",
    8: "BET_7X",
    9: "BET_8X",
    10: "BET_9X",
    11: "BET_10X",
    12: "PK_ONE",
    13: "COMPARE_ALL",
}


# ============================================================
# AI Service（生产级）
# ============================================================

class ZJHAIService:

    def __init__(self, model_path: str):
        # ---- load model once ----
        self.cfg = A30ConfigHybrid()
        self.agent = a30Agent(self.cfg)
        self.agent.load(model_path)

        self.agent.policy_net.eval()
        self.agent.target_net.eval()

        # ---- helpers ----
        self.state_builder = ZJHStateBuilder49Normalized()
        self.difficulty_policy = DifficultyPolicy(self.agent)

        # ---- deterministic inference ----
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ========================================================
    # LOOK 两阶段决策（核心）
    # ========================================================

    def _handle_look(self, raw_action: int, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        如果触发 LOOK 且尚未看牌：
        - 返回 sideEffect=LOOK
        - 不返回真实 action
        - 上层应立即更新 has_looked 并重新请求 AI
        """
        if raw_action == LOOK and not payload.get("has_looked", False):
            return {
                "action": -1,                 # NO_ACTION
                "actionName": "NO_ACTION",
                "sideEffect": "LOOK",
            }
        return None

    # ========================================================
    # BET 合法性裁剪（核心）
    # ========================================================

    def _min_raise_multiplier(self, payload: Dict[str, Any]) -> int:
        """
        计算当前最小合法加注倍数（基于 ante）
        """
        ante = float(payload.get("ante", 1))
        if ante <= 0:
            return 0

        max_bet = float(payload.get("max_bet", 0))
        hero_bet = float(payload.get("hero_bet", 0))

        need = max(0.0, max_bet - hero_bet)
        # 向上取整
        return int((need + ante - 1) // ante)

    def _apply_bet_constraints(self, action: int, payload: Dict[str, Any]) -> int:
        """
        防止出现：
        - 已有较高下注，却返回 BET_2X / BET_3X 等非法动作
        """
        if action < BET_2X or action > BET_10X:
            return action

        # action 3~11 → 倍数 2~10
        mult = action - 1
        min_mult = self._min_raise_multiplier(payload)

        if mult < min_mult:
            # 升级到最小合法倍数
            new_mult = min(min_mult, 10)
            return new_mult + 1

        return action

    # ========================================================
    # 基础规则兜底
    # ========================================================

    def _apply_basic_constraints(self, action: int, payload: Dict[str, Any]) -> int:
        """
        一些不会破坏策略的最小规则兜底
        """
        # 已看牌，不允许再 LOOK
        if payload.get("has_looked", False) and action == LOOK:
            return CALL

        # 无筹码，不允许下注 / PK
        if payload.get("chips", 0) <= 0 and action >= BET_2X:
            return CALL

        return action

    # ========================================================
    # 对外接口
    # ========================================================

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload: 你现在设计的线上标准参数
        """
        # ---- build state ----
        state = self.state_builder.build(payload)

        # ---- difficulty ----
        diff_str = payload.get("difficulty", "hard").lower()
        difficulty = (
            Difficulty(diff_str)
            if diff_str in Difficulty._value2member_map_
            else Difficulty.HARD
        )

        # ---- model decision ----
        raw_action = self.difficulty_policy.select(state, difficulty)

        # ---- LOOK 两阶段处理 ----
        look_resp = self._handle_look(raw_action, payload)
        if look_resp is not None:
            look_resp["difficulty"] = difficulty.value
            return look_resp

        # ---- BET 合法性裁剪 ----
        action = self._apply_bet_constraints(raw_action, payload)

        # ---- 基础规则兜底 ----
        action = self._apply_basic_constraints(action, payload)

        # ---- final ----
        return {
            "action": int(action),
            "actionName": ACTION_NAMES.get(action, "UNKNOWN"),
            "difficulty": difficulty.value,
        }
