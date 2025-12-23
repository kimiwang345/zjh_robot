# state_builder_normalized.py
import numpy as np
from typing import Dict, Any, List, Tuple


# ============================================================
# 牌型评估（必须与 ZJHEnvMulti14_8p 完全一致）
# ============================================================

def evaluate_hand(cards: List[Tuple[int, int]]) -> int:
    """
    cards: [(rank, suit), ...]
    rank: 2~14
    suit: 0~3
    """
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    r1, r2, r3 = ranks

    is_flush = (len(set(suits)) == 1)
    is_straight = (r1 == r2 + 1 == r3 + 2) or (r1 == 14 and r2 == 3 and r3 == 2)

    if r1 == r2 == r3:
        return 5
    if is_flush and is_straight:
        return 4
    if is_flush:
        return 3
    if is_straight:
        return 2
    if r1 == r2 or r2 == r3:
        return 1
    return 0


# ============================================================
# 49 维 State Builder（金额线性归一化版）
# ============================================================

class ZJHStateBuilder49Normalized:
    """
    作用：
    - 把线上任意筹码 / 底注
    - 映射回训练时的尺度（ante = 1, stack ≈ 100）
    """

    def build(self, payload: Dict[str, Any]) -> np.ndarray:
        state = np.zeros(49, dtype=np.float32)

        # =========================
        # 一、Hero 手牌
        # =========================
        cards = [(c["rank"], c["suit"]) for c in payload["my_cards"]]
        hand_type = evaluate_hand(cards)

        ranks = sorted([c[0] for c in cards], reverse=True)
        hi, mid, lo = ranks
        strength = (hand_type + hi / 14.0) / 6.0

        state[0] = hand_type
        state[1] = hi
        state[2] = mid
        state[3] = lo
        state[4] = strength

        # =========================
        # 二、牌局结构信息（不归一化）
        # =========================
        state[5] = payload.get("round", 0)
        state[6] = payload.get("num_players", payload.get("alive_players", 1))
        state[7] = payload.get("alive_players", 1)
        state[11] = 1.0 if payload.get("has_looked", False) else 0.0

        # =========================
        # 三、金额归一化（核心）
        # =========================
        ante = float(payload.get("ante", 1.0))
        if ante <= 0:
            ante = 1.0  # 防止除 0

        # 训练环境中 ante 恒为 1
        state[8] = 1.0
        state[9] = payload.get("pot", 0.0) / ante
        state[10] = payload.get("max_bet", 0.0) / ante
        state[12] = payload.get("chips", 0.0) / ante
        state[13] = payload.get("hero_bet", 0.0) / ante

        # =========================
        # 四、对手状态（1~7）
        # =========================
        opponents = {o["seat"]: o for o in payload.get("opponents", [])}
        idx = 14

        for seat in range(1, 8):
            o = opponents.get(seat)
            if o and o.get("alive", False):
                state[idx]     = 1.0
                state[idx + 1] = 1.0 if o.get("has_looked", False) else 0.0
                state[idx + 2] = o.get("chips", 0.0) / ante
                state[idx + 3] = o.get("bet", 0.0) / ante
                state[idx + 4] = o.get("last_act", 0)
            else:
                state[idx:idx + 5] = 0.0
            idx += 5

        return state
