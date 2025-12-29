# policy_mask.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from action_codec import encode_bet_mul, PK_BASE
from cards import Card, hand_strength
from policy_constraints import ConstraintEngine

_engine = ConstraintEngine("config/policy_constraints.json")

def _hand_cat_from_obs(obs: Dict[str, Any]) -> Optional[int]:
    me = obs["me"]
    if not me.get("seen"):
        return None
    cards = me.get("cards")
    if not cards:
        return None
    cs = [Card(int(c["rank"]), int(c["suit"])) for c in cards]
    cat, _ = hand_strength(cs)  # 0~5
    return int(cat)

def build_action_mask(obs: Dict[str, Any]) -> np.ndarray:
    me = obs["me"]
    base_bet = int(obs["base_bet"])
    round_id = int(obs.get("round", 0))

    my_seen = bool(me["seen"])
    my_stack = int(me["stack"])
    min_mul = int(me["min_mul"])

    # action space = 29
    mask = np.zeros(29, dtype=np.float32)

    # 基础：FOLD 默认允许（后面会被强牌约束关掉）
    mask[0] = 1.0

    hand_cat = _hand_cat_from_obs(obs)
    cons = _engine.get_constraint(hand_cat)

    # ========= 强牌：自动约束动作空间（关键） =========
    if hand_cat is not None and hand_cat >= cons.force_no_fold_hand_cat_gte:
        if not cons.allow_fold:
            mask[0] = 0.0  # 禁止弃牌（炸弹/顺金阶段不会再弃牌）

    # ========= PK =========
    if cons.allow_pk:
        # PK 简化按 min_mul 需要能支付
        if my_stack >= base_bet * min_mul:
            # 如果强牌要求“必须继续”，PK 仍允许
            mask[PK_BASE] = 1.0

    # ========= BET（按相对 min_mul 限制上下界） =========
    if cons.allow_bet:
        # 下注倍数合法范围：由看牌决定上限
        mul_max_abs = 10 if (not my_seen) else 20

        # 约束后的最大倍数：min_mul * ratio
        mul_max_by_ratio = int(np.floor(min_mul * cons.bet_mul_max_ratio))
        mul_min_by_ratio = int(np.ceil(min_mul * cons.bet_mul_min_ratio))

        # 前期回合更保守（可配置）
        if round_id <= cons.early_round_lte:
            mul_max_by_ratio = min(mul_max_by_ratio, int(np.floor(min_mul * cons.early_round_max_ratio)))

        # 统一 clamp
        mul_min = max(min_mul, mul_min_by_ratio)
        mul_max = min(mul_max_abs, mul_max_by_ratio)
        if mul_max < mul_min:
            mul_max = mul_min

        for mul in range(1, 21):
            # 看牌/不看牌倍数体系
            if not my_seen and mul > 10:
                continue
            if my_seen and (mul < 2 or mul % 2 != 0):
                continue

            # 绝对下限：min_mul
            if mul < min_mul:
                continue

            # 约束上下界（相对 min_mul）
            if mul < mul_min or mul > mul_max:
                continue

            # 筹码
            if mul * base_bet > my_stack:
                continue

            mask[encode_bet_mul(mul)] = 1.0

    # ========= 强牌“必须继续”：如果 fold 被禁且 bet/pk 都被卡死，兜底放开最小 bet =========
    if mask[0] == 0.0 and mask.sum() == 0.0:
        # 兜底：允许最小合法下注
        for mul in range(min_mul, 21):
            if not my_seen and mul > 10:
                continue
            if my_seen and (mul < 2 or mul % 2 != 0):
                continue
            if mul * base_bet <= my_stack:
                mask[encode_bet_mul(mul)] = 1.0
                break

    # 兜底：至少允许 FOLD（除非强牌禁 fold）
    if mask.sum() == 0.0:
        mask[0] = 1.0

    return mask
