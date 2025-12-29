# policy_constraints.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

from config.config_loader import HotConfig

@dataclass
class Constraint:
    allow_fold: bool
    allow_pk: bool
    allow_bet: bool
    bet_mul_max_ratio: float
    bet_mul_min_ratio: float

    # 强牌“禁止弃牌/必须继续”的阈值
    force_no_fold_hand_cat_gte: int
    force_bet_or_pk_hand_cat_gte: int

    # 前期回合额外限制
    early_round_lte: int
    early_round_max_ratio: float

def _match_band(band: Dict[str, Any], hand_cat: int) -> bool:
    if "hand_cat_eq" in band and hand_cat != int(band["hand_cat_eq"]):
        return False
    if "hand_cat_lte" in band and hand_cat > int(band["hand_cat_lte"]):
        return False
    if "hand_cat_gte" in band and hand_cat < int(band["hand_cat_gte"]):
        return False
    return True

class ConstraintEngine:
    def __init__(self, path: str = "config/policy_constraints.json"):
        self.cfg = HotConfig(path)

    def get_constraint(self, hand_cat: Optional[int]) -> Constraint:
        data = self.cfg.get()
        d = data["defaults"]

        # 默认约束
        c = Constraint(
            allow_fold=bool(d["allow_fold"]),
            allow_pk=bool(d["allow_pk"]),
            allow_bet=bool(d["allow_bet"]),
            bet_mul_max_ratio=float(d["bet_mul_max_ratio"]),
            bet_mul_min_ratio=float(d["bet_mul_min_ratio"]),
            force_no_fold_hand_cat_gte=int(d["force_no_fold_hand_cat_gte"]),
            force_bet_or_pk_hand_cat_gte=int(d["force_bet_or_pk_hand_cat_gte"]),
            early_round_lte=int(d["early_round_lte"]),
            early_round_max_ratio=float(d["early_round_max_ratio"]),
        )

        # 未看牌 / 无法评估牌力：返回默认
        if hand_cat is None:
            return c

        # 命中 band 覆盖
        for band in data.get("bands", []):
            if _match_band(band, hand_cat):
                c.allow_fold = bool(band.get("allow_fold", c.allow_fold))
                c.allow_pk = bool(band.get("allow_pk", c.allow_pk))
                c.allow_bet = bool(band.get("allow_bet", c.allow_bet))
                c.bet_mul_max_ratio = float(band.get("bet_mul_max_ratio", c.bet_mul_max_ratio))
                c.bet_mul_min_ratio = float(band.get("bet_mul_min_ratio", c.bet_mul_min_ratio))
                break

        return c
