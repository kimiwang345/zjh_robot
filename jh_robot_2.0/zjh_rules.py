from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from zjh_constants import GameConfig

@dataclass
class BetContext:
    last_bet_mul: int            # 当前桌面最高倍数（单调不减）
    last_bet_seen: bool          # 最后一次“产生/更新 last_bet_mul 的下注者”是否看牌

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def normalize_min_mul(min_mul: int, peeked: bool, cfg: GameConfig) -> int:
    """
    把 min_mul 落到合法倍数集合：
    - 没看牌：1..10
    - 看牌：2..20 且偶数
    """
    if peeked:
        min_mul = max(min_mul, cfg.min_mul_seen)
        if min_mul % 2 != 0:
            min_mul += 1
        min_mul = clamp(min_mul, cfg.min_mul_seen, cfg.max_mul_seen)
        # 仍需保证偶数
        if min_mul % 2 != 0:
            min_mul = clamp(min_mul + 1, cfg.min_mul_seen, cfg.max_mul_seen)
        return min_mul
    else:
        min_mul = clamp(min_mul, 1, cfg.max_mul_unseen)
        return min_mul

def calc_min_bet_mul(ctx: BetContext, peeked: bool, cfg: GameConfig) -> int:
    """
    按你给的口径：
    - lastPeeked && !peeked => ceil(lastBetMul / 2)
    - !lastPeeked && peeked => lastBetMul * 2
    - else => lastBetMul
    然后再做：看牌偶数倍 + 上下限裁剪
    """
    x = ctx.last_bet_mul
    last_peeked = ctx.last_bet_seen

    if last_peeked and (not peeked):
        raw = int(math.ceil(x / 2.0))
    elif (not last_peeked) and peeked:
        raw = x * 2
    else:
        raw = x

    return normalize_min_mul(raw, peeked, cfg)

def is_valid_bet_mul(mul: int, peeked: bool, cfg: GameConfig) -> bool:
    if peeked:
        return (cfg.min_mul_seen <= mul <= cfg.max_mul_seen) and (mul % 2 == 0)
    return 1 <= mul <= cfg.max_mul_unseen

def bet_cost(mul: int, base: int) -> int:
    return mul * base

def pk_cost(ctx: BetContext, base: int) -> int:
    # 你定义：PK 也要下 last_bet_mul
    return ctx.last_bet_mul * base
