# action_codec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

MAX_OPP_SLOTS = 8

# 动作空间：
# 0: FOLD
# 1..8: PK_SLOT_0..7
# 9..28: BET_MUL_1..20  (20个)
FOLD_ID = 0
PK_BASE = 1
BET_BASE = PK_BASE + MAX_OPP_SLOTS  # 9
BET_COUNT = 20
ACT_DIM = BET_BASE + BET_COUNT      # 29

@dataclass(frozen=True)
class DecodedAction:
    kind: str                    # "FOLD"|"PK"|"BET"
    bet_mul: Optional[int] = None
    pk_slot: Optional[int] = None

def decode_action(aid: int) -> DecodedAction:
    if aid == FOLD_ID:
        return DecodedAction("FOLD")
    if PK_BASE <= aid < PK_BASE + MAX_OPP_SLOTS:
        return DecodedAction("PK", pk_slot=aid - PK_BASE)
    if BET_BASE <= aid < BET_BASE + BET_COUNT:
        mul = (aid - BET_BASE) + 1  # 1..20
        return DecodedAction("BET", bet_mul=mul)
    return DecodedAction("FOLD")

def encode_bet_mul(mul: int) -> int:
    mul = max(1, min(20, int(mul)))
    return BET_BASE + (mul - 1)

def build_opp_slots(alive_opps: List[Tuple[int, bool, int]]) -> List[Tuple[int, bool, int]]:
    """
    输入：[(seat, seen, stack), ...] 只包含“存活对手”
    输出：固定最多8个slot，用于PK_SLOT_i映射
    默认按 seat 升序，超出截断
    """
    return sorted(alive_opps, key=lambda x: x[0])[:MAX_OPP_SLOTS]

def pk_slot_to_seat(slots: List[Tuple[int, bool, int]], slot: int) -> Optional[int]:
    if 0 <= slot < len(slots):
        return slots[slot][0]
    return None
