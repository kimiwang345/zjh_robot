from __future__ import annotations
from typing import List, Tuple
from zjh_constants import (
    HAND_CAT_BOMB, HAND_CAT_STRAIGHT_FLUSH, HAND_CAT_FLUSH,
    HAND_CAT_STRAIGHT, HAND_CAT_PAIR, HAND_CAT_HIGH
)

# card: (rank, suit) rank 2..14, suit 0..3

def _is_flush(cards: List[Tuple[int, int]]) -> bool:
    s = cards[0][1]
    return cards[1][1] == s and cards[2][1] == s

def _sorted_ranks(cards: List[Tuple[int, int]]) -> List[int]:
    return sorted([c[0] for c in cards])

def _is_straight(ranks_sorted: List[int]) -> Tuple[bool, int]:
    # A23 作为最小顺子（可改；但必须固定）
    a, b, c = ranks_sorted
    # 常规顺子
    if b == a + 1 and c == b + 1:
        return True, c
    # A23
    if ranks_sorted == [2, 3, 14]:
        return True, 3
    return False, 0

from typing import List, Tuple

Card = Tuple[int, int]  # (rank, suit), rank: 2~14(A), suit: 0~3


def get_hand_rank(cards: List[Card]) -> int:
    """
    返回 0~5 的牌型 bucket
    0 高牌
    1 对子
    2 顺子
    3 金花
    4 顺金
    5 炸弹
    """

    assert len(cards) == 3, "炸金花必须是 3 张牌"

    ranks = sorted([c[0] for c in cards])
    suits = [c[1] for c in cards]

    # ========= 炸弹（3 张同点数） =========
    if ranks[0] == ranks[1] == ranks[2]:
        return 5

    # ========= 是否同花 =========
    is_flush = (suits[0] == suits[1] == suits[2])

    # ========= 是否顺子 =========
    # 特判 A-2-3
    if ranks == [2, 3, 14]:
        is_straight = True
    else:
        is_straight = (ranks[0] + 1 == ranks[1] and ranks[1] + 1 == ranks[2])

    # ========= 顺金 =========
    if is_flush and is_straight:
        return 4

    # ========= 金花 =========
    if is_flush:
        return 3

    # ========= 顺子 =========
    if is_straight:
        return 2

    # ========= 对子 =========
    if ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        return 1

    # ========= 高牌 =========
    return 0

def evaluate_3cards(cards: List[Tuple[int, int]]) -> Tuple[int, Tuple[int, ...]]:
    """
    返回： (category, tiebreaker)
    category 越大越强
    tiebreaker 按字典序比较（越大越强）
    """
    ranks = _sorted_ranks(cards)
    flush = _is_flush(cards)
    straight, top = _is_straight(ranks)

    # 三条（炸弹/豹子）
    if ranks[0] == ranks[1] == ranks[2]:
        return HAND_CAT_BOMB, (ranks[2],)

    # 顺金
    if flush and straight:
        return HAND_CAT_STRAIGHT_FLUSH, (top,)

    # 金花
    if flush:
        return HAND_CAT_FLUSH, tuple(sorted(ranks, reverse=True))

    # 顺子
    if straight:
        return HAND_CAT_STRAIGHT, (top,)

    # 对子
    if ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        if ranks[0] == ranks[1]:
            pair = ranks[0]
            kicker = ranks[2]
        else:
            pair = ranks[1]
            kicker = ranks[0]
        return HAND_CAT_PAIR, (pair, kicker)

    # 高牌
    return HAND_CAT_HIGH, tuple(sorted(ranks, reverse=True))

def compare_hands(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    """
    返回：1 表示 a 赢，-1 表示 b 赢，0 平
    """
    ca, ta = evaluate_3cards(a)
    cb, tb = evaluate_3cards(b)
    if ca != cb:
        return 1 if ca > cb else -1
    if ta == tb:
        return 0
    return 1 if ta > tb else -1
