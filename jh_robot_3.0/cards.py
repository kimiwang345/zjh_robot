# cards.py
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple

@dataclass(frozen=True)
class Card:
    rank: int   # 2..14
    suit: int   # 0..3

def _is_straight(ranks: List[int]) -> Tuple[bool, int]:
    r = sorted(ranks)
    # A23 特例
    if r == [2, 3, 14]:
        return True, 3
    if r[0] + 1 == r[1] and r[1] + 1 == r[2]:
        return True, r[2]
    return False, 0

def hand_strength(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    """
    category: 5炸弹 > 4顺金 > 3金花 > 2顺子 > 1对子 > 0高牌
    """
    assert len(cards) == 3
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    cnt = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)
    uniq = sorted(cnt.keys(), reverse=True)

    flush = (len(set(suits)) == 1)
    straight, high = _is_straight(ranks)

    if counts == [3]:
        return 5, (uniq[0],)

    if straight and flush:
        return 4, (high,)

    if flush:
        return 3, tuple(sorted(ranks, reverse=True))

    if straight:
        return 2, (high,)

    if counts == [2, 1]:
        pair_rank = max(k for k, v in cnt.items() if v == 2)
        kicker = max(k for k, v in cnt.items() if v == 1)
        return 1, (pair_rank, kicker)

    return 0, tuple(sorted(ranks, reverse=True))
