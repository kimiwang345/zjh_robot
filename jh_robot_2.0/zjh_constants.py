from dataclasses import dataclass

# 花色：0♦ 1♣ 2♥ 3♠
SUITS = [0, 1, 2, 3]
RANKS = list(range(2, 15))  # 2..14 (A=14)

# 牌型：炸弹>顺金>金花>顺子>对子>高牌
HAND_CAT_BOMB = 6
HAND_CAT_STRAIGHT_FLUSH = 5
HAND_CAT_FLUSH = 4
HAND_CAT_STRAIGHT = 3
HAND_CAT_PAIR = 2
HAND_CAT_HIGH = 1

@dataclass
class GameConfig:
    num_players: int = 6          # 2..9
    base: int = 1                 # 底注
    max_rounds: int = 20          # 最多 20 回合
    init_stack: int = 200         # 每人初始筹码（单位=筹码，不是倍数）
    seed: int = 42

    # 倍数规则
    # 没看牌：1..10
    # 看牌：2..20 且偶数
    max_mul_unseen: int = 10
    min_mul_seen: int = 2
    max_mul_seen: int = 20
