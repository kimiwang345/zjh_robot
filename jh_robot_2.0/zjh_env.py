from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from zjh_constants import GameConfig
from zjh_hand_eval import compare_hands
from zjh_rules import BetContext, calc_min_bet_mul, bet_cost, pk_cost, is_valid_bet_mul

Card = Tuple[int, int]  # (rank, suit)


# ======================================================
# Hand rank bucket (0~5)
# ======================================================
def hand_rank_to_bucket(cards: List[Card]) -> int:
    """
    返回 0~5
    0 高牌
    1 对子
    2 顺子
    3 金花
    4 顺金
    5 炸弹

    说明：
    - 复用 compare_hands 的内部逻辑
    - 这里只做 bucket 映射，不泄露精确信息
    """

    # 构造一个“最弱对手”，通过 compare_hands 推断牌型层级
    # 你也可以直接替换成你已有的 classify 函数（如果有）
    # 这里假设 compare_hands 内部已经区分 6 类牌型

    # ⚠️ 推荐方式：如果 zjh_hand_eval 里有 get_hand_rank(cards)，优先用它
    try:
        from zjh_hand_eval import get_hand_rank
        return int(get_hand_rank(cards))  # 必须返回 0~5
    except ImportError:
        # 兜底：根据 compare_hands 的返回区间映射
        # （如果你没有 get_hand_rank，就必须自己补一个）
        raise RuntimeError(
            "请在 zjh_hand_eval.py 中提供 get_hand_rank(cards) -> 0~5"
        )


# ======================================================
# Player State
# ======================================================
@dataclass
class PlayerState:
    seat: int
    alive: bool
    has_seen: bool
    stack: int
    bet_total: int
    cards: List[Card]
    hand_rank: int = 0   # 0~5，默认未知=0


# ======================================================
# ZJH Environment
# ======================================================
class ZJHEnv:
    """
    金花 / 诈金花 环境（支持 DQN / DDQN）
    """

    def __init__(self, cfg: GameConfig):
        assert 2 <= cfg.num_players <= 9
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)

        self.players: List[PlayerState] = []
        self.pot: int = 0

        self.dealer_seat: int = 0
        self.acting_seat: int = 0

        self.round_index: int = 0
        self._turn_marks: set[int] = set()

        self.last_bet_mul: int = 1
        self.last_bet_seen: bool = False

        self.hero_seat: int = 0
        self.hero_init_stack: int = 0

        self.done: bool = False
        self.winner_seat: Optional[int] = None

    # ======================================================
    # Reset
    # ======================================================
    def _new_deck(self) -> List[Card]:
        deck = [(r, s) for r in range(2, 15) for s in range(4)]
        self.rng.shuffle(deck)
        return deck

    def reset(self, seed: Optional[int] = None, hero_seat: int = 0) -> Dict[str, Any]:
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.done = False
        self.winner_seat = None
        self.hero_seat = hero_seat
        self.hero_init_stack = self.cfg.init_stack

        self.pot = 0
        self.round_index = 1
        self._turn_marks = set()

        self.players = []
        deck = self._new_deck()

        for i in range(self.cfg.num_players):
            self.players.append(
                PlayerState(
                    seat=i,
                    alive=True,
                    has_seen=False,
                    stack=self.cfg.init_stack,
                    bet_total=0,
                    cards=[deck.pop(), deck.pop(), deck.pop()],
                    hand_rank=0,
                )
            )

        # 底注
        for p in self.players:
            pay = min(p.stack, self.cfg.base)
            p.stack -= pay
            p.bet_total += pay
            self.pot += pay

        self.dealer_seat = self.rng.randrange(self.cfg.num_players)
        self.acting_seat = self._next_alive_seat(self.dealer_seat)

        self.last_bet_mul = 1
        self.last_bet_seen = False

        return self.get_obs(self.hero_seat)

    # ======================================================
    # Helpers
    # ======================================================
    def _get_player(self, seat: int) -> PlayerState:
        return self.players[seat]

    def _alive_seats(self) -> List[int]:
        return [p.seat for p in self.players if p.alive]

    def _count_alive(self) -> int:
        return sum(1 for p in self.players if p.alive)

    def _next_alive_seat(self, seat: int) -> int:
        for k in range(1, self.cfg.num_players + 1):
            s = (seat + k) % self.cfg.num_players
            if self.players[s].alive:
                return s
        return seat

    # ======================================================
    # Observation
    # ======================================================
    def get_obs(self, seat: int) -> Dict[str, Any]:
        p = self._get_player(seat)
        ctx = BetContext(self.last_bet_mul, self.last_bet_seen)
        min_mul = calc_min_bet_mul(ctx, p.has_seen, self.cfg)
        min_need = bet_cost(min_mul, self.cfg.base)

        return {
            "round": self.round_index,
            "alive_count": self._count_alive(),
            "pot": self.pot,
            "last_bet_mul": self.last_bet_mul,
            "last_bet_seen": self.last_bet_seen,
            "acting_seat": self.acting_seat,
            "hero_seat": self.hero_seat,

            "self": {
                "seat": p.seat,
                "alive": p.alive,
                "has_seen": p.has_seen,
                "stack": p.stack,
                "bet_total": p.bet_total,
                "cards": p.cards,
            },

            "min_bet_mul": min_mul,
            "min_bet_need": min_need,
            "force_allin_or_fold": p.stack <= min_need,

            # ===== 核心：牌力 bucket（只在看牌后有效）=====
            "hand_rank": p.hand_rank if p.has_seen else 0,

            "opponents": [
                {
                    "seat": o.seat,
                    "alive": o.alive,
                    "has_seen": o.has_seen,
                    "stack": o.stack,
                    "bet_total": o.bet_total,
                }
                for o in self.players
                if o.seat != seat
            ],
        }

    # ======================================================
    # Legal actions
    # ======================================================
    def legal_actions(self, seat: int) -> List[Dict[str, Any]]:
        p = self._get_player(seat)
        if self.done or not p.alive:
            return []

        actions: List[Dict[str, Any]] = []

        if not p.has_seen:
            actions.append({"type": "SEE"})

        if seat != self.acting_seat:
            return actions

        ctx = BetContext(self.last_bet_mul, self.last_bet_seen)
        min_mul = calc_min_bet_mul(ctx, p.has_seen, self.cfg)
        min_need = bet_cost(min_mul, self.cfg.base)

        if p.stack <= min_need:
            actions.append({"type": "FOLD"})
            actions.append({"type": "ALL_IN"})
            return actions

        actions.append({"type": "FOLD"})

        if p.has_seen:
            muls = range(min_mul, self.cfg.max_mul_seen + 1, 2)
        else:
            muls = range(min_mul, self.cfg.max_mul_unseen + 1)

        for mul in muls:
            cost = bet_cost(mul, self.cfg.base)
            if cost <= p.stack and is_valid_bet_mul(mul, p.has_seen, self.cfg):
                actions.append({"type": "BET", "mul": mul})

        for t in self._alive_seats():
            if t != seat:
                need = pk_cost(ctx, self.cfg.base)
                if need <= p.stack:
                    actions.append({"type": "PK", "target": t})

        return actions

    # ======================================================
    # Step
    # ======================================================
    def step(self, seat: int, action: Dict[str, Any]):
        if self.done:
            return self.get_obs(self.hero_seat), 0.0, True, {}

        p = self._get_player(seat)
        reward = 0.0
        info: Dict[str, Any] = {}

        # ===== SEE =====
        if action["type"] == "SEE":
            if p.has_seen:
                return self.get_obs(self.hero_seat), -0.05, self.done, {"error": "DOUBLE_SEE"}
            p.has_seen = True
            p.hand_rank = hand_rank_to_bucket(p.cards)
            return self.get_obs(self.hero_seat), -0.02, self.done, {"info": "SEE"}

        if seat != self.acting_seat:
            return self.get_obs(self.hero_seat), -0.02, self.done, {"error": "NOT_YOUR_TURN"}

        # ===== FOLD =====
        if action["type"] == "FOLD":
            p.alive = False
            reward -= 0.5

        # ===== ALL IN =====
        elif action["type"] == "ALL_IN":
            reward -= 0.1
            self.pot += p.stack
            p.bet_total += p.stack
            p.stack = 0

        # ===== BET =====
        elif action["type"] == "BET":
            mul = action["mul"]
            cost = bet_cost(mul, self.cfg.base)
            p.stack -= cost
            p.bet_total += cost
            self.pot += cost
            reward += 0.05
            if mul > self.last_bet_mul:
                self.last_bet_mul = mul
                self.last_bet_seen = p.has_seen

        # ===== PK =====
        elif action["type"] == "PK":
            target = action["target"]
            need = pk_cost(BetContext(self.last_bet_mul, self.last_bet_seen), self.cfg.base)
            p.stack -= need
            p.bet_total += need
            self.pot += need
            reward += 0.1

            t = self._get_player(target)
            if compare_hands(p.cards, t.cards) > 0:
                t.alive = False
            else:
                p.alive = False

        else:
            return self.get_obs(self.hero_seat), -0.02, self.done, {"error": "UNKNOWN"}

        self._turn_marks.add(seat)

        if self._count_alive() <= 1:
            self.done = True
            self.winner_seat = self._alive_seats()[0]
            self._get_player(self.winner_seat).stack += self.pot
            self.pot = 0
            hero = self._get_player(self.hero_seat)
            reward += hero.stack - self.hero_init_stack
            return self.get_obs(self.hero_seat), reward, True, info

        self.acting_seat = self._next_alive_seat(seat)

        alive_now = set(self._alive_seats())
        if alive_now.issubset(self._turn_marks):
            self.round_index += 1
            self._turn_marks = set()

        return self.get_obs(self.hero_seat), reward, False, info
