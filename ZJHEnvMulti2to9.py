# ZJHEnvMulti2to9.py
import numpy as np
import random
from enum import Enum
import math


# ============================================================
# 对手动作类型（保持不变）
# ============================================================

class OppAct(Enum):
    FOLD = 0
    CALL = 1
    LOOK = 2
    BET = 3
    PK = 4
    COMPARE_ALL = 5


# ============================================================
# 牌型评估（保持原规则）
# ============================================================

def evaluate_hand(cards):
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    r1, r2, r3 = ranks

    is_flush = (len(set(suits)) == 1)
    is_straight = (r1 == r2 + 1 == r3 + 2) or (r1 == 14 and r2 == 3 and r3 == 2)

    if r1 == r2 == r3:
        return 5, ranks
    if is_flush and is_straight:
        return 4, ranks
    if is_flush:
        return 3, ranks
    if is_straight:
        return 2, ranks
    if r1 == r2 or r2 == r3:
        return 1, ranks
    return 0, ranks


def compare_hands(c1, c2):
    t1, r1 = evaluate_hand(c1)
    t2, r2 = evaluate_hand(c2)
    if t1 != t2:
        return 1 if t1 > t2 else -1
    return 1 if r1 > r2 else -1


# ============================================================
# ZJHEnvMulti2to9
# ============================================================

class ZJHEnvMulti2to9:

    def __init__(self,
                 max_round=20,
                 min_players=2,
                 max_players=9):

        self.min_players = min_players
        self.max_players = max_players
        self.max_round = max_round

        self.big_blind = 1.0

        self.action_dim = 14
        self.state_dim = 49

        self.reset()

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def _sample_env_params(self):
        self.num_players = random.randint(self.min_players, self.max_players)

        # 初始筹码（BB，LogUniform）
        lo, hi = math.log(20), math.log(300)
        self.starting_stack_bb = math.exp(random.uniform(lo, hi))

        # ante（BB）
        self.ante_bb = random.choice([0.0, 0.5, 1.0, 2.0])

    def _init_players(self):
        self.players = []
        for _ in range(self.num_players):
            self.players.append(dict(
                cards=None,
                stack_bb=self.starting_stack_bb,
                bet_bb=0.0,
                alive=True,
                has_seen=False,
                last_act=0,
            ))

        while len(self.players) < self.max_players:
            self.players.append(dict(
                cards=None,
                stack_bb=0.0,
                bet_bb=0.0,
                alive=False,
                has_seen=False,
                last_act=0,
            ))

    def _deal_cards(self):
        deck = [(r, s) for r in range(2, 15) for s in range(4)]
        random.shuffle(deck)
        for i in range(self.num_players):
            self.players[i]["cards"] = deck[i * 3:(i + 1) * 3]

    # --------------------------------------------------------
    # Hero 牌力缓存
    # --------------------------------------------------------

    def _cache_hero_hand(self):
        hero = self.players[0]
        t, ranks = evaluate_hand(hero["cards"])
        hi, mid, lo = ranks
        self.hero_hand_type = float(t)
        self.hero_hi = float(hi)
        self.hero_mid = float(mid)
        self.hero_lo = float(lo)
        self.hero_strength = (t + hi / 14.0) / 6.0

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    def _get_state(self):
        hero = self.players[0]

        avg_stack = sum(
            p["stack_bb"] for p in self.players[:self.num_players] if p["alive"]
        ) / max(1, self.alive_cnt)

        state = [
            self.hero_hand_type,
            self.hero_hi,
            self.hero_mid,
            self.hero_lo,
            self.hero_strength,
            float(self.round_index),
            float(self.num_players) / 9.0,
            float(self.alive_cnt) / float(self.num_players),
            float(self.ante_bb),
            float(self.pot_bb),
            float(self.max_bet_bb),
            float(int(hero["has_seen"])),
            hero["stack_bb"],
            hero["stack_bb"] / max(1e-6, avg_stack),
        ]

        for idx in range(1, self.max_players):
            if idx < self.num_players:
                p = self.players[idx]
                if p["alive"]:
                    state.extend([
                        1.0,
                        float(int(p["has_seen"])),
                        p["stack_bb"],
                        p["bet_bb"],
                        float(p["last_act"]),
                    ])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0, float(p["last_act"])])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    # --------------------------------------------------------
    # reset
    # --------------------------------------------------------

    def reset(self):
        self._sample_env_params()
        self._init_players()
        self._deal_cards()

        self.pot_bb = 0.0
        self.hero_contrib_bb = 0.0
        self.round_index = 0
        self.done = False

        for i in range(self.num_players):
            p = self.players[i]
            p["stack_bb"] -= self.ante_bb
            p["bet_bb"] = 0.0
            p["alive"] = True
            p["has_seen"] = False
            p["last_act"] = 0
            self.pot_bb += self.ante_bb
            if i == 0:
                self.hero_contrib_bb += self.ante_bb

        self.alive_cnt = self.num_players
        self.max_bet_bb = 0.0
        self._cache_hero_hand()

        return self._get_state()

    # --------------------------------------------------------
    # step（Hero）
    # --------------------------------------------------------

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        hero = self.players[0]

        if action == 0:  # FOLD
            hero["alive"] = False
            self.done = True
            return self._get_state(), -self.hero_contrib_bb, True, {}

        if action == 1:  # CALL
            need = self.max_bet_bb - hero["bet_bb"]
            add = min(need, hero["stack_bb"])
            hero["stack_bb"] -= add
            hero["bet_bb"] += add
            self.pot_bb += add
            self.hero_contrib_bb += add

        elif action == 2:  # LOOK
            hero["has_seen"] = True

        elif 3 <= action <= 11:  # BET
            mult = action - 1
            amount = mult * self.ante_bb
            if hero["has_seen"]:
                amount *= 2
            desired = max(self.max_bet_bb, hero["bet_bb"]) + amount
            add = min(hero["stack_bb"], desired - hero["bet_bb"])
            hero["stack_bb"] -= add
            hero["bet_bb"] += add
            self.pot_bb += add
            self.hero_contrib_bb += add
            self.max_bet_bb = max(self.max_bet_bb, hero["bet_bb"])

        elif action == 12:
            return self._get_state(), self._pk_one(), True, {}

        elif action == 13:
            return self._get_state(), self._compare_all(), True, {}

        self.round_index += 1
        if self.round_index >= self.max_round:
            return self._get_state(), self._compare_all(), True, {}

        return self._get_state(), 0.0, False, {}

    # --------------------------------------------------------
    # PK / 通比
    # --------------------------------------------------------

    def _compare_all(self):
        hero_cards = self.players[0]["cards"]
        for i in range(1, self.num_players):
            p = self.players[i]
            if p["alive"] and compare_hands(hero_cards, p["cards"]) < 0:
                return -self.hero_contrib_bb
        return self.pot_bb - self.hero_contrib_bb

    def _pk_one(self):
        alive = [i for i in range(1, self.num_players) if self.players[i]["alive"]]
        if not alive:
            return self.pot_bb - self.hero_contrib_bb
        opp = random.choice(alive)
        res = compare_hands(self.players[0]["cards"], self.players[opp]["cards"])
        return self.pot_bb - self.hero_contrib_bb if res >= 0 else -self.hero_contrib_bb
